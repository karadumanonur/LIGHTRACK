"""The application's own store: what was recorded, and what each engine made of it.

One SQLite file beside the deliveries. It holds three things and nothing else:
the records that exist, the power each sensor wrote during each passage, and
what a counting engine said about a record. Everything else the interface shows
is derived from a record on demand, because a spectrum is cheap to compute and
expensive to keep.

Counts are keyed by engine as well as by record and sensor, so two engines can
disagree about the same passage and the store keeps both. That is the point of
the switch at the top of the window.

Nothing here imports numpy or an engine. The store must open on a machine that
cannot run either.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS record (
    id       INTEGER PRIMARY KEY,
    folder   TEXT NOT NULL,
    file     TEXT NOT NULL,
    utc      TEXT NOT NULL,
    local    TEXT NOT NULL,
    epoch    INTEGER NOT NULL,
    samples  INTEGER,
    seconds  REAL,
    UNIQUE (folder, file)
);
CREATE INDEX IF NOT EXISTS record_epoch ON record (epoch);

CREATE TABLE IF NOT EXISTS power (
    record_id INTEGER NOT NULL REFERENCES record (id) ON DELETE CASCADE,
    pad       INTEGER NOT NULL,
    mean      REAL NOT NULL,
    sd        REAL NOT NULL,
    PRIMARY KEY (record_id, pad)
);

CREATE TABLE IF NOT EXISTS count (
    record_id INTEGER NOT NULL REFERENCES record (id) ON DELETE CASCADE,
    pad       INTEGER NOT NULL,
    engine    TEXT NOT NULL,
    axles     INTEGER NOT NULL,
    trains    INTEGER NOT NULL,
    speed_kmh REAL,
    first_s   REAL,
    last_s    REAL,
    detail    TEXT,
    PRIMARY KEY (record_id, pad, engine)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def default_path(root: Path) -> Path:
    return root / "lightrack.db"


@contextmanager
def open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def put_record(con, folder: str, file: str, utc: str, local: str, epoch: int,
               samples: int, seconds: float) -> int:
    con.execute(
        "INSERT INTO record (folder, file, utc, local, epoch, samples, seconds)"
        " VALUES (?,?,?,?,?,?,?)"
        " ON CONFLICT (folder, file) DO UPDATE SET"
        " utc=excluded.utc, local=excluded.local, epoch=excluded.epoch,"
        " samples=excluded.samples, seconds=excluded.seconds",
        (folder, file, utc, local, epoch, samples, seconds))
    row = con.execute("SELECT id FROM record WHERE folder=? AND file=?",
                      (folder, file)).fetchone()
    return int(row["id"])


def put_power(con, record_id: int, values: dict) -> None:
    con.executemany(
        "INSERT INTO power (record_id, pad, mean, sd) VALUES (?,?,?,?)"
        " ON CONFLICT (record_id, pad) DO UPDATE SET"
        " mean=excluded.mean, sd=excluded.sd",
        [(record_id, pad, m, s) for pad, (m, s) in sorted(values.items())])


def put_count(con, record_id: int, pad: int, engine: str, r: dict) -> None:
    con.execute(
        "INSERT INTO count (record_id, pad, engine, axles, trains, speed_kmh,"
        " first_s, last_s, detail) VALUES (?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT (record_id, pad, engine) DO UPDATE SET"
        " axles=excluded.axles, trains=excluded.trains,"
        " speed_kmh=excluded.speed_kmh, first_s=excluded.first_s,"
        " last_s=excluded.last_s, detail=excluded.detail",
        (record_id, pad, engine, int(r.get("axles", 0)), int(r.get("trains", 0)),
         r.get("speed_kmh"), r.get("first_s"), r.get("last_s"),
         r.get("detail", "")))


def folders(con) -> list[dict]:
    rows = con.execute(
        "SELECT folder, COUNT(*) n, MIN(epoch) a, MAX(epoch) b"
        " FROM record GROUP BY folder ORDER BY a").fetchall()
    return [dict(folder=r["folder"], records=r["n"], first=r["a"], last=r["b"])
            for r in rows]


def records(con, folder: str | None, engine: str, pad: int,
            query: str = "", limit: int = 4000) -> list[dict]:
    sql = ("SELECT r.id, r.folder, r.file, r.local, r.epoch, r.seconds,"
           " c.axles, c.trains, c.speed_kmh, c.first_s"
           " FROM record r LEFT JOIN count c"
           " ON c.record_id = r.id AND c.engine = ? AND c.pad = ?")
    args: list = [engine, pad]
    where = []
    if folder:
        where.append("r.folder = ?")
        args.append(folder)
    if query:
        where.append("(r.file LIKE ? OR r.local LIKE ?)")
        args += [f"%{query}%", f"%{query}%"]
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.epoch LIMIT ?"
    args.append(limit)
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def series(con, pad: int, engine: str, folder: str | None = None) -> list[dict]:
    sql = ("SELECT r.id, r.file, r.folder, r.local, r.epoch,"
           " p.mean, p.sd, c.axles, c.speed_kmh"
           " FROM record r JOIN power p ON p.record_id = r.id AND p.pad = ?"
           " LEFT JOIN count c ON c.record_id = r.id AND c.pad = ?"
           " AND c.engine = ?")
    args: list = [pad, pad, engine]
    if folder:
        sql += " WHERE r.folder = ?"
        args.append(folder)
    sql += " ORDER BY r.epoch"
    return [dict(r) for r in con.execute(sql, args).fetchall()]


def one(con, record_id: int) -> dict | None:
    row = con.execute("SELECT * FROM record WHERE id = ?",
                      (record_id,)).fetchone()
    return dict(row) if row else None


def counted(con, engine: str, pad: int) -> int:
    row = con.execute("SELECT COUNT(*) n FROM count WHERE engine=? AND pad=?",
                      (engine, pad)).fetchone()
    return int(row["n"])


def totals(con, engine: str, pad: int) -> dict:
    row = con.execute(
        "SELECT COUNT(*) records, COALESCE(SUM(c.axles), 0) axles"
        " FROM record r LEFT JOIN count c ON c.record_id = r.id"
        " AND c.engine = ? AND c.pad = ?", (engine, pad)).fetchone()
    return dict(records=int(row["records"]), axles=int(row["axles"]))


def set_meta(con, key: str, value: str) -> None:
    con.execute("INSERT INTO meta (key, value) VALUES (?,?)"
                " ON CONFLICT (key) DO UPDATE SET value=excluded.value",
                (key, value))


def get_meta(con, key: str, default: str = "") -> str:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default

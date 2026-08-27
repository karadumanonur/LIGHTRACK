"""A local window onto the store, served from the standard library.

No web framework is a dependency. Someone who clones this repository can run
the application with the Python they already have, which is the whole reason
for not reaching for one: the alternative is that the person who most needs to
look at these records cannot open them.

Most of what the browser asks for is a lookup in the store. Two things are not.
Selecting a record reads it off disk and computes its spectra, its modes and
the engine's own trace, which takes about a second. Zooming the raw signal asks
for that window again at full resolution, because an overview thinned to nine
hundred bins is a picture of the whole record and becomes a picture of nothing
when you stretch a second of it across the panel.

Adding a folder runs in a worker thread and reports its progress, so the window
stays usable while eighteen hundred records are read.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import db, engines, home, signals

ROOT = home.bundle()
STATIC = home.static()
PADS = list(range(1, 13))


def _analysis_on_path() -> None:
    home.code_on_path()


def _read(path: Path):
    _analysis_on_path()
    from analysis.config import CONFIG                          # noqa: E402
    from analysis.reader import read_channels                   # noqa: E402
    return read_channels(path), float(CONFIG["acquisition"]["fs_hz"])


class Job:
    """One folder being taken in, and how far it has got."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = dict(running=False, folder="", stage="", done=0, total=0,
                          message="", finished_at=0.0)

    def read(self) -> dict:
        with self.lock:
            return dict(self.state)

    def set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)


class App:
    def __init__(self, database: Path, roots: dict[str, Path]):
        self.database = database
        self.roots = dict(roots)
        self.job = Job()

    def setup(self) -> dict:
        with db.open_db(self.database) as con:
            fold = db.folders(con)
        for f in fold:
            f["path"] = str(self.roots.get(f["folder"], ""))
        return dict(engines=engines.available(), folders=fold, pads=PADS,
                    database=str(self.database), root=str(ROOT))

    def browse(self, q) -> dict:
        raw = unquote(q.get("path", [""])[0]).strip()
        here = Path(raw) if raw else home.beside()
        try:
            here = here.expanduser().resolve()
        except Exception:
            return dict(error="that path cannot be read", path=raw, entries=[])
        if not here.is_dir():
            return dict(error="not a folder", path=str(here), entries=[])
        entries = []
        try:
            for child in sorted(here.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                try:
                    n = sum(1 for _ in child.glob("*.txt"))
                except Exception:
                    n = 0
                entries.append(dict(name=child.name, path=str(child), records=n))
        except PermissionError:
            return dict(error="no permission to read that folder",
                        path=str(here), entries=[])
        try:
            n_here = sum(1 for _ in here.glob("*.txt"))
        except Exception:
            n_here = 0
        return dict(path=str(here), parent=str(here.parent), records=n_here,
                    entries=entries)

    def ingest(self, q) -> dict:
        raw = unquote(q.get("path", [""])[0]).strip()
        engine = q.get("engine", ["onur"])[0]
        pad = int(q.get("pad", ["1"])[0])
        if self.job.read()["running"]:
            return dict(error="already taking a folder in")
        folder = Path(raw).expanduser()
        if not folder.is_dir():
            return dict(error="not a folder")
        name = folder.name
        self.roots[name] = folder
        t = threading.Thread(target=self._ingest, args=(name, folder, engine, pad),
                             daemon=True)
        t.start()
        return dict(started=name)

    def _ingest(self, name: str, folder: Path, engine: str, pad: int) -> None:
        _analysis_on_path()
        from lightrack import parse_filename_utc, to_local        # noqa: E402
        from analysis.config import CONFIG                        # noqa: E402
        from analysis.reader import read_channels                 # noqa: E402
        from datetime import timezone
        fs = float(CONFIG["acquisition"]["fs_hz"])
        self.job.set(running=True, folder=name, stage="looking", done=0,
                     total=0, message="")
        try:
            files = sorted(folder.glob("*.txt"))
            self.job.set(stage="listing", total=len(files))
            ids = []
            with db.open_db(self.database) as con:
                for f in files:
                    try:
                        utc = parse_filename_utc(f.name)
                    except Exception:
                        continue
                    local = to_local(utc).replace(tzinfo=None)
                    samples = int(f.stat().st_size / 175)
                    ids.append(db.put_record(
                        con, name, f.name, utc.isoformat(),
                        local.isoformat(sep=" "),
                        int(utc.replace(tzinfo=timezone.utc).timestamp()),
                        samples, round(samples / fs, 1)))
            self.job.set(stage="measuring", total=len(ids), done=0)
            for n, rid in enumerate(ids, 1):
                with db.open_db(self.database) as con:
                    r = db.one(con, rid)
                path = folder / r["file"]
                try:
                    w = read_channels(path)
                    c = engines.count(engine, path, pad)
                except Exception as exc:
                    self.job.set(message=f"{r['file']}: {exc}")
                    continue
                marks = c.get("marks_s") or []
                power = {p: signals.passage_peaks(w[:, p - 1], fs, marks)
                         for p in PADS}
                with db.open_db(self.database) as con:
                    db.put_count(con, rid, pad, engine, c)
                    db.put_power(con, rid, power)
                self.job.set(done=n)
            self.job.set(stage="done", message=f"{len(ids)} records")
        except Exception as exc:
            self.job.set(stage="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            self.job.set(running=False, finished_at=time.time())

    def progress(self) -> dict:
        return self.job.read()

    def records(self, q) -> dict:
        engine = q.get("engine", ["onur"])[0]
        pad = int(q.get("pad", ["1"])[0])
        folder = (q.get("folder", [""])[0] or None)
        text = q.get("q", [""])[0]
        with db.open_db(self.database) as con:
            rows = db.records(con, folder, engine, pad, text)
            tot = db.totals(con, engine, pad)
        return dict(records=rows, totals=tot)

    def series(self, q) -> dict:
        engine = q.get("engine", ["onur"])[0]
        pad = int(q.get("pad", ["1"])[0])
        folder = (q.get("folder", [""])[0] or None)
        with db.open_db(self.database) as con:
            rows = db.series(con, pad, engine, folder)
        return dict(series=rows, panels=panels_of(rows))

    def _locate(self, rid: int):
        with db.open_db(self.database) as con:
            r = db.one(con, rid)
        if r is None:
            return None, None
        root = self.roots.get(r["folder"])
        if root is None:
            return r, None
        return r, root / r["file"]

    def raw(self, rid: int, q) -> dict:
        """The raw signal over one window, at whatever resolution it needs.

        With the baseline taken out by default. Each grating sits at its own
        wavelength — 1516 nm, 1520, 1525 — so twelve raw traces stack up the
        axis a nanometre apart and cannot be compared. Removing the baseline
        puts them all about zero, where the only difference left between them
        is the one worth looking at: how big a train they write.
        """
        pad = int(q.get("pad", ["1"])[0])
        keep = q.get("base", ["remove"])[0] == "keep"
        r, path = self._locate(rid)
        if path is None:
            return dict(error="record not found on disk")
        w, fs = _read(path)
        col = w[:, pad - 1]
        base = float(np.median(col))
        if not keep:
            col = signals.remove_baseline(col, fs)
        t0 = float(q.get("t0", ["0"])[0])
        t1 = float(q.get("t1", [str(len(col) / fs)])[0])
        a = max(0, int(t0 * fs))
        b = min(len(col), int(t1 * fs))
        if b - a < 8:
            a, b = 0, len(col)
        seg = col[a:b]
        n = 4 if keep else 2
        unit = "nm" if keep else "pm"
        bins = 900
        if len(seg) <= bins * 2:
            t = (np.arange(a, b) / fs)
            return dict(mode="points", unit=unit, base=round(base, 4),
                        t=signals.round_to(t, 4),
                        lo=signals.round_to(seg, n), hi=signals.round_to(seg, n),
                        span=[round(a / fs, 4), round(b / fs, 4)])
        lo, hi = signals.envelope(seg, bins)
        step = len(seg) / len(lo)
        t = (a + np.arange(len(lo)) * step) / fs
        return dict(mode="envelope", unit=unit, base=round(base, 4),
                    t=signals.round_to(t, 4),
                    lo=signals.round_to(lo, n), hi=signals.round_to(hi, n),
                    span=[round(a / fs, 4), round(b / fs, 4)])

    def trace(self, rid: int, q) -> dict:
        """The engine's view of one window, at whatever resolution it needs.

        Zooming into how an axle was found has to show more of the signal, not
        the same points stretched, so the step shrinks with the window until it
        is every sample.
        """
        engine = q.get("engine", ["onur"])[0]
        pad = int(q.get("pad", ["1"])[0])
        r, path = self._locate(rid)
        if path is None:
            return dict(error="record not found on disk")
        t0 = float(q.get("t0", ["0"])[0])
        t1 = float(q.get("t1", ["0"])[0])
        span = (t0, t1) if t1 > t0 else None
        step = 1
        if span:
            step = max(1, int((t1 - t0) * 500.0 / 1400.0))
        try:
            return engines.trace(engine, path, pad, span=span, step=step)
        except Exception as exc:
            return dict(error=f"{type(exc).__name__}: {exc}")

    def record(self, rid: int, q) -> dict:
        engine = q.get("engine", ["onur"])[0]
        pad = int(q.get("pad", ["1"])[0])
        r, path = self._locate(rid)
        if r is None:
            return dict(error="no such record")
        if path is None or not path.exists():
            return dict(error=f"{r['file']} is not where the store expects it")
        w, fs = _read(path)
        col = w[:, pad - 1]
        try:
            tr = engines.trace(engine, path, pad, span=None, step=6)
        except Exception as exc:
            tr = dict(error=str(exc))
        a = tr.get("first_s", 0.0) or 0.0
        b = tr.get("last_s", 0.0) or 0.0
        if b - a < 1.0:
            a, b = 0.0, len(col) / fs
        window = (max(0.0, a - 2.0), min(len(col) / fs, b + 2.0))
        if "error" not in tr:
            tr = engines.trace(engine, path, pad, span=window, step=6)
        mid = 0.5 * (window[0] + window[1])
        return dict(
            record=r, pad=pad, engine=engine, fs=fs,
            seconds=round(len(col) / fs, 2),
            overview=signals.overview(col, fs),
            spectra=signals.spectra(col, fs),
            emd=signals.modes(col, fs, (max(0.0, mid - 4.0), mid + 4.0)),
            trace=tr, window=[round(window[0], 2), round(window[1], 2)])


def panels_of(rows: list[dict], break_days: float = 5.0,
              gutter: float = 0.038) -> dict:
    """Split the time axis where nothing was recorded.

    Forty-four days separate the two deliveries. Drawn on one continuous axis
    that is two thirds of the picture spent on a period we have no data for, so
    the axis is broken and the break is labelled instead.
    """
    if not rows:
        return dict(panels=[], breaks=[])
    ep = [int(r["epoch"]) for r in rows]
    cuts = [i for i, (a, b) in enumerate(zip(ep, ep[1:]))
            if b - a > break_days * 86400]
    blocks, start = [], 0
    for i in cuts:
        blocks.append((start, i))
        start = i + 1
    blocks.append((start, len(ep) - 1))

    spans = []
    for a, b in blocks:
        lo, hi = ep[a], ep[b]
        pad = max((hi - lo) * 0.02, 3600)
        spans.append((lo - pad, hi + pad))
    total = sum(hi - lo for lo, hi in spans) or 1.0
    free = 1.0 - gutter * (len(spans) - 1)

    panels, x = [], 0.0
    for (lo, hi), (a, b) in zip(spans, blocks):
        w = free * (hi - lo) / total
        panels.append(dict(t0=lo, t1=hi, x0=round(x, 5), x1=round(x + w, 5),
                           first=a, last=b))
        x += w + gutter
    breaks = [dict(x0=panels[k]["x1"], x1=panels[k + 1]["x0"],
                   days=round((ep[i + 1] - ep[i]) / 86400))
              for k, i in enumerate(cuts)]
    return dict(panels=panels, breaks=breaks)


def handler_for(app: App):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(STATIC), **kw)

        def log_message(self, fmt, *args):
            pass

        def end_headers(self):
            """Never cache. The window is edited while it is open."""
            self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_index(self):
            """Stamp the script and stylesheet with their own modified time.

            The window is edited while it is open. A browser that keeps
            yesterday's script will show yesterday's bugs and no amount of
            reloading persuades it otherwise, so the name changes when the
            file does.
            """
            html = (STATIC / "index.html").read_text(encoding="utf-8")
            for name in ("app.js", "app.css"):
                f = STATIC / name
                if f.exists():
                    html = html.replace(name + '"',
                                        f"{name}?v={int(f.stat().st_mtime)}\"")
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self.send_index()
            if not u.path.startswith("/api/"):
                return super().do_GET()
            q = parse_qs(u.query)
            try:
                if u.path == "/api/setup":
                    return self.send_json(app.setup())
                if u.path == "/api/browse":
                    return self.send_json(app.browse(q))
                if u.path == "/api/ingest":
                    return self.send_json(app.ingest(q))
                if u.path == "/api/progress":
                    return self.send_json(app.progress())
                if u.path == "/api/records":
                    return self.send_json(app.records(q))
                if u.path == "/api/series":
                    return self.send_json(app.series(q))
                if u.path.startswith("/api/trace/"):
                    return self.send_json(
                        app.trace(int(u.path.rsplit("/", 1)[-1]), q))
                if u.path.startswith("/api/raw/"):
                    return self.send_json(
                        app.raw(int(u.path.rsplit("/", 1)[-1]), q))
                if u.path.startswith("/api/record/"):
                    return self.send_json(
                        app.record(int(u.path.rsplit("/", 1)[-1]), q))
            except Exception as exc:
                return self.send_json(dict(error=f"{type(exc).__name__}: {exc}"),
                                      status=500)
            self.send_json(dict(error="unknown endpoint"), status=404)

    return Handler


def serve(database: Path, roots: dict[str, Path], port: int = 8765,
          open_browser: bool = True) -> None:
    app = App(database, roots)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"LIGHTRACK is at {url}")
    print(f"   store    {database}")
    for name, path in roots.items():
        print(f"   records  {name} -> {path}")
    print("   stop with ctrl-c")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()

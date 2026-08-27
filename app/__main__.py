"""Three commands: find the records, measure them, look at them.

    python -m lightrack.app scan detected Valid_data_new poc1
    python -m lightrack.app measure --engine onur --pad 1
    python -m lightrack.app serve

`scan` only reads the file names and their length, so it is quick and can be
rerun whenever a delivery arrives. `measure` is the slow one: it reads every
record once, asks an engine how many axles are in it, and takes the power every
sensor wrote while that train was overhead. `serve` opens the window; anything
`measure` has not done yet simply shows as blank rather than blocking it.

One pad is COUNTED and all twelve are MEASURED, and that asymmetry is the
point. Counting is a job for the head that works; the size of the response is
how you find out which heads still do. The counting head says where the axles
are; every sensor's baseline-removed signal is then read at those same moments,
so the twelve numbers are responses to the same axles of the same train and can
be put beside each other.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import db, engines, home, signals, server

ROOT = home.bundle()
DEFAULT_FOLDERS = ("detected", "Valid_data_new", "poc1")


def _lightrack_on_path() -> None:
    home.code_on_path()


def roots_for(names) -> dict[str, Path]:
    out = {}
    for n in names:
        p = (ROOT / n)
        if p.is_dir():
            out[n] = p
    return out


def scan(args) -> None:
    _lightrack_on_path()
    from lightrack import parse_filename_utc, to_local           # noqa: E402
    from analysis.config import CONFIG                           # noqa: E402
    fs = float(CONFIG["acquisition"]["fs_hz"])

    roots = roots_for(args.folders or DEFAULT_FOLDERS)
    if not roots:
        print("no delivery folder found; give one by name")
        return
    with db.open_db(args.database) as con:
        for name, path in roots.items():
            files = sorted(path.glob("*.txt"))
            n = 0
            for f in files:
                try:
                    utc = parse_filename_utc(f.name)
                except Exception:
                    continue
                local = to_local(utc).replace(tzinfo=None)
                size = f.stat().st_size
                samples = int(size / 175)
                db.put_record(con, name, f.name, utc.isoformat(),
                              local.isoformat(sep=" "),
                              int(utc.replace(tzinfo=timezone.utc).timestamp()),
                              samples, round(samples / fs, 1))
                n += 1
            print(f"{name:<18}{n:>6} records")
        db.set_meta(con, "scanned", datetime.now().isoformat(timespec="seconds"))
    print(f"store {args.database}")


def measure(args) -> None:
    _lightrack_on_path()
    from analysis.config import CONFIG                           # noqa: E402
    from analysis.reader import read_channels                    # noqa: E402
    fs = float(CONFIG["acquisition"]["fs_hz"])
    roots = roots_for(args.folders or DEFAULT_FOLDERS)
    count_pad = args.pad
    power_pads = list(range(1, 13))

    with db.open_db(args.database) as con:
        todo = db.records(con, None, args.engine, count_pad, "", limit=10 ** 6)
        todo = [r for r in todo if r["folder"] in roots]
        if not args.again:
            todo = [r for r in todo if r["axles"] is None]
        print(f"{len(todo)} records to measure with {args.engine}", flush=True)
        t0 = time.time()
        for n, r in enumerate(todo, 1):
            path = roots[r["folder"]] / r["file"]
            try:
                w = read_channels(path)
            except Exception as exc:
                print(f"   !! {r['file']}: {exc}", flush=True)
                continue
            try:
                c = engines.count(args.engine, path, count_pad)
            except Exception as exc:
                print(f"   !! {r['file']}: {exc}", flush=True)
                c = dict(axles=0, trains=0, speed_kmh=None, first_s=0.0,
                         last_s=0.0, detail="")
            db.put_count(con, r["id"], count_pad, args.engine, c)
            marks = c.get("marks_s") or []
            power = {p: signals.passage_peaks(w[:, p - 1], fs, marks)
                     for p in power_pads}
            db.put_power(con, r["id"], power)
            if n % 50 == 0:
                con.commit()
                print(f"   [{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
        db.set_meta(con, f"measured_{args.engine}",
                    datetime.now().isoformat(timespec="seconds"))
    print(f"done in {time.time() - t0:.0f}s")


def run_server(args) -> None:
    roots = roots_for(args.folders or DEFAULT_FOLDERS)
    with db.open_db(args.database):
        pass
    server.serve(args.database, roots, port=args.port,
                 open_browser=not args.no_browser)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m lightrack.app",
                                description=__doc__.splitlines()[0])
    p.add_argument("--database", type=Path, default=home.database())
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="find the records and remember them")
    s.add_argument("folders", nargs="*")
    s.set_defaults(func=scan)

    m = sub.add_parser("measure", help="count the axles and take the power")
    m.add_argument("folders", nargs="*")
    m.add_argument("--engine", default="onur", choices=list(engines.ENGINES))
    m.add_argument("--pad", type=int, default=1,
                   help="the head to count on; power is taken on all twelve")
    m.add_argument("--again", action="store_true",
                   help="measure records that already have a result")
    m.set_defaults(func=measure)

    v = sub.add_parser("serve", help="open the window")
    v.add_argument("folders", nargs="*")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-browser", action="store_true")
    v.set_defaults(func=run_server)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

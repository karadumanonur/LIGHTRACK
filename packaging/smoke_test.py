"""Start the built program and make it count something.

A build that starts is not a build that works. This asks the program the only
question it exists to answer — how many axles are in this record — using a
record made by `make_fake_record.py`, and checks the answer against what the
same record gives on the machine the program was developed on.

The number is pinned rather than merely required to be positive. Both builds
run the same code over the same bytes, so they must agree; if a Mac and a
Windows machine disagree about a made-up train they would disagree about a
real one, and that is worth failing a build over.

What is checked on Asude's engine is that it runs at all and returns a trace
of the right length, since its failure mode is an import that never happens;
its count on a made-up train says nothing about a real one.

    python packaging/smoke_test.py dist/LIGHTRACK.exe records/
    python packaging/smoke_test.py /Applications/LIGHTRACK.app records/

Given a bundle it starts it through LaunchServices, the way a double-click
does, rather than reaching inside for the binary. That distinction is not
pedantic: a bundle can be built so that the binary runs and the bundle does
not, and testing the binary hides exactly the failure a stranger will meet.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PORT = 8899
AXLES = 13
SPEED = 104.3
SAMPLES = 20000


def get(path: str, tries: int = 1):
    url = f"http://127.0.0.1:{PORT}{path}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, TimeoutError):
            if k == tries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def stop(proc) -> bool:
    """Put the program down, and wait until it has actually let go.

    The one-file build runs a bootloader that starts the real program as a
    child, so signalling the parent alone can leave the child holding the
    bundle open. Packaging it while that is true fails with `Resource busy`,
    which is a confusing way to be told the program is still running.
    """
    posix = os.name == "posix"
    if not posix:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    try:
        if posix:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            if posix:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for _ in range(30):
        if not port_open():
            return True
        time.sleep(1)
    return False


def stop_bundle(app: Path) -> bool:
    """Stop something LaunchServices started, which we have no handle on."""
    subprocess.run(["pkill", "-f", str(app / "Contents" / "MacOS")],
                   capture_output=True)
    for _ in range(30):
        if not port_open():
            return True
        time.sleep(1)
    return False


def main() -> int:
    target, folder = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    bundle = target.suffix == ".app"

    tmp, proc = None, None
    if bundle:
        # Started the way a person starts it. `open` hands off to
        # LaunchServices and returns at once, so there is no process to own
        # here; it is found and stopped by name afterwards.
        subprocess.run(["open", "-a", str(target), "--args",
                        "--no-browser", "--port", str(PORT)], check=True)
    else:
        # The program keeps its store beside itself, so running this twice in
        # the same place would measure a store that already holds another
        # run's records. A copy in a fresh directory is also the honest test:
        # what a stranger gets is a program with no store at all.
        tmp = Path(tempfile.mkdtemp(prefix="lightrack-smoke-"))
        here = tmp / target.name
        shutil.copy2(target, here)
        proc = subprocess.Popen([str(here), "--no-browser", "--port", str(PORT)],
                                start_new_session=(os.name == "posix"))
    checks: list[tuple[bool, str]] = []
    try:
        setup = get("/api/setup", tries=45)
        ready = {e["id"]: e["ready"] for e in setup["engines"]}
        checks.append((ready.get("onur") is True, f"Onur's engine ready: {ready.get('onur')}"))
        checks.append((ready.get("asude") is True, f"Asude's engine ready: {ready.get('asude')}"))
        checks.append((len(setup["pads"]) == 12, f"twelve pads: {len(setup['pads'])}"))

        q = urllib.parse.quote(str(folder))
        started = get(f"/api/ingest?path={q}&engine=onur&pad=1")
        checks.append(("error" not in started, f"folder taken in: {started}"))
        for _ in range(150):
            time.sleep(2)
            job = get("/api/progress")
            if not job["running"]:
                break
        checks.append((job["stage"] == "done", f"ingest finished: {job['stage']}"))

        rows = get("/api/records")
        rows = rows.get("records", rows)
        checks.append((len(rows) == 1, f"one record in the store: {len(rows)}"))
        want = sorted(p.name for p in folder.glob("*.txt"))[0]
        mine = [r for r in rows if r["file"] == want] or rows
        rid = mine[0]["id"]
        checks.append((rows[0]["axles"] == AXLES,
                       f"stored count is {AXLES}: {rows[0]['axles']}"))

        t = get(f"/api/trace/{rid}?pad=1&engine=onur")
        checks.append((t.get("axles") == AXLES, f"Onur counts {AXLES}: {t.get('axles')}"))
        checks.append((abs(float(t.get("speed_kmh", 0)) - SPEED) < 0.05,
                       f"Onur reads {SPEED} km/h: {t.get('speed_kmh')}"))
        checks.append((len(t.get("y", [])) == SAMPLES,
                       f"trace is {SAMPLES} points: {len(t.get('y', []))}"))

        a = get(f"/api/trace/{rid}?pad=1&engine=asude")
        checks.append(("error" not in a, f"Asude's engine runs: {a.get('error', 'yes')}"))
        checks.append((len(a.get("y", [])) == SAMPLES,
                       f"her trace is {SAMPLES} points: {len(a.get('y', []))}"))

        raw = get(f"/api/raw/{rid}?pads=1,9,12&base=remove")
        checks.append((raw.get("unit") == "pm" and len(raw.get("t", [])) > 10,
                       f"raw comes back in pm: {raw.get('unit')}, {len(raw.get('t', []))} points"))
    finally:
        released = stop(proc) if proc is not None else stop_bundle(target)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    checks.append((released,
                   "the program let go: " + ("yes" if released else
                                             "no, it is still answering")))

    for good, line in checks:
        print(("  ok   " if good else "  FAIL ") + line)
    bad = [line for good, line in checks if not good]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

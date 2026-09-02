"""LIGHTRACK as a program someone can be handed.

Double-clicking the built file does what `python -m lightrack.app serve` does,
with two differences that matter away from this machine. There may be no store
yet, so one is made rather than refused; and there are no delivery folders to
guess at, so none are assumed — the window's own folder picker is the way in,
and it scans and measures what it is given.

    LIGHTRACK.exe                 open the window on port 8765
    LIGHTRACK.exe --port 9000     somewhere else
    LIGHTRACK.exe --no-browser    start it, open the page yourself
"""

from __future__ import annotations

import sys
import traceback

from lightrack.app import db, home, server

LOG = "lightrack.log"


def plumbing():
    """Give the program somewhere to speak when nobody is listening.

    A Mac bundle has no terminal attached, and a windowed build can hand back
    a `sys.stdout` of None — so the first `print` would end the program with
    an AttributeError nobody ever sees. Everything it says goes to a file
    beside the store instead, which is also the only evidence there will be
    if it fails to start on someone else's machine.
    """
    if sys.stdout is not None and sys.stdout.isatty():
        return None
    try:
        fh = open(home.store() / LOG, "a", encoding="utf-8", buffering=1)
    except OSError:
        return None
    sys.stdout = fh
    sys.stderr = fh
    return fh


def main() -> int:
    port, browser = 8765, True
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
        elif a.startswith("--port="):
            port = int(a.split("=", 1)[1])
        elif a == "--no-browser":
            browser = False

    database = home.database()
    with db.open_db(database):
        pass
    server.serve(database, {}, port=port, open_browser=browser)
    return 0


if __name__ == "__main__":
    fh = plumbing()
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stderr)
        if fh is None:
            # There is a terminal, so someone is there to read it.
            try:
                input("LIGHTRACK could not start. Press enter to close.")
            except (EOFError, OSError):
                pass
        raise SystemExit(1)

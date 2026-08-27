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
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print()
        input("LIGHTRACK could not start. Press enter to close.")
        raise SystemExit(1)

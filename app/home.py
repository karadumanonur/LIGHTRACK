"""Where the program's pieces are, run from source or built into one file.

Built, the code and the static pages are unpacked to a temporary folder that is
different every run, so nothing may be written there. The store must survive
between runs, so it goes beside the program instead: copy the folder and the
work goes with it. Only when that folder cannot be written to — a program run
from a read-only place — does the store fall back to the user's application
data.

Run from source none of this applies and every path is the repository, which is
why the two cases are answered by the same three functions rather than by
scattered tests for `sys.frozen`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))


def bundle() -> Path:
    """The folder the code and its data were unpacked into."""
    if FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def beside() -> Path:
    """The folder the program itself sits in."""
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def store() -> Path:
    """Where the database goes."""
    here = beside()
    if not FROZEN:
        return here
    probe = here / ".lightrack-probe"
    try:
        probe.touch()
        probe.unlink()
        return here
    except OSError:
        d = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "LIGHTRACK"
        d.mkdir(parents=True, exist_ok=True)
        return d


def database() -> Path:
    return store() / "lightrack.db"


def static() -> Path:
    return bundle() / "lightrack" / "app" / "static"


def code_on_path() -> None:
    """Make `analysis` importable, wherever it came from."""
    p = str(bundle() / "lightrack")
    if p not in sys.path:
        sys.path.insert(0, p)

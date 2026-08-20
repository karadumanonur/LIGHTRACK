"""Run the counter on Onur-Branch over one record.

The branch is imported, not copied: the harness puts its worktree on sys.path
and calls the package the same way `python -m analysis` does, so what is
compared here is the code as it stands on the branch and not a transcription of
it.

One record can hold more than one train. The counter reports them separately
and this adds them up, because the other side counts peaks over the whole
record and the two numbers have to mean the same thing to be comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAME = "Onur"
DESCRIPTION = ("streaming counter, hysteresis and derivative chains, "
               "speed-adaptive thresholds")


def prepare(worktree: Path) -> None:
    if str(worktree) not in sys.path:
        sys.path.insert(0, str(worktree))


def run(path: Path, pad: int = 1) -> dict:
    from analysis import StreamingCounter                       # noqa: E402
    from analysis.config import CONFIG                          # noqa: E402
    from analysis.reader import blocks, read_channels           # noqa: E402

    w = read_channels(path)
    c = StreamingCounter()
    for b in blocks(w[:, pad - 1]):
        c.push(b)
    trains = c.flush()
    fs = CONFIG["acquisition"]["fs_hz"]
    return {
        "axles": int(sum(t.axles for t in trains)),
        "trains": len(trains),
        "detail": " + ".join(str(t.axles) for t in trains),
        "first_s": round(min((t.first_index / fs for t in trains), default=0.0), 2),
        "speed_kmh": round(trains[0].speed_kmh, 1) if trains else 0.0,
        "sensor": f"PAD {pad}",
    }

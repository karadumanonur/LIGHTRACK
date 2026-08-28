"""Write a record that looks like the interrogator's, with axles we chose.

The real recordings are railway data and cannot be committed, but a build that
never counts anything has not been tested. So the build makes its own record:
twelve channels of drifting baseline, and on PAD 1 a train of bogies whose
axles sit where we put them.

The point is not that the counter gets our number right — it is that it gets
the SAME number on every machine. The workflow pins what this file counts to
on Windows and asks a Mac to agree, so a difference in numpy, in scipy, or in
how a platform rounds shows up as a failing build rather than as a wrong axle
count months later.

    python packaging/make_fake_record.py somewhere/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FS = 500.0
SECONDS = 40.0
CENTRES = [1566.3144, 1561.6522, 1556.4243, 1551.7734, 1547.2564, 1542.5399,
           1537.4269, 1533.7194, 1528.4216, 1524.3371, 1518.9932, 1515.6147]
SPEED_KMH = 100.0
WHEELBASE_M = 2.84
COUPLING_M = 5.8
EXTENT_M = 2.2
BOGIES = 8
PEAK_NM = 0.060
NAME = "20260515T032854,182Z.txt"


def axle_samples() -> list[int]:
    v = SPEED_KMH / 3.6
    wheel = WHEELBASE_M / v * FS
    couple = COUPLING_M / v * FS
    at, t = [], 6.0 * FS
    for _ in range(BOGIES):
        at.append(t)
        at.append(t + wheel)
        t += wheel + couple
    return [int(round(x)) for x in at]


def build() -> np.ndarray:
    n = int(SECONDS * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(20260515)
    w = np.empty((n, 12), dtype=float)
    for c, centre in enumerate(CENTRES):
        drift = 0.0008 * np.sin(2 * np.pi * t / 33.0 + c)
        w[:, c] = centre + drift + rng.normal(0.0, 0.00015, n)
    v = SPEED_KMH / 3.6
    sigma = (EXTENT_M / v * FS) / 2.355
    for i in axle_samples():
        half = int(4 * sigma)
        lo, hi = max(0, i - half), min(n, i + half)
        k = np.arange(lo, hi)
        w[lo:hi, 11] += PEAK_NM * np.exp(-0.5 * ((k - i) / sigma) ** 2)
    return w


def write(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / NAME
    w = build()
    t0 = 58893.682
    lines = ["Time [sec]\tWavelength 1[nm]\t"]
    for i, row in enumerate(w):
        stamp = f"{t0 + i / FS:.5f}".replace(".", ",")
        vals = "\t".join(f"{v:.5f}".replace(".", ",") for v in row)
        lines.append(f"{stamp}\t{vals}")
    with open(out, "w", encoding="ascii", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    return out


if __name__ == "__main__":
    where = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    p = write(where)
    print(f"{p}  {p.stat().st_size} bytes  {BOGIES * 2} axles planted")

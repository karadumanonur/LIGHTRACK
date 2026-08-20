"""Run the peak tracker on Asude-Branch over one record.

Her module is imported and its own functions are called with the defaults from
its command line, so the numbers here are what `sensor12_peak_tracker.py`
produces, not a re-implementation: detrend, a fourth-order Butterworth band
pass, then positive peaks above an adaptive prominence.

Two differences from the other side are deliberate and left alone, because
hiding them would defeat the comparison:

  the column is taken by position. `--sensor 12` means the twelfth wavelength
  column of the file, whichever PAD that turns out to be. The harness prints
  the mapping so the two sides can be read as looking at the same rail.

  a peak is not an axle. This side numbers peaks in the record; it has no
  notion of a train, so a record holding two trains yields one number.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAME = "Asude"
DESCRIPTION = ("detrend, Butterworth band pass, positive peaks above an "
               "adaptive prominence")

HIGHPASS_HZ = 0.5
LOWPASS_HZ = 40.0
PROMINENCE_SIGMA = 8.0
HEIGHT_FRACTION = 0.06
MIN_DISTANCE_S = 0.04


def prepare(worktree: Path) -> None:
    if str(worktree) not in sys.path:
        sys.path.insert(0, str(worktree))


def run(path: Path, sensor: int = 12) -> dict:
    import sensor12_peak_tracker as tracker                     # noqa: E402

    time_s, raw_nm = tracker.load_fbg_file(path, sensor)
    fs = tracker.sampling_rate(time_s)
    processed = tracker.butterworth_process(raw_nm, fs, HIGHPASS_HZ, LOWPASS_HZ)
    indices, signs, _ = tracker.detect_peaks(
        processed, fs, "positive", PROMINENCE_SIGMA, None, HEIGHT_FRACTION,
        MIN_DISTANCE_S)
    return {
        "axles": int(len(indices)),
        "trains": 1 if len(indices) else 0,
        "detail": "",
        "first_s": round(float(time_s[indices[0]] - time_s[0]), 2)
                   if len(indices) else 0.0,
        "speed_kmh": 0.0,
        "sensor": f"column {sensor}",
    }

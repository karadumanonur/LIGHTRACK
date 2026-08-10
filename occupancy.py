"""Stationary-state detection — the stopped, crawling or traffic-held train.

Briefing §7.4 records this as a known D-crossing failure: a train halted over a
sensor produces a long excursion carrying more than the six samples the rule
requires, and is counted as an axle. X-crossing correctly rejects it, which is
why the solution-set approach exists — but neither rule should be *counting*
while a train is standing on the head.

One subtlety that decides the implementation. The diagnostic baseline is a
**1 s rolling median**, which erases any strain sustained for longer than about
a second — a stopped train's static load is removed by the very filter that
makes axle peaks visible. So this detector cannot run on the diagnostic
baseline. It uses a **long-window** baseline instead, over which a multi-second
plateau survives.

Output is an abstention interval, not a count. Occupancy logic must treat the
section as OCCUPIED while it holds, not as clear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .baseline import quiet_noise_floor

FS = CONFIG["acquisition"]["fs_hz"]
OCC = CONFIG["occupancy"]


@dataclass
class StationaryEvent:
    start: int
    end: int
    duration_s: float
    mean_pm: float
    peak_density: float      # detected peaks per peak-width inside the plateau

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end,
                "duration_s": self.duration_s, "mean_pm": self.mean_pm,
                "peak_density": self.peak_density}


def slow_baseline(raw_nm: pd.Series | pd.DataFrame,
                  window_s: float | None = None):
    """Baseline-remove with a LONG window so sustained strain survives."""
    w = int((window_s if window_s is not None else OCC["slow_baseline_s"]) * FS)
    return (raw_nm - raw_nm.rolling(w, center=True, min_periods=1).median()) * 1000.0


def find_stationary(sig_slow: np.ndarray, v_kmh: float | None = None
                    ) -> list[StationaryEvent]:
    """Plateaus of sustained strain that carry no axle sequence.

    A stopped train gives a long, roughly flat excursion. A passing train gives
    a burst of narrow peaks. The discriminator is therefore duration together
    with the ABSENCE of peak structure: a long plateau containing many peaks is
    a slow train, not a stopped one, and must not be suppressed.
    """
    sigma = quiet_noise_floor(sig_slow)
    if sigma <= 0:
        return []
    thr = OCC["plateau_sigmas"] * sigma
    hot = np.abs(sig_slow) > thr
    if not hot.any():
        return []

    min_len = int(OCC["min_stationary_s"] * FS)
    idx = np.flatnonzero(hot)
    splits = np.flatnonzero(np.diff(idx) > int(OCC["join_gap_s"] * FS)) + 1
    out: list[StationaryEvent] = []
    for run in np.split(idx, splits):
        if run.size < min_len:
            continue
        a, b = int(run[0]), int(run[-1]) + 1
        seg = sig_slow[a:b]
        # peak density inside the plateau, in peaks per nominal peak width
        from .detect import samples_per_peak
        from scipy.signal import find_peaks
        spp = samples_per_peak(v_kmh) if v_kmh else 40.0
        pk, _ = find_peaks(np.abs(seg), prominence=OCC["inner_peak_sigmas"] * sigma,
                           distance=max(2, int(0.5 * spp)))
        density = len(pk) / max(1.0, (b - a) / spp)
        if density > OCC["max_peak_density"]:
            continue                      # peaky = a slow train, not a stop
        out.append(StationaryEvent(a, b, (b - a) / FS, float(np.mean(seg)),
                                   float(density)))
    return out


def stationary_mask(events: list[StationaryEvent], n: int) -> np.ndarray:
    """Boolean mask over the record where counting must ABSTAIN."""
    m = np.zeros(n, dtype=bool)
    for e in events:
        m[e.start:e.end] = True
    return m

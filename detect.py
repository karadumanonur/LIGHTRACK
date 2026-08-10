"""Axle detection.

SCOPE WARNING. What lives here is the **diagnostic chain's** axle *timing*
detector: it finds where the wheels were so that peak amplitudes can be
measured. The **safety chain** detectors (X-crossing, D-crossing, briefing
§7.4) are NOT here and must not reuse these parameters — briefing §7.5
quantifies the tension: raising the threshold improves counting and degrades
anomaly detection. They arrive in Phase 3 with their own config block.

Every time-domain parameter is speed-adaptive (briefing §3.4, §10).

Design note — why detection is two-stage
----------------------------------------
A first implementation gated peaks on prominence measured against the
record-wide noise floor. That accepted **789 "axles" spread over the whole
145 s record**, because a train occupies only a few seconds of it and the noise
elsewhere clears any fixed multiple of sigma. Every downstream amplitude was
then a median over noise (~0.6 pm instead of ~150 pm).

The fix is to locate the passage first and gate on a fraction of the passage's
own amplitude. A threshold relative to the noise floor answers "is this sample
unusual?"; the question that matters is "is this a wheel?", and only the
passage amplitude sets that scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import CONFIG
from .baseline import quiet_noise_floor

FS = CONFIG["acquisition"]["fs_hz"]
DET = CONFIG["detect"]


def samples_per_peak(v_kmh: float) -> float:
    """Width of one axle peak, in samples. Briefing §3.4: 3960 / v_kmh at 500 Hz."""
    return CONFIG["geometry"]["samples_per_peak_numerator_at_500hz"] / v_kmh


def min_axle_separation(v_kmh: float) -> float:
    """Smallest credible peak-to-peak spacing, in samples.

    From a minimum physical axle spacing (config), not a fixed sample count. At
    the observed 32-141 km/h this varies 4x; a fixed value would merge bogie
    pairs at high speed or split single axles at low speed.
    """
    return DET["min_axle_spacing_m"] * FS / (v_kmh / 3.6)


def has_train(sig: np.ndarray) -> bool:
    """Absolute amplitude test: does this record contain a train at all?

    This is INPUT VALIDATION, and it belongs here because everything downstream
    assumes it has already passed. Every other threshold in this module is
    relative — `passage_window` normalises to the envelope's own maximum — so a
    record of pure noise normalises to itself and produces a window, peaks, an
    axle count and an occupancy verdict, all from nothing.

    POC1 could not have caught this. Its 93 records were hand-selected to contain
    trains, so the "no train" case never entered the data. The July delivery
    contains 41 empty records; the frozen pipeline reported a median of 901
    "axles" in them, and 3 stationary-occupancy events (A-53).

    The threshold is `absolute_gate_pm`, read off an empty interval in the
    observed excursion distribution rather than chosen.
    """
    return float(np.abs(sig).max()) >= DET["absolute_gate_pm"]


def passage_window(sig: np.ndarray, v_kmh: float) -> tuple[int, int] | None:
    """Bracket the train passage on a baseline-removed trace.

    Returns None when the record contains no train — see `has_train`. That check
    is first, before any relative threshold, because a relative threshold cannot
    distinguish an empty record from a loud one.

    Envelope of |sig| smoothed over one peak width, thresholded at a fraction of
    its own maximum. Runs separated by less than `passage_gap_s` are joined —
    that gap has to exceed the quiet interval between two bogies, or a single
    train is split into several passages and each is scored separately.
    """
    if not has_train(sig):
        return None
    w = max(3, int(round(samples_per_peak(v_kmh))))
    env = pd.Series(np.abs(sig)).rolling(w, center=True, min_periods=1).mean().to_numpy()
    peak = float(env.max())
    if peak <= 0:
        return None
    hot = np.flatnonzero(env > DET["passage_envelope_fraction"] * peak)
    if hot.size == 0:
        return None
    splits = np.flatnonzero(np.diff(hot) > DET["passage_gap_s"] * FS) + 1
    run = max(np.split(hot, splits), key=len)
    pad = int(DET["passage_pad_peaks"] * samples_per_peak(v_kmh))
    return max(0, int(run[0]) - pad), min(len(sig), int(run[-1]) + pad + 1)


def axle_peaks(sig: np.ndarray, v_kmh: float,
               window: tuple[int, int] | None = None) -> np.ndarray:
    """Indices of positive axle peaks in a baseline-removed trace.

    Prominence rather than bare amplitude: the 2026 benchmark in briefing §7.4
    found scipy's `prominence` criterion the most robust of four detectors,
    holding up under both wheel flats and polygonisation. Prominence is also
    what rejects the local maximum in the valley *between* two bogies — the
    documented D-crossing failure mode.

    The gate is the larger of (a) a multiple of the quiet noise floor and
    (b) a fraction of the passage amplitude. (a) alone admits noise across the
    silent 95 % of the record; (b) alone would chase a spurious spike in a
    record with no train.
    """
    if window is None:
        window = passage_window(sig, v_kmh)
        if window is None:
            return np.array([], dtype=int)
    i0, i1 = window
    seg = sig[i0:i1]
    if seg.size < 3:
        return np.array([], dtype=int)

    sigma = quiet_noise_floor(sig)
    scale = float(np.percentile(seg, DET["amplitude_percentile"]))
    prom = max(DET["prominence_sigmas"] * sigma,
               DET["prominence_amplitude_fraction"] * scale)
    if prom <= 0:
        return np.array([], dtype=int)

    idx, _ = find_peaks(
        seg,
        prominence=prom,
        height=DET["height_amplitude_fraction"] * scale,
        distance=max(2, int(round(DET["separation_fraction"] * min_axle_separation(v_kmh)))),
    )
    return idx + i0

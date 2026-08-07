"""Phase 4 — sensor self-diagnosis features.

This is the project's actual goal (briefing §2, WP3): replace *hardware*
redundancy (4 FBGs per head) with *diagnostic* redundancy (2 FBGs), by proving
a sensor can tell you when it has stopped being trustworthy.

Diagnostic chain. Not safety-rated. ML is permitted here and nowhere near the
counting path.

Feature design follows one rule: **prefer features that need no history and no
calibration**, because a deployed head cannot wait fifteen sessions to notice
its own bond failing. Features are grouped by what they require:

  self / single-passage  — computable from one train, no memory, no neighbour.
                           The strongest form of self-diagnosis.
  self / trend           — needs the sensor's own history.
  pair                   — needs the other FBG in the same head. Still within
                           the 2-FBG budget, since direction sensing already
                           requires two.

All time-domain features are normalised by `samples_per_peak`, so they are
speed-invariant by construction (briefing §3.4, §10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .baseline import quiet_noise_floor
from .detect import axle_peaks, passage_window, samples_per_peak
from .diagnostic import PAD_OF_CHANNEL, head_offsets
from .speed import HEAD_BLOCKS, SpeedResult, head_signal

FS = CONFIG["acquisition"]["fs_hz"]

# Ground truth for validation, from briefing §8.
DEGRADING = {6, 9, 11}
BORDERLINE = {7}          # deck lists PAD 7 both as stable and as declining to ~0.75
HEALTHY = {1, 2, 3, 4, 5, 8, 10, 12}


@dataclass
class SensorFeatures:
    channel: str
    pad: int
    head: int
    n_axles: int
    # --- amplitude
    pos_pm: float
    neg_pm: float
    p2p_pm: float
    # --- shape, self / single-passage, dimensionless
    neg_over_pos: float      # |trough| / peak. A partially bonded pad is
                             # mechanically freer and over-responds downward.
    fwhm_norm: float         # full width at half max / samples_per_peak
    asymmetry: float         # (fall - rise) / (fall + rise) about the peak
    # --- noise, self / single-passage
    noise_pm: float
    snr: float
    # --- head-level
    count_disagree: int


def _shape(seg: np.ndarray, spp: float) -> tuple[float, float]:
    """(FWHM / samples_per_peak, rise-fall asymmetry) for one axle window."""
    if seg.size < 5:
        return np.nan, np.nan
    k = int(np.argmax(seg))
    top = seg[k]
    if top <= 0:
        return np.nan, np.nan
    half = 0.5 * top
    i = k
    while i > 0 and seg[i] > half:
        i -= 1
    j = k
    while j < len(seg) - 1 and seg[j] > half:
        j += 1
    rise, fall = k - i, j - k
    if rise + fall <= 0:
        return np.nan, np.nan
    return (rise + fall) / spp, (fall - rise) / (rise + fall)


def sensor_features(y: pd.DataFrame, spd: SpeedResult) -> list[SensorFeatures]:
    """Per-sensor feature vector for one record.

    `y` must be baseline-removed and in pm. Axle timing is transferred from the
    best-conditioned head — never self-detected — for the reason in A-16: a
    degrading sensor that finds its own peaks reports the amplitude of whatever
    noise looked most peak-like, biased upward, hiding the failure.
    """
    v = spd.v_kmh
    spp = samples_per_peak(v)
    half = max(3, int(round(CONFIG["detect"]["amplitude_window_fraction"] * spp)))
    wide = max(5, int(round(1.0 * spp)))       # wider window for shape fitting
    n = len(y)

    qual = {}
    for h in HEAD_BLOCKS:
        hs = head_signal(y, h)
        w = passage_window(hs, v)
        if w is None:
            qual[h] = -np.inf
            continue
        quiet = np.concatenate([hs[: w[0]], hs[w[1]:]])
        nz = quiet_noise_floor(quiet) if quiet.size > FS else quiet_noise_floor(hs)
        qual[h] = float(np.percentile(hs[w[0]:w[1]], 99.0) / nz) if nz > 0 else -np.inf
    ref_head = max(qual, key=qual.get)
    ref_sig = head_signal(y, ref_head)
    ref_idx = axle_peaks(ref_sig, v, window=passage_window(ref_sig, v))
    off = head_offsets(spd)
    n_ref = len(ref_idx)

    out: list[SensorFeatures] = []
    for head, chans in HEAD_BLOCKS.items():
        idx = ref_idx + (off[head] - off[ref_head])
        idx = idx[(idx >= 0) & (idx < n)]
        hs = head_signal(y, head)
        win = passage_window(hs, v)
        n_indep = len(axle_peaks(hs, v, window=win))
        for ch in chans:
            sig = y[ch].to_numpy()
            if win is not None:
                quiet = np.concatenate([sig[: win[0]], sig[win[1]:]])
                noise = quiet_noise_floor(quiet) if quiet.size > FS else quiet_noise_floor(sig)
            else:
                noise = quiet_noise_floor(sig)

            pos, neg, fw, asy = [], [], [], []
            for i in idx:
                a, b = max(0, i - half), min(n, i + half + 1)
                s = sig[a:b]
                if s.size:
                    pos.append(float(s.max()))
                    neg.append(float(s.min()))
                a2, b2 = max(0, i - wide), min(n, i + wide + 1)
                f, g = _shape(sig[a2:b2], spp)
                if np.isfinite(f):
                    fw.append(f)
                    asy.append(g)

            if not pos:
                continue
            p = float(np.median(pos))
            q = float(np.median(neg))
            out.append(SensorFeatures(
                channel=ch, pad=PAD_OF_CHANNEL[ch], head=head, n_axles=len(idx),
                pos_pm=p, neg_pm=q, p2p_pm=p - q,
                neg_over_pos=abs(q) / p if p > 0 else np.nan,
                fwhm_norm=float(np.median(fw)) if fw else np.nan,
                asymmetry=float(np.median(asy)) if asy else np.nan,
                noise_pm=noise, snr=p / noise if noise > 0 else np.nan,
                count_disagree=n_indep - n_ref,
            ))
    return out


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add same-head pair-agreement features.

    Within the 2-FBG budget: direction sensing already requires two gratings per
    head, so their mutual agreement is free. Each sensor is compared against the
    MEDIAN of the other members of its head, which is robust to the comparison
    partner being the faulty one.
    """
    df = df.copy()
    grp = df.groupby(["name", "head"])["p2p_pm"]
    tot = grp.transform("sum")
    cnt = grp.transform("count")
    # leave-one-out mean of the rest of the head
    others = (tot - df["p2p_pm"]) / (cnt - 1).replace(0, np.nan)
    df["pair_ratio"] = df["p2p_pm"] / others
    df["pair_log2"] = np.log2(df["pair_ratio"].replace(0, np.nan))
    return df


def label(pad: int) -> str:
    if pad in DEGRADING:
        return "degrading"
    if pad in BORDERLINE:
        return "borderline"
    return "healthy"

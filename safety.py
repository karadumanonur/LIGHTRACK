"""Phase 2 — the SAFETY chain: deterministic axle counting.

SIL4-candidate path. Everything here is thresholds, derivatives and voting;
nothing is learned, nothing is stochastic, and every decision is traceable to a
config value with a physical justification. ML belongs in the diagnostic chain
(`health.py`) and nowhere near this file.

**This chain shares nothing with the diagnostic chain below baseline removal.**
It has its own config block (`[safety]`), its own low-pass filter and its own
thresholds. Briefing §7.5 quantifies why: raising the detection threshold
improves axle counting and degrades anomaly detection, with an inflection past
which it hurts both. One filter cannot serve both jobs.

Two algorithms are implemented because the five-month field trial in briefing
§7.4 reached 100 % counting with zero phantom axles by running both and taking
a solution set — each catches what the other misses. Their documented failure
modes are complementary:

  * X-crossing fails on rare vehicle types whose troughs are high and peaks low.
  * D-crossing fails on a stopped train, whose noise peak has enough samples to
    satisfy the point-count rule. X-crossing correctly rejects that one.

Every time-domain parameter is speed-adaptive (briefing §3.4, §10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from .config import CONFIG

FS = CONFIG["acquisition"]["fs_hz"]
SAF = CONFIG["safety"]


def peak_duration_s(v_kmh: float) -> float:
    """Briefing §3.4: an axle peak spans ~2.2 m of track."""
    return CONFIG["geometry"]["peak_track_extent_m"] / (v_kmh / 3.6)


def samples_per_peak(v_kmh: float) -> float:
    return FS * peak_duration_s(v_kmh)


def lowpass_cutoff_hz(v_kmh: float) -> float:
    """Speed-adaptive cutoff. At 60 km/h this reproduces the reference 10 Hz."""
    return SAF["lpf_cycles_per_peak"] / peak_duration_s(v_kmh)


def lowpass(sig: np.ndarray, v_kmh: float) -> np.ndarray:
    """Zero-phase Butterworth. Zero-phase matters: a causal filter would shift
    crossing times and corrupt the head-to-head time-of-flight.

    Optional half-wave rectification first (`rectify_before_filter`). A-39 left
    one residual path by which compression-lobe depth still reaches the
    detector: the lobe sits adjacent to the axle peak, so low-passing the signed
    trace lets it pull the filtered peak down, altering width and prominence
    before any threshold is applied. Clipping negatives to zero removes the lobe
    from the filter's input entirely. The cost is harmonic distortion from the
    rectifier, which the low-pass then has to absorb - so this is measured, not
    assumed (see A-41).
    """
    if SAF.get("rectify_before_filter", False):
        sig = np.maximum(sig, 0.0)
    fc = lowpass_cutoff_hz(v_kmh)
    wn = min(0.99, fc / (FS / 2))
    b, a = butter(SAF["lpf_order"], wn, btype="low")
    return filtfilt(b, a, sig)


@dataclass
class Levels:
    """Robust peak/trough levels and the hysteresis band derived from them."""
    trough: float
    peak: float
    X: float
    thr_rise: float
    thr_fall: float
    d_cut: float


def levels(sig: np.ndarray, v_kmh: float) -> Levels | None:
    """Estimate X = (lowest peak) - (highest trough) within a passage.

    Two things this must get right, both learned the hard way:

    1. **Only significant maxima count as peaks.** A first implementation took
       percentiles over *every* local maximum. The passage window includes
       padding and the quiet gaps between bogies, where the filtered trace
       ripples around zero and contributes dozens of near-zero extrema. Those
       dragged the peak level down and the trough level up, collapsing X to
       almost nothing, and both detectors then fired on noise: within-head
       unanimity came out at 0-1 %. Peaks are gated at a fraction of the
       passage maximum first.

    2. **Troughs are the minima BETWEEN consecutive peaks.** The briefing's
       "maximum of the troughs" means the inter-axle valleys, not any local
       minimum anywhere in the record.

    Quantiles rather than true min/max, so one outlier cannot set the band.
    """
    M = float(sig.max())
    if M <= 0:
        return None
    d = np.diff(sig)
    up = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1     # local maxima
    if up.size < 2:
        return None
    up = up[sig[up] > SAF["x_peak_floor_fraction"] * M]
    if up.size < 2:
        return None

    valleys = [float(sig[a:b + 1].min()) for a, b in zip(up[:-1], up[1:])]
    if not valleys:
        return None

    peak = float(np.percentile(sig[up], SAF["x_peak_percentile"]))
    trough = float(np.percentile(valleys, SAF["x_trough_percentile"]))
    X = peak - trough
    if X <= 0:
        return None

    if SAF["anchor_mode"] == "peak":
        # Thresholds referenced to the POSITIVE-PEAK level only, with no
        # statistic that touches the trough. The trough-anchored form leaks
        # compression-lobe depth into the band, and lobe depth is a function of
        # sleeper-bay position (A-25/A-26): PAD 3, the deepest lobe of the
        # twelve at 0.76, over-counts on 26 % of passages where its head-mates
        # manage 0-4 %, correlation +0.823 across the head.
        return Levels(
            trough=trough, peak=peak, X=X,
            thr_rise=SAF["peak_rise_fraction"] * peak,
            thr_fall=SAF["peak_fall_fraction"] * peak,
            d_cut=SAF["d_lower_cut_fraction"] * peak,
        )
    return Levels(
        trough=trough, peak=peak, X=X,
        thr_rise=trough + SAF["x_rise_fraction"] * X,
        thr_fall=trough + SAF["x_fall_fraction"] * X,
        # The D-crossing lower cut is anchored to the PEAK level, not to the
        # trough. Anchoring it to the trough inherits the depth of the
        # compression lobe, which is a property of sleeper-bay position rather
        # than of the wheel: PAD 3 (+7.5 cm off-centre) has a trough 14 pm
        # deeper than its neighbours, which pushed its cut down to ~0 pm and
        # let the negative-lobe structure through as extra "axles" - 18 counted
        # against a true 12. Referenced to the peak, the cut sits where the
        # briefing intends: above the valley between two bogies.
        d_cut=SAF["d_lower_cut_fraction"] * peak,
    )


def x_crossing(sig: np.ndarray, v_kmh: float,
               lv: Levels | None = None) -> np.ndarray:
    """Hysteresis dual threshold. Returns confirmed peak indices.

    A peak is confirmed by a rising edge through `thr_rise` followed by a
    falling edge through `thr_fall`. Peaks whose rise->fall interval is shorter
    than the speed-scaled minimum duration are rejected as noise.
    """
    lv = lv or levels(sig, v_kmh)
    if lv is None:
        return np.array([], dtype=int)
    t_min = SAF["x_min_duration_fraction"] * samples_per_peak(v_kmh)

    out, armed, i_rise = [], False, 0
    for i, v in enumerate(sig):
        if not armed and v > lv.thr_rise:
            armed, i_rise = True, i
        elif armed and v < lv.thr_fall:
            if (i - i_rise) >= t_min:
                out.append(i_rise + int(np.argmax(sig[i_rise:i + 1])))
            armed = False
    return np.asarray(out, dtype=int)


def d_crossing(sig: np.ndarray, v_kmh: float,
               lv: Levels | None = None) -> np.ndarray:
    """Derivative sign-change detector with a lower cut.

    The lower cut is applied FIRST and is not optional: without it the local
    maximum in the valley between two bogies is read as a wheel (briefing §7.4).
    A peak is confirmed only if it is supported by enough consecutive samples —
    the reference used 6 at 250 Hz; here that is a floor and the requirement
    scales with the peak width.
    """
    lv = lv or levels(sig, v_kmh)
    if lv is None:
        return np.array([], dtype=int)
    need = max(SAF["d_min_points"],
               int(round(SAF["d_points_per_peak_fraction"] * samples_per_peak(v_kmh))))

    above = sig > lv.d_cut
    out = []
    i = 0
    n = len(sig)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        seg = sig[i:j]
        if seg.size >= need:
            k = int(np.argmax(seg))
            d = np.diff(seg)
            # require at least 2 rising and 2 falling derivative samples
            if np.count_nonzero(d[:k] > 0) >= 2 and np.count_nonzero(d[k:] < 0) >= 2:
                out.append(i + k)
        i = j
    return np.asarray(out, dtype=int)


@dataclass
class CountResult:
    n_x: int
    n_d: int
    agree: bool
    count: int              # the solution-set answer
    X_pm: float
    cutoff_hz: float
    idx_x: np.ndarray
    idx_d: np.ndarray


def count_axles(sig_pm: np.ndarray, v_kmh: float) -> CountResult:
    """Run both detectors on one baseline-removed trace and combine.

    Solution set: when the two agree, the count is returned with agreement
    asserted. When they disagree the DISAGREEMENT IS THE OUTPUT — it is not
    silently resolved. A counting head that cannot decide must say so rather
    than emit a number that might be a phantom axle.
    """
    f = lowpass(sig_pm, v_kmh)
    lv = levels(f, v_kmh)
    ix = x_crossing(f, v_kmh, lv)
    idd = d_crossing(f, v_kmh, lv)
    return CountResult(
        n_x=len(ix), n_d=len(idd), agree=len(ix) == len(idd),
        count=len(ix) if len(ix) == len(idd) else max(len(ix), len(idd)),
        X_pm=lv.X if lv else float("nan"),
        cutoff_hz=lowpass_cutoff_hz(v_kmh),
        idx_x=ix, idx_d=idd,
    )

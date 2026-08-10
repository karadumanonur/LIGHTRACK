"""Speed and direction by geometry-constrained time-of-flight.

Promoted from scripts/07 after Phase 0 showed both naive estimators fail:
free cross-correlation aliases by one carriage length on the quasi-periodic
axle sequence, and amplitude onset is biased by per-channel sensitivity
(ASSUMPTIONS A-13).

The fix is structural, not a tuning parameter. The three heads sit at FIXED
distances, so one unknown — the slowness — predicts BOTH inter-head lags at
once. An alias must then coincidentally satisfy the 15:10 ratio in order to
win, which it almost never does.

Every record carries a fit `score`. Records are rejected on score, never on
speed plausibility: rejecting on plausibility would hide exactly the outliers
that reveal a broken estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .loader import CH_COLS

FS = CONFIG["acquisition"]["fs_hz"]
D12 = CONFIG["geometry"]["head1_to_head2_m"]     # 15 m  Head1 <-> Head2
D23 = CONFIG["geometry"]["head2_to_head3_m"]     # 10 m  Head2 <-> Head3
MIN_SCORE = CONFIG["speed"]["min_fit_score"]

# Spectral-order channel blocks, established in Phase 0 (DATA_DICTIONARY §10.2).
HEAD_BLOCKS = {3: CH_COLS[0:4], 2: CH_COLS[4:8], 1: CH_COLS[8:12]}


@dataclass
class SpeedResult:
    v_kmh: float
    score: float
    alt_score: float
    head1_first: bool          # True: train reaches Head 1 before Head 3
    lag_ab: int                # samples, Head3-block vs Head2-block
    lag_bc: int                # samples, Head2-block vs Head1-block
    ok: bool                   # score cleared the gate

    @property
    def samples_per_peak(self) -> float:
        """Briefing §3.4: peak spans ~2.2 m of track -> 3960 / v_kmh at 500 Hz."""
        k = CONFIG["geometry"]["samples_per_peak_numerator_at_500hz"]
        return k / self.v_kmh


def head_signal(y: pd.DataFrame, head: int) -> np.ndarray:
    """Robustly averaged trace for one head.

    Each channel is scaled to unit MAD before averaging so that a loud sensor
    (ch04 / PAD 9) and a nearly dead one (ch02 / PAD 11) contribute equally.
    Without this the head's timing would track its loudest member, which is
    exactly the sensor most likely to be misbehaving.
    """
    m = y[HEAD_BLOCKS[head]].to_numpy()
    mad = np.median(np.abs(m - np.median(m, axis=0)), axis=0) * 1.4826
    mad[mad <= 0] = 1.0
    return (m / mad).mean(axis=1)


def _xcorr_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Normalised correlation at a fixed lag.

    SIGN CONVENTION: for lag > 0 we align a[lag:] with b[:-lag], so a[lag+i] ~
    b[i] — the event seen at index i in b turns up later, at lag+i, in a.
    **Positive lag means `a` is DELAYED relative to `b`.**
    """
    n = len(a)
    if lag >= 0:
        x, z = a[lag:], b[: n - lag]
    else:
        x, z = a[: n + lag], b[-lag:]
    if len(x) < FS:
        return 0.0
    x = x - x.mean()
    z = z - z.mean()
    d = np.linalg.norm(x) * np.linalg.norm(z)
    return float(np.dot(x, z) / d) if d > 1e-9 else 0.0


def _windowed_heads(y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three head traces, cropped to the loudest 20 s of the record."""
    a = head_signal(y, 3)      # ch01-04
    b = head_signal(y, 2)      # ch05-08
    c = head_signal(y, 1)      # ch09-12
    env = pd.Series(np.abs(a) + np.abs(b) + np.abs(c)).rolling(
        int(0.5 * FS), center=True, min_periods=1).mean().to_numpy()
    ctr = int(np.argmax(env))
    half = int(10 * FS)
    sl = slice(max(0, ctr - half), min(len(a), ctr + half))
    return a[sl], b[sl], c[sl]


def refine_lags(y: pd.DataFrame, spd: "SpeedResult",
                frac: float = 0.25) -> tuple[int, int]:
    """Re-measure both inter-head lags FREELY, near the constrained solution.

    `estimate` forces lag_bc / lag_ab = 15/10 by construction, so it can never
    reveal that the true head spacing differs from nominal. This searches each
    lag independently within +/-`frac` of its predicted value: wide enough to
    move off the imposed ratio, narrow enough that the carriage-length alias
    (which broke the free estimator in Phase 0) stays out of range.

    The ratio of the two refined lags is a direct measurement of the head
    spacing ratio, independent of speed.
    """
    a, b, c = _windowed_heads(y)
    out = []
    for x, z, pred in ((a, b, spd.lag_ab), (b, c, spd.lag_bc)):
        if pred == 0:
            out.append(0)
            continue
        lo, hi = sorted((int(pred * (1 - frac)), int(pred * (1 + frac))))
        cand = range(lo, hi + 1)
        best = max(cand, key=lambda L: _xcorr_at(x, z, L))
        out.append(int(best))
    return out[0], out[1]


def xcorr_curve(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised cross-correlation at every lag in [-max_lag, +max_lag].

    Exact per-lag normalisation (means and norms computed over the actual
    overlap, not globally) via prefix sums, with the dot products from an FFT
    correlation. Same convention as `_xcorr_at`: positive lag means `a` is
    delayed relative to `b`.

    Needed because the accelerated solver searches a 2-D (v0, a) grid, which is
    thousands of lag evaluations per record — recomputing each one directly is
    ~10^9 operations per record and not tractable.
    """
    from scipy.signal import correlate

    n = len(a)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ca = np.concatenate([[0.0], np.cumsum(a)])
    ca2 = np.concatenate([[0.0], np.cumsum(a * a)])
    cb = np.concatenate([[0.0], np.cumsum(b)])
    cb2 = np.concatenate([[0.0], np.cumsum(b * b)])
    dots = correlate(a, b, mode="full", method="fft")      # index n-1 == lag 0

    out = np.zeros(2 * max_lag + 1)
    for k, L in enumerate(range(-max_lag, max_lag + 1)):
        if L >= 0:
            i0, i1, j0, j1 = L, n, 0, n - L
        else:
            i0, i1, j0, j1 = 0, n + L, -L, n
        m = i1 - i0
        if m < FS:
            continue
        sx, sx2 = ca[i1] - ca[i0], ca2[i1] - ca2[i0]
        sz, sz2 = cb[j1] - cb[j0], cb2[j1] - cb2[j0]
        vx = sx2 - sx * sx / m
        vz = sz2 - sz * sz / m
        if vx <= 0 or vz <= 0:
            continue
        d = dots[n - 1 + L] - sx * sz / m
        out[k] = d / np.sqrt(vx * vz)
    return out


def _arrival_times(v0_kmh: float, accel: float, dists: list[float]) -> list[float] | None:
    """Arrival time at each distance for uniform acceleration, or None if the
    train would stop before reaching the last head."""
    v0 = v0_kmh / 3.6
    ts = []
    for d in dists:
        disc = v0 * v0 + 2.0 * accel * d
        if disc <= 0:
            return None
        ts.append((np.sqrt(disc) - v0) / accel if accel != 0 else d / v0)
    return ts


@dataclass
class AccelResult:
    v0_kmh: float
    accel_ms2: float
    score: float
    head1_first: bool
    lag_ab: int
    lag_bc: int

    @property
    def v_exit_kmh(self) -> float:
        v0 = self.v0_kmh / 3.6
        disc = v0 * v0 + 2 * self.accel_ms2 * (D12 + D23)
        return 3.6 * np.sqrt(max(disc, 0.0))


def estimate_accelerated(y: pd.DataFrame,
                         v_min: float = 15.0, v_max: float = 200.0,
                         v_step: float = 2.5,
                         a_max: float = 2.5, a_step: float = 0.125) -> AccelResult:
    """Two-parameter time-of-flight: entry speed and uniform acceleration.

    The constant-speed solver forces lag_bc/lag_ab = 15/10. That ratio only
    holds at constant speed — a train braking or accelerating between heads
    violates it, the fit score collapses, and the record is discarded. But the
    deviation is a MEASUREMENT of acceleration, not a failure: the geometry is
    known, so the ratio carries the kinematics.

    Head positions along travel are 0 / 15 / 25 m in the Head1→Head3 direction
    and 0 / 10 / 25 m in the reverse, so both directions are handled by the
    distance list rather than by a special case.
    """
    a_sig, b_sig, c_sig = _windowed_heads(y)
    max_lag = int(4.0 * FS)
    cab = xcorr_curve(a_sig, b_sig, max_lag)     # head3 vs head2
    cbc = xcorr_curve(b_sig, c_sig, max_lag)     # head2 vs head1

    def look(curve, lag):
        i = int(round(lag)) + max_lag
        return curve[i] if 0 <= i < len(curve) else -1.0

    best = None
    for v0 in np.arange(v_min, v_max + v_step, v_step):
        for acc in np.arange(-a_max, a_max + a_step, a_step):
            for h1_first in (True, False):
                dists = [0.0, D12, D12 + D23] if h1_first else [0.0, D23, D12 + D23]
                ts = _arrival_times(v0, acc, dists)
                if ts is None:
                    continue
                if h1_first:
                    t1, t2, t3 = ts
                else:
                    t3, t2, t1 = ts
                lag_bc = (t2 - t1) * FS
                lag_ab = (t3 - t2) * FS
                if abs(lag_ab) > max_lag or abs(lag_bc) > max_lag:
                    continue
                s = look(cab, lag_ab) + look(cbc, lag_bc)
                if best is None or s > best[0]:
                    best = (s, v0, acc, h1_first, lag_ab, lag_bc)

    s, v0, acc, h1, lab, lbc = best
    return AccelResult(v0_kmh=float(v0), accel_ms2=float(acc), score=float(s),
                       head1_first=bool(h1), lag_ab=int(round(lab)),
                       lag_bc=int(round(lbc)))


def estimate(y: pd.DataFrame,
             v_min: float = 20.0, v_max: float = 200.0,
             v_step: float = 0.25) -> SpeedResult:
    """Estimate speed and direction from a baseline-removed 12-channel frame."""
    a, b, c = _windowed_heads(y)

    grid = np.arange(v_min, v_max + v_step, v_step)
    best = best_alt = None
    for v in grid:
        tau = FS / (v / 3.6)                       # samples per metre
        for sign in (+1, -1):
            # H1 (established): Head3-block <-> Head2-block spans 10 m,
            #                   Head2-block <-> Head1-block spans 15 m.
            s = _xcorr_at(a, b, int(round(sign * D23 * tau))) + \
                _xcorr_at(b, c, int(round(sign * D12 * tau)))
            if best is None or s > best[0]:
                best = (s, v, sign)
            # H2 (counter-hypothesis): the head order is reversed.
            s2 = _xcorr_at(a, b, int(round(sign * D12 * tau))) + \
                 _xcorr_at(b, c, int(round(sign * D23 * tau)))
            if best_alt is None or s2 > best_alt[0]:
                best_alt = (s2, v, sign)

    score, v, sign = best
    tau = FS / (v / 3.6)
    return SpeedResult(
        v_kmh=float(v),
        score=float(score),
        alt_score=float(best_alt[0]),
        # sign=+1 makes the Head3 block lag the Head2 block, i.e. Head 1 first.
        head1_first=bool(sign > 0),
        lag_ab=int(round(sign * D23 * tau)),
        lag_bc=int(round(sign * D12 * tau)),
        ok=bool(score >= MIN_SCORE),
    )

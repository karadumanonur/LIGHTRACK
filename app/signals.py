"""What the interface draws, computed from one record on demand.

Everything here works on the RAW wavelength with its baseline left in place,
because that is what the instrument wrote and what the window is meant to show.
The engines remove their own baselines their own way; that belongs to them, not
to this.

Two decisions worth stating, both about not lying with a picture:

  the overview keeps a minimum and a maximum per bin, never a mean. An axle
  peak is a hundredth of the record wide, so a mean over eighty samples erases
  it, and a picture of a train with no axles in it is worse than no picture.

  the spectra are returned in full and windowed in the browser. Zooming into a
  spectrum must show more of it, not the same points stretched, so nothing is
  thinned below what the panel can resolve at its deepest zoom.

Empirical mode decomposition is written out rather than imported: no EMD
package is a dependency of this project, and the sifting loop is twenty lines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import periodogram, welch

BAND = (0.5, 60.0)
FMAX = 60.0


def extrema(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(y)
    hi = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    lo = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    return hi, lo


def sift(y: np.ndarray, rounds: int = 8) -> np.ndarray:
    h = y.astype(float).copy()
    n = np.arange(len(y))
    for _ in range(rounds):
        hi, lo = extrema(h)
        if len(hi) < 3 or len(lo) < 3:
            break
        top = CubicSpline(np.r_[0, hi, len(h) - 1],
                          np.r_[h[hi[0]], h[hi], h[lo[-1]]])(n)
        bot = CubicSpline(np.r_[0, lo, len(h) - 1],
                          np.r_[h[lo[0]], h[lo], h[hi[-1]]])(n)
        h = h - 0.5 * (top + bot)
    return h


def emd(y: np.ndarray, modes: int = 4) -> list[np.ndarray]:
    out, res = [], np.asarray(y, dtype=float).copy()
    for _ in range(modes):
        hi, lo = extrema(res)
        if len(hi) < 3 or len(lo) < 3:
            break
        m = sift(res)
        out.append(m)
        res = res - m
    return out


def envelope(y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, len(y) // bins)
    m = np.asarray(y[:(len(y) // k) * k], dtype=float).reshape(-1, k)
    return m.min(axis=1), m.max(axis=1)


def thin(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) <= n:
        return np.asarray(y, dtype=float)
    k = len(y) // n
    return np.asarray(y[:k * n], dtype=float).reshape(n, k).mean(axis=1)


def round_to(y, n: int = 4) -> list[float]:
    return [round(float(v), n) for v in y]


def band_power(seg: np.ndarray, fs: float) -> float:
    f, p = periodogram(seg, fs=fs)
    k = (f >= BAND[0]) & (f <= BAND[1])
    return float(np.trapezoid(p[k], f[k]))


def remove_baseline(col: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    """The counter's own baseline: a trailing median, then picometres.

    A rolling median over a second is wider than any axle and narrower than any
    drift, so it takes the slow wander out and leaves the axles standing.

    Pandas does the median, not a strided view and numpy: the view builds a
    seventy-thousand by five-hundred array per sensor, which is thirty-six
    million numbers to sort twelve times a record, and measuring the delivery
    that way takes three hours instead of four minutes.
    """
    n = max(3, int(window_s * fs))
    x = np.asarray(col, dtype=float)
    base = pd.Series(x).rolling(n, min_periods=1).median().to_numpy()
    return (x - base) * 1000.0


def passage_peaks(col: np.ndarray, fs: float, marks_s, half_s: float = 0.12
                  ) -> tuple[float, float]:
    """How tall this sensor's axle peaks are, and how much they vary.

    The axles are already located — the counting engine found them — so this
    does not look for peaks again. It reads the height of the baseline-removed
    signal around each axle the engine marked, on THIS sensor. Every sensor is
    therefore measured on the same axles of the same train at the same moment,
    which is what makes twelve numbers comparable at all.

    Only the positive excursion counts. An axle presses the rail down and the
    grating stretches; the number wanted is how far above the baseline that
    goes, so this is a maximum and never an absolute value or a peak-to-peak.
    Nothing below zero enters it.

    The mean is the size of the response, what a train writes on this grating.
    The spread is how much that varies from axle to axle inside one passage:
    small on a healthy sensor under an even train, large where the bond is
    going and the response depends on which wheel is overhead.

    With no marks there is nothing to measure a response to, so the whole
    record is taken and the largest excursions stand in for it — an honest
    number for a file with no train in it, which is near zero.
    """
    y = remove_baseline(col, fs)
    if len(marks_s):
        half = max(2, int(half_s * fs))
        vals = []
        for t in marks_s:
            i = int(t * fs)
            a, b = max(0, i - half), min(len(y), i + half + 1)
            if b > a:
                vals.append(max(0.0, float(np.max(y[a:b]))))
        if vals:
            return float(np.mean(vals)), float(np.std(vals))
    k = max(1, len(y) // 200)
    tops = np.max(y[:(len(y) // k) * k].reshape(-1, k), axis=1)
    tops = np.clip(np.sort(tops)[-20:], 0.0, None)
    return float(np.mean(tops)), float(np.std(tops))


def spectra(col: np.ndarray, fs: float, points: int = 900) -> dict:
    base = float(np.median(col))
    pm = (np.asarray(col, dtype=float) - base) * 1000.0
    f_psd, pxx = welch(pm, fs=fs, nperseg=2048, noverlap=1024)
    k = f_psd <= FMAX
    f_fft = np.fft.rfftfreq(len(pm), 1 / fs)
    j = f_fft <= FMAX
    sp = np.abs(np.fft.rfft(pm * np.hanning(len(pm))))
    return dict(
        base=round(base, 4),
        psd=dict(f=round_to(thin(f_psd[k], points), 3),
                 v=[float(f"{v:.4g}") for v in thin(pxx[k], points)],
                 peak=round(float(f_psd[k][int(np.argmax(pxx[k]))]), 2)),
        fft=dict(f=round_to(thin(f_fft[j], points), 3),
                 v=[float(f"{v:.4g}") for v in thin(sp[j], points)]))


def overview(col: np.ndarray, fs: float, bins: int = 900) -> dict:
    lo, hi = envelope(col, bins)
    t = np.arange(len(lo)) * (len(col) / len(lo)) / fs
    return dict(t=round_to(t, 3), lo=round_to(lo), hi=round_to(hi),
                span=[round(float(lo.min()), 4), round(float(hi.max()), 4)])


def modes(col: np.ndarray, fs: float, span: tuple[float, float],
          step: int = 8, n: int = 4) -> dict:
    a, b = int(span[0] * fs), int(span[1] * fs)
    a, b = max(0, a), min(len(col), b)
    pm = (np.asarray(col[a:b], dtype=float)
          - float(np.median(col[a:b]))) * 1000.0
    out = emd(pm, modes=n)
    return dict(span=[round(a / fs, 2), round(b / fs, 2)],
                t=round_to(np.arange(a, b, step) / fs, 3),
                imf=[round_to(m[::step], 2) for m in out])

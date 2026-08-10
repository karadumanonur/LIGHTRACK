"""Baseline (thermal drift) removal.

Shared by both chains — this is the ONE stage they are allowed to share,
because it removes a physical artefact (temperature) rather than shaping the
signal band. Everything downstream of here must be duplicated per chain.

Physics: a single FBG cannot separate strain from temperature (briefing §3.3).
A 50 °C seasonal swing is ~500 pm, larger than the entire train signal. The
saving grace is timescale: a passage lasts seconds, drift takes hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG

NM_TO_PM = CONFIG["units"]["nm_to_pm"]


def rolling_median_baseline(x: pd.DataFrame | pd.Series,
                            window: int | None = None) -> pd.DataFrame | pd.Series:
    """Subtract a centred rolling median and convert nm -> pm.

    `window` defaults to the config value (500 samples = 1 s at 500 Hz), which
    reproduces what the project itself appears to do: slide 21 is titled
    "λ − λΔ (500)" (briefing §5.4).

    Median rather than mean: a mean baseline is dragged upward by the passage
    itself, which flattens peaks by several percent — precisely the amplitude we
    are trying to measure in the diagnostic chain.
    """
    w = int(window if window is not None else CONFIG["baseline"]["rolling_window_samples"])
    base = x.rolling(w, center=True, min_periods=1).median()
    return (x - base) * NM_TO_PM


def quiet_noise_floor(y: np.ndarray) -> float:
    """Robust sigma of the inter-train baseline, in whatever unit y is in.

    MAD-based, so a train passage occupying a few percent of the record does not
    inflate it. This is the denominator for every SNR-like quantity downstream
    and a diagnostic feature in its own right (briefing §9, item 2).
    """
    med = np.median(y)
    return float(np.median(np.abs(y - med)) * 1.4826)

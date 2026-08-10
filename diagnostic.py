"""Diagnostic chain: per-sensor peak amplitude and health features.

NOT safety-rated (briefing §2, §7). ML would be permitted here; nothing in this
module is a candidate for the SIL4 counting path.

The central measurement is per-axle peak amplitude per channel. Two design
decisions matter, and both were forced by observed failures.

**1. Axle TIMES never come from the channel being measured.**
A collapsed sensor (PAD 11 after measurement 3) has no peaks above noise. If
each channel found its own, a dead sensor would report the amplitude of
whatever noise excursion looked most peak-like — biased *upward*, hiding the
very failure WP3 exists to detect.

**2. Axle times come from ONE reference head, transferred by time-of-flight —
not from each head's own mean.**
Deriving them per head looked equivalent and was not. Head 3 contains PAD 9
(large negative lobes) and PAD 11 (collapsed), and its robust mean produced
13–64 "axles" per record where Heads 1 and 2 both consistently found 12. The
surplus are noise times, and averaging amplitude over them drags the *healthy*
Head 3 sensors down: PAD 10 and PAD 12 both read ~0.15 in session 11 purely as
an artefact. Head 3 is the mounting-method test head, so this corrupted exactly
the comparison the experiment exists to make.

The fix uses machinery already validated in Phase 0: all three heads see the
same axles, separated by known distances, and `speed.estimate` already returns
both inter-head lags. Detect once on the best-conditioned head, shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .baseline import quiet_noise_floor
from .detect import axle_peaks, passage_window, samples_per_peak
from .loader import CH_COLS
from .speed import HEAD_BLOCKS, SpeedResult, head_signal

FS = CONFIG["acquisition"]["fs_hz"]
HEAD_OF_CHANNEL = dict(zip(CH_COLS, CONFIG["channel_to_pad"]["head_of_channel"]))
PAD_OF_CHANNEL = dict(zip(CH_COLS, CONFIG["channel_to_pad"]["pad_of_channel"]))
CH_OF_PAD = {p: c for c, p in PAD_OF_CHANNEL.items()}


@dataclass
class ChannelAmplitude:
    channel: str
    pad: int
    head: int
    n_axles: int
    pos_pm: float        # median positive peak height above baseline
    neg_pm: float        # median negative excursion around each axle
    pos_max_pm: float
    noise_pm: float      # MAD sigma of the quiet baseline
    snr: float
    ref_head: int        # head whose axle timing was used
    # Timing-transfer quality: where the true local maximum actually sat,
    # relative to the transferred index. Small = the transfer is accurate.
    # Approaching half the search window = the sampling point is on the flank
    # and the amplitude is being under-read.
    offset_med: float    # median signed offset, samples
    offset_abs_p90: float
    search_half: int     # half-width of the search window, samples
    # Head-level health feature (A-18): this head's OWN axle count minus the
    # reference head's. All heads see the same train, so a persistent surplus
    # means this head is manufacturing detections.
    n_indep: int
    count_disagree: int


def head_axle_times(y: pd.DataFrame, head: int, v_kmh: float) -> np.ndarray:
    """Axle sample indices for one head, from its own robust mean trace.

    Kept for diagnostics and cross-head agreement checks. It is NOT what
    `channel_amplitudes` uses — see the module docstring.
    """
    return axle_peaks(head_signal(y, head), v_kmh)


def head_offsets(spd: SpeedResult) -> dict[int, int]:
    """Sample offset of each head relative to Head 1, from the fitted geometry.

    Sign convention comes straight from `speed._xcorr_at`: `lag_ab` is the
    Head-3 block's delay behind the Head-2 block, `lag_bc` the Head-2 block's
    delay behind the Head-1 block. Both are signed, so this handles either
    travel direction without a special case.
    """
    return {1: 0, 2: spd.lag_bc, 3: spd.lag_bc + spd.lag_ab}


def _head_quality(y: pd.DataFrame, head: int, v_kmh: float) -> float:
    """Peak-to-noise of a head's mean trace. Picks the timing reference."""
    hs = head_signal(y, head)
    win = passage_window(hs, v_kmh)
    if win is None:
        return -np.inf
    quiet = np.concatenate([hs[: win[0]], hs[win[1]:]])
    n = quiet_noise_floor(quiet) if quiet.size > FS else quiet_noise_floor(hs)
    if n <= 0:
        return -np.inf
    return float(np.percentile(hs[win[0]:win[1]], 99.0) / n)


def channel_amplitudes(y: pd.DataFrame, spd: SpeedResult) -> list[ChannelAmplitude]:
    """Per-channel amplitude statistics for one record.

    `y` must already be baseline-removed and in pm.

    The noise floor is measured OUTSIDE the passage window. Including the
    passage inflates it by an order of magnitude on a healthy sensor and barely
    at all on a dead one, which would make SNR look *better* the more degraded
    the sensor is — the exact inversion this module exists to avoid.
    """
    v_kmh = spd.v_kmh
    half = max(2, int(round(CONFIG["detect"]["amplitude_window_fraction"]
                           * samples_per_peak(v_kmh))))
    n = len(y)

    # Reference head = best peak-to-noise. Chosen per record, on head-level
    # signal quality only, so it cannot be circular with the per-sensor
    # degradation being measured.
    quality = {h: _head_quality(y, h, v_kmh) for h in HEAD_BLOCKS}
    ref_head = max(quality, key=quality.get)
    ref_sig = head_signal(y, ref_head)
    ref_idx = axle_peaks(ref_sig, v_kmh, window=passage_window(ref_sig, v_kmh))

    off = head_offsets(spd)
    n_ref = len(ref_idx)
    out: list[ChannelAmplitude] = []

    for head, chans in HEAD_BLOCKS.items():
        idx = ref_idx + (off[head] - off[ref_head])
        idx = idx[(idx >= 0) & (idx < n)]
        hs = head_signal(y, head)
        win = passage_window(hs, v_kmh)
        n_indep = len(axle_peaks(hs, v_kmh, window=win))
        for ch in chans:
            sig = y[ch].to_numpy()
            if win is not None:
                quiet = np.concatenate([sig[: win[0]], sig[win[1]:]])
                noise = quiet_noise_floor(quiet) if quiet.size > FS else quiet_noise_floor(sig)
            else:
                noise = quiet_noise_floor(sig)
            if len(idx) == 0:
                out.append(ChannelAmplitude(ch, PAD_OF_CHANNEL[ch], head, 0,
                                            np.nan, np.nan, np.nan, noise, np.nan,
                                            ref_head, np.nan, np.nan, half,
                                            n_indep, n_indep - n_ref))
                continue
            pos, neg, offs = [], [], []
            for i in idx:
                a, b = max(0, i - half), min(n, i + half + 1)
                seg = sig[a:b]
                if seg.size:
                    pos.append(float(seg.max()))
                    neg.append(float(seg.min()))
                    offs.append(int(np.argmax(seg)) + a - int(i))
            p = float(np.median(pos))
            o = np.asarray(offs, dtype=float)
            out.append(ChannelAmplitude(
                channel=ch, pad=PAD_OF_CHANNEL[ch], head=head, n_axles=len(idx),
                pos_pm=p, neg_pm=float(np.median(neg)), pos_max_pm=float(np.max(pos)),
                noise_pm=noise, snr=p / noise if noise > 0 else np.nan,
                ref_head=ref_head,
                offset_med=float(np.median(o)),
                offset_abs_p90=float(np.percentile(np.abs(o), 90)),
                search_half=half,
                n_indep=n_indep, count_disagree=n_indep - n_ref,
            ))
    return out


def to_frame(rows: list[ChannelAmplitude], **extra) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    for k, v in extra.items():
        df[k] = v
    return df

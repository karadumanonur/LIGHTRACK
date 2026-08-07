"""Lightrack POC1 analysis package.

Current scope: loader, baseline removal, geometry-constrained speed/direction,
and the DIAGNOSTIC chain (axle timing + per-sensor amplitude).

The SAFETY chain — X-crossing and D-crossing counting, direction voting — is
deliberately absent until Phase 3, and when it arrives it gets its own config
block and its own filter. Briefing §7.5: the two chains need opposite operating
points and must never share a stage below baseline removal.
"""

from .config import load_config, CONFIG
from .loader import (
    FileMeta,
    build_index,
    parse_filename_utc,
    to_local,
    sniff_decimal,
    load_file,
    load_raw_array,
    scan_file,
    list_files,
    CH_COLS,
    TIME_COL,
)
from .baseline import rolling_median_baseline, quiet_noise_floor
from .speed import (
    estimate as estimate_speed, refine_lags, SpeedResult, head_signal, HEAD_BLOCKS,
)
from .detect import axle_peaks, samples_per_peak, min_axle_separation
from .diagnostic import (
    channel_amplitudes,
    head_axle_times,
    head_offsets,
    ChannelAmplitude,
    to_frame,
    PAD_OF_CHANNEL,
    CH_OF_PAD,
    HEAD_OF_CHANNEL,
)

__all__ = [
    "load_config", "CONFIG",
    "FileMeta", "build_index", "parse_filename_utc", "to_local", "sniff_decimal",
    "load_file", "load_raw_array", "scan_file", "list_files", "CH_COLS", "TIME_COL",
    "rolling_median_baseline", "quiet_noise_floor",
    "estimate_speed", "refine_lags", "SpeedResult", "head_signal", "HEAD_BLOCKS",
    "axle_peaks", "samples_per_peak", "min_axle_separation",
    "channel_amplitudes", "head_axle_times", "head_offsets",
    "ChannelAmplitude", "to_frame",
    "PAD_OF_CHANNEL", "CH_OF_PAD", "HEAD_OF_CHANNEL",
]

"""Lightrack — FBG axle counting, direction and speed on line 130.

Two detection chains live here and they are deliberately kept apart, because
they need opposite operating points and must never share a stage below baseline
removal:

    safety      X-crossing and D-crossing counting. Deterministic and
                explainable, and it is what the occupancy balance acts on.
    detect      prominence peaks. More sensitive, used for diagnosis and for
                the gap sequence.

Everything below baseline removal is shared: loading, drift removal, and the
geometry-constrained speed and direction solver.
"""

from .config import load_config, CONFIG
from .loader import (
    FileMeta,
    parse_filename_utc,
    to_local,
    sniff_decimal,
    sniff_column_order,
    load_file,
    scan_file,
    list_files,
    CH_COLS,
    TIME_COL,
    PAD_OF_CHANNEL,
    CH_OF_PAD,
)
from .baseline import rolling_median_baseline, quiet_noise_floor
from .speed import estimate as estimate_speed, SpeedResult, head_signal, HEAD_BLOCKS
from .detect import axle_peaks, samples_per_peak, min_axle_separation, has_train

__all__ = [
    "load_config", "CONFIG",
    "FileMeta", "parse_filename_utc", "to_local", "sniff_decimal",
    "sniff_column_order", "load_file", "scan_file", "list_files",
    "CH_COLS", "TIME_COL", "PAD_OF_CHANNEL", "CH_OF_PAD",
    "rolling_median_baseline", "quiet_noise_floor",
    "estimate_speed", "SpeedResult", "head_signal", "HEAD_BLOCKS",
    "axle_peaks", "samples_per_peak", "min_axle_separation", "has_train",
]

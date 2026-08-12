"""Lightrack realtime — the live axle counter.

    from realtime import StreamingCounter

    c = StreamingCounter()
    for block in feed:                 # blocks of raw wavelength, in nm
        for axle in c.push(block):
            ...                        # an axle was just confirmed
    c.count, c.n_x, c.n_d, c.agree, c.speed_kmh
"""

from .config import CONFIG, load_config
from .counter import (
    Axle, StreamingCounter, count_stream,
    peak_duration_s, samples_per_peak, lowpass_cutoff_hz,
)

__all__ = ["CONFIG", "load_config", "Axle", "StreamingCounter", "count_stream",
           "peak_duration_s", "samples_per_peak", "lowpass_cutoff_hz"]

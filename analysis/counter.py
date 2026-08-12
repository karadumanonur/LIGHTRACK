from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

from .config import CONFIG

FS = CONFIG["acquisition"]["fs_hz"]
SAF = CONFIG["safety"]
RT = CONFIG["realtime"]
EXTENT_M = CONFIG["geometry"]["peak_track_extent_m"]


def peak_duration_s(v_kmh: float) -> float:
 
    return EXTENT_M / (v_kmh / 3.6)


def samples_per_peak(v_kmh: float) -> float:
    return FS * peak_duration_s(v_kmh)


def lowpass_cutoff_hz(v_kmh: float) -> float:
    return SAF["lpf_cycles_per_peak"] / peak_duration_s(v_kmh)


@dataclass
class Axle:
    """One confirmed axle: when it was seen, and how tall it was."""
    index: int          # sample index of the peak, in the input stream
    height_pm: float


@dataclass
class Train:
    """One train, closed when the line has been quiet long enough."""
    axles: int
    n_x: int
    n_d: int
    agree: bool
    speed_kmh: float
    first_index: int
    last_index: int


@dataclass
class StreamingCounter:

    v0_kmh: float = RT["initial_speed_kmh"]
    block: int = RT["block_samples"]
    n_peaks: int = RT["peak_buffer"]
    taps: int = RT["fir_taps"]

    # --- state -------------------------------------------------------------
    _base: deque = field(default_factory=lambda: deque(maxlen=int(
        CONFIG["baseline"]["rolling_window_samples"])), repr=False)
    _peaks: deque = field(default_factory=lambda: deque(maxlen=16), repr=False)
    _noise: deque = field(default_factory=lambda: deque(maxlen=40), repr=False)
    _gaps: deque = field(default_factory=lambda: deque(maxlen=32), repr=False)
    _peak_max: float = 0.0
    _last_peak_i: int | None = None
    _armed: bool = False
    _i_rise: int = 0
    _run_max: float = 0.0
    _run_max_i: int = 0
    _n: int = 0                      # samples consumed
    _n_x: int = 0                    # confirmed by the hysteresis chain
    _n_d: int = 0                    # confirmed by the derivative chain
    _d_seg: list = field(default_factory=list, repr=False)
    _d_idx: list = field(default_factory=list, repr=False)
    _first_axle: int | None = None
    _last_axle: int | None = None
    _done: list = field(default_factory=list, repr=False)
    _d_start: int = 0
    _in_train_until: int = -1
    _levels_seen: deque = field(default_factory=lambda: deque(maxlen=20), repr=False)
    _provisional: bool = True
    _zi: np.ndarray | None = None
    _b: np.ndarray | None = None
    _v: float = 0.0

    def __post_init__(self) -> None:

        self._peaks = deque(maxlen=int(self.n_peaks))
        self._v = float(self.v0_kmh)
        self._design(self._v)

    # ------------------------------------------------------------- internals
    def _design(self, v_kmh: float) -> None:
        """(Re)build the FIR for the speed we currently believe."""
        cut = min(lowpass_cutoff_hz(v_kmh), FS / 2 * 0.9)
        b = firwin(self.taps, cut, fs=FS)
        if self._zi is None:
            self._zi = lfilter_zi(b, 1.0) * 0.0
        self._b = b

    @property
    def speed_kmh(self) -> float:
        """Current speed. Provisional until a time-of-flight value arrives."""
        return self._v

    @property
    def speed_is_provisional(self) -> bool:
        return self._provisional

    @property
    def level_pm(self) -> float:

        lv = self._level()
        return float(lv) if np.isfinite(lv) else float("nan")

    @property
    def n_x(self) -> int:
        """Axles confirmed by the hysteresis chain."""
        return self._n_x

    @property
    def n_d(self) -> int:
        """Axles confirmed by the derivative chain."""
        return self._n_d

    @property
    def first_axle_index(self) -> int | None:
        """Sample index of the first axle this head confirmed, if any."""
        return self._first_axle

    @property
    def d_indices(self) -> list[int]:
        """Where the derivative chain confirmed an axle."""
        return list(self._d_idx)

    @property
    def agree(self) -> bool:
        return self._n_x == self._n_d

    @property
    def count(self) -> int:
 
        return self._n_x if self._n_x == self._n_d else max(self._n_x, self._n_d)

    def _new_train(self) -> None:

        if len(self._peaks) >= 3:
            self._levels_seen.append(
                float(np.percentile(self._peaks, SAF["x_peak_percentile"])))
        if self._first_axle is not None:
            self._done.append(Train(
                axles=self.count, n_x=self._n_x, n_d=self._n_d, agree=self.agree,
                speed_kmh=self._v, first_index=self._first_axle,
                last_index=self._last_axle))
        self._gaps.clear()
        self._peaks.clear()
        self._d_idx.clear()
        self._peak_max = 0.0
        self._n_x = self._n_d = 0
        self._first_axle = self._last_axle = None

    def pop_trains(self) -> list[Train]:
        """Trains finished since the last call."""
        out, self._done = self._done, []
        return out

    def flush(self) -> list[Train]:
        """Close the train still in progress, if any, and return everything."""
        self._new_train()
        return self.pop_trains()

    def _update_speed(self) -> None:

        if len(self._gaps) < 2:
            return
        g = np.asarray(self._gaps, dtype=float)
        short = g[g <= np.median(g)]
        if not len(short):
            return
        v = 3.6 * RT["wheelbase_m"] / (float(np.median(short)) / FS)
        v = float(np.clip(v, RT["min_speed_kmh"], RT["max_speed_kmh"]))
        redesign = abs(v - self._v) / self._v > 0.05
        self._v = v                      # always keep the current estimate
        if redesign:                     # but only rebuild the filter on a real move
            self._design(v)

    def set_speed(self, v_kmh: float) -> None:

        v = float(np.clip(v_kmh, RT["min_speed_kmh"], RT["max_speed_kmh"]))
        if abs(v - self._v) / self._v > 0.05:
            self._design(v)
        self._v = v
        self._provisional = False

    def _level(self) -> float:

        if len(self._peaks) >= 3:
            return float(np.percentile(self._peaks, SAF["x_peak_percentile"]))

        floor = (float(np.median(self._levels_seen)) if self._levels_seen
                 else float(CONFIG["detect"]["absolute_gate_pm"]))
        if self._noise:
            return max(floor, float(RT["bootstrap_snr"] * np.median(self._noise)))
        return floor

    def _gate_pm(self) -> float:

        if self._levels_seen:
            return SAF["peak_rise_fraction"] * float(np.median(self._levels_seen))
        return float(CONFIG["detect"]["absolute_gate_pm"])

    def _close_d_segment(self, need: int) -> None:

        seg = np.asarray(self._d_seg, dtype=float)
        self._d_seg = []
        if seg.size < need:
            return
        kmax = int(np.argmax(seg))
        d = np.diff(seg)
        if (np.count_nonzero(d[:kmax] > 0) >= 2
                and np.count_nonzero(d[kmax:] < 0) >= 2):
            self._n_d += 1
            self._d_idx.append(self._d_start + kmax)

    # ------------------------------------------------------------------ API
    def push(self, raw_nm: np.ndarray) -> list[Axle]:
        """Consume one block of raw wavelengths, return the axles it confirmed."""
        out: list[Axle] = []
        for s in range(0, len(raw_nm), self.block):
            chunk = np.asarray(raw_nm[s:s + self.block], dtype=float)
            self._base.extend(chunk)
            base = float(np.median(self._base))           # trailing, causal
            pm = (chunk - base) * 1000.0

            y, self._zi = lfilter(self._b, 1.0, pm, zi=self._zi)

            if not self._armed and np.max(np.abs(y)) < RT["quiet_block_pm"]:
                self._noise.append(float(np.median(np.abs(y)) * 1.4826) or 1e-6)

            lv = self._level()
            thr_rise = SAF["peak_rise_fraction"] * lv
            thr_fall = SAF["peak_fall_fraction"] * lv
            t_min = SAF["x_min_duration_fraction"] * samples_per_peak(self._v)

            d_cut = SAF["d_lower_cut_fraction"] * lv
            if self._noise:
                d_cut = max(d_cut, RT["d_noise_sigmas"] * float(np.median(self._noise)))
            need = max(SAF["d_min_points"],
                       int(round(SAF["d_points_per_peak_fraction"]
                                 * samples_per_peak(self._v))))

            for k, v in enumerate(y):
                i = self._n + k

                if v > d_cut:
                    if not self._d_seg:
                        self._d_start = i
                    self._d_seg.append(v)
                elif self._d_seg:
                    self._close_d_segment(need)

                if not self._armed:
                    if v > thr_rise:
                        self._armed, self._i_rise = True, i
                        self._run_max, self._run_max_i = v, i
                else:
                    if v > self._run_max:
                        self._run_max, self._run_max_i = v, i
                    if v < thr_fall:
                        if (i - self._i_rise) >= t_min:

                            gap = (self._run_max_i - self._last_peak_i
                                   if self._last_peak_i is not None else None)
                            if gap is not None and not (
                                    0 < gap < RT["max_axle_gap_s"] * FS):
                                self._new_train()
                                gap = None

                            out.append(Axle(self._run_max_i, self._run_max))
                            self._n_x += 1
                            if self._first_axle is None:
                                self._first_axle = self._run_max_i
                            self._last_axle = self._run_max_i

                            if self._run_max >= (SAF["x_peak_floor_fraction"]
                                                 * self._peak_max):
                                self._peaks.append(self._run_max)
                            self._peak_max = max(self._peak_max, self._run_max)
                            if gap is not None:
                                self._gaps.append(gap)
                                self._update_speed()
                            self._last_peak_i = self._run_max_i
                        self._armed = False
            self._n += len(chunk)
        return out


def count_stream(raw_nm: np.ndarray, **kw) -> tuple[int, list[Axle], float]:

    c = StreamingCounter(**kw)
    axles = c.push(np.asarray(raw_nm, dtype=float))
    return c.count, axles, c.speed_kmh

@dataclass
class Passage:
    """What two heads together say about one train."""
    head_first: int                 # which head the train reached first
    lag_s: float                    # how much later the other one saw it
    speed_kmh: float                # measured, not assumed
    axles: int


def compare_heads(a: StreamingCounter, b: StreamingCounter,
                  separation_m: float, head_a: int = 1, head_b: int = 2
                  ) -> Passage | None:
                   
    if not (a.first_axle_index and b.first_axle_index):
        return None
    lag = (b.first_axle_index - a.first_axle_index) / FS
    if abs(lag) < 0.1 * separation_m / (RT["max_speed_kmh"] / 3.6):
        return None
    v = 3.6 * separation_m / abs(lag)
    if not RT["min_speed_kmh"] <= v <= RT["max_speed_kmh"]:
        return None
    return Passage(head_first=head_a if lag > 0 else head_b,
                   lag_s=abs(lag), speed_kmh=v,
                   axles=max(a.count, b.count))

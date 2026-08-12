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
    """How long one axle peak lasts, at this speed.

    An axle loads about EXTENT_M of track, so the time it spends over a sensor
    is that length divided by the speed. Every speed-adaptive number in the
    counter comes from this one line: the filter cutoff, the minimum time a
    peak must hold above the arm threshold, and the shortest segment the
    derivative chain will judge.
    """
    return EXTENT_M / (v_kmh / 3.6)


def samples_per_peak(v_kmh: float) -> float:
    """Width of one axle peak, in samples."""
    return FS * peak_duration_s(v_kmh)


def lowpass_cutoff_hz(v_kmh: float) -> float:
    """Speed-adaptive cutoff. At 60 km/h this reproduces the reference 10 Hz."""
    return SAF["lpf_cycles_per_peak"] / peak_duration_s(v_kmh)


@dataclass
class Axle:
    """One confirmed axle: when it was seen, and how tall it was."""
    index: int          # sample index of the peak, in the input stream
    height_pm: float


@dataclass
class StreamingCounter:
    """Feed it blocks of raw wavelength; it returns axles as they are confirmed.

    Every value it holds comes from samples it has already seen. Nothing is
    revisited and nothing is buffered for a second pass.
    """

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
    _d_start: int = 0
    _in_train_until: int = -1
    _levels_seen: deque = field(default_factory=lambda: deque(maxlen=20), repr=False)
    _provisional: bool = True
    _zi: np.ndarray | None = None
    _b: np.ndarray | None = None
    _v: float = 0.0

    def __post_init__(self) -> None:
        # the configured buffer length was being ignored: the deque default
        # hard-coded its own maxlen
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
        """The peak level the thresholds currently sit on, in pm.

        Exposed because it MOVES: unlike the offline chain, which fixes one
        level per passage after seeing all of it, this one is re-derived from
        the last few peaks as the train goes by. Plotting it is the clearest
        way to show that the threshold follows the signal rather than the
        other way round.
        """
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
    def d_indices(self) -> list[int]:
        """Where the derivative chain confirmed an axle."""
        return list(self._d_idx)

    @property
    def agree(self) -> bool:
        return self._n_x == self._n_d

    @property
    def count(self) -> int:
        """The two chains combined, exactly as the offline chain combines them.

        When they agree the count is that number. When they do not, the
        disagreement is not resolved quietly — the larger is returned and
        `agree` says the head could not decide. Freight is where this earns its
        keep: an axle load that varies several fold defeats any threshold set as
        a FRACTION of the peak level, and the hysteresis chain drops the light
        axles. The derivative chain does not, because it tests the shape of the
        peak rather than its height.
        """
        return self._n_x if self._n_x == self._n_d else max(self._n_x, self._n_d)

    def _new_train(self) -> None:
        """Forget what belonged to the previous train, keep what is the sensor's.

        The peak level and the wheelbase belong to the train that just left. The
        noise floor and the LEVELS SEEN belong to the installation and survive —
        that is what lets a weak sensor calibrate its own floor instead of
        inheriting one measured on a loud one.
        """
        if len(self._peaks) >= 3:
            self._levels_seen.append(
                float(np.percentile(self._peaks, SAF["x_peak_percentile"])))
        self._gaps.clear()
        self._peaks.clear()
        self._peak_max = 0.0

    def _update_speed(self) -> None:
        """Wheelbase speed from the SHORT family of peak intervals.

        A consist has two kinds of gap: within a bogie (the wheelbase) and
        between bogies. Taking the single smallest interval would let one
        split peak set the speed, so the estimate is the median of the
        intervals below the median — the short family as a group.
        """
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
        """Adopt a better speed from outside — the head-to-head time of flight.

        The wheelbase estimate exists to get a filter running in the first 0.2 s
        of a train. It inherits the assumed wheelbase, so it is only as accurate
        as that number. Once the train reaches the second head, about a second
        later, the time-of-flight speed is available and is measured rather than
        assumed: this is where it enters.
        """
        v = float(np.clip(v_kmh, RT["min_speed_kmh"], RT["max_speed_kmh"]))
        if abs(v - self._v) / self._v > 0.05:
            self._design(v)
        self._v = v
        self._provisional = False

    def _level(self) -> float:
        """The peak level the thresholds are a fraction of.

        BEFORE ANY PEAK EXISTS there is nothing to take a percentile of, and the
        fallback matters more than it looks: set it too low and the first small
        ripple of an approaching train is counted as an axle. Every over-count
        by one in this module came from exactly that.

        It is NOT a new tuned number. `absolute_gate_pm` is the floor the
        pipeline already uses to decide whether a record contains a train at
        all, read off an empty interval in the observed excursion distribution
        rather than chosen. An excursion too small to clear it would not have
        been called a train either. The noise-relative form is kept alongside so
        that a noisy installation raises the floor, never lowers it.
        """
        if len(self._peaks) >= 3:
            return float(np.percentile(self._peaks, SAF["x_peak_percentile"]))
        # Cold start uses the pipeline's own train/no-train gate. Once trains
        # have been seen, the sensor's OWN history replaces it: a pad reading at
        # half the amplitude of its neighbours needs half the floor, and nobody
        # has to configure that per sensor.
        floor = (float(np.median(self._levels_seen)) if self._levels_seen
                 else float(CONFIG["detect"]["absolute_gate_pm"]))
        if self._noise:
            return max(floor, float(RT["bootstrap_snr"] * np.median(self._noise)))
        return floor

    def _gate_pm(self) -> float:
        """Excursion that says a train is present, not an empty stretch.

        Cold start uses , the value the pipeline already reads
        off an empty interval in the observed distribution. Once trains have
        been seen the sensor scales it to itself, so a pad reading at half the
        amplitude of its neighbours is not held to a floor measured on a loud
        one.
        """
        if self._levels_seen:
            return SAF["peak_rise_fraction"] * float(np.median(self._levels_seen))
        return float(CONFIG["detect"]["absolute_gate_pm"])

    def _close_d_segment(self, need: int) -> None:
        """Decide a finished above-cut segment, exactly as the offline test does.

        The segment is buffered rather than judged sample by sample because the
        test is anchored on the segment's maximum, which is only known once the
        signal has come back down. That is not a look-ahead: the hysteresis
        chain waits for the same event before it confirms anything.
        """
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

            # Noise floor from quiet blocks only, so a train cannot inflate it.
            if not self._armed and np.max(np.abs(y)) < RT["quiet_block_pm"]:
                self._noise.append(float(np.median(np.abs(y)) * 1.4826) or 1e-6)

            lv = self._level()
            thr_rise = SAF["peak_rise_fraction"] * lv
            thr_fall = SAF["peak_fall_fraction"] * lv
            t_min = SAF["x_min_duration_fraction"] * samples_per_peak(self._v)

            # The derivative chain enters on a LOWER cut than the hysteresis
            # chain arms on, which is exactly why it keeps the light freight
            # axles the other one drops. On a weak sensor that same low cut sits
            # close to the noise, so it is floored: a fraction of the peak level
            # OR a multiple of the measured noise, whichever is higher.
            d_cut = SAF["d_lower_cut_fraction"] * lv
            if self._noise:
                d_cut = max(d_cut, RT["d_noise_sigmas"] * float(np.median(self._noise)))
            need = max(SAF["d_min_points"],
                       int(round(SAF["d_points_per_peak_fraction"]
                                 * samples_per_peak(self._v))))

            for k, v in enumerate(y):
                i = self._n + k

                # A train-present gate was tried here, mirroring the offline
                # `has_train`. It was removed after measurement: the absolute
                # excursion it keys on was read off head-1 amplitudes, so on a
                # sensor reading at half that it opens late or not at all, and
                # PAD 10 fell from 26/44 to 16/44. The single-sensor case is the
                # wrong place to solve this — the head vote below does it.

                # ---- derivative chain, run in parallel with the hysteresis one
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
                            # Time the axle at its MAXIMUM, not at the moment it
                            # crossed the threshold: a taller peak crosses
                            # earlier, so arming times are biased by height and
                            # the intervals between them would be too short —
                            # which would inflate the wheelbase speed.
                            out.append(Axle(self._run_max_i, self._run_max))
                            self._n_x += 1
                            # The 30 % floor decides which peaks DEFINE THE
                            # LEVEL, exactly as it does offline. It must not
                            # decide which peaks are counted: a long freight
                            # consist carries axle loads that differ several
                            # fold, and rejecting the light ones halves the
                            # train. Counted here, admitted to the statistic
                            # only if it is a full-sized peak.
                            if self._run_max >= (SAF["x_peak_floor_fraction"]
                                                 * self._peak_max):
                                self._peaks.append(self._run_max)
                            self._peak_max = max(self._peak_max, self._run_max)
                            if self._last_peak_i is not None:
                                g = self._run_max_i - self._last_peak_i
                                # gaps longer than a coach are between trains,
                                # not between axles
                                if 0 < g < RT["max_axle_gap_s"] * FS:
                                    self._gaps.append(g)
                                    self._update_speed()
                                else:
                                    self._new_train()
                            self._last_peak_i = self._run_max_i
                        self._armed = False
            self._n += len(chunk)
        return out


def count_stream(raw_nm: np.ndarray, **kw) -> tuple[int, list[Axle], float]:
    """Convenience: run a whole channel through the streaming counter.

    Returns (count, axles, final speed estimate). The signal is fed in blocks,
    so the result is exactly what a live feed would have produced.
    """
    c = StreamingCounter(**kw)
    axles = c.push(np.asarray(raw_nm, dtype=float))
    return c.count, axles, c.speed_kmh

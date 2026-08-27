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
    _x_idx: list = field(default_factory=list, repr=False)
    _first_axle: int | None = None
    _last_axle: int | None = None
    _done: list = field(default_factory=list, repr=False)
    _d_start: int = 0
    _in_train_until: int = -1
    _expect_from: int = -1
    _expect_to: int = -1
    _expect_h: float = 0.0
    _since_max: float = 0.0
    _unlocked_at: int = -1
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

    def _distinct(self) -> int:
        """How many separate places the two chains marked between them.

        The two totals can agree without agreeing on where. On 20 May 14:34
        both chains counted 127 and the figure carries 128 marks: one axle the
        hysteresis chain found and the derivative chain did not, another the
        derivative chain found and the hysteresis chain did not. Comparing
        totals cannot tell that apart from the two chains marking the same 127,
        and an axle that was detected is lost to the arithmetic.

        The indices come from the same peak maximum in both chains, so anything
        this close is the same axle and anything further apart is not: two real
        axles are a bogie wheelbase apart, which is an order of magnitude more.

        Widening it to a fraction of the peak, so that a flat top on a slow
        train may let the two maxima wander further apart, changes nothing at
        any fraction up to a whole peak. Twenty milliseconds is already past
        where the two agree.
        """
        idx = sorted(self._x_idx + self._d_idx)
        if not idx:
            return 0
        tol = SAF["same_axle_tolerance_s"] * FS
        n, last = 1, idx[0]
        for i in idx[1:]:
            if i - last > tol:
                n += 1
            last = i
        return n

    @property
    def count(self) -> int:
        """What the head reports: everything either chain found.

        Never below either total. The axle that crests as a record ends is
        added to both without an index, so the count of places can read low and
        must not be allowed to lower anything.

        Held back on short vehicles. One place marked by only one chain among a
        hundred both agree on is a chain missing a single axle; the same thing
        among four is the two chains disagreeing about what they are looking
        at, and believing both then invents an axle. Measured on a slow freight
        the counter had split into pieces of four: the union added one to each
        of five of them.
        """
        n = max(self._n_x, self._n_d)
        if n < SAF["same_axle_min_axles"]:
            return n
        return max(n, self._distinct())

    def _new_train(self) -> None:
        """Close the current train and keep it if it was really a train.

        One or two peaks with no companion is not a train: no bogie, no wagon.
        A live counter never sees a train cut in half, so it can say that
        outright. Reading a delivery as separate files does cut trains in half,
        and half a train is short through no fault of its own — so the rule is
        held back for one whose first axle sits at the very start of the data
        or whose last axle is still running when the data ends.
        """

        if len(self._peaks) >= 3:
            self._levels_seen.append(
                float(np.percentile(self._peaks, SAF["x_peak_percentile"])))
        grace = RT["edge_grace_s"] * FS
        clipped = self._first_axle is not None and (
            self._first_axle < grace or (self._n - self._last_axle) < grace)
        if self._first_axle is not None and (clipped
                                             or self.count >= RT["min_axles"]):
            self._done.append(Train(
                axles=self.count, n_x=self._n_x, n_d=self._n_d, agree=self.agree,
                speed_kmh=self._v, first_index=self._first_axle,
                last_index=self._last_axle))
        self._gaps.clear()
        self._peaks.clear()
        self._provisional = True
        self._expect_from = self._expect_to = -1
        self._since_max = 0.0
        self._unlocked_at = -1
        self._d_idx.clear()
        self._x_idx.clear()
        self._n_x = self._n_d = 0
        self._first_axle = self._last_axle = None

    def pop_trains(self) -> list[Train]:
        """Trains finished since the last call."""
        out, self._done = self._done, []
        return out

    def _close_pending(self) -> None:
        """The axle that armed but never fell back before the data ran out.

        A train crossing the boundary between two recordings leaves one axle
        armed at the end of the first: it crested inside this record, but the
        samples that would have carried it back under the confirm threshold
        belong to the next one, so the hysteresis never closes and the axle is
        dropped without a sound. The next record does not save it either — it
        opens on the falling side, which never arms.

        It is only counted here if it has already passed its maximum. That is
        what says the next record sees a tail rather than a rise, and so will
        not count the same axle a second time.
        """
        if not self._armed or self._first_axle is None:
            return
        t_min = SAF["x_min_duration_fraction"] * samples_per_peak(self._v)
        if (self._n - self._i_rise) < t_min:
            return
        if self._n - self._run_max_i < 2:
            return
        self._n_x += 1
        self._n_d += 1
        self._last_axle = self._run_max_i
        self._observe(self._run_max)
        self._armed = False

    def flush(self) -> list[Train]:
        """Close the train still in progress, if any, and return everything."""
        self._close_pending()
        self._new_train()
        return self.pop_trains()

    def _update_speed(self) -> None:
        """Infer the speed from the gaps, unless it has been measured.

        This reads the shortest gaps as bogie gaps and divides the wheelbase by
        them, so it is only ever as right as `wheelbase_m` is for the vehicle
        passing. A passenger bogie is about 2.6 m and a freight Y25 about 1.8,
        which is a 45 % spread, and a missed axle turns a bogie gap into a
        coupling gap and inflates it further. A measured speed replaces it and
        is not overwritten afterwards.
        """
        if not self._provisional:
            return
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

    def _expect_partner(self, index: int, height: float, gap: int | None) -> None:
        """Open a lower threshold where this axle's bogie partner should be.

        The trailing axle of a bogie is about half the leading one on every
        pair measured, and the arm sits at half the level, so a leading axle
        that only just arms leaves its partner exactly on the line. This does
        not assume the pattern holds: it only lowers the threshold inside one
        short window at one predicted distance, and never raises it, so a
        vehicle that writes something else is counted exactly as before.

        Behind a trailing axle a third is not expected at the same spacing, but
        an axle is still expected: the measured pattern puts the next bogie one
        coupling away. So the window moves rather than closing. It is the same
        rule either way — one short window, at one predicted distance, where
        the threshold is lowered and nowhere else.

        Both windows are needed because a bogie is lost as a bogie. When the
        leading axle falls under the arm the trailing one has nothing to open a
        window for it, and the pair disappears together; the coupling window is
        what reaches the next leading axle without waiting to have caught one.

        The coupling window waits for the train to have shown a few gaps first.
        The distance it predicts comes from the speed, the speed comes from the
        gaps, and at the very start of a passage there are none — so on a train
        clipped by the start of a record it would be predicting from a number
        nothing has measured yet.
        """
        frac = SAF["partner_height_fraction"]
        win = SAF["partner_window_fraction"]
        d = RT["wheelbase_m"] / max(self._v, 1.0) * 3.6 * FS
        if gap is not None and abs(gap - d) < win * d:
            if (RT["coupling_m"] <= 0.0
                    or len(self._gaps) < RT["coupling_min_gaps"]):
                self._expect_from = self._expect_to = -1
                return
            d = RT["coupling_m"] / max(self._v, 1.0) * 3.6 * FS
            frac = SAF["coupling_height_fraction"]
        floor = (SAF["partner_noise_sigmas"] * float(np.median(self._noise))
                 if self._noise else 0.0)
        self._expect_h = max(frac * height, floor)
        self._expect_from = int(index + (1 - win) * d)
        self._expect_to = int(index + (1 + win) * d)

    def _unlock_level(self, i: int, noise: float) -> None:
        """Let the level fall when a train is running but nothing confirms.

        The level is a percentile of confirmed peaks, so when a train quietens
        far enough that nothing arms, nothing enters the buffer and nothing can
        lower it again. The train is still there and still visible; this
        believes what is visible rather than what was confirmed.

        The silence has to be longer than any gap this train has already shown,
        not merely longer than a few peak widths. A vehicle is quiet between its
        bogies too — seventeen metres of it on a passenger coach — and a rule
        that cannot tell that apart from a fading train lowers the level in
        every wagon gap and starts counting ripple.

        Which is why it does nothing until the train has shown enough gaps to
        say what normal is. Measured the hard way: with a fallback of a few peak
        widths it fired inside the first wagon gap of an ordinary train, before
        any gap was known, and a 5 pm ripple became a thirteenth axle on a
        twelve-axle train. Two hundred and thirty-six trains gained one.

        The seed also has to be a real peak and not the ripple it is trying to
        rescue, so it is held to a fraction of the level it is about to lower.
        """
        if self._last_axle is None or noise <= 0.0:
            return
        if len(self._gaps) < RT["unlock_min_gaps"]:
            return
        since = max(self._last_axle, self._unlocked_at)
        if (i - since) < RT["unlock_gap_multiple"] * max(self._gaps):
            return
        if self._since_max < RT["unlock_noise_sigmas"] * noise:
            return
        if self._since_max < RT["unlock_min_fraction"] * self._level():
            return
        for _ in range(RT["unlock_seeds"]):
            self._peaks.append(self._since_max)
        self._unlocked_at = i
        self._since_max = 0.0

    def _observe(self, amplitude: float) -> None:
        """Let an axle inform the level, whichever chain confirmed it.

        The level is a percentile of recent peaks, so what is kept here
        decides where the thresholds sit next. Keeping only the peaks that
        ARMED closes a loop: a locomotive sets a level its own wagons cannot
        reach, so nothing further arms, so nothing further can lower it, and
        the rest of the train is lost. The derivative chain still sees those
        wagons at its lower cut, so it is allowed to speak here even though it
        is not what counts.

        The floor is measured against the recent peaks rather than the largest
        seen so far, because a running maximum only ever ratchets upward.
        """
        recent = max(self._peaks) if self._peaks else amplitude
        if amplitude >= SAF["x_peak_floor_fraction"] * recent:
            self._peaks.append(amplitude)

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
            self._observe(float(seg[kmax]))

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

            noise = float(np.median(self._noise)) if self._noise else 0.0
            self._unlock_level(self._n, noise)
            lv = self._level()
            thr_rise = SAF["peak_rise_fraction"] * lv
            thr_fall = SAF["peak_fall_fraction"] * lv
            t_min = SAF["x_min_duration_fraction"] * samples_per_peak(self._v)

            d_cut = SAF["d_lower_cut_fraction"] * lv
            if noise > 0.0:
                d_cut = max(d_cut, RT["d_noise_sigmas"] * noise)
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

                if v > self._since_max:
                    self._since_max = float(v)

                thr = thr_rise
                if self._expect_from <= i <= self._expect_to:
                    thr = min(thr_rise, self._expect_h)

                if not self._armed:
                    if v > thr:
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
                            self._x_idx.append(self._run_max_i)
                            if self._first_axle is None:
                                self._first_axle = self._run_max_i
                            self._last_axle = self._run_max_i

                            self._observe(self._run_max)
                            self._expect_partner(self._run_max_i,
                                                 self._run_max, gap)
                            self._since_max = 0.0
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

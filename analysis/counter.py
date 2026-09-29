from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

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
    """One train, closed when the line has been quiet long enough.

    The marks are where each chain found the axles it kept, once the train's
    edges have been judged; what the counter threw away is not among them.
    """
    axles: int
    n_x: int
    n_d: int
    agree: bool
    speed_kmh: float
    first_index: int
    last_index: int
    x_marks: list = field(default_factory=list)
    d_marks: list = field(default_factory=list)


def safety_for(pad: int | None) -> dict | None:
    """The settings this grating is counted with, if it has its own."""
    if pad is None:
        return None
    return CONFIG.get("safety_by_pad", {}).get(str(pad))


@dataclass
class StreamingCounter:

    # Settings for a grating that is not the one the thresholds were built on.
    # PAD 1 writes about 65 pm per axle and PAD 10 about half of it, and the
    # thresholds are fractions of a level taken from recent peaks, so they
    # follow the sensor down: structure between bogies that is a small
    # fraction of the axle peak on a well-bonded grating becomes a large one
    # on a loose one, and starts being counted. Overriding here rather than
    # editing the shared numbers keeps PAD 1 at what it scores.
    safety: dict | None = None

    v0_kmh: float = RT["initial_speed_kmh"]
    block: int = RT["block_samples"]
    n_peaks: int = RT["peak_buffer"]
    taps: int = RT["fir_taps"]

    # --- state -------------------------------------------------------------
    _base: deque = field(default_factory=lambda: deque(maxlen=int(
        CONFIG["baseline"]["rolling_window_samples"])), repr=False)
    _peaks: deque = field(default_factory=lambda: deque(maxlen=16), repr=False)
    _noise: deque = field(default_factory=lambda: deque(
        maxlen=int(RT["noise_blocks"])), repr=False)
    _gaps: deque = field(default_factory=lambda: deque(maxlen=32), repr=False)
    _last_peak_i: int | None = None
    _armed: bool = False
    _hold: bool = False
    _i_rise: int = 0
    _run_max: float = 0.0
    _run_max_i: int = 0
    _n: int = 0                      # samples consumed
    _n_x: int = 0                    # confirmed by the hysteresis chain
    _n_d: int = 0                    # confirmed by the derivative chain
    _d_seg: list = field(default_factory=list, repr=False)
    _d_idx: list = field(default_factory=list, repr=False)
    _x_idx: list = field(default_factory=list, repr=False)
    _d_h: list = field(default_factory=list, repr=False)
    _x_h: list = field(default_factory=list, repr=False)
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
    _recent: deque = field(default_factory=deque, repr=False)
    _provisional: bool = True
    _zi: np.ndarray | None = None
    _b: np.ndarray | None = None
    _v: float = 0.0

    def __post_init__(self) -> None:
        self._saf = SAF if not self.safety else {**SAF, **self.safety}

        self._peaks = deque(maxlen=int(self.n_peaks))
        blocks = int(np.ceil(RT["presence_window_s"] * FS / self.block))
        self._recent = deque(maxlen=max(1, blocks))
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
        tol = self._saf["same_axle_tolerance_s"] * FS
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
        if n < self._saf["same_axle_min_axles"]:
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

        self._drop_edges()
        if len(self._peaks) >= 3:
            self._levels_seen.append(
                float(np.percentile(self._peaks, self._saf["x_peak_percentile"])))
        grace = RT["edge_grace_s"] * FS
        clipped = self._first_axle is not None and (
            self._first_axle < grace or (self._n - self._last_axle) < grace)
        if self._first_axle is not None and (clipped
                                             or self.count >= RT["min_axles"]):
            self._done.append(Train(
                axles=self.count, n_x=self._n_x, n_d=self._n_d, agree=self.agree,
                speed_kmh=self._v, first_index=self._first_axle,
                last_index=self._last_axle, x_marks=list(self._x_idx),
                d_marks=list(self._d_idx)))
        self._gaps.clear()
        self._peaks.clear()
        self._provisional = True
        self._expect_from = self._expect_to = -1
        self._since_max = 0.0
        self._unlocked_at = -1
        self._d_idx.clear()
        self._x_idx.clear()
        self._d_h.clear()
        self._x_h.clear()
        self._n_x = self._n_d = 0
        self._first_axle = self._last_axle = None

    def _drop_edges(self) -> None:
        """Throw away what the train pushed ahead of itself and left behind.

        A rail lifts before the load reaches it, and that shows on the grating
        as a slow bump a third of a second ahead of the first axle. It is one
        to five picometres against axles of a hundred and more — one to four
        per cent of them — but it is a hundred times the noise, so a level
        taken from what is visible takes it for the train and counts it.

        It is thrown away from the front of the train: anything leading that
        is under `lead_fraction` of the tallest axle behind it goes, and the
        first axle that is not stops the rule. It is measured against the axles
        behind it and not against the train's median, because those are the
        load that moved the rail: against the median a locomotive's bump sits
        among the wagons, and 20 May 18:38 pushed 5.5 pm ahead of a 149 pm
        axle, 4 % of it but 12 % of the train. A slow heavy vehicle moves the
        rail for seconds before it arrives, so nothing nearer than the whole
        train will do: on 22 Jul 11:35 three axles of 260 pm came four seconds
        after the first wiggle. No real first axle of the 1596 reference
        trains is under 16 % of the tallest behind it.

        The rail settles the same way once the load has gone, a swell a
        quarter of a second after the last axle: on 19 May 13:36 one of 3.1 pm
        was counted as a 137th axle of 136. It is judged against the last
        bogie, the axles in the `trail_s` before it, and not against the
        train, because the last axle of a train is a bogie's trailing one and
        writes a third of its partner: on 21 Jul 19:58 a real one of 16 pm
        was under a tenth of the locomotive at the other end. Against its
        bogie no real last axle of 1503 trains counted right is between a
        tenth and a quarter, and the swells reach an eighth, so `trail_fraction`
        sits at a fifth.

        An end the record cut off is not judged: what leads a record that
        opens on a train is the middle of it, and what trails one that closes
        on a train is not its end.
        """
        marks = sorted([(i, h, self._x_idx, self._x_h) for i, h in
                        zip(self._x_idx, self._x_h)]
                       + [(i, h, self._d_idx, self._d_h) for i, h in
                          zip(self._d_idx, self._d_h)], key=lambda m: m[0])
        if len(marks) < RT["edge_min_marks"]:
            return
        at = [m[0] for m in marks]
        hs = [m[1] for m in marks]
        grace, tail = RT["edge_grace_s"] * FS, RT["trail_s"] * FS
        front = []
        if at[0] >= grace:
            for k in range(len(marks)):
                if hs[k] >= RT["lead_fraction"] * max(hs[k + 1:], default=0.0):
                    break
                front.append(k)
        back = []
        if self._n - at[-1] >= grace:
            for k in range(len(marks) - 1, -1, -1):
                before = [h for a, h in zip(at[:k], hs[:k]) if a >= at[k] - tail]
                if k in front or hs[k] >= RT["trail_fraction"] * max(before, default=0.0):
                    break
                back.append(k)
        for k in front + back:
            i, _, idx, heights = marks[k]
            j = idx.index(i)
            del idx[j]
            del heights[j]
            if idx is self._x_idx:
                self._n_x = max(0, self._n_x - 1)
            else:
                self._n_d = max(0, self._n_d - 1)
        if self._x_idx or self._d_idx:
            self._first_axle = min(self._x_idx + self._d_idx)
            if back:
                self._last_axle = max(self._x_idx + self._d_idx)

    def carry(self, prev: "StreamingCounter") -> None:
        """Start from what the record before this one had learned.

        A live counter never opens in the middle of a train; one reading a
        record that continues another does, and then it has no quiet stretch
        to take the noise from and nothing but the absolute gate for a level.
        September on PAD 1 lost whole trains that way: 17 Sep 09:09 carried 84
        axles of 8-12 pm into a new file and the counter stood at 39.

        So the baseline, the noise, the levels, the last second and the speed
        go on from where `prev` stopped, as they would have in a single stream.
        What it counted does not: every axle here is this record's own. An axle
        `prev` counted as it ended is still falling here, and with the baseline
        carried over it is tall enough to arm again, so nothing arms until it
        has fallen back under the confirm threshold.
        """
        self._hold = prev._hold
        self._base.extend(prev._base)
        self._noise.extend(prev._noise)
        self._levels_seen.extend(prev._levels_seen)
        self._recent.extend(prev._recent)
        self._v, self._b, self._zi = prev._v, prev._b.copy(), prev._zi.copy()

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
        t_min = self._saf["x_min_duration_fraction"] * samples_per_peak(self._v)
        if (self._n - self._i_rise) < t_min:
            return
        if self._n - self._run_max_i < 2:
            return
        self._n_x += 1
        self._n_d += 1
        self._last_axle = self._run_max_i
        self._observe(self._run_max)
        self._armed = False
        self._hold = True

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

        A gap no bogie could take even at the slowest train is not a bogie gap,
        and is not read as one. When a weak grating loses the trailing axles,
        what is left is the gaps between vehicles, and dividing the wheelbase
        by those put a 100 km/h train at 16: the duration rule then asked for
        peaks five times too wide and nothing further counted (16 Sep 12:58).
        """
        if not self._provisional:
            return
        if len(self._gaps) < 2:
            return
        g = np.asarray(self._gaps, dtype=float)
        longest = RT["wheelbase_m"] / (RT["min_speed_kmh"] / 3.6) * FS
        short = g[(g <= np.median(g)) & (g <= longest)]
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
        """What the thresholds are fractions of.

        Once a train has given three peaks, a percentile of its recent peaks.
        Before that, a memory: the levels of the trains before it, or the
        absolute gate when there were none. A memory is only right while the
        grating writes what it used to. When it suddenly writes less, every
        axle stays under half the remembered level, nothing arms, nothing is
        confirmed, and the memory that is wrong is the only thing that could
        have been corrected — September on PAD 1, where a 4.5 pm freight
        train went past a level of 39.

        So while a passage is visibly under way the memory is capped by what
        the last second shows. Visible means clear of the grating's own noise
        by `presence_noise_sigmas`, which empty stretches do not reach. The
        cap only ever lowers the level: a grating writing what it always did
        shows more than the memory, and is counted exactly as before.
        """
        if len(self._peaks) >= 3:
            return float(np.percentile(self._peaks, self._saf["x_peak_percentile"]))

        floor = (float(np.median(self._levels_seen)) if self._levels_seen
                 else float(CONFIG["detect"]["absolute_gate_pm"]))
        if not self._noise:
            return floor
        noise = float(np.median(self._noise))
        seen = max(self._recent) if self._recent else 0.0
        if seen >= max(RT["presence_noise_sigmas"] * noise, RT["presence_min_pm"]):
            floor = min(floor, seen)
        return max(floor, float(RT["bootstrap_snr"] * noise))

    def _gate_pm(self) -> float:

        if self._levels_seen:
            return self._saf["peak_rise_fraction"] * float(np.median(self._levels_seen))
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
        frac = self._saf["partner_height_fraction"]
        win = self._saf["partner_window_fraction"]
        d = RT["wheelbase_m"] / max(self._v, 1.0) * 3.6 * FS
        if gap is not None and abs(gap - d) < win * d:
            if (RT["coupling_m"] <= 0.0
                    or len(self._gaps) < RT["coupling_min_gaps"]):
                self._expect_from = self._expect_to = -1
                return
            d = RT["coupling_m"] / max(self._v, 1.0) * 3.6 * FS
            frac = self._saf["coupling_height_fraction"]
        floor = (self._saf["partner_noise_sigmas"] * float(np.median(self._noise))
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

        It also looks back when a peak taller than all of them arrives: what
        came before and is under the floor of the new one stops defining the
        level. The rail moving ahead of a heavy vehicle is seen first, at a
        percent of what follows, and left in the buffer it held the level at
        1.2 pm under a 143 pm axle, so every rebound after it counted
        (24 Jul 10:27).
        """
        floor = self._saf["x_peak_floor_fraction"]
        recent = max(self._peaks) if self._peaks else amplitude
        if amplitude < floor * recent:
            return
        self._peaks.append(amplitude)
        if amplitude > recent:
            kept = [p for p in self._peaks if p >= floor * amplitude]
            self._peaks.clear()
            self._peaks.extend(kept)

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
            self._d_h.append(float(seg[kmax]))
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
            self._recent.append(float(np.max(y)))

            quiet = RT["quiet_block_pm"]
            if len(self._noise) >= 5:
                quiet = min(quiet, RT["quiet_noise_sigmas"] * float(np.median(self._noise)))
            if not self._armed and np.max(np.abs(y)) < quiet:
                self._noise.append(float(np.median(np.abs(y)) * 1.4826) or 1e-6)

            noise = float(np.median(self._noise)) if self._noise else 0.0
            self._unlock_level(self._n, noise)
            lv = self._level()
            thr_rise = self._saf["peak_rise_fraction"] * lv
            thr_fall = self._saf["peak_fall_fraction"] * lv
            t_min = self._saf["x_min_duration_fraction"] * samples_per_peak(self._v)

            d_cut = self._saf["d_lower_cut_fraction"] * lv
            if noise > 0.0:
                d_cut = max(d_cut, RT["d_noise_sigmas"] * noise)
            need = max(self._saf["d_min_points"],
                       int(round(self._saf["d_points_per_peak_fraction"]
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
                    if self._hold:
                        self._hold = v >= thr_fall
                    elif v > thr:
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
                            self._x_h.append(float(self._run_max))
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


@lru_cache(maxsize=4)
def _channels(path: str) -> np.ndarray:
    from .reader import read_channels
    return read_channels(path)


def before(path, pad: int, depth: int = 3) -> StreamingCounter | None:
    """The record `path` carries straight on from, counted to its end.

    None when the record starts on its own. A train long enough to run through
    several files is followed back `depth` records.
    """
    from .reader import previous
    prev = previous(path) if depth > 0 else None
    if prev is None:
        return None
    p = counter_for(prev, pad, depth - 1)
    col = _channels(str(prev))[:, pad - 1]
    for s in range(0, len(col), p.block):
        p.push(col[s:s + p.block])
    p.flush()
    return p


def counter_for(path, pad: int, depth: int = 3) -> StreamingCounter:
    """A counter for one record of `pad`, carrying on from the record before.

    See `StreamingCounter.carry` for why.
    """
    c = StreamingCounter(safety=safety_for(pad))
    p = before(path, pad, depth)
    if p is not None:
        c.carry(p)
    return c


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

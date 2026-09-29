"""The counting engines, behind one interface.

Each engine is imported rather than copied, so what the window shows is the
code itself and never a transcription of it.

Which copy matters. Onur's counter is read from the WORKING TREE of this
repository, so the window shows the algorithm as it stands right now, including
edits that have not been committed — the person changing it needs to see what
they changed. Asude's is read from the branch worktree `lightrack compare` lays
down under `.sides/`, because her code does not live in this repository at all.
Pass `worktree=True` to read Onur's from the branch instead, which is what to
do when the question is what the committed branch does rather than what the
working tree does.

An engine answers two questions and they are deliberately separate:

  how many axles are in this record — the number the store keeps
  what did you look at to decide — the trace the detection panel draws

The second is what makes a count checkable instead of asserted, and the two
engines answer it differently on purpose. One has thresholds that move with a
level and two chains that can disagree; the other band-passes, learns one
height threshold per recording from its quiet background, and has one kind of
mark. Flattening
that into a common shape would hide the only interesting thing about comparing
them, so the trace carries whatever guides and marks the engine actually has.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import home

ROOT = home.bundle()
SIDES = ROOT / "lightrack" / ".sides"
INK, PURPLE, GREEN, GREY, TERRA = "#1d1b17", "#7b4ea8", "#2f7d4f", "#9a958a", "#a4553a"

ENGINES = {
    "onur": dict(
        name="Onur",
        worktree=ROOT / "lightrack",
        fallback=SIDES / "onur",
        about="streaming counter, hysteresis and derivative chains, "
              "thresholds that move with the level"),
    "asude": dict(
        name="Asude",
        worktree=SIDES / "asude",
        fallback=None,
        about="detrend, 0.5-40 Hz Butterworth, a 20 pm event gate, then peaks "
              "above height and prominence thresholds learned from each "
              "recording's quiet background"),
}


def home(engine: str, worktree: bool = False) -> Path | None:
    e = ENGINES.get(engine)
    if not e:
        return None
    order = [e["fallback"], e["worktree"]] if worktree else [e["worktree"],
                                                             e["fallback"]]
    for p in order:
        if p and p.exists() and any(p.iterdir()):
            return p
    return None


def ready(engine: str) -> bool:
    h = home(engine)
    if h is None:
        return False
    if engine == "onur":
        return (h / "analysis" / "counter.py").exists()
    return (h / "axle_detector.py").exists() and (h / "recording_io.py").exists()


def available() -> list[dict]:
    return [dict(id=k, name=v["name"], about=v["about"], ready=ready(k))
            for k, v in ENGINES.items()]


def _path(engine: str) -> None:
    h = home(engine)
    if h and str(h) not in sys.path:
        sys.path.insert(0, str(h))


def _onur_trace(path: Path, pad: int) -> dict:
    _path("onur")
    from analysis.config import CONFIG                          # noqa: E402
    from analysis.counter import (counter_for, safety_for,      # noqa: E402
                                  samples_per_peak)
    from analysis.reader import read_channels                   # noqa: E402
    from scipy.signal import firwin, lfilter, lfilter_zi
    import pandas as pd

    fs = CONFIG["acquisition"]["fs_hz"]
    rt, saf = CONFIG["realtime"], CONFIG["safety"]
    win = CONFIG["baseline"]["rolling_window_samples"]
    col = read_channels(path)[:, pad - 1]

    c = counter_for(path, pad)
    lv, tm = [], []
    block = int(rt["block_samples"])
    for s in range(0, len(col), block):
        c.push(col[s:s + block])
        lv.append(c.level_pm)
        tm.append(s + block)
    seam, maybe = [], []
    if c._armed and c._first_axle is not None:
        t_min = saf["x_min_duration_fraction"] * samples_per_peak(c.speed_kmh)
        if (c._n - c._i_rise) >= t_min and (c._n - c._run_max_i) >= 2:
            seam.append(int(c._run_max_i))
        else:
            maybe.append(int(c._run_max_i))
    trains = c.flush()
    hits = sorted({int(i) for t in trains for i in t.x_marks})
    tol = (safety_for(pad) or {}).get("same_axle_tolerance_s",
                                      saf["same_axle_tolerance_s"]) * fs
    d_only = [int(i) for t in trains for i in t.d_marks
              if not any(abs(int(i) - h) <= tol for h in hits)]

    trail = pd.Series(col).rolling(win, min_periods=1).median().to_numpy()
    from analysis.counter import lowpass_cutoff_hz               # noqa: E402
    b = firwin(int(rt["fir_taps"]),
               min(lowpass_cutoff_hz(c.speed_kmh), fs / 2 * 0.9), fs=fs)
    y = lfilter(b, 1.0, (col - trail) * 1000.0, zi=lfilter_zi(b, 1.0) * 0.0)[0]
    level = (pd.Series(lv, index=tm).reindex(np.arange(len(y)), method="ffill")
             .bfill().to_numpy())

    return dict(
        y=y, fs=fs,
        guides=[("arm", TERRA, saf["peak_rise_fraction"] * level),
                ("confirm", "#c98b3a", saf["peak_fall_fraction"] * level)],
        marks=[("hysteresis chain", INK, hits),
               ("derivative only", PURPLE, d_only),
               ("cut by the record end", GREEN, seam),
               ("possible, cut too early", GREY, maybe)],
        units="filtered [pm]",
        axles=int(sum(t.axles for t in trains)),
        trains=len(trains),
        speed_kmh=round(float(c.speed_kmh), 1),
        first_s=round(min((t.first_index / fs for t in trains), default=0.0), 2),
        last_s=round(max((t.last_index / fs for t in trains), default=0.0), 2),
        detail=" + ".join(str(t.axles) for t in trains),
        n_x=int(sum(t.n_x for t in trains)),
        n_d=int(sum(t.n_d for t in trains)))


def _asude_trace(path: Path, pad: int) -> dict:
    """Her final detector, called the way her Train Monitoring window calls it.

    Her own reader picks the PAD's column: it compares the median wavelength
    of the first and last columns, so a delivery written from the long
    wavelength down and one written from the short wavelength up both give the
    grating that was asked for. Her detector then band-passes, gates the
    recording at 20 pm and learns its thresholds from the quiet background.
    The height threshold is drawn where it sits on the signal: the learned
    background plus the threshold, in the direction of the chosen polarity.
    """
    _path("asude")
    from recording_io import (load_physical_fbg_from_frame,      # noqa: E402
                              load_recording_frame)
    from axle_detector import detect_axles                      # noqa: E402

    got = load_physical_fbg_from_frame(load_recording_frame(str(path)), pad)
    time_s = np.asarray(got["time_s"], dtype=float)
    det = detect_axles(got["raw_nm"], time_s, pad,
                       stored_column=int(got["stored_column"]),
                       stored_order=str(got["stored_order"]))
    y = np.asarray(det.filtered_signal_nm, dtype=float) * 1000.0
    idx = [int(i) for i in det.indices]
    guides = []
    side = det.polarity_result
    if side is not None and np.isfinite(side.height_threshold_nm):
        level = (det.background_location_nm
                 + side.sign * side.height_threshold_nm) * 1000.0
        guides.append(("learned height", TERRA, np.full(len(y), level)))
    return dict(
        y=y, fs=500.0,
        units="band-passed [pm]",
        guides=guides,
        marks=[(f"{det.selected_polarity} peak, {det.quality_label}", INK, idx)],
        axles=len(idx), trains=1 if idx else 0, speed_kmh=None,
        first_s=round(float(time_s[idx[0]] - time_s[0]), 2) if idx else 0.0,
        last_s=round(float(time_s[idx[-1]] - time_s[0]), 2) if idx else 0.0,
        detail=det.quality_label, n_x=len(idx), n_d=len(idx))


RUNNERS = {"onur": _onur_trace, "asude": _asude_trace}


def run(engine: str, path: Path, pad: int) -> dict:
    if engine not in RUNNERS:
        raise KeyError(engine)
    if not ready(engine):
        raise RuntimeError(
            f"{ENGINES[engine]['name']} is not available. Run "
            f"`python -m lightrack.compare` once to lay down its worktree.")
    return RUNNERS[engine](Path(path), pad)


def count(engine: str, path: Path, pad: int) -> dict:
    """The number, and where the axles are.

    The marks come back with it because every sensor is measured at the axles
    this PAD found; `head_marks` says when each other PAD saw them.
    """
    r = run(engine, path, pad)
    fs = r["fs"]
    marks = sorted(i for _, _, xs in r["marks"] for i in xs)
    out = {k: r[k] for k in ("axles", "trains", "speed_kmh", "first_s",
                             "last_s", "detail")}
    out["marks_s"] = [round(i / fs, 4) for i in marks]
    return out


def heads() -> dict[int, int]:
    """Which of the three heads along the rail each PAD belongs to."""
    from ..config import CONFIG as SITE
    by_channel = SITE["channel_to_pad"]["head_of_channel"]
    return {pad: int(by_channel[12 - pad]) for pad in range(1, 13)}


def head_marks(engine: str, path: Path, pad: int, counted: dict) -> dict:
    """The axles one PAD counted, at the moment each other PAD saw them.

    The twelve gratings are on three heads, fifteen and then ten metres apart
    along the rail, so a train reaches them at three different moments: at
    87 km/h head 2 sees an axle 0.6 s after head 1 and head 3 a second after
    it, in either order depending on the direction. Reading every PAD at the
    counting PAD's moments reads the heads it is not on between their axles,
    and a grating writing 120 pm was measured at 18.

    So each other head is shifted by its own lag, taken from where the train
    starts on that head: the first PAD of that head whose count agrees with
    the counting PAD's to within a tenth gives the time of its first axle. The
    lag has to be one a train can have — the head distance at 20 to 200 km/h —
    or the next PAD of the head is tried; if none qualifies the head is left
    unmeasured for this record rather than measured at the wrong moment.
    """
    from ..config import CONFIG as SITE
    head = heads()
    geo = SITE["geometry"]
    along = {1: 0.0, 2: float(geo["head1_to_head2_m"]),
             3: float(geo["head1_to_head2_m"]) + float(geo["head2_to_head3_m"])}
    marks = list(counted.get("marks_s") or []) if counted.get("axles") else []
    if not marks:
        return {p: [] for p in head}

    lag_of = {head[pad]: 0.0}
    enough = max(1, 0.1 * counted["axles"])
    for h in sorted(set(head.values()) - {head[pad]}):
        d = abs(along[h] - along[head[pad]])
        lo, hi = d / (200.0 / 3.6), d / (20.0 / 3.6)
        for q in (p for p, hh in head.items() if hh == h):
            try:
                c = count(engine, path, q)
            except Exception:
                continue
            if not c.get("axles"):
                continue
            lag = float(c["first_s"]) - float(counted["first_s"])
            if (lo <= abs(lag) <= hi
                    and abs(int(c["axles"]) - int(counted["axles"])) <= enough):
                lag_of[h] = lag
                break
    return {p: ([round(t + lag_of[h], 4) for t in marks] if h in lag_of else None)
            for p, h in head.items()}


def trace(engine: str, path: Path, pad: int, span=None, step: int = 6) -> dict:
    """The engine's own view of the record, thinned for one panel."""
    r = run(engine, path, pad)
    fs, y = r["fs"], r["y"]
    a, b = (0, len(y)) if span is None else (int(span[0] * fs), int(span[1] * fs))
    a, b = max(0, a), min(len(y), b)
    if b - a < fs:
        a, b = 0, len(y)
    take = slice(a, b, step)
    inside = lambda xs: [round(i / fs, 3) for i in xs if a <= i < b]
    height = lambda xs: [round(float(y[i]), 1) for i in xs if a <= i < b]
    return dict(
        span=[round(a / fs, 2), round(b / fs, 2)],
        t=[round(v, 3) for v in (np.arange(a, b, step) / fs)],
        y=[round(float(v), 2) for v in y[take]],
        guides=[dict(name=n, colour=c, v=[round(float(x), 2) for x in g[take]])
                for n, c, g in r["guides"]],
        marks=[dict(name=n, colour=c, t=inside(xs), y=height(xs))
               for n, c, xs in r["marks"]],
        axles=r["axles"], trains=r["trains"], speed_kmh=r["speed_kmh"],
        first_s=r["first_s"], last_s=r["last_s"], detail=r["detail"],
        n_x=r["n_x"], n_d=r["n_d"], units=r.get("units", "signal"))

"""Gated records, and the independent check on speed.

PART A  WHAT THE GATE EXCLUDED. The constant-speed solver assumes one slowness
        across all 25 m. A train that accelerates, brakes or stops between heads
        violates that, the fit score falls, and the record is gated out — so the
        excluded set plausibly contains the stopping and traffic-held passages,
        exactly the cases the occupancy logic has to survive. Every excluded
        record is classified individually. NOTHING is re-admitted to any
        statistic here; this only establishes what they are.

PART B  TWO INDEPENDENT SPEEDS. There are two measurements per record and they
        share no assumption:

          TOF     across heads, 25 m baseline, from the constrained lag fit
          bogie   within ONE head, from the wheelbase interval

        Where they diverge materially something is wrong with that record —
        either the lag fit, or the assumption about the stock's wheelbase. Both
        are useful classifications, and neither can be reached from the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Works from either location: the working copy (data/scripts/) or the repo,
# where Scripts/ sits inside the package directory itself.
_UP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UP if (_UP / "lightrack").is_dir() else _UP.parent))
from lightrack import (                                    # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal, axle_peaks,
)
from lightrack.speed import estimate_accelerated           # noqa: E402
from lightrack.detect import passage_window, samples_per_peak  # noqa: E402
from lightrack.occupancy import slow_baseline, find_stationary  # noqa: E402


# ============================================== PART A — THE EXCLUDED RECORDS
FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
SESSIONS = {pd.Timestamp(d).date(): i + 1
            for i, d in enumerate(CONFIG["campaign"]["measurement_dates"])}
MIN_SCORE = CONFIG["speed"]["min_fit_score"]
EDGE_S = 2.0

pd.set_option("display.width", 250)
LINE = "=" * 112


def classify(rec: dict) -> str:
    if not rec["in_session"]:
        return "outside session series"
    if rec["n_stationary"] > 0:
        return "STOPPED / held"
    if rec["touches_start"] or rec["touches_end"]:
        return "split at file boundary"
    if rec["n_bursts"] > 1:
        return "multiple trains"
    if rec["acc_score"] - rec["score"] > 0.15 and abs(rec["accel"]) >= 0.25:
        return "ACCELERATING" if rec["accel"] > 0 else "DECELERATING"
    if rec["max_pm"] < 20:
        return "no train / weak"
    return "genuine bad fit"


def classify_gated() -> None:
    rows = []
    for n, p in enumerate(list_files(), 1):
        utc = parse_filename_utc(p.name)
        loc = to_local(utc)
        raw = load_file(p)
        y = rolling_median_baseline(raw[CH_COLS])
        spd = estimate_speed(y)
        acc = estimate_accelerated(y)

        hs = head_signal(y, 1)
        w = passage_window(hs, spd.v_kmh)
        nrec = len(y)

        # bursts: contiguous activity blocks across the whole record
        env = pd.Series(np.abs(hs)).rolling(int(0.3 * FS), center=True,
                                            min_periods=1).mean().to_numpy()
        sig = env > 0.15 * env.max() if env.max() > 0 else np.zeros(nrec, bool)
        ix = np.flatnonzero(sig)
        nb = 0
        touches_start = touches_end = False
        if ix.size:
            sp = np.flatnonzero(np.diff(ix) > int(3.0 * FS)) + 1
            runs = np.split(ix, sp)
            nb = len(runs)
            touches_start = runs[0][0] < EDGE_S * FS
            touches_end = runs[-1][-1] > nrec - EDGE_S * FS

        # stationary state, on a LONG baseline
        ys = slow_baseline(raw[CH_COLS])
        st = find_stationary(head_signal(ys, 1), spd.v_kmh)

        rows.append({
            "name": p.name, "local": loc, "date": loc.date(),
            "in_session": loc.date() in SESSIONS,
            "score": spd.score, "v_kmh": spd.v_kmh, "gated": spd.ok,
            "acc_score": acc.score, "v0": acc.v0_kmh, "accel": acc.accel_ms2,
            "v_exit": acc.v_exit_kmh,
            "n_bursts": nb, "touches_start": touches_start,
            "touches_end": touches_end,
            "n_stationary": len(st),
            "stationary_s": sum(e.duration_s for e in st),
            "max_pm": float(np.abs(y[CH_COLS].to_numpy()).max()),
            "n_axles": len(axle_peaks(hs, spd.v_kmh, window=w)) if w else 0,
        })
        if n % 15 == 0:
            print(f"  [{n:3d}/93] {p.name}", flush=True)

    df = pd.DataFrame(rows)
    df["class"] = df.apply(classify, axis=1)
    df["excluded"] = ~(df["gated"] & df["in_session"])
    df.to_csv(OUT / "gated_classification.csv", index=False)

    print(f"\n{LINE}\nITEM 4 — THE 23 EXCLUDED RECORDS, CLASSIFIED\n{LINE}")
    ex = df[df["excluded"]].sort_values("local")
    print(f"total files {len(df)}, retained {int((~df['excluded']).sum())}, "
          f"excluded {len(ex)}\n")
    print(f"{'local':<20}{'score':>7}{'acc sc':>8}{'v0':>7}{'accel':>8}"
          f"{'burst':>6}{'edge':>6}{'stat s':>8}{'axles':>7}  class")
    for _, r in ex.iterrows():
        edge = ("S" if r["touches_start"] else "") + ("E" if r["touches_end"] else "")
        print(f"{r['local']:%Y-%m-%d %H:%M:%S}  {r['score']:7.2f}{r['acc_score']:8.2f}"
              f"{r['v0']:7.1f}{r['accel']:+8.2f}{r['n_bursts']:>6}{edge:>6}"
              f"{r['stationary_s']:>8.1f}{r['n_axles']:>7}  {r['class']}")

    print(f"\n{LINE}\nBREAKDOWN\n{LINE}")
    print(ex["class"].value_counts().to_string())
    print(f"\nof the {len(ex)} excluded: "
          f"{int((~ex['in_session']).sum())} are outside the 15-session series "
          f"(install day and off-hours records), leaving "
          f"{int(ex['in_session'].sum())} gated on fit score alone.")
    ing = ex[ex["in_session"]]
    if len(ing):
        print("\nscore-gated only:")
        print(ing["class"].value_counts().to_string())

    print(f"\n{LINE}\nITEM 5 — ACCELERATION FROM THE LAG RATIO\n{LINE}")
    print("The geometry forces lag_bc/lag_ab = 15/10 ONLY at constant speed.")
    print("Under uniform acceleration the ratio moves, and the shift measures it:")
    print("  accelerating -> the second 10 m is covered faster -> ratio RISES")
    print("  decelerating -> ratio FALLS\n")
    ok = df[df["gated"] & df["in_session"]]
    print(f"{'population':<34}{'n':>5}{'median accel':>14}{'p5':>9}{'p95':>9}"
          f"{'|a|>0.25 m/s2':>15}")
    for lab, sub in (("retained (score-gated, in session)", ok),
                     ("excluded, in session", ing),
                     ("all files", df)):
        if not len(sub):
            continue
        print(f"{lab:<34}{len(sub):>5}{sub['accel'].median():>+14.2f}"
              f"{sub['accel'].quantile(0.05):>+9.2f}{sub['accel'].quantile(0.95):>+9.2f}"
              f"{100*(sub['accel'].abs()>0.25).mean():>14.0f}%")

    print(f"\ntwo-branch solver — which branch wins, over all {len(df)} files:")
    df["branch"] = np.where(df["acc_score"] > df["score"] + 0.05,
                            "accelerated", "constant-speed")
    print(df["branch"].value_counts().to_string())
    gain = df["acc_score"] - df["score"]
    print(f"\nscore gain from the accelerated branch: median {gain.median():+.3f}, "
          f"p95 {gain.quantile(0.95):+.3f}, max {gain.max():+.3f}")
    rescued = df[(~df["gated"]) & df["in_session"] & (df["acc_score"] >= MIN_SCORE)]
    print(f"\nrecords currently gated out whose ACCELERATED fit clears "
          f"{MIN_SCORE}: {len(rescued)}/{len(ing)}")
    for _, r in rescued.iterrows():
        print(f"    {r['local']:%Y-%m-%d %H:%M}  score {r['score']:.2f} -> "
              f"{r['acc_score']:.2f}   v0 {r['v0']:.0f} km/h, "
              f"a {r['accel']:+.2f} m/s2, exit {r['v_exit']:.0f} km/h  [{r['class']}]")

    print(f"\n{LINE}\nITEM 6 — STATIONARY STATE\n{LINE}")
    st = df[df["n_stationary"] > 0].sort_values("stationary_s", ascending=False)
    print(f"records with a detected stationary plateau: {len(st)}/{len(df)}")
    if len(st):
        print(f"\n{'local':<20}{'stat s':>8}{'events':>8}{'score':>7}"
              f"{'v_kmh':>8}{'axles':>7}  gated?")
        for _, r in st.iterrows():
            print(f"{r['local']:%Y-%m-%d %H:%M:%S}  {r['stationary_s']:>8.1f}"
                  f"{r['n_stationary']:>8}{r['score']:>7.2f}{r['v_kmh']:>8.1f}"
                  f"{r['n_axles']:>7}  {'retained' if r['gated'] else 'GATED OUT'}")
    print("\nWhile a plateau holds, BOTH counting rules must abstain rather than")
    print("count — a halted axle over the head satisfies D-crossing's six-sample")
    print("requirement indefinitely. Abstention means the section stays OCCUPIED,")
    print("which is the safe state; counting would risk a phantom axle.")
    print(f"\nwrote {OUT/'gated_classification.csv'}")


# ================================================ PART B — SPEED CROSS-CHECK
FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
WHEELBASE_M = 2.84          # class-0 template median; see docstring
pd.set_option("display.width", 250)
LINE = "=" * 116


def bogie_speed(idx: np.ndarray) -> tuple[float, int]:
    """Speed from the short (wheelbase) intervals at one head, km/h."""
    if len(idx) < 4:
        return np.nan, 0
    dt = np.diff(idx) / FS
    short = dt[dt < np.median(dt)]
    if short.size < 3:
        return np.nan, int(short.size)
    return float(WHEELBASE_M / np.median(short) * 3.6), int(short.size)


def speed_crosscheck() -> None:
    g = pd.read_csv(OUT / "gated_classification.csv")
    g["local"] = pd.to_datetime(g["local"])
    rows = []
    for n, p in enumerate(list_files(), 1):
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        per_head = {}
        for h in (1, 2, 3):
            hs = head_signal(y, h)
            w = passage_window(hs, spd.v_kmh)
            idx = axle_peaks(hs, spd.v_kmh, window=w) if w else np.array([], int)
            vb, ns = bogie_speed(idx)
            per_head[h] = {"n": len(idx), "v": vb, "ns": ns,
                           "dur": (w[1] - w[0]) / FS if w else 0.0}
        vb = np.nanmedian([per_head[h]["v"] for h in (1, 2, 3)])
        counts = [per_head[h]["n"] for h in (1, 2, 3)]
        rows.append({
            "name": p.name, "v_tof": spd.v_kmh, "v_bogie": vb,
            "div_pct": 100 * (vb - spd.v_kmh) / spd.v_kmh if np.isfinite(vb) else np.nan,
            "n_h1": counts[0], "n_h2": counts[1], "n_h3": counts[2],
            "count_spread": max(counts) - min(counts),
            "dur_s": per_head[2]["dur"],
        })
        if n % 20 == 0:
            print(f"  [{n:3d}/93]", flush=True)

    w = pd.DataFrame(rows).merge(g, on="name")
    w.to_csv(OUT / "speed_crosscheck.csv", index=False)

    print(f"\n{LINE}\nITEM 2 — TWO INDEPENDENT SPEED MEASUREMENTS, ALL 93 RECORDS\n{LINE}")
    ok = w[np.isfinite(w["div_pct"])]
    print(f"records with both estimates: {len(ok)}/{len(w)}")
    print(f"divergence (bogie - TOF) / TOF: median {ok['div_pct'].median():+.1f} %, "
          f"IQR {ok['div_pct'].quantile(.25):+.1f} to {ok['div_pct'].quantile(.75):+.1f} %")
    print(f"correlation between the two speeds: "
          f"{ok['v_tof'].corr(ok['v_bogie']):+.3f}")
    for lab, sub in (("retained", ok[ok["gated"] & ok["in_session"]]),
                     ("excluded", ok[~(ok["gated"] & ok["in_session"])])):
        print(f"  {lab:<10} n={len(sub):>3}  median |divergence| "
              f"{sub['div_pct'].abs().median():>5.1f} %   "
              f">25 % divergent: {100*(sub['div_pct'].abs()>25).mean():>4.0f} %")

    # -------------------------------------------------- the 15
    print(f"\n{LINE}\nITEM 1 — THE 15 UNEXPLAINED RECORDS\n{LINE}")
    ex = w[(~w["gated"]) & w["in_session"] &
           ~(w["touches_start"] | w["touches_end"])].sort_values("local")
    print(f"n = {len(ex)}   (excluded, in-session, not a boundary split)\n")
    print(f"{'local':<20}{'score':>7}{'v_TOF':>8}{'v_bogie':>9}{'div':>8}"
          f"{'axles H1/H2/H3':>16}{'dur s':>7}{'max pm':>8}  verdict")

    verdicts = []
    for _, r in ex.iterrows():
        v = []
        if r["count_spread"] >= 6:
            v.append("head count disagreement")
        if abs(r["div_pct"]) > 25:
            v.append("speed estimates diverge")
        if r["n_h2"] <= 6:
            v.append("very short consist")
        elif r["n_h2"] >= 40:
            v.append("very long consist")
        if r["max_pm"] < 60:
            v.append("low amplitude")
        if r["dur_s"] > 8:
            v.append("long passage")
        verdict = "; ".join(v) if v else "unexplained"
        verdicts.append(verdict)
        print(f"{r['local']:%Y-%m-%d %H:%M:%S}  {r['score']:>7.2f}{r['v_tof']:>8.1f}"
              f"{r['v_bogie']:>9.1f}{r['div_pct']:>+7.0f}%"
              f"{f'{int(r.n_h1)}/{int(r.n_h2)}/{int(r.n_h3)}':>16}"
              f"{r['dur_s']:>7.1f}{r['max_pm']:>8.0f}  {verdict}")
    ex = ex.assign(verdict=verdicts)

    print(f"\n{LINE}\nDO THEY SHARE A PROPERTY?\n{LINE}")
    from collections import Counter
    tags = Counter(t for v in verdicts for t in v.split("; "))
    for t, c in tags.most_common():
        print(f"  {c:>3}/{len(ex)}  {t}")
    ret = w[w["gated"] & w["in_session"]]
    print(f"\nsame tags in the RETAINED population, for contrast:")
    print(f"  {(ret['count_spread']>=6).sum():>3}/{len(ret)}  head count disagreement "
          f"({100*(ret['count_spread']>=6).mean():.0f} %)")
    print(f"  {(ret['div_pct'].abs()>25).sum():>3}/{len(ret)}  speed estimates diverge "
          f"({100*(ret['div_pct'].abs()>25).mean():.0f} %)")
    print(f"  {(ret['n_h2']<=6).sum():>3}/{len(ret)}  very short consist "
          f"({100*(ret['n_h2']<=6).mean():.0f} %)")

    # -------------------------------------------------- montage
    fig, axes = plt.subplots(5, 3, figsize=(17, 13))
    fig.subplots_adjust(hspace=0.55, wspace=0.14, top=0.93, bottom=0.04,
                        left=0.04, right=0.99)
    for ax_, (_, r) in zip(axes.ravel(), ex.iterrows()):
        y = rolling_median_baseline(load_file(
            Path(CONFIG["paths"]["data_dir"]) / r["name"])[CH_COLS])
        t = np.arange(len(y)) / FS
        for h, c in ((1, "#1a7f37"), (2, "#2166ac"), (3, "#b2182b")):
            ax_.plot(t, head_signal(y, h), lw=0.35, color=c, alpha=0.85)
        ax_.set_title(f"{r['local']:%m-%d %H:%M}  score {r['score']:.2f}  "
                      f"{int(r['n_h2'])} axles\n{r['verdict'][:52]}", fontsize=7.5)
        ax_.tick_params(labelsize=6)
        ax_.margins(x=0)
        ax_.grid(alpha=0.25, lw=0.3)
    for ax_ in axes.ravel()[len(ex):]:
        ax_.axis("off")
    fig.suptitle("The 15 unexplained gated records — head-mean traces "
                 "(green H1, blue H2, red H3)", fontsize=12)
    png = OUT / "gated_15_montage.png"
    fig.savefig(png, dpi=115)
    ex.to_csv(OUT / "gated_15_verdicts.csv", index=False)
    print(f"\nwrote {png} and gated_15_verdicts.csv")


if __name__ == "__main__":
    classify_gated()
    speed_crosscheck()

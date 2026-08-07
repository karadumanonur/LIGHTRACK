"""Items 4-6 — what are the 23 gated-out records, and is the gate discarding
the cases the installation exists to detect?

The constant-speed solver assumes one slowness across all 25 m. A train that
accelerates, brakes or stops between heads violates that, the fit score falls,
and the record is gated out. Those are plausibly the stopping, starting and
traffic-held passages — exactly the cases Head 1 comptage / Head 2 decomptage
occupancy logic has to survive.

Every excluded record is classified individually. NOTHING is re-admitted to any
statistic here; this only establishes what they are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal, axle_peaks,
)
from lightrack.speed import estimate_accelerated                # noqa: E402
from lightrack.detect import passage_window, samples_per_peak   # noqa: E402
from lightrack.occupancy import slow_baseline, find_stationary  # noqa: E402

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


def main() -> None:
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


if __name__ == "__main__":
    main()

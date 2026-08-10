"""Passages — extract every train passage, then run the occupancy logic.

PART A  EXTRACTION. Matching anything against an external reference needs the
        time the TRAIN passed, not the time the recorder started a slot.
        Recording is contiguous, so a file boundary falls at an arbitrary point
        relative to the train and using the file start smears every match by up
        to a full slot. Times are the first/last axle at whichever head is best
        conditioned in that record. A gap larger than SPLIT_M metres separates
        two trains rather than two axles, so a record holding two passages
        yields two rows.

PART B  OCCUPANCY. Head 1 counts in, Head 2 counts out, and the section between
        them is clear when the two balance. Two things the naive reading gets
        wrong and this does not:

          1. THE HEAD ROLES SWAP WITH DIRECTION. Direction is resolved first and
             the entry/exit roles are assigned from it; Head 1 is never
             hard-coded as the entry.
          2. A TRAIN CAN SPAN TWO FILES. A passage touching a record edge
             continues in the temporally adjacent file, and counting each half
             separately would produce two imbalances from one train.

        Errors are reported separately and never merged: an OVER-count can make
        an occupied block look clear (SAFETY), an UNDER-count leaves a clear
        block occupied (AVAILABILITY).

Run with --smoke to process a few files first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Works from either location: the working copy (data/scripts/) or the repo,
# where Scripts/ sits inside the package directory itself.
_UP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UP if (_UP / "lightrack").is_dir() else _UP.parent))
from lightrack import (                                    # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, list_files, load_file, parse_filename_utc,
    to_local, rolling_median_baseline, estimate_speed, head_signal, axle_peaks,
)
from lightrack.detect import passage_window                # noqa: E402
from lightrack.safety import count_axles                   # noqa: E402
from lightrack.baseline import quiet_noise_floor           # noqa: E402
from lightrack.occupancy import slow_baseline, find_stationary  # noqa: E402


# ======================================================= PART A — EXTRACTION
FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
SPLIT_M = 30.0                      # inter-train gap, not inter-axle
pd.set_option("display.width", 240)


def best_head(y, v):
    q = {}
    for h in (1, 2, 3):
        hs = head_signal(y, h)
        w = passage_window(hs, v)
        if w is None:
            q[h] = -np.inf
            continue
        quiet = np.concatenate([hs[: w[0]], hs[w[1]:]])
        nz = quiet_noise_floor(quiet) if quiet.size > FS else quiet_noise_floor(hs)
        q[h] = float(np.percentile(hs[w[0]:w[1]], 99.0) / nz) if nz > 0 else -np.inf
    return max(q, key=q.get)


def extract_passages() -> None:
    smoke = "--smoke" in sys.argv
    files = list_files()
    if smoke:
        files = files[:3]
    rows, gaps, failed = [], [], []
    for n, p in enumerate(files, 1):
        utc = parse_filename_utc(p.name)
        loc = to_local(utc)
        try:
            y = rolling_median_baseline(load_file(p)[CH_COLS])
            spd = estimate_speed(y)
            if not spd.ok:
                failed.append((p.name, "speed solver did not converge"))
                continue
            h = best_head(y, spd.v_kmh)
            hs = head_signal(y, h)
            idx = axle_peaks(hs, spd.v_kmh, window=passage_window(hs, spd.v_kmh))
        except Exception as e:                                  # noqa: BLE001
            failed.append((p.name, repr(e)))
            continue
        if len(idx) < 4:
            failed.append((p.name, f"only {len(idx)} peaks"))
            continue
        v_mps = spd.v_kmh / 3.6
        x = (idx - idx[0]) / FS * v_mps
        cuts = np.flatnonzero(np.diff(x) > SPLIT_M)
        starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(x) - 1]
        for si, (a, b) in enumerate(zip(starts, ends)):
            if b - a < 3:
                continue
            # axle-gap signature: the short-gap population is what distinguishes
            # freight bogies from passenger stock, so it is carried here rather
            # than reconstructed later from n_axles and length alone.
            g = np.diff(x[a:b + 1])
            rows.append({
                "name": p.name, "utc": utc, "file_local": loc,
                "sub": si, "n_sub": len(starts),
                "off_s": float(idx[a]) / FS,
                "off_end_s": float(idx[b]) / FS,
                "train_local": loc + pd.Timedelta(seconds=float(idx[a]) / FS),
                "train_end_local": loc + pd.Timedelta(seconds=float(idx[b]) / FS),
                "head": h, "v_kmh": spd.v_kmh,
                "head1_first": spd.head1_first, "score": spd.score,
                "alt_score": spd.alt_score,
                "n_axles": int(b - a + 1), "length_m": float(x[b] - x[a]),
                "gap_min_m": float(g.min()), "gap_med_m": float(np.median(g)),
                "gap_p25_m": float(np.percentile(g, 25)),
                "n_gap_lt_2m5": int((g < 2.5).sum()),
            })
            # Full gap vector, persisted. Every spatial quantity in this project
            # is v * dt, so a speed-scale error rescales all of them together;
            # keeping the raw gaps makes that sensitivity a multiplication rather
            # than a re-run.
            gaps.extend({"name": p.name, "sub": si, "k": k, "gap_m": float(gv)}
                        for k, gv in enumerate(g))
        if n % 15 == 0 or smoke:
            print(f"  [{n:3d}/{len(files)}] {p.name}  "
                  f"{len(starts)} segment(s)", flush=True)

    g = pd.DataFrame(gaps)
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} passages from {d['name'].nunique()}/{len(files)} records")
    print(f"records yielding >1 passage: "
          f"{int((d.drop_duplicates('name')['n_sub'] > 1).sum())}")
    if failed:
        print(f"\n{len(failed)} records produced no passage:")
        for nm, why in failed:
            print(f"   {nm[:30]:<32} {why}")
    if smoke:
        print("\nSMOKE OK — rerun without --smoke")
        return
    d.to_csv(OUT / "passage_times.csv", index=False)
    g.to_csv(OUT / "passage_gaps.csv", index=False)
    print(f"\nwrote passage_times.csv, passage_gaps.csv ({len(g)} gaps)")


# ======================================================== PART B — OCCUPANCY
FS = CONFIG["acquisition"]["fs_hz"]
SLOT_S = CONFIG["acquisition"]["slot_s"]
OUT = Path(CONFIG["paths"]["out_dir"])
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}     # head -> PADs (Head 3 excluded)
EDGE_S = 1.5

pd.set_option("display.width", 250)
LINE = "=" * 116


def head_count(y, head: int, v_kmh: float) -> tuple[int, int, bool]:
    """Deterministic vote across a head's four sensors -> (count, spread, unanimous)."""
    hs = head_signal(y, head)
    w = passage_window(hs, v_kmh)
    if w is None:
        return 0, 0, False
    counts = []
    for pad in OPERATIONAL[head]:
        sig = y[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
        counts.append(count_axles(sig, v_kmh).count)
    c = int(np.median(counts))
    return c, int(max(counts) - min(counts)), len(set(counts)) == 1


def occupancy() -> None:
    files = list_files()
    recs = []
    for n, p in enumerate(files, 1):
        utc = parse_filename_utc(p.name)
        raw = load_file(p)[CH_COLS]
        y = rolling_median_baseline(raw)
        spd = estimate_speed(y)
        nrec = len(y)

        hs1 = head_signal(y, 1)
        w = passage_window(hs1, spd.v_kmh)
        touch_start = bool(w and w[0] < EDGE_S * FS)
        touch_end = bool(w and w[1] > nrec - EDGE_S * FS)

        c1, s1, u1 = head_count(y, 1, spd.v_kmh)
        c2, s2, u2 = head_count(y, 2, spd.v_kmh)
        st = find_stationary(head_signal(slow_baseline(raw), 1), spd.v_kmh)

        recs.append({
            "name": p.name, "utc": utc, "local": to_local(utc),
            "v_kmh": spd.v_kmh, "score": spd.score, "gated": spd.ok,
            "head1_first": spd.head1_first,
            "c_h1": c1, "c_h2": c2, "spread_h1": s1, "spread_h2": s2,
            "unanimous": u1 and u2,
            "touch_start": touch_start, "touch_end": touch_end,
            "n_stationary": len(st),
        })
        if n % 20 == 0:
            print(f"  [{n:3d}/93]", flush=True)

    df = pd.DataFrame(recs).sort_values("utc").reset_index(drop=True)

    # ---------------------------------------------------- role assignment
    df["entry_head"] = np.where(df["head1_first"], 1, 2)
    df["exit_head"] = np.where(df["head1_first"], 2, 1)
    df["count_in"] = np.where(df["head1_first"], df["c_h1"], df["c_h2"])
    df["count_out"] = np.where(df["head1_first"], df["c_h2"], df["c_h1"])
    df["balance"] = df["count_in"] - df["count_out"]

    # ---------------------------------------------------- stitching
    df["gap_to_prev_s"] = df["utc"].diff().dt.total_seconds()
    df["stitch_prev"] = (df["touch_start"] &
                         (df["gap_to_prev_s"].between(SLOT_S - 5, SLOT_S + 5)) &
                         df["touch_end"].shift(1).fillna(False))
    print(f"\n{LINE}\nSTITCHING — passages spanning a file boundary\n{LINE}")
    print(f"records whose passage touches the record START : "
          f"{int(df['touch_start'].sum())}")
    print(f"records whose passage touches the record END   : "
          f"{int(df['touch_end'].sum())}")
    st = df[df["touch_start"] | df["touch_end"]]
    if len(st):
        print(f"\n{'local':<20}{'edge':>6}{'gap to prev s':>15}{'in':>5}{'out':>5}"
              f"{'bal':>5}  stitchable?")
        for _, r in st.iterrows():
            e = ("S" if r["touch_start"] else "") + ("E" if r["touch_end"] else "")
            gp = f"{r['gap_to_prev_s']:.1f}" if np.isfinite(r["gap_to_prev_s"]) else "-"
            print(f"{r['local']:%Y-%m-%d %H:%M:%S}  {e:>6}{gp:>15}"
                  f"{int(r['count_in']):>5}{int(r['count_out']):>5}"
                  f"{int(r['balance']):>5}  {'YES' if r['stitch_prev'] else 'no adjacent file'}")
    print("\nRecording is contiguous, so a passage at a record edge continues in")
    print("the next slot IF that slot was retained. Where it was not, the train")
    print("is unrecoverable and the record must be EXCLUDED from any balance")
    print("statistic rather than counted as an imbalance.")

    # ---------------------------------------------------- direction
    print(f"\n{LINE}\nDIRECTION AND ROLE ASSIGNMENT\n{LINE}")
    print(df.groupby(["head1_first"]).size().rename(
        index={True: "Head1 first -> H1 entry, H2 exit",
               False: "Head2 first -> H2 entry, H1 exit"}).to_string())

    # ---------------------------------------------------- balance
    print(f"\n{LINE}\nOCCUPANCY BALANCE — in-count vs out-count\n{LINE}")
    clean = df[~(df["touch_start"] | df["touch_end"]) & (df["n_stationary"] == 0)]
    for lab, sub in (("all records", df),
                     ("gate-retained only", df[df["gated"]]),
                     ("clean (no edge, no stop)", clean)):
        b = sub["balance"]
        print(f"{lab:<28} n={len(sub):>3}   balanced {100*(b==0).mean():>5.1f} %   "
              f"|bal|<=1 {100*(b.abs()<=1).mean():>5.1f} %   "
              f"phantom (in>out) {100*(b>0).mean():>5.1f} %   "
              f"missed (in<out) {100*(b<0).mean():>5.1f} %")

    print(f"\n{LINE}\nEVERY FAILURE TO BALANCE\n{LINE}")
    bad = df[df["balance"] != 0].sort_values("utc")
    print(f"{'local':<20}{'gate':>6}{'v':>7}{'entry':>6}{'in':>5}{'out':>5}"
          f"{'bal':>5}{'spread H1/H2':>14}  reason")
    for _, r in bad.iterrows():
        reasons = []
        if r["touch_start"] or r["touch_end"]:
            reasons.append("passage at record edge")
        if r["n_stationary"]:
            reasons.append("stationary state")
        if max(r["spread_h1"], r["spread_h2"]) >= 2:
            reasons.append("sensors disagree within a head")
        if not reasons:
            reasons.append("unexplained")
        print(f"{r['local']:%Y-%m-%d %H:%M:%S}  {'ok' if r['gated'] else 'GATED':>6}"
              f"{r['v_kmh']:>7.0f}{int(r['entry_head']):>6}{int(r['count_in']):>5}"
              f"{int(r['count_out']):>5}{int(r['balance']):>+5}"
              f"{f'{int(r.spread_h1)}/{int(r.spread_h2)}':>14}  {'; '.join(reasons)}")

    print(f"\n{LINE}\nGATE STATUS vs COUNTING QUALITY\n{LINE}")
    g = df[df["gated"]]
    ng = df[~df["gated"]]
    print(f"{'population':<24}{'n':>5}{'balanced':>11}{'unanimous vote':>17}")
    for lab, sub in (("gate-retained", g), ("gate-rejected", ng)):
        print(f"{lab:<24}{len(sub):>5}{100*(sub['balance']==0).mean():>10.1f} %"
              f"{100*sub['unanimous'].mean():>16.1f} %")
    print("\nIf the rejected records balance as well as the retained ones, the")
    print("speed-fit gate is not a counting-quality criterion and must not be")
    print("applied to occupancy statistics.")

    df.to_csv(OUT / "occupancy_balance.csv", index=False)
    print(f"\nwrote {OUT/'occupancy_balance.csv'}")


if __name__ == "__main__":
    extract_passages()
    occupancy()

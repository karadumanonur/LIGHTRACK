"""Item 3 — the occupancy state machine.

What the layout is for (briefing §4.3): Head 1 "Comptage" counts in, Head 2
"Decomptage" counts out, the section between them is clear when the two balance.
Head 3 is the mounting-test zone and takes no part.

Two things the naive reading gets wrong and this does not:

1. **The head roles swap with direction.** 11 of 93 records run Head 3 -> Head 1.
   For those, Head 2 is the ENTRY and Head 1 the EXIT. Direction is resolved
   first and roles are assigned from it; Head 1 is never hard-coded as entry.

2. **A train can span two files.** Recording is contiguous (A-01), so a passage
   touching a record edge continues in the temporally adjacent file. Counting
   each half separately produces two imbalances from one train.

Run over the 70 retained records PLUS the 2 boundary-split cases and the 15
score-gated ones — the speed gate is a speed-quality gate, not a counting-quality
gate, and every gated record has Head 1 and Head 2 agreeing exactly on count.
Records are labelled by gate status so nothing is silently mixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, list_files, load_file, parse_filename_utc,
    to_local, rolling_median_baseline, estimate_speed, head_signal,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.safety import count_axles                       # noqa: E402
from lightrack.occupancy import slow_baseline, find_stationary  # noqa: E402

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


def main() -> None:
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
    main()

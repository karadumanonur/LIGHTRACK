"""Items 4-5 — the counting bugs, the stitched consist, and the axle distribution.

Four questions:

  A. The stitched 130-axle consist is three axles out (135 in, 132 out). It is
     the longest train in the set and the only stitched one, so where the
     discrepancy accumulates matters: at the join, at one head, or spread along?
  B. Are PAD 5's X/D disagreement and the residual phantoms the SAME bug?
     PAD 5 has the shallowest compression lobe of all twelve (0.06) and PAD 3
     the deepest (0.76), and both broke the safety rules in opposite directions
     before the p90/peak-anchor fix. If phantoms still concentrate at the
     extremes of lobe depth, the anchoring inherits lobe depth despite the fix,
     and it is one bug rather than two.
  C. Over-count broken down by speed, sensor and consist size.
  D. Axle-count distribution PER HEAD and BY RESOLVED DIRECTION, not pooled,
     with the tail eyeballed for further stitch artefacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.safety import count_axles                       # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
DATA = Path(CONFIG["paths"]["data_dir"])
OUT = Path(CONFIG["paths"]["out_dir"])
D12 = CONFIG["geometry"]["head1_to_head2_m"]
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}

pd.set_option("display.width", 250)
LINE = "=" * 116

# ================================================================== A
print(f"{LINE}\nA. THE STITCHED 130-AXLE CONSIST — where do the 3 axles go?\n{LINE}")
# NB filenames are UTC; the occupancy report prints local (UTC+2).
FA, FB = "20260528T120311,441Z.txt", "20260528T120536,855Z.txt"
ta, tb = parse_filename_utc(FA), parse_filename_utc(FB)
ya = rolling_median_baseline(load_file(DATA / FA)[CH_COLS])
yb = rolling_median_baseline(load_file(DATA / FB)[CH_COLS])
dur_a, dur_b = len(ya) / FS, len(yb) / FS
gap = (tb - ta).total_seconds() - dur_a
print(f"file A {FA}  {dur_a:.3f} s")
print(f"file B {FB}  {dur_b:.3f} s")
print(f"filename spacing {(tb-ta).total_seconds():.3f} s  ->  "
      f"DEAD TIME BETWEEN RECORDS = {gap:+.3f} s")

sa = estimate_speed(ya)
sb = estimate_speed(yb)
v = np.mean([sa.v_kmh, sb.v_kmh]) / 3.6
print(f"speed {sa.v_kmh:.1f} / {sb.v_kmh:.1f} km/h  -> {v:.1f} m/s")
print(f"train length lost in the dead time: {gap*v:.1f} m")

print(f"\n{'head':>5}{'file A':>9}{'file B':>9}{'total':>8}")
tot = {}
for h in (1, 2):
    ca, cb = [], []
    for y, c in ((ya, ca), (yb, cb)):
        hs = head_signal(y, h)
        w = passage_window(hs, estimate_speed(y).v_kmh)
        for pad in OPERATIONAL[h]:
            sig = y[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
            c.append(count_axles(sig, estimate_speed(y).v_kmh).count)
    tot[h] = int(np.median(ca)) + int(np.median(cb))
    print(f"{h:>5}{int(np.median(ca)):>9}{int(np.median(cb)):>9}{tot[h]:>8}")
print(f"\nimbalance H1 - H2 = {tot[1]-tot[2]:+d} axles")

print(f"""
The dead time is {gap:+.3f} s. The two records are CONTIGUOUS to within 6 ms,
so no axle was lost to a recording gap - the discrepancy is produced by
processing the halves SEPARATELY, not by missing data.

Note also the per-file imbalances: +8 in file A, -5 in file B, partly
cancelling. That pattern is expected and is not itself an error. Head 1 sees
the train {D12:.0f} m / {D12/v:.2f} s before Head 2, so at the instant the file
boundary falls, Head 1 has already seen more of the consist. File B reverses it.
Only the TOTAL should balance.

TEST: rejoin the two records and count on the continuous signal. If the joined
counts balance, the 3 axles were lost to per-file window and threshold effects
at the boundary, not to the data.""")

joined = pd.concat([load_file(DATA / FA)[CH_COLS],
                    load_file(DATA / FB)[CH_COLS]], ignore_index=True)
yj = rolling_median_baseline(joined)
sj = estimate_speed(yj)
print(f"\njoined record: {len(yj)/FS:.1f} s, fitted speed {sj.v_kmh:.1f} km/h")
print(f"{'head':>5}{'split (A+B)':>14}{'joined':>9}{'recovered':>11}")
jt = {}
for h in (1, 2):
    hs = head_signal(yj, h)
    w = passage_window(hs, sj.v_kmh)
    cs = []
    for pad in OPERATIONAL[h]:
        sig = yj[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
        cs.append(count_axles(sig, sj.v_kmh).count)
    jt[h] = int(np.median(cs))
    print(f"{h:>5}{tot[h]:>14}{jt[h]:>9}{jt[h]-tot[h]:>+11}")
print(f"\nimbalance: split {tot[1]-tot[2]:+d} axles  ->  joined {jt[1]-jt[2]:+d} axles")
print("""
Whichever way this lands, the operational conclusion is the same: a consist
spanning a file boundary must be STITCHED BEFORE COUNTING, not counted per file
and summed. The heads are offset along the train, so the boundary cuts each
head's view at a different point in the consist, and per-file windows and
thresholds are then fitted to two different fragments of the same train.""")

# ================================================================== B, C
print(f"\n{LINE}\nB. IS PAD 5's DISAGREEMENT THE SAME BUG AS THE PHANTOMS?\n{LINE}")
cnt = pd.read_csv(OUT / "phase2_counts.csv")
feat = pd.read_csv(OUT / "phase4_features_scored.csv")
lobe = (feat[feat["session"] == 1].groupby("pad")["neg_over_pos"].median()
        .rename("lobe_commissioning"))
cnt = cnt.merge(lobe, on="pad", how="left")

# per-sensor deviation from its own head's median count on the same record
cnt["head_med"] = cnt.groupby(["name", "head"])["count"].transform("median")
cnt["dev"] = cnt["count"] - cnt["head_med"]

g = cnt.groupby("pad").agg(
    lobe=("lobe_commissioning", "first"),
    xd_disagree=("agree", lambda s: 100 * (1 - s.mean())),
    over=("dev", lambda s: 100 * (s > 0).mean()),
    under=("dev", lambda s: 100 * (s < 0).mean()),
    n=("count", "size")).reset_index().sort_values("lobe")

print(f"{'PAD':>4}{'ch':>6}{'lobe |trough|/peak':>20}{'X/D disagree %':>16}"
      f"{'over-count %':>14}{'under-count %':>15}")
for _, r in g.iterrows():
    print(f"{int(r['pad']):>4}{CH_OF_PAD[int(r['pad'])]:>6}{r['lobe']:>20.2f}"
          f"{r['xd_disagree']:>16.0f}{r['over']:>14.0f}{r['under']:>15.0f}")

lo, hi = g["lobe"].quantile(0.25), g["lobe"].quantile(0.75)
extreme = g[(g["lobe"] <= lo) | (g["lobe"] >= hi)]
middle = g[(g["lobe"] > lo) & (g["lobe"] < hi)]
print(f"\ncorrelation(lobe depth, X/D disagreement) = "
      f"{g['lobe'].corr(g['xd_disagree']):+.3f}")
print(f"correlation(lobe depth, over-count rate)   = "
      f"{g['lobe'].corr(g['over']):+.3f}")
print(f"\nX/D disagreement: extremes of lobe depth {extreme['xd_disagree'].mean():.0f} % "
      f"vs middle {middle['xd_disagree'].mean():.0f} %")
print(f"over-count:       extremes {extreme['over'].mean():.0f} % "
      f"vs middle {middle['over'].mean():.0f} %")

# ================================================================== C
print(f"\n{LINE}\nC. OVER-COUNT BREAKDOWN\n{LINE}")
cnt["v_bin"] = pd.cut(cnt["v_kmh"], [30, 60, 80, 100, 120, 145])
cnt["consist"] = pd.cut(cnt["head_med"], [0, 8, 16, 32, 300],
                        labels=["<=8", "9-16", "17-32", ">32"])
for by, lab in (("v_bin", "speed km/h"), ("consist", "consist size"),
                ("head", "head")):
    t = cnt.groupby(by, observed=True).agg(
        n=("dev", "size"), over=("dev", lambda s: 100 * (s > 0).mean()),
        under=("dev", lambda s: 100 * (s < 0).mean())).reset_index()
    print(f"\nby {lab}:")
    print(f"{'':<12}{'n':>7}{'over %':>9}{'under %':>10}")
    for _, r in t.iterrows():
        print(f"{str(r[by]):<12}{int(r['n']):>7}{r['over']:>9.1f}{r['under']:>10.1f}")

# ================================================================== D
print(f"\n{LINE}\nD. AXLE-COUNT DISTRIBUTION, PER HEAD AND BY DIRECTION\n{LINE}")
occ = pd.read_csv(OUT / "occupancy_balance.csv")
occ["local"] = pd.to_datetime(occ["local"])
occ["dir"] = np.where(occ["head1_first"], "H1->H3", "H3->H1")
for h, col in ((1, "c_h1"), (2, "c_h2")):
    print(f"\nHead {h}:")
    for d, sub in occ.groupby("dir"):
        vc = sub[col].value_counts().sort_index()
        print(f"  {d}  n={len(sub):>3}  median {sub[col].median():>5.0f}   "
              f"counts: " + ", ".join(f"{int(k)}x{v}" for k, v in vc.items() if v))

print(f"\n{LINE}\nTAIL — everything away from the 12-axle norm\n{LINE}")
tail = occ[(occ["c_h1"] < 8) | (occ["c_h1"] > 16)].sort_values("c_h1")
print(f"{'local':<20}{'dir':>8}{'v':>7}{'H1':>5}{'H2':>5}{'bal':>5}"
      f"{'edge':>6}{'gate':>7}  reading")
for _, r in tail.iterrows():
    edge = ("S" if r["touch_start"] else "") + ("E" if r["touch_end"] else "")
    if edge:
        read = "STITCH ARTEFACT — passage clipped at a record boundary"
    elif r["c_h1"] >= 100:
        read = "long consist (freight?) — verify against the tail below"
    elif r["c_h1"] <= 6:
        read = "very short consist or clipped"
    elif r["c_h1"] in (24, 36):
        read = f"coupled units ({int(r['c_h1'])//12} x 12-axle)"
    else:
        read = "non-standard consist"
    print(f"{r['local']:%Y-%m-%d %H:%M:%S}{r['dir']:>8}{r['v_kmh']:>7.0f}"
          f"{int(r['c_h1']):>5}{int(r['c_h2']):>5}{int(r['balance']):>+5}"
          f"{edge:>6}{'ok' if r['gated'] else 'GATED':>7}  {read}")

print(f"\nmultiples of 12 (coupled class-0 units): "
      f"{int((occ['c_h1'] % 12 == 0).sum())}/{len(occ)} records")
cnt.to_csv(OUT / "counting_bug_analysis.csv", index=False)
print(f"\nwrote counting_bug_analysis.csv")

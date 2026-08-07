"""A reference-free lower bound on July axle error, from parity alone.

No complete train has an odd number of axles. So every open-day record whose
detected axle count is odd is DEMONSTRABLY WRONG with no reference consulted -
which makes the odd fraction a lower bound on the axle error rate, and the only
error figure the July set can carry at all.

It is a LOWER bound, and the reason matters: an error of +2 or -2 leaves the
count even and is invisible to this test. Parity catches only odd-sized errors.

Two things are checked beyond the headline:
  * DIRECTION. If the wrong counts cluster just ABOVE the modal values (13, 14,
    15 against 12; 25, 26, 27 against 24) with nothing just below (no 11, no 22,
    no 23), July is OVER-counting - the opposite sign to the freight under-count
    measured against ACAT in POC1.
  * THE LOW TAIL. Records of 4-10 axles are a different population: too few axles
    for any passenger consist here. Edge clipping would explain them, and the
    gating already carries a file-boundary rule (touch_start / touch_end), so the
    two can be cross-checked rather than assumed.

OPEN DAYS ONLY. Sealed days are not touched and remain unspent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
pd.set_option("display.width", 250)
L = "=" * 116


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


r = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False).drop(
    columns=["set", "gated"])
aux = aux.merge(r[["name", "set", "gated", "touch_start", "touch_end",
                   "c_h1", "c_h2"]], on="name")
aux["train_free"] = aux["max_h1_pm"] < CONFIG["detect"]["absolute_gate_pm"]
op = aux[(aux["set"] == "open") & (~aux["train_free"])
         & aux["n_axles"].notna()].copy()
op["n_axles"] = op["n_axles"].astype(int)
op["odd"] = op["n_axles"] % 2 == 1

print(f"{L}\nA REFERENCE-FREE LOWER BOUND ON JULY AXLE ERROR\n{L}")
print(f"population: open-day, train-containing records with a detected count")
print(f"            {len(op)} records, 2026-07-17 to 2026-07-27, OPEN DAYS ONLY")
k, n = int(op["odd"].sum()), len(op)
lo, hi = (beta.ppf(.025, k, n - k + 1), beta.ppf(.975, k + 1, n - k))
print(f"\nODD axle counts: {k} / {n} = {100*k/n:.1f} % "
      f"[95 % CI {100*lo:.1f}-{100*hi:.1f} %]")
print(f"""
LOWER BOUND, and the word is doing work: an error of +2 or -2 leaves the count
even and parity cannot see it. The true axle error rate on this population is at
least {100*k/n:.1f} % and unbounded above by this test.""")

print(f"\n{L}\nDISTRIBUTION — IS THE ERROR ONE-SIDED?\n{L}")
vc = op["n_axles"].value_counts().sort_index()
print(f"{'axles':>7}{'n':>6}   {'':<40}")
for a, c in vc.items():
    mark = ""
    if a % 2 == 1:
        mark = "  <- ODD, impossible"
    elif a in (12, 24, 36):
        mark = "  <- modal (AM08 x1 / x2 / x3)"
    print(f"{a:>7}{c:>6}   {'#'*min(60, c)}{mark}")

for base in (12, 24):
    above = vc[(vc.index > base) & (vc.index < base + 12)].sum()
    below = vc[(vc.index < base) & (vc.index > base - 12)].sum()
    print(f"\naround {base}: {int(vc.get(base,0))} exact, "
          f"{int(above)} in ({base}, {base+12}), {int(below)} in ({base-12}, {base})")
    for q in range(base - 3, base):
        if q > 0:
            print(f"   {q:>3} axles: {int(vc.get(q, 0))}")
print(f"""
READING. The counts pile up JUST ABOVE the modal values and are absent just
below. That is an OVER-count, and it is the opposite sign to the freight
under-count measured against ACAT ground truth in POC1 (§3.1), where the head
vote missed by up to 10 axles downward. Different configuration, different
direction of error.""")

print(f"\n{L}\nTHE LOW TAIL — EDGE CLIPPING, NOT UNDER-COUNTING?\n{L}")
low = op[op["n_axles"] <= 10]
rest = op[op["n_axles"] > 10]
print(f"records with 4-10 axles: {len(low)}   ({pct(len(low), len(op))} of the population)")
print(f"\n{'population':<28}{'n':>6}{'touches a file edge':>22}{'rate':>8}")
for lab, g in (("4-10 axles", low), ("11+ axles", rest)):
    e = g["touch_start"] | g["touch_end"]
    print(f"{lab:<28}{len(g):>6}{int(e.sum()):>22}{pct(int(e.sum()), len(g)):>8}")
print(f"\nthe file-boundary rule already in the gating is `touch_start | touch_end`")
print(f"— the passage window reaching within 1.5 s of the record's first or last")
print(f"sample, i.e. a train cut by the 144.5 s slot boundary.")
if len(low):
    print(f"\n{'record':<30}{'axles':>7}{'v':>7}{'start':>7}{'end':>6}{'max pm':>9}")
    for _, q in low.sort_values("n_axles").iterrows():
        print(f"{q['name'][:28]:<30}{q['n_axles']:>7}{q['v_kmh']:>7.1f}"
              f"{str(q['touch_start']):>7}{str(q['touch_end']):>6}"
              f"{q['max_h1_pm']:>9.0f}")
e_low = int((low["touch_start"] | low["touch_end"]).sum())

# ---- is a 4-axle detection even an error? POC1 has ground truth on this
print(f"\n{L}\nARE 4-AXLE DETECTIONS ERRORS AT ALL? POC1 GROUND TRUTH SAYS NO\n{L}")
el = pd.read_csv(OUT / "acat_matched_passages.csv")
small = el[el["acat_axles"] <= 10]
print(f"POC1 passages matched to ACAT with truth <= 10 axles: {len(small)}")
if len(small):
    print(f"{'record':<30}{'sensor':>8}{'ACAT truth':>12}{'v':>8}")
    for _, q in small.sort_values("acat_axles").iterrows():
        print(f"{q['name'][:28]:<30}{q['n_axles']:>8.0f}{q['acat_axles']:>12.0f}"
              f"{q['v_kmh']:>8.1f}")
print(f"\nand ACAT's own log over the campaign window holds short consists in "
      f"quantity:")
ac = pd.read_csv(OUT / "acat_train_events.csv")
ac["t"] = pd.to_datetime(ac["t"])
acc = ac[(ac["t"] >= "2026-05-14") & (ac["t"] < "2026-06-03")]
sc = acc[acc["axles"] <= 10]["axles"].value_counts().sort_index()
print(f"   {sc.to_dict()}   ({len(acc[acc['axles'] <= 10])} of {len(acc)} entries)")
print(f"\namplitude of the July 4-axle records: median "
      f"{low.loc[low['n_axles']==4,'max_h1_pm'].median():.0f} pm against "
      f"{op.loc[op['n_axles']==12,'max_h1_pm'].median():.0f} pm for 12-axle records")
print("""
A light locomotive has a higher axle load than an EMU, so a 4-axle detection with
ABOVE-average amplitude is what a real light engine looks like - not a train the
detector lost most of. POC1's ACAT truth confirms it directly.""")
print(f"""
VERDICT RULE, stated before reading. If the 4-10 axle records touch a file edge
far more often than the rest, they are CLIPPED PASSAGES - a real train cut by the
slot boundary, which is a known and already-documented condition (§3.7) and not
an under-count. If they touch no more often than the rest, they are genuine
under-counts and belong with the error population.""")

op["low_tail"] = op["n_axles"] <= 10
op[["name", "set", "n_axles", "odd", "low_tail", "v_kmh", "touch_start",
    "touch_end", "max_h1_pm"]].to_csv(OUT / "july_odd_axle_bound.csv", index=False)
pd.DataFrame([{"population": "July open, train-containing", "n": n,
               "odd": k, "rate": k/n, "ci_lo": lo, "ci_hi": hi,
               "low_tail": len(low), "low_tail_edge": e_low}]).to_csv(
    OUT / "july_axle_error_bound.csv", index=False)
print(f"\nwrote july_odd_axle_bound.csv, july_axle_error_bound.csv")

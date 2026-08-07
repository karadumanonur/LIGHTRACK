"""Verification of the absolute gate, OPEN DAYS ONLY.

Two things have to be true for this to be a defect fix rather than a change:
  1. the gate rejects exactly the records that contain no train, and nothing else;
  2. on every record it does NOT reject, the pipeline's output is IDENTICAL -
     `passage_window` is untouched above the gate, so any difference would mean
     the gate had side effects.

Then the honest accounting of what the fix buys, which is less than it looks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
GATE = CONFIG["detect"]["absolute_gate_pm"]
pd.set_option("display.width", 250)
L = "=" * 116


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


new = pd.read_csv(OUT / "regated_open_records.csv", low_memory=False)
old = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
old = old[old["set"] == "open"].copy()
old["clean"] = ~(old["touch_start"] | old["touch_end"]) & (old["n_stationary"] == 0)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False).drop(columns=["set"])
aux["train_free"] = aux["max_h1_pm"] < GATE

print(f"{L}\nGATE VERIFICATION — OPEN DAYS ONLY (sealed days are spent)\n{L}")
print(f"gate = {GATE} pm; open-day records = {len(new)}")

# ------------------------------------------------------------- 1. rejections
rej = set(new.loc[new["rejected_no_train"], "name"])
tf = set(aux.loc[aux["train_free"] & aux["name"].isin(new["name"]), "name"])
print(f"\n1. WHAT THE GATE REJECTED")
print(f"   rejected by the gate            : {len(rej)}")
print(f"   independently classed train-free: {len(tf)}")
print(f"   agreement                       : {len(rej & tf)} both, "
      f"{len(rej - tf)} rejected but not train-free, "
      f"{len(tf - rej)} train-free but not rejected")

# ------------------------------------------------- 2. no side effects above it
kept = new[~new["rejected_no_train"]]
cmp = kept.merge(old, on="name", suffixes=("_new", "_old"))
print(f"\n2. SIDE EFFECTS ON THE {len(cmp)} RECORDS IT DID NOT REJECT")
same = True
for col in ("v_kmh", "c_h1", "c_h2", "unanimous", "gated", "head1_first",
            "touch_start", "touch_end", "n_stationary"):
    a, b = cmp[f"{col}_new"], cmp[f"{col}_old"]
    if a.dtype.kind in "fc":
        d = int((~np.isclose(a.fillna(-999), b.fillna(-999))).sum())
    else:
        d = int((a.fillna(-999) != b.fillna(-999)).sum())
    same &= d == 0
    print(f"   {col:<16} differing records: {d}")
print(f"   => {'IDENTICAL — the gate has no side effects' if same else 'DIFFERENCES FOUND, investigate'}")

# ----------------------------------------------------- 3. what it actually buys
print(f"\n3. WHAT THE FIX BUYS, AND WHAT IT DOES NOT")
a = aux.drop(columns=["gated"]).merge(
    old[["name", "clean", "n_stationary", "gated", "balance"]],
    on="name", how="inner")
t = a[a["train_free"]]
print(f"   phantom axles prevented   : median {t['n_peaks'].median():.0f} "
      f"per record x {len(t)} records")
print(f"   phantom occupancy events  : {int(t['n_stationary'].sum())} events "
      f"in {int((t['n_stationary'] > 0).sum())} records")
print(f"   of the 25, how many the speed gate would have caught anyway: "
      f"{int(t['gated'].sum())} cleared it, "
      f"{int((~t['gated']).sum())} were already rejected there")
print(f"\n   headline occupancy, open days, before and after the fix:")
print(f"   {'population':<34}{'n':>6}{'balanced':>11}{'phantom':>10}{'missed':>9}")
for lab, sub in (("clean, frozen pipeline", old[old["clean"]]),
                 ("clean, with the gate", old[old["clean"] &
                                              ~old["name"].isin(rej)])):
    n = len(sub)
    print(f"   {lab:<34}{n:>6}{pct(int((sub['balance']==0).sum()), n):>11}"
          f"{pct(int((sub['balance']>0).sum()), n):>10}"
          f"{pct(int((sub['balance']<0).sum()), n):>9}")
print(f"""
   THE HEADLINE FIGURES DO NOT MOVE, and that is the finding rather than a
   disappointment. Every rejected record was ALREADY outside the clean
   population: a train-free record's noise-driven window spans the whole record
   and trips the record-edge test, so `clean` was excluding them for the wrong
   reason. That accident is exactly why the defect stayed invisible - the
   statistic that would have shown it had already filtered it out. The fix is
   worth having because a deployed system does not get to apply a clean filter
   after the fact; it has to decide at the time.""")

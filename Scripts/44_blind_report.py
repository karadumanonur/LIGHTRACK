"""Step 3 report — frozen pipeline vs Valid_data_new. Sealed and open, side by side.

Nothing here fits anything. Every threshold is the frozen POC1 value:
    pair  < -0.727      trend < -0.423      shape > +0.736
    speed gate score >= 0.70

Self-diagnosis is reported under two clearly separated names:
  FALSE-ALARM TEST  - baseline = this set's own first day. Valid regardless of
                      what happened in the 45-day gap.
  CONTINUITY TEST   - baseline = POC1 session 1 (2026-05-15). Valid only if
                      nothing was re-bonded; confounded otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD                          # noqa: E402
from lightrack.health import DEGRADING, HEALTHY                  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SEALED = {pd.Timestamp(d).date() for d in
          ("2026-07-17", "2026-07-20", "2026-07-23", "2026-07-26")}
THR_PAIR, THR_TREND, THR_SHAPE = -0.727, -0.423, 0.736
KEEP2 = {1: [1, 2], 2: [5, 6], 3: [9, 10]}
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}
pd.set_option("display.width", 250)
LINE = "=" * 116


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


r = pd.read_csv(OUT / "blind_records.csv")
r["local"] = pd.to_datetime(r["local"])
r["date"] = r["local"].dt.date
r["set"] = np.where(r["date"].isin(SEALED), "SEALED", "open")
s = pd.read_csv(OUT / "blind_sensors.csv")
cnt = s[s.get("diag").isna()] if "diag" in s else s
dia = s[s.get("diag") == True] if "diag" in s else pd.DataFrame()
cnt = cnt.merge(r[["name", "date", "set"]], on="name", how="left")
if len(dia):
    dia = dia.merge(r[["name", "date", "set", "local"]], on="name", how="left")

print(f"{LINE}\nSTEP 3 — BLIND RUN, FROZEN CONFIGURATION\n{LINE}")
print(f"records processed: {len(r)}/609    "
      f"sealed {int((r['set']=='SEALED').sum())}, open {int((r['set']=='open').sum())}")
print(f"days: sealed {sorted(SEALED)}")

# ---------------------------------------------------- speed and direction
print(f"\n{LINE}\nSPEED AND DIRECTION\n{LINE}")
print(f"{'set':<9}{'n':>6}{'gate pass':>11}{'v p10':>8}{'v med':>8}{'v p90':>8}"
      f"{'H1 first':>11}")
for k, g in list(r.groupby("set")) + [("ALL", r)]:
    print(f"{k:<9}{len(g):>6}{100*g['gated'].mean():>10.1f}%"
          f"{g['v_kmh'].quantile(.1):>8.0f}{g['v_kmh'].median():>8.0f}"
          f"{g['v_kmh'].quantile(.9):>8.0f}{100*g['head1_first'].mean():>10.1f}%")
print(f"\nPOC1 for comparison: gate pass 76/93 = 81.7 %, speed 32.5-141.0 "
      f"(median 83.5), H1-first 88.2 %")

# ---------------------------------------------------- occupancy
print(f"\n{LINE}\nOCCUPANCY BALANCE\n{LINE}")
r["count_in"] = np.where(r["head1_first"], r["c_h1"], r["c_h2"])
r["count_out"] = np.where(r["head1_first"], r["c_h2"], r["c_h1"])
r["balance"] = r["count_in"] - r["count_out"]
r["clean"] = ~(r["touch_start"] | r["touch_end"]) & (r["n_stationary"] == 0)
print(f"{'population':<24}{'set':<9}{'n':>6}{'balanced':>11}{'phantom':>10}"
      f"{'missed':>9}")
for pop, mask in (("all records", r["name"].notna()), ("clean only", r["clean"])):
    for k in ("SEALED", "open", "ALL"):
        g = r[mask & ((r["set"] == k) if k != "ALL" else True)]
        g = g.dropna(subset=["balance"])
        if not len(g):
            continue
        print(f"{pop:<24}{k:<9}{len(g):>6}{100*(g['balance']==0).mean():>10.1f}%"
              f"{100*(g['balance']>0).mean():>9.1f}%{100*(g['balance']<0).mean():>8.1f}%")
print(f"\nPOC1 clean population: balanced 97.7 %, phantom 1.1 %, missed 1.1 %")

# ---------------------------------------------------- by consist size
print(f"\n{LINE}\nBY CONSIST SIZE — deviation from the sensor's own head median\n{LINE}")
print("(dispersion, NOT a phantom rate - see A-45)\n")
cnt["hm"] = cnt.groupby(["name", "head"])["count"].transform("median")
cnt["dev"] = cnt["count"] - cnt["hm"]
cnt["bin"] = pd.cut(cnt["hm"], [0, 8, 16, 32, 400], labels=["<=8", "9-16", "17-32", ">32"])
print(f"{'consist':<9}{'set':<9}{'trains':>8}{'reads high':>12}{'reads low':>11}")
for b in ["<=8", "9-16", "17-32", ">32"]:
    for k in ("SEALED", "open"):
        g = cnt[(cnt["bin"] == b) & (cnt["set"] == k)]
        if not len(g):
            continue
        print(f"{b:<9}{k:<9}{g['name'].nunique():>8}"
              f"{100*(g['dev']>0).mean():>11.1f}%{100*(g['dev']<0).mean():>10.1f}%")
print(f"\nPOC1: <=8 0.0 %, 9-16 0.2 %, 17-32 1.3 %, >32 26.3 % (reads high)")

# ---------------------------------------------------- detector agreement
print(f"\n{LINE}\nDETECTOR AGREEMENT\n{LINE}")
print(f"{'set':<9}{'n sensors':>11}{'X/D agree':>12}{'within-head unanimous':>24}"
      f"{'cross-head |H1-H2|<=1':>24}")
for k in ("SEALED", "open", "ALL"):
    g = cnt[(cnt["set"] == k) if k != "ALL" else cnt["name"].notna()]
    rr = r[(r["set"] == k) if k != "ALL" else r["name"].notna()].dropna(subset=["balance"])
    print(f"{k:<9}{len(g):>11}{100*g['agree'].mean():>11.1f}%"
          f"{100*rr['unanimous'].mean():>23.1f}%"
          f"{100*(rr['balance'].abs()<=1).mean():>23.1f}%")
print(f"\nPOC1: X/D agree 91.7 %, within-head unanimous 78.9 %")

# ---------------------------------------------------- train-free / stationary
print(f"\n{LINE}\nTRAIN-FREE RECORDS AND STATIONARY STATE\n{LINE}")
notrain = r[r["c_h1"].isna() | (r["c_h1"] < 2)]
print(f"records with no detectable passage: {len(notrain)}/{len(r)} "
      f"({100*len(notrain)/len(r):.1f} %)")
if len(notrain):
    print("   -> these are the route to the OPERATIONAL DISTURBANCE FLOOR (A-15),")
    print("      which POC1 could not bound at all.")
    print(notrain.groupby("date").size().to_string())
st = r[r["n_stationary"] > 0]
print(f"\nrecords with a stationary plateau: {len(st)}/{len(r)}")
if len(st):
    print(f"   total stationary time {st['stationary_s'].sum():.0f} s; "
          f"longest {st['stationary_s'].max():.1f} s")
    print(st[["local", "v_kmh", "stationary_s", "c_h1", "c_h2"]].head(15).to_string(index=False))

# ---------------------------------------------------- long consists
print(f"\n{LINE}\nLONG CONSISTS — the >32 bin POC1 had only 10 of\n{LINE}")
lng = r[r["c_h1"] > 32]
print(f"records with >32 axles: {len(lng)} (POC1 had 10)")
if len(lng):
    print(f"   axle-count distribution: "
          f"{lng['c_h1'].describe()[['min','50%','max']].to_dict()}")
    print(f"   balanced {100*(lng['balance']==0).mean():.1f} %   "
          f"|bal|<=1 {100*(lng['balance'].abs()<=1).mean():.1f} %")
print(f"\nmultiples of 12: {int((r['c_h1'] % 12 == 0).sum())}/{len(r)} records")

r.to_csv(OUT / "blind_records_scored.csv", index=False)
print(f"\nwrote blind_records_scored.csv")

"""TASK 2 and TASK 3 — count-blind matching, then the accuracy figure.

TASK 2's rule is load-bearing. 75 % of ACAT trains are 12 or 24 axles, so a
matcher that used the count to choose a partner would mostly succeed by
construction and the accuracy figure that followed would be circular. NO AXLE
COUNT IS READ ANYWHERE IN THE MATCHER. The window comes from 63_acat_calibrate.py,
which derived it from the timetable and the geometry alone.

WINDOW, and why it is wide. The site is 1577 m before Le Campinaire (chainage)
and the through-train fit gives 1680 m with a 15.5 s residual MAD. At the fastest
observed speed that is 43 s; a train that STOPS at Le Campinaire — S61-1 and
S61-2 do, with a 39-51 s dwell — arrives 286 s after the sensor at the median.
The window must span both or the accuracy figure is measured on through trains
only. It therefore spans [+0, +400] s, and the price is ambiguity, reported
separately as required.

TASK 3 reports exact agreement and the signed error, per head and per sensor,
by consist size, with edge-clipped records excluded BY RULE.
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
WIN_LO, WIN_HI = 0.0, 400.0
pd.set_option("display.width", 250)
L = "=" * 116


def ci(k, n):
    if n == 0:
        return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1),
            1.0 if k >= n else beta.ppf(.975, k + 1, n - k))


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


acat = pd.read_csv(OUT / "acat_train_events.csv")
acat["t"] = pd.to_datetime(acat["t"])
d = pd.read_csv(OUT / "passage_times.csv")
d["train_local"] = pd.to_datetime(d["train_local"]).dt.tz_localize(None)
occ = pd.read_csv(OUT / "occupancy_balance.csv")
d = d.merge(occ[["name", "touch_start", "touch_end", "gated", "c_h1", "c_h2"]],
            on="name", how="left")

AT = np.sort(acat["t"].to_numpy("datetime64[s]").astype("int64"))
AX = acat.sort_values("t")["axles"].to_numpy()
st = d["train_local"].to_numpy("datetime64[s]").astype("int64")

print(f"{L}\nTASK 2 — MATCHING ON TIME ALONE\n{L}")
print(f"window: sensor passage + [{WIN_LO:.0f}, {WIN_HI:.0f}] s. No count is read.")
lo = np.searchsorted(AT, st + WIN_LO)
hi = np.searchsorted(AT, st + WIN_HI)
d["n_in_window"] = hi - lo
d["acat_i"] = np.where(hi - lo == 1, lo, -1)
d["acat_axles"] = np.where(d["acat_i"] >= 0, AX[np.clip(d["acat_i"], 0, None)], np.nan)
d["acat_dt"] = np.where(d["acat_i"] >= 0,
                        AT[np.clip(d["acat_i"], 0, None)] - st, np.nan)

n = len(d)
n0 = int((d["n_in_window"] == 0).sum())
n1 = int((d["n_in_window"] == 1).sum())
n2 = int((d["n_in_window"] >= 2).sum())
print(f"\n{'outcome':<44}{'n':>6}{'of 76':>9}")
print(f"{'MATCHED — exactly one ACAT event in the window':<44}{n1:>6}{pct(n1,n):>9}")
print(f"{'AMBIGUOUS — two or more in the window':<44}{n2:>6}{pct(n2,n):>9}")
print(f"{'UNMATCHED — none in the window':<44}{n0:>6}{pct(n0,n):>9}")
print(f"\nambiguous cases carry {d.loc[d['n_in_window']>=2,'n_in_window'].median():.0f} "
      f"candidates at the median (max "
      f"{int(d['n_in_window'].max())}); they are EXCLUDED from the accuracy "
      f"figure rather than resolved by count, which would be circular.")
print(f"\ndelta on the matched set: median {d.loc[d['acat_i']>=0,'acat_dt'].median():.0f} s, "
      f"p10 {d.loc[d['acat_i']>=0,'acat_dt'].quantile(.1):.0f}, "
      f"p90 {d.loc[d['acat_i']>=0,'acat_dt'].quantile(.9):.0f}")

# --------------------------------------------------- exclusion rule, named
d["edge_clipped"] = d["touch_start"].fillna(False) | d["touch_end"].fillna(False)
elig = d[(d["n_in_window"] == 1) & (~d["edge_clipped"])].copy()
print(f"\n{L}\nEXCLUSION RULE, APPLIED BEFORE LOOKING AT ANY COUNT\n{L}")
print(f"edge-clipped records (passage touches the first or last 1.5 s of the")
print(f"file) cannot contain the whole train and MUST mismatch by construction.")
print(f"   matched passages                : {n1}")
print(f"   of which edge-clipped, excluded : "
      f"{int(d.loc[d['n_in_window']==1,'edge_clipped'].sum())}")
print(f"   ELIGIBLE FOR THE ACCURACY FIGURE: {len(elig)}")

# ================================================================== TASK 3
print(f"\n{L}\nTASK 3 — COUNTING ACCURACY AGAINST GROUND TRUTH\n{L}")
sens = pd.read_csv(OUT / "phase2_counts.csv")
sens = sens[sens["name"].isin(elig["name"])].merge(
    elig[["name", "acat_axles", "n_axles", "v_kmh"]], on="name")
elig["err_h1"] = elig["c_h1"] - elig["acat_axles"]
elig["err_h2"] = elig["c_h2"] - elig["acat_axles"]
elig["err_diag"] = elig["n_axles"] - elig["acat_axles"]

print(f"HEAD MEDIAN VOTE vs ACAT truth, n = {len(elig)} passages")
print(f"{'output':<28}{'exact':>8}{'rate':>9}{'95 % CI':>18}"
      f"{'median err':>12}{'mean err':>10}{'|err| p90':>11}")
for lab, col in (("Head 1 median (PADs 1-4)", "err_h1"),
                 ("Head 2 median (PADs 5-8)", "err_h2"),
                 ("diagnostic chain, head 1", "err_diag")):
    e = elig[col].dropna()
    k = int((e == 0).sum())
    c = ci(k, len(e))
    print(f"{lab:<28}{k:>8}{pct(k,len(e)):>9}"
          f"{f'{100*c[0]:.1f}-{100*c[1]:.1f}%':>18}"
          f"{e.median():>12.1f}{e.mean():>10.2f}{e.abs().quantile(.9):>11.1f}")

print(f"\nsigned error distribution, head median vote:")
for lab, col in (("Head 1", "err_h1"), ("Head 2", "err_h2")):
    e = elig[col].dropna()
    vc = e.value_counts().sort_index()
    print(f"   {lab}: " + "  ".join(f"{int(k):+d}:{v}" for k, v in vc.items()))

print(f"\n{L}\nPER SENSOR, NOT JUST THE HEAD VOTE\n{L}")
sens["err"] = sens["count"] - sens["acat_axles"]
print(f"{'pad':>4}{'head':>6}{'n':>5}{'exact':>8}{'rate':>9}{'median err':>12}"
      f"{'mean err':>10}{'over':>7}{'under':>7}")
rows = []
for pad, g in sens.groupby("pad"):
    e = g["err"].dropna()
    k = int((e == 0).sum())
    print(f"{int(pad):>4}{int(g['head'].iloc[0]):>6}{len(e):>5}{k:>8}"
          f"{pct(k,len(e)):>9}{e.median():>12.1f}{e.mean():>10.2f}"
          f"{int((e>0).sum()):>7}{int((e<0).sum()):>7}")
    rows.append(dict(pad=int(pad), n=len(e), exact=k, rate=k/len(e),
                     median_err=e.median(), mean_err=e.mean()))

print(f"\n{L}\nBY CONSIST SIZE — THE >32-AXLE BIN IS THE ONE THAT MATTERS\n{L}")
elig["bin"] = pd.cut(elig["acat_axles"], [0, 12, 24, 32, 400],
                     labels=["<=12", "13-24", "25-32", ">32"])
print(f"{'bin':<10}{'n':>5}{'H1 exact':>11}{'H2 exact':>11}"
      f"{'H1 med err':>13}{'H2 med err':>13}{'H1 mean':>10}{'H2 mean':>10}")
for b, g in elig.groupby("bin", observed=True):
    k1 = int((g["err_h1"] == 0).sum()); k2 = int((g["err_h2"] == 0).sum())
    print(f"{str(b):<10}{len(g):>5}{pct(k1,len(g)):>11}{pct(k2,len(g)):>11}"
          f"{g['err_h1'].median():>13.1f}{g['err_h2'].median():>13.1f}"
          f"{g['err_h1'].mean():>10.2f}{g['err_h2'].mean():>10.2f}")
big = elig[elig["acat_axles"] > 32]
print(f"\nthe >32-axle passages individually (n={len(big)}):")
if len(big):
    print(f"{'record':<28}{'v':>7}{'ACAT':>7}{'H1':>6}{'H2':>6}{'diag':>7}"
          f"{'err H1':>8}{'err H2':>8}")
    for _, r in big.sort_values("acat_axles").iterrows():
        print(f"{r['name'][:26]:<28}{r['v_kmh']:>7.1f}{r['acat_axles']:>7.0f}"
              f"{r['c_h1']:>6.0f}{r['c_h2']:>6.0f}{r['n_axles']:>7.0f}"
              f"{r['err_h1']:>+8.0f}{r['err_h2']:>+8.0f}")

elig.to_csv(OUT / "acat_matched_passages.csv", index=False)
sens.to_csv(OUT / "acat_per_sensor_error.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "acat_per_pad_accuracy.csv", index=False)
d.to_csv(OUT / "acat_match_outcomes.csv", index=False)
print(f"\nwrote acat_matched_passages.csv, acat_per_sensor_error.csv, "
      f"acat_per_pad_accuracy.csv, acat_match_outcomes.csv")

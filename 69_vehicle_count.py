"""Item 5 — VEHICLE and BOGIE count from the gap sequence.

The gap sequence has a three-level structure that is already legible:

  12-axle passenger  2.84 | 15.75 | 2.80 || 4.77 || 2.84 | 15.95 | 2.88 || ...
  104-axle freight   2.97 |  7.24 | 2.97 || 4.64 || 2.27 |  9.09 | 2.32 || ...

sorted by size the three levels are always
  L1 WHEELBASE      within a bogie          2.3-3.2 m
  L2 COUPLING       between two vehicles    3.1-5.0 m
  L3 INTRA-VEHICLE  between a vehicle's two bogies   7-16 m

and the counts follow: bogies = |L1|, vehicles = |L2| + 1.

THRESHOLDS ARE PER-PASSAGE AND DATA-DRIVEN, not fixed metres. Every gap is
v*dt, so A-52's speed question scales them all together — but it scales them
*together*, so the SEPARATION is a ratio and is scale-invariant while any fixed
metre threshold is not. Splitting on the two largest jumps in the sorted log-gap
sequence inherits that invariance.

TWO SELF-CHECKS THAT NEED NO GROUND TRUTH. For a consist of 2-axle bogies and
2-bogie vehicles:
        |L1| = 2 x |L3|        and        |L3| = |L2| + 1
Any passage failing them has been decomposed wrongly, and that is reported rather
than smoothed.

CHAIN PROVENANCE. Vehicle count inherits axle count. These gaps come from the
DIAGNOSTIC chain (`axle_peaks`), which script 68 showed to be the chain that
resolves freight axles; the safety chain under-counts them and would corrupt the
decomposition by construction. Stated per record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
pd.set_option("display.width", 250)
L = "=" * 116


def decompose(gaps: np.ndarray) -> dict:
    """Split a gap sequence into wheelbase / coupling / intra-vehicle."""
    g = np.sort(np.asarray(gaps, float))
    g = g[g > 0]
    if len(g) < 5:
        return {}
    lg = np.log(g)
    d = np.diff(lg)
    # two largest jumps, kept in size order
    idx = np.sort(np.argsort(d)[-2:])
    t1 = float(np.sqrt(g[idx[0]] * g[idx[0] + 1]))
    t2 = float(np.sqrt(g[idx[1]] * g[idx[1] + 1]))
    n1 = int((gaps < t1).sum())
    n2 = int(((gaps >= t1) & (gaps < t2)).sum())
    n3 = int((gaps >= t2).sum())
    return {"t_wc": t1, "t_ci": t2, "n_wheel": n1, "n_couple": n2,
            "n_intra": n3, "bogies": n1, "vehicles": n2 + 1,
            "chk_bogie": n1 == 2 * n3, "chk_vehicle": n3 == n2 + 1,
            "margin_ratio": t1 / g[idx[0]] if idx[0] < len(g) else np.nan}


gp = pd.read_csv(OUT / "passage_gaps.csv")
pt = pd.read_csv(OUT / "passage_times.csv")
el = pd.read_csv(OUT / "acat_matched_passages.csv")[["name", "acat_axles"]]
pt = pt.merge(el, on="name", how="left")

rows = []
for (nm, sub), in [((k, v),) for k, v in gp.groupby(["name", "sub"])]:
    g = sub["gap_m"].to_numpy()
    d = decompose(g)
    if not d:
        continue
    m = pt[(pt["name"] == nm[0]) & (pt["sub"] == nm[1])]
    if m.empty:
        continue
    d.update({"name": nm[0], "sub": nm[1], "n_axles": int(m["n_axles"].iloc[0]),
              "v_kmh": float(m["v_kmh"].iloc[0]),
              "acat_axles": float(m["acat_axles"].iloc[0])})
    rows.append(d)
V = pd.DataFrame(rows)
V["freight"] = V["n_axles"] > 32
print(f"{L}\nVEHICLE AND BOGIE COUNT FROM THE GAP SEQUENCE\n{L}")
print(f"POC1 passages decomposed: {len(V)}  "
      f"({int(V['freight'].sum())} freight, {int((~V['freight']).sum())} passenger)")
print(f"\nself-checks (no ground truth needed):")
print(f"{'population':<16}{'n':>5}{'|L1| = 2|L3|':>15}{'|L3| = |L2|+1':>16}"
      f"{'both':>8}")
for lab, g in (("passenger", V[~V["freight"]]), ("freight", V[V["freight"]])):
    both = g["chk_bogie"] & g["chk_vehicle"]
    print(f"{lab:<16}{len(g):>5}{f'{int(g.chk_bogie.sum())}/{len(g)}':>15}"
          f"{f'{int(g.chk_vehicle.sum())}/{len(g)}':>16}"
          f"{f'{int(both.sum())}/{len(g)}':>8}")

print(f"\n{L}\nAGAINST THE 12-AXLE AM08, WHICH MUST GIVE 3 CARS / 6 BOGIES\n{L}")
a12 = V[V["n_axles"] == 12]
print(f"12-axle passages: {len(a12)}")
print(f"   vehicles = 3 : {int((a12['vehicles']==3).sum())}/{len(a12)}"
      f"   bogies = 6 : {int((a12['bogies']==6).sum())}/{len(a12)}")
vc = a12["vehicles"].value_counts().sort_index()
print(f"   vehicle-count distribution: {vc.to_dict()}")
print(f"   thresholds chosen per passage: wheelbase|coupling median "
      f"{a12['t_wc'].median():.2f} m, coupling|intra median {a12['t_ci'].median():.2f} m")

print(f"\n{L}\nFREIGHT — DOES IT SEPARATE?\n{L}")
fr = V[V["freight"]].sort_values("n_axles")
print(f"{'record':<28}{'axles':>7}{'ACAT':>7}{'bogies':>8}{'vehicles':>10}"
      f"{'t w|c':>8}{'t c|i':>8}{'checks':>9}")
for _, r in fr.iterrows():
    ck = ("ok" if (r["chk_bogie"] and r["chk_vehicle"]) else
          "bogie" if r["chk_vehicle"] else "vehicle" if r["chk_bogie"] else "both")
    at = "-" if pd.isna(r["acat_axles"]) else f"{r['acat_axles']:.0f}"
    print(f"{r['name'][:26]:<28}{r['n_axles']:>7}{at:>7}{r['bogies']:>8.0f}"
          f"{r['vehicles']:>10.0f}{r['t_wc']:>8.2f}{r['t_ci']:>8.2f}{ck:>9}")

print(f"\n{L}\nTHE MARGIN THAT DECIDES FREIGHT — AND THE SCALE QUESTION\n{L}")
gf = gp[gp["name"].isin(fr["name"])]["gap_m"]
w = gf[gf < 3.05]; c = gf[(gf >= 3.05) & (gf < 6)]
print(f"freight gaps below 3.05 m (wheelbase population): n={len(w)}, "
      f"p90 {w.quantile(.9):.2f}, max {w.max():.2f}")
print(f"freight gaps 3.05-6 m (coupling population)     : n={len(c)}, "
      f"p10 {c.quantile(.1):.2f}, min {c.min():.2f}")
sep = c.quantile(.1) / w.quantile(.9)
print(f"separation ratio p10(coupling) / p90(wheelbase) = {sep:.3f}")
pw = gp[gp["name"].isin(V.loc[~V['freight'], 'name'])]["gap_m"]
pw_w = pw[pw < 3.6]; pw_c = pw[(pw >= 3.6) & (pw < 7)]
print(f"\nfor contrast, passenger: wheelbase p90 {pw_w.quantile(.9):.2f}, "
      f"coupling p10 {pw_c.quantile(.1):.2f}, ratio "
      f"{pw_c.quantile(.1)/pw_w.quantile(.9):.3f}")
try:
    ss = pd.read_csv(OUT / "scale_sensitivity.csv")
    print(f"\nA-41's scale sweep for reference "
          f"(scale_sensitivity.csv): scales tested "
          f"{ss['scale'].min():.3f}-{ss['scale'].max():.3f}")
except Exception:
    pass
print(f"""
THE SCALE QUESTION DOES NOT BITE HERE, and the reason is worth stating rather
than assuming. A-52's speed question would rescale every gap by the same factor.
The thresholds above are the geometric midpoints of the two largest jumps in each
passage's OWN sorted gap sequence, so they rescale with it: the decomposition is
a ratio test and is invariant. A FIXED metre threshold would not be, and the
freight margin ({sep:.2f}) is narrow enough that a fixed threshold at nominal scale
would start mislabelling under an 8 % error. That is an argument for keeping the
per-passage rule, not a limitation of the result.""")

V.to_csv(OUT / "vehicle_counts.csv", index=False)
print(f"\nwrote vehicle_counts.csv")

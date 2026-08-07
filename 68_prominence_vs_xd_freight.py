"""Item 3 — the prerequisite test POC2 §7 names, run with ground truth.

THE QUESTION. Is the freight under-count ALGORITHMIC or is it the
sensor-configuration problem §7 describes? The test is to run BOTH detectors over
THE SAME baseline-removed trace, in the SAME window, on the four POC1 freight
consists that carry ACAT truth. Same data, same geometry, same influence extent -
any difference is the algorithm and nothing else.

  X/D          lightrack.safety.count_axles  — X-crossing and D-crossing,
               solution-set combination. The SIL4 safety chain.
  prominence   lightrack.detect.axle_peaks   — scipy find_peaks with a
               speed-adaptive minimum separation. The diagnostic chain.

AND NAME THE REFERENCE. A-41 forbids quoting the two chains against each other
without saying what is differenced against what. The July table used
max(c_h1, c_h2) — the better of the two head MEDIAN VOTES — which was not stated.
Four references are reported here so the choice is visible.

Nothing is applied. The safety chain keeps X/D.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, load_file, rolling_median_baseline,
    estimate_speed, head_signal, axle_peaks,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.safety import count_axles                       # noqa: E402

DATA = Path(CONFIG["paths"]["data_dir"])
OUT = Path(CONFIG["paths"]["out_dir"])
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}
pd.set_option("display.width", 250)
L = "=" * 116

el = pd.read_csv(OUT / "acat_matched_passages.csv")
fr = el[el["acat_axles"] > 32].sort_values("acat_axles")
print(f"{L}\nSAME TRACE, TWO DETECTORS, GROUND TRUTH\n{L}")
print(f"POC1 freight consists with ACAT truth: {len(fr)}")

rows = []
for _, r in fr.iterrows():
    p = DATA / r["name"]
    y = rolling_median_baseline(load_file(p)[CH_COLS])
    spd = estimate_speed(y)
    for h, pads in OPERATIONAL.items():
        hs = head_signal(y, h)
        w = passage_window(hs, spd.v_kmh)
        if w is None:
            continue
        for pad in pads:
            sig = y[CH_OF_PAD[pad]].to_numpy()
            xd = count_axles(sig[w[0]:w[1]], spd.v_kmh)
            pk = axle_peaks(sig, spd.v_kmh, window=w)
            rows.append({"name": r["name"], "truth": r["acat_axles"],
                         "v_kmh": spd.v_kmh, "head": h, "pad": pad,
                         "xd": xd.count, "n_x": xd.n_x, "n_d": xd.n_d,
                         "prom": len(pk)})
    print(f"  {r['name'][:28]:<30} truth {r['acat_axles']:.0f}", flush=True)

s = pd.DataFrame(rows)
s["err_xd"] = s["xd"] - s["truth"]
s["err_prom"] = s["prom"] - s["truth"]

print(f"\nPER SENSOR, all 8 operational pads x {len(fr)} consists = {len(s)} traces")
print(f"{'':<22}{'X/D':>26}{'prominence':>26}")
print(f"{'':<22}{'exact':>7}{'median err':>12}{'mean':>7}"
      f"{'exact':>9}{'median err':>12}{'mean':>7}")
for lab, g in (("all freight traces", s),):
    print(f"{lab:<22}{int((g['err_xd']==0).sum()):>7}{g['err_xd'].median():>12.0f}"
          f"{g['err_xd'].mean():>7.1f}"
          f"{int((g['err_prom']==0).sum()):>9}{g['err_prom'].median():>12.0f}"
          f"{g['err_prom'].mean():>7.1f}")

print(f"\nPER CONSIST, per-pad counts side by side")
for nm, g in s.groupby("name", sort=False):
    t = g["truth"].iloc[0]
    print(f"\n{nm[:30]:<32} truth {t:.0f} axles, v {g['v_kmh'].iloc[0]:.1f} km/h")
    print(f"   {'pad':>5}" + "".join(f"{p:>6}" for p in range(1, 9)))
    print(f"   {'X/D':>5}" + "".join(
        f"{int(g.loc[g['pad']==p,'xd'].iloc[0]):>6}" if (g['pad']==p).any() else f"{'-':>6}"
        for p in range(1, 9)))
    print(f"   {'prom':>5}" + "".join(
        f"{int(g.loc[g['pad']==p,'prom'].iloc[0]):>6}" if (g['pad']==p).any() else f"{'-':>6}"
        for p in range(1, 9)))

print(f"\n{L}\nHEAD-LEVEL OUTPUTS, FOUR NAMED REFERENCES\n{L}")
ref = []
for nm, g in s.groupby("name", sort=False):
    t = g["truth"].iloc[0]
    d = {"name": nm, "truth": t}
    for lab, sub in (("h1", g[g["head"] == 1]), ("h2", g[g["head"] == 2])):
        d[f"xd_{lab}"] = sub["xd"].median()
        d[f"prom_{lab}"] = sub["prom"].median()
    d["xd_med2"] = np.median([d["xd_h1"], d["xd_h2"]])
    d["xd_best2"] = max(d["xd_h1"], d["xd_h2"])
    d["xd_med8"] = g["xd"].median()
    d["prom_med8"] = g["prom"].median()
    ref.append(d)
R = pd.DataFrame(ref)
print(f"{'reference':<34}{'exact/4':>9}{'median err':>12}{'mean err':>10}")
for lab, col in (("X/D, head 1 median (PADs 1-4)", "xd_h1"),
                 ("X/D, head 2 median (PADs 5-8)", "xd_h2"),
                 ("X/D, median of the two heads", "xd_med2"),
                 ("X/D, better of the two heads", "xd_best2"),
                 ("X/D, median over all 8 sensors", "xd_med8"),
                 ("prominence, head 1 median", "prom_h1"),
                 ("prominence, median over all 8", "prom_med8")):
    e = R[col] - R["truth"]
    print(f"{lab:<34}{int((e==0).sum()):>9}{e.median():>12.1f}{e.mean():>10.2f}")
print(f"""
THE ANSWER. Both detectors see the same samples in the same window. If the
influence extent had merged the peaks irrecoverably, NO algorithm reading those
samples could recover them and both columns would fail together. They do not.

That makes the freight deficit ALGORITHMIC at this consist size and speed range,
and POC2 §7's premise - that only a narrower influence length can resolve freight
axles - too strong as stated. It does NOT make prominence a drop-in replacement:
X/D is the deterministic, explainable detector the SIL4 argument rests on, and
n = {len(fr)} consists is not a basis for changing a safety chain.""")

# ------------------------------------------ July table, references named
print(f"\n{L}\nJULY FREIGHT GAP — REFERENCE NAMED, 33-60 BIN RESTORED\n{L}")
r2 = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False).drop(
    columns=["set", "gated"])
aux["train_free"] = aux["max_h1_pm"] < CONFIG["detect"]["absolute_gate_pm"]
aux["freight"] = aux["n_gap_lt_2m5"].fillna(0) > 22
j = aux.merge(r2[["name", "set", "gated", "c_h1", "c_h2"]], on="name")
j = j[(~j["train_free"]) & j["n_axles"].notna() & j["gated"] & (j["set"] == "open")]
j["xd_best2"] = j[["c_h1", "c_h2"]].max(axis=1)
j["xd_med2"] = j[["c_h1", "c_h2"]].median(axis=1)
print(f"OPEN DAYS ONLY, n = {len(j)}. The diagnostic count is `axle_peaks` on the")
print(f"best-conditioned head (52_passage_times.py). Gaps against three references:")
print(f"\n{'reference':<34}{'freight median':>16}{'freight mean':>14}"
      f"{'passenger median':>18}{'passenger mean':>16}")
for lab, col in (("X/D, better of the two heads", "xd_best2"),
                 ("X/D, median of the two heads", "xd_med2"),
                 ("X/D, head 1 median", "c_h1")):
    gf = j.loc[j["freight"], "n_axles"] - j.loc[j["freight"], col]
    gp = j.loc[~j["freight"], "n_axles"] - j.loc[~j["freight"], col]
    print(f"{lab:<34}{gf.median():>16.0f}{gf.mean():>14.2f}"
          f"{gp.median():>18.0f}{gp.mean():>16.2f}")
j["gap"] = j["n_axles"] - j["xd_best2"]
j["bin"] = pd.cut(j["n_axles"], [0, 12, 24, 32, 60, 400],
                  labels=["<=12", "13-24", "25-32", "33-60", ">60"])
print(f"\nby consist size, reference = better of the two head medians "
      f"(ALL bins, none skipped):")
print(f"{'axles':<10}{'n':>5}{'median gap':>12}{'mean gap':>10}{'diag>X/D':>10}"
      f"{'diag<X/D':>10}")
for b, g in j.groupby("bin", observed=True):
    print(f"{str(b):<10}{len(g):>5}{g['gap'].median():>12.0f}{g['gap'].mean():>10.2f}"
          f"{int((g['gap']>0).sum()):>10}{int((g['gap']<0).sum()):>10}")

s.to_csv(OUT / "freight_prominence_vs_xd.csv", index=False)
R.to_csv(OUT / "freight_head_references.csv", index=False)
j.to_csv(OUT / "july_freight_referenced.csv", index=False)
print(f"\nwrote freight_prominence_vs_xd.csv, freight_head_references.csv, "
      f"july_freight_referenced.csv")

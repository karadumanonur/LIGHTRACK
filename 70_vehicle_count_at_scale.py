"""Item 3 — vehicle and bogie count on the full populations.

Vehicle count is the ONLY quantity in this project that validates without an
external reference. A 12-axle AM08 must decompose to 3 vehicles and 6 bogies;
the arithmetic of 2-axle bogies in 2-bogie vehicles is self-checking. So the
44/46 denominator from the ACAT work was the wrong one - that denominator existed
because AXLE accuracy needed ACAT, and vehicle count does not.

POPULATIONS
  POC1   all 76 passages with a gap vector.        14/05 - 02/06 2026
  July   train-containing PASSENGER records on the OPEN DAYS ONLY.
         17-27/07 2026, sealed days untouched and unspent.

FREIGHT IS NOT ATTEMPTED HERE. A-63 established the negative on POC1 freight -
separation ratio 1.05 against 1.28 for passenger, 3 of 8 long consists passing
internal checks - and that result stands. Re-running it on July would be a second
attempt at a question already answered, so freight records are excluded and
reported as NOT ATTEMPTED, not as a second failure.

THIS IS A CONSISTENCY MEASURE, NOT AN ACCURACY MEASURE. It says the decomposition
is internally coherent and matches the AM08's known layout. It does not say the
axle count underneath it was right - that is §3.1, and it is POC1-only.
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


def ci(k, n):
    if n == 0:
        return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1),
            1.0 if k >= n else beta.ppf(.975, k + 1, n - k))


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


def decompose(gaps: np.ndarray) -> dict:
    """Three-level split on the two largest jumps in the sorted log-gap sequence.

    Per-passage and data-driven, so it is a RATIO test: every gap is v*dt, and a
    speed-scale error (A-52) rescales the thresholds with the gaps. A fixed metre
    threshold would not survive that.
    """
    g = np.sort(np.asarray(gaps, float))
    g = g[g > 0]
    if len(g) < 5:
        return {}
    d = np.diff(np.log(g))
    idx = np.sort(np.argsort(d)[-2:])
    t1 = float(np.sqrt(g[idx[0]] * g[idx[0] + 1]))
    t2 = float(np.sqrt(g[idx[1]] * g[idx[1] + 1]))
    n1 = int((gaps < t1).sum())
    n2 = int(((gaps >= t1) & (gaps < t2)).sum())
    n3 = int((gaps >= t2).sum())
    return {"t_wc": t1, "t_ci": t2, "n_wheel": n1, "n_couple": n2,
            "n_intra": n3, "bogies": n1, "vehicles": n2 + 1}


def score(df: pd.DataFrame, gaps: pd.DataFrame, label: str,
          key=("name", "sub")) -> pd.DataFrame:
    rows = []
    for k, sub in gaps.groupby(list(key)):
        m = df
        for col, val in zip(key, k if isinstance(k, tuple) else (k,)):
            m = m[m[col] == val]
        if m.empty:
            continue
        d = decompose(sub["gap_m"].to_numpy())
        if not d:
            continue
        ax = int(m["n_axles"].iloc[0])
        d.update({"name": k[0] if isinstance(k, tuple) else k,
                  "n_axles": ax, "v_kmh": float(m["v_kmh"].iloc[0]),
                  "pop": label})
        # the four self-checks
        d["c_bogie"] = d["bogies"] == 2 * d["n_intra"]
        d["c_vehicle"] = d["n_intra"] == d["n_couple"] + 1
        d["c_axles4"] = ax == 4 * d["vehicles"]
        d["c_mult12"] = (ax % 12 == 0) == (d["vehicles"] % 3 == 0)
        d["all4"] = bool(d["c_bogie"] and d["c_vehicle"] and
                         d["c_axles4"] and d["c_mult12"])
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- POC1
p1 = pd.read_csv(OUT / "passage_times.csv")
g1 = pd.read_csv(OUT / "passage_gaps.csv")
V1 = score(p1, g1, "POC1")
V1["freight"] = V1["n_axles"] > 32

# ---------------------------------------------------------------- July
r = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False).drop(
    columns=["set", "gated"])
aux = aux.merge(r[["name", "set", "gated"]], on="name")
aux["train_free"] = aux["max_h1_pm"] < CONFIG["detect"]["absolute_gate_pm"]
aux["freight"] = aux["n_gap_lt_2m5"].fillna(0) > 22
gj = pd.read_csv(OUT / "july_aux_gaps.csv").drop(columns=["set"])
gj["sub"] = 0
aux["sub"] = 0

n_all_tc = int((~aux["train_free"]).sum())
op = aux[(aux["set"] == "open") & (~aux["train_free"]) & aux["n_axles"].notna()]
op_pass = op[~op["freight"]]
op_fr = op[op["freight"]]
VJ = score(op_pass, gj[gj["name"].isin(op_pass["name"])], "July open, passenger")

print(f"{L}\nPOPULATIONS\n{L}")
print(f"POC1, all passages with a gap vector      : {len(V1)}  "
      f"(of 76; {76-len(V1)} had fewer than 5 gaps)")
print(f"July, train-containing records, ALL days  : {n_all_tc}")
print(f"July, OPEN DAYS only, train-containing    : {len(op)}")
print(f"   passenger, decomposed                  : {len(VJ)}")
print(f"   freight, NOT ATTEMPTED (A-63 stands)   : {len(op_fr)}")
print(f"\nSEALED DAYS ARE NOT INCLUDED and remain unspent. The all-days figure")
print(f"({n_all_tc}) is shown only so the open-day subset can be located inside it.")

print(f"\n{L}\nSELF-CONSISTENCY — four checks, no external reference\n{L}")
print("""  C1  bogies   = 2 x vehicles          (2-bogie vehicles)
  C2  vehicles = couplings + 1         (n couplings join n+1 vehicles)
  C3  axles    = 4 x vehicles          (4-axle vehicles)
  C4  axles multiple of 12  <=>  vehicles multiple of 3   (AM08 3-car units)""")
print(f"\n{'population':<26}{'n':>5}{'C1':>8}{'C2':>8}{'C3':>8}{'C4':>8}"
      f"{'ALL FOUR':>11}{'95 % CI':>16}")
rows = []
for lab, V in (("POC1 passenger", V1[~V1["freight"]]),
               ("POC1 freight (A-63)", V1[V1["freight"]]),
               ("July open, passenger", VJ)):
    if not len(V):
        continue
    k = int(V["all4"].sum())
    lo, hi = ci(k, len(V))
    print(f"{lab:<26}{len(V):>5}{pct(int(V['c_bogie'].sum()), len(V)):>8}"
          f"{pct(int(V['c_vehicle'].sum()), len(V)):>8}"
          f"{pct(int(V['c_axles4'].sum()), len(V)):>8}"
          f"{pct(int(V['c_mult12'].sum()), len(V)):>8}"
          f"{pct(k, len(V)):>11}{f'{100*lo:.1f}-{100*hi:.1f}%':>16}")
    rows.append(dict(population=lab, n=len(V), all4=k, rate=k/len(V),
                     lo=lo, hi=hi))

print(f"\n{L}\nTHE AM08 CHECK — 12 AXLES MUST GIVE 3 VEHICLES AND 6 BOGIES\n{L}")
print(f"{'population':<26}{'n':>6}{'vehicles = 3':>14}{'bogies = 6':>13}"
      f"{'both':>9}{'95 % CI (both)':>18}")
for lab, V in (("POC1", V1[~V1["freight"]]), ("July open", VJ)):
    a = V[V["n_axles"] == 12]
    if not len(a):
        continue
    both = int(((a["vehicles"] == 3) & (a["bogies"] == 6)).sum())
    lo, hi = ci(both, len(a))
    print(f"{lab:<26}{len(a):>6}{pct(int((a['vehicles']==3).sum()), len(a)):>14}"
          f"{pct(int((a['bogies']==6).sum()), len(a)):>13}"
          f"{pct(both, len(a)):>9}{f'{100*lo:.1f}-{100*hi:.1f}%':>18}")
    rows.append(dict(population=f"{lab} 12-axle AM08", n=len(a), all4=both,
                     rate=both/len(a), lo=lo, hi=hi))

print(f"\n{L}\nVEHICLE-COUNT DISTRIBUTION BY AXLE COUNT (July open, passenger)\n{L}")
tab = pd.crosstab(VJ["n_axles"], VJ["vehicles"])
print(tab.head(12).to_string())
print(f"""
The diagonal is the expectation: 12 axles -> 3 vehicles, 24 -> 6, 36 -> 9. Every
off-diagonal entry is a decomposition that disagrees with its own axle count.""")

print(f"\n{L}\nTHRESHOLDS THE RULE CHOSE FOR ITSELF\n{L}")
for lab, V in (("POC1 passenger", V1[~V1["freight"]]),
               ("July open, passenger", VJ)):
    print(f"{lab:<24} wheelbase|coupling {V['t_wc'].median():.2f} m "
          f"[{V['t_wc'].quantile(.1):.2f}-{V['t_wc'].quantile(.9):.2f}]   "
          f"coupling|intra {V['t_ci'].median():.2f} m "
          f"[{V['t_ci'].quantile(.1):.2f}-{V['t_ci'].quantile(.9):.2f}]")
print("""
Chosen per passage from that passage's own sorted gap sequence, so the rule
carries no metre constant and rescales with any speed-scale error (A-52).""")

pd.concat([V1, VJ], ignore_index=True).to_csv(
    OUT / "vehicle_counts_at_scale.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "vehicle_consistency.csv", index=False)
print(f"\nwrote vehicle_counts_at_scale.csv, vehicle_consistency.csv")

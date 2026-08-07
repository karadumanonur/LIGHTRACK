"""Item 2 — what the 28 unmatched passages actually are; Item 4 — does the
freight result hold at n = 29?

ITEM 2. The Task-2 window was ONE-SIDED, [0, +400] s, justified by the geometry:
Le Campinaire is 1577 m in the PK-increasing direction, so a Charleroi-bound
train must reach the sensor first. That made any population on the negative side
structurally invisible, and 15 of the 28 unmatched passages have an ACAT event in
[-400, 0). Two hypotheses:
  (a) they are unrelated NEIGHBOURING trains, and the unmatched passages really
      are absent from ACAT;
  (b) the offset model has a sign or geometry error and those events ARE the
      partners.
Axle agreement cannot decide between them - 58 % of ACAT trains carry 12 axles,
so a 12<->12 coincidence is nearly free. The arbiter is the punctuality feed at
PTCAR 700, count-blind: if the negative-side ACAT event lines up with the SAME
train's timetabled Le Campinaire arrival it is (b); if it lines up with a
DIFFERENT train it is (a).

ITEM 4. POC2 §7 recommends a sensor-configuration change on the premise that
freight axle peaks are irrecoverably merged at the 2.2 m influence extent. The
POC1 ground truth showed the diagnostic chain resolving 36/104/128/130 exactly
where X/D missed by up to 10 - but at n = 4. The July set holds 25 freight
consists. There is no ground truth there, and none is needed for the question
asked: does the diagnostic chain systematically resolve MORE axles than X/D on
the same records, and is the gap specific to freight?

OPEN DAYS ONLY for the July part. The sealed days stay unspent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
pd.set_option("display.width", 250)
L = "=" * 116


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


# ==================================================================== ITEM 2
print(f"{L}\nITEM 2 — THE 28 UNMATCHED PASSAGES, CHARACTERISED\n{L}")
mo = pd.read_csv(OUT / "acat_match_outcomes.csv")
mo["train_local"] = pd.to_datetime(mo["train_local"]).dt.tz_localize(None)
acat = pd.read_csv(OUT / "acat_train_events.csv")
acat["t"] = pd.to_datetime(acat["t"])
AT = np.sort(acat["t"].to_numpy("datetime64[s]").astype("int64"))
AX = acat.sort_values("t")["axles"].to_numpy()

un = mo[mo["n_in_window"] == 0].copy()
ut = un["train_local"].to_numpy("datetime64[s]").astype("int64")
lo = np.searchsorted(AT, ut - 400)
hi = np.searchsorted(AT, ut)
un["n_neg"] = hi - lo
un["neg_i"] = np.where(hi - lo >= 1, hi - 1, -1)
un["neg_dt"] = np.where(un["neg_i"] >= 0, AT[np.clip(un["neg_i"], 0, None)] - ut, np.nan)
un["neg_axles"] = np.where(un["neg_i"] >= 0, AX[np.clip(un["neg_i"], 0, None)], np.nan)
print(f"unmatched passages: {len(un)}")
print(f"   with an ACAT event in [-400, 0) s : {int((un['n_neg'] >= 1).sum())}")
print(f"   of those, head1_first (forward)   : "
      f"{int(un.loc[un['n_neg'] >= 1, 'head1_first'].sum())}")
print(f"   axle agreement with that event    : "
      f"{int((un['neg_axles'] == un['n_axles']).sum())}"
      f" — uninformative: 58 % of ACAT trains are 12-axle")

# ---- the count-blind arbiter
cp = pd.read_csv(OUT / "punctuality_campinaire.csv", low_memory=False)
cp["arr"] = (pd.to_datetime(cp["REAL_DATE_ARR"], format="%d%b%Y")
             + pd.to_timedelta(cp["REAL_TIME_ARR"]))
cp = cp.dropna(subset=["arr"]).sort_values("arr").reset_index(drop=True)
CT = cp["arr"].to_numpy("datetime64[s]").astype("int64")
LAT = 53.0                      # ACAT logging latency after timetabled arrival

neg = un[un["n_neg"] >= 1].copy()
nt = AT[neg["neg_i"].to_numpy()]
# which timetabled train does the NEGATIVE-side ACAT event belong to?
j = np.searchsorted(CT, nt - LAT)
b_ev = np.full(len(nt), np.inf); i_ev = np.zeros(len(nt), int)
for off in (-1, 0, 1):
    k = np.clip(j + off, 0, len(CT) - 1)
    r = (nt - LAT - CT[k]).astype(float)
    m = np.abs(r) < np.abs(b_ev)
    b_ev = np.where(m, r, b_ev); i_ev = np.where(m, k, i_ev)
# which timetabled train does the SENSOR PASSAGE belong to? through-train model
# 1680 m at the record's own speed, the A-57 fit, no count involved
st = neg["train_local"].to_numpy("datetime64[s]").astype("int64")
pred = st + 1680.0 * 3.6 / neg["v_kmh"].to_numpy()
j2 = np.searchsorted(CT, pred)
b_pa = np.full(len(pred), np.inf); i_pa = np.zeros(len(pred), int)
for off in (-1, 0, 1):
    k = np.clip(j2 + off, 0, len(CT) - 1)
    r = (pred - CT[k]).astype(float)
    m = np.abs(r) < np.abs(b_pa)
    b_pa = np.where(m, r, b_pa); i_pa = np.where(m, k, i_pa)
neg["ev_train"] = cp["TRAIN_NO"].to_numpy()[i_ev]
neg["ev_off"] = b_ev
neg["pa_train"] = cp["TRAIN_NO"].to_numpy()[i_pa]
neg["pa_off"] = b_pa
neg["same_train"] = (neg["ev_train"] == neg["pa_train"]) & \
                    (np.abs(b_ev) <= 120) & (np.abs(b_pa) <= 240)

print(f"\nTHE ARBITER — does the negative-side ACAT event belong to the SAME "
      f"timetabled train as the sensor passage?")
print(f"{'record':<28}{'neg dt':>8}{'ACAT ax':>8}{'sens ax':>8}"
      f"{'event train':>13}{'passage train':>15}{'same':>7}")
for _, r in neg.sort_values("neg_dt").iterrows():
    print(f"{r['name'][:26]:<28}{r['neg_dt']:>8.0f}{r['neg_axles']:>8.0f}"
          f"{r['n_axles']:>8.0f}{str(r['ev_train']):>13}{str(r['pa_train']):>15}"
          f"{str(r['same_train']):>7}")
ns = int(neg["same_train"].sum())
print(f"\nsame train: {ns}/{len(neg)}   different train: {len(neg)-ns}/{len(neg)}")
print(f"""
VERDICT. A sign or geometry error in the offset model would make the negative-side
event the SAME train's ACAT report, on most of these. Unrelated neighbours would
make it a DIFFERENT train, on most of these. The count is above.""")
if ns <= len(neg) / 3:
    print("""=> HYPOTHESIS (a). The negative-side events belong to other trains.
   The one-sided window was correct, these passages really are absent from the
   ACAT log, and the reference is incomplete as A-58 states. The 36.8 % shortfall
   stands and the capture denominator must be bracketed for it (A-59).""")
else:
    print("""=> HYPOTHESIS (b). The offset model has a sign or geometry error and
   the Task-2 window excluded genuine partners. Task 2 and Task 3 must be re-run
   with a two-sided window before the accuracy figure is quoted.""")

# ==================================================================== ITEM 4
print(f"\n{L}\nITEM 4 — DOES THE FREIGHT RESULT HOLD AT n = 29?\n{L}")
r = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False).drop(
    columns=["set", "gated"])
aux["train_free"] = aux["max_h1_pm"] < CONFIG["detect"]["absolute_gate_pm"]
aux["freight"] = aux["n_gap_lt_2m5"].fillna(0) > 22
j = aux.merge(r[["name", "set", "gated", "c_h1", "c_h2"]], on="name")
j = j[(~j["train_free"]) & j["n_axles"].notna() & j["gated"]]
op = j[j["set"] == "open"].copy()
print(f"OPEN DAYS ONLY. gated, train-containing passages with a diagnostic count: "
      f"{len(op)}   (sealed set untouched)")
op["xd"] = op[["c_h1", "c_h2"]].max(axis=1)     # best of the two safety heads
op["gap"] = op["n_axles"] - op["xd"]
print(f"\ndiagnostic-chain count minus the better X/D head vote:")
print(f"{'population':<24}{'n':>5}{'median gap':>12}{'mean gap':>10}"
      f"{'diag > X/D':>12}{'diag < X/D':>12}{'equal':>8}")
rows = []
for lab, g in (("freight (>22 short gaps)", op[op["freight"]]),
               ("passenger", op[~op["freight"]])):
    gp = g["gap"]
    print(f"{lab:<24}{len(g):>5}{gp.median():>12.0f}{gp.mean():>10.2f}"
          f"{int((gp > 0).sum()):>12}{int((gp < 0).sum()):>12}"
          f"{int((gp == 0).sum()):>8}")
    rows.append(dict(pop=lab, n=len(g), median_gap=gp.median(),
                     mean_gap=gp.mean(), n_pos=int((gp > 0).sum()),
                     n_neg=int((gp < 0).sum()), n_eq=int((gp == 0).sum())))
fr = op[op["freight"]]
pa = op[~op["freight"]]
if len(fr) >= 6:
    w = wilcoxon(fr["gap"], alternative="greater") if (fr["gap"] != 0).any() else None
    print(f"\nWilcoxon signed-rank on the freight gaps (H1: diagnostic > X/D): "
          f"p = {w.pvalue:.4f}" if w else "\n(all freight gaps zero)")
print(f"""
THE COMPARISON THAT MATTERS is freight AGAINST passenger, not freight alone. If
the diagnostic chain simply counts higher everywhere, the gap is a detector
offset and says nothing about merged peaks. If the gap is near zero on passenger
stock and positive on freight, it is specific to the geometry A-41 is about.""")
print(f"\n{'consist size':<16}{'n':>5}{'median gap':>12}{'mean gap':>10}")
op["bin"] = pd.cut(op["n_axles"], [0, 12, 24, 32, 60, 400],
                   labels=["<=12", "13-24", "25-32", "33-60", ">60"])
for b, g in op.groupby("bin", observed=True):
    print(f"{str(b):<16}{len(g):>5}{g['gap'].median():>12.0f}{g['gap'].mean():>10.2f}")
print(f"\nthe {len(fr)} freight consists individually:")
print(f"{'record':<28}{'v':>7}{'diag':>7}{'c_h1':>7}{'c_h2':>7}{'gap':>7}")
for _, q in fr.sort_values("n_axles").iterrows():
    print(f"{q['name'][:26]:<28}{q['v_kmh']:>7.1f}{q['n_axles']:>7.0f}"
          f"{q['c_h1']:>7.0f}{q['c_h2']:>7.0f}{q['gap']:>+7.0f}")

n_tot = 4 + len(fr)
print(f"""
POC1 contributed 4 freight consists WITH ground truth; July contributes {len(fr)} on
open days WITHOUT it. Combined n = {n_tot}. July cannot say which chain is RIGHT — only
whether they DISAGREE, in which direction, and whether that is specific to
freight. Ground truth remains the POC1 four.""")
pd.DataFrame(rows).to_csv(OUT / "july_freight_chain_gap.csv", index=False)
op.to_csv(OUT / "july_freight_detail.csv", index=False)
neg.to_csv(OUT / "acat_unmatched_arbitration.csv", index=False)
print(f"\nwrote july_freight_chain_gap.csv, july_freight_detail.csv, "
      f"acat_unmatched_arbitration.csv")

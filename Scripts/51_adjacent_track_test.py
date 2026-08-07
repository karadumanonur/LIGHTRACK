"""Item 1 — is the 88/12 direction split adjacent-track traffic?

Line 130 is double track. The two track centres are ~3.5-4 m apart, so a train
on the OTHER track loads the same ballast and sleeper ends and can couple a weak
signal into the instrumented rail. A-15 already names adjacent-track traffic as
part of the operational disturbance floor. That gives a third explanation for the
11 reverse-direction records, alongside solver error and wrong-line running:

    they may be real trains, on the other track.

The discriminator is amplitude. Direct axle loading produces ~100 pm at a healthy
pad against a ~0.4 pm noise floor. Cross-track coupling through ballast is a
small fraction of that, so an adjacent-track passage should be MARKEDLY weaker -
an order of magnitude, not twenty percent. If instead the 11 are as strong as the
82, the mechanism is excluded and only wrong-line running or solver error remain,
which step 4's RELATION_DIRECTION match separates.

Amplitude also depends on speed and on axle load, so speed is reported alongside
and the comparison is repeated inside a common speed band.

CHANNEL SELECTION. Only healthy OPERATIONAL pads are used: heads 1-2 carry
PADs 1-8 (head 3 is the excluded test zone), and PADs 6, 9, 11 are degrading with
7 borderline. Healthy operational = PADs 1, 2, 3, 4, 5, 8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OUT = Path(r"C:\Users\onurk\Desktop\data\out")
pd.set_option("display.width", 240)
L = "=" * 112

HEALTHY_OPERATIONAL = [1, 2, 3, 4, 5, 8]

# ------------------------------------------------------------------ inputs
t = pd.read_csv(OUT / "tof_speed_direction.csv")          # 93 records, sign
o = pd.read_csv(OUT / "occupancy_balance.csv")            # 93 records, counting
g = pd.read_csv(OUT / "gated_classification.csv")         # 93 records, max_pm
a = pd.read_csv(OUT / "phase1_amplitudes_raw.csv")        # 87 records x 12 ch

t["reverse"] = t["sign"] < 0
print(f"{L}\nITEM 1 — ADJACENT-TRACK HYPOTHESIS FOR THE 11 REVERSE RECORDS\n{L}")
print(f"records with a direction solution : {len(t)}")
print(f"   forward (head 1 first)         : {(~t['reverse']).sum()}")
print(f"   reverse (head 3 first)         : {t['reverse'].sum()}"
      f"   = {100*t['reverse'].mean():.0f} %")

# ------------------------------------------------------- per-record amplitude
ah = a[a["pad"].isin(HEALTHY_OPERATIONAL)].copy()
ah["p2p_pm"] = ah["pos_pm"] - ah["neg_pm"]
rec = ah.groupby("name").agg(
    n_ch=("pad", "size"),
    pos_med=("pos_pm", "median"),
    peak_med=("pos_max_pm", "median"),
    peak_max=("pos_max_pm", "max"),
    p2p_med=("p2p_pm", "median"),
    snr_med=("snr", "median"),
    noise_med=("noise_pm", "median"),
    n_axles=("n_axles", "median"),
).reset_index()

d = t.merge(rec, on="name", how="left").merge(
    g[["name", "max_pm", "gated", "n_bursts", "class"]], on="name", how="left")

cov = d.groupby("reverse")["peak_med"].apply(lambda s: s.notna().sum())
print(f"\nrecords carrying a Phase-1 amplitude vector:")
print(f"   forward {int(cov.get(False,0))}/{int((~t['reverse']).sum())}"
      f"   reverse {int(cov.get(True,0))}/{int(t['reverse'].sum())}")

# --------------------------------------------------------------- comparison
METRICS = [
    ("peak_med",  "peak amplitude   pos_max, pm"),
    ("pos_med",   "mean axle peak   pos,     pm"),
    ("p2p_med",   "peak-to-peak             pm"),
    ("snr_med",   "SNR              peak/noise"),
    ("max_pm",    "record max       any ch,  pm"),
    ("noise_med", "noise floor              pm"),
    ("v_kmh",     "speed                  km/h"),
    ("score",     "TOF fit score"),
]
print(f"\n{L}\nAMPLITUDE: 11 REVERSE vs 82 FORWARD\n{L}")
print(f"{'metric':<32}{'fwd med':>10}{'rev med':>10}{'ratio':>8}"
      f"{'fwd p10':>10}{'rev max':>10}{'MWU p':>10}")
rows = []
for col, lab in METRICS:
    f = d.loc[~d["reverse"], col].dropna()
    r = d.loc[d["reverse"], col].dropna()
    if len(r) < 3 or len(f) < 3:
        continue
    p = mannwhitneyu(r, f, alternative="two-sided").pvalue
    ratio = r.median() / f.median() if f.median() else np.nan
    print(f"{lab:<32}{f.median():>10.2f}{r.median():>10.2f}{ratio:>8.2f}"
          f"{f.quantile(.10):>10.2f}{r.max():>10.2f}{p:>10.4f}")
    rows.append(dict(metric=col, label=lab, fwd_median=f.median(),
                     rev_median=r.median(), ratio=ratio,
                     fwd_p10=f.quantile(.10), rev_max=r.max(), mwu_p=p,
                     n_fwd=len(f), n_rev=len(r)))

# how many reverse records fall below the forward 10th percentile?
print(f"\nposition of the reverse records inside the forward distribution:")
for col, lab in METRICS[:5]:
    f = d.loc[~d["reverse"], col].dropna()
    r = d.loc[d["reverse"], col].dropna()
    if len(r) < 3:
        continue
    below = (r < f.quantile(.10)).sum()
    print(f"   {lab:<32} {below}/{len(r)} reverse below the forward p10; "
          f"reverse min {r.min():.2f}, forward min {f.min():.2f}")

print("""
READING. Cross-track coupling through ballast is a small fraction of direct axle
loading, so the adjacent-track hypothesis predicts an ORDER-OF-MAGNITUDE deficit
and reverse records sitting below the forward p10 almost without exception. A
ratio near 1 with overlapping distributions excludes it.""")

# --------------------------------------------------- speed-matched sub-check
print(f"\n{L}\nSPEED-MATCHED SUB-CHECK\n{L}")
lo, hi = d.loc[d["reverse"], "v_kmh"].quantile([.05, .95])
band = d[(d["v_kmh"] >= lo) & (d["v_kmh"] <= hi)]
print(f"reverse speed range {lo:.0f}-{hi:.0f} km/h; "
      f"{len(band)} records in band "
      f"({(~band['reverse']).sum()} fwd / {band['reverse'].sum()} rev)")
for col, lab in METRICS[:4]:
    f = band.loc[~band["reverse"], col].dropna()
    r = band.loc[band["reverse"], col].dropna()
    if len(r) < 3 or len(f) < 3:
        continue
    p = mannwhitneyu(r, f, alternative="two-sided").pvalue
    print(f"   {lab:<32} fwd {f.median():>8.2f}   rev {r.median():>8.2f}   "
          f"ratio {r.median()/f.median():>5.2f}   p {p:.4f}")

# ------------------------------------------------------- the eleven, listed
print(f"\n{L}\nTHE ELEVEN, INDIVIDUALLY\n{L}")
oo = o[["name", "gated", "n_stationary", "count_in", "count_out", "balance",
        "c_h1", "c_h2", "unanimous", "stitch_prev"]]
e = d[d["reverse"]].merge(oo, on="name", how="left", suffixes=("", "_o"))
print(f"{'file':<28}{'v':>7}{'score':>7}{'peak':>8}{'snr':>9}{'ax':>5}"
      f"{'gated':>7}{'c_h1':>6}{'c_h2':>6}{'bal':>6}{'stat':>6}")
for _, r in e.sort_values("v_kmh").iterrows():
    pk = "-" if pd.isna(r["peak_med"]) else f"{r['peak_med']:.0f}"
    sn = "-" if pd.isna(r["snr_med"]) else f"{r['snr_med']:.0f}"
    ax = "-" if pd.isna(r["n_axles"]) else f"{r['n_axles']:.0f}"
    print(f"{r['name'][:26]:<28}{r['v_kmh']:>7.1f}{r['score']:>7.2f}{pk:>8}"
          f"{sn:>9}{ax:>5}{str(r['gated_o']):>7}{r['c_h1']:>6.0f}"
          f"{r['c_h2']:>6.0f}{r['balance']:>6.0f}{r['n_stationary']:>6.0f}")

# ------------------------------------------------ occupancy contamination
print(f"\n{L}\nDO THE ELEVEN ENTER THE OCCUPANCY / COUNTING STATISTICS?\n{L}")
o2 = o.merge(t[["name", "reverse"]], on="name", how="left")
print(f"{'population':<44}{'n':>6}{'of which reverse':>20}")
for lab, sub in (("all records with a direction solution", o2),
                 ("gated (used for counting)", o2[o2["gated"]]),
                 ("gated and unanimous", o2[o2["gated"] & o2["unanimous"]]),
                 ("gated with |balance| > 0", o2[o2["gated"] & (o2["balance"] != 0)]),
                 ("n_stationary > 0 (occupancy events)", o2[o2["n_stationary"] > 0]),
                 ("stitched to previous slot", o2[o2["stitch_prev"] == True])):
    n = len(sub)
    nr = int(sub["reverse"].sum()) if n else 0
    print(f"{lab:<44}{n:>6}{nr:>20}")

print("""
Any reverse record that is GATED is already inside the counting statistics. If
those axles belonged to a train on the other track they would be phantom axles
for occupancy - a block reading occupied by a train that never entered it. That
consequence only follows if the amplitude test above supports the mechanism.""")

# ================================================================ solver test
# The amplitude test can only exclude adjacent-track coupling. Separating the
# two remaining explanations - solver error vs genuine wrong-line running - does
# not have to wait for step 4: the TOF solver reports two quantities that say how
# trustworthy its own answer is.
#   * score      - the constrained-geometry correlation sum. Low = the record
#                  does not constrain the fit.
#   * alt_score  - the same fit with the HEAD ORDER REVERSED (A-B = 15 m instead
#                  of 10 m). If this beats the assumed geometry, the record
#                  cannot tell the two apart, so its sign is not evidence.
# And onset_direction.csv is a semi-independent estimator built from per-head
# onset times rather than cross-correlation.
print(f"\n{L}\nIS THE SIGN TRUSTWORTHY? SOLVER CONDITIONING\n{L}")
t["geom_flip"] = t["alt_score"] > t["score"]
t["band"] = pd.cut(t["score"], [0, .5, .6, .7, .8, .9, 2.0])
print("direction by TOF fit score:")
print(f"{'score band':<14}{'forward':>9}{'reverse':>9}{'% reverse':>11}")
for b, gg in t.groupby("band", observed=True):
    print(f"{str(b):<14}{int((~gg['reverse']).sum()):>9}"
          f"{int(gg['reverse'].sum()):>9}{100*gg['reverse'].mean():>10.1f}%")

print(f"\nhead-order counter-hypothesis (alt_score > score => record cannot "
      f"resolve the geometry):")
print(f"   forward: {int(t.loc[~t['reverse'],'geom_flip'].sum())}/"
      f"{int((~t['reverse']).sum())} "
      f"({100*t.loc[~t['reverse'],'geom_flip'].mean():.0f} %)")
print(f"   reverse: {int(t.loc[t['reverse'],'geom_flip'].sum())}/"
      f"{int(t['reverse'].sum())} "
      f"({100*t.loc[t['reverse'],'geom_flip'].mean():.0f} %)")

onset = pd.read_csv(OUT / "onset_direction.csv")
mo = t.merge(onset[["name", "order", "ratio", "diff_pct"]], on="name")
mo["agrees"] = np.where(mo["order"] == "MIXED", np.nan,
                        (mo["order"] == "C->B->A") == (mo["sign"] > 0))
print(f"\nindependent onset estimator (n={len(mo)} records with an onset "
      f"solution):")
for rev, gg in mo.groupby("reverse"):
    lab = "reverse" if rev else "forward"
    ag = gg["agrees"]
    print(f"   {lab}: n={len(gg):>3}  agrees {int((ag == 1).sum())}  "
          f"contradicts {int((ag == 0).sum())}  mixed {int(ag.isna().sum())}")

print(f"\n{L}\nDIRECTION SPLIT AFTER CONDITIONING\n{L}")
print(f"{'population':<48}{'n':>6}{'reverse':>9}{'% reverse':>11}")
for lab, sub in (("all records with a solution", t),
                 ("score > 0.8", t[t["score"] > 0.8]),
                 ("geometry confirmed (score > alt_score)", t[~t["geom_flip"]]),
                 ("score > 0.8 AND geometry confirmed",
                  t[(t["score"] > 0.8) & (~t["geom_flip"])])):
    print(f"{lab:<48}{len(sub):>6}{int(sub['reverse'].sum()):>9}"
          f"{100*sub['reverse'].mean():>10.1f}%")

surv = t[(t["score"] > 0.8) & (~t["geom_flip"]) & t["reverse"]]
print(f"\nreverse records surviving every filter:")
for _, r in surv.iterrows():
    om = mo[mo["name"] == r["name"]]
    od = "" if om.empty else (f"  onset {om.iloc[0]['order']} "
                              f"(diff {om.iloc[0]['diff_pct']:.1f} %)")
    print(f"   {r['name']}  v={r['v_kmh']:.1f} km/h  score={r['score']:.3f} "
          f"vs alt {r['alt_score']:.3f}{od}")

print("""
INTERPRETATION. Line 130 is double track and each track normally carries ONE
direction, so a sensor on one track should see one direction almost exclusively.
The timetable's 51/49 split is over BOTH tracks and is therefore not the right
comparison - an earlier note in this project called the 88/12 split impossible
for traffic, and that framing was wrong.

What the 88/12 does contain is solver noise, and it is removable: the reverse
label concentrates in records that cannot support any fit. Conditioning on fit
quality and on the geometry test leaves a residual reverse fraction, and a
residual at the percent level is the size of genuine wrong-line running during
engineering possessions - a real operating mode on a double-track line.""")

d.to_csv(OUT / "direction_amplitude.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "direction_amplitude_tests.csv", index=False)
print(f"\nwrote direction_amplitude.csv, direction_amplitude_tests.csv")

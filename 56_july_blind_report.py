"""July blind pass — the full report, sealed and open side by side.

WHAT THIS MEASURES. Whether the frozen POC1 configuration's predictions hold on
609 records it has never seen, at 6.5x the sample size. The July punctuality file
is not published, so there is no timetable to match against and NOTHING here is
an accuracy measurement. Every figure is INTERNAL CONSISTENCY: agreement between
detectors, between heads, between days, and against POC1's own values.

SEALED AND OPEN ARE REPORTED SEPARATELY EVERYWHERE. Nothing is fitted, so they
should agree; where they do not, that is information about day-to-day variability
and it is stated rather than pooled away.

NO TUNING. Every threshold is either `config_frozen_poc1.toml` or a split read
off an empty interval in an observed distribution. Where the frozen pipeline
breaks, the break is reported and a fix is proposed, not applied.
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
POC1_HEALTHY = [1, 2, 3, 4, 5, 8, 10, 12]
POC1_DEGRADING = [6, 9, 11]
pd.set_option("display.width", 250)
L = "=" * 118


def ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1),
            1.0 if k >= n else beta.ppf(.975, k + 1, n - k))


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


# ------------------------------------------------------------------ inputs
r = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
r["local"] = pd.to_datetime(r["local"])
r["clean"] = ~(r["touch_start"] | r["touch_end"]) & (r["n_stationary"] == 0)
s = pd.read_csv(OUT / "blind_sensors.csv")
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False)
ag = pd.read_csv(OUT / "july_aux_gaps.csv")
aux = aux.merge(r[["name", "clean", "c_h1", "c_h2", "balance", "unanimous"]],
                on="name", how="left")

# POC1 comparators, recomputed on identical definitions rather than quoted
p1 = pd.read_csv(OUT / "occupancy_balance.csv")
p1["clean"] = ~(p1["touch_start"] | p1["touch_end"]) & (p1["n_stationary"] == 0)
p1c = pd.read_csv(OUT / "phase2_counts.csv")
p1g = pd.read_csv(OUT / "passage_times.csv")

SETS = [("SEALED", r["set"] == "SEALED"), ("open", r["set"] == "open"),
        ("both", r["name"].notna())]

print(f"{L}\nJULY BLIND PASS — INTERNAL-CONSISTENCY TEST OF THE FROZEN "
      f"CONFIGURATION\n{L}")
print("Not an accuracy measurement: no timetable exists for July.")
day = r.groupby(["date", "set"]).agg(n=("name", "size"), first=("local", "min"),
                                     last=("local", "max")).reset_index()
day["span_h"] = (day["last"] - day["first"]).dt.total_seconds() / 3600
print(f"\n{'date':<12}{'set':<9}{'records':>9}{'span h':>9}{'recorded h':>12}")
for _, q in day.iterrows():
    print(f"{str(q['date']):<12}{q['set']:<9}{q['n']:>9}{q['span_h']:>9.1f}"
          f"{q['n']*144.5/3600:>12.2f}")
print(f"{'TOTAL':<21}{len(r):>9}{'':>9}{len(r)*144.5/3600:>12.2f}")
print(f"\nPOC1 for scale: 93 records, 3.73 h recorded, sessions ~1 h.")

# ============================================================ HEADLINE TABLE
print(f"\n{L}\nHEADLINE — FROZEN PIPELINE vs POC1 BASELINE\n{L}")
rows = []


def add(q, poc1, getter, note=""):
    vals = {}
    for lab, m in SETS:
        k, n = getter(r[m])
        vals[lab] = (k, n)
    rows.append(dict(quantity=q, poc1=poc1, note=note, **{
        f"{lab}": vals[lab] for lab, _ in SETS}))


add("speed-gate pass rate", 81.7,
    lambda d: (int(d["gated"].sum()), len(d)))
add("direction, Head 1 first", 88.2,
    lambda d: (int(d["head1_first"].sum()), len(d)))
add("within-head unanimity (gated)", 78.9,
    lambda d: (int(d.loc[d["gated"], "unanimous"].sum()), int(d["gated"].sum())))
add("occupancy balanced, clean", 97.7,
    lambda d: (int((d.loc[d["clean"], "balance"] == 0).sum()),
               int(d["clean"].sum())))
add("phantom (in > out), clean", 1.1,
    lambda d: (int((d.loc[d["clean"], "balance"] > 0).sum()),
               int(d["clean"].sum())))
add("missed (in < out), clean", 1.1,
    lambda d: (int((d.loc[d["clean"], "balance"] < 0).sum()),
               int(d["clean"].sum())))

# X/D agreement. POC1's headline 91.7 % is over ALL 93 records x 8 operational
# pads (rectify_0.csv, 744 rows); the gated-only subset (phase2_counts.csv, 560
# rows) is 79.8 %. Both are reported against their own comparator - quoting the
# gated July figure against the all-records POC1 figure would be a definition
# mismatch masquerading as a result.
xd = s[s["agree"].notna()].merge(r[["name", "set", "gated"]], on="name")
for lab_pop, sub, poc1v in (("X/D agreement, all records", xd, 91.7),
                            ("X/D agreement, gated only", xd[xd["gated"]], 79.8)):
    e = {"quantity": lab_pop, "poc1": poc1v, "note": ""}
    for lab, _ in SETS:
        m = sub if lab == "both" else sub[sub["set"] == lab]
        e[lab] = (int(m["agree"].sum()), len(m))
    rows.append(e)

print(f"{'quantity':<32}{'POC1':>8}{'SEALED':>18}{'open':>18}{'both':>18}"
      f"{'95 % CI (both)':>20}{'':>4}")
for q in rows:
    k, n = q["both"]
    lo, hi = ci(k, n)
    dev = 100 * k / n - q["poc1"] if n else np.nan
    flag = "  <== " if abs(dev) >= 3.0 else ""
    line = f"{q['quantity']:<32}{q['poc1']:>7.1f}%"
    for lab, _ in SETS:
        kk, nn = q[lab]
        line += f"{pct(kk, nn) + f' ({kk}/{nn})':>18}"
    line += f"{f'{100*lo:.1f}-{100*hi:.1f}%':>20}{flag}{f'{dev:+.1f} pt' if flag else ''}"
    print(line)
print(f"\n'<==' marks a deviation of 3 percentage points or more from POC1.")

# sealed vs open disagreement, which is the point of splitting them
print(f"\n{'quantity':<32}{'SEALED':>10}{'open':>10}{'difference':>13}")
for q in rows:
    ks, ns = q["SEALED"]; ko, no = q["open"]
    if not ns or not no:
        continue
    d_ = 100 * ks / ns - 100 * ko / no
    print(f"{q['quantity']:<32}{pct(ks,ns):>10}{pct(ko,no):>10}{d_:>+12.1f} pt"
          + ("   <== material" if abs(d_) >= 3.0 else ""))

# ================================================== POPULATION DECOMPOSITION
# The deviations above are large. Before anything is said about the pipeline, the
# population has to be separated from the processing: this delivery contains
# record types POC1's hand-picked 93 could not contain. This is a decomposition,
# not an explanation - the frozen pipeline still produced those numbers on the
# data as delivered, and the top line of each column is the one that counts.
print(f"\n{L}\nDECOMPOSITION — IS IT THE PIPELINE OR THE POPULATION?\n{L}")
_mx = np.sort(aux["max_h1_pm"].to_numpy())
_lg = np.log10(np.maximum(_mx, 1e-3))
_h = len(_mx) // 2
_gi = int(np.argmax(np.diff(_lg[:_h + 1])))
_cut = float(np.sqrt(_mx[_gi] * _mx[_gi + 1]))
aux["train_free"] = aux["max_h1_pm"] < _cut
aux["freight_sig"] = aux["n_gap_lt_2m5"].fillna(0) > 22
rr = r.merge(aux[["name", "train_free", "freight_sig"]], on="name", how="left")
POPS = [
    ("clean (POC1 definition)", rr["clean"]),
    ("  ... minus train-free records", rr["clean"] & ~rr["train_free"].fillna(False)),
    ("  ... minus freight too", rr["clean"] & ~rr["train_free"].fillna(False)
     & ~rr["freight_sig"].fillna(False)),
    ("  ... and gated as well", rr["clean"] & ~rr["train_free"].fillna(False)
     & ~rr["freight_sig"].fillna(False) & rr["gated"]),
]
print(f"{'population':<34}{'n':>7}{'balanced':>11}{'phantom':>10}{'missed':>9}"
      f"{'unanimous':>12}")
for lab, m in POPS:
    a = rr[m]
    n = len(a)
    print(f"{lab:<34}{n:>7}{pct(int((a['balance']==0).sum()), n):>11}"
          f"{pct(int((a['balance']>0).sum()), n):>10}"
          f"{pct(int((a['balance']<0).sum()), n):>9}"
          f"{pct(int(a['unanimous'].sum()), n):>12}")
print(f"{'POC1 clean':<34}{87:>7}{'97.7%':>11}{'1.1%':>10}{'1.1%':>9}{'78.9%':>12}")
print("""
Read the FIRST row against POC1: that is the like-for-like comparison and it is
where the pipeline stands on this delivery. The rows below it say how much of the
gap is attributable to record types POC1 never saw. Whatever remains after the
last row is the pipeline itself.""")

# ==================================================== 1. TRAIN-FREE RECORDS
print(f"\n{L}\n1. TRAIN-FREE RECORDS AND THE OPERATIONAL DISTURBANCE FLOOR\n{L}")
print("""A-15 records the disturbance floor as BLOCKED: POC1's 93 records were
hand-picked to contain trains, so a record with no train in it never appeared.
This set spans 4.7-23.4 h per day, so overnight windows should exist.

The classification is read off the data, not asserted. `passage_window` always
returns a window - it thresholds at a fraction of its OWN maximum - so a record
of pure noise yields a window and a peak count. The discriminating quantity is
the absolute excursion.""")
mx = np.sort(aux["max_h1_pm"].to_numpy())
lg = np.log10(np.maximum(mx, 1e-3))
# search the LOWER HALF only: a train-free population sits below the median, and
# the largest gap in the full range would otherwise be found in the upper tail,
# between two unusually loud trains.
half = len(mx) // 2
gi = int(np.argmax(np.diff(lg[:half + 1])))
cut = float(np.sqrt(mx[gi] * mx[gi + 1]))
print(f"\nrecord peak |head-1| excursion, sorted: "
      f"min {mx[0]:.2f}, p5 {np.percentile(mx,5):.2f}, "
      f"median {np.median(mx):.1f}, max {mx[-1]:.0f} pm")
print(f"largest empty interval on a log scale: {mx[gi]:.2f} .. {mx[gi+1]:.2f} pm"
      f"  => split at {cut:.2f} pm (geometric midpoint)")
aux["train_free"] = aux["max_h1_pm"] < cut
print(f"\n{'set':<9}{'records':>9}{'train-free':>12}{'rate':>9}")
for lab, _ in SETS:
    a = aux if lab == "both" else aux[aux["set"] == lab]
    print(f"{lab:<9}{len(a):>9}{int(a['train_free'].sum()):>12}"
          f"{pct(int(a['train_free'].sum()), len(a)):>9}")

tf, tr = aux[aux["train_free"]], aux[~aux["train_free"]]
print(f"""
TWO DIFFERENT NOISE QUANTITIES, kept apart because A-15 turns on the difference:
  (a) OPERATIONAL DISTURBANCE FLOOR - a record with no train at all. Whatever is
      present is ambient: adjacent-track traffic, thermal, plant, wind.
  (b) QUIET INTER-TRAIN BASELINE - the part of a record outside the passage
      window, in a record that does contain a train.""")
# Only ONE quantity is measured on comparable footing in both populations: the
# MAD noise floor, taken over the whole trace in (a) and outside the passage
# window in (b). Peak excursion and the excursion counts are whole-trace, so in
# (b) they describe the TRAIN, not the baseline, and putting them in a shared
# column would compare a train against a noise floor.
a, b = tf["noise_all_pm"].median(), tr["noise_outside_pm"].median()
print(f"\n{'quantity':<40}{'(a) train-free':>18}{'(b) inter-train':>18}{'ratio':>9}")
print(f"{'noise floor (MAD), pm':<40}{a:>18.3f}{b:>18.3f}{a/b:>9.2f}")
print(f"""
THAT RATIO IS THE RESULT. A-15 recorded the disturbance floor as unmeasurable and
warned that the inter-train baseline might understate it. It does not: a record
with no train anywhere in it is within 3 % of the quiet stretch of a record that
has one. Every threshold in this project that was set against the inter-train
baseline was set against the right number.""")
print(f"\ncharacterising the disturbance floor itself, on the {len(tf)} "
      f"train-free records (one-sided — there is no (b) comparator for these):")
for lab, col in (("noise floor (MAD), pm", "noise_all_pm"),
                 ("peak |excursion|, pm", "max_h1_pm"),
                 ("p99.9 |excursion|, pm", "p999_h1_pm"),
                 ("samples > 10x own noise floor", "n_exc_10s"),
                 ("samples > 20x own noise floor", "n_exc_20s")):
    q = tf[col]
    print(f"   {lab:<36} median {q.median():>9.3f}   p90 {q.quantile(.9):>9.3f}"
          f"   max {q.max():>9.3f}")
print(f"""   For scale, a healthy pad's axle peak is ~100-200 pm (A-09), and the
   loudest excursion in any train-free record is {tf['max_h1_pm'].max():.1f} pm.
   The floor is ~{tf['max_h1_pm'].max()/150:.0%} of a passenger axle peak. Adjacent-track traffic, if
   it reaches this fibre at all, does not reach it above that.""")
if len(tf):
    print(f"\ntrain-free records by hour of day (local):")
    h = pd.to_datetime(r.set_index('name').loc[tf['name'], 'local'].values).hour
    vc = pd.Series(h).value_counts().sort_index()
    for hr in range(24):
        n = int(vc.get(hr, 0))
        if n:
            print(f"   {hr:02d}:00  {n:>3}  {'#'*min(n, 60)}")
    print(f"""
WHAT THE FROZEN PIPELINE DOES WITH THEM - this is the important part.""")
    print(f"   gated by the speed fit           : "
          f"{int(tf['gated'].sum())}/{len(tf)} ({pct(int(tf['gated'].sum()), len(tf))})")
    print(f"   'axles' detected, median         : {tf['n_peaks'].median():.0f}"
          f"   max {int(tf['n_peaks'].max())}")
    print(f"   passed the plausibility bracket  : "
          f"{int(((tf['n_peaks'] >= CONFIG['detect']['min_axles_for_passage']) & (tf['n_peaks'] <= CONFIG['detect']['max_axles_for_passage'])).sum())}/{len(tf)}")
    print(f"""
   The frozen pipeline has no train/no-train test. `passage_window` normalises to
   the record's own maximum, so an empty record produces a window and peaks in
   pure noise. Any record reaching the counting stage is counted.
   PROPOSED, NOT APPLIED: an absolute amplitude gate ahead of `passage_window`.""")

# ============================================================== 2. FREIGHT
print(f"\n{L}\n2. FREIGHT — A-41 ON A REAL SAMPLE\n{L}")
FCUT = 22           # empty interval 8..37 in the POC1 gap-count distribution
tr2 = aux[(~aux["train_free"]) & aux["n_axles"].notna()].copy()
tr2["freight"] = tr2["n_gap_lt_2m5"] > FCUT
print(f"labelling rule: > {FCUT} inter-axle gaps below 2.5 m — the midpoint of "
      f"the empty interval 8..37 derived from POC1 (A-51). Not re-derived here.")
print(f"\n{'set':<9}{'passages':>10}{'freight':>9}{'rate':>8}{'axles med':>11}"
      f"{'length med':>12}")
for lab, _ in SETS:
    a = tr2 if lab == "both" else tr2[tr2["set"] == lab]
    f = a[a["freight"]]
    print(f"{lab:<9}{len(a):>10}{len(f):>9}{pct(len(f), len(a)):>8}"
          f"{(f['n_axles'].median() if len(f) else np.nan):>11.0f}"
          f"{(f['length_m'].median() if len(f) else np.nan):>12.1f}")
print(f"\nPOC1 held 5 freight consists in 76 passages (6.6 %); "
      f"A-41's mechanism rests on 10 long consists in total.")
fr = tr2[tr2["freight"]]
if len(fr):
    gf = ag[ag["name"].isin(fr["name"])]["gap_m"].to_numpy()
    gc = ag[ag["name"].isin(tr2.loc[~tr2["freight"], "name"])]["gap_m"].to_numpy()
    sm = lambda x: float(np.median(x[x < 5])) if (x < 5).any() else np.nan
    print(f"\nshort-gap statistic, median of gaps below 5 m "
          f"(DIAGNOSTIC chain — see A-41's detector note):")
    print(f"{'population':<28}{'n gaps':>9}{'July':>9}{'POC1 (same chain)':>20}")
    print(f"{'freight / long consists':<28}{len(gf):>9}{sm(gf):>9.2f}{2.36:>20.2f}")
    print(f"{'passenger':<28}{len(gc):>9}{sm(gc):>9.2f}{3.04:>20.2f}")
    print(f"""
A-41's own figures (2.01 m / 2.90 m) come from the SAFETY chain's X-crossing
detector and are NOT comparable with the two above. Re-estimating them on this
sample needs a safety-chain pass, which is proposed, not run here.""")
    print(f"\ndistinct freight passages: {len(fr)}  "
          f"(POC1: 5 labelled, 10 long consists total)")
    print(f"axle-count distribution: "
          f"{sorted(fr['n_axles'].astype(int).tolist())[:24]}")

# ==================================================== 3. STOPPED / CRAWLING
print(f"\n{L}\n3. STOPPED OR CRAWLING TRAINS — FIRST REAL-DATA FIRINGS\n{L}")
print(f"POC1: 0 firings in 93 records. The detector was validated synthetically "
      f"only (A-36).")
print(f"\n{'set':<9}{'records':>9}{'n_stationary>0':>16}{'rate':>8}"
      f"{'events':>9}{'total s':>10}{'median s':>10}")
for lab, m in SETS:
    a = r[m]
    st = a[a["n_stationary"] > 0]
    print(f"{lab:<9}{len(a):>9}{len(st):>16}{pct(len(st), len(a)):>8}"
          f"{int(a['n_stationary'].sum()):>9}{a['stationary_s'].sum():>10.0f}"
          f"{(st['stationary_s'].median() if len(st) else np.nan):>10.1f}")
st = r[r["n_stationary"] > 0].merge(aux[["name", "train_free", "max_h1_pm",
                                         "n_peaks"]], on="name", how="left")
if len(st):
    print(f"\nOF THE FIRINGS, how many are in records with no train in them?")
    print(f"   train-free : {int(st['train_free'].sum())}/{len(st)}")
    print(f"   with train : {int((~st['train_free'].fillna(False)).sum())}/{len(st)}")
    print(f"""
A firing in a TRAIN-FREE record is a false positive, not a stopped train: the
occupancy detector runs on a 30 s baseline and a flat, empty trace has no
signature to distinguish from a stationary axle load. This is the same absence of
a train/no-train test as in section 1, surfacing in a different stage.""")
    print(f"\n{'record':<30}{'set':<9}{'events':>8}{'sec':>8}{'max pm':>9}"
          f"{'train-free':>12}")
    for _, q in st.sort_values("stationary_s", ascending=False).head(12).iterrows():
        print(f"{q['name'][:28]:<30}{q['set']:<9}{int(q['n_stationary']):>8}"
              f"{q['stationary_s']:>8.1f}{q['max_h1_pm']:>9.1f}"
              f"{str(q['train_free']):>12}")

# ==================================================== 4. SELF-DIAGNOSIS
print(f"\n{L}\n4. SELF-DIAGNOSIS — TWO BASELINES, NEVER MERGED\n{L}")
fa = pd.read_csv(OUT / "blind_selfdiag_falsealarm.csv")
co = pd.read_csv(OUT / "blind_selfdiag_continuity.csv")
print("""(a) FALSE-ALARM TEST, baseline = this set's own first day (2026-07-17).
    Valid regardless of what happened in the 45-day gap. This is the headline.
(b) CONTINUITY TEST, baseline = POC1 session 1 (2026-05-15).
    VALID ONLY IF NOTHING WAS RE-BONDED OR RE-INSTALLED IN THE GAP. That question
    is still unanswered by the operator, so (b) cannot be interpreted yet.""")
h_fa = fa[fa["pad"].isin(POC1_HEALTHY)]["flag_pct"]
print(f"\n(a) FALSE-ALARM, pooled over the 8 POC1-healthy pads: "
      f"{h_fa.mean():.2f} %   POC1 LOSO baseline 1.5 %")
print(f"{'pad':>5}{'head':>6}{'POC1':>12}{'flag %':>9}{'pair med':>10}"
      f"{'trend med':>11}{'first flag':>12}")
for _, q in fa.iterrows():
    mark = "  <==" if (q["pad"] in POC1_HEALTHY and q["flag_pct"] > 5) else ""
    print(f"{int(q['pad']):>5}{int(q['head']):>6}{q['poc1']:>12}"
          f"{q['flag_pct']:>9.2f}{q['pair_med']:>10.3f}{q['trend_med']:>11.3f}"
          f"{q['first_flag'] if pd.notna(q['first_flag']) else '-':>12}{mark}")
print(f"""
DEVIATION. Pooled over healthy pads the false-alarm rate is {h_fa.mean():.1f} % against POC1's
1.5 %. PAD 3 and PAD 4 carry most of it. Both are POC1-healthy and both flag from
the first passage, which is the signature of a baseline offset rather than a
degradation: a sensor that has genuinely drifted crosses the threshold LATER, not
at passage 1. Reported, not corrected.""")
print(f"\n(b) CONTINUITY, baseline = POC1 session 1")
print(f"{'pad':>5}{'head':>6}{'POC1':>12}{'flag %':>9}{'pair med':>10}"
      f"{'trend med':>11}")
for _, q in co.iterrows():
    print(f"{int(q['pad']):>5}{int(q['head']):>6}{q['poc1']:>12}"
          f"{q['flag_pct']:>9.2f}{q['pair_med']:>10.3f}{q['trend_med']:>11.3f}")
n_hi = int((co["flag_pct"] > 90).sum())
n_hi_h = int(((co["flag_pct"] > 90) & (co["pad"].isin(POC1_HEALTHY))).sum())
print(f"""
{n_hi} of 12 pads flag on essentially every July passage, and {n_hi_h} of those are
POC1-HEALTHY. A detector cannot be 'right' about {n_hi} simultaneous failures and
also useful. Two readings, and this data cannot separate them:
  * the installation changed in the gap (re-bonding, re-seating, replacement),
    in which case the POC1 baseline is simply void; or
  * the whole array shifted for a common reason (season, railhead condition,
    a different interrogator setting), which the pair rule is designed to reject
    but the POC1-referenced version cannot see because the reference moved too.
UNTIL THE OPERATOR ANSWERS WHAT HAPPENED IN THE GAP, (b) IS UNINTERPRETABLE.""")

# =============================================== 5. NEW FAILURE INSTANCES
print(f"\n{L}\n5. NEW FAILURE INSTANCES — ALL TWELVE PADS ACROSS THE JULY DAYS\n{L}")
print("""POC1 rests on three failure instances, and the 63 % miss-rate bound comes
straight from that. Any sensor degrading across these 11 days would be the single
most valuable thing this delivery contains. All twelve are checked against their
JULY DAY-1 value - not against POC1, which section 4(b) has just shown is not a
usable reference.""")
sd = s[s["diag"] == True].merge(r[["name", "date", "set", "gated"]], on="name")
sd = sd[sd["gated"]]
piv = sd.pivot_table(index="date", columns="pad", values="p2p_pm", aggfunc="median")
d1 = piv.iloc[0]
rel = piv.div(d1, axis=1)
print(f"\nper-pad median peak-to-peak, normalised to July day 1 "
      f"({piv.index[0]}), gated passages only")
print(f"{'date':<12}{'set':<8}" + "".join(f"{f'P{p}':>7}" for p in range(1, 13)))
dset = r.drop_duplicates("date").set_index("date")["set"]
for dt, q in rel.iterrows():
    print(f"{str(dt):<12}{str(dset.get(dt, '')):<8}"
          + "".join(f"{q[p]:>7.2f}" if p in q and pd.notna(q[p]) else f"{'-':>7}"
                    for p in range(1, 13)))
print(f"\n{'pad':>5}{'POC1':>12}{'day1 pm':>10}{'last pm':>10}{'ratio':>8}"
      f"{'min ratio':>11}{'slope /day':>12}  verdict")
new_fail = []
for p in range(1, 13):
    if p not in rel.columns:
        continue
    v = rel[p].dropna()
    if len(v) < 3:
        continue
    x = np.arange(len(v))
    slope = float(np.polyfit(x, v.to_numpy(), 1)[0])
    lab = "healthy" if p in POC1_HEALTHY else "DEGRADING"
    verdict = ("DECLINE" if (v.iloc[-1] < 0.70 and slope < -0.01)
               else "watch" if v.iloc[-1] < 0.85 else "stable")
    if verdict == "DECLINE":
        new_fail.append(p)
    print(f"{p:>5}{lab:>12}{piv[p].iloc[0]:>10.1f}{piv[p].iloc[-1]:>10.1f}"
          f"{v.iloc[-1]:>8.2f}{v.min():>11.2f}{slope:>+12.4f}  {verdict}")
print(f"""
'DECLINE' requires the last day below 70 % of day 1 AND a negative trend - the
70 % criterion POC1 used for durability (§1.8). Anything short of that is not a
failure instance and is not counted as one.

new failure instances found: {len(new_fail)}   {new_fail if new_fail else ''}""")

# =============================================== 6. MULTI-TRAIN SIGNATURE
print(f"\n{L}\n6. MULTI-TRAIN RECORDS — SIGNATURE ONLY, NO TIMETABLE\n{L}")
print("""POC1: 7 of 77 occupied windows held two timetabled trains, all in opposite
directions, and the pipeline emits one passage per record. Without a timetable
the count is not directly available, but two signatures are:
  (i) a sub-2.0 m gap inside an otherwise passenger-looking consist - the
      interleaving artefact seen on the two POC1 reverse records;
  (ii) an axle count that is not a multiple of 12, where class-0 stock is.""")
ps = tr2[~tr2["freight"]].copy()
ps["short_gap"] = ps["n_gap_lt_2m0"] > 0
ps["not_mult12"] = (ps["n_axles"] % 12) != 0
ps["either"] = ps["short_gap"] | ps["not_mult12"]
print(f"\n{'set':<9}{'passenger-like':>16}{'sub-2.0 m gap':>15}{'rate':>8}"
      f"{'not mult of 12':>16}{'rate':>8}{'either':>9}{'rate':>8}")
for lab, _ in SETS:
    a = ps if lab == "both" else ps[ps["set"] == lab]
    print(f"{lab:<9}{len(a):>16}{int(a['short_gap'].sum()):>15}"
          f"{pct(int(a['short_gap'].sum()), len(a)):>8}"
          f"{int(a['not_mult12'].sum()):>16}"
          f"{pct(int(a['not_mult12'].sum()), len(a)):>8}"
          f"{int(a['either'].sum()):>9}{pct(int(a['either'].sum()), len(a)):>8}")
p1p = p1g[p1g["n_gap_lt_2m5"] <= 8]
print(f"\nPOC1, same signatures on 76 passages: sub-2.0 m gap "
      f"{int((p1p['gap_min_m'] < 2.0).sum())}/{len(p1p)} "
      f"({pct(int((p1p['gap_min_m']<2.0).sum()), len(p1p))}), "
      f"not a multiple of 12 {int((p1p['n_axles'] % 12 != 0).sum())}/{len(p1p)} "
      f"({pct(int((p1p['n_axles']%12!=0).sum()), len(p1p))})")
print(f"""
NEITHER SIGNATURE IS SPECIFIC. A sub-2.0 m gap can be a split peak; a non-multiple
of 12 can be a locomotive, a short set, or a clipped passage. The rate is an UPPER
BOUND on multi-train records, not a count, and POC1's 9 % timetable-verified rate
is the only anchor available for what fraction of it is real.""")

aux.to_csv(OUT / "july_aux_scored.csv", index=False)
print(f"\nwrote july_aux_scored.csv")

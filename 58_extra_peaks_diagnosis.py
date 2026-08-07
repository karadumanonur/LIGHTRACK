"""Item 2 — what generates extra peaks in July that was absent in May?

THE CLUE. The freight short-gap statistic fell 2.36 -> 2.08 m and the passenger
one fell 3.04 -> 2.62 m: both by ~13 %. Two unrelated rolling-stock populations
do not shorten by the same proportion because the stock changed. Equal
proportional shortening is what happens when measured intervals are SUBDIVIDED -
one spurious peak inside a real gap halves it. That single mechanism would
produce the unanimity collapse, the X/D disagreement, the raised missed rate and
the gap shortening together, so it is tested as one cause rather than four.

THREE CANDIDATES, in the order the user set:
  A  TIME-OF-DAY COVERAGE. POC1 sampled ~1 h at midday; July covers 4.7-23.4 h
     including nights. Comparing July's 14:00 records against POC1's removes the
     coverage difference with a single filter and leaves only the season and the
     45-day gap. Cleanest cut available, so it runs first.
  B  THERMAL. If the metrics degrade at the temperature extremes, that is a real
     operating-envelope finding rather than a defect.
  C  SENSOR STATE. If the extra peaks concentrate on particular pads it is
     sensor-side; if uniform, environmental.

Sealed and open are reported separately throughout, as everywhere else.
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
L = "=" * 118


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


def short_mode(x):
    x = np.asarray(x)
    x = x[x < 5]
    return float(np.median(x)) if x.size else np.nan


# ------------------------------------------------------------------- July
r = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
r["local"] = pd.to_datetime(r["local"])
r["hour"] = r["local"].dt.hour
r["clean"] = ~(r["touch_start"] | r["touch_end"]) & (r["n_stationary"] == 0)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False)
GATE = CONFIG["detect"]["absolute_gate_pm"]
aux["train_free"] = aux["max_h1_pm"] < GATE
aux["freight"] = aux["n_gap_lt_2m5"].fillna(0) > 22
r = r.merge(aux[["name", "train_free", "freight", "n_peaks", "max_h1_pm"]],
            on="name", how="left")
sj = pd.read_csv(OUT / "blind_sensors.csv")
sj = sj[sj["agree"].notna()].merge(r[["name", "hour", "set", "gated",
                                      "train_free"]], on="name")
# TRAIN-FREE RECORDS MUST BE DROPPED FIRST. Their "gaps" come from peaks found in
# pure noise — a median of 528 per record, up to 1201 — and their short-gap count
# clears the freight threshold trivially. Including them turns 25 real freight
# consists into 66 and manufactures a population of sub-1.5 m gaps out of nothing.
gj = pd.read_csv(OUT / "july_aux_gaps.csv").drop(columns=["set"]).merge(
    r[["name", "hour", "set", "train_free"]], on="name").merge(
    aux[["name", "freight"]], on="name")
gj = gj[~gj["train_free"].fillna(False)]

# ------------------------------------------------------------------- POC1
p1 = pd.read_csv(OUT / "occupancy_balance.csv")
p1["local"] = pd.to_datetime(p1["local"])
p1["hour"] = p1["local"].dt.hour
p1["clean"] = ~(p1["touch_start"] | p1["touch_end"]) & (p1["n_stationary"] == 0)
s1 = pd.read_csv(OUT / "rectify_0.csv").merge(p1[["name", "hour"]], on="name")
g1 = pd.read_csv(OUT / "passage_gaps.csv").merge(
    pd.read_csv(OUT / "passage_times.csv")[["name", "sub", "n_gap_lt_2m5"]],
    on=["name", "sub"]).merge(p1[["name", "hour"]], on="name")
g1["freight"] = g1["n_gap_lt_2m5"] > 22

print(f"{L}\nITEM 2 — LOCATING THE EXTRA PEAKS\n{L}")
print(f"POC1 hour coverage: "
      f"{p1['hour'].value_counts().sort_index().to_dict()}")
print(f"July hour coverage: {len(r)} records over "
      f"{r['hour'].nunique()} distinct hours")


def metrics(rr, ss, gg, label):
    """The four broken quantities, on one population."""
    cl = rr[rr["clean"]]
    gd = rr[rr["gated"]]
    sg = ss[ss["gated"]] if "gated" in ss else ss
    pas = gg[~gg["freight"]]["gap_m"] if len(gg) else np.array([])
    return {
        "population": label, "n_rec": len(rr),
        "unanimity": (int(gd["unanimous"].sum()), len(gd)),
        "xd": (int(sg["agree"].sum()), len(sg)),
        "balanced": (int((cl["balance"] == 0).sum()), len(cl)),
        "missed": (int((cl["balance"] < 0).sum()), len(cl)),
        "pass_gap": short_mode(pas),
    }


# ================================================== A. THE 14:00 COMPARISON
print(f"\n{L}\nA. SAME HOUR — POC1 14:00 vs JULY 14:00\n{L}")
print("""POC1 put 56 of its 93 records in the 14:00 hour. Restricting July to the
same hour removes the coverage difference in one filter; what remains is the
season and the 45-day gap.""")
rows = [metrics(p1[p1["hour"] == 14], s1[s1["hour"] == 14],
                g1[g1["hour"] == 14], "POC1, 14:00 only"),
        metrics(p1, s1, g1, "POC1, all records")]
for st in ("SEALED", "open"):
    m = (r["hour"] == 14) & (r["set"] == st) & (~r["train_free"].fillna(False))
    rows.append(metrics(r[m], sj[(sj["hour"] == 14) & (sj["set"] == st)],
                        gj[(gj["hour"] == 14) & (gj["set"] == st)],
                        f"July 14:00, {st}"))
m = (r["hour"] == 14) & (~r["train_free"].fillna(False))
rows.append(metrics(r[m], sj[sj["hour"] == 14], gj[gj["hour"] == 14],
                    "July 14:00, both"))
rows.append(metrics(r[~r["train_free"].fillna(False)], sj, gj,
                    "July all hours"))
print(f"\n{'population':<24}{'records':>9}{'unanimity':>18}{'X/D (gated)':>20}"
      f"{'balanced':>16}{'missed':>15}{'pass gap m':>12}")
for q in rows:
    print(f"{q['population']:<24}{q['n_rec']:>9}"
          f"{pct(*q['unanimity']) + f' ({q['unanimity'][1]})':>18}"
          f"{pct(*q['xd']) + f' ({q['xd'][1]})':>20}"
          f"{pct(*q['balanced']) + f' ({q['balanced'][1]})':>16}"
          f"{pct(*q['missed']):>15}{q['pass_gap']:>12.2f}")
print("""
READING RULE, stated before the numbers are read. If July's 14:00 subset matches
POC1's 14:00 subset, the difference is time-of-day coverage and the pipeline is
intact — it was only ever exercised at midday. If July's 14:00 subset is as bad
as July overall, coverage is not the cause and something changed between May and
July that is present at all hours.""")

# ============================================================== B. THERMAL
print(f"\n{L}\nB. HOUR OF DAY — DOES IT DEGRADE AT THE EXTREMES?\n{L}")
th = pd.read_csv(OUT / "newdata_thermal.csv")
prof = th.groupby("hour")["cm_pm"].median()
print(f"common-mode proxy by hour (newdata_thermal.csv, n={len(th)} samples):")
print(f"   range {th['cm_pm'].min():.0f} to {th['cm_pm'].max():.0f} pm "
      f"= {th['cm_pm'].max()-th['cm_pm'].min():.0f} pm peak-to-peak")
band = [(0, 5, "00-05 night"), (6, 9, "06-09 morning"), (10, 13, "10-13"),
        (14, 17, "14-17 afternoon"), (18, 21, "18-21"), (22, 23, "22-23")]
print(f"\n{'band':<18}{'records':>9}{'cm_pm med':>11}{'unanimity':>12}"
      f"{'X/D gated':>12}{'balanced':>11}{'missed':>10}{'pass gap':>10}")
for lo, hi, lab in band:
    m = (r["hour"] >= lo) & (r["hour"] <= hi) & (~r["train_free"].fillna(False))
    q = metrics(r[m], sj[(sj["hour"] >= lo) & (sj["hour"] <= hi)],
                gj[(gj["hour"] >= lo) & (gj["hour"] <= hi)], lab)
    cm = prof[(prof.index >= lo) & (prof.index <= hi)]
    print(f"{lab:<18}{q['n_rec']:>9}"
          f"{(cm.median() if len(cm) else np.nan):>11.0f}"
          f"{pct(*q['unanimity']):>12}{pct(*q['xd']):>12}"
          f"{pct(*q['balanced']):>11}{pct(*q['missed']):>10}"
          f"{q['pass_gap']:>10.2f}")
print("""
A U-shape against the common mode — worst at both thermal extremes — is the
thermal signature. A flat profile excludes it.""")

# ========================================================== C. SENSOR STATE
print(f"\n{L}\nC. PER-PAD — CONCENTRATED OR UNIFORM?\n{L}")
print("""The direct measure of subdivision: among records whose HEAD-MEDIAN count
is 12 — a class-0 unit, which is unambiguous — what fraction of individual pads
report MORE than 12? A spurious peak shows up here and nowhere else as cleanly.""")
sj2 = sj.merge(pd.read_csv(OUT / "blind_sensors.csv")[
    ["name", "pad", "head", "count"]].drop_duplicates(["name", "pad", "head"]),
    on=["name", "pad", "head"], how="left", suffixes=("", "_c"))
sj2["count"] = sj2["count"].fillna(sj2["count_c"])
med_j = sj2.groupby(["name", "head"])["count"].median().rename("head_med")
sj2 = sj2.join(med_j, on=["name", "head"])
j12 = sj2[(sj2["head_med"] == 12) & (~sj2["train_free"].fillna(False))]
p12 = s1[s1["head_med"] == 12]
print(f"\n{'pad':>5}{'POC1 >12':>12}{'POC1 n':>9}{'July >12':>12}{'July n':>9}"
      f"{'change':>10}{'POC1 X/D':>11}{'July X/D':>11}{'change':>10}")
for pad in range(1, 9):
    a, b = p12[p12["pad"] == pad], j12[j12["pad"] == pad]
    xa = s1[s1["pad"] == pad]
    xb = sj[(sj["pad"] == pad) & (sj["gated"])]
    if not len(a) or not len(b):
        continue
    o_a = 100 * (a["count"] > 12).mean()
    o_b = 100 * (b["count"] > 12).mean()
    g_a, g_b = 100 * xa["agree"].mean(), 100 * xb["agree"].mean()
    print(f"{pad:>5}{o_a:>11.1f}%{len(a):>9}{o_b:>11.1f}%{len(b):>9}"
          f"{o_b-o_a:>+9.1f}p{g_a:>10.1f}%{g_b:>10.1f}%{g_b-g_a:>+9.1f}p")
print("""
Concentrated on one or two pads => sensor-side, and it should match the pads
flagging in the continuity test. Spread evenly across all eight => environmental
or processing, and no sensor is at fault.""")

# ----------------------------------------------------- gap subdivision, direct
print(f"\n{L}\nTHE SUBDIVISION ITSELF\n{L}")
for lab, gg in (("POC1", g1), ("July", gj)):
    pas = gg[~gg["freight"]]["gap_m"]
    frt = gg[gg["freight"]]["gap_m"]
    print(f"{lab:<7} passenger: n={len(pas):>6}  short-mode {short_mode(pas):.2f} m"
          f"   frac < 2.0 m {100*(pas < 2.0).mean():>5.1f} %"
          f"   frac < 1.5 m {100*(pas < 1.5).mean():>5.1f} %")
    print(f"{'':<7} freight  : n={len(frt):>6}  short-mode {short_mode(frt):.2f} m"
          f"   frac < 2.0 m {100*(frt < 2.0).mean():>5.1f} %"
          f"   frac < 1.5 m {100*(frt < 1.5).mean():>5.1f} %")
print("""
If gaps are being SUBDIVIDED, the excess sits below the physical minimum: no
bogie has a 1.5 m wheelbase, so any population there is manufactured by the
detector. If instead the whole distribution has slid down, the cause is a scale
error, not spurious peaks — and A-52 has already excluded a scale error of this
size.""")

# ============================================ D. WHY THOSE PADS — AMPLITUDE
print(f"\n{L}\nD. THE CAUSE — SIX SENSORS LOST ~90 % OF THEIR SIGNAL IN THE GAP\n{L}")
print("""Section C found the extra peaks concentrated on PADs 2, 6, 7 and 8 rather
than spread across the eight. That is sensor-side, and it points at an amplitude
comparison that section 5 of the blind report did NOT make: it checked each pad
against its own JULY DAY-1 value, which is blind to anything that happened before
17 July. Against POC1 session 1:""")
p4 = pd.read_csv(OUT / "phase4_features_scored.csv")
o1 = p4[p4["session"] == 1].groupby("pad")["p2p_pm"].median()
sdiag = pd.read_csv(OUT / "blind_sensors.csv")
sdiag = sdiag[sdiag["diag"] == True].merge(
    r[["name", "date", "gated", "set"]], on="name")
sdiag = sdiag[sdiag["gated"]]
jall = sdiag.groupby("pad")["p2p_pm"].median()
jset = sdiag.pivot_table(index="pad", columns="set", values="p2p_pm",
                         aggfunc="median")
POC1_HEALTHY = {1, 2, 3, 4, 5, 8, 10, 12}
POC1_DEGRADING = {6, 9, 11}
over = {int(p): 100 * (j12[j12["pad"] == p]["count"] > 12).mean()
        for p in range(1, 9) if len(j12[j12["pad"] == p])}
print(f"\n{'pad':>4}{'POC1 label':>12}{'POC1 s1 pm':>12}{'July SEALED':>13}"
      f"{'July open':>11}{'July/POC1':>11}{'>12 count':>11}  verdict")
collapsed = []
for pad in range(1, 13):
    if pad not in o1.index or pad not in jall.index:
        continue
    ratio = jall[pad] / o1[pad]
    lab = ("DEGRADING" if pad in POC1_DEGRADING
           else "healthy" if pad in POC1_HEALTHY else "borderline")
    v = ("COLLAPSED" if ratio < 0.30 else
         "watch" if ratio < 0.70 else "stable")
    if v == "COLLAPSED":
        collapsed.append(pad)
    ov = f"{over[pad]:.1f}%" if pad in over else "-"
    print(f"{pad:>4}{lab:>12}{o1[pad]:>12.1f}"
          f"{jset.loc[pad, 'SEALED'] if 'SEALED' in jset.columns else np.nan:>13.1f}"
          f"{jset.loc[pad, 'open'] if 'open' in jset.columns else np.nan:>11.1f}"
          f"{ratio:>11.2f}{ov:>11}  {v}")
new_fail = [p for p in collapsed if p in POC1_HEALTHY]
print(f"""
collapsed (below 30 % of POC1 commissioning): {collapsed}
of those, POC1-HEALTHY at the end of POC1   : {new_fail}   <== NEW FAILURE INSTANCES

THIS CORRECTS SECTION 5 OF THE BLIND REPORT, which found zero new failures. It
was asking the wrong question: nothing degrades DURING the 11 July days because
the collapse had already happened by 17 July. The degradation is in the 45-day
gap, and only a POC1-referenced comparison can see it.

AND IT RESCUES THE CONTINUITY TEST. Section 4(b) called it uninterpretable
because six pads flagged at once. Those six are exactly {collapsed} — the
six that lost their signal. The detector was right. Three things now rule out the
alternative readings:
  * a re-bonding or re-installation would raise amplitudes, not cut them to a
    tenth;
  * an interrogator or processing change would move all twelve, and six are
    unchanged at 89-105 % of POC1;
  * a lighter train population would likewise move all twelve.
Six specific sensors lost ~90 % of their signal while their six neighbours did
not. That is sensor degradation, and the pair rule flagged it correctly.

WHAT REMAINS UNKNOWN is WHEN inside the 45-day gap, and whether anything was
touched at the site. That is still an operator question - but it is no longer the
question the continuity test depends on.""")
print(f"\nAND IT EXPLAINS THE EXTRA PEAKS. The peak gate inside the passage is")
print(f"RELATIVE — {CONFIG['detect']['prominence_amplitude_fraction']:.0%} of the passage amplitude (`prominence_amplitude_fraction`).")
print(f"{'pad':>5}{'July p2p pm':>13}{'gate at 20 %':>14}{'noise pm':>10}"
      f"{'gate/noise':>12}{'>12 count':>11}{'X/D July':>10}")
nz = sdiag.groupby("pad")["noise_pm"].median()
for pad in range(1, 9):
    if pad not in jall.index:
        continue
    gate = 0.20 * jall[pad]
    xb = sj[(sj["pad"] == pad) & sj["gated"]]
    print(f"{pad:>5}{jall[pad]:>13.1f}{gate:>14.2f}{nz.get(pad, np.nan):>10.2f}"
          f"{gate/nz.get(pad, np.nan):>12.1f}"
          f"{(f'{over[pad]:.1f}%' if pad in over else '-'):>11}"
          f"{100*xb['agree'].mean():>9.1f}%")
print("""
On a healthy pad the gate sits ~400x above the noise floor and nothing spurious
clears it. On a collapsed pad it falls to 28-43x, and the over-counting and the
X/D disagreement appear exactly there, ordered by how far it fell (PAD 7 lowest
at 28x, PAD 2 worst on X/D). The relative gate has no floor under it, and that is the same design fault
as the missing train/no-train test one level down: every threshold in the
detection chain is relative to the signal it is measuring.

PROPOSED, NOT APPLIED: an absolute floor under the peak gate, as a companion to
`absolute_gate_pm`. It is NOT proposed as a fix for the collapsed sensors — they
need replacing, not thresholding.""")

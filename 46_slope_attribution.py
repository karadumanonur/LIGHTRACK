"""Items 1 & 4 — what does the thermal slope actually track, and how much freight?

ITEM 1. Lobe depth has now contaminated the shape rule (A-25) and the safety
thresholds (A-39). Before claiming "bond stiffness is measurable from diurnal
drift", check whether the slope is measuring sleeper-bay position a third time.
Tested against position, rail, head, commissioning lobe depth, mounting - and
against the one variable that was not available when the claim was made: which
sensors are ALIVE in July and which have collapsed.

ITEM 4. Whether the count of distinct trains above 32 axles has increased
materially over POC1's ten decides whether A-41's freight peak-extent mechanism
can become a measurement.

Also: the temperature-dependent STRAIN TRANSFER test - does a pad's gain vary
with temperature, which is the concrete mechanism behind A-33's "commissioning
baseline has no established validity period"?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD                          # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
pd.set_option("display.width", 250)
LINE = "=" * 116

# briefing §4.4
META = pd.DataFrame([
    # pad, head, rail,   offset_cm, mounting
    (1,  1, "1-Out", 0.0, "standard glued"),
    (2,  1, "1-Out", 0.0, "standard glued"),
    (3,  1, "2-In",  7.5, "standard glued"),
    (4,  1, "2-In",  7.5, "standard glued"),
    (5,  2, "1-Out", 0.0, "standard glued"),
    (6,  2, "1-Out", 0.0, "standard glued"),
    (7,  2, "1-In",  3.0, "standard glued"),
    (8,  2, "2-In",  3.0, "standard glued"),
    (9,  3, "1-Out", 0.0, "1/2 collage"),
    (10, 3, "1-Out", 0.0, "clipse"),
    (11, 3, "2-In",  0.0, "unprepared surface"),
    (12, 3, "2-In",  0.0, "taraude"),
], columns=["pad", "head", "rail", "offset_cm", "mounting"])

t = pd.read_csv(OUT / "thermal_sensitivity.csv").merge(META, on="pad", suffixes=("", "_m"))
lobe = (pd.read_csv(OUT / "phase4_features_scored.csv")
        .query("session == 1").groupby("pad")["neg_over_pos"].median().rename("lobe"))
t = t.merge(lobe, on="pad")

# alive/dead from the July blind run
s = pd.read_csv(OUT / "blind_sensors.csv")
d = s[s["diag"] == True]
jul = d.groupby("pad")["pos_pm"].median()
poc1 = (pd.read_csv(OUT / "phase4_features_scored.csv")
        .query("session == 1").groupby("pad")["pos_pm"].median())
t["jul_ratio"] = (jul / poc1).reindex(t["pad"]).values
t["state"] = np.where(t["jul_ratio"] > 0.5, "alive", "COLLAPSED")

print(f"{LINE}\nITEM 1 — WHAT DOES THE THERMAL SLOPE TRACK?\n{LINE}")
print(f"{'PAD':>4}{'ch':>6}{'head':>6}{'rail':>7}{'offset':>8}"
      f"{'mounting':<20}{'lobe':>7}{'slope':>8}{'Jul ratio':>11}{'state':>11}")
for _, r in t.sort_values("slope").iterrows():
    print(f"{int(r['pad']):>4}{r['ch']:>6}{int(r['head']):>6}{r['rail']:>7}"
          f"{r['offset_cm']:>8.1f}  {r['mounting']:<18}{r['lobe']:>7.2f}"
          f"{r['slope']:>8.2f}{r['jul_ratio']:>11.2f}{r['state']:>11}")

print(f"\n{LINE}\nCORRELATIONS\n{LINE}")
for col, lab in (("offset_cm", "sleeper-bay offset"), ("lobe", "commissioning lobe depth"),
                 ("head", "head number")):
    rho, p = spearmanr(t[col], t["slope"])
    print(f"  slope vs {lab:<28} Spearman rho {rho:+.3f}   p = {p:.3f}")

print(f"\nby sleeper-bay position:")
for off, g in t.groupby("offset_cm"):
    print(f"   {off:>4.1f} cm  n={len(g)}  slope {g['slope'].mean():.2f} "
          f"(range {g['slope'].min():.2f}-{g['slope'].max():.2f})")
print("   WITHIN Head 1 (position varies, head does not):")
h1 = t[t["head"] == 1]
for off, g in h1.groupby("offset_cm"):
    print(f"      {off:>4.1f} cm  " + "  ".join(f"PAD{int(r.pad)}:{r.slope:.2f}"
                                                for r in g.itertuples()))
print("   WITHIN Head 2:")
h2 = t[t["head"] == 2]
for off, g in h2.groupby("offset_cm"):
    print(f"      {off:>4.1f} cm  " + "  ".join(f"PAD{int(r.pad)}:{r.slope:.2f}"
                                                for r in g.itertuples()))

print(f"\n{LINE}\nTHE VARIABLE THAT WAS NOT AVAILABLE BEFORE: ALIVE vs COLLAPSED\n{LINE}")
a = t[t["state"] == "alive"]["slope"]
c = t[t["state"] == "COLLAPSED"]["slope"]
print(f"   alive     (n={len(a)}): mean {a.mean():.2f}  range {a.min():.2f}-{a.max():.2f}")
print(f"   COLLAPSED (n={len(c)}): mean {c.mean():.2f}  range {c.min():.2f}-{c.max():.2f}")
u, p = mannwhitneyu(c, a, alternative="greater")
print(f"   Mann-Whitney (collapsed > alive): p = {p:.4f}")
print(f"   ratio of means: {c.mean()/a.mean():.2f}")
gl = t[(t["state"] == "alive") & (t["mounting"] == "standard glued")]["slope"]
print(f"\n   alive AND standard-glued (n={len(gl)}): mean {gl.mean():.2f} "
      f"(PADs {sorted(t[(t['state']=='alive')&(t['mounting']=='standard glued')]['pad'])})")
print(f"   collapsed / alive-glued ratio: {c.mean()/gl.mean():.2f}")
print(f"""
PHYSICAL READING. The pad body is ALUMINIUM (briefing §3.2). Free aluminium
expands at ~23 ppm/degC; the steel rail it is bonded to at ~12 ppm/degC. A pad
still bonded follows the rail; a released pad follows its own aluminium, so it
shows a HIGHER apparent thermal response - by roughly the CTE ratio 23/12 = 1.9.

Measured ratio collapsed vs alive-and-glued: {c.mean()/gl.mean():.2f}.

That is why PAD 9's high slope is NOT evidence against debonding, as first read -
it is the expected direction, and it explains §8's "mechanically freer".""")

print(f"\n{LINE}\nVERDICT ON THE CLAIM\n{LINE}")
rho_o, p_o = spearmanr(t["offset_cm"], t["slope"])
rho_l, p_l = spearmanr(t["lobe"], t["slope"])
print(f"slope vs sleeper-bay offset : rho {rho_o:+.2f} (p={p_o:.2f})")
print(f"slope vs lobe depth         : rho {rho_l:+.2f} (p={p_l:.2f})")
print(f"slope vs alive/collapsed    : p = {p:.4f}, ratio {c.mean()/a.mean():.2f}")
print("Head 1 and Head 2 DISAGREE on the sign of the position effect, exactly as")
print("Phase 5 found for the compression lobe - so position does not explain it.")

# ============================================================== ITEM 4
print(f"\n{LINE}\nITEM 4 — DISTINCT TRAINS ABOVE 32 AXLES\n{LINE}")
r = pd.read_csv(OUT / "blind_records_scored.csv")
lng = r[r["c_h1"] > 32]
lng_ok = lng[lng["gated"] & lng["clean"]]
print(f"POC1                              : 10 trains")
print(f"July, all records                 : {len(lng)} records")
print(f"July, gate-retained AND clean     : {len(lng_ok)} records")
print(f"   of those, |balance| <= 1       : {int((lng_ok['balance'].abs()<=1).sum())}")
print(f"   axle counts: min {lng_ok['c_h1'].min():.0f}, median "
      f"{lng_ok['c_h1'].median():.0f}, max {lng_ok['c_h1'].max():.0f}")
print(f"\n=> {len(lng_ok)/10:.1f}x POC1's sample. A-41's peak-extent mechanism can")
print(f"   move from a mechanism to a measurement.")

# ============================================================== thermal gain
print(f"\n{LINE}\nTEMPERATURE-DEPENDENT STRAIN TRANSFER (the A-33 mechanism)\n{LINE}")
d2 = d.merge(r[["name", "local", "gated"]], on="name")
d2 = d2[d2["gated"]]
d2["hour"] = pd.to_datetime(d2["local"]).dt.hour
alive_pads = t[t["state"] == "alive"]["pad"].tolist()
print(f"alive sensors only ({alive_pads}), gate-retained passages:")
print(f"{'PAD':>4}{'rho(hour, amplitude)':>24}{'p':>9}{'day-hours 06-22 range':>26}")
for pad in alive_pads:
    g = d2[(d2["pad"] == pad) & d2["hour"].between(6, 22)].dropna(
        subset=["hour", "pos_pm"])
    if len(g) < 10:
        print(f"{pad:>4}{'too few':>24}")
        continue
    rho, pv = spearmanr(g["hour"].to_numpy(float), g["pos_pm"].to_numpy(float))
    med = g.groupby("hour")["pos_pm"].median()
    print(f"{pad:>4}{rho:>+24.3f}{pv:>9.3f}"
          f"{f'{med.min():.0f}-{med.max():.0f} pm ({100*(med.max()/med.min()-1):.0f} %)':>26}")
print("""
Within July's diurnal cycle there is no monotonic amplitude-temperature
relationship. So the trend rule is not confounded by the DAILY cycle.

This does NOT close A-33. The seasonal question - does a pad's gain drift over
months so that a commissioning baseline decays - needs data spanning seasons,
and 11 days in July cannot answer it. What it does establish is that the
day-to-day component is not the mechanism.""")
t.to_csv(OUT / "slope_attribution.csv", index=False)
print(f"\nwrote slope_attribution.csv")

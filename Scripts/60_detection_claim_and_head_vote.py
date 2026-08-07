"""Items 1 and 4 — was the collapse DETECTED, and does the head vote still work?

ITEM 1. Five failure instances only improve the miss-rate bound if all five were
DETECTED. PAD 2 and PAD 8 were found here by an amplitude comparison against POC1
session 1 — that is an analyst noticing, not a detector firing. The question is
whether the frozen monitoring rules would have flagged them. It is answerable:
the continuity test already runs those rules with a POC1 baseline.

ITEM 4. The absolute floor under the peak gate is NOT applied. The right fix is
for the self-diagnosis chain to exclude collapsed sensors from the head vote, so
the question is whether it does. Head 2 is the test case: three of its four pads
have collapsed, leaving PAD 5 as the only healthy sensor in that head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
COLLAPSED = {2, 6, 7, 8, 9, 11}
pd.set_option("display.width", 250)
L = "=" * 116


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


# ==================================================================== ITEM 1
print(f"{L}\nITEM 1 — WERE PAD 2 AND PAD 8 DETECTED, OR ONLY NOTICED?\n{L}")
fa = pd.read_csv(OUT / "blind_selfdiag_falsealarm.csv").set_index("pad")
co = pd.read_csv(OUT / "blind_selfdiag_continuity.csv").set_index("pad")
print(f"{'pad':>5}{'POC1 label':>12}{'July/POC1':>11}"
      f"{'(a) July-baseline flag %':>26}{'(b) POC1-baseline flag %':>26}{'first':>7}")
ratio = {1: 1.05, 2: 0.10, 3: 0.93, 4: 1.02, 5: 1.03, 6: 0.12,
         7: 0.07, 8: 0.10, 9: 0.13, 10: 0.95, 11: 0.10, 12: 0.89}
for pad in (2, 8, 6, 7, 9, 11, 1, 3, 4, 5, 10, 12):
    lab = co.loc[pad, "poc1"]
    ff = co.loc[pad, "first_flag"]
    print(f"{pad:>5}{lab:>12}{ratio[pad]:>11.2f}{fa.loc[pad,'flag_pct']:>25.2f}%"
          f"{co.loc[pad,'flag_pct']:>25.2f}%"
          f"{(int(ff) if pd.notna(ff) else '-'):>7}")

print(f"""
WHAT EACH TEST LICENSES — they answer different questions and neither substitutes
for the other.

(a) FALSE-ALARM TEST, baseline = July day 1. Asks: do stable sensors raise a
    flag? Valid whatever happened in the 45-day gap. It CANNOT see the collapse,
    because by 17 July the collapse had already happened and it is baked into the
    baseline — PAD 2 reads 0.37 % and PAD 8 1.83 % here, correctly, because
    relative to their own July day-1 state they are stable. Reading that as
    "PAD 2 is fine" would be a category error.
    What it licenses: a false-alarm rate of 4.78 % pooled over POC1-healthy pads
    against POC1's 1.5 % LOSO figure, carried by PAD 3 and PAD 4 flagging from
    passage 1 — a baseline-offset signature, since a genuine drift crosses later.

(b) CONTINUITY TEST, baseline = POC1 session 1. Asks: does the detector catch
    what changed across the gap? A-54 removed the confound that made this
    uninterpretable, so it now reads directly.""")
det = [p for p in (2, 8) if co.loc[p, "flag_pct"] > 90]
print(f"\nPAD 2 and PAD 8 under the frozen rules with a POC1 baseline: "
      f"flagged on {co.loc[2,'flag_pct']:.1f} % and {co.loc[8,'flag_pct']:.1f} % of "
      f"July passages, first flag at passage {int(co.loc[2,'first_flag'])} and "
      f"{int(co.loc[8,'first_flag'])}.")
print(f"=> DETECTED: {len(det)}/2")

print(f"""
BUT STATE THE LIMIT PLAINLY. There is no data between 2026-06-02 and 2026-07-17,
so this is detection at the FIRST AVAILABLE OPPORTUNITY, not detection in flight.
Lead time is zero by construction — not because the rule was slow, but because
nothing was recorded while the sensors were failing. These two instances add to
the DETECTION count and add nothing to the LEAD-TIME evidence, which still rests
entirely on PAD 6 (A-22, §3.4).

They are also easy detections: a 90 % amplitude loss against a pair threshold of
-0.727 in log2 (a ratio of 0.605) is not a demanding test. None of the five
instances probes sensitivity to a small loss.""")

for n_inst, lab in ((3, "POC1 alone"), (5, "with PAD 2 and PAD 8")):
    b = 1 - 0.05 ** (1 / n_inst)
    print(f"\n{lab:<24} n={n_inst}, 0 misses -> exact 95 % upper bound on the "
          f"miss rate = 1 - 0.05^(1/{n_inst}) = {100*b:.1f} %")
print("""
The bound improves from 63 % to 45 %. It is the one headline number that moves in
the right direction, and it is still barely a bound: a SIL4 claim needs failure
counts three orders of magnitude larger.""")

# ==================================================================== ITEM 4
print(f"\n{L}\nITEM 4 — DOES THE HEAD VOTE STILL FUNCTION?\n{L}")
print("""Head 1 = PADs 1-4, of which PAD 2 has collapsed: one of four.
Head 2 = PADs 5-8, of which 6, 7 and 8 have collapsed: THREE of four, leaving
PAD 5 as the only healthy sensor in that head. The head output is the median of
four counts, so in Head 2 the median is decided by collapsed sensors.""")

s = pd.read_csv(OUT / "blind_sensors.csv")
r = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
aux = pd.read_csv(OUT / "july_aux_records.csv", low_memory=False).drop(columns=["set"])
aux["train_free"] = aux["max_h1_pm"] < CONFIG["detect"]["absolute_gate_pm"]
c = s[s["agree"].notna()].merge(
    r[["name", "set", "gated"]], on="name").merge(
    aux[["name", "train_free"]], on="name")
c = c[c["gated"] & ~c["train_free"].fillna(False)]

piv = c.pivot_table(index=["name", "set"], columns="pad", values="count")
piv = piv.dropna(subset=[1, 2, 3, 4, 5, 6, 7, 8])
print(f"\nrecords with all eight operational counts: {len(piv)}")

h1 = piv[[1, 2, 3, 4]]
h2 = piv[[5, 6, 7, 8]]
h1_med, h2_med = h1.median(axis=1), h2.median(axis=1)
print(f"\n{'head':<8}{'collapsed pads':<18}{'unanimous':>12}{'spread med':>12}"
      f"{'spread p90':>12}")
for lab, blk in (("Head 1", h1), ("Head 2", h2)):
    coll = sorted(set(blk.columns) & COLLAPSED)
    un = (blk.nunique(axis=1) == 1).mean()
    sp = (blk.max(axis=1) - blk.min(axis=1))
    print(f"{lab:<8}{str(coll):<18}{100*un:>11.1f}%{sp.median():>12.0f}"
          f"{sp.quantile(.9):>12.0f}")

print(f"""
THE SHARP TEST. In Head 2, PAD 5 is the only sensor whose amplitude is unchanged
from commissioning, so its count is the best available estimate for that head.
How often does the four-sensor median disagree with it?""")
dis = (h2_med != piv[5])
print(f"   Head 2 median != PAD 5              : {pct(int(dis.sum()), len(piv))}")
print(f"   median ABOVE PAD 5 (over-count)     : "
      f"{pct(int((h2_med > piv[5]).sum()), len(piv))}")
print(f"   median BELOW PAD 5 (under-count)    : "
      f"{pct(int((h2_med < piv[5]).sum()), len(piv))}")
d2 = (h2_med - piv[5])[dis]
print(f"   where they differ: mean {d2.mean():+.1f} axles, "
      f"p10 {d2.quantile(.1):+.0f}, p90 {d2.quantile(.9):+.0f}, "
      f"|diff| p90 {d2.abs().quantile(.9):.0f}")
print(f"\n   for contrast, Head 1 median vs its healthy majority (PADs 1,3,4):")
h1_healthy = piv[[1, 3, 4]].median(axis=1)
dis1 = (h1_med != h1_healthy)
print(f"   Head 1 median != healthy-3 median   : "
      f"{pct(int(dis1.sum()), len(piv))}")

print(f"\n{'set':<9}{'records':>9}{'H2 med != PAD5':>17}{'H1 med != healthy3':>21}")
for st in ("SEALED", "open"):
    m = piv.index.get_level_values("set") == st
    print(f"{st:<9}{int(m.sum()):>9}{pct(int(dis[m].sum()), int(m.sum())):>17}"
          f"{pct(int(dis1[m].sum()), int(m.sum())):>21}")

print(f"""
IS THE MONITORING RULE EXCLUDING THEM? No. The head vote is
`np.median(counts_over_four_pads)` in the driver, and `lightrack/safety.py`
`count_axles` takes a single trace and a speed — it has no health argument, no
health import, and no way to know a sensor is degraded. The diagnostic chain
computes the flags; nothing consumes them.

That separation was deliberate (briefing §7.5: the two chains need opposite
operating points and must not share a filter below baseline removal) and it is
still right at the FILTER level. But "must not share a filter" was read as "must
not communicate", and the result is a safety chain that keeps voting sensors its
own diagnostic chain has already condemned. The missing link is one bit per
sensor per passage — healthy / excluded — flowing diagnostic to safety, which
shares no filter and no operating point.

THE ABSOLUTE FLOOR UNDER THE PEAK GATE IS NOT APPLIED, and should not be: a gate
that keeps a sensor at 8 % of commissioning inside the head vote is worse than
one that lets it subdivide visibly. Exclusion is the fix; thresholding is not.

WHAT EXCLUSION WOULD COST. With three of four pads excluded, Head 2 falls to a
single sensor and has no vote left at all - it becomes a 1-FBG head, which cannot
self-check. That is not an argument against exclusion; it is the 4->2 FBG
argument arriving from a new direction, and it belongs in POC2 §4.""")

# ============================================================ MOUNTING TYPES
print(f"\n{L}\nTHE COLLAPSE BY MOUNTING TYPE\n{L}")
p5 = pd.read_csv(OUT / "phase5_position.csv")
p5["july_frac"] = p5["pad"].map(ratio)
p5["collapsed"] = p5["pad"].isin(COLLAPSED)
print(f"{'mount':<22}{'pads':<28}{'n':>4}{'collapsed':>16}{'rate':>8}")
for mnt, g in p5.groupby("mount"):
    cc = g[g["collapsed"]]
    print(f"{mnt:<22}{str(sorted(g['pad'].tolist())):<28}{len(g):>4}"
          f"{str(sorted(cc['pad'].tolist())):>16}{pct(len(cc), len(g)):>8}")
print(f"\n{'pad':>4}  {'mount':<22}{'commissioning':>15}{'POC1 final':>12}"
      f"{'POC1 70% day':>14}{'July frac':>11}  verdict")
for _, q in p5.sort_values(["mount", "pad"]).iterrows():
    dd = "-" if pd.isna(q["days_to_cross"]) else f"{q['days_to_cross']:.1f}"
    print(f"{int(q['pad']):>4}  {q['mount']:<22}{q['commissioning_pm']:>15.1f}"
          f"{q['final_frac']:>12.2f}{dd:>14}{q['july_frac']:>11.2f}"
          f"  {'COLLAPSED' if q['collapsed'] else 'stable'}")
std = p5[p5["mount"] == "standard glued"]
alt = p5[p5["mount"].isin(["clipse (clamped)", "taraude (screwed)"])]
print(f"""
STANDARD GLUED: {int(std['collapsed'].sum())} of {len(std)} collapsed — exactly half, not a majority.
Two of those four (PADs 6 and 7) had already crossed 70 % during POC1's 18 days,
at day 7.0 and day 18.0. PADs 2 and 8 are the new ones.

ALTERNATIVE MOUNTINGS: {int(alt['collapsed'].sum())} of {len(alt)} collapsed. PAD 10 (clipse) is at
{ratio[10]:.2f} and PAD 12 (taraude) at {ratio[12]:.2f} of commissioning after ~65 days.

DELIBERATE WEAK VARIANTS: PAD 9 (1/2 collage) and PAD 11 (unprepared surface)
both collapsed, as they were installed to do.

CAVEAT THAT LIMITS ALL OF IT: the timing inside the 45-day gap is unknown, so
this is a 65-DAY OUTCOME, not a durability curve. Two mountings surviving and
half the standard ones failing is an outcome at one time point with n=1 per
alternative mounting.""")

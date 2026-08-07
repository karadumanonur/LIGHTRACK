"""Items 1-3 — the one-track hypothesis, restated capture rates, hourly profile.

ITEM 1. EBP counts ~82/day where the passenger timetable alone gives ~137. Line
130 is double track and the installation monitors ONE track (Fil1/Fil2 are the
two rails of one track). If EBP counts a single direction, the ratio should be
close to 2. Physical direction is taken from the ORDER of the two station
timestamps, not from the relation names - a train recorded at Aiseau before
Farciennes is travelling toward Charleroi (PK increasing).

ITEM 3. The 2026-05-19 detection rate needed a traffic-distribution assumption
and spanned 55-95 % as a result. With per-train timestamps no assumption is
needed: count the trains actually timetabled inside the covered window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

OUT = Path(r"C:\Users\onurk\Desktop\data\out")
REF = Path(r"C:\Users\onurk\Desktop\data\reference\ebp_daily_train_counts.csv")
pd.set_option("display.width", 240)
L = "=" * 112

p = pd.read_csv(OUT / "punctuality_site.csv", low_memory=False)
p["t"] = pd.to_datetime(p["t"])
p["date"] = p["t"].dt.date

# ---------------------------------------------------------------- ITEM 1
print(f"{L}\nITEM 1 — DOES EBP COUNT ONE TRACK?\n{L}")
w = p.pivot_table(index=["DATDEP", "TRAIN_NO"], columns="station",
                  values="t", aggfunc="first")
w = w.dropna(subset=["Aiseau", "Farciennes"])
w["toward_charleroi"] = w["Farciennes"] > w["Aiseau"]   # Aiseau first => PK increasing
w["leg_s"] = (w["Farciennes"] - w["Aiseau"]).dt.total_seconds().abs()
print(f"trains seen at BOTH stations: {len(w)}")
print(f"   toward Charleroi (Aiseau first): {int(w['toward_charleroi'].sum())}")
print(f"   toward Namur     (Farciennes first): {int((~w['toward_charleroi']).sum())}")
print(f"   inter-station transit: median {w['leg_s'].median():.0f} s, "
      f"p10 {w['leg_s'].quantile(.1):.0f}, p90 {w['leg_s'].quantile(.9):.0f}")
print(f"   implied avg speed over 2.726 km: median "
      f"{2.726/(w['leg_s'].median()/3600):.1f} km/h")

# map relation-direction labels to physical direction
w2 = w.reset_index().merge(
    p[["DATDEP", "TRAIN_NO", "RELATION_DIRECTION", "RELATION"]].drop_duplicates(
        ["DATDEP", "TRAIN_NO"]), on=["DATDEP", "TRAIN_NO"], how="left")
print(f"\nrelation-direction -> physical direction:")
for rd, g in w2.groupby("RELATION_DIRECTION"):
    if len(g) < 20:
        continue
    frac = g["toward_charleroi"].mean()
    print(f"   {str(rd)[:52]:<54} n={len(g):>5}  toward Charleroi {100*frac:5.1f} %")

# per-day, per-direction vs EBP
ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
w2["date"] = pd.to_datetime(w2["Aiseau"]).dt.date
per = w2.groupby(["date", "toward_charleroi"]).size().unstack()
per.columns = ["to_Namur", "to_Charleroi"]
per["both"] = per.sum(axis=1)
m = per.reset_index().merge(ebp[["date", "vns", "cvt", "total", "day_class"]], on="date")
print(f"\n{L}\nPER-DAY: PUNCTUALITY BY DIRECTION vs EBP\n{L}")
print(f"{'date':<12}{'class':<16}{'->Namur':>9}{'->Chrl':>8}{'both':>7}"
      f"{'EBP vns':>9}{'EBP tot':>9}{'vns/1dir':>10}")
for _, r in m.iterrows():
    one = max(r["to_Namur"], r["to_Charleroi"])
    print(f"{str(r['date']):<12}{r['day_class']:<16}{int(r['to_Namur']):>9}"
          f"{int(r['to_Charleroi']):>8}{int(r['both']):>7}{int(r['vns']):>9}"
          f"{int(r['total']):>9}{r['vns']/one:>10.2f}")
nw = m[m["day_class"] == "normal_weekday"]
print(f"\nnormal weekdays only (n={len(nw)}):")
print(f"   median per direction : {nw[['to_Namur','to_Charleroi']].median().mean():.0f}")
print(f"   median both          : {nw['both'].median():.0f}")
print(f"   median EBP vns       : {nw['vns'].median():.0f}")
print(f"   ratio both / EBP tot : {(nw['both']/nw['total']).median():.2f}")
print(f"   ratio 1-dir / EBP vns: "
      f"{(nw[['to_Namur','to_Charleroi']].max(axis=1)/nw['vns']).median():.2f}")
print("""
READING. If EBP counted one direction of passenger traffic the last ratio would
sit near 1.00. Report what it actually is; a value materially below 1 means EBP
counts MORE than one direction of passengers (i.e. includes freight), and a
value above 1 means it counts less.""")

# ---------------------------------------------------------------- ITEM 3
print(f"\n{L}\nITEM 3 — 2026-05-19 DETECTION RATE, NO DISTRIBUTION ASSUMPTION\n{L}")
day = pd.Timestamp("2026-05-19").date()
d19 = p[(p["date"] == day) & (p["station"] == "Aiseau")].copy()
print(f"passenger trains at the site on {day}: {len(d19)}")
hh = d19["t"].dt.hour.value_counts().sort_index()
print(f"\nhourly profile (local):")
for hr in range(24):
    n = int(hh.get(hr, 0))
    if n:
        print(f"   {hr:02d}:00  {n:>3}  {'#'*n}")

# the sensor's covered window on 05-19, local
lo = pd.Timestamp("2026-05-19 01:36:23")
hi = pd.Timestamp("2026-05-19 11:36:14")
inwin = d19[(d19["t"] >= lo) & (d19["t"] <= hi)]
print(f"\nsensor covered window (local): {lo:%H:%M:%S} - {hi:%H:%M:%S}")
print(f"passenger trains timetabled INSIDE that window: {len(inwin)}")
print(f"sensor files retained in the window: 33")
k, n = 33, len(inwin)
lo_ci, hi_ci = (0.0 if k == 0 else beta.ppf(0.025, k, n-k+1),
                1.0 if k >= n else beta.ppf(0.975, k+1, n-k))
print(f"\nCAPTURE RATE = 33 / {n} = {100*33/n:.1f} %"
      f"   [95 % CI {100*lo_ci:.1f}-{100*hi_ci:.1f} %]")
print(f"   previous estimate (A-43, EBP + uniform assumption): 55-95 %")
print("""
This is a CAPTURE rate, not a detection rate: retention and detection are still
confounded, and the denominator excludes freight entirely, so the true capture
of ALL traffic is lower again. What has changed is that the traffic-distribution
assumption is gone - the trains are counted, not modelled.""")

# ---------------------------------------------------------------- campaign
print(f"\n{L}\nCAMPAIGN-WIDE RESTATEMENT (14/05 - 02/06)\n{L}")
camp = p[(p["station"] == "Aiseau") &
         (p["date"] >= pd.Timestamp("2026-05-14").date()) &
         (p["date"] <= pd.Timestamp("2026-06-02").date())]
tot_pass = len(camp)
print(f"passenger trains 14/05-02/06 : {tot_pass}")
print(f"EBP total for the same period : 1187")
print(f"\n{'denominator':<40}{'files 93':>12}{'gated 70':>12}")
for lab, den in (("EBP total (A-30, superseded)", 1187),
                 ("passenger trains (punctuality)", tot_pass)):
    print(f"{lab:<40}{100*93/den:>11.1f}%{100*70/den:>11.1f}%")
print("\nBoth exclude freight. The punctuality denominator is the better-founded")
print("of the two, and it moves the headline capture rate DOWN.")
m.to_csv(OUT / "direction_vs_ebp.csv", index=False)
print(f"\nwrote direction_vs_ebp.csv")

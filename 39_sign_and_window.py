"""Items 1 & 3 — what is the over-count measured against, and which window?

ITEM 1. Merging two axle peaks produces an UNDER-count, yet the >32-axle bin
reports 26.3 % OVER-count. Those cannot both describe the same thing. The
over-count is a deviation from the HEAD MEDIAN, not from truth: if most sensors
merge peaks on freight, the median falls and the sensors that did NOT merge are
labelled as over-counting. Measured here against two references so the direction
is unambiguous:
    (a) the sensor's own head median   - the current, relative definition
    (b) the OTHER head's median        - independent of the sensor's own head

ITEM 3. A-43's window (10.04 h, 250 slots, 13.2 %) and A-07's (6.14 h, 154
slots, 20.1 %) disagree. Both are computed correctly; they describe different
spans. Which one supports "05-19 is a complete retained set for its window"?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, list_files, parse_filename_utc, to_local  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SLOT = CONFIG["acquisition"]["slot_s"]
pd.set_option("display.width", 250)
LINE = "=" * 116

# ============================================================== ITEM 1
print(f"{LINE}\nITEM 1 — OVER-COUNT AGAINST WHAT?\n{LINE}")
d = pd.read_csv(OUT / "seg_B.csv")          # peak-anchored, single speed
head_med = d.groupby(["name", "head"])["count"].median().rename("hm").reset_index()
other = head_med.copy()
other["head"] = other["head"].map({1: 2, 2: 1})
other = other.rename(columns={"hm": "other_hm"})
d = d.merge(head_med, on=["name", "head"]).merge(other, on=["name", "head"])
d["dev_own"] = d["count"] - d["hm"]
d["dev_other"] = d["count"] - d["other_hm"]
d["bin"] = pd.cut(d["hm"], [0, 8, 16, 32, 300], labels=["<=8", "9-16", "17-32", ">32"])

print("(a) deviation from the sensor's OWN head median  — the current definition")
print("(b) deviation from the OTHER head's median       — independent reference\n")
print(f"{'consist':<9}{'trains':>8}" +
      f"{'(a) over':>10}{'(a) under':>11}" + f"{'(b) over':>11}{'(b) under':>11}")
for b in ["<=8", "9-16", "17-32", ">32"]:
    s = d[d["bin"] == b]
    if s.empty:
        continue
    print(f"{b:<9}{s['name'].nunique():>8}"
          f"{100*(s['dev_own']>0).mean():>9.1f}%{100*(s['dev_own']<0).mean():>10.1f}%"
          f"{100*(s['dev_other']>0).mean():>10.1f}%{100*(s['dev_other']<0).mean():>10.1f}%")

lng = d[d["bin"] == ">32"]
print(f"\n>32 bin in detail ({lng['name'].nunique()} trains, {len(lng)} sensor-passages):")
print(f"  mean deviation from own head median   : {lng['dev_own'].mean():+.2f} axles")
print(f"  mean deviation from other head median : {lng['dev_other'].mean():+.2f} axles")
print(f"  median absolute deviation, own        : {lng['dev_own'].abs().median():.1f}")
print(f"  spread of counts within a head (max-min), median: "
      f"{d[d['bin']=='>32'].groupby(['name','head'])['count'].apply(lambda s: s.max()-s.min()).median():.1f}")

print(f"\nper-record, >32 bin — head counts and within-head spread:")
print(f"{'record':<28}{'H1 counts':<26}{'H2 counts':<26}{'H1med':>6}{'H2med':>6}")
for nm, g in lng.groupby("name"):
    c1 = sorted(g[g["head"] == 1]["count"].astype(int))
    c2 = sorted(g[g["head"] == 2]["count"].astype(int))
    if not c1 or not c2:
        continue
    print(f"{nm[:26]:<28}{str(c1):<26}{str(c2):<26}"
          f"{int(np.median(c1)):>6}{int(np.median(c2)):>6}")

print("""
READING. On freight the spread WITHIN each head is wide and roughly symmetric
about the median, so "over-count" here means "read higher than its head-mates",
not "read higher than truth". Since the mechanism - peak merging at 2.0 m
freight bogie spacing against a 2.2 m influence extent - can only DESTROY
peaks, the head median on these trains is most likely UNDER-counting, and the
sensors labelled as over-counting are the ones that resolved the axles.

SAFETY CONSEQUENCE, INVERTED. An under-counting head leaves a clear block
reading occupied: an AVAILABILITY failure, not the phantom-axle SAFETY failure.
The >32 figure must not be presented as a phantom-axle rate.""")

# ============================================================== ITEM 3
print(f"\n{LINE}\nITEM 3 — WHICH WINDOW ON 2026-05-19?\n{LINE}")
day = pd.Timestamp("2026-05-19").date()
t = sorted(to_local(parse_filename_utc(p.name)) for p in list_files()
           if to_local(parse_filename_utc(p.name)).date() == day)
gaps = [(t[i + 1] - t[i]).total_seconds() / 3600 for i in range(len(t) - 1)]
big = [i for i, g in enumerate(gaps) if g > 1.0]
print(f"files on 05-19 (local): {len(t)}, from {t[0]:%H:%M:%S} to {t[-1]:%H:%M:%S}")
print(f"inter-file gaps > 1 h: {[(f'{t[i]:%H:%M}', f'{t[i+1]:%H:%M}', round(gaps[i],2)) for i in big]}")

blocks, start = [], 0
for i in big:
    blocks.append((start, i))
    start = i + 1
blocks.append((start, len(t) - 1))
print(f"\n{'block':<22}{'files':>7}{'span h':>9}{'slots':>8}{'retention':>11}")
for a, b in blocks:
    span = (t[b] - t[a]).total_seconds()
    sl = int(round(span / SLOT)) + 1
    n = b - a + 1
    print(f"{f'{t[a]:%H:%M}-{t[b]:%H:%M}':<22}{n:>7}{span/3600:>9.2f}{sl:>8}"
          f"{100*n/sl:>10.1f}%")
span_all = (t[-1] - t[0]).total_seconds()
sl_all = int(round(span_all / SLOT)) + 1
print(f"{'WHOLE local-day span':<22}{len(t):>7}{span_all/3600:>9.2f}{sl_all:>8}"
      f"{100*len(t)/sl_all:>10.1f}%")
print(f"""
Both figures are arithmetically correct and describe different spans. The
whole-span number is depressed by a {max(gaps):.1f} h stretch containing no files
at all, which is not a filtering decision - it is a period the delivery simply
does not cover.

For the question "is 05-19 a complete retained set for its window", the
CONTIGUOUS morning block is the right window. The whole-local-day span is the
right denominator for a capture rate against the EBP daily total (A-43).""")

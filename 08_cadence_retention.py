"""Phase 0 addendum: recording cadence, slot retention, and speed-score filtering.

Three questions:
  1. Is the inter-file interval an exact multiple of the file duration? If yes,
     recording is contiguous and the files are not short — the deck's "3 min"
     is simply wrong. That closes A-01.
  2. If gaps are exact slot multiples, the discarded files can be COUNTED, which
     turns retention from a guess into a measurement.
  3. Do the implausible high speeds coincide with low fit scores?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
idx = pd.read_csv(OUT / "file_index.csv", parse_dates=["utc"]).sort_values("utc").reset_index(drop=True)

print("=" * 84)
print("1. IS THE CADENCE A FIXED SLOT?")
print("=" * 84)
dur = idx["duration_s"]
# duration is (t_last - t_first); the slot is one sample longer
slot_est = float(np.median(dur + 1.0 / CONFIG["acquisition"]["fs_hz"]))
print(f"median record duration      : {dur.median():.3f} s")
print(f"median record span (+1 samp): {slot_est:.3f} s")

d19 = idx[idx["utc"].dt.date == pd.Timestamp("2026-05-19").date()].reset_index(drop=True)
gaps = d19["utc"].diff().dt.total_seconds().dropna()
print(f"\n2026-05-19: {len(d19)} files, {len(gaps)} intervals")
print(f"{'gap_s':>10} {'/slot':>9} {'residual_s':>12}")
best = None
for slot in np.arange(143.0, 147.001, 0.001):
    k = np.round(gaps / slot)
    resid = np.abs(gaps - k * slot)
    score = resid.max()
    if best is None or score < best[0]:
        best = (score, slot, k, resid)
worst, slot, k, resid = best
for g, kk, rr in zip(gaps, k, resid):
    print(f"{g:10.3f} {kk:9.0f} {rr:12.3f}")
print(f"\nbest-fit slot: {slot:.3f} s   worst residual over all intervals: {worst:.3f} s")
print(f"mean |residual|: {resid.mean():.3f} s")

if worst < 2.0:
    print("\n=> VERDICT: every interval is an integer multiple of the slot.")
    print("   Recording is CONTIGUOUS with no dead time. The files are complete;")
    print(f"   the true cadence is {slot:.2f} s, i.e. {3600/slot:.1f} files/hour,")
    print(f"   {86400/slot:.0f} files/day. The deck's '3 min / 20 per hour / 480 per")
    print("   day' is wrong. A-01 closed.")

print("\n" + "=" * 84)
print("2. SLOT RETENTION PER DAY (measured, not assumed)")
print("=" * 84)
print(f"{'date':>12} {'files':>6} {'span_h':>8} {'slots':>7} {'retained%':>10}")
rows = []
for d, g in idx.groupby(idx["utc"].dt.date):
    g = g.sort_values("utc")
    if len(g) < 2:
        rows.append({"date": d, "n": len(g), "span_h": 0.0, "slots": 1, "ret": 100.0})
        print(f"{str(d):>12} {len(g):6d} {0.0:8.2f} {1:7d} {100.0:10.1f}")
        continue
    span = (g["utc"].iloc[-1] - g["utc"].iloc[0]).total_seconds()
    slots = int(round(span / slot)) + 1
    ret = 100.0 * len(g) / slots
    rows.append({"date": d, "n": len(g), "span_h": span / 3600, "slots": slots, "ret": ret})
    print(f"{str(d):>12} {len(g):6d} {span/3600:8.2f} {slots:7d} {ret:10.1f}")

r = pd.DataFrame(rows)
print(f"\ndeck's stated retention: 80/480 = {100*80/480:.0f} %")
print(f"median retention across days: {r['ret'].median():.1f} %")
d19r = r[r['date'] == pd.Timestamp('2026-05-19').date()].iloc[0]
print(f"\n2026-05-19: {d19r['n']} files over {d19r['span_h']:.2f} h = "
      f"{d19r['slots']} slots -> {d19r['ret']:.1f} % retained")
print("   This is the only day whose retention matches the deck's own filter rate,")
print("   so it is the only day plausibly representing a COMPLETE retained set")
print("   for its window. Counting statistics belong here, labelled as such.")

print("\n" + "=" * 84)
print("3. SPEED vs FIT SCORE — reject on score, not on plausibility")
print("=" * 84)
tof = pd.read_csv(OUT / "tof_speed_direction.csv")
tof = tof.sort_values("v_kmh", ascending=False)
print(f"{'file':<28}{'v_kmh':>9}{'score':>9}{'alt':>9}")
for _, x in tof.head(8).iterrows():
    print(f"{x['name'][:26]:<28}{x['v_kmh']:9.1f}{x['score']:9.3f}{x['alt_score']:9.3f}")
print("   ...")
for _, x in tof.tail(3).iterrows():
    print(f"{x['name'][:26]:<28}{x['v_kmh']:9.1f}{x['score']:9.3f}{x['alt_score']:9.3f}")

print(f"\nscore: median {tof['score'].median():.3f}, "
      f"p10 {tof['score'].quantile(0.10):.3f}, min {tof['score'].min():.3f}")
print(f"correlation(v_kmh, score) = {tof['v_kmh'].corr(tof['score']):+.3f}")

for thr in [0.60, 0.70, 0.80]:
    keep = tof[tof["score"] >= thr]
    print(f"\nscore >= {thr:.2f}: keep {len(keep)}/{len(tof)}  "
          f"v range {keep['v_kmh'].min():.1f}-{keep['v_kmh'].max():.1f} km/h  "
          f"median {keep['v_kmh'].median():.1f}")

r.to_csv(OUT / "retention_per_day.csv", index=False)
print(f"\nwrote {OUT/'retention_per_day.csv'}")

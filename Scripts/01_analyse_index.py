"""Phase 0: interrogate the index built by 00_inventory.py.

Answers reconnaissance questions 1, 3, 5, 7, 8, 9 from the Phase 0 brief.
Descriptive only.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
idx = pd.read_csv(OUT / "file_index.csv", parse_dates=["utc", "local"])
idx = idx.sort_values("utc").reset_index(drop=True)
ch = pd.read_csv(OUT / "channel_stats.csv", parse_dates=["utc"])

LINE = "=" * 78


def hdr(s: str) -> None:
    print(f"\n{LINE}\n{s}\n{LINE}")


# ---------------------------------------------------------------- 1. inventory
hdr("1. FILE INVENTORY")
print(f"files            : {len(idx)}")
print(f"total bytes      : {idx['bytes'].sum():,}  ({idx['bytes'].sum()/2**30:.2f} GiB)")
print(f"size min/max     : {idx['bytes'].min():,} / {idx['bytes'].max():,}")
print(f"decimal separator: {idx['decimal'].value_counts().to_dict()}")
print(f"UTC coverage     : {idx['utc'].min()}  ->  {idx['utc'].max()}")
print(f"local coverage   : {idx['local'].min()}  ->  {idx['local'].max()}")

# --------------------------------------------------------- 3. shape / 5. rate
hdr("3 & 5. SHAPE AND SAMPLING RATE")
print(f"rows   min/median/max : {idx['n_rows'].min()} / "
      f"{idx['n_rows'].median():.0f} / {idx['n_rows'].max()}")
print(f"briefing expectation  : {CONFIG['acquisition']['expected_rows_briefing']} "
      f"(3 min x 500 Hz)")
print(f"shortfall vs briefing : "
      f"{100*(1-idx['n_rows'].median()/90000):.1f} % fewer rows than briefed")
print(f"duration s min/med/max: {idx['duration_s'].min():.3f} / "
      f"{idx['duration_s'].median():.3f} / {idx['duration_s'].max():.3f}")
print(f"dt median (all files) : {idx['dt_median'].min():.6f} .. {idx['dt_median'].max():.6f} s")
print(f"dt min / dt max       : {idx['dt_min'].min():.6f} / {idx['dt_max'].max():.6f} s")
print(f"fs implied            : {idx['fs_hz'].min():.4f} .. {idx['fs_hz'].max():.4f} Hz")
print(f"files with dt outliers: {(idx['n_dt_outliers']>0).sum()}  "
      f"(total outlier samples: {idx['n_dt_outliers'].sum()})")
print(f"columns per row       : {CONFIG['format']['n_columns']} "
      f"(1 time + 12 wavelength)")

# ------------------------------------------------- 6/A-06. instrument clock
hdr("A-06. INSTRUMENT CLOCK vs FILENAME TIMESTAMP")
d_name = idx["utc"].diff().dt.total_seconds()
d_clock = idx["t_first"].diff()
same_run = (d_clock > 0) & (np.abs(d_clock - d_name) < 5.0)
print("Consecutive-file spacing, filename vs Time[sec] column:")
print(f"  pairs where the two agree within 5 s : {int(same_run.sum())} / {len(idx)-1}")
resid = (d_clock - d_name)[same_run]
if len(resid):
    print(f"  residual (clock - filename) on those: mean {resid.mean():+.3f} s, "
          f"sd {resid.std():.3f} s, max |.| {resid.abs().max():.3f} s")
breaks = idx.index[1:][~same_run[1:]]
print(f"\n  clock discontinuities (resets / gaps): {len(breaks)}")
for i in breaks:
    print(f"    before {idx.loc[i,'name']}: filename +{d_name[i]:,.1f} s, "
          f"clock {d_clock[i]:+,.1f} s")

# ------------------------------------------- 7/8. date coverage and the gap
hdr("7 & 8. DATE COVERAGE, GAPS, AND THE 12/06 OUTAGE")
idx["date_utc"] = idx["utc"].dt.date
per_day = idx.groupby("date_utc").size()
print("files per UTC day:")
for d, n in per_day.items():
    print(f"  {d}  {n:3d}")
print(f"\ndistinct days: {len(per_day)}")

all_days = pd.date_range(idx["utc"].min().date(), idx["utc"].max().date(), freq="D").date
missing = [d for d in all_days if d not in set(per_day.index)]
print(f"days with NO data inside the covered span: {missing}")

print("\ninter-file gaps > 1 hour:")
gap = idx["utc"].diff().dt.total_seconds()
for i in idx.index[1:]:
    if gap[i] > 3600:
        print(f"  {gap[i]/3600:7.2f} h  after {idx.loc[i-1,'name']}  "
              f"-> {idx.loc[i,'name']}")

print("\n12/06/2026 -> 03/07/2026 outage:")
print(f"  last file in this delivery: {idx['utc'].max()} UTC")
print("  VERDICT: the outage window lies entirely AFTER the last file here.")
print("  It cannot be confirmed or bounded from this dataset — there is simply")
print("  no data on either side of it. Not a loader bug; the delivery stops on")
print("  2026-06-02, ten days before the outage is said to begin.")

# ------------------------------------------------------- 9. the golden window
hdr("9. GOLDEN TEST WINDOW — 2026-05-18, 08:45-09:07")
def window_hits(day: str, h0: int, m0: int, h1: int, m1: int, tzlabel: str,
                col: str) -> pd.DataFrame:
    d = pd.Timestamp(day).date()
    sel = idx[idx[col].dt.date == d]
    t = sel[col].dt
    mask = ((t.hour * 60 + t.minute) >= h0 * 60 + m0) & \
           ((t.hour * 60 + t.minute) <= h1 * 60 + m1)
    hits = sel[mask]
    print(f"  {day} 08:45-09:07 interpreted as {tzlabel}: {len(hits)} file(s)")
    for _, r in hits.iterrows():
        print(f"      {r['name']}")
    return hits

print("2026-05-18, both time-zone readings:")
window_hits("2026-05-18", 8, 45, 9, 7, "LOCAL (UTC+2)", "local")
window_hits("2026-05-18", 8, 45, 9, 7, "UTC", "utc")
print("\n  all files on 2026-05-18:")
for _, r in idx[idx["utc"].dt.date == pd.Timestamp("2026-05-18").date()].iterrows():
    print(f"      {r['name']}   UTC {r['utc'].time()}   local {r['local'].time()}")

print("\n2026-05-19, same window (the adjacent day):")
window_hits("2026-05-19", 8, 45, 9, 7, "LOCAL (UTC+2)", "local")
window_hits("2026-05-19", 8, 45, 9, 7, "UTC", "utc")

# ----------------------------------------------------- deck-referenced files
hdr("FILES NAMED IN THE BRIEFING'S PLOT TITLES (§5.2)")
for stem in ["20260602T122326", "20260515T122333"]:
    hit = idx[idx["name"].str.startswith(stem)]
    print(f"  {stem}: {'PRESENT -> ' + hit.iloc[0]['name'] if len(hit) else 'absent'}")

# ------------------------------------------------------------- 4. units
hdr("4. UNITS — decided from range, not assumption")
g = ch.groupby("channel")
print(f"{'ch':>5} {'min_nm':>10} {'max_nm':>10} {'median_nm':>11} "
      f"{'med p2p_pm':>11} {'max p2p_pm':>11} {'med std_pm':>11}")
for c, sub in g:
    print(f"{c:>5} {sub['min_nm'].min():10.4f} {sub['max_nm'].max():10.4f} "
          f"{sub['median_nm'].median():11.4f} {sub['p2p_pm'].median():11.1f} "
          f"{sub['p2p_pm'].max():11.1f} {sub['std_pm'].median():11.1f}")

centres = g["median_nm"].median()
print(f"\nabsolute range across archive: {ch['min_nm'].min():.4f} .. "
      f"{ch['max_nm'].max():.4f} nm  (span {ch['max_nm'].max()-ch['min_nm'].min():.2f} nm)")
print("=> values are ABSOLUTE WAVELENGTH in nm; not pm, not baseline-subtracted.")
print(f"\nchannel spacing (nm), spectral order:")
sp = -np.diff(centres.values)
print("  " + "  ".join(f"{s:.3f}" for s in sp))
print(f"  min {sp.min():.3f}  max {sp.max():.3f}  mean {sp.mean():.3f}")

OUT.mkdir(exist_ok=True)
centres.to_csv(OUT / "channel_centres.csv")
print(f"\nwrote {OUT/'channel_centres.csv'}")

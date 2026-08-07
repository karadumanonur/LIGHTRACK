"""ACAT ground truth — load, characterise, and settle the questions that must be
settled before any number is quoted.

SOURCE: L130 Comptage ACAT KJ du 13-05 au 24-06-2026.xlsx, an axle-counter log
from section 06 / CAT_KJ on line 130 at Le Campinaire. Structure was supplied and
verified; it is not re-derived here.

THIS FILE DOES NOT REACH JULY. Span 2026-05-13 to 2026-06-24. The July sealed
days are not involved in any part of this work and the seal is NOT spent on it.

Four things are established here, in order, because each depends on the last:
  1. what the 310 AxleCount rows that are neither an entry nor an exit actually
     are - characterised before being used or discarded;
  2. the TIME ZONE, from the traffic profile rather than assumed (the same method
     as 49_punctuality_extract.py, and for the same reason);
  3. whether section 06 sees one track or both - which decides Task 5 and cannot
     be assumed either way;
  4. the raw material for the offset model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

DATA = Path(r"C:\Users\onurk\Desktop\data")
OUT = Path(CONFIG["paths"]["out_dir"])
XLS = DATA / "L130 Comptage ACAT KJ du 13-05 au 24-06-2026.xlsx"
pd.set_option("display.width", 250)
L = "=" * 116

raw = pd.read_excel(XLS, skiprows=5, header=None)
raw.columns = ["ts", "section", "code", "event", "v_new", "v_old"]
raw["ts"] = pd.to_datetime(raw["ts"])
print(f"{L}\nACAT SOURCE\n{L}")
print(f"rows: {len(raw)}   span {raw['ts'].min()} .. {raw['ts'].max()}")
print(f"sections: {raw['section'].dropna().unique().tolist()}")
print(f"events  : {raw['event'].value_counts().to_dict()}")
print(f"""
COVERAGE. This reference covers ALL of POC1 (14/05-02/06) and NONE of the July
blind set (17-27/07). Nothing here validates the blind set, and the July sealed
days are untouched by it.""")

ac = raw[raw["event"] == "AxleCount"].copy()
dl = raw[raw["event"] == "DirectionLastAxle"].copy()
ac["v_new"] = pd.to_numeric(ac["v_new"], errors="coerce")
ac["v_old"] = pd.to_numeric(ac["v_old"], errors="coerce")

# ------------------------------------------------------------ 1. the 310 rows
ac["kind"] = np.where((ac["v_old"] == 0) & (ac["v_new"] > 0), "entry",
                      np.where(ac["v_new"] == 0, "exit", "other"))
print(f"\n{L}\n1. THE ROWS THAT ARE NEITHER AN ENTRY NOR AN EXIT\n{L}")
print(ac["kind"].value_counts().to_dict())
oth = ac[ac["kind"] == "other"].copy()
oth["delta"] = oth["v_new"] - oth["v_old"]
print(f"\nn = {len(oth)}")
print(f"v_old -> v_new step size: {oth['delta'].value_counts().head(8).to_dict()}")
print(f"v_old distribution      : {oth['v_old'].value_counts().head(8).to_dict()}")
print(f"v_new distribution      : {oth['v_new'].value_counts().head(8).to_dict()}")

# are they interleaved inside a train, or standalone?
ac = ac.sort_values("ts").reset_index(drop=True)
ac["prev_kind"] = ac["kind"].shift()
ac["gap_prev_s"] = ac["ts"].diff().dt.total_seconds()
print(f"\nwhat precedes an 'other' row:")
print(ac.loc[ac["kind"] == "other", "prev_kind"].value_counts().to_dict())
print(f"time from the previous row, 'other' rows: "
      f"median {ac.loc[ac['kind']=='other','gap_prev_s'].median():.1f} s, "
      f"p90 {ac.loc[ac['kind']=='other','gap_prev_s'].quantile(.9):.1f} s")
print(f"time from the previous row, 'entry' rows: "
      f"median {ac.loc[ac['kind']=='entry','gap_prev_s'].median():.1f} s")
print("""
READING. An axle counter reports a running occupancy value. `v_old -> v_new` with
both non-zero is the SAME train still crossing - a partial update, not a new
train. If these rows sit seconds after an entry and carry small positive steps,
they are intermediate reports inside one passage and must NOT be counted as
trains. The entry rows (v_old == 0, v_new > 0) are the train events.""")

# ---------------------------------------------------------------- 2. timezone
ent = ac[ac["kind"] == "entry"].copy()
print(f"\n{L}\n2. TIME ZONE — from the traffic profile, not assumed\n{L}")
h = ent["ts"].dt.hour.value_counts().sort_index()
mx = h.max()
for hr in range(24):
    n = int(h.get(hr, 0))
    print(f"{hr:>5}{n:>7}   {'#' * int(56 * n / mx)}")
first = min(hr for hr in range(24) if h.get(hr, 0) > 0.05 * mx)
last = max(hr for hr in range(24) if h.get(hr, 0) > 0.05 * mx)
print(f"\nservice window (>5 % of peak): {first:02d}:00-{last:02d}:00, "
      f"busiest {int(h.idxmax()):02d}:00")
print(f"=> {'LOCAL (UTC+2)' if first >= 4 else 'UTC'}; "
      f"sensor filenames are UTC (A-05), sensor `local` is UTC+2.")

# ------------------------------------------------------- 3. one track or two?
print(f"\n{L}\n3. DOES SECTION 06 SEE ONE TRACK OR BOTH?\n{L}")
d = dl.copy()
d["dir"] = d["v_new"].astype(str).str.strip()
print(f"DirectionLastAxle values: {d['dir'].value_counts().to_dict()}")
print(f"""
An axle-counting SECTION is a track circuit on ONE track: a counting head sits on
one rail pair. If section 06 were somehow reporting both tracks the direction
split would be near 50/50 and the daily count would be roughly double the
single-direction traffic. Both are checked below.""")
per_day = ent.groupby(ent["ts"].dt.date).size()
print(f"\nentries per day: median {per_day.median():.0f}, "
      f"p10 {per_day.quantile(.1):.0f}, p90 {per_day.quantile(.9):.0f}, "
      f"max {per_day.max()}")
camp = ent[(ent["ts"] >= "2026-05-14") & (ent["ts"] < "2026-06-03")]
print(f"entries inside 2026-05-14..2026-06-02: {len(camp)}")

print(f"\n{L}\n4. COUNT DISTRIBUTION\n{L}")
vc = ent["v_new"].value_counts().sort_index()
print(f"distinct counts: {len(vc)}   total entries: {len(ent)}")
for k, v in vc.head(14).items():
    print(f"   {int(k):>4} axles  {v:>5}  {'#'*min(60, v//10)}")
print(f"   ... tail to {int(vc.index.max())}")
print(f"multiples of 12: {int(ent['v_new'].mod(12).eq(0).sum())}/{len(ent)} "
      f"= {100*ent['v_new'].mod(12).eq(0).mean():.1f} %")
print(f"12 or 24 axles : {int(ent['v_new'].isin([12,24]).sum())}/{len(ent)} "
      f"= {100*ent['v_new'].isin([12,24]).mean():.1f} %")

ent[["ts", "v_new", "v_old"]].rename(columns={"v_new": "axles"}).to_csv(
    OUT / "acat_entries.csv", index=False)
ac.to_csv(OUT / "acat_axlecount_all.csv", index=False)
d[["ts", "dir"]].to_csv(OUT / "acat_direction.csv", index=False)
print(f"\nwrote acat_entries.csv ({len(ent)}), acat_axlecount_all.csv, "
      f"acat_direction.csv ({len(d)})")

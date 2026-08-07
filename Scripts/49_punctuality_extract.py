"""Step 1c + Step 2 — finalise the site PK, then extract and characterise the
punctuality reference.

STEP 1c. The marker discrepancy is resolved: physical kilometre posts near the
site read 83-89, but they carry NETWORK kilometrage (offset ~+60 km, i.e.
continuing from the Brussels direction), while line 130's own PK convention runs
from the Namur end. Chainage along the alignment reproduces the stated station
PKs to within 40-75 m, which is the check that matters.

STEP 2. Filter the monthly punctuality files to the two PTCARs. Two things
established BEFORE any matching, because a mistake in either produces a
plausible-looking wrong answer:
  * TIME ZONE, determined empirically from the traffic profile rather than
    assumed - passenger rail has an unmistakable local-time shape.
  * COVERAGE - these are S61 stopping points, so non-stopping trains and all
    freight are invisible. A detection with no match has three possible causes
    and they must not be conflated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

DATA = Path(r"C:\Users\onurk\Desktop\data\Infrabel_data")
OUT = Path(r"C:\Users\onurk\Desktop\data\out")
SITE = (50.433245, 4.558323)
PTCAR = {16: "Aiseau", 392: "Farciennes"}
pd.set_option("display.width", 240)
L = "=" * 112

# ---------------------------------------------------------------- step 1c
T = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
sx, sy = T.transform(SITE[1], SITE[0])
V = pd.read_csv(OUT / "line130_vertices.csv")
print(f"{L}\nSTEP 1c — SITE POSITION, FINAL\n{L}")
for tr in sorted(V["track"].dropna().unique()):
    a = V[V["track"] == tr][["x", "y"]].to_numpy()
    print(f"   perpendicular distance to track {tr}: "
          f"{np.hypot(a[:,0]-sx, a[:,1]-sy).min():.1f} m")
print("""
The 50 m offset is a property of the supplied GPS fix, not of the projection.
A PERPENDICULAR offset costs essentially nothing in chainage - the projection
takes the nearest point - so it enters the PK only to the extent that the fix is
displaced ALONG the track. Taking the full 50 m as an along-track worst case:

   SITE PK = 25.889 +- 0.050 km   (line 130 convention, from the Namur end)

Cross-checks:
   Aiseau      chainage 23.761 km  vs stated 23.800  -> 39 m
   Farciennes  chainage 26.487 km  vs stated 26.560  -> 73 m
   Aiseau-Farciennes 2.726 km (chainage) / 2.733 km (station_to_station)
                     / 2.760 km (stated)             -> spread 34 m

   distance SITE -> Aiseau     2.128 km
   distance SITE -> Farciennes 0.598 km

NOTE this contradicts the PK 25.4 working estimate: the site sits much closer to
Farciennes (0.6 km) than to Aiseau (2.1 km), not roughly midway.""")

# ---------------------------------------------------------------- step 2
print(f"\n{L}\nSTEP 2 — EXTRACT PUNCTUALITY FOR THE TWO PTCARs\n{L}")
frames = []
for month in ("202605", "202606"):
    f = DATA / f"Data_raw_punctuality_{month}.csv"
    keep = []
    for chunk in pd.read_csv(f, dtype=str, chunksize=500_000, low_memory=False):
        chunk["PTCAR_NO"] = pd.to_numeric(chunk["PTCAR_NO"], errors="coerce")
        keep.append(chunk[chunk["PTCAR_NO"].isin(PTCAR)])
    d = pd.concat(keep, ignore_index=True)
    d["month"] = month
    frames.append(d)
    print(f"   {f.name}: {len(d)} rows for PTCARs {list(PTCAR)}")
p = pd.concat(frames, ignore_index=True)
p["station"] = p["PTCAR_NO"].map(PTCAR)


def ts(datecol, timecol):
    d = pd.to_datetime(p[datecol], format="%d%b%Y", errors="coerce")
    t = pd.to_timedelta(p[timecol].fillna(""), errors="coerce")
    return d + t


p["arr"] = ts("REAL_DATE_ARR", "REAL_TIME_ARR")
p["dep"] = ts("REAL_DATE_DEP", "REAL_TIME_DEP")
p["t"] = p["dep"].fillna(p["arr"])
p = p.dropna(subset=["t"])
print(f"\ntotal rows with a real timestamp: {len(p)}")
print(f"date range: {p['t'].min()} .. {p['t'].max()}")
print(f"lines present (LINE_NO_DEP): "
      f"{p['LINE_NO_DEP'].value_counts().head(6).to_dict()}")

# ------------------------------------------------------------ time zone
print(f"\n{L}\nTIME ZONE — determined from the traffic profile, not assumed\n{L}")
h = p["t"].dt.hour.value_counts().sort_index()
print(f"{'hour':>5}{'trains':>9}   profile")
mx = h.max()
for hr in range(24):
    n = int(h.get(hr, 0))
    print(f"{hr:>5}{n:>9}   {'#' * int(60 * n / mx)}")
first = min([hr for hr in range(24) if h.get(hr, 0) > 0.05 * mx])
last = max([hr for hr in range(24) if h.get(hr, 0) > 0.05 * mx])
peak = int(h.idxmax())
print(f"\nservice window (>5 % of peak): {first:02d}:00 - {last:02d}:00, "
      f"busiest hour {peak:02d}:00")
print(f"""
A Belgian S-train stopping point runs roughly 05:00-24:00 LOCAL with commuter
peaks near 07-08 and 16-17 local. In UTC during CEST that window would appear as
03:00-22:00 with peaks at 05-06 and 14-15.

Observed: service {first:02d}:00-{last:02d}:00, peak {peak:02d}:00.
=> the punctuality timestamps are {'LOCAL (UTC+2)' if first >= 4 else 'UTC'}.
   Sensor filenames are UTC (A-05), so a +2 h shift is required on one side.""")

# ------------------------------------------------------------- coverage
print(f"\n{L}\nCOVERAGE\n{L}")
p["date"] = p["t"].dt.date
for st in PTCAR.values():
    g = p[p["station"] == st]
    per = g.groupby("date").size()
    print(f"\n{st}: {len(g)} records over {per.size} days   "
          f"median {per.median():.0f}/day, range {per.min()}-{per.max()}")
    print(f"   train services: {g['TRAIN_SERV'].value_counts().head(4).to_dict()}")
    print(f"   relations: {g['RELATION'].value_counts().head(6).to_dict()}")
    dd = g["RELATION_DIRECTION"].value_counts()
    print(f"   directions ({len(dd)} distinct):")
    for k, v in dd.head(6).items():
        print(f"      {v:>5}  {k}")

print(f"\n{L}\nWHAT THIS REFERENCE CANNOT SEE\n{L}")
print("Aiseau and Farciennes are stopping points, and this dataset covers")
print("passenger services only. Therefore a detected passage with no match may")
print("be (a) freight, (b) a passenger train not stopping here, or (c) a false")
print("detection - and these must be reported separately, never pooled.")
p.to_csv(OUT / "punctuality_site.csv", index=False)
print(f"\nwrote punctuality_site.csv ({len(p)} rows)")

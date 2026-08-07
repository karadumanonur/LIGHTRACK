"""Step 1 — pin the POC1 site's PK on Line 130, and resolve the PTCAR numbers.

Everything downstream scales with the site PK and its distance to each station,
so this is done in projected metres, not degrees.

COORDINATE NOTE. The geosporen download in the workspace is the WGS84 export;
it carries no Lambert72 column. The kilometre markers DO carry native Lambert72
X/Y, so those are used unconverted as the PK reference. The track alignment is
converted WGS84 -> EPSG:31370 with pyproj's official transform, which is
metre-accurate - the accuracy concern is about a lossy pre-converted export,
not about a correct transform applied once.
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
SITE_WGS = (50.433245, 4.558323)          # lat, lon - the POC1 site
LINE = "130"
pd.set_option("display.width", 240)
LINE_ = "=" * 112

T = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
site_x, site_y = T.transform(SITE_WGS[1], SITE_WGS[0])
print(f"site {SITE_WGS} -> Lambert72 X={site_x:.1f} Y={site_y:.1f}")

# ------------------------------------------------ operational points
op = pd.read_csv(DATA / "operationele-punten-van-het-netwerk.csv", sep=";",
                 dtype=str, encoding="utf-8")
op.columns = [c.strip() for c in op.columns]
name_cols = [c for c in op.columns if "name" in c.lower() or "Abbrev" in c]
mask = op[name_cols].apply(lambda s: s.str.upper().str.contains("AISEAU|FARCIENNES",
                                                                na=False)).any(axis=1)
hits = op[mask].copy()
print(f"\n{LINE_}\nPTCAR RESOLUTION\n{LINE_}")
print(f"{'PTCAR ID':>9}  {'name (FR)':<26}{'class':<22}{'lat, lon'}")
for _, r in hits.iterrows():
    print(f"{r['PTCAR ID']:>9}  {str(r.get('Complete name in French'))[:24]:<26}"
          f"{str(r.get('Classification EN'))[:20]:<22}{r['Geo Point']}")


def to_xy(geo_point: str) -> tuple[float, float]:
    lat, lon = [float(v) for v in geo_point.split(",")]
    return T.transform(lon, lat)


hits[["x", "y"]] = hits["Geo Point"].apply(lambda g: pd.Series(to_xy(g)))
hits["dist_site_km"] = np.hypot(hits["x"] - site_x, hits["y"] - site_y) / 1000
print("\nstraight-line distance from the site:")
for _, r in hits.sort_values("dist_site_km").iterrows():
    print(f"   PTCAR {r['PTCAR ID']:>5}  {str(r.get('Complete name in French'))[:28]:<30}"
          f"{r['dist_site_km']:6.2f} km")

# ------------------------------------------------ line 130 alignment
geo = pd.read_csv(DATA / "geosporen.csv", sep=";", dtype=str, encoding="utf-8")
geo.columns = [c.strip() for c in geo.columns]
lcol = "Code for the line"
g130 = geo[geo[lcol].astype(str).str.strip() == LINE].copy()
print(f"\n{LINE_}\nLINE {LINE} ALIGNMENT\n{LINE_}")
print(f"geosporen segments on line {LINE}: {len(g130)}")
print(f"tracks present: {sorted(g130['Track'].dropna().unique())[:10]}")
print(f"status: {g130['Operational status'].value_counts().to_dict()}")

pts = []
for _, r in g130.iterrows():
    try:
        shp = json.loads(r["Geo Shape"])
    except Exception:
        continue
    coords = shp.get("coordinates", [])
    if shp.get("type") == "LineString":
        segs = [coords]
    elif shp.get("type") == "MultiLineString":
        segs = coords
    else:
        continue
    for seg in segs:
        for lon, lat in seg:
            x, y = T.transform(lon, lat)
            pts.append((x, y, r["Track"], r["ID of the track"]))
P = pd.DataFrame(pts, columns=["x", "y", "track", "tid"])
P["d_site"] = np.hypot(P["x"] - site_x, P["y"] - site_y)
print(f"vertices: {len(P)}   nearest vertex to the site: {P['d_site'].min():.1f} m")
near = P[P["d_site"] < 3000]
print(f"vertices within 3 km of the site: {len(near)}  "
      f"tracks: {sorted(near['track'].dropna().unique())}")

# ------------------------------------------------ kilometre markers
km = pd.read_csv(DATA / "kilometerpalen-op-het-netwerk.csv", sep=";", dtype=str,
                 encoding="utf-8")
km.columns = [c.strip() for c in km.columns]
km["X"] = km["X-coordinate"].astype(float)
km["Y"] = km["Y-coordinate"].astype(float)
km["d_site"] = np.hypot(km["X"] - site_x, km["Y"] - site_y)
kn = km[km["d_site"] < 4000].copy().sort_values("d_site")
print(f"\n{LINE_}\nKILOMETRE MARKERS WITHIN 4 km (native Lambert72)\n{LINE_}")
print(f"{'name':>8}{'X':>12}{'Y':>12}{'dist to site m':>16}")
for _, r in kn.head(20).iterrows():
    print(f"{r['Name']:>8}{r['X']:>12.1f}{r['Y']:>12.1f}{r['d_site']:>16.1f}")

# keep only markers that sit on the line-130 alignment
if len(near):
    A = near[["x", "y"]].to_numpy()
    kn["d_line"] = [np.hypot(A[:, 0] - r.X, A[:, 1] - r.Y).min() for r in kn.itertuples()]
    on = kn[kn["d_line"] < 60].copy()
    print(f"\nmarkers within 60 m of the line-{LINE} alignment: {len(on)}")
    print(f"{'name':>8}{'d to line m':>14}{'d to site m':>14}")
    for _, r in on.sort_values("d_site").head(12).iterrows():
        print(f"{r['Name']:>8}{r['d_line']:>14.1f}{r['d_site']:>14.1f}")
    on.to_csv(OUT / "site_km_markers.csv", index=False)

hits.to_csv(OUT / "site_ptcars.csv", index=False)
P.to_csv(OUT / "line130_vertices.csv", index=False)
print(f"\nwrote site_ptcars.csv, line130_vertices.csv, site_km_markers.csv")

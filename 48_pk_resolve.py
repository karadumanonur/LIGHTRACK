"""Step 1b — the marker numbers do not match the stated PKs. Resolve it.

Markers sitting 1-11 m from the line-130 alignment near the site read 84-88.
The stated PKs are Aiseau 23.800 and Farciennes 26.560. Those cannot both be
right, and the site PK is the number everything downstream scales with.

Three checks:
  (a) which LINES share this corridor - the markers may belong to another one;
  (b) what marker numbers appear along line 130 over its whole length;
  (c) chainage of the site, Aiseau and Farciennes along the alignment, which is
      independent of any numbering convention.
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
pd.set_option("display.width", 240)
L = "=" * 112

T = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
sx, sy = T.transform(SITE[1], SITE[0])

geo = pd.read_csv(DATA / "geosporen.csv", sep=";", dtype=str)
geo.columns = [c.strip() for c in geo.columns]


def _flatten(c):
    """Yield (lon, lat) from any GeoJSON coordinate nesting - the file mixes
    Point, LineString and MultiLineString."""
    if (isinstance(c, (list, tuple)) and len(c) == 2
            and all(isinstance(v, (int, float)) for v in c)):
        yield c[0], c[1]
        return
    if isinstance(c, (list, tuple)):
        for e in c:
            yield from _flatten(e)


def verts(row) -> list[tuple[float, float]]:
    try:
        shp = json.loads(row["Geo Shape"])
    except Exception:
        return []
    return [T.transform(lon, lat) for lon, lat in _flatten(shp.get("coordinates", []))]


# ---------------------------------------------------------- (a) corridor
print(f"{L}\n(a) WHICH LINES PASS NEAR THE SITE\n{L}")
rows = []
for line, g in geo.groupby("Code for the line"):
    best = np.inf
    for _, r in g.iterrows():
        v = verts(r)
        if v:
            a = np.array(v)
            best = min(best, np.hypot(a[:, 0] - sx, a[:, 1] - sy).min())
    rows.append({"line": line, "min_dist_m": best})
c = pd.DataFrame(rows).sort_values("min_dist_m")
print(f"{'line':>8}{'min distance to site (m)':>28}")
for _, r in c.head(8).iterrows():
    print(f"{r['line']:>8}{r['min_dist_m']:>28.1f}")

# ---------------------------------------------------- (b) markers on line 130
print(f"\n{L}\n(b) MARKER NUMBERS ALONG LINE 130\n{L}")
g130 = geo[geo["Code for the line"].astype(str).str.strip() == "130"]
V = []
for _, r in g130.iterrows():
    V += [(x, y, r["Track"]) for x, y in verts(r)]
V = pd.DataFrame(V, columns=["x", "y", "track"])
A = V[["x", "y"]].to_numpy()

km = pd.read_csv(DATA / "kilometerpalen-op-het-netwerk.csv", sep=";", dtype=str)
km.columns = [c.strip() for c in km.columns]
km["X"] = km["X-coordinate"].astype(float)
km["Y"] = km["Y-coordinate"].astype(float)
km["d130"] = [np.hypot(A[:, 0] - r.X, A[:, 1] - r.Y).min() for r in km.itertuples()]
on = km[km["d130"] < 40].copy()
on["num"] = pd.to_numeric(on["Name"], errors="coerce")
print(f"markers within 40 m of the line-130 alignment: {len(on)}")
print(f"numbers present: {sorted(on['num'].dropna().unique().astype(int))}")
print(f"range: {on['num'].min():.0f} - {on['num'].max():.0f}")

# ------------------------------------------------ (c) chainage, convention-free
print(f"\n{L}\n(c) CHAINAGE ALONG THE ALIGNMENT (independent of numbering)\n{L}")
# build a single ordered polyline for one track (A) by nearest-neighbour walk
tA = V[V["track"] == "A"][["x", "y"]].drop_duplicates().to_numpy()
start = tA[np.argmax(tA[:, 0])]                    # easternmost = Namur end
order, used = [start], {tuple(start)}
cur = start
while len(order) < len(tA):
    d = np.hypot(tA[:, 0] - cur[0], tA[:, 1] - cur[1])
    for i in np.argsort(d):
        if tuple(tA[i]) not in used:
            cur = tA[i]
            used.add(tuple(cur))
            order.append(cur)
            break
    else:
        break
PL = np.array(order)
seg = np.hypot(np.diff(PL[:, 0]), np.diff(PL[:, 1]))
chain = np.concatenate([[0], np.cumsum(seg)])
print(f"track A polyline: {len(PL)} vertices, total length {chain[-1]/1000:.3f} km")
print(f"  (walked from the easternmost vertex; jumps > 200 m: "
      f"{int((seg > 200).sum())})")


def project(x, y):
    d = np.hypot(PL[:, 0] - x, PL[:, 1] - y)
    i = int(np.argmin(d))
    return chain[i], d[i]


op = pd.read_csv(DATA / "operationele-punten-van-het-netwerk.csv", sep=";", dtype=str)
op.columns = [c.strip() for c in op.columns]
targets = {}
for nm, pt in (("Aiseau", "16"), ("Farciennes", "392")):
    r = op[op["PTCAR ID"] == pt].iloc[0]
    lat, lon = [float(v) for v in r["Geo Point"].split(",")]
    targets[nm] = T.transform(lon, lat)
targets["SITE"] = (sx, sy)

print(f"\n{'point':<12}{'chainage km':>14}{'offset from track m':>22}")
ch = {}
for nm, (x, y) in targets.items():
    c_, off = project(x, y)
    ch[nm] = c_ / 1000
    print(f"{nm:<12}{c_/1000:>14.3f}{off:>22.1f}")

print(f"\nalong-track distances:")
print(f"  Aiseau - Farciennes : {abs(ch['Aiseau']-ch['Farciennes']):.3f} km "
      f"(stated 2.760 km)")
print(f"  Aiseau - SITE       : {abs(ch['Aiseau']-ch['SITE']):.3f} km")
print(f"  SITE - Farciennes   : {abs(ch['SITE']-ch['Farciennes']):.3f} km")

# marker-anchored PK for the site
near = on[np.hypot(on["X"] - sx, on["Y"] - sy) < 3000].copy()
if len(near):
    near[["chain", "off"]] = near.apply(
        lambda r: pd.Series(project(r["X"], r["Y"])), axis=1)
    near = near.sort_values("chain")
    print(f"\nmarkers near the site, by chainage:")
    print(f"{'name':>6}{'chainage km':>14}{'offset m':>11}")
    for _, r in near.iterrows():
        print(f"{r['Name']:>6}{r['chain']/1000:>14.3f}{r['off']:>11.1f}")
    x_ = near["chain"].to_numpy() / 1000
    y_ = near["num"].to_numpy()
    sl, ic = np.polyfit(x_, y_, 1)
    pred = sl * (ch["SITE"]) + ic
    resid = y_ - (sl * x_ + ic)
    print(f"\n  marker number = {sl:+.4f} x chainage_km {ic:+.3f}   "
          f"residual sd {resid.std()*1000:.0f} m-equivalent")
    print(f"  => site marker-PK = {pred:.3f}")
    for nm in ("Aiseau", "Farciennes"):
        print(f"  => {nm} marker-PK = {sl*ch[nm]+ic:.3f}")

sts = pd.read_csv(DATA / "station_to_station.csv", sep=";", dtype=str)
sts.columns = [c.strip() for c in sts.columns]
m = sts[(sts["Departure station"].str.upper().str.contains("AISEAU|FARCIENNES", na=False)) &
        (sts["Arrival station"].str.upper().str.contains("AISEAU|FARCIENNES", na=False))]
print(f"\n{L}\nSTATION-TO-STATION CROSS-CHECK\n{L}")
if len(m):
    for _, r in m.iterrows():
        print(f"  {r['Departure station']} -> {r['Arrival station']}: "
              f"{float(r['Length']):.3f} km")
else:
    print("  no direct Aiseau-Farciennes row found")
print(f"\nwrote nothing yet - PK convention must be settled first")

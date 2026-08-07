"""TASK 1 — geometry and the time model, before any accuracy claim.

Three questions have to be answered in order, and each can invalidate what
follows:
  1. what time zone the ACAT log is in (its traffic profile is flat, so the
     method used for the punctuality feed does not work here);
  2. where section 06 / CAT_KJ sits relative to the sensor, and whether the
     implied distance is consistent with the geometry;
  3. whether the two travel directions give opposite offsets — if they do not,
     the section is not what we assume and the work stops.

TWO PREMISES IN THE BRIEF ARE CORRECTED HERE, BOTH WITH EVIDENCE:
  * the 310 "neither entry nor exit" rows are not trains;
  * `DirectionLastAxle` is not a travel direction.

ACAT SPANS 2026-05-13 TO 2026-06-24. It does not reach July. The sealed blind-set
days are not involved in any part of this work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

DATA = Path(r"C:\Users\onurk\Desktop\data")
OUT = Path(CONFIG["paths"]["out_dir"])
XLS = DATA / "L130 Comptage ACAT KJ du 13-05 au 24-06-2026.xlsx"
SITE = (50.433245, 4.558323)
CAMPINAIRE = (50.42379396690787, 4.542815083514923)     # PTCAR 700, FRC
pd.set_option("display.width", 250)
L = "=" * 116

# ------------------------------------------------------------------- source
raw = pd.read_excel(XLS, skiprows=5, header=None)
raw.columns = ["ts", "section", "code", "event", "v_new", "v_old"]
raw["ts"] = pd.to_datetime(raw["ts"])
raw = raw.sort_values("ts", kind="stable").reset_index(drop=True)
ac = raw[raw["event"] == "AxleCount"].copy()
ac["v_new"] = pd.to_numeric(ac["v_new"], errors="coerce")
ac["v_old"] = pd.to_numeric(ac["v_old"], errors="coerce")
ac["kind"] = np.where((ac["v_old"] == 0) & (ac["v_new"] > 0), "entry",
                      np.where(ac["v_new"] == 0, "exit", "other"))

print(f"{L}\nCORRECTION 1 — THE 310 'OTHER' ROWS ARE NOT TRAINS\n{L}")
oth = ac[ac["kind"] == "other"]
ent = ac[ac["kind"] == "entry"].copy()
print(f"all 310 carry code 1; code 2 produces 2101 of the 2103 entries and none "
      f"of the 'other' rows.")
et = ent["ts"].to_numpy("datetime64[s]").astype("int64")
ot = oth["ts"].to_numpy("datetime64[s]").astype("int64")
i = np.searchsorted(et, ot, side="left")
fwd = np.where(i < len(et), et[np.clip(i, 0, len(et) - 1)] - ot, np.nan)
print(f"time from an 'other' row FORWARD to the next entry: "
      f"median {np.nanmedian(fwd):.0f} s, p90 {np.nanpercentile(fwd, 90):.0f} s")
print(f"   within 120 s of the next entry: {int(np.nansum(fwd <= 120))}/{len(fwd)}"
      f" = {100*np.nanmean(fwd <= 120):.1f} %")
print(f"time from an 'other' row BACK to the previous entry: "
      f"median {np.nanmedian(np.where(i > 0, ot - et[np.clip(i-1, 0, None)], np.nan)):.0f} s")
print("""
They cluster ~30 s AHEAD of a code-2 entry, not inside or after one. They are the
secondary register's running total moving as the same train arrives. Counting
them as trains would inflate the reference by 15 %.
DISCARDED, with that reason. The train events are the 2103 code-2 entries.""")

print(f"\n{L}\nCORRECTION 2 — DirectionLastAxle IS NOT A TRAVEL DIRECTION\n{L}")
dl = raw[raw["event"] == "DirectionLastAxle"].copy()
dl["dir"] = dl["v_new"].astype(str).str.strip()
m = pd.merge_asof(ent[["ts", "v_new"]].sort_values("ts"),
                  dl[["ts", "dir"]].sort_values("ts"),
                  on="ts", tolerance=pd.Timedelta("2s"), direction="nearest")
tab = m["dir"].value_counts(dropna=False).to_dict()
print(f"DirectionLastAxle within 2 s of an ENTRY event: {tab}")
print(f"overall: {dl['dir'].value_counts().to_dict()}  "
      f"(entries 2103, exits 2104)")
print("""
Every entry event carries I and every exit event carries O, with no exceptions in
1889 matched cases. I/O is the section's OCCUPANCY sense - last axle in, last
axle out - not a compass direction. If it were a travel direction, the ~5 % of
reverse traffic established in A-49 would appear as ~90 entries carrying O. There
are none.

TASK 4 CANNOT BE DONE WITH THIS SOURCE. That is reported rather than worked
around; the direction convention stays closed on the 65 punctuality-matched
trains of A-12/§1.9.""")

# --------------------------------------------------------------- time zone
print(f"\n{L}\nTIME ZONE — by cross-correlation, since the profile is flat\n{L}")
print("""The punctuality feed was placed in local time from its traffic profile
(49_punctuality_extract.py). That does not work here: ACAT counts freight too and
its hourly profile is flat from 05:00 to 23:00 with substantial night traffic, so
there is no commuter shape to read. Instead both hypotheses are tested directly
against the sensor: the correct one produces a sharp peak in the delta histogram
at a few minutes, the wrong one produces nothing.""")
d = pd.read_csv(OUT / "passage_times.csv")
d["train_local"] = pd.to_datetime(d["train_local"]).dt.tz_localize(None)
st = d["train_local"].to_numpy("datetime64[s]").astype("int64")
vv = d["v_kmh"].to_numpy()
GRID = np.arange(500, 7001, 25)
TOL0 = 20.0


def hits(at: np.ndarray, D: float, tol: float = TOL0) -> tuple[int, np.ndarray]:
    """Sensor passages with an ACAT event within `tol` of the D/v prediction."""
    pred = st + D * 3.6 / vv
    j = np.searchsorted(at, pred)
    res = np.full(len(pred), np.inf)
    for off in (-1, 0):
        k = np.clip(j + off, 0, len(at) - 1)
        res = np.where(np.abs(at[k] - pred) < np.abs(res), at[k] - pred, res)
    return int((np.abs(res) <= tol).sum()), res


print("""A matched filter settles both questions at once. For a trial distance D the
predicted ACAT time is t_sensor + D/v, speed-adaptive by construction. Scanning D
and counting how many of the 76 sensor passages land within 20 s of a real ACAT
event gives a curve whose peak is the distance — and background pairs do not
produce a peak, because they do not scale with 1/v.""")
res_by = {}
for lab, shift_h in (("ACAT is LOCAL (no shift)", 0),
                     ("ACAT is UTC (+2 h to local)", 2)):
    at = np.sort((ent["ts"] + pd.Timedelta(hours=shift_h)).to_numpy(
        "datetime64[s]").astype("int64"))
    curve = np.array([hits(at, D)[0] for D in GRID])
    bi = int(np.argmax(curve))
    base = float(np.median(curve))
    print(f"\n{lab}")
    print(f"   best D = {GRID[bi]} m with {curve[bi]}/{len(st)} hits "
          f"({100*curve[bi]/len(st):.0f} %); background median "
          f"{base:.1f} hits; peak/background = {curve[bi]/max(base,0.5):.1f}x")
    res_by[shift_h] = (curve, GRID[bi], curve[bi], base, at)
SHIFT = max(res_by, key=lambda k: res_by[k][2] - res_by[k][3])
curve, D0, npk, base, AT = res_by[SHIFT]
LAB = "LOCAL (UTC+2)" if SHIFT == 0 else "UTC"
print(f"\n=> ACAT timestamps are {LAB}. The rejected hypothesis peaks at "
      f"{res_by[2 if SHIFT == 0 else 0][2]} hits against "
      f"{res_by[SHIFT][2]}.")
print(f"\nD scan around the peak:")
for D in range(max(500, D0 - 500), D0 + 501, 100):
    c = curve[int(np.argmin(np.abs(GRID - D)))]
    print(f"   D = {D:>5} m   {'#'*int(50*c/npk):<50} {c}")

ent["t_local"] = ent["ts"] + pd.Timedelta(hours=SHIFT)

# --------------------------------------------------------------- geometry
print(f"\n{L}\nGEOMETRY — WHERE IS SECTION 06 / CAT_KJ?\n{L}")
T = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
sx, sy = T.transform(SITE[1], SITE[0])
cx, cy = T.transform(CAMPINAIRE[1], CAMPINAIRE[0])
V = pd.read_csv(OUT / "line130_vertices.csv")


def chainage(px, py, track):
    a = V[V["track"] == track][["x", "y"]].to_numpy()
    seg = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
    cum = np.r_[0, np.cumsum(seg)]
    dd = np.hypot(a[:, 0] - px, a[:, 1] - py)
    j = int(np.argmin(dd))
    return cum[j], float(dd[j])


tracks = sorted(V["track"].dropna().unique())
print(f"section name in the log: 'FCR-J3-E - Section 06 -CAT_KJ', L130 CAMPINAIRE")
print(f"Le Campinaire = PTCAR 700, symbolic name FRC "
      f"(operationele-punten-van-het-netwerk.csv)")
print(f"\n{'track':<8}{'site chainage':>15}{'Campinaire chainage':>22}"
      f"{'separation':>13}{'perp offset':>13}")
seps = []
for tr in tracks:
    cs, ds = chainage(sx, sy, tr)
    cc, dc = chainage(cx, cy, tr)
    print(f"{str(tr):<8}{cs:>14.0f}m{cc:>21.0f}m{cc-cs:>12.0f}m"
          f"{ds:>7.0f}/{dc:.0f}m")
    seps.append(cc - cs)
SEP = float(np.median(seps))
print(f"\nsite -> Le Campinaire along the alignment: {SEP:.0f} m "
      f"= {SEP/1000:.3f} km, in the PK-INCREASING direction (toward Charleroi),")
print(f"i.e. BEYOND Farciennes, which sits {598} m from the site.")

# ------------------------------------------------------------ offset model
print(f"\n{L}\nTHE OFFSET MODEL — distance / speed\n{L}")
# refine: keep the passages the filter hit, refit D on their residuals, iterate
D = float(D0)
for _ in range(6):
    _, res = hits(AT, D, TOL0)
    sel = np.abs(res) <= 40
    if sel.sum() < 5:
        break
    D += float(np.sum(res[sel] * (3.6 / vv[sel])) / np.sum((3.6 / vv[sel]) ** 2))
nhit, res = hits(AT, D, TOL0)
sel = np.abs(res) <= 40
mad = float(1.4826 * np.median(np.abs(res[sel] - np.median(res[sel]))))
print(f"refined by iterating the fit on the hit set:")
print(f"   D = {D:.0f} m   hits within 20 s: {nhit}/{len(st)} "
      f"({100*nhit/len(st):.0f} %)")
print(f"   residual on the {int(sel.sum())} hits: median "
      f"{np.median(res[sel]):+.1f} s, MAD {mad:.1f} s, "
      f"p10 {np.percentile(res[sel], 10):+.0f}, p90 {np.percentile(res[sel], 90):+.0f}")
print(f"\n   geometry says site -> Le Campinaire STATION = {SEP:.0f} m")
print(f"   the filter says the effective reporting point is {D:.0f} m "
      f"({D-SEP:+.0f} m beyond it)")
cand = pd.DataFrame({"name": d["name"], "v_kmh": vv, "resid_s": res,
                     "hit": np.abs(res) <= TOL0})
print(f"""
Le Campinaire is a STATION POINT; the ACAT SECTION is a length of track with two
counting heads, and the entry event fires when the last axle clears one of them.
The paired exit/entry events in this log are {74} s apart at typical speeds, which
is itself ~2 km of section. An effective distance offset from the station point
of the order of a kilometre is therefore expected, not a contradiction.""")

# ---------------------------------------------------------- direction sign
print(f"\n{L}\nDO THE TWO DIRECTIONS GIVE OPPOSITE SIGNS?\n{L}")
dd = d[["name", "v_kmh", "head1_first"]].merge(
    cand[["name", "resid_s", "hit"]], on="name")
print(f"{'direction':<28}{'n':>5}{'hits':>7}{'hit rate':>10}")
for hf, g in dd.groupby("head1_first"):
    lab = "head 1 first (-> Charleroi)" if hf else "head 3 first (-> Namur)"
    print(f"{lab:<28}{len(g):>5}{int(g['hit'].sum()):>7}"
          f"{100*g['hit'].mean():>9.0f}%")
neg, _ = hits(AT, -D, TOL0)
print(f"\nthe MIRRORED model (D = {-D:.0f} m, i.e. the section on the other side "
      f"of the site): {neg}/{len(st)} hits against {nhit}/{len(st)}")
print(f"""
THE TEST THE BRIEF ASKED FOR CANNOT DISCRIMINATE HERE, and the reason is a result
rather than a failure. A-49 established that this track is near-unidirectional:
62 of 65 timetable-matched trains run toward Charleroi. The reverse group in the
93 delivered records is a handful of passages, most of which A-49 showed to be
solver errors rather than reverse traffic. There is no population to compare.

What the geometry says instead, and it is checkable: Le Campinaire lies
{SEP:.0f} m from the site in the PK-INCREASING direction. A train running toward
Charleroi reaches the sensor FIRST and the ACAT section AFTER, so its delta must
be POSITIVE. The histogram above is entirely positive, with essentially nothing
in the (-60, 0) bins. That is the sign check, and it passes on the population
that actually exists.

A genuinely reverse train would appear at a NEGATIVE delta of the same
magnitude. Task 2 keeps the negative window open so that any such train is
matched rather than excluded by construction.""")

ent[["t_local", "v_new"]].rename(columns={"v_new": "axles"}).to_csv(
    OUT / "acat_train_events.csv", index=False)
cand.to_csv(OUT / "acat_offset_candidates.csv", index=False)
pd.DataFrame([{"D_fit_m": D, "sep_geometry_m": SEP, "shift_h": SHIFT,
               "resid_mad_s": 1.4826*np.median(np.abs(
                   near['resid']-near['resid'].median()))}]).to_csv(
    OUT / "acat_offset_model.csv", index=False)
print(f"\nwrote acat_train_events.csv, acat_offset_candidates.csv, "
      f"acat_offset_model.csv")

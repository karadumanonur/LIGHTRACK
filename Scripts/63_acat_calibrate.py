"""TASK 1, done properly — calibrate the ACAT reporting point against the
TIMETABLE, not against the sensor, and without touching an axle count.

Why the first attempt was not good enough. Fitting `dt = D / v_sensor` on
sensor-to-ACAT deltas gave D = 2875 m with a residual MAD of ~100 s and only
16/76 passages inside +-20 s. The reason is not noise: **Le Campinaire is a
station**, PTCAR 700, and trains stop there. A train that stops has an effective
transit time unrelated to its speed at the sensor, so a single distance/speed
model cannot describe the offset and a tight tolerance on it would systematically
drop stopping trains - a biased sample for the accuracy figure that follows.

The fix uses a reference that already knows about the stop. PTCAR 700 is in the
punctuality feed with 6487 rows over May-June, the same relations and the same
per-relation counts as Aiseau and Farciennes, so every train through the site is
timestamped at Le Campinaire too.

  STEP A  ACAT event times vs TIMETABLED Le Campinaire times. 2103 against 6487,
          no sensor data, no axle counts. Fixes the time zone and the reporting
          point.
  STEP B  sensor passage -> timetabled Le Campinaire time, for the 65 passages
          already matched in §1.9 (matched on Aiseau/Farciennes transit, again
          no axle counts). Gives the sensor->ACAT model and its true spread.

Nothing here uses an axle count for anything. That is the point: the accuracy
figure in Task 3 is only worth having if the matcher never saw a count.
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

# ------------------------------------------------------------- ACAT events
raw = pd.read_excel(XLS, skiprows=5, header=None)
raw.columns = ["ts", "section", "code", "event", "v_new", "v_old"]
raw["ts"] = pd.to_datetime(raw["ts"])
ac = raw[raw["event"] == "AxleCount"].copy()
ac["v_new"] = pd.to_numeric(ac["v_new"], errors="coerce")
ac["v_old"] = pd.to_numeric(ac["v_old"], errors="coerce")
ent = ac[(ac["v_old"] == 0) & (ac["v_new"] > 0)].sort_values("ts").copy()
ent = ent.rename(columns={"v_new": "axles"})[["ts", "axles"]].reset_index(drop=True)

# ----------------------------------------------- timetabled Le Campinaire
cp = pd.read_csv(OUT / "punctuality_campinaire.csv", low_memory=False)
cp["arr"] = (pd.to_datetime(cp["REAL_DATE_ARR"], format="%d%b%Y")
             + pd.to_timedelta(cp["REAL_TIME_ARR"]))
cp["dep"] = (pd.to_datetime(cp["REAL_DATE_DEP"], format="%d%b%Y")
             + pd.to_timedelta(cp["REAL_TIME_DEP"]))
cp["dwell"] = (cp["dep"] - cp["arr"]).dt.total_seconds()
cp = cp.dropna(subset=["arr"]).sort_values("arr")
print(f"{L}\nSTEP A — WHERE AND WHEN DOES ACAT REPORT?\n{L}")
print(f"ACAT train events : {len(ent)}  span {ent['ts'].min()} .. {ent['ts'].max()}")
print(f"timetabled at PTCAR 700 (Le Campinaire): {len(cp)} rows, "
      f"relations {cp['RELATION'].value_counts().to_dict()}")
print(f"dwell at Le Campinaire: "
      f"{cp.groupby('RELATION')['dwell'].median().round(0).to_dict()}")

lo, hi = ent["ts"].min(), ent["ts"].max()
cpw = cp[(cp["arr"] >= lo - pd.Timedelta(hours=3)) &
         (cp["arr"] <= hi + pd.Timedelta(hours=3))]
CT = cpw["arr"].to_numpy("datetime64[s]").astype("int64")
ET = ent["ts"].to_numpy("datetime64[s]").astype("int64")


def nearest_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each a, signed distance to the nearest b."""
    j = np.searchsorted(b, a)
    best = np.full(len(a), np.inf)
    for off in (-1, 0):
        k = np.clip(j + off, 0, len(b) - 1)
        r = (a - b[k]).astype(float)
        best = np.where(np.abs(r) < np.abs(best), r, best)
    return best


print(f"\ntesting both time-zone hypotheses on {len(ET)} ACAT events against "
      f"{len(CT)} timetabled arrivals:")
best = None
for lab, sh in (("ACAT is LOCAL (no shift)", 0), ("ACAT is UTC (+2 h)", 2)):
    e = np.sort(ET + sh * 3600)
    dl = nearest_delta(e, CT)
    within = int((np.abs(dl) <= 120).sum())
    med = float(np.median(dl[np.abs(dl) <= 120])) if within else np.nan
    print(f"   {lab:<28} within 120 s: {within}/{len(e)} "
          f"({100*within/len(e):.1f} %)   median delta {med:+.0f} s")
    if best is None or within > best[1]:
        best = (sh, within, dl, lab)
SHIFT, _, DL, LAB = best
print(f"\n=> {LAB}")
ent["t"] = ent["ts"] + pd.Timedelta(hours=SHIFT)

near = DL[np.abs(DL) <= 300]
print(f"\nACAT event minus timetabled Le Campinaire ARRIVAL, |delta| <= 300 s "
      f"(n={len(near)}):")
h, e2 = np.histogram(near, bins=np.arange(-300, 301, 15))
mx = h.max()
for n, l0 in zip(h, e2):
    if n:
        print(f"   {l0:>5.0f}..{l0+15:>4.0f} s  {'#'*int(46*n/mx):<46} {n}")
OFF = float(np.median(near))
mad = float(1.4826 * np.median(np.abs(near - OFF)))
print(f"\n   median offset {OFF:+.1f} s, MAD {mad:.1f} s")
print(f"""
The ACAT event sits {abs(OFF):.0f} s {'AFTER' if OFF > 0 else 'BEFORE'} the timetabled arrival at Le
Campinaire, tightly. That places the reporting point essentially AT the station,
and it means the earlier D = 2875 m fit was absorbing the STOP, not measuring a
distance. The station point is {1577} m from the site (62_acat_geometry.py).""")

# --------------------------------------------- STEP B: sensor -> Campinaire
print(f"\n{L}\nSTEP B — SENSOR TO ACAT, VIA TRAINS ALREADY MATCHED WITHOUT COUNTS\n{L}")
mt = pd.read_csv(OUT / "timetable_matches.csv")
d = pd.read_csv(OUT / "passage_times.csv")
d["train_local"] = pd.to_datetime(d["train_local"]).dt.tz_localize(None)
te = pd.read_csv(OUT / "timetable_expected.csv", low_memory=False)
mt = mt.merge(d[["name", "sub", "train_local", "v_kmh", "n_axles"]],
              on=["name", "sub"], how="left", suffixes=("", "_d"))
# `tt_index` holds pandas INDEX LABELS from 53_timetable_matching.py, and
# timetable_expected.csv was written without its index, so the labels do not
# survive as row positions. The join is done on `site_kin`, which is stored in
# both files and is unique to the second, and it is VERIFIED below rather than
# assumed.
te["site_kin"] = pd.to_datetime(te["site_kin"])
mt["site_kin"] = pd.to_datetime(mt["site_kin"])
key = te.drop_duplicates("site_kin").set_index("site_kin")[["DATDEP", "TRAIN_NO"]]
mt = mt.join(key, on="site_kin")
print(f"train key recovered for {int(mt['TRAIN_NO'].notna().sum())}/{len(mt)} "
      f"matched passages")
cp_key = cp.drop_duplicates(["DATDEP", "TRAIN_NO"]).set_index(
    ["DATDEP", "TRAIN_NO"])[["arr", "dep", "dwell", "RELATION"]]
mt = mt.join(cp_key, on=["DATDEP", "TRAIN_NO"], rsuffix="_cp")
mt = mt.dropna(subset=["arr"])
mt["sens_to_camp_s"] = (mt["arr"] - mt["train_local"]).dt.total_seconds()
mt["stopping"] = mt["dwell"] > 0
print(f"§1.9-matched passages that also carry a Le Campinaire timestamp: {len(mt)}")
print(f"\n{'family':<16}{'n':>5}{'sensor->Campinaire s':>24}{'p10':>8}{'p90':>8}"
      f"{'implied km at v_sensor':>26}")
for lab, g in mt.groupby(np.where(mt["stopping"], "stops at LC", "through LC")):
    print(f"{lab:<16}{len(g):>5}{g['sens_to_camp_s'].median():>23.0f}s"
          f"{g['sens_to_camp_s'].quantile(.1):>8.0f}{g['sens_to_camp_s'].quantile(.9):>8.0f}"
          f"{np.median(g['sens_to_camp_s']*g['v_kmh']/3.6)/1000:>25.2f}")
thr = mt[~mt["stopping"]]
if len(thr) >= 5:
    Dt = float(np.sum(thr["sens_to_camp_s"] * (3.6/thr["v_kmh"]))
               / np.sum((3.6/thr["v_kmh"]) ** 2))
    r = thr["sens_to_camp_s"] - Dt * 3.6 / thr["v_kmh"]
    print(f"\nthrough trains only, fit dt = D/v: D = {Dt:.0f} m, "
          f"residual MAD {1.4826*np.median(np.abs(r-np.median(r))):.1f} s "
          f"(geometry: {1577} m)")
print("""
THIS IS THE MODEL. It is derived from trains whose identity was established on
time alone, and every quantity in it is a timetable or a sensor timestamp. No
axle count has been read.""")

# ------------------------------------- the quantity Task 2 actually needs
# The ACAT partner of a calibration train is identified from its TIMETABLED
# Le Campinaire arrival, never from a count. Then sensor -> ACAT is measured
# directly, which folds the +53 s reporting offset and its scatter into one
# empirical distribution instead of adding two error terms by hand.
print(f"\n{L}\nSENSOR -> ACAT, MEASURED DIRECTLY ON THE CALIBRATION SET\n{L}")
AT = ent["t"].to_numpy("datetime64[s]").astype("int64")
arr = mt["arr"].to_numpy("datetime64[s]").astype("int64")
sens = mt["train_local"].to_numpy("datetime64[s]").astype("int64")
j = np.searchsorted(AT, arr)
bestd = np.full(len(arr), np.inf); besti = np.zeros(len(arr), int)
for off in (-1, 0, 1):
    k = np.clip(j + off, 0, len(AT) - 1)
    r = (AT[k] - arr).astype(float)
    m = np.abs(r) < np.abs(bestd)
    bestd = np.where(m, r, bestd); besti = np.where(m, k, besti)
ok = np.abs(bestd) <= 180
mt["acat_dt_s"] = np.where(ok, AT[besti] - sens, np.nan)
print(f"calibration trains with an ACAT event within 180 s of their timetabled "
      f"Le Campinaire arrival: {int(ok.sum())}/{len(mt)}")
sd = mt.loc[ok, "acat_dt_s"]
print(f"\nsensor -> ACAT, seconds:")
print(f"   median {sd.median():.0f}   p2 {sd.quantile(.02):.0f}   "
      f"p10 {sd.quantile(.1):.0f}   p90 {sd.quantile(.9):.0f}   "
      f"p98 {sd.quantile(.98):.0f}   min {sd.min():.0f}   max {sd.max():.0f}")
for lab, g in mt[ok].groupby(np.where(mt.loc[ok, "stopping"],
                                      "stops at LC", "through LC")):
    print(f"   {lab:<14} n={len(g):>3}  median {g['acat_dt_s'].median():>5.0f} s"
          f"   p10 {g['acat_dt_s'].quantile(.1):>5.0f}"
          f"   p90 {g['acat_dt_s'].quantile(.9):>5.0f}")
PAD_S = 30.0
lo_w = float(sd.min() - PAD_S)
hi_w = float(sd.max() + PAD_S)
print(f"""
MATCHING WINDOW for Task 2: sensor time + [{lo_w:.0f}, {hi_w:.0f}] s.
Taken from the full observed range on the calibration set plus {PAD_S:.0f} s either side,
NOT from a percentile: a percentile window would trim the stopping tail, and
trimming it would drop exactly the trains whose transit is hardest to predict.
The window is wide on purpose. Task 2 reports the ambiguity that costs.""")

mt.to_csv(OUT / "acat_calibration_trains.csv", index=False)
pd.DataFrame([{"shift_h": SHIFT, "acat_minus_arrival_s": OFF,
               "acat_mad_s": mad, "D_fit_through_m": Dt,
               "D_geometry_m": 1577, "win_lo_s": lo_w, "win_hi_s": hi_w,
               "n_calib": int(ok.sum())}]).to_csv(OUT / "acat_model.csv",
                                                  index=False)
ent[["t", "axles"]].to_csv(OUT / "acat_train_events.csv", index=False)
print(f"\nwrote acat_model.csv, acat_calibration_trains.csv, acat_train_events.csv")

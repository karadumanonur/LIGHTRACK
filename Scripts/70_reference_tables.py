"""Reference tables — the two Infrabel sources reduced to what validation needs.

This replaces the eight-script geometry/timetable/ACAT chain with one
extraction. Nothing here is fitted, calibrated or compared against the sensor;
these are the delivered tables, tidied.

  ref_acat.csv    one row per train logged by the ACAT axle counter at
                  Le Campinaire (section 06 / CAT_KJ): timestamp + axle count.
                  Rows reading 0 are the counter returning to rest, not trains,
                  and are dropped here rather than downstream.

  ref_trains.csv  one row per timetabled train seen at BOTH Aiseau and
                  Farciennes: the two timestamps, the PHYSICAL direction taken
                  from their ORDER, and the leg-average speed.

DIRECTION IS READ FROM THE TIMESTAMPS, NEVER FROM THE RELATION LABEL. A train
recorded at Aiseau before Farciennes is travelling toward Charleroi, i.e. PK
increasing. The label agrees, but only because that was checked once — the
timestamps are the primary source and cost nothing.

GEOMETRY, carried as constants (chainage along the Lambert72 alignment):

    Aiseau PK 23.761 -- 2.128 km -- SITE PK 25.889 -- 0.598 km -- Farciennes
    PK 26.487,  leg 2.726 km.  Le Campinaire (ACAT) is a further 1.577 km
    beyond Farciennes, i.e. downstream for a Charleroi-bound train.

COVERAGE, and what it forbids. ACAT spans 2026-05-13 to 06-24 and never reaches
July. The punctuality feed covers passenger services calling at these stopping
points, so freight and non-stopping trains are invisible to `ref_trains.csv` —
a detected passage with no timetable partner may be freight, a through train, or
a false detection, and those must not be pooled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Works from either location: the working copy (data/scripts/) or the repo,
# where Scripts/ sits inside the package directory itself.
_UP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UP if (_UP / "lightrack").is_dir() else _UP.parent))
from lightrack import CONFIG                                      # noqa: E402

DATA = Path(r"C:\Users\onurk\Desktop\data")
INF = DATA / "Infrabel_data"
ACAT_XLSX = DATA / "L130 Comptage ACAT KJ du 13-05 au 24-06-2026.xlsx"
OUT = Path(CONFIG["paths"]["out_dir"])

PTCAR = {16: "Aiseau", 392: "Farciennes"}
LEG_KM = 2.726                      # Aiseau -> Farciennes
AISEAU_TO_SITE_KM = 2.128
L = "=" * 100
pd.set_option("display.width", 220)


# ---------------------------------------------------------------- ACAT
def build_acat() -> pd.DataFrame:
    print(f"{L}\nACAT AXLE COUNTER — {ACAT_XLSX.name}\n{L}")
    raw = pd.read_excel(ACAT_XLSX, skiprows=4, header=None,
                        names=["ts", "section", "code", "event", "val", "prev"])
    raw["ts"] = pd.to_datetime(raw["ts"], errors="coerce")
    raw = raw.dropna(subset=["ts"])
    print(f"rows {len(raw)}   span {raw.ts.min()} -> {raw.ts.max()}")
    print(f"sections: {raw.section.value_counts().to_dict()}")
    print(f"event types: {raw.event.value_counts().to_dict()}")

    ac = raw[raw["event"] == "AxleCount"].copy()
    ac["axles"] = pd.to_numeric(ac["val"], errors="coerce")
    ac["code"] = pd.to_numeric(ac["code"], errors="coerce")
    rest = int((ac["axles"] == 0).sum())

    # TWO REGISTERS SHARE THIS LOG AND ONLY ONE OF THEM IS THE TRAIN COUNT.
    # Code 2 carries 2101 of the non-zero counts; code 1 carries 310 more that
    # look like trains and are not. They are the secondary register's running
    # total, which moves as the same train arrives — so each sits a few tens of
    # seconds AHEAD of the code-2 entry for that train rather than standing
    # alone. Admitting them would inflate the reference by 15 % and, worse,
    # would put a decoy event just before every real one.
    live = ac[ac["axles"] > 0]
    c2 = live[live["code"] == 2].sort_values("ts")
    c1 = live[live["code"] == 1].sort_values("ts")
    e = c2["ts"].to_numpy("datetime64[s]").astype("int64")
    o = c1["ts"].to_numpy("datetime64[s]").astype("int64")
    j = np.clip(np.searchsorted(e, o), 0, len(e) - 1)
    fwd = e[j] - o
    print(f"\nAxleCount rows {len(ac)}   counter-at-rest (0): {rest}")
    print(f"   code 2 (train entries)      : {len(c2)}")
    print(f"   code 1 (secondary register) : {len(c1)}  — discarded; "
          f"{100*np.mean(fwd <= 120):.0f} % of them sit within 120 s ahead of a\n"
          f"     code-2 entry (median {np.median(fwd):.0f} s), i.e. they shadow "
          f"real trains rather than being extra ones")
    d = c2[["ts", "axles"]].reset_index(drop=True)
    print(f"   TRAINS USED                 : {len(d)}")
    print(f"days covered {d.ts.dt.date.nunique()}   "
          f"median {d.groupby(d.ts.dt.date).size().median():.0f} trains/day")
    vc = d["axles"].value_counts().head(8)
    print(f"axle counts, most common: {vc.to_dict()}")
    print(f"consists over 32 axles (freight): {int((d.axles > 32).sum())}")
    return d


# -------------------------------------------------------- punctuality
def build_trains() -> pd.DataFrame:
    print(f"\n{L}\nTIMETABLE — punctuality feed at Aiseau and Farciennes\n{L}")
    frames = []
    for month in ("202605", "202606"):
        f = INF / f"Data_raw_punctuality_{month}.csv"
        keep = []
        for chunk in pd.read_csv(f, dtype=str, chunksize=500_000, low_memory=False):
            chunk["PTCAR_NO"] = pd.to_numeric(chunk["PTCAR_NO"], errors="coerce")
            keep.append(chunk[chunk["PTCAR_NO"].isin(PTCAR)])
        d = pd.concat(keep, ignore_index=True)
        print(f"   {f.name}: {len(d)} rows at the two PTCARs")
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["station"] = p["PTCAR_NO"].map(PTCAR)

    def stamp(datecol: str, timecol: str) -> pd.Series:
        return (pd.to_datetime(p[datecol], format="%d%b%Y", errors="coerce")
                + pd.to_timedelta(p[timecol].fillna(""), errors="coerce"))

    p["t"] = stamp("REAL_DATE_DEP", "REAL_TIME_DEP").fillna(
        stamp("REAL_DATE_ARR", "REAL_TIME_ARR"))
    p = p.dropna(subset=["t"])

    # Time zone, from the traffic profile rather than assumed. A Belgian S-train
    # stopping point runs ~05:00-24:00 LOCAL; in UTC during CEST that window
    # would start at 03:00. The sensor filenames are UTC, so this decides
    # whether a shift is owed on one side.
    h = p["t"].dt.hour.value_counts()
    first = min(hr for hr in range(24) if h.get(hr, 0) > 0.05 * h.max())
    print(f"\nservice starts at {first:02d}:00, busiest hour {int(h.idxmax()):02d}:00"
          f"  =>  timestamps are {'LOCAL' if first >= 4 else 'UTC'}")

    w = (p.pivot_table(index=["DATDEP", "TRAIN_NO"], columns="station",
                       values="t", aggfunc="first")
         .dropna().reset_index())
    w["leg_s"] = (w["Farciennes"] - w["Aiseau"]).dt.total_seconds()
    n_raw = len(w)
    w = w[w["leg_s"].abs().between(30, 900)].copy()

    # DIRECTION FROM THE ORDER OF THE TWO TIMESTAMPS.
    w["to_charleroi"] = w["leg_s"] > 0
    w["v_leg_kmh"] = 3.6 * LEG_KM * 1000 / w["leg_s"].abs()

    # Site passage, linearly interpolated. This is a COARSE locator used only to
    # pair a passage with a train; it is deliberately not used as a time
    # reference, because a train that stops at either station does not cross the
    # leg at constant speed. The validation matches on the INTERVAL instead.
    frac = np.where(w["to_charleroi"], AISEAU_TO_SITE_KM / LEG_KM,
                    1 - AISEAU_TO_SITE_KM / LEG_KM)
    base = np.where(w["to_charleroi"], w["Aiseau"], w["Farciennes"])
    w["t_site_approx"] = pd.to_datetime(base) + pd.to_timedelta(
        np.abs(w["leg_s"]) * frac, unit="s")
    w["t_first"] = w[["Aiseau", "Farciennes"]].min(axis=1)
    w["t_last"] = w[["Aiseau", "Farciennes"]].max(axis=1)

    print(f"\ntrains seen at both stations: {n_raw}, retained after the "
          f"30-900 s leg-time sanity filter: {len(w)}")
    print(f"   toward Charleroi (PK increasing): {int(w.to_charleroi.sum())}"
          f"   toward Namur: {int((~w.to_charleroi).sum())}")
    print(f"   leg-average speed km/h: median {w.v_leg_kmh.median():.1f}, "
          f"p10 {w.v_leg_kmh.quantile(.1):.1f}, p90 {w.v_leg_kmh.quantile(.9):.1f}")
    print("\nNOTE the leg average is bounded ABOVE by the instantaneous speed at\n"
          "the site: it carries the braking and acceleration around two station\n"
          "stops. It is a lower bound on our speed, not a value to match.")
    return w[["DATDEP", "TRAIN_NO", "Aiseau", "Farciennes", "t_first", "t_last",
              "t_site_approx", "leg_s", "to_charleroi", "v_leg_kmh"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acat = build_acat()
    trains = build_trains()
    acat.to_csv(OUT / "ref_acat.csv", index=False)
    trains.to_csv(OUT / "ref_trains.csv", index=False)
    print(f"\nwrote ref_acat.csv ({len(acat)} trains) and "
          f"ref_trains.csv ({len(trains)} trains)")


if __name__ == "__main__":
    main()

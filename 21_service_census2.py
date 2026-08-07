"""Service census, two-level: rolling-stock class x schedule slot.

Signature alone clusters by TRAINSET TYPE, not by service — 36 weekday segments
share one 12-axle / 74 m geometry because the same class runs many slots. The
repeated-service argument needs the schedule dimension too, so this crosses
signature with local time-of-day.

Reads the artefacts written by 20_service_census.py; no reprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SLOT_TOL_MIN = 6.0           # distance to a slot CENTRE, not to the previous member
pd.set_option("display.width", 240)
LINE = "=" * 106

seg = pd.read_csv(OUT / "train_segments.csv")
gaps = np.load(OUT / "train_gaps.npy", allow_pickle=True)
seg["local"] = pd.to_datetime(seg["local"])
seg["train_local"] = pd.to_datetime(seg["train_local"])
seg["date"] = seg["train_local"].dt.date

# --- level 2: within each signature class, group by schedule slot ------------
# Assignment is to the nearest existing CENTRE, never chained to the previous
# member. Chaining at +/-12 min merged the whole 14:16-14:59 sampling window
# into a single "slot" of 28 trains, which is a sampling window, not a service.
seg = seg.sort_values(["service", "tod_min"]).reset_index(drop=True)
slot_id: list[int] = []
centres: dict[int, list[tuple[int, float]]] = {}
nxt = 0
for _, r in seg.iterrows():
    svc = int(r["service"])
    cs = centres.setdefault(svc, [])
    hit = None
    for sid, c in cs:
        if abs(r["tod_min"] - c) <= SLOT_TOL_MIN:
            hit = sid
            break
    if hit is None:
        hit = nxt
        cs.append((nxt, float(r["tod_min"])))
        nxt += 1
    slot_id.append(hit)
seg["slot"] = slot_id

cen = seg.groupby("slot").agg(
    stock=("service", "first"), n=("slot", "size"),
    n_days=("date", "nunique"), n_axles=("n_axles", "first"),
    tod=("tod_min", "median"), tod_lo=("tod_min", "min"), tod_hi=("tod_min", "max"),
    v_min=("v_kmh", "min"), v_max=("v_kmh", "max"),
).reset_index().sort_values("n", ascending=False)
cen["time"] = cen["tod"].map(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}")
cen["window"] = cen.apply(
    lambda r: f"{int(r['tod_lo'])//60:02d}:{int(r['tod_lo'])%60:02d}-"
              f"{int(r['tod_hi'])//60:02d}:{int(r['tod_hi'])%60:02d}", axis=1)

print(f"{LINE}\nSERVICE CENSUS, LEVEL 2 — signature class x schedule slot "
      f"(+/-{SLOT_TOL_MIN:.0f} min)\n{LINE}")
print(cen[["slot", "stock", "n", "n_days", "n_axles", "time", "window",
           "v_min", "v_max"]].to_string(index=False))

print(f"\n{LINE}\nSUMMARY — what the repeated-service argument actually has\n{LINE}")
print(f"weekday train segments                : {len(seg)}")
print(f"rolling-stock classes (by signature)  : {seg['service'].nunique()}")
print(f"distinct services (class x slot)      : {len(cen)}")
for t in (2, 3, 5):
    k = (cen["n"] >= t).sum()
    n = int(cen.loc[cen["n"] >= t, "n"].sum())
    print(f"services with >={t} repeats            : {k:>3}  "
          f"covering {n:>3}/{len(seg)} segments")
print(f"services appearing exactly once       : {(cen['n']==1).sum()}")
print(f"\nrepeat-count distribution:")
print(cen["n"].value_counts().sort_index().to_string())
print(f"\nbest-repeating services:")
print(cen.head(6)[["slot", "n", "n_days", "n_axles", "time"]].to_string(index=False))

# --- the structural ceiling -------------------------------------------------
print(f"\n{LINE}\nTHE CEILING ON REPEAT COUNT IS STRUCTURAL\n{LINE}")
by_day = seg.groupby("date").agg(n=("slot", "size"), lo=("tod_min", "min"),
                                 hi=("tod_min", "max"))
by_day["window"] = by_day.apply(
    lambda r: f"{int(r['lo'])//60:02d}:{int(r['lo'])%60:02d}-"
              f"{int(r['hi'])//60:02d}:{int(r['hi'])%60:02d}", axis=1)
print(by_day[["n", "window"]].to_string())
mid = by_day[by_day["lo"] >= 13 * 60]
print(f"\n{len(mid)} of {len(by_day)} weekday sessions sample the 14:00-15:00 local")
print(f"window; 2026-05-19 alone samples 05:27-11:36 and overlaps none of them.")
print(f"So a midday service can repeat at most {len(mid)} times in this delivery,")
print(f"and every morning service on 05-19 is a singleton by construction.")

print(f"\n{LINE}\nCONSEQUENCE FOR RULE 3 (item 1)\n{LINE}")
top = cen.iloc[0]
usable = cen[cen["n"] >= 3]
print(f"Best service: slot {int(top['slot'])}, {int(top['n'])} repeats over "
      f"{int(top['n_days'])} days, {int(top['n_axles'])} axles at {top['time']}.")
print(f"Services with >=3 repeats: {len(usable)}, covering "
      f"{int(usable['n'].sum())}/{len(seg)} segments.")
print(f"Median repeats among those: {usable['n'].median():.0f}.")
print("\nThe fixed-stimulus argument holds PER SERVICE, but a 3-5 point trend is")
print("weak evidence for a monotonic degradation claim. Rule 3 built this way")
print("must be reported as a per-service consistency check, not as a trend fit.")
print("Speed still varies widely within a service "
      f"({top['v_min']:.0f}-{top['v_max']:.0f} km/h on the best one), so the")
print("speed normalisation stays mandatory - repetition does not remove it.")

cen.to_csv(OUT / "service_census_level2.csv", index=False)
seg.to_csv(OUT / "train_segments_slotted.csv", index=False)
print(f"\nwrote service_census_level2.csv, train_segments_slotted.csv")

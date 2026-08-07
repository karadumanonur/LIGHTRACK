"""Items 3-4 — train-level detection rate on 2026-05-19, and matching without slots.

ITEM 3. The EBP log gives 83 trains on 05-19. The delivery holds 33 files
covering a known window, and because recording is contiguous every discarded
slot is countable. That supports a train-level rate against an independent
count — not axle accuracy, but the only detection rate this dataset can carry.
The traffic-distribution assumption is stated explicitly because it does the
work.

ITEM 4. Slot-level service matching uses 27 of 58 weekday passages and excludes
05-19 entirely: that day samples 05:27-11:37 local while every other session
samples 14:00-15:00, so no service is shared. Try matching on axle-spacing
SIGNATURE plus SPEED BAND instead of time-of-day, which controls geometry and
kinematics without requiring schedule overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, list_files, parse_filename_utc, to_local,
)
from lightrack.health import HEALTHY                            # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
SLOT = CONFIG["acquisition"]["slot_s"]
OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
pd.set_option("display.width", 250)
LINE = "=" * 116

# ============================================================== ITEM 3
print(f"{LINE}\nITEM 3 — TRAIN-LEVEL DETECTION RATE, 2026-05-19\n{LINE}")
ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
day = pd.Timestamp("2026-05-19").date()
n_ebp = int(ebp.loc[ebp["date"] == day, "total"].iloc[0])

files = [(p, to_local(parse_filename_utc(p.name))) for p in list_files()]
d19 = sorted([t for _, t in files if t.date() == day])
t0, t1 = d19[0], d19[-1]
span_h = (t1 - t0).total_seconds() / 3600 + SLOT / 3600
slots = int(round((t1 - t0).total_seconds() / SLOT)) + 1

seg = pd.read_csv(OUT / "train_segments.csv")
seg["local"] = pd.to_datetime(seg["local"])
det_wd = int((seg["local"].dt.date == day).sum())

print(f"EBP total for the day            : {n_ebp} trains")
print(f"files delivered on 05-19 (local) : {len(d19)}")
print(f"window covered                   : {t0:%H:%M:%S} - {t1:%H:%M:%S} local "
      f"= {span_h:.2f} h")
print(f"contiguous slots in that window   : {slots}   "
      f"(retention {100*len(d19)/slots:.1f} %)")
print(f"train segments detected (weekday-gated set): {det_wd}")
print(f"files each containing >=1 detected passage : {len(d19)} (every retained file)")

print(f"\n{'assumption':<44}{'expected in window':>20}{'detected/expected':>20}")
for lab, hours in (("uniform over 24 h", 24.0),
                   ("uniform over an 18 h operating day", 18.0),
                   ("uniform over a 14 h day (06:00-20:00)", 14.0)):
    exp = n_ebp * span_h / hours
    print(f"{lab:<44}{exp:>20.1f}{len(d19)/exp:>19.0%}")

print(f"""
THE ANSWER DEPENDS ENTIRELY ON THE ASSUMPTION, AND THAT IS THE RESULT.

Note the window is {span_h:.1f} h, not the ~6 h of the midday-session days: two
files named 2026-05-18 in UTC fall at 01:36 and 01:43 LOCAL on 05-19, so the
covered span runs 01:36-11:36.

{len(d19)} retained files each contain at least one detected passage, against
{n_ebp*span_h/24:.0f} trains expected under a uniform 24 h day and
{n_ebp*span_h/14:.0f} under a 14 h operating day. The implied rate therefore
ranges from {100*len(d19)/(n_ebp*span_h/24):.0f} % to
{100*len(d19)/(n_ebp*span_h/14):.0f} % - a spread driven by the assumption, not
by the measurement. Rail traffic is not uniform, so the low end is the more
realistic reading and the high end is an artefact of pretending 02:00 carries
the same traffic as 08:00.

{slots - len(d19)} of the {slots} slots in the window were discarded by the
project's own filter. If they were train-free, {len(d19)} trains passed and all
were detected. If they carried trains, those trains were never delivered. The
daily aggregate cannot separate the two, so RETENTION and DETECTION remain
confounded.

What can be stated with no distribution assumption at all:
  * {len(d19)} of the day's {n_ebp} trains ({100*len(d19)/n_ebp:.0f} %) appear in retained
    files, within a window covering {100*span_h/24:.0f} % of the day;
  * every retained file contains at least one detected passage - the pipeline
    found a train in {len(d19)}/{len(d19)} of them.
That is a CAPTURE rate plus a null miss-rate on retained files. It is not a
detection rate. Closing this needs an HOURLY EBP profile, not a daily total.""")

# ============================================================== ITEM 4
print(f"\n{LINE}\nITEM 4 — MATCHING ON SIGNATURE + SPEED BAND\n{LINE}")
ebp_cls = dict(zip(ebp["date"], ebp["day_class"]))
df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["date"] = df["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
df["day_class"] = df["date"].map(ebp_cls)
df["utc"] = df["name"].map(parse_filename_utc)
df = df[df["day_class"] == "normal_weekday"].copy()
o = (df[["name", "utc"]].drop_duplicates().sort_values("utc")
     .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
df = df.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
amp = df.pivot_table(index="qi", columns="pad", values="p2p_pm")

slot = pd.read_csv(OUT / "train_segments_slotted.csv")[
    ["name", "service", "slot", "v_kmh"]].drop_duplicates("name")
m = df.drop_duplicates("qi")[["qi", "name"]].merge(slot, on="name", how="left")
m = m.set_index("qi")

REP = (m.groupby("slot").size().pipe(lambda s: s[s >= 3]).index.tolist())


def score(group: pd.Series, keep=None, label="") -> dict:
    a = amp.copy()
    gg = group.reindex(a.index)
    if keep is not None:
        a = a[gg.isin(keep)]
        gg = gg.reindex(a.index)
    parts = []
    for g, idx in a.groupby(gg).groups.items():
        blk = a.loc[idx]
        if len(blk) < 2:
            continue
        parts.append(blk / blk.iloc[0])
    if not parts:
        return {}
    r = pd.concat(parts).sort_index()
    h = pd.concat([r[q] for q in HEALTHY]).dropna()
    mad = float(np.median(np.abs(h - h.median())) * 1.4826)
    out = {"label": label, "n": len(r), "groups": len(parts), "mad": mad}
    for pad in (6, 9):
        s = r[pad].dropna()
        fin = float(s.iloc[-3:].median()) if len(s) >= 3 else np.nan
        out[f"p{pad}"] = fin
        out[f"z{pad}"] = (h.median() - fin) / mad if mad else np.nan
    return out


rows = [score(pd.Series("all", index=amp.index), None, "unmatched (all weekday)"),
        score(m["service"], [0], "class level (geometry only)"),
        score(m["slot"], REP, "slot level (load + geometry)")]
for w in (10, 15, 20, 30):
    key = m["service"].astype(str) + "_" + (m["v_kmh"] // w).astype("Int64").astype(str)
    keep = key.value_counts().pipe(lambda s: s[s >= 3]).index.tolist()
    rows.append(score(key, keep, f"signature + speed band {w} km/h"))
t = pd.DataFrame([r for r in rows if r])

print(f"{'normalisation':<36}{'n':>5}{'groups':>8}{'healthy MAD':>13}"
      f"{'PAD 6 z':>10}{'PAD 9 z':>10}")
for _, r in t.iterrows():
    print(f"{r['label']:<36}{int(r['n']):>5}{int(r['groups']):>8}{r['mad']:>13.4f}"
          f"{r['z6']:>10.1f}{r['z9']:>10.1f}")

base = t[t["label"].str.startswith("class")].iloc[0]
best_sb = t[t["label"].str.startswith("signature +")].sort_values("mad").iloc[0]
print(f"\nclass level      : MAD {base['mad']:.4f}, PAD 6 z {base['z6']:.1f}, "
      f"n {int(base['n'])}")
print(f"best signature+sb: MAD {best_sb['mad']:.4f}, PAD 6 z {best_sb['z6']:.1f}, "
      f"n {int(best_sb['n'])}  [{best_sb['label']}]")
if best_sb["mad"] < base["mad"] and best_sb["z6"] > base["z6"]:
    print("=> signature + speed band BEATS class level on both measures.")
else:
    print("=> NEGATIVE: signature + speed band does not beat class level.")
    print("   Splitting by speed shrinks each group without removing variance")
    print("   the speed normalisation had not already removed.")
t.to_csv(OUT / "rule3_signature_speed.csv", index=False)
print(f"\nwrote rule3_signature_speed.csv")

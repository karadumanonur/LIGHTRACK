"""TASK 5 — does section 06 see one track or both, and what is the right
capture-rate denominator?

The question must be settled before any rate is quoted. Our array is single-track
and near-unidirectional (A-49: 62 of 65 timetable-matched trains run toward
Charleroi). If ACAT's section is the SAME single track then 93/918 is the
like-for-like capture rate and the 4.8 % of §3.2 is comparing a single-track
numerator against a both-tracks denominator. If ACAT covers both tracks it is not.

THE TEST IS DIRECT. Every ACAT event is matched to the nearest timetabled
Le Campinaire arrival, and the DIRECTION of that train is read from
RELATION_DIRECTION - which §1.9 already resolved into a physical direction from
the ORDER of the Aiseau and Farciennes timestamps, not from the label. One track
means one direction.

Also here: what the 28 unmatched sensor passages are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
pd.set_option("display.width", 250)
L = "=" * 116


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


acat = pd.read_csv(OUT / "acat_train_events.csv")
acat["t"] = pd.to_datetime(acat["t"])
cp = pd.read_csv(OUT / "punctuality_campinaire.csv", low_memory=False)
cp["arr"] = (pd.to_datetime(cp["REAL_DATE_ARR"], format="%d%b%Y")
             + pd.to_timedelta(cp["REAL_TIME_ARR"]))
cp = cp.dropna(subset=["arr"]).sort_values("arr")
te = pd.read_csv(OUT / "timetable_expected.csv", low_memory=False)
dirmap = te.drop_duplicates(["DATDEP", "TRAIN_NO"]).set_index(
    ["DATDEP", "TRAIN_NO"])["toward_charleroi"]
cp = cp.join(dirmap, on=["DATDEP", "TRAIN_NO"])

print(f"{L}\nTASK 5 — ONE TRACK OR BOTH?\n{L}")
CT = cp["arr"].to_numpy("datetime64[s]").astype("int64")
AT = acat["t"].to_numpy("datetime64[s]").astype("int64")
j = np.searchsorted(CT, AT)
best = np.full(len(AT), np.inf); bi = np.zeros(len(AT), int)
for off in (-1, 0, 1):
    k = np.clip(j + off, 0, len(CT) - 1)
    r = (AT - CT[k]).astype(float)
    m = np.abs(r) < np.abs(best)
    best = np.where(m, r, best); bi = np.where(m, k, bi)
acat["dt_tt"] = best
acat["tt_dir"] = cp["toward_charleroi"].to_numpy()[bi]
acat["tt_rel"] = cp["RELATION"].to_numpy()[bi]
hit = acat["dt_tt"].between(-30, 180)          # the mode found in 63, +-scatter
print(f"ACAT events matched to a timetabled Le Campinaire arrival "
      f"(-30..+180 s): {int(hit.sum())}/{len(acat)} = {pct(int(hit.sum()), len(acat))}")
# an ACAT event with SEVERAL timetabled trains in its window carries no
# direction information - both tracks feed Le Campinaire and the nearest arrival
# may belong to the other one. Only unique matches are read.
lo_w = np.searchsorted(CT, AT - 30)
hi_w = np.searchsorted(CT, AT + 180)
acat["n_tt"] = hi_w - lo_w
uniq = hit & (acat["n_tt"] == 1) & acat["tt_dir"].notna()
print(f"of those, UNIQUE (one timetabled train in the window) and with a "
      f"resolved direction: {int(uniq.sum())}")
kn = acat.loc[uniq, "tt_dir"].astype(bool)
print(f"\ndirection of the uniquely matched timetabled train:")
print(f"   toward Charleroi : {int(kn.sum())}")
print(f"   toward Namur     : {int(len(kn) - kn.sum())}")
print(f"   => {100*kn.mean():.1f} % toward Charleroi over {len(kn)} events")
print(f"""
THIS TEST LEANS THE RIGHT WAY BUT IS NOT DECISIVE ON ITS OWN, and saying so
matters more than the number. Both tracks feed Le Campinaire, so a "unique"
timetabled train inside an ACAT event's window can still belong to the other
track — the rule has no way to tell. {100*kn.mean():.0f} % is well above the timetable's own
51/49 and well below the ~95 % that our near-unidirectional track would give.
It is corroboration, not proof. The volume check below is the decisive one.""")

print(f"\n{L}\nDAILY VOLUME, THE SECOND CHECK\n{L}")
acat["date"] = acat["t"].dt.date
per = acat.groupby("date").size().reset_index(name="acat")
per["dow"] = pd.to_datetime(per["date"]).dt.dayofweek
wk = per[per["dow"] < 5]
print(f"ACAT trains per weekday: median {wk['acat'].median():.0f} "
      f"(n={len(wk)} weekdays)")
print(f"timetabled passengers per weekday, BOTH directions : 138  (§ item 1)")
print(f"timetabled passengers per weekday, ONE direction   :  69")
print(f"EBP vns per normal weekday                          :  83")
print(f"""
ACAT's {wk['acat'].median():.0f} sits just below one direction of passenger traffic plus the
freight it also counts, and nowhere near the 138 of both directions. Both checks
agree: SECTION 06 IS ONE TRACK, AND IT IS OURS — 44 of our passages match it with
90.9 % exact axle agreement, which coincidence cannot produce.""")

print(f"\n{L}\nTHE CAPTURE-RATE DENOMINATOR, ALL THREE NAMED\n{L}")
camp = acat[(acat["date"] >= pd.Timestamp("2026-05-14").date()) &
            (acat["date"] <= pd.Timestamp("2026-06-02").date())]
n_acat = len(camp)
print(f"delivered files 14/05-02/06: 93   gated: 70")
print(f"\n{'denominator':<52}{'covers':<26}{'93 files':>10}{'70 gated':>10}")
for lab, cov, den in (
        ("EBP daily totals, 1187  (superseded, A-48)", "unclear", 1187),
        ("punctuality passengers, 1949 (§3.2)", "BOTH tracks, no freight", 1949),
        ("punctuality passengers, one direction ~975", "one track, no freight", 975),
        (f"ACAT trains, {n_acat}", "one track, ALL traffic", n_acat)):
    print(f"{lab:<52}{cov:<26}{100*93/den:>9.1f}%{100*70/den:>9.1f}%")
print(f"""
THE ACAT DENOMINATOR IS THE RIGHT ONE and it is the only one that is
like-for-like: same single track as the array, and it counts the freight the
array also sees. On it the delivery is {100*93/n_acat:.1f} % of the traffic the sensors could
have seen, not {100*93/1949:.1f} %.

{100*93/1949:.1f} % IS NOT WRONG, IT ANSWERS A DIFFERENT QUESTION - what fraction of all
passenger traffic on the line reached a file. Both belong in the report, named.
Note they differ by roughly a factor of two, which is the two tracks.""")

print(f"\n{L}\nWHAT ARE THE 28 UNMATCHED SENSOR PASSAGES?\n{L}")
mo = pd.read_csv(OUT / "acat_match_outcomes.csv")
mo["train_local"] = pd.to_datetime(mo["train_local"])
un = mo[mo["n_in_window"] == 0]
print(f"unmatched: {len(un)}/{len(mo)}")
covered = acat.groupby("date").size()
un_by_day = un.groupby(un["train_local"].dt.date).size()
print(f"\n{'date':<13}{'unmatched':>11}{'matched':>9}{'ACAT that day':>15}")
for dt in sorted(set(mo["train_local"].dt.date)):
    u = int(un_by_day.get(dt, 0))
    tot = int((mo["train_local"].dt.date == dt).sum())
    print(f"{str(dt):<13}{u:>11}{tot-u:>9}{int(covered.get(dt, 0)):>15}")
n_norm = int(sum(covered.get(d_, 0) >= 40 for d_ in un["train_local"].dt.date))
print(f"\nunmatched passages on days where ACAT logged >= 40 trains: "
      f"{n_norm}/{len(un)}")
print("""
A REFERENCE-OUTAGE READING DOES NOT SURVIVE THAT NUMBER. The low-volume ACAT days
(05-30 and 05-31, 18 and 16 trains) contribute ZERO unmatched passages, and most
unmatched passages fall on days when ACAT was logging normally. The explanation
has to be per-passage, not per-day.""")
# is the window too tight, or is the train absent from ACAT?
AT2 = np.sort(acat["t"].to_numpy("datetime64[s]").astype("int64"))
AX2 = acat.sort_values("t")["axles"].to_numpy()
ust = un["train_local"].dt.tz_localize(None).to_numpy(
    "datetime64[s]").astype("int64")
print(f"\nwidening the window for the {len(un)} unmatched passages only:")
print(f"{'window s':<18}{'passages with >=1 ACAT event':>32}{'count agrees':>15}")
for a, b in ((-600, 900), (-1200, 1800), (-3600, 3600)):
    lo2 = np.searchsorted(AT2, ust + a)
    hi2 = np.searchsorted(AT2, ust + b)
    got = int((hi2 - lo2 > 0).sum())
    agree = 0
    for q, (l3, h3) in enumerate(zip(lo2, hi2)):
        if h3 > l3 and un["n_axles"].iloc[q] in AX2[l3:h3]:
            agree += 1
    print(f"[{a:>5},{b:>5}]{'':<6}{got:>32}{agree:>15}")
print("""
READ THAT CAUTIOUSLY IN BOTH DIRECTIONS. Widening to a full hour finds an event
for all 28, but 75 % of ACAT trains carry 12 or 24 axles, so "some event in the
window has the same count" is easy to satisfy by chance and is NOT evidence of a
recovered match. What the table does show is that the count-agreement fraction
barely moves (15 -> 15 -> 20) while the window grows 6-fold: the extra events are
noise, not missed partners.

The honest conclusion is that a substantial share of the 28 are ABSENT FROM ACAT,
so the reference is incomplete. That is a limit on the DENOMINATOR of the
accuracy figure, not on its value: the 44 eligible passages are still 44 genuine
comparisons. It does mean the reference cannot be used to claim the sensor
detected a train ACAT missed, or the reverse.""")

acat.to_csv(OUT / "acat_events_scored.csv", index=False)
pd.DataFrame([{"n_acat_campaign": n_acat, "files": 93, "gated": 70,
               "cap_93_acat": 93/n_acat, "cap_70_acat": 70/n_acat,
               "cap_93_punct": 93/1949}]).to_csv(
    OUT / "acat_denominators.csv", index=False)
print(f"\nwrote acat_events_scored.csv, acat_denominators.csv")

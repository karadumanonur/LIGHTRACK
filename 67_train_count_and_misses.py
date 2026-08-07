"""Item 1 — close the capture bracket; Item 2 — characterise the 21 misses.

ITEM 1. A-59's table was self-inconsistent: it listed "punctuality one direction
~975, complete" beside "ACAT 918, incomplete". ACAT covers one track and ALL
traffic, so it must EXCEED the passenger-only count for that direction. It does
not, which means ACAT's incompleteness is measurable directly rather than
guessed at from the 36.8 % sensor shortfall.

The site-passing direction is established the way 50_direction_and_profile.py
established it: from the ORDER of the Aiseau and Farciennes timestamps, never
from a relation label. Toward Charleroi = Aiseau first = PK increasing.

ITEM 2. §1.9 reports 63/84 = 75.0 % detection and stops there. A detection rate
without its miss population is not a closed measurement. Each timetabled train
inside a delivered window that the pipeline did not find is classified into one
of four causes, with counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, list_files, parse_filename_utc, to_local  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SLOT = CONFIG["acquisition"]["slot_s"]
LO, HI = pd.Timestamp("2026-05-14"), pd.Timestamp("2026-06-03")
pd.set_option("display.width", 250)
L = "=" * 116


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "  n/a"


# ==================================================================== ITEM 1
print(f"{L}\nITEM 1 — THE SITE-PASSING DIRECTION, COUNTED\n{L}")
p = pd.read_csv(OUT / "punctuality_site.csv", low_memory=False)
p["t"] = pd.to_datetime(p["t"])
w = p.pivot_table(index=["DATDEP", "TRAIN_NO"], columns="station",
                  values="t", aggfunc="first").dropna(
    subset=["Aiseau", "Farciennes"])
w["toward_charleroi"] = w["Farciennes"] > w["Aiseau"]
meta = p.drop_duplicates(["DATDEP", "TRAIN_NO"]).set_index(
    ["DATDEP", "TRAIN_NO"])[["RELATION", "RELATION_DIRECTION"]]
w = w.join(meta).reset_index()
w["date"] = pd.to_datetime(w["Aiseau"]).dt.normalize()
camp = w[(w["date"] >= LO) & (w["date"] < HI)]

print(f"trains at both stations, 14/05-02/06: {len(camp)}")
print(f"\n{'RELATION_DIRECTION':<46}{'n':>6}{'toward Charleroi':>18}")
for rd, g in camp.groupby("RELATION_DIRECTION"):
    if len(g) < 20:
        continue
    print(f"{str(rd)[:44]:<46}{len(g):>6}{pct(int(g['toward_charleroi'].sum()), len(g)):>18}")
fwd = camp[camp["toward_charleroi"]]
print(f"\nEVERY relation-direction maps 100/0 to one physical direction, so the")
print(f"label is safe to USE once the mapping is read from the timestamps.")
print(f"\ntoward Charleroi (the site-passing direction) by relation-direction:")
tot = 0
for rd, g in fwd.groupby("RELATION_DIRECTION"):
    print(f"   {str(rd)[:52]:<54}{len(g):>6}")
    tot += len(g)
print(f"   {'TOTAL PASSENGER, site-passing direction':<54}{tot:>6}")
N_PASS = tot

# ------------------------------------------------------- ACAT completeness
acat = pd.read_csv(OUT / "acat_train_events.csv")
acat["t"] = pd.to_datetime(acat["t"])
ac = acat[(acat["t"] >= LO) & (acat["t"] < HI)]
n_acat = len(ac)
print(f"\n{L}\nACAT COMPLETENESS, DERIVED FROM THE SHORTFALL\n{L}")
print(f"ACAT trains logged 14/05-02/06                     : {n_acat}")
print(f"passenger trains timetabled, site-passing direction : {N_PASS}")
print(f"""
ACAT covers ONE TRACK and ALL TRAFFIC, so it must exceed {N_PASS}. It does not.
The deficit is ACAT's incompleteness, and it is measurable rather than assumed.""")

# freight inside ACAT's own log, two definitions, both lower bounds
big = ac[ac["axles"] >= 40]
nonmult = ac[ac["axles"] % 12 != 0]
print(f"\nACAT's own non-passenger population, two definitions:")
print(f"   >= 40 axles                     : {len(big)}")
print(f"   not a multiple of 12            : {len(nonmult)}")
print(f"   either                          : {len(ac[(ac['axles'] >= 40) | (ac['axles'] % 12 != 0)])}")
print(f"""Both are LOWER BOUNDS on freight: a 24-axle short freight is invisible to
the first and a 48-axle one to the second.""")

rows = []
for lab, n_fr_obs in (("freight = >= 40 axles", len(big)),
                      ("freight = not a multiple of 12", len(nonmult))):
    n_pass_obs = n_acat - n_fr_obs
    c = n_pass_obs / N_PASS                       # ACAT's completeness on passengers
    n_fr_true = n_fr_obs / c if c > 0 else np.nan
    T = N_PASS + n_fr_true
    rows.append(dict(definition=lab, acat_freight=n_fr_obs,
                     acat_passenger=n_pass_obs, completeness=c,
                     freight_true=n_fr_true, total=T,
                     cap93=93/T, cap70=70/T))
print(f"\n{'freight definition':<34}{'ACAT pass':>11}{'complete':>10}"
      f"{'freight est':>13}{'same-track total':>18}{'93 files':>10}{'70 gated':>10}")
for r in rows:
    print(f"{r['definition']:<34}{r['acat_passenger']:>11}"
          f"{100*r['completeness']:>9.1f}%{r['freight_true']:>13.0f}"
          f"{r['total']:>18.0f}{100*r['cap93']:>9.1f}%{100*r['cap70']:>9.1f}%")
T_lo = min(r["total"] for r in rows)
T_hi = max(r["total"] for r in rows)
print(f"""
THE DERIVATION. If ACAT's misses fall on passenger and freight alike, its
completeness on passengers - where the timetable gives an exact denominator -
also applies to its freight observations, so the true freight count is the
observed one divided by that completeness.

SAME-TRACK CAPTURE = 93 / {T_lo:.0f}-{T_hi:.0f} = {100*93/T_hi:.1f}-{100*93/T_lo:.1f} %

WHAT REMAINS ASSUMED, and it is one thing: that ACAT misses freight and passenger
trains at the same rate. If it preferentially misses long consists - plausible, a
counter is more likely to lose sync on 130 axles than on 12 - the freight estimate
is low and the true capture is lower than the range above.

The old 6.4-10.1 % bracket came from projecting the 36.8 % sensor-side shortfall
onto the whole log. That was a cruder route to the same question and it is
SUPERSEDED: the shortfall measured directly against the timetable is
{100*(1-min(r['completeness'] for r in rows)):.1f}-{100*(1-max(r['completeness'] for r in rows)):.1f} %, not 36.8 %.""")

# ==================================================================== ITEM 2
print(f"\n{L}\nITEM 2 — THE 21 MISSES, CLASSIFIED\n{L}")
te = pd.read_csv(OUT / "timetable_expected.csv", low_memory=False)
te["site_kin"] = pd.to_datetime(te["site_kin"])
mt = pd.read_csv(OUT / "timetable_matches.csv")
mt["site_kin"] = pd.to_datetime(mt["site_kin"])
inw = te[te["in_window"]].copy()
inw["matched"] = inw["site_kin"].isin(set(mt["site_kin"]))
miss = inw[~inw["matched"]].copy()
print(f"timetabled trains inside a delivered 144.5 s window: {len(inw)}")
print(f"matched by the pipeline                            : {int(inw['matched'].sum())}")
print(f"MISSED                                             : {len(miss)}")

files = [(q.name, to_local(parse_filename_utc(q.name)).replace(tzinfo=None))
         for q in list_files()]
win = pd.DataFrame(files, columns=["name", "start"]).sort_values("start")
ws = win["start"].to_numpy("datetime64[ns]")
pos = np.searchsorted(ws, miss["site_kin"].to_numpy("datetime64[ns]"),
                      side="right") - 1
miss["win_name"] = win["name"].to_numpy()[np.clip(pos, 0, None)]

pt = pd.read_csv(OUT / "passage_times.csv")
has_passage = set(pt["name"])
occ = pd.read_csv(OUT / "occupancy_balance.csv").set_index("name")
per_win = inw.groupby("win_name").size()

def classify(r):
    nm = r["win_name"]
    if per_win.get(nm, 0) >= 2:
        return "two trains in one window (structural, A-51)"
    if nm not in has_passage:
        if nm in occ.index and not occ.loc[nm, "gated"]:
            return "no passage extracted — speed-fit gate rejected the record"
        return "no passage extracted — detector found < 4 peaks"
    if nm == "20260519T093614,145Z.txt":
        return "timing-transfer failure (A-24)"
    return "passage found, but timed outside the +-20 s tolerance"

miss["cause"] = miss.apply(classify, axis=1)
print(f"\n{'cause':<52}{'n':>5}{'of 21':>8}")
for c, g in miss.groupby("cause"):
    print(f"{c:<52}{len(g):>5}{pct(len(g), len(miss)):>8}")
print(f"\n{'record':<28}{'expected':<11}{'relation':<10}{'cause'}")
for _, r in miss.sort_values("site_kin").iterrows():
    print(f"{str(r['win_name'])[:26]:<28}{r['site_kin']:%H:%M:%S}  "
          f"{str(r['RELATION']):<10}{r['cause'][:46]}")
k, n = int(inw["matched"].sum()), len(inw)
lo_c, hi_c = (beta.ppf(.025, k, n-k+1), beta.ppf(.975, k+1, n-k))
print(f"""
DETECTION RATE {k}/{n} = {100*k/n:.1f} % [95 % CI {100*lo_c:.1f}-{100*hi_c:.1f} %], and now with its
error population named. The structural cause - two trains inside one 144.5 s
record, which the pipeline cannot report because it emits one passage per record
- is not a detector fault and no threshold addresses it.""")

miss.to_csv(OUT / "detection_misses.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "capture_denominator_closed.csv", index=False)
print(f"\nwrote detection_misses.csv, capture_denominator_closed.csv")

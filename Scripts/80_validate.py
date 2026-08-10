"""Validation — our three claims against the two reference tables.

  AXLES      our count            vs the ACAT axle counter at Le Campinaire
  DIRECTION  our resolved direction vs the ORDER of the Aiseau/Farciennes stamps
  SPEED      our speed at the site vs the leg-average speed of the same train

Inputs are `ref_acat.csv` and `ref_trains.csv` (70_reference_tables.py) and the
pipeline's own outputs: passage_times.csv, occupancy_balance.csv,
phase2_counts.csv. Nothing is fitted here and no threshold is chosen.

STEP 0 — THE ACAT CLOCK, SETTLED BEFORE ANY COUNT IS READ. The log carries no
time-zone marker and its hourly profile is flat — it counts freight through the
night — so there is no commuter shape to read it from.

What does NOT work, and why it is worth stating: fitting a distance from
t_ACAT - t_sensor = D/v. Le Campinaire is a station, the S61 services stop
there, and a stopping train's transit time has nothing to do with its speed at
the site. That fit returns D = 2875 m against 1577 m of chainage — it is
absorbing dwell time into distance, not measuring geometry.

What is used instead needs no transit model at all: count how many sensor
passages have an ACAT event in a fixed window after them, and compare each
candidate clock against PLACEBO offsets that cannot be right. A candidate that
only reaches the placebo band carries no information; the true one stands clear
of it. NO AXLE COUNT ENTERS THIS STEP.

WHY THE AXLE WINDOW IS WIDE. Le Campinaire is a station and the S61 services
stop there, so a stopping train reaches the counter several minutes after
crossing the site while a through train takes well under a minute. A window that
covered only through trains would measure accuracy on a biased sample, so it
spans [0, +400] s and the price — ambiguity when two trains fall inside — is
reported separately instead of being resolved by count, which would be circular.

MATCHING IS COUNT-BLIND. 72 % of ACAT trains carry 12 or 24 axles, so a matcher
allowed to see the count would mostly succeed by construction and the agreement
figure that followed would mean nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

_UP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UP if (_UP / "lightrack").is_dir() else _UP.parent))
from lightrack import CONFIG                                      # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
WIN_LO, WIN_HI = 0.0, 400.0          # sensor -> ACAT, seconds
TOL_S = 30.0                         # slack on the station-to-station interval
L = "=" * 100
pd.set_option("display.width", 220)


def ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    return (0.0 if k == 0 else beta.ppf(.025, k, n - k + 1),
            1.0 if k >= n else beta.ppf(.975, k + 1, n - k))


def pct(k: int, n: int) -> str:
    return f"{100*k/n:.1f}%" if n else "  n/a"


def epoch(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).dt.tz_localize(None).to_numpy(
        "datetime64[s]").astype("int64")


# ------------------------------------------------------------------ inputs
acat = pd.read_csv(OUT / "ref_acat.csv")
trains = pd.read_csv(OUT / "ref_trains.csv")
d = pd.read_csv(OUT / "passage_times.csv")
occ = pd.read_csv(OUT / "occupancy_balance.csv")
d = d.merge(occ[["name", "touch_start", "touch_end", "gated", "c_h1", "c_h2"]],
            on="name", how="left")
st = epoch(d["train_local"])
print(f"{L}\nINPUTS\n{L}")
print(f"sensor passages {len(d)} from {d['name'].nunique()} records   "
      f"{pd.to_datetime(d['train_local']).min()} -> "
      f"{pd.to_datetime(d['train_local']).max()}")
print(f"ACAT trains {len(acat)}   timetabled trains {len(trains)}")

# =========================================================== STEP 0 — clock
print(f"\n{L}\nSTEP 0 — THE ACAT CLOCK, AGAINST PLACEBO OFFSETS (no counts read)\n{L}")
CANDIDATES = {0: "ACAT is LOCAL (no shift)", 2: "ACAT is UTC (+2 h to local)"}
PLACEBOS = (1, -1, 3, 6, 12, -6)
acat_t = pd.to_datetime(acat["ts"])
score = {}
print(f"{'shift':>7}{'hit':>7}{'rate':>8}{'unique':>8}{'delay med':>11}"
      f"{'delay MAD':>11}   ")
for sh in list(CANDIDATES) + list(PLACEBOS):
    at = np.sort(epoch(acat_t + pd.Timedelta(hours=sh)))
    lo_, hi_ = np.searchsorted(at, st + WIN_LO), np.searchsorted(at, st + WIN_HI)
    uniq = hi_ - lo_ == 1
    dt = (at[np.clip(lo_, 0, len(at) - 1)] - st)[uniq]
    score[sh] = int((hi_ - lo_ > 0).sum())
    print(f"{sh:>+7d}{score[sh]:>7}{pct(score[sh], len(st)):>8}{int(uniq.sum()):>8}"
          f"{np.median(dt):>10.0f}s{np.median(np.abs(dt - np.median(dt))):>10.0f}s"
          f"   {CANDIDATES.get(sh, 'placebo')}")
band = max(score[s] for s in PLACEBOS)
live = [s for s in CANDIDATES if score[s] > band]
print(f"\nplacebo band tops out at {band}/{len(st)}; "
      f"candidates above it: {[CANDIDATES[s] for s in live] or 'NONE'}")
if len(live) != 1:
    raise SystemExit("clock not settled by this test — stop rather than guess")
SHIFT = live[0]
AT = np.sort(epoch(acat_t + pd.Timedelta(hours=SHIFT)))
print(f"=> ACAT timestamps are {'LOCAL' if SHIFT == 0 else 'UTC'}, and the delay\n"
      f"   spread above is consistent with a station stop at Le Campinaire "
      f"rather\n   than with a constant-speed transit.")

# =========================================================== STEP 1 — axles
print(f"\n{L}\nSTEP 1 — AXLES vs THE ACAT COUNTER\n{L}")
AX = acat.sort_values("ts")["axles"].to_numpy()
lo = np.searchsorted(AT, st + WIN_LO)
hi = np.searchsorted(AT, st + WIN_HI)
d["n_in_window"] = hi - lo
one = d["n_in_window"] == 1
d["acat_axles"] = np.where(one, AX[np.clip(lo, 0, len(AX) - 1)], np.nan)
d["acat_dt_s"] = np.where(one, AT[np.clip(lo, 0, len(AT) - 1)] - st, np.nan)

n = len(d)
print(f"window: passage + [{WIN_LO:.0f}, {WIN_HI:.0f}] s")
print(f"   MATCHED   (exactly one ACAT event) {int(one.sum()):>4}  {pct(int(one.sum()), n)}")
print(f"   AMBIGUOUS (two or more)            "
      f"{int((d['n_in_window'] >= 2).sum()):>4}  {pct(int((d['n_in_window']>=2).sum()), n)}")
print(f"   UNMATCHED (none)                   "
      f"{int((d['n_in_window'] == 0).sum()):>4}  {pct(int((d['n_in_window']==0).sum()), n)}")
print(f"delay on the matched set: median {d.loc[one,'acat_dt_s'].median():.0f} s "
      f"(p10 {d.loc[one,'acat_dt_s'].quantile(.1):.0f}, "
      f"p90 {d.loc[one,'acat_dt_s'].quantile(.9):.0f})")

# Edge-clipped records cannot hold a whole train and must mismatch by
# construction, so they are excluded BEFORE any count is compared.
d["edge_clipped"] = d["touch_start"].fillna(False) | d["touch_end"].fillna(False)
elig = d[one & ~d["edge_clipped"]].copy()
print(f"\nedge-clipped among the matched, excluded by rule: "
      f"{int(d.loc[one, 'edge_clipped'].sum())}   ELIGIBLE: {len(elig)}")

for col, src in (("err_h1", "c_h1"), ("err_h2", "c_h2"), ("err_diag", "n_axles")):
    elig[col] = elig[src] - elig["acat_axles"]
print(f"\n{'output':<26}{'n':>5}{'exact':>7}{'rate':>8}{'95 % CI':>17}"
      f"{'median err':>12}{'mean err':>10}")
for lab, col in (("Head 1 median vote", "err_h1"),
                 ("Head 2 median vote", "err_h2"),
                 ("diagnostic chain", "err_diag")):
    e = elig[col].dropna()
    k = int((e == 0).sum())
    c = ci(k, len(e))
    print(f"{lab:<26}{len(e):>5}{k:>7}{pct(k, len(e)):>8}"
          f"{f'{100*c[0]:.1f}-{100*c[1]:.1f}%':>17}{e.median():>12.1f}{e.mean():>10.2f}")

print(f"\nsigned error, head median vote (over-count is the SAFETY direction):")
for lab, col in (("Head 1", "err_h1"), ("Head 2", "err_h2")):
    vc = elig[col].dropna().value_counts().sort_index()
    print(f"   {lab}: " + "  ".join(f"{int(k):+d}:{v}" for k, v in vc.items()))

elig["bin"] = pd.cut(elig["acat_axles"], [0, 12, 24, 32, 400],
                     labels=["<=12", "13-24", "25-32", ">32 (freight)"])
print(f"\n{'consist':<16}{'n':>5}{'H1 exact':>10}{'H2 exact':>10}"
      f"{'H1 med err':>12}{'H2 med err':>12}")
for b, g in elig.groupby("bin", observed=True):
    print(f"{str(b):<16}{len(g):>5}"
          f"{pct(int((g['err_h1']==0).sum()), len(g)):>10}"
          f"{pct(int((g['err_h2']==0).sum()), len(g)):>10}"
          f"{g['err_h1'].median():>12.1f}{g['err_h2'].median():>12.1f}")

sens = pd.read_csv(OUT / "phase2_counts.csv")
sens = sens[sens["name"].isin(elig["name"])].merge(
    elig[["name", "acat_axles"]], on="name")
sens["err"] = sens["count"] - sens["acat_axles"]
print(f"\nper sensor, not just the head vote:")
print(f"{'pad':>4}{'head':>6}{'n':>5}{'exact':>7}{'rate':>8}{'median err':>12}"
      f"{'over':>6}{'under':>7}")
for pad, g in sens.groupby("pad"):
    e = g["err"].dropna()
    print(f"{int(pad):>4}{int(g['head'].iloc[0]):>6}{len(e):>5}"
          f"{int((e == 0).sum()):>7}{pct(int((e==0).sum()), len(e)):>8}"
          f"{e.median():>12.1f}{int((e>0).sum()):>6}{int((e<0).sum()):>7}")

# ======================================================= STEP 2 — direction
print(f"\n{L}\nSTEP 2 — DIRECTION vs THE ORDER OF THE STATION TIMESTAMPS\n{L}")
tf, tl = epoch(trains["t_first"]), epoch(trains["t_last"])
order = np.argsort(tf)
tf, tl = tf[order], tl[order]
dirn = trains["to_charleroi"].to_numpy()[order]
vleg = trains["v_leg_kmh"].to_numpy()[order]

hit_i, hit_n = [], []
for t in st:
    inside = np.nonzero((tf - TOL_S <= t) & (t <= tl + TOL_S))[0]
    hit_n.append(len(inside))
    hit_i.append(inside[0] if len(inside) == 1 else -1)
d["tt_n"] = hit_n
d["tt_i"] = hit_i
m = d[d["tt_i"] >= 0].copy()
m["tt_to_charleroi"] = dirn[m["tt_i"].to_numpy()]
m["tt_v_leg"] = vleg[m["tt_i"].to_numpy()]
print(f"passages falling inside exactly one station-to-station interval "
      f"(+-{TOL_S:.0f} s): {len(m)} of {len(d)}")
print(f"   inside two or more (not usable): {int((d['tt_n'] >= 2).sum())}   "
      f"inside none: {int((d['tt_n'] == 0).sum())}")
print("""
The sensor reports which head fires first; which physical direction that
corresponds to is a CONVENTION and is read off the table below rather than
assumed. What the table can therefore prove is CONSISTENCY: if the mapping is
real, one diagonal holds almost everything. A near-even split would mean our
direction carries no information, whatever the convention.""")
ct = pd.crosstab(m["head1_first"], m["tt_to_charleroi"])
ct.index = [f"head1 first = {i}" for i in ct.index]
ct.columns = [f"timetable: {'to Charleroi' if c else 'to Namur'}" for c in ct.columns]
print("\n" + ct.to_string())
diag = max(int(np.trace(ct.to_numpy())),
           int(np.trace(np.fliplr(ct.to_numpy()))))
tot = int(ct.to_numpy().sum())
c = ci(diag, tot)
print(f"\nconsistent with a single mapping: {diag}/{tot} = {pct(diag, tot)}"
      f"   95 % CI {100*c[0]:.1f}-{100*c[1]:.1f}%")

# =========================================================== STEP 3 — speed
print(f"\n{L}\nSTEP 3 — SPEED vs THE LEG AVERAGE\n{L}")
print("""The leg average is not a target value. It covers 2.726 km bracketed by two
stops, so it is bounded ABOVE by the instantaneous speed at the site: the correct
result is a ratio comfortably above 1, and a ratio near or below 1 would mean our
speed reads low. The DISTRIBUTION is the measurement; a single number would hide
the stopping/through split that drives it.""")
r = (m["v_kmh"] / m["tt_v_leg"]).dropna()
print(f"\nn = {len(r)}   sensor speed {m['v_kmh'].median():.1f} km/h median, "
      f"leg average {m['tt_v_leg'].median():.1f} km/h median")
print(f"ratio sensor/leg: median {r.median():.2f}   p10 {r.quantile(.1):.2f}   "
      f"p90 {r.quantile(.9):.2f}   below 1.0: {int((r < 1).sum())}/{len(r)}")
fast = m[m["tt_v_leg"] >= m["tt_v_leg"].quantile(.75)]
rf = (fast["v_kmh"] / fast["tt_v_leg"]).dropna()
print(f"\nrestricted to the fastest quartile of leg averages — the trains least\n"
      f"affected by a station stop, so the closest thing to a like-for-like\n"
      f"comparison: n = {len(rf)}, ratio median {rf.median():.2f} "
      f"(p10 {rf.quantile(.1):.2f}, p90 {rf.quantile(.9):.2f})")
print(f"\nmaximum speed observed by the sensor : {d['v_kmh'].max():.1f} km/h")
print(f"maximum leg average in the same window: "
      f"{m['tt_v_leg'].max():.1f} km/h over {len(m)} matched trains")

elig.to_csv(OUT / "validation_axles.csv", index=False)
m.to_csv(OUT / "validation_direction_speed.csv", index=False)
print(f"\nwrote validation_axles.csv ({len(elig)}) and "
      f"validation_direction_speed.csv ({len(m)})")

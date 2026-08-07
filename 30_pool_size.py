"""Item 5 — the minimum head count for cross-head referencing.

POC2_RECOMMENDATIONS §1 says "at least two heads" without deriving it, and two
is probably not enough: with two heads at 2 FBGs each, a sensor in head A is
referenced against head B's two gratings, and if head B degrades the pool is
gone. That is the within-head failure case again, one level up.

Derived from POC1's own data by shrinking the reference pool and measuring the
false-alarm rate as a function of pool size.

Two views:
  (a) ARBITRARY subsets of size k - the general curve, averaged over subsets;
  (b) COMPLETE-HEAD pools - the realistic configurations, since a deployment
      adds sensors a whole head at a time. With H heads at 2 FBGs, a sensor's
      cross-head pool is 2(H-1).

This number sets the cost line in a POC2 proposal, so it should not be left
qualitative.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, parse_filename_utc, to_local   # noqa: E402
from lightrack.health import DEGRADING, HEALTHY             # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
K_MON, TREND_WIN, CONFIRM_N, CONFIRM_K = 3.0, 5, 3, 2
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}
RNG = np.random.default_rng(0)
MAX_SUBSETS = 40

pd.set_option("display.width", 240)
LINE = "=" * 106

ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
cls = dict(zip(ebp["date"], ebp["day_class"]))

df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["date"] = df["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
df["day_class"] = df["date"].map(cls)
df["utc"] = df["name"].map(parse_filename_utc)
df = df[df["day_class"] == "normal_weekday"].copy()
o = (df[["name", "utc"]].drop_duplicates().sort_values("utc")
     .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
df = df.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
df["snr_trend"] = (df.groupby("pad")["snr_delta"]
                   .transform(lambda v: v.rolling(TREND_WIN, min_periods=2).median()))
P = df.pivot_table(index="qi", columns="pad", values="p2p_pm")
TRD = df.pivot_table(index="qi", columns="pad", values="snr_trend")
S1 = sorted(df.loc[df["session"] == 1, "qi"].unique())


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def rate_for(pool_fn) -> tuple[int, int, dict]:
    """LOSO false alarms with a caller-supplied reference pool per sensor."""
    w = {}
    for pad in range(1, 13):
        pool = pool_fn(pad)
        if not pool:
            w[pad] = pd.Series(np.nan, index=P.index)
            continue
        ref = P[pool].median(axis=1) if len(pool) > 1 else P[pool[0]]
        w[pad] = np.log2(P[pad] / ref.replace(0, np.nan))
    W = pd.DataFrame(w)
    W = W - W.loc[S1].median()
    fa = tot = 0
    det = {}
    for pad in sorted(HEALTHY):
        cal = HEALTHY - {pad}
        v = pd.concat([W[q] for q in cal]).dropna()
        if v.empty:
            continue
        m = v.median()
        thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
        tv = pd.concat([TRD[q] for q in cal]).dropna()
        tt = tv.median() - K_MON * float(np.median(np.abs(tv - tv.median())) * 1.4826)
        f = (W[pad] < thr) | (TRD[pad] < tt)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        fa += int(c.sum())
        tot += len(c)
    v = pd.concat([W[q] for q in HEALTHY]).dropna()
    thr = v.median() - K_MON * float(np.median(np.abs(v - v.median())) * 1.4826)
    tv = pd.concat([TRD[q] for q in HEALTHY]).dropna()
    tt = tv.median() - K_MON * float(np.median(np.abs(tv - tv.median())) * 1.4826)
    for pad in sorted(DEGRADING):
        f = (W[pad] < thr) | (TRD[pad] < tt)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        det[pad] = int(c.index[c][0]) if c.any() else None
    return fa, tot, det


# ------------------------------------------------- (a) arbitrary subsets
print(f"{LINE}\n(a) GENERAL CURVE — arbitrary cross-head reference pools\n{LINE}")
print(f"{'pool size':>10}{'subsets':>9}{'mean FA %':>11}{'min':>8}{'max':>8}"
      f"   detection 6/9/11 (modal)")
rows = []
for k in range(1, 9):
    rates, dets = [], []
    for pad_ref in range(1):     # pool choice is per-sensor; sample globally
        pass
    combos = []
    others = {p: [q for q in range(1, 13) if HEAD_OF[q] != HEAD_OF[p]] for p in range(1, 13)}
    n_sub = min(MAX_SUBSETS, len(list(itertools.combinations(others[1], k))))
    for _ in range(n_sub):
        choice = {p: list(RNG.choice(others[p], size=k, replace=False)) for p in range(1, 13)}
        fa, tot, det = rate_for(lambda pad: choice[pad])
        rates.append(100 * fa / tot)
        dets.append(tuple(det.get(x) for x in (6, 9, 11)))
    modal = max(set(dets), key=dets.count)
    rows.append({"pool": k, "mean": np.mean(rates), "min": np.min(rates),
                 "max": np.max(rates), "det": modal})
    print(f"{k:>10}{n_sub:>9}{np.mean(rates):>11.1f}{np.min(rates):>8.1f}"
          f"{np.max(rates):>8.1f}   {'/'.join(str(x) for x in modal)}")

# ------------------------------------------- (b) complete-head pools
print(f"\n{LINE}\n(b) REALISTIC — complete-head pools, 2 FBG per head\n{LINE}")
print("A deployment adds sensors a head at a time. With H heads at 2 FBG each,")
print("a sensor's cross-head pool is 2(H-1).\n")
KEEP2 = {1: [1, 2], 2: [5, 6], 3: [9, 10]}    # the comb pair retained per head
STATE = {p: ("DEGRADED" if p in DEGRADING else "healthy") for p in range(1, 13)}
print("Retained 2-FBG pair per head (the comb pairs, A-11):")
for h, ps in KEEP2.items():
    print(f"  head {h}: " + ", ".join(f"PAD{p}[{STATE[p]}]" for p in ps))
print("\nOnly sensors inside the simulated deployment are evaluated, and for a")
print("sensor in head h the pool is every retained sensor in the OTHER heads.\n")


def eval_subset(heads: list[int]) -> tuple[int, int, dict, int]:
    live = [q for h in heads for q in KEEP2[h]]
    healthy_live = sorted(set(live) & HEALTHY)
    w = {}
    for pad in live:
        pool = [q for h in heads if h != HEAD_OF[pad] for q in KEEP2[h]]
        if not pool:
            continue
        ref = P[pool].median(axis=1) if len(pool) > 1 else P[pool[0]]
        w[pad] = np.log2(P[pad] / ref.replace(0, np.nan))
    W = pd.DataFrame(w)
    W = W - W.loc[S1].median()
    fa = tot = 0
    for pad in healthy_live:
        cal = [q for q in healthy_live if q != pad]
        if not cal:
            continue
        v = pd.concat([W[q] for q in cal]).dropna()
        m = v.median()
        thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
        tv = pd.concat([TRD[q] for q in cal]).dropna()
        tt = tv.median() - K_MON * float(np.median(np.abs(tv - tv.median())) * 1.4826)
        f = (W[pad] < thr) | (TRD[pad] < tt)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        fa += int(c.sum())
        tot += len(c)
    det = {}
    v = pd.concat([W[q] for q in healthy_live]).dropna()
    thr = v.median() - K_MON * float(np.median(np.abs(v - v.median())) * 1.4826)
    for pad in sorted(set(live) & DEGRADING):
        f = (W[pad] < thr)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        det[pad] = int(c.index[c][0]) if c.any() else None
    pool_n = len(KEEP2[heads[0]]) * (len(heads) - 1)
    return fa, tot, det, pool_n


print(f"{'H':>3}{'heads used':<16}{'pool':>6}{'pool contains':<26}"
      f"{'LOSO false alarms':>24}")
res = []
for heads in ([1, 2], [1, 3], [2, 3], [1, 2, 3]):
    fa, tot, det, pool_n = eval_subset(heads)
    if tot == 0:
        continue
    lo, hi = cp(fa, tot)
    # what a typical sensor's pool contains
    ex = [q for h in heads if h != heads[0] for q in KEEP2[h]]
    tag = ", ".join(f"PAD{q}" + ("*" if q in DEGRADING else "") for q in ex)
    print(f"{len(heads):>3}{str(heads):<16}{pool_n:>6}{tag:<26}"
          f"{f'{fa}/{tot} = {100*fa/tot:5.1f} % [{100*lo:.1f}-{100*hi:.1f}]':>24}")
    res.append({"H": len(heads), "heads": str(heads), "pool": pool_n,
                "fa": fa, "tot": tot, "rate": 100 * fa / tot,
                "lo": 100 * lo, "hi": 100 * hi})
print("  * = degrading sensor inside the reference pool")

r = pd.DataFrame(res)
print(f"\n{LINE}\nVERDICT — the minimum head count\n{LINE}")
two = r[r["H"] == 2]
print(f"H = 2 (pool of 2): {two['rate'].min():.1f} - {two['rate'].max():.1f} % "
      f"across the three head pairings")
print(f"H = 3 (pool of 4): {r[r['H']==3]['rate'].iloc[0]:.1f} %")
print("""
The spread across WHICH heads are paired is the finding, not the mean. At H = 2
the reference pool is one head's two gratings, so the rate is decided entirely
by whether that head happens to be healthy - and A-31 showed the failure is
one-sided, so a pool containing an OVER-RESPONDING sensor is far worse than one
containing a collapsing sensor. A design cannot choose its healthy head in
advance.

H = 2 therefore reproduces the within-head failure one level up: a single point
of reference, no redundancy in the reference itself, and no way to tell a
degrading reference from a degrading subject.

MINIMUM: three heads, giving a cross-head pool of four. That is the smallest
configuration in which the reference survives losing one member, and it is the
number a POC2 proposal should cost.""")

pd.DataFrame(rows).to_csv(OUT / "pool_size_curve.csv", index=False)
r.to_csv(OUT / "pool_size_heads.csv", index=False)
print(f"\nwrote pool_size_curve.csv, pool_size_heads.csv")

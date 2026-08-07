"""Points 1 & 2 — is X4 an implicit oracle, and does the trend rule cover common mode?

PART A. X4 works partly because every contaminating member happens to sit inside
Head 3. Real degradation will not oblige. Two things are checked: whether the
degrading PAD 6 is actually inside the cross-head reference pool of the Head 3
sensors, and how much of X4's performance comes from the MEDIAN rejecting a
degraded pool member rather than from cross-head referencing itself.

PART B. Cross-head does not solve common mode. All three heads share ballast,
sleepers, thermal cycle and weather; if adhesive ageing is temperature-driven it
advances everywhere at once and BOTH pair variants go silent, because a ratio
between two equally-degraded sensors is unchanged. The self-referential trend
rule is then the only defence, and it has never been tested because no
correlated failure occurred in POC1.

Synthetic check: attenuate sensors multiplicatively (signal down, noise
unchanged, so SNR scales with the attenuation) and sweep the depth. Reports the
attenuation at which the trend rule stops firing. Synthetic, so it bounds the
claim rather than proving it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, parse_filename_utc, to_local   # noqa: E402
from lightrack.health import DEGRADING, BORDERLINE, HEALTHY  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
K_MON, TREND_WIN, CONFIRM_N, CONFIRM_K = 3.0, 5, 3, 2
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}
KEEP = {1: [1, 2], 2: [5, 6], 3: [9, 10]}          # X4 as run
KEEP_STRESS = {1: [1, 2], 2: [6, 7], 3: [9, 10]}   # 2 impaired members in pool

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

P2P = df.pivot_table(index="qi", columns="pad", values="p2p_pm")
SNR = df.pivot_table(index="qi", columns="pad", values="snr")
N = len(P2P)
# Commissioning baseline = the MEDIAN over session 1, not the single first
# passage. Using one passage makes the trend rule so noisy that it fires on its
# own and masks every pair-reference difference - which is exactly what happened
# on the first run of this script (X4 came out at 2.8 % instead of 1.5 %, and
# the stress pool gave results identical to the normal pool).
S1 = sorted(df.loc[df["session"] == 1, "qi"].unique())
print(f"commissioning baseline: session 1 = passages {S1}")

# ============================================================ PART A
print(f"{LINE}\nPART A — is the X4 reference pool an implicit oracle?\n{LINE}")
state = {p: ("DEGRADED" if p in DEGRADING else
             "borderline" if p in BORDERLINE else "healthy") for p in range(1, 13)}
print(f"{'sensor':>7}{'head':>6}  cross-head reference pool (X4)")
for pad in (10, 12, 1, 5):
    pool = [q for h, ps in KEEP.items() if h != HEAD_OF[pad] for q in ps]
    tags = ", ".join(f"PAD{q}[{state[q]}]" for q in pool)
    print(f"{pad:>7}{HEAD_OF[pad]:>6}  {tags}")
pool10 = [q for h, ps in KEEP.items() if h != HEAD_OF[10] for q in ps]
print(f"\nIs PAD 6 (the degrading control) inside PAD 10's pool? "
      f"{'YES' if 6 in pool10 else 'NO'}")
print("So X4 is NOT an oracle on that axis: the Head 3 sensors are referenced")
print("against a pool that contains a genuinely degrading sensor.")


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def make_ratio(p2p: pd.DataFrame, kind: str, agg: str, keep=KEEP) -> pd.DataFrame:
    out = {}
    for pad in range(1, 13):
        if kind == "within":
            mates = [q for q in range(1, 13) if HEAD_OF[q] == HEAD_OF[pad] and q != pad]
        else:
            mates = [q for h, ps in keep.items() if h != HEAD_OF[pad] for q in ps]
        r = p2p[mates]
        ref = r.median(axis=1) if agg == "median" else r.mean(axis=1)
        out[pad] = np.log2(p2p[pad] / ref.replace(0, np.nan))
    w = pd.DataFrame(out)
    return w - w.loc[S1].median()


def trend_of(snr: pd.DataFrame) -> pd.DataFrame:
    d = np.log2(snr / snr.loc[S1].median())
    return d.rolling(TREND_WIN, min_periods=2).median()


def loso(w: pd.DataFrame, tr: pd.DataFrame) -> dict:
    fa = tot = 0
    per = {}
    for pad in sorted(HEALTHY):
        cal = HEALTHY - {pad}
        v = pd.concat([w[q] for q in cal]).dropna()
        m = v.median()
        thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
        tv = pd.concat([tr[q] for q in cal]).dropna()
        tm = tv.median()
        tt = tm - K_MON * float(np.median(np.abs(tv - tm)) * 1.4826)
        f = (w[pad] < thr) | (tr[pad] < tt)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        per[pad] = int(c.sum())
        fa += int(c.sum())
        tot += len(c)
    return {"fa": fa, "tot": tot, "ci": cp(fa, tot), "per": per}


TR = trend_of(SNR)
print(f"\n{LINE}\nHOW MUCH OF X4 COMES FROM THE MEDIAN REJECTING A BAD POOL MEMBER?\n{LINE}")
print(f"{'config':<44}{'aggregator':<11}{'LOSO false alarms':>24}  PAD10/PAD12")
for kind, keep, label in [("cross", KEEP, "X4 cross-head, pool {1,2,5,6}"),
                          ("cross", KEEP_STRESS,
                           "X4-stress, pool {1,2,6,7} (2 impaired)")]:
    for agg in ("median", "mean"):
        w = make_ratio(P2P, kind, agg, keep)
        r = loso(w, TR)
        print(f"{label:<44}{agg:<11}"
              f"{f'{r[chr(102)+chr(97)]}/{r[chr(116)+chr(111)+chr(116)]} = {100*r[chr(102)+chr(97)]/r[chr(116)+chr(111)+chr(116)]:5.1f} % [{100*r[chr(99)+chr(105)][0]:.1f}-{100*r[chr(99)+chr(105)][1]:.1f}]':>24}"
              f"  {r['per'][10]:>3}/{r['per'][12]:<3}")

# ============================================================ PART B
print(f"\n{LINE}\nPART B — synthetic COMMON-MODE degradation\n{LINE}")
print("Attenuate the signal multiplicatively from passage 20 onward; noise is")
print("unchanged, so SNR scales with the attenuation. Two scenarios:")
print("  ONE HEAD : only Head 1's four sensors attenuated")
print("  ALL HEADS: every sensor attenuated equally  <- the common-mode case\n")

START = 20
VICTIM = 1     # the sensor whose detection we track


def synth(att: float, scope: str) -> dict:
    p2p, snr = P2P.copy(), SNR.copy()
    targets = ([q for q in range(1, 13) if HEAD_OF[q] == 1] if scope == "one_head"
               else list(range(1, 13)))
    idx = p2p.index[START:]
    p2p.loc[idx, targets] *= att
    snr.loc[idx, targets] *= att
    tr = trend_of(snr)
    res = {}
    # Thresholds are calibrated on the PRE-attenuation passages only. Fitting
    # them on the whole record lets the calibration population fall along with
    # the victim - in the one-head scenario three of the seven calibration
    # sensors are themselves attenuated - which drags the threshold down and
    # hides the very degradation the test is injecting. Pre-attenuation
    # calibration is also what a deployed system would actually have.
    pre = slice(0, START)
    for kind, agg, name in (("within", "median", "pair_within"),
                            ("cross", "median", "pair_cross")):
        w = make_ratio(p2p, kind, agg)
        cal = HEALTHY - {VICTIM}
        v = pd.concat([w[q].iloc[pre] for q in cal]).dropna()
        m = v.median()
        thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
        f = (w[VICTIM] < thr).fillna(False)
        c = f.rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        res[name] = bool(c.iloc[START:].any())
    tv = pd.concat([tr[q].iloc[pre] for q in HEALTHY - {VICTIM}]).dropna()
    tt = tv.median() - K_MON * float(np.median(np.abs(tv - tv.median())) * 1.4826)
    f = (tr[VICTIM] < tt).fillna(False)
    c = f.rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
    res["trend"] = bool(c.iloc[START:].any())
    return res


for scope, title in (("one_head", "ONE HEAD attenuated (Head 1 only)"),
                     ("all", "ALL HEADS attenuated (common mode)")):
    print(f"{title}")
    print(f"{'attenuation':>12}{'loss %':>9}{'pair within-head':>19}"
          f"{'pair cross-head':>18}{'trend (self-ref)':>19}")
    for att in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.30):
        r = synth(att, scope)
        print(f"{att:>12.2f}{100*(1-att):>9.0f}"
              + "".join(f"{'FIRES' if r[k] else '.':>19}" if k != 'pair_cross'
                        else f"{'FIRES' if r[k] else '.':>18}"
                        for k in ("pair_within", "pair_cross", "trend")))
    print()

print(f"{LINE}\nWHERE THE TREND RULE STOPS\n{LINE}")
for scope in ("one_head", "all"):
    fired = [a for a in np.arange(0.99, 0.50, -0.01) if synth(round(a, 2), scope)["trend"]]
    if fired:
        print(f"  {scope:<10}: trend fires for attenuation <= {max(fired):.2f} "
              f"({100*(1-max(fired)):.0f} % signal loss)")
    else:
        print(f"  {scope:<10}: trend never fires")
print("\nAnalytic check: the trend gate is snr_trend < "
      f"{float(pd.concat([TR[q] for q in HEALTHY]).dropna().median() - K_MON*np.median(np.abs(pd.concat([TR[q] for q in HEALTHY]).dropna() - pd.concat([TR[q] for q in HEALTHY]).dropna().median()))*1.4826):+.3f}, "
      "and attenuation a shifts")
print("snr_delta by log2(a), so firing needs a below 2^threshold. Everything")
print("above that attenuation is INVISIBLE to all three rules under common mode.")

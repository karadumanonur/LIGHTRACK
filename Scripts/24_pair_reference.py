"""Point 2 — the pair reference is contaminated by degraded head-mates.

24 of 30 LOSO false alarms come from PAD 12, which is healthy. PAD 12 sits in
Head 3, so its pair reference is PAD 9 and PAD 11 — both collapsed. A ratio
measured against degraded neighbours must drift. Same class of error as PAD 3
contaminating the shape rule with sleeper-bay position (A-25).

Four references compared:

  V0  leave-one-out MEAN of the rest of the head        (current)
  V1  leave-one-out MEDIAN of the rest of the head      (robustification)
  V2  LOO mean over head-mates NOT currently flagged by the self-referential
      trend rule                                        (deployable)
  V3  LOO mean excluding the known-degraded PADs 6/9/11 (ORACLE - diagnostic
      upper bound only, uses ground truth, not deployable)

V2 is the interesting one: rule 3 needs no neighbour, so it can decide which
neighbours are fit to serve as rule 2's reference. That is self-consistent
rather than circular.

Also breaks the day-class question down to the sensor that actually drives it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD, parse_filename_utc, to_local  # noqa: E402
from lightrack.health import DEGRADING, BORDERLINE, HEALTHY            # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
K_MON, TREND_WIN, CONFIRM_N, CONFIRM_K, ONSET_FRAC = 3.0, 5, 3, 2, 0.70
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}

pd.set_option("display.width", 240)
LINE = "=" * 106

ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
cls = dict(zip(ebp["date"], ebp["day_class"]))

df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["utc"] = df["name"].map(parse_filename_utc)
df["date"] = df["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
df["day_class"] = df["date"].map(cls)
df["head"] = df["pad"].map(HEAD_OF)
df = df[df["day_class"] == "normal_weekday"].copy()

o = (df[["name", "utc"]].drop_duplicates().sort_values("utc")
     .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
df = df.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
df["snr_trend"] = (df.groupby("pad")["snr_delta"]
                   .transform(lambda v: v.rolling(TREND_WIN, min_periods=2).median()))
N = df["qi"].nunique()
print(f"normal-weekday passages: {N}")

# ---------------------------------------- who actually drives the false alarms
print(f"\n{LINE}\nWHERE THE LOWER TAIL OF pair_delta LIVES\n{LINE}")
full = pd.read_csv(OUT / "phase4_features_scored.csv")
full["date"] = full["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
full["day_class"] = full["date"].map(cls)
print(f"{'PAD':>4}{'head':>6}{'day class':<18}{'n':>5}{'median':>9}{'p5':>9}"
      f"{'p1':>9}{'min':>9}")
for pad in sorted(HEALTHY):
    for k in ("normal_weekday", "weekend"):
        s = full[(full["pad"] == pad) & (full["day_class"] == k)]["pair_delta"].dropna()
        if s.empty:
            continue
        print(f"{pad:>4}{HEAD_OF[pad]:>6}{k:<18}{len(s):>5}{s.median():>9.3f}"
              f"{s.quantile(0.05):>9.3f}{s.quantile(0.01):>9.3f}{s.min():>9.3f}")
print("\nthreshold is pair_delta < -0.727; only sensors whose lower tail reaches")
print("past it can produce false alarms at all.")

# ------------------------------------------------------------- the variants
def trend_thr(x: pd.DataFrame, cal: set[int]) -> float:
    v = x[x["pad"].isin(cal)]["snr_trend"].dropna()
    m = v.median()
    return m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)


t_thr = trend_thr(df, HEALTHY)
df["trend_flag"] = df["snr_trend"] < t_thr
print(f"\nself-referential trend gate for V2: snr_trend < {t_thr:+.3f}")

piv = df.pivot_table(index="qi", columns="pad", values="p2p_pm")
tfl = df.pivot_table(index="qi", columns="pad", values="trend_flag").astype(bool)

def build(variant: str) -> pd.DataFrame:
    out = {}
    for pad in range(1, 13):
        mates = [q for q in range(1, 13) if HEAD_OF[q] == HEAD_OF[pad] and q != pad]
        if variant == "V3":
            mates = [q for q in mates if q not in DEGRADING]
        vals = []
        for qi in piv.index:
            m = mates
            if variant == "V2":
                m = [q for q in mates if not tfl.loc[qi, q]]
            if not m:
                vals.append(np.nan)
                continue
            r = piv.loc[qi, m]
            ref = r.median() if variant == "V1" else r.mean()
            vals.append(np.log2(piv.loc[qi, pad] / ref) if ref > 0 else np.nan)
        out[pad] = pd.Series(vals, index=piv.index)
    w = pd.DataFrame(out)
    return w - w.iloc[0]          # commissioning baseline


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def loso(pd_delta: pd.DataFrame) -> dict:
    fa = tot = 0
    per = {}
    for pad in sorted(HEALTHY):
        cal = HEALTHY - {pad}
        v = pd.concat([pd_delta[q] for q in cal]).dropna()
        m = v.median()
        thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
        tv = df[df["pad"].isin(cal)]["snr_trend"].dropna()
        tm = tv.median()
        tt = tm - K_MON * float(np.median(np.abs(tv - tm)) * 1.4826)
        f = (pd_delta[pad] < thr) | (df[df["pad"] == pad].set_index("qi")["snr_trend"] < tt)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        per[pad] = int(c.sum())
        fa += int(c.sum())
        tot += len(c)
    lo, hi = cp(fa, tot)
    # detection, thresholds fitted on all healthy
    v = pd.concat([pd_delta[q] for q in HEALTHY]).dropna()
    m = v.median()
    thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
    det = {}
    for pad in sorted(DEGRADING):
        f = (pd_delta[pad] < thr) | (df[df["pad"] == pad].set_index("qi")["snr_trend"] < t_thr)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        det[pad] = int(c.index[c][0]) if c.any() else None
    return {"fa": fa, "tot": tot, "ci": (lo, hi), "per": per, "det": det, "thr": thr}


print(f"\n{LINE}\nPAIR-REFERENCE VARIANTS — LOSO on normal-weekday records\n{LINE}")
NAMES = {"V0": "LOO mean (current)", "V1": "LOO median",
         "V2": "LOO mean, trend-unflagged mates", "V3": "LOO mean, ORACLE excl. 6/9/11"}
res = {}
print(f"{'':<4}{'reference':<34}{'LOSO false alarms':>22}{'PAD 12':>8}{'PAD 10':>8}"
      f"   detection PAD 6/9/11")
for v in ("V0", "V1", "V2", "V3"):
    w = build(v)
    r = loso(w)
    res[v] = (w, r)
    print(f"{v:<4}{NAMES[v]:<34}"
          f"{f'{r[chr(102)+chr(97)]}/{r[chr(116)+chr(111)+chr(116)]} = {100*r[chr(102)+chr(97)]/r[chr(116)+chr(111)+chr(116)]:.1f} % [{100*r[chr(99)+chr(105)][0]:.1f}-{100*r[chr(99)+chr(105)][1]:.1f}]':>22}"
          f"{r['per'][12]:>8}{r['per'][10]:>8}"
          f"   {'/'.join(str(r['det'][p]) for p in (6, 9, 11))}")

print(f"\n{LINE}\nPER-SENSOR FALSE ALARMS BY VARIANT\n{LINE}")
print(f"{'PAD':>4}" + "".join(f"{v:>7}" for v in ("V0", "V1", "V2", "V3")))
for pad in sorted(HEALTHY):
    print(f"{pad:>4}" + "".join(f"{res[v][1]['per'][pad]:>7}" for v in ("V0", "V1", "V2", "V3")))

print(f"\n{LINE}\nV2 ABSTENTION — how often is no trustworthy mate available?\n{LINE}")
w2 = res["V2"][0]
for pad in sorted(HEALTHY | DEGRADING):
    n_na = int(w2[pad].isna().sum())
    if n_na:
        print(f"  PAD {pad:>2} (head {HEAD_OF[pad]}): reference undefined in "
              f"{n_na}/{len(w2)} passages")
print("  A head whose every other member is flagged cannot form a pair feature.")
print("  The rule abstains rather than guessing - and in a 2-FBG head that means")
print("  ONE flagged neighbour silences it completely.")

for v in ("V0", "V1", "V2", "V3"):
    res[v][0].to_csv(OUT / f"pair_delta_{v}.csv")
print(f"\nwrote pair_delta_V0..V3.csv")

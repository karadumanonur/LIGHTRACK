"""Points 1 & 2 — does the pair rule survive the 2-FBG target configuration?

A-29 showed the leave-one-out MEDIAN cuts false alarms 7x and beats the oracle.
But that robustness needs three neighbours to take a median over, and the design
the project is arguing for has ONE. So POC1's headline pair-rule performance may
be a property of the 4-FBG layout it is trying to eliminate.

Configurations compared, all on normal-weekday records:

  V1   leave-one-out MEDIAN of the 3 head-mates      4-FBG, current best
  P2   the single spectral COMB PARTNER              2-FBG, the target design
  X8   median of all 8 sensors in the other 2 heads  cross-head, POC1-optimistic
  X4   median of a 2-per-head subset of the others   cross-head, 2-FBG realistic

The comb pairs (A-11) are the physical 2-FBG units:
  (ch01,ch02)=(PAD12,PAD11)  (ch03,ch04)=(PAD10,PAD9)  (ch05,ch06)=(PAD8,PAD7)
  (ch07,ch08)=(PAD6,PAD5)    (ch09,ch10)=(PAD4,PAD3)   (ch11,ch12)=(PAD2,PAD1)
Three of the six pairs contain a degraded member, which is exactly the case the
4->2 reduction has to survive.

Cross-head needs two things handled, both reported: heads sit at different
sleeper-bay positions (PAD 3 / PAD 5 already showed position-dependent lobe
depth), so every ratio is normalised to its own commissioning value; and the
amplitudes come from TOF-transferred timing, so the item-9 guards are applied
and the result is reported with and without them.
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
K_MON, TREND_WIN, CONFIRM_N, CONFIRM_K = 3.0, 5, 3, 2
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}
PARTNER = {12: 11, 11: 12, 10: 9, 9: 10, 8: 7, 7: 8,
           6: 5, 5: 6, 4: 3, 3: 4, 2: 1, 1: 2}
KEEP_2FBG = {1: [1, 2], 2: [5, 6], 3: [9, 10]}   # one comb pair per head
GUARD_OFFSET_FRAC = 0.80

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

# --- item 9 guards, from the Phase 1 run -----------------------------------
g = pd.read_csv(OUT / "phase1_amplitudes_raw.csv")[
    ["name", "pad", "offset_abs_p90", "search_half", "count_disagree", "n_indep"]]
df = df.merge(g, on=["name", "pad"], how="left", suffixes=("", "_p1"))
df["offset_frac"] = df["offset_abs_p90"] / df["search_half"]
df["guard_ok"] = (df["offset_frac"] < GUARD_OFFSET_FRAC) & \
                 (df["count_disagree_p1"].abs() <= 2)

o = (df[["name", "utc"]].drop_duplicates().sort_values("utc")
     .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
df = df.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
df["snr_trend"] = (df.groupby("pad")["snr_delta"]
                   .transform(lambda v: v.rolling(TREND_WIN, min_periods=2).median()))
N = df["qi"].nunique()

piv = df.pivot_table(index="qi", columns="pad", values="p2p_pm")
grd = df.pivot_table(index="qi", columns="pad", values="guard_ok").astype(bool)
print(f"normal-weekday passages: {N}")
print(f"item-9 guards: |offset|p90 < {GUARD_OFFSET_FRAC:.2f} x half-window "
      f"AND |cross-head count disagreement| <= 2")
print(f"  guard pass rate: {100*grd.mean().mean():.1f} % of sensor-passages")
print(f"  per head: " + "  ".join(
    f"H{h} {100*grd[[p for p in range(1,13) if HEAD_OF[p]==h]].mean().mean():.0f} %"
    for h in (1, 2, 3)))
# The two guards do different jobs and must be separable: the count guard
# rejects a head whose OWN detector over-counts, which for Head 3 is a property
# of its degraded members rather than of the transferred timing.
ga = (df.pivot_table(index="qi", columns="pad", values="offset_frac")
      < GUARD_OFFSET_FRAC)
gb = (df.pivot_table(index="qi", columns="pad", values="count_disagree_p1")
      .abs() <= 2)
for nm, gg in (("offset only", ga), ("count only", gb)):
    print(f"  {nm:<12} overall {100*gg.mean().mean():5.1f} %   " + "  ".join(
        f"H{h} {100*gg[[p for p in range(1,13) if HEAD_OF[p]==h]].mean().mean():3.0f} %"
        for h in (1, 2, 3)))


def ratio(kind: str, guarded: bool = False) -> pd.DataFrame:
    out = {}
    for pad in range(1, 13):
        if kind == "V1":
            mates = [q for q in range(1, 13) if HEAD_OF[q] == HEAD_OF[pad] and q != pad]
            ref = piv[mates].median(axis=1)
        elif kind == "P2":
            ref = piv[PARTNER[pad]]
        elif kind == "X8":
            mates = [q for q in range(1, 13) if HEAD_OF[q] != HEAD_OF[pad]]
            ref = piv[mates].median(axis=1)
        elif kind == "X4":
            mates = [q for h, ps in KEEP_2FBG.items() if h != HEAD_OF[pad] for q in ps]
            ref = piv[mates].median(axis=1)
        else:
            raise ValueError(kind)
        v = np.log2(piv[pad] / ref.replace(0, np.nan))
        if guarded:
            v = v.where(grd[pad])
        out[pad] = v
    w = pd.DataFrame(out)
    return w - w.apply(lambda c: c.dropna().iloc[0])      # commissioning baseline


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def evaluate(w: pd.DataFrame) -> dict:
    """LOSO false alarms, ABSTENTION-AWARE.

    A guarded configuration returns NaN where it refuses to decide. Counting
    those as "not flagged" silently rewards abstention: the guards reject 86 %
    of Head 3 sensor-passages, which is exactly where the false alarms live, so
    a naive denominator makes the guarded rate look like the unguarded one for
    the wrong reason. The rate below is over DECIDED passages only, and the
    abstention rate is reported next to it.
    """
    fa = dec = tot = 0
    per, per_dec = {}, {}
    for pad in sorted(HEALTHY):
        cal = HEALTHY - {pad}
        v = pd.concat([w[q] for q in cal]).dropna()
        m = v.median()
        thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
        tv = df[df["pad"].isin(cal)]["snr_trend"].dropna()
        tm = tv.median()
        tt = tm - K_MON * float(np.median(np.abs(tv - tm)) * 1.4826)
        trend = df[df["pad"] == pad].set_index("qi")["snr_trend"]
        # decided = the pair feature exists, or the trend rule alone can decide
        decided = w[pad].notna() | trend.notna()
        f = (w[pad] < thr) | (trend < tt)
        c = (f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K) & decided
        per[pad] = int(c.sum())
        per_dec[pad] = int(decided.sum())
        fa += int(c.sum())
        dec += int(decided.sum())
        tot += len(c)
    v = pd.concat([w[q] for q in HEALTHY]).dropna()
    m = v.median()
    thr = m - K_MON * float(np.median(np.abs(v - m)) * 1.4826)
    tv = df[df["pad"].isin(HEALTHY)]["snr_trend"].dropna()
    tt = tv.median() - K_MON * float(np.median(np.abs(tv - tv.median())) * 1.4826)
    det = {}
    for pad in sorted(DEGRADING):
        f = (w[pad] < thr) | (df[df["pad"] == pad].set_index("qi")["snr_trend"] < tt)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        det[pad] = int(c.index[c][0]) if c.any() else None
    return {"fa": fa, "tot": tot, "dec": dec, "ci": cp(fa, dec),
            "per": per, "per_dec": per_dec, "det": det}


CONF = [("V1", "LOO median of 3 head-mates", "4-FBG", False),
        ("P2", "single comb partner", "2-FBG TARGET", False),
        ("X8", "cross-head, all 8 others", "cross, optimistic", False),
        ("X4", "cross-head, 2-per-head", "cross, 2-FBG", False),
        ("X4", "cross-head, 2-per-head + guards", "cross, 2-FBG", True)]

print(f"\n{LINE}\nPAIR-REFERENCE CONFIGURATIONS\n{LINE}")
print(f"{'cfg':<5}{'reference':<32}{'layout':<19}{'FA / decided':>22}"
      f"{'abstain':>9}   det 6/9/11")
res = {}
for kind, label, layout, guarded in CONF:
    w = ratio(kind, guarded)
    r = evaluate(w)
    res[label] = (w, r)
    rate = 100 * r["fa"] / r["dec"]
    abst = 100 * (1 - r["dec"] / r["tot"])
    print(f"{kind:<5}{label:<32}{layout:<19}"
          f"{f'{r[chr(102)+chr(97)]}/{r[chr(100)+chr(101)+chr(99)]} = {rate:5.1f} % [{100*r[chr(99)+chr(105)][0]:.1f}-{100*r[chr(99)+chr(105)][1]:.1f}]':>22}"
          f"{abst:>8.1f}%   {'/'.join(str(r['det'][p]) for p in (6, 9, 11))}")

print(f"\n{LINE}\nPER-SENSOR FALSE ALARMS — who pays, and why\n{LINE}")
labels = [c[1] for c in CONF]
print(f"{'PAD':>4}{'partner':>9}{'partner state':<16}" + "".join(f"{l[:13]:>15}" for l in labels))
state = {p: ("DEGRADED" if p in DEGRADING else
             "borderline" if p in BORDERLINE else "healthy") for p in range(1, 13)}
for pad in sorted(HEALTHY):
    print(f"{pad:>4}{PARTNER[pad]:>9}{state[PARTNER[pad]]:<16}"
          + "".join(f"{res[l][1]['per'][pad]:>15}" for l in labels))

print(f"\n{LINE}\nTHE MECHANISM\n{LINE}")
print("The pair rule is ONE-SIDED: it fires on 'amplitude below reference'.")
print("So a partner that COLLAPSES pushes the healthy sensor's ratio UP and is")
print("harmless; a partner that AMPLIFIES pushes it DOWN and is a false alarm.")
print("PAD 9 amplifies to 2.6x before collapsing, so its partner PAD 10 is the")
print("sensor at risk in the 2-FBG configuration - and PAD 12, whose partner")
print("PAD 11 only ever collapses, should be clean.")
for pad in (10, 12, 5):
    w = res["single comb partner"][0][pad]
    print(f"  PAD {pad:>2} (partner {PARTNER[pad]:>2}, {state[PARTNER[pad]]:<10}): "
          f"pair_delta median {w.median():+.3f}, p5 {w.quantile(0.05):+.3f}, "
          f"min {w.min():+.3f}")

for k, (w, _) in res.items():
    w.to_csv(OUT / f"pair_delta_{k.replace(' ', '_').replace(',', '')}.csv")
print(f"\nwrote per-configuration pair_delta tables")

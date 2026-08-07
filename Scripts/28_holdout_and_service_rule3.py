"""Item 4 (weekend regression check) and the service-matched rule 3.

ITEM 4 is a REGRESSION CHECK, not a validation. The weekend holdout is 9
passages x 8 healthy sensors = 72 observations. At the weekday false-alarm rate
the expected count is under one, so the holdout can show that nothing broke and
nothing more. The binomial interval is reported next to the point estimate so
the resolution limit is visible rather than implied.

RULE 3, SERVICE-MATCHED. Repeated weekday services give a fixed stimulus: axle
count and spacing are constant and tare mass nearly so. Two levels, both
reported and labelled, because they control different things:

  slot level   n = 6-8 per service, 4 services   controls load AND geometry
  class level  n = 36                            controls geometry only

Built as an ADDITION to the existing peak/noise trend, not a replacement. If
service matching does not improve separation on PAD 6 and PAD 9, that is a
negative result and the simpler self-referential trend stays.
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

OUT = Path(CONFIG["paths"]["data_dir"]) and Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
K_MON, TREND_WIN, CONFIRM_N, CONFIRM_K = 3.0, 5, 3, 2
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}
KEEP = {1: [1, 2], 2: [5, 6], 3: [9, 10]}

pd.set_option("display.width", 240)
LINE = "=" * 106


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
cls = dict(zip(ebp["date"], ebp["day_class"]))
cvt = dict(zip(ebp["date"], ebp["cvt"]))

df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["date"] = df["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
df["day_class"] = df["date"].map(cls)
df["cvt"] = df["date"].map(cvt)
df["utc"] = df["name"].map(parse_filename_utc)

# ===================================================== ITEM 2 framing numbers
print(f"{LINE}\nITEM 2 — CVT AS AN OPERATING REGIME (framing for the holdout)\n{LINE}")
w = df.drop_duplicates("name")
print(f"{'day class':<18}{'passages':>10}{'days':>7}{'EBP CVT>0 days':>17}")
for k, g in w.groupby("day_class"):
    d = sorted(set(g["date"]))
    ncvt = sum(1 for x in d if cvt.get(x, 0) > 0)
    print(f"{k:<18}{len(g):>10}{len(d):>7}{ncvt:>17}")
print("\nCVT is zero on every normal weekday and non-zero only on 18/05, the")
print("Pentecost weekend (23-24/05), Whit Monday (25/05) and twice in June.")
print("It reads as a DIVERSION ROUTE, not a direction. So weekend records differ")
print("from weekday records in operating regime, not only in rolling stock, and")
print("CVT trains may not pass the POC1 site at all - which would mean the")
print("weekend EBP totals overstate the traffic the sensors actually saw.")
print("Meaning unconfirmed; a question for Infrabel (A-30).")

# ============================================================ ITEM 4
print(f"\n{LINE}\nITEM 4 — WEEKEND REGRESSION CHECK (not a validation)\n{LINE}")
nw = df[df["day_class"] == "normal_weekday"].copy()
we = df[df["day_class"] == "weekend"].copy()


def seq(x):
    o = (x[["name", "utc"]].drop_duplicates().sort_values("utc")
         .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
    y = x.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
    y["snr_trend"] = (y.groupby("pad")["snr_delta"]
                      .transform(lambda v: v.rolling(TREND_WIN, min_periods=2).median()))
    return y


nw, we = seq(nw), seq(we)
P_nw = nw.pivot_table(index="qi", columns="pad", values="p2p_pm")
P_we = we.pivot_table(index="qi", columns="pad", values="p2p_pm")
S1 = sorted(nw.loc[nw["session"] == 1, "qi"].unique())


def xratio(p2p, base):
    out = {}
    for pad in range(1, 13):
        mates = [q for h, ps in KEEP.items() if h != HEAD_OF[pad] for q in ps]
        out[pad] = np.log2(p2p[pad] / p2p[mates].median(axis=1).replace(0, np.nan))
    return pd.DataFrame(out) - base


base = None
w_nw_raw = {}
for pad in range(1, 13):
    mates = [q for h, ps in KEEP.items() if h != HEAD_OF[pad] for q in ps]
    w_nw_raw[pad] = np.log2(P_nw[pad] / P_nw[mates].median(axis=1).replace(0, np.nan))
W_nw = pd.DataFrame(w_nw_raw)
base = W_nw.loc[S1].median()
W_nw = W_nw - base
W_we = xratio(P_we, base)          # same commissioning baseline, weekday-derived

v = pd.concat([W_nw[q] for q in HEALTHY]).dropna()
thr_pair = v.median() - K_MON * float(np.median(np.abs(v - v.median())) * 1.4826)
tv = nw[nw["pad"].isin(HEALTHY)]["snr_trend"].dropna()
thr_trend = tv.median() - K_MON * float(np.median(np.abs(tv - tv.median())) * 1.4826)
print(f"thresholds calibrated on NORMAL WEEKDAY records only: "
      f"pair < {thr_pair:+.3f}, trend < {thr_trend:+.3f}")

fa = tot = 0
per = {}
for pad in sorted(HEALTHY):
    t = we[we["pad"] == pad].set_index("qi")["snr_trend"]
    f = (W_we[pad] < thr_pair) | (t < thr_trend)
    c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
    per[pad] = int(c.sum())
    fa += int(c.sum())
    tot += len(c)
lo, hi = cp(fa, tot)
wd_rate = 0.015
print(f"\nweekend holdout: {tot} observations "
      f"({we['qi'].nunique()} passages x {len(HEALTHY)} healthy sensors)")
print(f"false alarms   : {fa}/{tot} = {100*fa/tot:.1f} %  "
      f"[95 % CI {100*lo:.1f}-{100*hi:.1f} %]")
print(f"expected at the weekday cross-head rate ({100*wd_rate:.1f} %): "
      f"{wd_rate*tot:.1f}")
print(f"per sensor: {({p: n for p, n in per.items() if n}) or 'none'}")
print("\nRESOLUTION LIMIT: with an expected count under one, this holdout can")
print("distinguish 'nothing broke' from 'something broke badly' and nothing")
print("finer. The upper confidence bound alone spans the entire plausible range.")
print("It is a REGRESSION CHECK. It does not validate the weekday rate, and it")
print("is not evidence that the detector generalises across rolling stock.")

# ============================================================ RULE 3
print(f"\n{LINE}\nSERVICE-MATCHED RULE 3\n{LINE}")
seg = pd.read_csv(OUT / "train_segments_slotted.csv")[["name", "service", "slot"]]
m = nw.merge(seg, on="name", how="left")
print(f"weekday passages with a service label: "
      f"{m['slot'].notna().sum() // 12}/{nw['qi'].nunique()}")

REP_SLOTS = (m.drop_duplicates("name").groupby("slot").size()
             .pipe(lambda s: s[s >= 3]).index.tolist())
print(f"slots with >=3 repeats: {[int(s) for s in REP_SLOTS]}")

amp = m.pivot_table(index="qi", columns="pad", values="p2p_pm")
slot_of = m.drop_duplicates("qi").set_index("qi")["slot"]
svc_of = m.drop_duplicates("qi").set_index("qi")["service"]


def norm_and_report(label: str, group: pd.Series, keep=None) -> dict:
    """Normalise each sensor to its own commissioning value WITHIN each group."""
    a = amp.copy()
    if keep is not None:
        a = a[group.reindex(a.index).isin(keep)]
        gg = group.reindex(a.index)
    else:
        gg = pd.Series("all", index=a.index)
    out = {}
    for g, idx in a.groupby(gg).groups.items():
        blk = a.loc[idx]
        if len(blk) < 2:
            continue
        out[g] = blk / blk.iloc[0]
    if not out:
        return {}
    r = pd.concat(out.values()).sort_index()
    h = pd.concat([r[q] for q in HEALTHY]).dropna()
    mad = float(np.median(np.abs(h - h.median())) * 1.4826)
    res = {"label": label, "n": len(r), "healthy_mad": mad}
    for pad in (6, 9):
        s = r[pad].dropna()
        res[f"pad{pad}_final"] = float(s.iloc[-3:].median()) if len(s) >= 3 else np.nan
        res[f"pad{pad}_z"] = (h.median() - res[f"pad{pad}_final"]) / mad if mad else np.nan
    return res


rows = [
    norm_and_report("unmatched (all weekday passages)", pd.Series("all", index=amp.index)),
    norm_and_report("class level  (geometry only, n=36)", svc_of, keep=[0]),
    norm_and_report("slot level   (load + geometry)", slot_of, keep=REP_SLOTS),
]
rows = [r for r in rows if r]
t = pd.DataFrame(rows)
print(f"\n{'normalisation':<38}{'n':>6}{'healthy MAD':>13}"
      f"{'PAD 6 final':>13}{'PAD 6 z':>10}{'PAD 9 final':>13}{'PAD 9 z':>10}")
for _, r in t.iterrows():
    print(f"{r['label']:<38}{int(r['n']):>6}{r['healthy_mad']:>13.4f}"
          f"{r['pad6_final']:>13.3f}{r['pad6_z']:>10.1f}"
          f"{r['pad9_final']:>13.3f}{r['pad9_z']:>10.1f}")

print("\n'healthy MAD' is the dispersion the threshold must clear; lower is")
print("better separation. 'z' is how many MADs the degrading sensor sits below")
print("the healthy median at the end of the campaign; higher is better.")
best = t.loc[t["healthy_mad"].idxmin(), "label"]
print(f"\ntightest healthy dispersion: {best}")
if "unmatched" in best:
    print("=> NEGATIVE RESULT: service matching does not improve separation.")
    print("   The simpler self-referential trend rule stays. Likely cause: the")
    print("   dominant variance is speed and passenger load WITHIN a service,")
    print("   which repetition does not remove - speeds span 33-116 km/h inside")
    print("   a single slot (service census), so the speed normalisation was")
    print("   already doing the work that service matching was meant to add.")
else:
    print(f"=> service matching at the {best} level tightens dispersion.")
t.to_csv(OUT / "rule3_service_matched.csv", index=False)
print(f"\nwrote rule3_service_matched.csv")

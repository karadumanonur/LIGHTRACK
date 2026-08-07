"""Item 1 — three-way day classification, and what it does to Phase 4.

Item 0 split the campaign weekday/weekend. That was incomplete: **Whit Monday,
2026-05-25, is session 8** and was counted as a normal weekday. Ascension
(2026-05-14) was already outside the session series.

Both are corroborated by the EBP counts independently of the calendar: normal
weekdays run a median of 84 trains, weekends 35, and these two days run 38 and
53. 05-25 also carries the highest CVT of the whole period (18), matching the
weekend diversion pattern.

Reports, for three populations (all / weekday-incl-holiday / normal weekday):
thresholds, LOSO false-alarm rate with a Clopper-Pearson interval, and lead time.
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
ONSET_FRAC, TREND_WIN, CONFIRM_N, CONFIRM_K = 0.70, 5, 3, 2
K_MON, MON_RULES = 3.0, ["pair", "trend"]
RULES = [("shape", "neg_over_pos", True), ("pair", "pair_delta", False),
         ("trend", "snr_trend", False)]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

pd.set_option("display.width", 240)
LINE = "=" * 106

ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date

df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["utc"] = df["name"].map(parse_filename_utc)
df["local"] = df["utc"].map(to_local)
df["date"] = df["local"].map(lambda d: d.date())
df = df.merge(ebp[["date", "day_class", "total", "vns", "cvt"]], on="date", how="left")

# ------------------------------------------------------------- classification
print(f"{LINE}\nTHREE-WAY DAY CLASSIFICATION\n{LINE}")
s = (df.drop_duplicates("name")
     .groupby("session")
     .agg(date=("date", "first"), dow=("local", lambda x: DAYS[x.iloc[0].weekday()]),
          n_pass=("name", "count"), day_class=("day_class", "first"),
          ebp_total=("total", "first"), cvt=("cvt", "first")).reset_index())
s["capture_%"] = 100 * s["n_pass"] / s["ebp_total"]
print(s.to_string(index=False))
print("\nEBP median trains/day: "
      f"normal weekday {ebp.loc[ebp['day_class']=='normal_weekday','total'].median():.0f}, "
      f"weekend {ebp.loc[ebp['day_class']=='weekend','total'].median():.0f}, "
      f"holiday {list(ebp.loc[ebp['day_class']=='holiday','total'])}")
hol = s[s["day_class"] == "holiday"]
print(f"\nHOLIDAY SESSIONS INSIDE THE SERIES: "
      f"{list(hol['session'])} -> {list(hol['date'])} "
      f"({int(hol['n_pass'].sum())} passages)")

# ----------------------------------------------------------------- populations
POPS = {
    "all (item 0 original)": lambda d: d["day_class"].notna(),
    "weekday incl. holiday": lambda d: d["day_class"] != "weekend",
    "normal weekday only": lambda d: d["day_class"] == "normal_weekday",
}


def prep(sub: pd.DataFrame) -> pd.DataFrame:
    o = (sub[["name", "utc"]].drop_duplicates().sort_values("utc")
         .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
    x = sub.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
    x["snr_trend"] = (x.groupby("pad")["snr_delta"]
                      .transform(lambda v: v.rolling(TREND_WIN, min_periods=2).median()))
    x["amp_trend"] = (x.groupby("pad")["amp_rel"]
                      .transform(lambda v: v.rolling(TREND_WIN, min_periods=2).median()))
    return x


def thresholds(x: pd.DataFrame, cal: set[int], k: float) -> dict[str, float]:
    out = {}
    for tag, feat, upper in RULES:
        v = x[x["pad"].isin(cal)][feat].dropna()
        m = v.median()
        sd = np.median(np.abs(v - m)) * 1.4826
        out[tag] = m + k * sd if upper else m - k * sd
    return out


def flags(x: pd.DataFrame, pad: int, thr: dict, rules: list[str]) -> pd.Series:
    sub = x[x["pad"] == pad].set_index("qi").sort_index()
    f = pd.Series(False, index=sub.index)
    for tag, feat, upper in RULES:
        if tag in rules:
            f |= (sub[feat] > thr[tag]) if upper else (sub[feat] < thr[tag])
    return f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K


def cp_interval(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95 % interval for k successes in n trials."""
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


print(f"\n{LINE}\nEFFECT ON THE PHASE 4 NUMBERS\n{LINE}")
results = {}
for label, mask in POPS.items():
    x = prep(df[mask(df)].copy())
    npass = x["qi"].nunique()
    thr_all = thresholds(x, HEALTHY, K_MON)

    fa = tot = 0
    per = {}
    for pad in sorted(HEALTHY):
        thr = thresholds(x, HEALTHY - {pad}, K_MON)
        c = flags(x, pad, thr, MON_RULES)
        per[pad] = int(c.sum())
        fa += int(c.sum())
        tot += len(c)
    lo, hi = cp_interval(fa, tot)

    lead = {}
    for pad in sorted(DEGRADING | BORDERLINE):
        sub = x[x["pad"] == pad].set_index("qi").sort_index()
        bad = sub["amp_trend"] <= ONSET_FRAC
        onset = next((int(i) for i in sub.index if bad.loc[i] and bad.loc[i:].all()), None)
        c = flags(x, pad, thr_all, MON_RULES)
        fl = int(c.index[c][0]) if c.any() else None
        if onset is None or fl is None:
            lead[pad] = (onset, fl, None, None)
            continue
        tu = x.drop_duplicates("qi").set_index("qi")["utc"]
        lead[pad] = (onset, fl, onset - fl,
                     (tu.loc[onset] - tu.loc[fl]).total_seconds() / 3600)
    results[label] = dict(n=npass, thr=thr_all, fa=fa, tot=tot, ci=(lo, hi),
                          per=per, lead=lead)

    print(f"\n{label}  ({npass} passages)")
    print(f"  thresholds        : pair < {thr_all['pair']:+.3f}   "
          f"trend < {thr_all['trend']:+.3f}   shape > {thr_all['shape']:+.3f}")
    print(f"  LOSO false alarms : {fa}/{tot} = {100*fa/tot:.1f} %   "
          f"[95 % CI {100*lo:.1f}-{100*hi:.1f} %]")
    nz = {p: v for p, v in per.items() if v}
    print(f"  by sensor         : {nz if nz else 'none'}")
    for pad in sorted(DEGRADING | BORDERLINE):
        o, fl, lp, lh = lead[pad]
        tag = "" if pad not in DEGRADING else ""
        print(f"  PAD {pad:>2} lead        : "
              + (f"{lp:+d} passages, {lh:+.1f} h  (onset {o}, flag {fl})"
                 if lp is not None else f"onset {o}, flag {fl} — not derivable"))

print(f"\n{LINE}\nVERDICT\n{LINE}")
a, b, c = (results[k] for k in POPS)
print(f"{'population':<26}{'passages':>10}{'pair thr':>10}{'trend thr':>11}"
      f"{'LOSO FA':>18}{'PAD 6 lead':>18}")
for label in POPS:
    r = results[label]
    l6 = r["lead"][6]
    print(f"{label:<26}{r['n']:>10}{r['thr']['pair']:>10.3f}{r['thr']['trend']:>11.3f}"
          f"{f'{r[chr(102)+chr(97)]}/{r[chr(116)+chr(111)+chr(116)]} = {100*r[chr(102)+chr(97)]/r[chr(116)+chr(111)+chr(116)]:.1f} %':>18}"
          f"{f'{l6[2]:+d} pass / {l6[3]:+.1f} h' if l6[2] is not None else 'n/a':>18}")

pd.DataFrame([
    {"population": k, "n_passages": v["n"], "pair_thr": v["thr"]["pair"],
     "trend_thr": v["thr"]["trend"], "shape_thr": v["thr"]["shape"],
     "fa": v["fa"], "fa_tot": v["tot"], "fa_pct": 100 * v["fa"] / v["tot"],
     "ci_lo_pct": 100 * v["ci"][0], "ci_hi_pct": 100 * v["ci"][1],
     "pad6_lead_pass": v["lead"][6][2], "pad6_lead_h": v["lead"][6][3]}
    for k, v in results.items()
]).to_csv(OUT / "phase4_daytype_sensitivity.csv", index=False)
s.to_csv(OUT / "session_daytype.csv", index=False)
print(f"\nwrote phase4_daytype_sensitivity.csv, session_daytype.csv")

"""Item 0 — weekday/weekend contamination check on the Phase 4 lead times.

Lead times were computed on a chronologically mixed passage sequence. If any of
PAD 6's +7 passages or PAD 11's +1 fall on weekend records, they were measured
across a change of rolling stock and are not comparable.

Reports the contamination, then re-derives onset and flag on weekday-only
records. Nothing else is touched until this is answered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD, parse_filename_utc, to_local  # noqa: E402
from lightrack.health import DEGRADING, BORDERLINE, HEALTHY            # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
ONSET_FRAC, TREND_WIN, CONFIRM_N, CONFIRM_K = 0.70, 5, 3, 2
K_MON, MON_RULES = 3.0, ["pair", "trend"]
RULES = [("shape", "neg_over_pos", True), ("pair", "pair_delta", False),
         ("trend", "snr_trend", False)]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

pd.set_option("display.width", 240)
LINE = "=" * 104

df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["utc"] = df["name"].map(parse_filename_utc)
df["local"] = df["utc"].map(to_local)
df["dow"] = df["local"].map(lambda d: d.weekday())
df["weekend"] = df["dow"] >= 5

order = (df[["name", "pi", "utc", "local", "dow", "weekend"]].drop_duplicates("name")
         .sort_values("pi").reset_index(drop=True))

print(f"{LINE}\nPASSAGE INDEX -> DATE -> WEEKDAY  (local Belgian time)\n{LINE}")
print(f"{'pi':>4}  {'local date/time':<21}{'day':<5}{'sess':>5}  file")
sess_of = df.drop_duplicates("name").set_index("name")["session"].to_dict()
for _, r in order.iterrows():
    mark = "  <-- WEEKEND" if r["weekend"] else ""
    print(f"{int(r['pi']):>4}  {r['local']:%Y-%m-%d %H:%M:%S}    {DAYS[r['dow']]:<5}"
          f"{sess_of[r['name']]:>5}  {r['name']}{mark}")

n_we = int(order["weekend"].sum())
print(f"\npassages: {len(order)} total, {len(order)-n_we} weekday, {n_we} weekend")
print("\nsessions by weekday:")
s = df.drop_duplicates("name").groupby(["session"]).agg(
    date=("local", lambda x: x.iloc[0].date()),
    dow=("dow", "first"), n=("name", "count")).reset_index()
s["day"] = s["dow"].map(lambda d: DAYS[d])
s["weekend"] = s["dow"] >= 5
print(s[["session", "date", "day", "n", "weekend"]].to_string(index=False))
we_sessions = sorted(s.loc[s["weekend"], "session"].tolist())
print(f"\nweekend sessions: {we_sessions}")

# ------------------------------------------------- the specific lead windows
print(f"\n{LINE}\nCONTAMINATION OF THE REPORTED LEAD WINDOWS\n{LINE}")
WINDOWS = {6: (28, 35), 11: (5, 6), 7: (67, 68), 9: (0, 61)}
for pad, (a, b) in WINDOWS.items():
    seg = order[(order["pi"] >= a) & (order["pi"] <= b)]
    nwe = int(seg["weekend"].sum())
    print(f"\nPAD {pad}: flag at passage {a}, onset at {b} "
          f"-> {b-a} passages spanned, {nwe} of them weekend")
    if pad == 9:
        print("   (PAD 9 is caught at acceptance, so this span is not a "
              "monitoring lead — see item 5)")
        continue
    for _, r in seg.iterrows():
        print(f"     pi {int(r['pi']):>3}  {r['local']:%Y-%m-%d %H:%M}  "
              f"{DAYS[r['dow']]}{'   WEEKEND' if r['weekend'] else ''}")

# --------------------------------------------- re-derive on weekdays only
print(f"\n{LINE}\nRE-DERIVATION ON WEEKDAY-ONLY RECORDS\n{LINE}")
wd = df[~df["weekend"]].copy()
wo = (wd[["name", "utc"]].drop_duplicates().sort_values("utc")
      .reset_index(drop=True).reset_index().rename(columns={"index": "wpi"}))
wd = wd.merge(wo[["name", "wpi"]], on="name")
print(f"weekday passages: {wo.shape[0]} (was {len(order)} mixed)")

# recompute the passage-ordered rolling features on the weekday sequence
wd = wd.sort_values(["pad", "wpi"])
wd["snr_trend"] = (wd.groupby("pad")["snr_delta"]
                   .transform(lambda x: x.rolling(TREND_WIN, min_periods=2).median()))
wd["amp_trend"] = (wd.groupby("pad")["amp_rel"]
                   .transform(lambda x: x.rolling(TREND_WIN, min_periods=2).median()))


def thresholds(cal: set[int], k: float) -> dict[str, float]:
    out = {}
    for tag, feat, upper in RULES:
        v = wd[wd["pad"].isin(cal)][feat].dropna()
        m = v.median()
        sd = np.median(np.abs(v - m)) * 1.4826
        out[tag] = m + k * sd if upper else m - k * sd
    return out


thr = thresholds(HEALTHY, K_MON)
print(f"weekday-calibrated monitoring thresholds: "
      f"pair < {thr['pair']:+.3f}, trend < {thr['trend']:+.3f}")

print(f"\n{'PAD':>4}{'onset wpi':>11}{'flag wpi':>10}{'lead (pass)':>13}"
      f"{'lead (h)':>11}   vs mixed-sequence result")
MIXED = {6: (7, 47.8), 11: (1, 0.2), 7: (1, 0.2)}
rows = []
for pad in sorted(DEGRADING | BORDERLINE):
    sub = wd[wd["pad"] == pad].set_index("wpi").sort_index()
    bad = sub["amp_trend"] <= ONSET_FRAC
    onset = None
    for i in sub.index:
        if bad.loc[i] and bad.loc[i:].all():
            onset = int(i)
            break
    f = pd.Series(False, index=sub.index)
    for tag, feat, upper in RULES:
        if tag in MON_RULES:
            f |= (sub[feat] > thr[tag]) if upper else (sub[feat] < thr[tag])
    conf = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
    flag = int(conf.index[conf][0]) if conf.any() else None
    if onset is None or flag is None:
        print(f"{pad:>4}{str(onset):>11}{str(flag):>10}{'-':>13}{'-':>11}")
        continue
    tu = wo.set_index("wpi")
    hrs = (tu.loc[onset, "utc"] - tu.loc[flag, "utc"]).total_seconds() / 3600
    m = MIXED.get(pad)
    note = f"was +{m[0]} pass / +{m[1]:.1f} h" if m else "(acceptance)"
    print(f"{pad:>4}{onset:>11}{flag:>10}{onset-flag:>+13}{hrs:>+11.1f}   {note}")
    rows.append({"pad": pad, "onset_wpi": onset, "flag_wpi": flag,
                 "lead_pass": onset - flag, "lead_h": hrs})

pd.DataFrame(rows).to_csv(OUT / "phase4_lead_weekday.csv", index=False)
order.to_csv(OUT / "passage_calendar.csv", index=False)
print(f"\nwrote passage_calendar.csv and phase4_lead_weekday.csv")

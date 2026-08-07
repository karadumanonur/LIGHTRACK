"""Step 3, self-diagnosis — frozen thresholds, two clearly separated baselines.

FALSE-ALARM TEST   baseline = this set's own first day (2026-07-17).
                   Asks: do sensors that are STABLE across the 11 July days
                   raise a flag? Valid regardless of what happened in the gap.

CONTINUITY TEST    baseline = POC1 session 1 (2026-05-15).
                   Asks: does the detector catch what changed across the gap?
                   Valid only if nothing was re-bonded; confounded otherwise.

Frozen thresholds, unchanged: pair < -0.727, trend < -0.423.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD                 # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
THR_PAIR, THR_TREND = -0.727, -0.423
TREND_WIN, CONFIRM_N, CONFIRM_K = 5, 3, 2
KEEP2 = {1: [1, 2], 2: [5, 6], 3: [9, 10]}
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}
POC1_HEALTHY = {1, 2, 3, 4, 5, 8, 10, 12}
POC1_DEGRADING = {6, 9, 11}
pd.set_option("display.width", 250)
LINE = "=" * 116

s = pd.read_csv(OUT / "blind_sensors.csv")
r = pd.read_csv(OUT / "blind_records_scored.csv")
r["local"] = pd.to_datetime(r["local"]); r["date"] = r["local"].dt.date
d = s[s["diag"] == True].merge(r[["name", "local", "date", "gated"]], on="name")
d = d[d["gated"]].sort_values("local")
order = d[["name", "local"]].drop_duplicates().sort_values("local").reset_index(drop=True)
order["qi"] = order.index
d = d.merge(order[["name", "qi"]], on="name")

P2P = d.pivot_table(index="qi", columns="pad", values="p2p_pm")
SNR = d.pivot_table(index="qi", columns="pad", values="snr")
DAY1 = sorted(d.loc[d["date"] == d["date"].min(), "qi"].unique())

old = pd.read_csv(OUT / "phase4_features_scored.csv")
o1 = old[old["session"] == 1].groupby("pad").agg(
    p2p=("p2p_pm", "median"), snr=("snr", "median"))


def run(label: str, p2p_base: pd.Series, snr_base: pd.Series) -> pd.DataFrame:
    pair = {}
    for pad in range(1, 13):
        pool = [q for h, ps in KEEP2.items() if h != HEAD_OF[pad] for q in ps]
        ref = P2P[pool].median(axis=1)
        pair[pad] = np.log2(P2P[pad] / ref.replace(0, np.nan))
    W = pd.DataFrame(pair)
    # baseline the ratio the same way the frozen detector does
    ref0 = {}
    for pad in range(1, 13):
        pool = [q for h, ps in KEEP2.items() if h != HEAD_OF[pad] for q in ps]
        ref0[pad] = np.log2(p2p_base[pad] / np.median([p2p_base[q] for q in pool]))
    W = W - pd.Series(ref0)
    T = np.log2(SNR / snr_base).rolling(TREND_WIN, min_periods=2).median()

    rows = []
    for pad in range(1, 13):
        f = (W[pad] < THR_PAIR) | (T[pad] < THR_TREND)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        rows.append({"pad": pad, "ch": CH_OF_PAD[pad], "head": HEAD_OF[pad],
                     "poc1": ("healthy" if pad in POC1_HEALTHY else "DEGRADING"),
                     "flag_pct": 100 * c.mean(),
                     "pair_med": W[pad].median(), "trend_med": T[pad].median(),
                     "first_flag": int(c.index[c][0]) if c.any() else None})
    t = pd.DataFrame(rows)
    print(f"\n{label}")
    print(f"{'PAD':>4}{'ch':>6}{'head':>6}{'POC1 state':>12}"
          f"{'pair med':>11}{'trend med':>11}{'flagged %':>12}")
    for _, x in t.iterrows():
        print(f"{int(x['pad']):>4}{x['ch']:>6}{int(x['head']):>6}{x['poc1']:>12}"
              f"{x['pair_med']:>+11.2f}{x['trend_med']:>+11.2f}{x['flag_pct']:>11.1f}%")
    return t


print(f"{LINE}\nSELF-DIAGNOSIS ON UNSEEN DATA — FROZEN THRESHOLDS\n{LINE}")
print(f"passages (gate-retained): {len(P2P)}")

t_fa = run("FALSE-ALARM TEST  (baseline = 2026-07-17, this set's own first day)",
           P2P.loc[DAY1].median(), SNR.loc[DAY1].median())
t_co = run("CONTINUITY TEST   (baseline = POC1 session 1, 2026-05-15)",
           o1["p2p"], o1["snr"])

print(f"\n{LINE}\nREADING\n{LINE}")
alive = [1, 3, 4, 5, 10, 12]
dead = [2, 6, 7, 8, 9, 11]
fa = t_fa[t_fa["pad"].isin(alive)]["flag_pct"]
print(f"FALSE-ALARM TEST: of the 6 sensors stable across all 11 July days,")
print(f"   flagged fraction: min {fa.min():.1f} %, max {fa.max():.1f} %, "
      f"median {fa.median():.1f} %")
print(f"   sensors flagged in >5 % of passages: "
      f"{sorted(t_fa[(t_fa['pad'].isin(alive)) & (t_fa['flag_pct']>5)]['pad'].tolist())}")
print(f"\nCONTINUITY TEST: of the 6 sensors that collapsed, "
      f"{int((t_co[t_co['pad'].isin(dead)]['flag_pct']>50).sum())}/6 flagged in >50 % "
      f"of passages")
print(f"   NEW failures (healthy at POC1 end): PADs 2, 7, 8 -> flagged "
      + ", ".join(f"{t_co[t_co['pad']==p]['flag_pct'].iloc[0]:.0f} %" for p in (2, 7, 8)))
print(f"   of the 6 survivors, flagged: "
      f"{sorted(t_co[(t_co['pad'].isin(alive)) & (t_co['flag_pct']>5)]['pad'].tolist())}")
t_fa.to_csv(OUT / "blind_selfdiag_falsealarm.csv", index=False)
t_co.to_csv(OUT / "blind_selfdiag_continuity.csv", index=False)
print(f"\nwrote blind_selfdiag_*.csv")

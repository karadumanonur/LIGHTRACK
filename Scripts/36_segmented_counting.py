"""Items 1-4 — segmented speed adaptation, intervals, PAD 3 anchoring, PAD 5.

Four configurations are compared on identical records so the effect of each
change is separable:

  A  trough-anchored, single speed     the state at the last report
  B  peak-anchored,   single speed     item 3 - does removing the trough
                                       statistic close PAD 3's over-count?
  C  trough-anchored, segmented        item 1 in isolation
  D  peak-anchored,   segmented        both

Confidence intervals on the consist-size table use the number of distinct
TRAINS, not sensor-passages: eight sensors on one record are eight measurements
of the same train and are not independent. The point estimate is the
sensor-level rate; the interval is computed at the train-level n.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightrack.safety as SF                                  # noqa: E402
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, list_files, load_file, parse_filename_utc,
    to_local, rolling_median_baseline, estimate_speed, head_signal,
)
from lightrack.detect import passage_window                    # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
DATA = Path(CONFIG["paths"]["data_dir"])
OUT = Path(CONFIG["paths"]["out_dir"])
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}
pd.set_option("display.width", 250)
LINE = "=" * 116


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def run(anchor: str, segmented: bool) -> pd.DataFrame:
    SF.SAF = dict(CONFIG["safety"])
    SF.SAF["anchor_mode"] = anchor
    rows = []
    for p in list_files():
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        for h, pads in OPERATIONAL.items():
            hs = head_signal(y, h)
            w = passage_window(hs, spd.v_kmh)
            if w is None:
                continue
            hint = None
            if segmented:
                hint = SF.count_axles(hs[w[0]:w[1]], spd.v_kmh).idx_x
            for pad in pads:
                sig = y[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
                r = (SF.count_axles_segmented(sig, spd.v_kmh, idx_hint=hint)
                     if segmented else SF.count_axles(sig, spd.v_kmh))
                rows.append({"name": p.name, "head": h, "pad": pad,
                             "v_kmh": spd.v_kmh, "count": r.count,
                             "n_x": r.n_x, "n_d": r.n_d, "agree": r.agree})
    d = pd.DataFrame(rows)
    d["head_med"] = d.groupby(["name", "head"])["count"].transform("median")
    d["dev"] = d["count"] - d["head_med"]
    return d


CFG = [("A trough / single", "trough", False), ("B peak   / single", "peak", False),
       ("C trough / segmented", "trough", True), ("D peak   / segmented", "peak", True)]
res = {}
for lab, anc, seg in CFG:
    print(f"running {lab} ...", flush=True)
    res[lab] = run(anc, seg)

BINS = [0, 8, 16, 32, 300]
LABS = ["<=8", "9-16", "17-32", ">32"]

print(f"\n{LINE}\nITEMS 1-3 — OVER-COUNT BY CONSIST SIZE, FOUR CONFIGURATIONS\n{LINE}")
print("point estimate = sensor-level rate; interval = Clopper-Pearson at the")
print("number of distinct TRAINS, since 8 sensors on one record are not")
print("independent measurements.\n")
print(f"{'consist':<9}{'trains':>8}{'sensor-p':>10}" +
      "".join(f"{c[0]:>26}" for c in CFG))
for lab in LABS:
    line = f"{lab:<9}"
    first = True
    for i, (clab, _, _) in enumerate(CFG):
        d = res[clab]
        d = d.assign(bin=pd.cut(d["head_med"], BINS, labels=LABS))
        s = d[d["bin"] == lab]
        if s.empty:
            continue
        ntr = s["name"].nunique()
        over = (s["dev"] > 0).mean()
        k = int(round(over * ntr))
        lo, hi = cp(k, ntr)
        if first:
            line += f"{ntr:>8}{len(s):>10}"
            first = False
        line += f"{f'{100*over:5.1f} % [{100*lo:4.1f}-{100*hi:4.1f}]':>26}"
    print(line)

print(f"\n{LINE}\nITEM 3 — PAD 3's OVER-COUNT UNDER EACH ANCHORING\n{LINE}")
print(f"{'PAD':<5}{'ch':<7}{'lobe':>7}" + "".join(f"{c[0][:18]:>20}" for c in CFG))
lobe = (pd.read_csv(OUT / "phase4_features_scored.csv")
        .query("session == 1").groupby("pad")["neg_over_pos"].median())
for pad in [3, 5, 8, 1, 2, 4, 6, 7]:
    line = f"{pad:<5}{CH_OF_PAD[pad]:<7}{lobe[pad]:>7.2f}"
    for clab, _, _ in CFG:
        s = res[clab].query("pad == @pad")
        line += f"{100*(s['dev']>0).mean():>19.0f}%"
    print(line)
for clab, _, _ in CFG:
    d = res[clab]
    g = d.groupby("pad")["dev"].apply(lambda s: 100 * (s > 0).mean())
    print(f"{clab:<22} correlation(lobe, over-count) = "
          f"{pd.Series(lobe[g.index].values).corr(pd.Series(g.values)):+.3f}   "
          f"head1 {100*(d.query('head==1')['dev']>0).mean():.1f} %  "
          f"head2 {100*(d.query('head==2')['dev']>0).mean():.1f} %")

print(f"\n{LINE}\nOVERALL, ALL CONSISTS\n{LINE}")
print(f"{'config':<22}{'over %':>9}{'under %':>10}{'X/D agree %':>14}")
for clab, _, _ in CFG:
    d = res[clab]
    print(f"{clab:<22}{100*(d['dev']>0).mean():>9.1f}{100*(d['dev']<0).mean():>10.1f}"
          f"{100*d['agree'].mean():>14.1f}")
for clab, _, _ in CFG:
    res[clab].to_csv(OUT / f"seg_{clab.split()[0]}.csv", index=False)

# ============================================================ ITEM 4
print(f"\n{LINE}\nITEM 4 — PAD 5, ONE DIAGNOSTIC\n{LINE}")
d = res["B peak   / single"]
p5 = d.query("pad == 5")
worst = p5.assign(gap=(p5["n_d"] - p5["n_x"]).abs()).sort_values("gap", ascending=False)
name = worst.iloc[0]["name"]
print(f"most divergent record for PAD 5: {name}  "
      f"X={int(worst.iloc[0]['n_x'])} D={int(worst.iloc[0]['n_d'])}")
SF.SAF = dict(CONFIG["safety"]); SF.SAF["anchor_mode"] = "peak"
y = rolling_median_baseline(load_file(DATA / name)[CH_COLS])
spd = estimate_speed(y)
hs = head_signal(y, 2)
w = passage_window(hs, spd.v_kmh)
fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
for ax_, pad in zip(axes, (5, 8)):
    sig = y[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
    f = SF.lowpass(sig, spd.v_kmh)
    lv = SF.levels(f, spd.v_kmh)
    ix = SF.x_crossing(f, spd.v_kmh, lv)
    idd = SF.d_crossing(f, spd.v_kmh, lv)
    t = np.arange(len(f)) / FS
    ax_.plot(t, sig, lw=0.4, color="#bbb", label="raw")
    ax_.plot(t, f, lw=1.3, color="#333", label="LPF")
    for nm, val, c in (("thr_rise", lv.thr_rise, "#b2182b"),
                       ("thr_fall", lv.thr_fall, "#e08214"),
                       ("d_cut", lv.d_cut, "#2166ac")):
        ax_.axhline(val, ls="--", lw=1.0, color=c, label=nm)
    ax_.plot(t[ix], f[ix], "rv", ms=9, label=f"X  n={len(ix)}")
    ax_.plot(t[idd], f[idd], "b^", ms=6, label=f"D  n={len(idd)}")
    ax_.set_ylabel(f"PAD {pad} ({CH_OF_PAD[pad]})\nlobe {lobe[pad]:.2f}  [pm]", fontsize=9)
    ax_.grid(alpha=0.3); ax_.legend(fontsize=7, ncol=7, loc="upper right")
    print(f"  PAD {pad}: X={len(ix)} D={len(idd)}  peak={lv.peak:.1f} "
          f"trough={lv.trough:.1f} rise={lv.thr_rise:.1f} fall={lv.thr_fall:.1f} "
          f"dcut={lv.d_cut:.1f}")
axes[-1].set_xlabel("time in passage [s]")
fig.suptitle(f"PAD 5 vs its head-mate PAD 8 — {name}\n"
             "where X-crossing and D-crossing part company", fontsize=11)
fig.savefig(OUT / "pad5_diagnostic.png", dpi=140)
print(f"\nwrote pad5_diagnostic.png")

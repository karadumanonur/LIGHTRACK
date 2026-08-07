"""Investigate 20260519T093614 — Head 1 read 6.4 pm against Head 2's 78.9 pm.

A factor of 12 cannot come from sampling off a peak flank. Either Head 1 did
not see this passage at all, or the passage is clipped at the record boundary.
The distinction matters: a genuinely missed passage will resurface in Phase 5 as
a counting error and must be characterised now, not explained away.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                       # noqa: E402
    CONFIG, CH_COLS, load_file, rolling_median_baseline, estimate_speed,
    head_signal, axle_peaks, quiet_noise_floor, PAD_OF_CHANNEL,
)
from lightrack.detect import passage_window                   # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
NAME = "20260519T093614,145Z.txt"
p = Path(CONFIG["paths"]["data_dir"]) / NAME

df = load_file(p)
y = rolling_median_baseline(df[CH_COLS])
spd = estimate_speed(y)
n = len(y)
print(f"{NAME}: {n} samples = {n/FS:.1f} s, v={spd.v_kmh:.1f} km/h, "
      f"score={spd.score:.2f}, head1_first={spd.head1_first}")
print(f"lags: ab={spd.lag_ab} bc={spd.lag_bc}\n")

print(f"{'head':>5}{'peak pm':>10}{'noise pm':>10}{'n_axles':>9}"
      f"{'window s':>20}{'frac of record':>16}")
wins = {}
for h in (1, 2, 3):
    hs = head_signal(y, h)
    w = passage_window(hs, spd.v_kmh)
    wins[h] = w
    idx = axle_peaks(hs, spd.v_kmh, window=w)
    ws = f"{w[0]/FS:.1f}-{w[1]/FS:.1f}" if w else "none"
    print(f"{h:>5}{hs.max():10.1f}{quiet_noise_floor(hs):10.2f}{len(idx):9d}"
          f"{ws:>20}{(w[1]-w[0])/n if w else 0:16.3f}")

print(f"\n{'channel':>8}{'PAD':>5}{'head':>5}{'record max pm':>15}"
      f"{'max in H2 window':>18}{'p99.9 pm':>10}")
w2 = wins[2]
for ch in CH_COLS:
    sig = y[ch].to_numpy()
    inw = sig[w2[0]:w2[1]].max() if w2 else np.nan
    print(f"{ch:>8}{PAD_OF_CHANNEL[ch]:>5}{(3 if ch<'ch05' else 2 if ch<'ch09' else 1):>5}"
          f"{sig.max():15.1f}{inw:18.1f}{np.percentile(sig,99.9):10.1f}")

# where in the record is each head's energy?
print("\nEnergy profile — max |signal| per 10 s block, per head:")
blk = int(10 * FS)
print(f"{'block s':>10}" + "".join(f"{'H'+str(h):>10}" for h in (1, 2, 3)))
for b in range(0, n, blk):
    row = f"{b/FS:>7.0f}-{min(n,b+blk)/FS:<3.0f}"
    for h in (1, 2, 3):
        hs = head_signal(y, h)
        row += f"{np.abs(hs[b:b+blk]).max():10.1f}"
    print(row)

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
fig.subplots_adjust(hspace=0.22, top=0.9, bottom=0.08, left=0.08, right=0.98)
t = np.arange(n) / FS
for ax_, h, col in zip(axes, (1, 2, 3), ("#1a7f37", "#2166ac", "#b2182b")):
    hs = head_signal(y, h)
    ax_.plot(t, hs, lw=0.5, color=col)
    if wins[h]:
        ax_.axvspan(wins[h][0]/FS, wins[h][1]/FS, color=col, alpha=0.12)
    ax_.set_ylabel(f"Head {h}\n(norm. mean)", fontsize=9)
    ax_.grid(alpha=0.3, lw=0.4)
    ax_.margins(x=0)
axes[-1].set_xlabel("time within record [s]", fontsize=10)
fig.suptitle(f"{NAME} — why did Head 1 read 6.4 pm?\n"
             f"robust head-mean traces, baseline-removed, v={spd.v_kmh:.0f} km/h",
             fontsize=12)
png = OUT / "investigate_093614.png"
fig.savefig(png, dpi=140)
print(f"\nwrote {png}")

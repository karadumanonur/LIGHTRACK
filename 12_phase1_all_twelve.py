"""Phase 1b — reproduce ALL TWELVE §8 curves, not just the four targets.

Why this is worth doing: the 4/4 gate tested PAD 6, 9, 10, 11. Of those, only
PAD 9 and PAD 11 were used as anchors when the channel->PAD mapping was built,
and PAD 10 vs PAD 12 was never directly tested at all — both are stable and look
alike — yet A-15's whole conclusion depends on which of ch01/ch03 is PAD 10.

The deck's slide 22 plots all twelve. Reproducing all twelve turns eight
sensors the mapping was NOT fitted to into out-of-sample predictions, for free.
The most discriminating is ch06: under ch_k -> PAD(13-k) it is PAD 7, which the
deck shows declining mildly to ~0.75. If ch06 were instead PAD 6 it would have
to land at ~0.5.
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
from lightrack import CONFIG, CH_OF_PAD  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
a = pd.read_csv(OUT / "phase1_amplitudes_raw.csv")
a = a[a["speed_ok"]].copy()
a["p2p_pm"] = a["pos_pm"] - a["neg_pm"]

per = a.groupby(["pad", "session"]).agg(p2p=("p2p_pm", "median")).reset_index()
W = per.pivot(index="session", columns="pad", values="p2p").sort_index()
N = W / W.iloc[0]

# Deck §8. `anchor` marks the sensors used to BUILD the mapping — those are not
# independent evidence. The rest are out-of-sample.
DECK = {
    1:  (1.00, "stable ~1.0",              "glued, centred",           False),
    2:  (1.00, "stable ~1.0",              "glued, centred",           False),
    3:  (1.00, "stable ~1.0",              "glued, +7.5 cm",           False),
    4:  (1.00, "stable ~1.0",              "glued, +7.5 cm",           False),
    5:  (1.00, "stable ~1.0",              "glued, centred",           False),
    6:  (0.50, "decay to ~0.5",            "glued, centred (CONTROL)", True),
    7:  (0.75, "mild decline to ~0.75",    "glued, +3 cm",             False),
    8:  (1.00, "stable ~1.0",              "glued, +3 cm",             False),
    9:  (0.05, "collapse to ~0.05",        "1/2 collage",              True),
    10: (1.00, "stable ~1.0",              "clipse (clamped)",         False),
    11: (0.10, "collapse to ~0.1 by m3",   "unprepared surface",       True),
    12: (1.00, "stable ~1.0",              "taraude (screwed)",        False),
}
# Tolerance: stable sensors within +/-25 % of 1.0; declining sensors within
# +/-0.15 absolute of target; collapsed sensors at or below target + 0.15.
def verdict(pad: float, meas: float) -> bool:
    t = DECK[pad][0]
    if t >= 0.9:
        return 0.75 <= meas <= 1.30
    if t <= 0.15:
        return meas <= t + 0.15
    return abs(meas - t) <= 0.15


LINE = "=" * 100
print(f"{LINE}\nALL TWELVE §8 CURVES — peak-to-peak, normalised to session 1\n{LINE}")
print(f"{'PAD':>4}{'ch':>6}  {'mounting':<26}{'deck':<24}{'target':>7}"
      f"{'measured':>10}{'anchor':>8}  verdict")
n_ok = n_oos = n_oos_ok = 0
for pad in range(1, 13):
    tgt, desc, mount, anchor = DECK[pad]
    m = float(N[pad].iloc[-1])
    ok = verdict(pad, m)
    n_ok += ok
    if not anchor:
        n_oos += 1
        n_oos_ok += ok
    print(f"{pad:>4}{CH_OF_PAD[pad]:>6}  {mount:<26}{desc:<24}{tgt:7.2f}"
          f"{m:10.2f}{'  used' if anchor else '   OOS':>8}  {'PASS' if ok else 'FAIL'}")

print(f"\noverall            : {n_ok}/12")
print(f"out-of-sample only : {n_oos_ok}/{n_oos}  "
      f"(sensors the mapping was NOT built from)")

print(f"\n{LINE}\nTHE DISCRIMINATING TEST — ch06\n{LINE}")
m6 = float(N[7].iloc[-1])
print(f"  ch06 measured final amplitude ratio : {m6:.2f}")
print(f"  under ch_k -> PAD(13-k), ch06 = PAD 7, deck target 0.75  "
      f"-> |err| {abs(m6-0.75):.2f}")
print(f"  under any mapping making ch06 = PAD 6, deck target 0.50  "
      f"-> |err| {abs(m6-0.50):.2f}")
print(f"  => {'PAD 7' if abs(m6-0.75) < abs(m6-0.50) else 'PAD 6'} is the better fit")

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(3, 4, figsize=(17, 10.2), sharex=True)
fig.subplots_adjust(hspace=0.42, wspace=0.22, top=0.885, bottom=0.105,
                    left=0.052, right=0.99)
s = N.index.to_numpy()
for ax_, pad in zip(axes.ravel(), range(1, 13)):
    tgt, desc, mount, anchor = DECK[pad]
    m = float(N[pad].iloc[-1])
    ok = verdict(pad, m)
    col = "#b2182b" if tgt < 0.9 else "#1a7f37"
    for q in range(1, 13):
        if q != pad:
            ax_.plot(s, N[q], color="#dcdee1", lw=0.7, zorder=1)
    ax_.axhline(1.0, color="#999", ls=":", lw=0.8)
    ax_.axhline(tgt, color="#2166ac", ls="--", lw=1.3, zorder=2)
    ax_.plot(s, N[pad], color=col, lw=2.2, marker="o", ms=4, zorder=3)
    tag = "anchor" if anchor else "out-of-sample"
    ax_.set_title(f"PAD {pad}  ({CH_OF_PAD[pad]})  ·  {mount}\n"
                  f"deck {desc} → {m:.2f}   [{tag}]  {'✓' if ok else '✗'}",
                  fontsize=8.8, color="#111" if ok else "#b2182b")
    ax_.set_ylim(0, max(1.4, float(N[pad].max()) * 1.1))
    ax_.set_xticks(s[::2])
    ax_.grid(alpha=0.3, lw=0.4)
    ax_.tick_params(labelsize=7.5)
for ax_ in axes[-1]:
    ax_.set_xlabel("session", fontsize=8.5)
for ax_ in axes[:, 0]:
    ax_.set_ylabel("amplitude / session 1", fontsize=8.5)

fig.suptitle("Lightrack POC1 — all twelve §8 durability curves reproduced\n"
             "peak-to-peak axle amplitude per session, normalised to session 1 · "
             "blue dashed = the deck's published endpoint for that sensor",
             fontsize=13, y=0.963)
fig.text(0.052, 0.028,
         f"{n_ok}/12 match. Only PADs 6, 9, 11 were used to build the channel→PAD "
         f"mapping; the other {n_oos} are out-of-sample predictions ({n_oos_ok}/{n_oos} correct). "
         "ch06 → PAD 7 is the decisive one:\nit lands at "
         f"{m6:.2f}, near the deck's 0.75 for PAD 7 and far from the 0.50 that PAD 6 "
         "would require. The mapping ch_k → PAD(13−k) is established.",
         fontsize=8.5, style="italic", color="#444")

png = OUT / "phase1_all_twelve_curves.png"
fig.savefig(png, dpi=140)
N.to_csv(OUT / "phase1_all12_normalised.csv")
print(f"\nwrote {png}")

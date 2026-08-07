"""Items 1-2 — the mechanism behind the >32-axle failure, and the filter path.

ITEM 1. Ten trains is too few to measure a rate but may be enough to identify a
cause. Three checks on the long consists specifically:
  (a) WHERE the miscounts accumulate along the consist. Uniform => threshold
      problem; concentrated => structural.
  (b) Whether the inter-axle SPACING distribution differs from class-0. The
      minimum-separation criterion is derived from a 2.0 m assumption tuned
      against 2.8 m passenger bogies; freight geometry is not that.
  (c) Whether the head disagreement is SYMMETRIC. Symmetric errors point at the
      train, asymmetric at the sensors.

ITEM 2. A-39 left one residual path: the compression lobe sits next to the peak,
so low-passing the signed trace lets it distort the peak before any threshold
applies. Test half-wave rectification before the filter and report
corr(lobe depth, over-count) before and after.
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

occ = pd.read_csv(OUT / "occupancy_balance.csv")
occ["local"] = pd.to_datetime(occ["local"])
LONG = occ[(occ["c_h1"] > 32) & ~(occ["touch_start"] | occ["touch_end"])]
print(f"long consists (>32 axles, not edge-clipped): {len(LONG)}")
for _, r in LONG.iterrows():
    print(f"  {r['local']:%Y-%m-%d %H:%M:%S}  H1={int(r['c_h1'])} "
          f"H2={int(r['c_h2'])}  bal {int(r['balance']):+d}  v={r['v_kmh']:.0f}")

SF.SAF = dict(CONFIG["safety"])


def head_axles(y, h, v):
    hs = head_signal(y, h)
    w = passage_window(hs, v)
    per = {}
    for pad in OPERATIONAL[h]:
        sig = y[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
        per[pad] = SF.count_axles(sig, v)
    return w, per


# ================================================================= 1(a)
print(f"\n{LINE}\n1(a) WHERE ALONG THE CONSIST DO THE MISCOUNTS ACCUMULATE?\n{LINE}")
print("Axles binned into deciles of the passage; count per decile per head.\n")
fig, axes = plt.subplots(1, len(LONG), figsize=(4.6 * len(LONG), 4.2), squeeze=False)
for ax_, (_, r) in zip(axes[0], LONG.iterrows()):
    y = rolling_median_baseline(load_file(DATA / r["name"])[CH_COLS])
    v = estimate_speed(y).v_kmh
    dec = {}
    for h in (1, 2):
        w, per = head_axles(y, h, v)
        idx = per[OPERATIONAL[h][np.argmin([abs(len(per[p].idx_x) - r[f"c_h{h}"])
                                            for p in OPERATIONAL[h]])]].idx_x
        n = w[1] - w[0]
        dec[h] = np.histogram(idx, bins=np.linspace(0, n, 11))[0]
    d = dec[1].astype(int) - dec[2].astype(int)
    print(f"{r['local']:%m-%d %H:%M}  H1 per decile {list(dec[1])}")
    print(f"{'':<12}  H2 per decile {list(dec[2])}")
    print(f"{'':<12}  diff          {list(d)}   cumulative {list(np.cumsum(d))}")
    ax_.bar(np.arange(1, 11), np.cumsum(d), color="#b2182b")
    ax_.axhline(0, color="k", lw=0.8)
    ax_.set_title(f"{r['local']:%m-%d %H:%M}\nH1 {int(r['c_h1'])} / H2 "
                  f"{int(r['c_h2'])}", fontsize=9)
    ax_.set_xlabel("decile along consist", fontsize=8)
    ax_.set_ylabel("cumulative H1 − H2", fontsize=8)
    ax_.grid(alpha=0.3)
fig.suptitle("Where the head-to-head count difference accumulates along long consists",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "long_consist_accumulation.png", dpi=130)

# ================================================================= 1(b)
print(f"\n{LINE}\n1(b) INTER-AXLE SPACING — freight vs class-0\n{LINE}")
seg = pd.read_csv(OUT / "train_segments.csv")
gaps_all = np.load(OUT / "train_gaps.npy", allow_pickle=True)
c0 = np.concatenate([gaps_all[i] for i in seg.index[seg["service"] == 0]])
print(f"class-0 (12-axle units, n={len(c0)} gaps): "
      f"short mode {np.median(c0[c0 < 5]):.2f} m, "
      f"p5 {np.percentile(c0, 5):.2f}, min {c0.min():.2f}")

longg = []
for _, r in LONG.iterrows():
    y = rolling_median_baseline(load_file(DATA / r["name"])[CH_COLS])
    v = estimate_speed(y).v_kmh
    w, per = head_axles(y, 1, v)
    idx = per[1].idx_x
    if len(idx) > 4:
        longg.append(np.diff(idx) / FS * (v / 3.6))
lg = np.concatenate(longg) if longg else np.array([])
print(f"long consists (n={len(lg)} gaps):        "
      f"short mode {np.median(lg[lg < 5]):.2f} m, "
      f"p5 {np.percentile(lg, 5):.2f}, min {lg.min():.2f}")
sep_m = CONFIG["detect"]["min_axle_spacing_m"] * CONFIG["safety"].get(
    "d_points_per_peak_fraction", 1) * 0 + CONFIG["detect"]["min_axle_spacing_m"]
print(f"\nminimum-separation criterion assumes {sep_m:.1f} m "
      f"(x{CONFIG['detect']['separation_fraction']} = "
      f"{sep_m*CONFIG['detect']['separation_fraction']:.2f} m effective)")
print(f"class-0 gaps below that: {100*(c0 < sep_m*0.7).mean():.1f} %")
print(f"long-consist gaps below: {100*(lg < sep_m*0.7).mean():.1f} %")
print(f"\ngap histogram, long consists (m):")
h, e = np.histogram(lg[lg < 25], bins=np.arange(0, 25, 1.0))
for c, lo in zip(h, e):
    if c:
        print(f"  {lo:4.1f}-{lo+1:4.1f}  {'#'*min(60, int(60*c/h.max())):<60} {c}")

# ================================================================= 1(c)
print(f"\n{LINE}\n1(c) IS THE HEAD DISAGREEMENT SYMMETRIC?\n{LINE}")
print(f"{'record':<18}{'H1':>5}{'H2':>5}{'bal':>6}   per-sensor counts H1 | H2")
for _, r in LONG.iterrows():
    y = rolling_median_baseline(load_file(DATA / r["name"])[CH_COLS])
    v = estimate_speed(y).v_kmh
    c = {}
    for h in (1, 2):
        _, per = head_axles(y, h, v)
        c[h] = [per[p].count for p in OPERATIONAL[h]]
    print(f"{r['local']:%m-%d %H:%M}  {int(r['c_h1']):>5}{int(r['c_h2']):>5}"
          f"{int(r['balance']):>+6}   {c[1]} | {c[2]}")
print("\nWide per-sensor spread within BOTH heads means the disagreement is not")
print("one bad sensor - it is every sensor struggling on the same train.")

# ================================================================= ITEM 2
print(f"\n{LINE}\nITEM 2 — DOES RECTIFYING BEFORE THE FILTER CLOSE THE LOBE PATH?\n{LINE}")
lobe = (pd.read_csv(OUT / "phase4_features_scored.csv")
        .query("session == 1").groupby("pad")["neg_over_pos"].median())


def sweep(rectify: bool) -> pd.DataFrame:
    SF.SAF = dict(CONFIG["safety"])
    SF.SAF["anchor_mode"] = "peak"
    SF.SAF["rectify_before_filter"] = rectify
    rows = []
    for p in list_files():
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        v = estimate_speed(y).v_kmh
        for h, pads in OPERATIONAL.items():
            hs = head_signal(y, h)
            w = passage_window(hs, v)
            if w is None:
                continue
            for pad in pads:
                sig = y[CH_OF_PAD[pad]].to_numpy()[w[0]:w[1]]
                r = SF.count_axles(sig, v)
                rows.append({"name": p.name, "head": h, "pad": pad,
                             "count": r.count, "agree": r.agree})
    d = pd.DataFrame(rows)
    d["head_med"] = d.groupby(["name", "head"])["count"].transform("median")
    d["dev"] = d["count"] - d["head_med"]
    return d


for rect in (False, True):
    d = sweep(rect)
    g = d.groupby("pad")["dev"].apply(lambda s: 100 * (s > 0).mean())
    corr = pd.Series(lobe[g.index].values).corr(pd.Series(g.values))
    print(f"\nrectify_before_filter = {rect}")
    print(f"  corr(lobe depth, over-count) = {corr:+.3f}")
    print(f"  over {100*(d['dev']>0).mean():.1f} %   under "
          f"{100*(d['dev']<0).mean():.1f} %   X/D agree {100*d['agree'].mean():.1f} %")
    print(f"  PAD 3 over-count {g[3]:.0f} %   head1 "
          f"{100*(d.query('head==1')['dev']>0).mean():.1f} %  head2 "
          f"{100*(d.query('head==2')['dev']>0).mean():.1f} %")
    d.to_csv(OUT / f"rectify_{int(rect)}.csv", index=False)
print(f"\nwrote long_consist_accumulation.png, rectify_*.csv")

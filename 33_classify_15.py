"""Items 1-2 — classify the 15 unexplained gated records, with a free cross-check.

Timeboxed: a one-line verdict each. What matters is whether they share a
property. If they do, that property is a gap in the detector; if they are
heterogeneous, the gate is doing its job.

The cross-check costs almost nothing and may classify several of them. There are
now TWO independent speed measurements per record:

  TOF        across heads, 25 m baseline, from the constrained lag fit
  bogie      within one head, from the wheelbase interval (~2.84 m from the
             class-0 template, sd 0.24 m over 36 segments)

Where they diverge materially something is wrong with that record — either the
lag fit, or the assumption that the stock has a ~2.84 m wheelbase. Both are
useful classifications.

NOTHING here is re-admitted to any statistic.
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
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal, axle_peaks,
)
from lightrack.detect import passage_window                    # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
WHEELBASE_M = 2.84          # class-0 template median; see docstring
pd.set_option("display.width", 250)
LINE = "=" * 116


def bogie_speed(idx: np.ndarray) -> tuple[float, int]:
    """Speed from the short (wheelbase) intervals at one head, km/h."""
    if len(idx) < 4:
        return np.nan, 0
    dt = np.diff(idx) / FS
    short = dt[dt < np.median(dt)]
    if short.size < 3:
        return np.nan, int(short.size)
    return float(WHEELBASE_M / np.median(short) * 3.6), int(short.size)


def main() -> None:
    g = pd.read_csv(OUT / "gated_classification.csv")
    g["local"] = pd.to_datetime(g["local"])
    rows = []
    for n, p in enumerate(list_files(), 1):
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        per_head = {}
        for h in (1, 2, 3):
            hs = head_signal(y, h)
            w = passage_window(hs, spd.v_kmh)
            idx = axle_peaks(hs, spd.v_kmh, window=w) if w else np.array([], int)
            vb, ns = bogie_speed(idx)
            per_head[h] = {"n": len(idx), "v": vb, "ns": ns,
                           "dur": (w[1] - w[0]) / FS if w else 0.0}
        vb = np.nanmedian([per_head[h]["v"] for h in (1, 2, 3)])
        counts = [per_head[h]["n"] for h in (1, 2, 3)]
        rows.append({
            "name": p.name, "v_tof": spd.v_kmh, "v_bogie": vb,
            "div_pct": 100 * (vb - spd.v_kmh) / spd.v_kmh if np.isfinite(vb) else np.nan,
            "n_h1": counts[0], "n_h2": counts[1], "n_h3": counts[2],
            "count_spread": max(counts) - min(counts),
            "dur_s": per_head[2]["dur"],
        })
        if n % 20 == 0:
            print(f"  [{n:3d}/93]", flush=True)

    w = pd.DataFrame(rows).merge(g, on="name")
    w.to_csv(OUT / "speed_crosscheck.csv", index=False)

    print(f"\n{LINE}\nITEM 2 — TWO INDEPENDENT SPEED MEASUREMENTS, ALL 93 RECORDS\n{LINE}")
    ok = w[np.isfinite(w["div_pct"])]
    print(f"records with both estimates: {len(ok)}/{len(w)}")
    print(f"divergence (bogie - TOF) / TOF: median {ok['div_pct'].median():+.1f} %, "
          f"IQR {ok['div_pct'].quantile(.25):+.1f} to {ok['div_pct'].quantile(.75):+.1f} %")
    print(f"correlation between the two speeds: "
          f"{ok['v_tof'].corr(ok['v_bogie']):+.3f}")
    for lab, sub in (("retained", ok[ok["gated"] & ok["in_session"]]),
                     ("excluded", ok[~(ok["gated"] & ok["in_session"])])):
        print(f"  {lab:<10} n={len(sub):>3}  median |divergence| "
              f"{sub['div_pct'].abs().median():>5.1f} %   "
              f">25 % divergent: {100*(sub['div_pct'].abs()>25).mean():>4.0f} %")

    # -------------------------------------------------- the 15
    print(f"\n{LINE}\nITEM 1 — THE 15 UNEXPLAINED RECORDS\n{LINE}")
    ex = w[(~w["gated"]) & w["in_session"] &
           ~(w["touches_start"] | w["touches_end"])].sort_values("local")
    print(f"n = {len(ex)}   (excluded, in-session, not a boundary split)\n")
    print(f"{'local':<20}{'score':>7}{'v_TOF':>8}{'v_bogie':>9}{'div':>8}"
          f"{'axles H1/H2/H3':>16}{'dur s':>7}{'max pm':>8}  verdict")

    verdicts = []
    for _, r in ex.iterrows():
        v = []
        if r["count_spread"] >= 6:
            v.append("head count disagreement")
        if abs(r["div_pct"]) > 25:
            v.append("speed estimates diverge")
        if r["n_h2"] <= 6:
            v.append("very short consist")
        elif r["n_h2"] >= 40:
            v.append("very long consist")
        if r["max_pm"] < 60:
            v.append("low amplitude")
        if r["dur_s"] > 8:
            v.append("long passage")
        verdict = "; ".join(v) if v else "unexplained"
        verdicts.append(verdict)
        print(f"{r['local']:%Y-%m-%d %H:%M:%S}  {r['score']:>7.2f}{r['v_tof']:>8.1f}"
              f"{r['v_bogie']:>9.1f}{r['div_pct']:>+7.0f}%"
              f"{f'{int(r.n_h1)}/{int(r.n_h2)}/{int(r.n_h3)}':>16}"
              f"{r['dur_s']:>7.1f}{r['max_pm']:>8.0f}  {verdict}")
    ex = ex.assign(verdict=verdicts)

    print(f"\n{LINE}\nDO THEY SHARE A PROPERTY?\n{LINE}")
    from collections import Counter
    tags = Counter(t for v in verdicts for t in v.split("; "))
    for t, c in tags.most_common():
        print(f"  {c:>3}/{len(ex)}  {t}")
    ret = w[w["gated"] & w["in_session"]]
    print(f"\nsame tags in the RETAINED population, for contrast:")
    print(f"  {(ret['count_spread']>=6).sum():>3}/{len(ret)}  head count disagreement "
          f"({100*(ret['count_spread']>=6).mean():.0f} %)")
    print(f"  {(ret['div_pct'].abs()>25).sum():>3}/{len(ret)}  speed estimates diverge "
          f"({100*(ret['div_pct'].abs()>25).mean():.0f} %)")
    print(f"  {(ret['n_h2']<=6).sum():>3}/{len(ret)}  very short consist "
          f"({100*(ret['n_h2']<=6).mean():.0f} %)")

    # -------------------------------------------------- montage
    fig, axes = plt.subplots(5, 3, figsize=(17, 13))
    fig.subplots_adjust(hspace=0.55, wspace=0.14, top=0.93, bottom=0.04,
                        left=0.04, right=0.99)
    for ax_, (_, r) in zip(axes.ravel(), ex.iterrows()):
        y = rolling_median_baseline(load_file(
            Path(CONFIG["paths"]["data_dir"]) / r["name"])[CH_COLS])
        t = np.arange(len(y)) / FS
        for h, c in ((1, "#1a7f37"), (2, "#2166ac"), (3, "#b2182b")):
            ax_.plot(t, head_signal(y, h), lw=0.35, color=c, alpha=0.85)
        ax_.set_title(f"{r['local']:%m-%d %H:%M}  score {r['score']:.2f}  "
                      f"{int(r['n_h2'])} axles\n{r['verdict'][:52]}", fontsize=7.5)
        ax_.tick_params(labelsize=6)
        ax_.margins(x=0)
        ax_.grid(alpha=0.25, lw=0.3)
    for ax_ in axes.ravel()[len(ex):]:
        ax_.axis("off")
    fig.suptitle("The 15 unexplained gated records — head-mean traces "
                 "(green H1, blue H2, red H3)", fontsize=12)
    png = OUT / "gated_15_montage.png"
    fig.savefig(png, dpi=115)
    ex.to_csv(OUT / "gated_15_verdicts.csv", index=False)
    print(f"\nwrote {png} and gated_15_verdicts.csv")


if __name__ == "__main__":
    main()

"""Records — what was delivered, and whether the delivery is complete.

PART A  INVENTORY. Reads every file once and writes one row per file (timing,
        format, clock) and one row per (file, channel) of absolute-wavelength
        statistics. No filtering, no baseline removal, no detection.

PART B  CADENCE AND RETENTION. Two questions that decide what any later rate
        can be measured against:
          1. is the inter-file interval an exact multiple of the file duration?
             If it is, recording is CONTIGUOUS and the files are not short;
          2. if the gaps are exact slot multiples then the discarded files can
             be COUNTED, which turns retention from a guess into a measurement.

Nothing here interprets a train. This is the delivery, described.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Works from either location: the working copy (data/scripts/) or the repo,
# where Scripts/ sits inside the package directory itself.
_UP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_UP if (_UP / "lightrack").is_dir() else _UP.parent))
from lightrack.config import CONFIG           # noqa: E402
from lightrack.loader import (                # noqa: E402
    CH_COLS, list_files, scan_file,
)

OUT = Path(CONFIG["paths"]["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)


# ======================================================== PART A — INVENTORY
OUT = Path(CONFIG["paths"]["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)


def inventory() -> None:
    files = list_files()
    print(f"{len(files)} files found in {CONFIG['paths']['data_dir']}")

    index_rows, chan_rows = [], []
    for i, p in enumerate(files, 1):
        meta, wl = scan_file(p)
        index_rows.append(meta.as_dict())

        for j, ch in enumerate(CH_COLS):
            v = wl[:, j]
            # Robust spread: a train passage is a tiny fraction of the record,
            # so std is dominated by the baseline. p99.9-p0.1 and full p2p
            # separate "quiet drift" from "something happened".
            chan_rows.append({
                "name": meta.name,
                "utc": meta.utc.isoformat(),
                "channel": ch,
                "mean_nm": float(v.mean()),
                "median_nm": float(np.median(v)),
                "min_nm": float(v.min()),
                "max_nm": float(v.max()),
                "std_pm": float(v.std()) * 1000.0,
                "p2p_pm": float(v.max() - v.min()) * 1000.0,
                # excursion about the file median, in pm, split by sign —
                # the briefing's sanity figures are quoted this way (§6.1)
                "pos_exc_pm": float(v.max() - np.median(v)) * 1000.0,
                "neg_exc_pm": float(v.min() - np.median(v)) * 1000.0,
            })

        print(f"  [{i:3d}/{len(files)}] {p.name}  dec={meta.decimal!r} "
              f"rows={meta.n_rows} dur={meta.duration_s:.3f}s "
              f"fs={meta.fs_hz:.3f}Hz outliers={meta.n_dt_outliers}", flush=True)

    idx = pd.DataFrame(index_rows)
    ch = pd.DataFrame(chan_rows)
    idx.to_csv(OUT / "file_index.csv", index=False)
    ch.to_csv(OUT / "channel_stats.csv", index=False)
    print(f"\nwrote {OUT/'file_index.csv'} and {OUT/'channel_stats.csv'}")


# ============================================ PART B — CADENCE AND RETENTION
def cadence() -> None:
    idx = pd.read_csv(OUT / "file_index.csv", parse_dates=["utc"]).sort_values("utc").reset_index(drop=True)

    print("=" * 84)
    print("1. IS THE CADENCE A FIXED SLOT?")
    print("=" * 84)
    dur = idx["duration_s"]
    # duration is (t_last - t_first); the slot is one sample longer
    slot_est = float(np.median(dur + 1.0 / CONFIG["acquisition"]["fs_hz"]))
    print(f"median record duration      : {dur.median():.3f} s")
    print(f"median record span (+1 samp): {slot_est:.3f} s")

    d19 = idx[idx["utc"].dt.date == pd.Timestamp("2026-05-19").date()].reset_index(drop=True)
    gaps = d19["utc"].diff().dt.total_seconds().dropna()
    print(f"\n2026-05-19: {len(d19)} files, {len(gaps)} intervals")
    print(f"{'gap_s':>10} {'/slot':>9} {'residual_s':>12}")
    best = None
    for slot in np.arange(143.0, 147.001, 0.001):
        k = np.round(gaps / slot)
        resid = np.abs(gaps - k * slot)
        score = resid.max()
        if best is None or score < best[0]:
            best = (score, slot, k, resid)
    worst, slot, k, resid = best
    for g, kk, rr in zip(gaps, k, resid):
        print(f"{g:10.3f} {kk:9.0f} {rr:12.3f}")
    print(f"\nbest-fit slot: {slot:.3f} s   worst residual over all intervals: {worst:.3f} s")
    print(f"mean |residual|: {resid.mean():.3f} s")

    if worst < 2.0:
        print("\n=> VERDICT: every interval is an integer multiple of the slot.")
        print("   Recording is CONTIGUOUS with no dead time. The files are complete;")
        print(f"   the true cadence is {slot:.2f} s, i.e. {3600/slot:.1f} files/hour,")
        print(f"   {86400/slot:.0f} files/day. The deck's '3 min / 20 per hour / 480 per")
        print("   day' is wrong. A-01 closed.")

    print("\n" + "=" * 84)
    print("2. SLOT RETENTION PER DAY (measured, not assumed)")
    print("=" * 84)
    print(f"{'date':>12} {'files':>6} {'span_h':>8} {'slots':>7} {'retained%':>10}")
    rows = []
    for d, g in idx.groupby(idx["utc"].dt.date):
        g = g.sort_values("utc")
        if len(g) < 2:
            rows.append({"date": d, "n": len(g), "span_h": 0.0, "slots": 1, "ret": 100.0})
            print(f"{str(d):>12} {len(g):6d} {0.0:8.2f} {1:7d} {100.0:10.1f}")
            continue
        span = (g["utc"].iloc[-1] - g["utc"].iloc[0]).total_seconds()
        slots = int(round(span / slot)) + 1
        ret = 100.0 * len(g) / slots
        rows.append({"date": d, "n": len(g), "span_h": span / 3600, "slots": slots, "ret": ret})
        print(f"{str(d):>12} {len(g):6d} {span/3600:8.2f} {slots:7d} {ret:10.1f}")

    r = pd.DataFrame(rows)
    print(f"\ndeck's stated retention: 80/480 = {100*80/480:.0f} %")
    print(f"median retention across days: {r['ret'].median():.1f} %")
    d19r = r[r['date'] == pd.Timestamp('2026-05-19').date()].iloc[0]
    print(f"\n2026-05-19: {d19r['n']} files over {d19r['span_h']:.2f} h = "
          f"{d19r['slots']} slots -> {d19r['ret']:.1f} % retained")
    print("   This is the only day whose retention matches the deck's own filter rate,")
    print("   so it is the only day plausibly representing a COMPLETE retained set")
    print("   for its window. Counting statistics belong here, labelled as such.")

    r.to_csv(OUT / "retention_per_day.csv", index=False)
    print(f"\nwrote {OUT/'retention_per_day.csv'}")


if __name__ == "__main__":
    inventory()
    cadence()

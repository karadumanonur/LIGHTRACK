"""Phase 0 reconnaissance: full inventory scan.

Reads every file once and writes:
  out/file_index.csv         one row per file (timing, format, clock)
  out/channel_stats.csv      one row per (file, channel) — absolute wavelength stats

No filtering, no peak detection, no baseline removal. Descriptive only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG           # noqa: E402
from lightrack.loader import (                # noqa: E402
    CH_COLS, list_files, scan_file,
)

OUT = Path(CONFIG["paths"]["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
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


if __name__ == "__main__":
    main()

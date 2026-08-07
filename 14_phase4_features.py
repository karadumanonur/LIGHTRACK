"""Phase 4 step 1 — build the per-sensor health feature table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc,
    rolling_median_baseline, estimate_speed,
)
from lightrack.health import sensor_features, add_pair_features, label  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SESSIONS = {pd.Timestamp(d).date(): i + 1
            for i, d in enumerate(CONFIG["campaign"]["measurement_dates"])}


def main() -> None:
    rows = []
    for n, p in enumerate(list_files(), 1):
        utc = parse_filename_utc(p.name)
        sess = SESSIONS.get(utc.date())
        if sess is None:
            continue
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        if not spd.ok:
            print(f"  [{n:3d}] s{sess:02d} {p.name}  REJECTED (score {spd.score:.2f})",
                  flush=True)
            continue
        feats = sensor_features(y, spd)
        t = pd.DataFrame([f.__dict__ for f in feats])
        t["name"] = p.name
        t["session"] = sess
        t["date"] = str(utc.date())
        t["v_kmh"] = spd.v_kmh
        t["score"] = spd.score
        rows.append(t)
        print(f"  [{n:3d}] s{sess:02d} {p.name}  v={spd.v_kmh:6.1f} "
              f"axles={t['n_axles'].max():3d}", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df = add_pair_features(df)
    df["label"] = df["pad"].map(label)
    df.to_csv(OUT / "phase4_features.csv", index=False)
    print(f"\n{len(df)} sensor-records from {df['name'].nunique()} passages")
    print(f"wrote {OUT/'phase4_features.csv'}")


if __name__ == "__main__":
    main()

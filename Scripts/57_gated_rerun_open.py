"""Re-run with the absolute amplitude gate — OPEN DAYS ONLY.

The gate (`lightrack.detect.has_train`, threshold `absolute_gate_pm = 38.93`) is
a defect fix, not a tuning change: it supplies input validation the pipeline
never had. The threshold is the geometric midpoint of an empty interval in the
observed excursion distribution (15.29 .. 99.10 pm), not a chosen value, and all
93 POC1 records clear it by a factor of 3.5, so no POC1 result moves.

THE SEALED DAYS ARE NOT RE-RUN. They were spent on the blind pass; anything
decided after looking at that data must be evaluated on the open days alone.
Open days: 07-18, 07-19, 07-21, 07-22, 07-24, 07-25, 07-27 — 372 records.

Amplitudes are not recomputed. The gate only removes records that contain no
train, and for every record above it `passage_window` is byte-for-byte what it
was, so per-sensor amplitudes on train-containing records cannot change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.safety import count_axles                       # noqa: E402
from lightrack.occupancy import slow_baseline, find_stationary  # noqa: E402

NEW = Path(r"C:\Users\onurk\Desktop\data\Valid_data_new")
OUT = Path(CONFIG["paths"]["out_dir"])
FS = CONFIG["acquisition"]["fs_hz"]
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}
EDGE_S = 1.5


def main() -> None:
    smoke = "--smoke" in sys.argv
    base = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
    open_names = set(base.loc[base["set"] == "open", "name"])
    files = [p for p in sorted(NEW.glob("*.txt")) if p.name in open_names]
    if smoke:
        files = files[:6]
    print(f"re-run over {len(files)} OPEN-day records, "
          f"gate = {CONFIG['detect']['absolute_gate_pm']} pm")

    rows, sens = [], []
    for n, p in enumerate(files, 1):
        try:
            raw = load_file(p)[CH_COLS]
            y = rolling_median_baseline(raw)
            spd = estimate_speed(y)
            nrec = len(y)
            w = passage_window(head_signal(y, 1), spd.v_kmh)
            if w is None:
                rows.append({"name": p.name, "rejected_no_train": True,
                             "gated": bool(spd.ok), "v_kmh": spd.v_kmh})
                continue
            st = find_stationary(head_signal(slow_baseline(raw), 1), spd.v_kmh)
            counts, spreads, unan = {}, {}, {}
            for h, pads in OPERATIONAL.items():
                hsh = head_signal(y, h)
                wh = passage_window(hsh, spd.v_kmh)
                if wh is None:
                    counts[h] = np.nan
                    continue
                cs = []
                for pad in pads:
                    r = count_axles(y[CH_OF_PAD[pad]].to_numpy()[wh[0]:wh[1]],
                                    spd.v_kmh)
                    cs.append(r.count)
                    sens.append({"name": p.name, "head": h, "pad": pad,
                                 "count": r.count, "agree": r.agree})
                counts[h] = float(np.median(cs))
                spreads[h] = int(max(cs) - min(cs))
                unan[h] = len(set(cs)) == 1
            utc = parse_filename_utc(p.name)
            rows.append({
                "name": p.name, "rejected_no_train": False,
                "utc": utc, "local": to_local(utc),
                "v_kmh": spd.v_kmh, "score": spd.score, "gated": bool(spd.ok),
                "head1_first": bool(spd.head1_first),
                "c_h1": counts.get(1, np.nan), "c_h2": counts.get(2, np.nan),
                "unanimous": bool(unan.get(1, False) and unan.get(2, False)),
                "touch_start": bool(w[0] < EDGE_S * FS),
                "touch_end": bool(w[1] > nrec - EDGE_S * FS),
                "n_stationary": len(st),
                "stationary_s": sum(e.duration_s for e in st),
            })
        except Exception as e:                                  # noqa: BLE001
            print(f"  !! {p.name}: {type(e).__name__}: {e}", flush=True)
        if n % 40 == 0 or smoke:
            print(f"  [{n:4d}/{len(files)}]", flush=True)

    d = pd.DataFrame(rows)
    print(f"\n{len(d)} records; rejected as train-free: "
          f"{int(d['rejected_no_train'].sum())}")
    if smoke:
        print("SMOKE OK — rerun without --smoke")
        return
    d.to_csv(OUT / "regated_open_records.csv", index=False)
    pd.DataFrame(sens).to_csv(OUT / "regated_open_sensors.csv", index=False)
    print("wrote regated_open_records.csv, regated_open_sensors.csv")


if __name__ == "__main__":
    main()

"""Step 3 — BLIND RUN of the frozen POC1 pipeline over Valid_data_new.

Nothing is tuned here. The configuration is `config_frozen_poc1.toml`
(SHA256 5314912...), byte-identical to the state at the end of the POC1 work.
The ONLY change to the codebase is `sniff_column_order`, which is parsing, not
tuning: the July delivery lists the twelve gratings in ascending wavelength
where POC1 listed them descending, and reading it unnormalised would assign
every sensor to the wrong PAD.

Every number below is a PREDICTION the frozen pipeline is making about data it
has never seen.

Self-diagnosis needs a commissioning baseline, and which one is used changes the
question:
  (a) NEW-SET baseline (its own first day)  -> false-alarm rate of the frozen
      thresholds on unseen data. This is the headline test.
  (b) POC1 baseline (2026-05-15)            -> continuity across the 45-day gap.
      Confounded by any re-installation, so reported separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, CH_OF_PAD, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal, axle_peaks,
    quiet_noise_floor,
)
from lightrack.detect import passage_window, samples_per_peak   # noqa: E402
from lightrack.safety import count_axles                        # noqa: E402
from lightrack.diagnostic import channel_amplitudes             # noqa: E402
from lightrack.occupancy import slow_baseline, find_stationary   # noqa: E402

NEW = Path(r"C:\Users\onurk\Desktop\data\Valid_data_new")
OUT = Path(CONFIG["paths"]["out_dir"])
FS = CONFIG["acquisition"]["fs_hz"]
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}
EDGE_S = 1.5

files = sorted(NEW.glob("*.txt"))
print(f"blind run over {len(files)} records with the frozen configuration")

rec_rows, sens_rows = [], []
for n, p in enumerate(files, 1):
    try:
        raw = load_file(p)[CH_COLS]
        y = rolling_median_baseline(raw)
        spd = estimate_speed(y)
        nrec = len(y)
        hs1 = head_signal(y, 1)
        w = passage_window(hs1, spd.v_kmh)
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
                r = count_axles(y[CH_OF_PAD[pad]].to_numpy()[wh[0]:wh[1]], spd.v_kmh)
                cs.append(r.count)
                sens_rows.append({"name": p.name, "head": h, "pad": pad,
                                  "count": r.count, "n_x": r.n_x, "n_d": r.n_d,
                                  "agree": r.agree, "v_kmh": spd.v_kmh})
            counts[h] = float(np.median(cs))
            spreads[h] = int(max(cs) - min(cs))
            unan[h] = len(set(cs)) == 1

        amps = channel_amplitudes(y, spd)
        for a in amps:
            # ChannelAmplitude carries pos_pm/neg_pm; peak-to-peak is derived.
            sens_rows.append({"name": p.name, "pad": a.pad, "head": a.head,
                              "p2p_pm": a.pos_pm - a.neg_pm, "pos_pm": a.pos_pm,
                              "neg_pm": a.neg_pm, "noise_pm": a.noise_pm,
                              "snr": a.snr, "n_axles": a.n_axles,
                              "neg_over_pos": (abs(a.neg_pm) / a.pos_pm
                                               if a.pos_pm and a.pos_pm > 0 else np.nan),
                              "diag": True})

        utc = parse_filename_utc(p.name)
        rec_rows.append({
            "name": p.name, "utc": utc, "local": to_local(utc),
            "date": to_local(utc).date(),
            "v_kmh": spd.v_kmh, "score": spd.score, "gated": spd.ok,
            "head1_first": spd.head1_first,
            "c_h1": counts.get(1, np.nan), "c_h2": counts.get(2, np.nan),
            "spread_h1": spreads.get(1, np.nan), "spread_h2": spreads.get(2, np.nan),
            "unanimous": bool(unan.get(1, False) and unan.get(2, False)),
            "touch_start": bool(w and w[0] < EDGE_S * FS),
            "touch_end": bool(w and w[1] > nrec - EDGE_S * FS),
            "n_stationary": len(st),
            "stationary_s": sum(e.duration_s for e in st),
        })
    except Exception as e:                                        # noqa: BLE001
        print(f"  !! {p.name}: {type(e).__name__}: {e}", flush=True)
    if n % 50 == 0:
        print(f"  [{n:4d}/{len(files)}]", flush=True)

pd.DataFrame(rec_rows).to_csv(OUT / "blind_records.csv", index=False)
pd.DataFrame(sens_rows).to_csv(OUT / "blind_sensors.csv", index=False)
print(f"\nwrote blind_records.csv ({len(rec_rows)}) and "
      f"blind_sensors.csv ({len(sens_rows)})")

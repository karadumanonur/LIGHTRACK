"""July blind pass, auxiliary extraction — the quantities 41_blind_run.py did not keep.

41_blind_run.py produced counts, balance, stationary events and per-sensor
amplitudes for all 609 July records. Four of the six questions this set can
answer need things it did not store:

  * the AXLE-GAP VECTOR, for freight labelling and for the multi-train signature;
  * the NOISE FLOOR measured OUTSIDE the passage, which is not the same quantity
    as the noise floor of a record with no train in it at all - A-15 turns on that
    distinction and the whole point of this delivery is that it may contain both;
  * per-record excursion statistics, so that "does this record contain a train"
    can be answered from the observed distribution rather than asserted;
  * the passage window itself.

NO TUNING. The speed and gate come from the frozen run's own output rather than
being recomputed, so nothing here can drift from it; every detector parameter is
`config_frozen_poc1.toml` as loaded by the package. Nothing in this script
chooses a threshold.

Run with --smoke to process ten records first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, load_file, rolling_median_baseline,
    head_signal, axle_peaks, quiet_noise_floor,
)
from lightrack.detect import passage_window                    # noqa: E402

NEW = Path(r"C:\Users\onurk\Desktop\data\Valid_data_new")
OUT = Path(CONFIG["paths"]["out_dir"])
FS = CONFIG["acquisition"]["fs_hz"]
SPLIT_M = 30.0                      # same constant as 20_service_census.py
pd.set_option("display.width", 240)


def main() -> None:
    smoke = "--smoke" in sys.argv
    base = pd.read_csv(OUT / "blind_records_scored.csv", low_memory=False)
    spd = base.set_index("name")[["v_kmh", "score", "gated", "head1_first",
                                  "set", "date"]]
    files = sorted(NEW.glob("*.txt"))
    if smoke:
        files = files[:10]
    rows, gaps, failed = [], [], []
    for n, p in enumerate(files, 1):
        if p.name not in spd.index:
            failed.append((p.name, "not in the frozen blind run"))
            continue
        s = spd.loc[p.name]
        v = float(s["v_kmh"])
        try:
            y = rolling_median_baseline(load_file(p)[CH_COLS])
            nrec = len(y)
            hs = {h: head_signal(y, h) for h in (1, 2, 3)}
            h1 = hs[1]
            w = passage_window(h1, v)
            idx = axle_peaks(h1, v, window=w) if w else np.array([], dtype=int)
        except Exception as e:                                  # noqa: BLE001
            failed.append((p.name, f"{type(e).__name__}: {e}"))
            continue

        # noise measured OUTSIDE the passage window (inter-train baseline) and
        # over the whole record (which equals the former when there is no train)
        if w is not None:
            outside = np.concatenate([h1[: w[0]], h1[w[1]:]])
        else:
            outside = h1
        nz_out = quiet_noise_floor(outside) if outside.size > FS \
            else quiet_noise_floor(h1)
        nz_all = quiet_noise_floor(h1)

        r = {
            "name": p.name, "set": s["set"], "date": s["date"],
            "v_kmh": v, "score": float(s["score"]), "gated": bool(s["gated"]),
            "head1_first": bool(s["head1_first"]),
            "n_samples": nrec, "duration_s": nrec / FS,
            "win_i0": (w[0] if w else np.nan), "win_i1": (w[1] if w else np.nan),
            "win_s": ((w[1] - w[0]) / FS if w else np.nan),
            "noise_outside_pm": float(nz_out), "noise_all_pm": float(nz_all),
            "n_peaks": int(len(idx)),
        }
        for h in (1, 2, 3):
            a = np.abs(hs[h])
            r[f"max_h{h}_pm"] = float(a.max())
            r[f"p999_h{h}_pm"] = float(np.percentile(a, 99.9))
        # excursion counts against the record's OWN outside-passage noise floor:
        # the operational disturbance floor is a count of events, not a sigma
        r["n_exc_10s"] = int((np.abs(h1) > 10 * nz_out).sum())
        r["n_exc_20s"] = int((np.abs(h1) > 20 * nz_out).sum())

        if len(idx) >= 4:
            x = (idx - idx[0]) / FS * (v / 3.6)
            g = np.diff(x)
            cuts = np.flatnonzero(g > SPLIT_M)
            starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(x) - 1]
            keep = [(a, b) for a, b in zip(starts, ends) if b - a >= 3]
            r["n_segments"] = len(keep)
            if keep:
                a, b = max(keep, key=lambda t: t[1] - t[0])
                gv = np.diff(x[a:b + 1])
                r.update({
                    "n_axles": int(b - a + 1),
                    "length_m": float(x[b] - x[a]),
                    "gap_min_m": float(gv.min()),
                    "gap_med_m": float(np.median(gv)),
                    "n_gap_lt_2m0": int((gv < 2.0).sum()),
                    "n_gap_lt_2m5": int((gv < 2.5).sum()),
                    "off_s": float(idx[a]) / FS,
                })
                gaps.extend({"name": p.name, "set": s["set"], "k": k,
                             "gap_m": float(q)} for k, q in enumerate(gv))
        else:
            r["n_segments"] = 0
        rows.append(r)
        if n % 50 == 0 or smoke:
            print(f"  [{n:4d}/{len(files)}] {p.name[:26]}  "
                  f"peaks {r['n_peaks']:>3}  max {r['max_h1_pm']:>7.1f} pm",
                  flush=True)

    d = pd.DataFrame(rows)
    g = pd.DataFrame(gaps)
    print(f"\n{len(d)} records, {len(g)} gaps")
    if failed:
        print(f"{len(failed)} failed:")
        for nm, why in failed[:20]:
            print(f"   {nm[:30]:<32} {why}")
    if smoke:
        print("\nSMOKE OK — rerun without --smoke")
        return
    d.to_csv(OUT / "july_aux_records.csv", index=False)
    g.to_csv(OUT / "july_aux_gaps.csv", index=False)
    print("wrote july_aux_records.csv, july_aux_gaps.csv")


if __name__ == "__main__":
    main()

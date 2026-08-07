"""Step 4a — passage time inside every delivered record.

Matching against a timetable needs the time the TRAIN passed, not the time the
recorder started a slot. Recording is contiguous (A-01), so a file boundary falls
at an arbitrary point relative to the train and using the file start smears every
match by up to 144.5 s - larger than the tightest tolerance worth testing.

This is the same extraction as 20_service_census.py, with the weekday and session
filters removed so that all delivered records are covered, and with the passage
END recorded as well as its start. Frozen config throughout; no parameter here is
chosen with the timetable in view.

REFERENCE POINT. Times are the first/last axle at whichever head is best
conditioned in that record. The three heads span 25 m, so the choice of head
moves the timestamp by at most 25 m / v - 0.8 s at 110 km/h. The site position
itself is only known to +-50 m, so head choice is not the limiting term.

MULTI-TRAIN RECORDS. A gap larger than SPLIT_M metres separates two trains rather
than two axles, so a record holding two passages yields two rows.

Run with --smoke to process three files first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal, axle_peaks,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.baseline import quiet_noise_floor               # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
SPLIT_M = 30.0                      # same constant as 20_service_census.py
pd.set_option("display.width", 240)


def best_head(y, v):
    q = {}
    for h in (1, 2, 3):
        hs = head_signal(y, h)
        w = passage_window(hs, v)
        if w is None:
            q[h] = -np.inf
            continue
        quiet = np.concatenate([hs[: w[0]], hs[w[1]:]])
        nz = quiet_noise_floor(quiet) if quiet.size > FS else quiet_noise_floor(hs)
        q[h] = float(np.percentile(hs[w[0]:w[1]], 99.0) / nz) if nz > 0 else -np.inf
    return max(q, key=q.get)


def main() -> None:
    smoke = "--smoke" in sys.argv
    files = list_files()
    if smoke:
        files = files[:3]
    rows, gaps, failed = [], [], []
    for n, p in enumerate(files, 1):
        utc = parse_filename_utc(p.name)
        loc = to_local(utc)
        try:
            y = rolling_median_baseline(load_file(p)[CH_COLS])
            spd = estimate_speed(y)
            if not spd.ok:
                failed.append((p.name, "speed solver did not converge"))
                continue
            h = best_head(y, spd.v_kmh)
            hs = head_signal(y, h)
            idx = axle_peaks(hs, spd.v_kmh, window=passage_window(hs, spd.v_kmh))
        except Exception as e:                                  # noqa: BLE001
            failed.append((p.name, repr(e)))
            continue
        if len(idx) < 4:
            failed.append((p.name, f"only {len(idx)} peaks"))
            continue
        v_mps = spd.v_kmh / 3.6
        x = (idx - idx[0]) / FS * v_mps
        cuts = np.flatnonzero(np.diff(x) > SPLIT_M)
        starts, ends = np.r_[0, cuts + 1], np.r_[cuts, len(x) - 1]
        for si, (a, b) in enumerate(zip(starts, ends)):
            if b - a < 3:
                continue
            # axle-gap signature: the short-gap population is what distinguishes
            # freight bogies from passenger stock, so it is carried here rather
            # than reconstructed later from n_axles and length alone.
            g = np.diff(x[a:b + 1])
            rows.append({
                "name": p.name, "utc": utc, "file_local": loc,
                "sub": si, "n_sub": len(starts),
                "off_s": float(idx[a]) / FS,
                "off_end_s": float(idx[b]) / FS,
                "train_local": loc + pd.Timedelta(seconds=float(idx[a]) / FS),
                "train_end_local": loc + pd.Timedelta(seconds=float(idx[b]) / FS),
                "head": h, "v_kmh": spd.v_kmh,
                "head1_first": spd.head1_first, "score": spd.score,
                "alt_score": spd.alt_score,
                "n_axles": int(b - a + 1), "length_m": float(x[b] - x[a]),
                "gap_min_m": float(g.min()), "gap_med_m": float(np.median(g)),
                "gap_p25_m": float(np.percentile(g, 25)),
                "n_gap_lt_2m5": int((g < 2.5).sum()),
            })
            # Full gap vector, persisted. Every spatial quantity in this project
            # is v * dt, so a speed-scale error rescales all of them together;
            # keeping the raw gaps makes that sensitivity a multiplication rather
            # than a re-run.
            gaps.extend({"name": p.name, "sub": si, "k": k, "gap_m": float(gv)}
                        for k, gv in enumerate(g))
        if n % 15 == 0 or smoke:
            print(f"  [{n:3d}/{len(files)}] {p.name}  "
                  f"{len(starts)} segment(s)", flush=True)

    g = pd.DataFrame(gaps)
    d = pd.DataFrame(rows)
    print(f"\n{len(d)} passages from {d['name'].nunique()}/{len(files)} records")
    print(f"records yielding >1 passage: "
          f"{int((d.drop_duplicates('name')['n_sub'] > 1).sum())}")
    if failed:
        print(f"\n{len(failed)} records produced no passage:")
        for nm, why in failed:
            print(f"   {nm[:30]:<32} {why}")
    if smoke:
        print("\nSMOKE OK — rerun without --smoke")
        return
    d.to_csv(OUT / "passage_times.csv", index=False)
    g.to_csv(OUT / "passage_gaps.csv", index=False)
    print(f"\nwrote passage_times.csv, passage_gaps.csv ({len(g)} gaps)")


if __name__ == "__main__":
    main()

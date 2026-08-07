"""Service census — how often does the same train actually repeat?

The repeated-weekday-service idea is only as strong as the repeat count. This
counts it, before any of it is relied on.

Method
------
A trainset's axle-spacing signature is a fixed geometric property. Peak times
are converted to distances with the fitted speed, so the signature is
speed-invariant and can be compared across days:

    x_i = v * (t_i - t_0)        gaps = diff(x)

Records containing more than one train are segmented first: a gap larger than
`SPLIT_M` is an inter-train gap, not an inter-axle one.

Matching is on SIGNATURE, not on schedule slot, because operators rotate stock
and "same service" does not guarantee "same trainset" (and the converse: the
same unit may run different slots). Time-of-day is reported alongside as a
cross-check, never as the clustering key.
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
from lightrack.diagnostic import head_offsets                  # noqa: E402
from lightrack.baseline import quiet_noise_floor               # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
SESSIONS = {pd.Timestamp(d).date(): i + 1
            for i, d in enumerate(CONFIG["campaign"]["measurement_dates"])}
SPLIT_M = 30.0          # gap above this separates two trains
MATCH_TOL_M = 1.0       # RMS gap difference for "same signature"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

pd.set_option("display.width", 240)
LINE = "=" * 106


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
    trains = []
    for n, p in enumerate(list_files(), 1):
        utc = parse_filename_utc(p.name)
        sess = SESSIONS.get(utc.date())
        if sess is None:
            continue
        loc = to_local(utc)
        if loc.weekday() >= 5:                    # weekday only
            continue
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        if not spd.ok:
            continue
        h = best_head(y, spd.v_kmh)
        hs = head_signal(y, h)
        idx = axle_peaks(hs, spd.v_kmh, window=passage_window(hs, spd.v_kmh))
        if len(idx) < 4:
            continue
        v_mps = spd.v_kmh / 3.6
        x = (idx - idx[0]) / FS * v_mps
        gaps = np.diff(x)
        # segment into trains
        cuts = np.flatnonzero(gaps > SPLIT_M)
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(x) - 1]
        for si, (a, b) in enumerate(zip(starts, ends)):
            if b - a < 3:
                continue
            g = np.diff(x[a:b + 1])
            # The TRAIN's clock time, not the file's. Recording is contiguous
            # (A-01), so a file boundary falls at an arbitrary point relative to
            # the train; matching schedule slots on file-start time smears every
            # slot by up to 145 s.
            off_s = float(idx[a]) / FS
            tl = loc + pd.Timedelta(seconds=off_s)
            trains.append({
                "name": p.name, "session": sess, "utc": utc, "local": loc,
                "train_local": tl, "off_s": off_s,
                "dow": loc.weekday(), "tod_min": tl.hour * 60 + tl.minute + tl.second / 60,
                "v_kmh": spd.v_kmh, "sub": si, "n_axles": b - a + 1,
                "length_m": float(x[b] - x[a]),
                "gaps": g,
            })
        if n % 15 == 0:
            print(f"  [{n:3d}/93] {p.name}", flush=True)

    print(f"\n{len(trains)} train segments from "
          f"{len({t['name'] for t in trains})} weekday passages")

    # ------------------------------------------------------------- cluster
    clusters: list[list[int]] = []
    reps: list[dict] = []
    for i, t in enumerate(trains):
        placed = False
        for ci, r in enumerate(reps):
            if r["n_axles"] != t["n_axles"]:
                continue
            d = float(np.sqrt(np.mean((r["gaps"] - t["gaps"]) ** 2)))
            if d < MATCH_TOL_M:
                clusters[ci].append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(t)

    rows = []
    for ci, members in enumerate(clusters):
        m = [trains[j] for j in members]
        tod = np.array([x["tod_min"] for x in m])
        rows.append({
            "service": ci, "n_repeats": len(m),
            "n_axles": m[0]["n_axles"],
            "length_m": float(np.median([x["length_m"] for x in m])),
            "tod_median": f"{int(np.median(tod))//60:02d}:{int(np.median(tod))%60:02d}",
            "tod_spread_min": float(tod.max() - tod.min()),
            "n_days": len({x["local"].date() for x in m}),
            "sessions": sorted({x["session"] for x in m}),
            "v_min": float(min(x["v_kmh"] for x in m)),
            "v_max": float(max(x["v_kmh"] for x in m)),
        })
    cen = pd.DataFrame(rows).sort_values("n_repeats", ascending=False).reset_index(drop=True)

    print(f"\n{LINE}\nSERVICE CENSUS — weekday train segments clustered by axle signature\n{LINE}")
    print(f"match tolerance {MATCH_TOL_M} m RMS on inter-axle gaps; "
          f"train split at {SPLIT_M} m\n")
    print(cen[["service", "n_repeats", "n_days", "n_axles", "length_m",
               "tod_median", "tod_spread_min", "v_min", "v_max"]].to_string(index=False))

    print(f"\n{LINE}\nSUMMARY\n{LINE}")
    print(f"distinct services (signature clusters) : {len(cen)}")
    print(f"train segments                         : {len(trains)}")
    print(f"services with >=3 repeats              : {(cen['n_repeats']>=3).sum()}")
    print(f"services with >=5 repeats              : {(cen['n_repeats']>=5).sum()}")
    print(f"services appearing once only           : {(cen['n_repeats']==1).sum()}")
    print(f"segments in services with >=3 repeats   : "
          f"{int(cen.loc[cen['n_repeats']>=3,'n_repeats'].sum())} / {len(trains)}")
    print(f"\nrepeat-count distribution:")
    print(cen["n_repeats"].value_counts().sort_index().to_string())

    print(f"\n{LINE}\nWHY: the sampling window differs between sessions\n{LINE}")
    tod = pd.DataFrame([{"session": t["session"], "tod": t["tod_min"],
                         "date": t["local"].date()} for t in trains])
    g = tod.groupby("session").agg(
        date=("date", "first"), n=("tod", "size"),
        first=("tod", "min"), last=("tod", "max"))
    g["window_local"] = g.apply(
        lambda r: f"{int(r['first'])//60:02d}:{int(r['first'])%60:02d}-"
                  f"{int(r['last'])//60:02d}:{int(r['last'])%60:02d}", axis=1)
    print(g[["date", "n", "window_local"]].to_string())

    cen.to_csv(OUT / "service_census.csv", index=False)
    pd.DataFrame([{k: v for k, v in t.items() if k != "gaps"} for t in trains]).assign(
        service=[next(ci for ci, m in enumerate(clusters) if i in m)
                 for i in range(len(trains))]
    ).to_csv(OUT / "train_segments.csv", index=False)
    np.save(OUT / "train_gaps.npy",
            np.array([t["gaps"] for t in trains], dtype=object), allow_pickle=True)
    print(f"\nwrote service_census.csv, train_segments.csv, train_gaps.npy")


if __name__ == "__main__":
    main()

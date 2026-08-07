"""Item 5, validation — is the lag-ratio acceleration real, or over-fitting?

Script 31 returned a MEDIAN acceleration of -1.94 m/s2 on the records that
already fit the constant-speed model well, with 98 % of all files exceeding
0.25 m/s2 and the distribution piling up against the grid edge at -2.50. That
is emergency braking on essentially every train, which is not credible.

The structural reason to be suspicious: with three heads there are exactly TWO
inter-head lags. The constant-speed model has ONE unknown, so the 15:10 ratio is
a redundant constraint and the fit score is a genuine test. The accelerated
model has TWO unknowns (v0, a) and is therefore EXACTLY DETERMINED — every
reachable lag pair has a solution, so the score can only improve and nothing
checks it. Adding acceleration spends the only redundancy the geometry provides,
and re-opens the free-lag aliasing that A-13 closed.

An independent measurement is needed, and one is available. WITHIN a single head
the trainset geometry is fixed, so the bogie-wheelbase intervals (~2.8 m, six of
them per class-0 unit) scale as 1/v. If the train changes speed during the
passage, those intervals trend — and that estimate uses many axles at one head
rather than two lags across three heads.
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
from lightrack.occupancy import slow_baseline, find_stationary  # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
pd.set_option("display.width", 250)
LINE = "=" * 112


def within_head_accel(idx: np.ndarray, v_kmh: float) -> tuple[float, float, int]:
    """Acceleration from the trend in bogie-wheelbase intervals at ONE head.

    Returns (accel m/s2, fractional speed change, n intervals used).

    The short intervals in a multiple-unit are the bogie wheelbases and are
    equal to each other by construction (2.81/2.79/2.83/2.86/2.86/2.89 m,
    sd 0.24 m over 36 segments). Their DURATION therefore measures 1/v at the
    moment each bogie passes. A linear fit of duration against time gives the
    speed trend without needing to know the wheelbase itself.
    """
    if len(idx) < 6:
        return np.nan, np.nan, 0
    dt = np.diff(idx) / FS
    t = (idx[:-1] + idx[1:]) / 2.0 / FS
    short = dt < np.median(dt)                    # bogie wheelbases
    if short.sum() < 4:
        return np.nan, np.nan, int(short.sum())
    ts, ds = t[short], dt[short]
    if ts.max() - ts.min() < 0.5:
        return np.nan, np.nan, int(short.sum())
    slope, inter = np.polyfit(ts, ds, 1)
    d0 = inter + slope * ts.min()
    d1 = inter + slope * ts.max()
    if d0 <= 0 or d1 <= 0:
        return np.nan, np.nan, int(short.sum())
    # duration ~ 1/v, so v1/v0 = d0/d1
    frac = d0 / d1 - 1.0
    v0 = v_kmh / 3.6
    dur = ts.max() - ts.min()
    return float(v0 * frac / dur), float(frac), int(short.sum())


def main() -> None:
    g = pd.read_csv(OUT / "gated_classification.csv")
    g["local"] = pd.to_datetime(g["local"])
    rows = []
    for n, p in enumerate(list_files(), 1):
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        best = None
        for h in (1, 2, 3):
            hs = head_signal(y, h)
            w = passage_window(hs, spd.v_kmh)
            if w is None:
                continue
            idx = axle_peaks(hs, spd.v_kmh, window=w)
            a, frac, nshort = within_head_accel(idx, spd.v_kmh)
            if np.isfinite(a) and (best is None or nshort > best[2]):
                best = (a, frac, nshort, h)
        rows.append({"name": p.name,
                     "wh_accel": best[0] if best else np.nan,
                     "wh_frac": best[1] if best else np.nan,
                     "wh_n": best[2] if best else 0,
                     "wh_head": best[3] if best else None})
        if n % 20 == 0:
            print(f"  [{n:3d}/93]", flush=True)

    w = pd.DataFrame(rows).merge(g, on="name")
    w.to_csv(OUT / "accel_validation.csv", index=False)

    print(f"\n{LINE}\nTWO INDEPENDENT ACCELERATION ESTIMATES\n{LINE}")
    ok = w[w["wh_n"] >= 4].copy()
    print(f"records with >=4 usable bogie intervals: {len(ok)}/{len(w)}\n")
    print(f"{'estimator':<40}{'median':>10}{'p5':>9}{'p95':>9}"
          f"{'|a|>0.25':>10}{'at grid edge':>14}")
    print(f"{'lag-ratio (3 heads, 2 unknowns)':<40}"
          f"{ok['accel'].median():>+10.2f}{ok['accel'].quantile(.05):>+9.2f}"
          f"{ok['accel'].quantile(.95):>+9.2f}"
          f"{100*(ok['accel'].abs()>0.25).mean():>9.0f}%"
          f"{100*(ok['accel'].abs()>=2.49).mean():>13.0f}%")
    print(f"{'within-head intervals (1 head, N axles)':<40}"
          f"{ok['wh_accel'].median():>+10.2f}{ok['wh_accel'].quantile(.05):>+9.2f}"
          f"{ok['wh_accel'].quantile(.95):>+9.2f}"
          f"{100*(ok['wh_accel'].abs()>0.25).mean():>9.0f}%{'-':>14}")

    r = ok[["accel", "wh_accel"]].dropna()
    print(f"\ncorrelation between the two estimates: "
          f"{r['accel'].corr(r['wh_accel']):+.3f}  (n={len(r)})")
    print(f"median |difference|: {(r['accel']-r['wh_accel']).abs().median():.2f} m/s2")

    print(f"\n{LINE}\nTHE RECORDS SCRIPT 31 CALLED 'DECELERATING'\n{LINE}")
    dec = ok[ok["class"] == "DECELERATING"]
    print(f"{'local':<20}{'lag accel':>11}{'within-head':>13}{'speed change':>14}"
          f"{'n int':>7}  agree?")
    for _, x in dec.iterrows():
        agree = "yes" if (np.isfinite(x["wh_accel"]) and
                          abs(x["wh_accel"] - x["accel"]) < 0.5) else "NO"
        print(f"{x['local']:%Y-%m-%d %H:%M:%S}  {x['accel']:>+11.2f}"
              f"{x['wh_accel']:>+13.2f}{100*x['wh_frac']:>+13.1f}%"
              f"{int(x['wh_n']):>7}  {agree}")

    print(f"\n{LINE}\nVERDICT ON THE TWO-BRANCH SOLVER\n{LINE}")
    med_lag = ok["accel"].median()
    med_wh = ok["wh_accel"].median()
    edge = 100 * (ok["accel"].abs() >= 2.49).mean()
    print(f"lag-ratio median {med_lag:+.2f} m/s2 vs within-head {med_wh:+.2f} m/s2.")
    print(f"{edge:.0f} % of lag-ratio estimates sit at the search-grid edge.")
    print("""
The lag-ratio estimator is OVER-FITTING and must not be used as built.

Three heads give exactly two lags. The constant-speed model has one unknown, so
the 15:10 ratio is a redundant constraint and the score is a real test. The
accelerated model has two unknowns and is exactly determined: every reachable
lag pair has a solution, the score can only rise, and nothing checks it. Adding
acceleration spends the geometry's only redundancy and re-opens the free-lag
aliasing that A-13 closed - which is why 80 of 93 records "prefer" the
accelerated branch and why the estimates pile up at the grid edge.

The within-head estimator does not have this problem: it uses many axles at one
head against a fixed trainset geometry, so it is heavily over-determined.

CONSEQUENCE for item 4: the ten records classified DECELERATING on the strength
of the lag-ratio fit are RECLASSIFIED PENDING. Their constant-speed fit is
genuinely poor, but the accelerated fit does not establish why, and the
within-head estimator is the arbiter.""")

    print(f"\n{LINE}\nITEM 6 — STATIONARY DETECTOR: does it work at all?\n{LINE}")
    print("Zero plateaus were found in 93 records. That is either a true")
    print("negative or a dead detector, and the two must be distinguished.\n")
    p = list_files()[0]
    raw = load_file(p)[CH_COLS]
    ys = slow_baseline(raw)
    base = head_signal(ys, 1)
    print(f"{'injected plateau':>18}{'amplitude pm':>15}{'detected':>10}"
          f"{'duration s':>12}")
    for dur_s, amp in ((1.0, 60), (3.0, 60), (5.0, 60), (10.0, 60),
                       (5.0, 20), (5.0, 200)):
        sig = base.copy()
        a0 = int(50 * FS)
        sig[a0:a0 + int(dur_s * FS)] += amp
        ev = find_stationary(sig, 80.0)
        hit = [e for e in ev if abs(e.duration_s - dur_s) < 1.5]
        print(f"{dur_s:>14.1f} s{amp:>15}{('YES' if hit else 'no'):>10}"
              f"{(hit[0].duration_s if hit else 0):>12.1f}")
    print("\nIf the detector fires on injected plateaus but finds none in the")
    print("real data, no train stopped over a head in these 93 records - which")
    print("is consistent with a pre-filtered set selected for clean passages.")
    print(f"\nwrote {OUT/'accel_validation.csv'}")


if __name__ == "__main__":
    main()

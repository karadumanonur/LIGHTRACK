"""Phase 0: head order, direction and speed by a geometry-constrained search.

Both naive estimators failed, in opposite ways:
  * script 03 (free cross-correlation) aliases by one carriage length, because
    the axle sequence is quasi-periodic;
  * script 06 (amplitude onset) is biased by per-channel sensitivity — a
    degraded sensor crosses the gate late and fakes a lag.

Fix: stop estimating the two lags independently. The three heads are at FIXED
distances, so a single unknown — the slowness tau [s/m] — predicts BOTH lags at
once. Scanning tau and scoring the sum of the two correlations means an alias
must coincidentally satisfy the 15:10 ratio to win, which it almost never does.

One free parameter, two constraints. That is the whole idea.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG                        # noqa: E402
from lightrack.loader import CH_COLS, load_file, list_files  # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
WIN = CONFIG["baseline"]["rolling_window_samples"]
D12 = CONFIG["geometry"]["head1_to_head2_m"]   # 15 m  Head1<->Head2
D23 = CONFIG["geometry"]["head2_to_head3_m"]   # 10 m  Head2<->Head3
OUT = Path(CONFIG["paths"]["out_dir"])

GA, GB, GC = CH_COLS[0:4], CH_COLS[4:8], CH_COLS[8:12]
V_MIN, V_MAX = 20.0, 200.0          # km/h search bracket


def group_signal(y: pd.DataFrame, chs: list[str]) -> np.ndarray:
    """Robustly averaged group trace: each channel scaled to unit MAD first so
    a loud sensor (ch04) and a nearly dead one (ch02) contribute equally."""
    m = y[chs].to_numpy()
    mad = np.median(np.abs(m - np.median(m, axis=0)), axis=0) * 1.4826
    mad[mad <= 0] = 1.0
    return (m / mad).mean(axis=1)


def xcorr_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Normalised correlation of a and b at a given lag.

    SIGN CONVENTION — read this before touching the direction logic.
    For lag > 0 we align a[lag:] with b[:-lag], i.e. a[lag+i] ~ b[i]. The event
    seen at index i in b turns up at index lag+i in a, so **positive lag means
    `a` is DELAYED relative to `b`** — b is reached first.
    """
    n = len(a)
    if lag >= 0:
        x, z = a[lag:], b[: n - lag]
    else:
        x, z = a[: n + lag], b[-lag:]
    if len(x) < FS:
        return 0.0
    x = x - x.mean(); z = z - z.mean()
    d = np.linalg.norm(x) * np.linalg.norm(z)
    return float(np.dot(x, z) / d) if d > 1e-9 else 0.0


def solve(y: pd.DataFrame) -> dict | None:
    a, b, c = (group_signal(y, g) for g in (GA, GB, GC))
    # search window: the loudest 20 s, so the scan is not dominated by silence
    env = pd.Series(np.abs(a) + np.abs(b) + np.abs(c)).rolling(
        int(0.5 * FS), center=True, min_periods=1).mean().to_numpy()
    ctr = int(np.argmax(env))
    half = int(10 * FS)
    sl = slice(max(0, ctr - half), min(len(a), ctr + half))
    a, b, c = a[sl], b[sl], c[sl]

    best = None
    # tau = slowness in samples per metre; scan both travel directions
    for v_kmh in np.arange(V_MIN, V_MAX + 0.25, 0.25):
        tau = FS / (v_kmh / 3.6)                       # samples per metre
        for sign in (+1, -1):
            # Per the convention in xcorr_at, sign=+1 makes A lag B and B lag C,
            # i.e. the train is reached C first, then B, then A.
            lag_ab = int(round(sign * D23 * tau))      # A<->B spans 10 m
            lag_bc = int(round(sign * D12 * tau))      # B<->C spans 15 m
            score = xcorr_at(a, b, lag_ab) + xcorr_at(b, c, lag_bc)
            if best is None or score > best[0]:
                best = (score, v_kmh, sign, lag_ab, lag_bc)

    score, v, sign, lag_ab, lag_bc = best
    # counter-hypothesis: the head order is the other way round (A<->B = 15 m)
    best2 = None
    for v_kmh in np.arange(V_MIN, V_MAX + 0.25, 0.25):
        tau = FS / (v_kmh / 3.6)
        for sg in (+1, -1):
            s = xcorr_at(a, b, int(round(sg * D12 * tau))) + \
                xcorr_at(b, c, int(round(sg * D23 * tau)))
            if best2 is None or s > best2[0]:
                best2 = (s, v_kmh, sg)

    return {
        "score": score, "v_kmh": v, "sign": sign,
        "lag_ab": lag_ab, "lag_bc": lag_bc,
        "alt_score": best2[0], "alt_v": best2[1],
    }


def main() -> None:
    files = list_files()
    rows = []
    for p in files:
        try:
            x = load_file(p)[CH_COLS]
            y = (x - x.rolling(WIN, center=True, min_periods=1).median()) * 1000.0
            r = solve(y)
        except Exception as e:                                  # noqa: BLE001
            print(f"  !! {p.name}: {e}")
            continue
        if r is None:
            continue
        r["name"] = p.name
        rows.append(r)
        print(f"  {p.name}  v={r['v_kmh']:6.1f} km/h  dir={'C->B->A' if r['sign']>0 else 'A->B->C'}"
              f"  score={r['score']:.3f}  (alt {r['alt_score']:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tof_speed_direction.csv", index=False)

    print("\n" + "=" * 82)
    print("HEAD-ORDER HYPOTHESIS TEST")
    print("=" * 82)
    win = (df["score"] > df["alt_score"]).sum()
    print(f"H1  A-B = 10 m, B-C = 15 m  (A=Head3, B=Head2, C=Head1): "
          f"wins {win}/{len(df)} files")
    print(f"H2  A-B = 15 m, B-C = 10 m  (A=Head1, B=Head2, C=Head3): "
          f"wins {len(df)-win}/{len(df)} files")
    print(f"mean correlation score: H1 {df['score'].mean():.3f}   "
          f"H2 {df['alt_score'].mean():.3f}")

    print("\n" + "=" * 82)
    print("DIRECTION")
    print("=" * 82)
    d = df["sign"].map({1: "C->B->A  (Head1 -> Head2 -> Head3)",
                        -1: "A->B->C  (Head3 -> Head2 -> Head1)"})
    print(d.value_counts().to_string())

    print("\n" + "=" * 82)
    print("SPEED DISTRIBUTION (constrained time-of-flight)")
    print("=" * 82)
    q = df["v_kmh"].quantile([0, .1, .25, .5, .75, .9, 1.0])
    for k, v in q.items():
        print(f"  p{int(k*100):>3} : {v:6.1f} km/h")
    print(f"  files at the search edge ({V_MIN} or {V_MAX}): "
          f"{int(((df['v_kmh']<=V_MIN)|(df['v_kmh']>=V_MAX)).sum())}")

    k = CONFIG["geometry"]["samples_per_peak_numerator_at_500hz"]
    print("\nspeed-adaptive window implied by briefing §3.4 (3960 / v_kmh):")
    for lbl, v in [("fastest", df["v_kmh"].max()), ("median", df["v_kmh"].median()),
                   ("slowest", df["v_kmh"].min())]:
        print(f"  {lbl:>8} {v:6.1f} km/h -> samples_per_peak = {k/v:5.1f}")

    print(f"\nwrote {OUT/'tof_speed_direction.csv'}")


if __name__ == "__main__":
    main()

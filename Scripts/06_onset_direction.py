"""Phase 0: head order and direction from passage ONSET, not cross-correlation.

Why this exists: script 03 used cross-correlation over the whole passage. A
train is quasi-periodic (axle, bogie, carriage), so the correlation has
secondary maxima one carriage-length away from the true lag, and for roughly
half the files it locked onto one of those. The 12-channel raw plot shows the
truth directly — three groups firing at three distinct times — so the lag
estimator, not the grouping, was at fault.

Onset time is immune to that: the first arrival of the first axle happens once.

This is still reconnaissance. The threshold below is a crude robust-MAD gate to
find *where* the passage starts; it is NOT the counting detector and nothing
here should be reused in the safety chain.
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
D12 = CONFIG["geometry"]["head1_to_head2_m"]    # 15 m
D23 = CONFIG["geometry"]["head2_to_head3_m"]    # 10 m
OUT = Path(CONFIG["paths"]["out_dir"])

GROUPS = {"A": CH_COLS[0:4], "B": CH_COLS[4:8], "C": CH_COLS[8:12]}
K_MAD = 8.0          # onset gate, in robust sigmas above the quiet baseline


def debaseline(df: pd.DataFrame) -> pd.DataFrame:
    x = df[CH_COLS]
    return (x - x.rolling(WIN, center=True, min_periods=1).median()) * 1000.0


def onset(y: np.ndarray) -> float | None:
    """Index of the first sample of the largest coherent excursion block."""
    mad = np.median(np.abs(y - np.median(y))) * 1.4826
    if mad <= 0:
        return None
    thr = K_MAD * mad
    hot = np.abs(y) > thr
    if not hot.any():
        return None
    # largest contiguous-ish run, allowing gaps up to 1 s (between bogies)
    idx = np.flatnonzero(hot)
    splits = np.flatnonzero(np.diff(idx) > FS) + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    return float(best[0])


def main() -> None:
    files = list_files()
    rows = []
    for p in files[::2]:                      # every 2nd file: 47 passages
        try:
            y = debaseline(load_file(p))
        except Exception as e:                                  # noqa: BLE001
            print(f"  !! {p.name}: {e}")
            continue
        o = {}
        for g, chs in GROUPS.items():
            ons = [onset(y[c].to_numpy()) for c in chs]
            ons = [v for v in ons if v is not None]
            o[g] = float(np.median(ons)) if ons else np.nan
        if any(np.isnan(v) for v in o.values()):
            continue
        rows.append({"name": p.name, **{f"onset_{g}": v for g, v in o.items()}})

    df = pd.DataFrame(rows)

    print("=" * 96)
    print("PASSAGE ONSET PER CHANNEL GROUP (samples @500 Hz), and implied speed")
    print("=" * 96)
    print(f"{'file':<28}{'A':>8}{'B':>8}{'C':>8}  {'order':<12}"
          f"{'|AB| m/s->km/h':>16}{'|BC| km/h':>11}{'ratio':>8}")

    out = []
    for _, r in df.iterrows():
        a, b, c = r["onset_A"], r["onset_B"], r["onset_C"]
        # A=Head3, B=Head2, C=Head1  =>  |A-B| spans 10 m, |B-C| spans 15 m
        dab, dbc = abs(b - a) / FS, abs(c - b) / FS
        if dab <= 0 or dbc <= 0:
            continue
        v_ab = D23 / dab * 3.6
        v_bc = D12 / dbc * 3.6
        order = "C->B->A" if c < b < a else ("A->B->C" if a < b < c else "MIXED")
        ratio = dbc / dab
        print(f"{r['name'][:26]:<28}{a:8.0f}{b:8.0f}{c:8.0f}  {order:<12}"
              f"{v_ab:16.1f}{v_bc:11.1f}{ratio:8.2f}")
        out.append({"name": r["name"], "order": order, "v_ab": v_ab,
                    "v_bc": v_bc, "ratio": ratio,
                    "diff_pct": 100 * abs(v_ab - v_bc) / ((v_ab + v_bc) / 2)})

    res = pd.DataFrame(out)
    good = res[res["order"] != "MIXED"]

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"passages measured        : {len(res)}")
    print(f"monotonic head order     : {len(good)}  "
          f"({100*len(good)/max(len(res),1):.0f} %)")
    print(f"\ndirection split:\n{good['order'].value_counts().to_string()}")
    print(f"\nspacing ratio |BC|/|AB| : median {good['ratio'].median():.3f}  "
          f"(expected {D12/D23:.3f} if A=Head3, B=Head2, C=Head1)")
    print(f"speed agreement          : median {good['diff_pct'].median():.1f} %, "
          f"90th pct {good['diff_pct'].quantile(0.9):.1f} %")
    print(f"speed range              : {good[['v_ab','v_bc']].min().min():.0f} - "
          f"{good[['v_ab','v_bc']].max().max():.0f} km/h")

    # If the ratio is systematically off 1.5, the true A-B distance differs
    # from the nominal 10 m. Report it; do NOT silently adopt it.
    r_med = good["ratio"].median()
    implied_d23 = D12 / r_med
    print(f"\nimplied Head2<->Head3 separation, taking Head1<->Head2 = {D12} m as exact:")
    print(f"  {implied_d23:.2f} m  (nominal {D23} m, "
          f"{100*(implied_d23-D23)/D23:+.1f} %)")

    print("\nsamples_per_peak over the observed speed range (briefing §3.4):")
    k = CONFIG["geometry"]["samples_per_peak_numerator_at_500hz"]
    for v in [good[['v_ab','v_bc']].min().min(), good["v_ab"].median(),
              good[['v_ab','v_bc']].max().max()]:
        print(f"  v = {v:6.1f} km/h -> {k/v:5.1f} samples")

    res.to_csv(OUT / "onset_direction.csv", index=False)
    print(f"\nwrote {OUT/'onset_direction.csv'}")


if __name__ == "__main__":
    main()

"""Phase 0 reconnaissance: group the 12 spectral channels into counting heads.

RECONNAISSANCE ONLY. This is not the pipeline. It uses a rolling-median
baseline and a normalised cross-correlation purely to answer the structural
question "which channels see the train at the same instant?". No peak
detection, no thresholds, nothing here feeds Phase 1.

Logic
-----
The 12 FBGs sit at three trackside locations 15 m and 10 m apart (briefing
§4.3). Sensors in the SAME head see a given wheel at the same time; sensors in
DIFFERENT heads see it separated by d/v seconds. So the lag of peak
cross-correlation between channels partitions them into three groups of four,
and the ratio of the between-group lags must be 15:10 regardless of the train's
speed — which makes it a self-checking test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG                    # noqa: E402
from lightrack.loader import CH_COLS, load_file, list_files  # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
WIN = CONFIG["baseline"]["rolling_window_samples"]
D12 = CONFIG["geometry"]["head1_to_head2_m"]
D23 = CONFIG["geometry"]["head2_to_head3_m"]
MAX_LAG = int(3.0 * FS)          # +/- 3 s covers 15 m down to ~18 km/h
OUT = Path(CONFIG["paths"]["out_dir"])


def debaseline(df: pd.DataFrame) -> np.ndarray:
    """Remove thermal drift with a 1 s centred rolling median -> pm."""
    x = df[CH_COLS]
    y = (x - x.rolling(WIN, center=True, min_periods=1).median()) * 1000.0
    return y.to_numpy()


def find_passage(y: np.ndarray) -> tuple[int, int]:
    """Return (i0, i1) around the largest coherent excursion in the record."""
    # Normalise each channel so one loud channel cannot dominate the search.
    s = np.abs(y) / (np.median(np.abs(y), axis=0, keepdims=True) + 1e-9)
    env = pd.Series(s.mean(axis=1)).rolling(int(0.2 * FS), center=True,
                                            min_periods=1).mean().to_numpy()
    c = int(np.argmax(env))
    half = int(8.0 * FS)          # generous: a 2x3-car AM08 at 100 km/h ~ 6 s
    return max(0, c - half), min(len(env), c + half)


def lag_to(ref: np.ndarray, sig: np.ndarray) -> tuple[int, float]:
    """Lag (samples) maximising normalised cross-correlation, and its value.

    Positive lag => `sig` is DELAYED relative to `ref`.
    """
    a = ref - ref.mean()
    b = sig - sig.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0, 0.0
    full = np.correlate(a, b, mode="full") / (na * nb)
    lags = np.arange(-len(b) + 1, len(a))
    keep = np.abs(lags) <= MAX_LAG
    full, lags = full[keep], lags[keep]
    k = int(np.argmax(full))
    return int(lags[k]), float(full[k])


def analyse(path: Path, ref_ch: int = 0) -> dict | None:
    df = load_file(path)
    y = debaseline(df)
    i0, i1 = find_passage(y)
    seg = y[i0:i1]
    if seg.shape[0] < int(2 * FS):
        return None
    amp = seg.max(axis=0) - seg.min(axis=0)
    ref = seg[:, ref_ch]
    row = {"name": path.name, "i0": i0, "i1": i1}
    for j, ch in enumerate(CH_COLS):
        lag, r = lag_to(ref, seg[:, j])
        row[f"{ch}_lag"] = lag
        row[f"{ch}_r"] = r
        row[f"{ch}_amp"] = float(amp[j])
    return row


def main() -> None:
    files = list_files()
    # Spread the sample across the campaign rather than one day.
    picks = [f for f in files if f.name.startswith(("20260515T1223", "20260602T1223"))]
    picks += [f for f in files if f not in picks][::6]
    picks = picks[:24]

    rows = []
    for p in picks:
        try:
            r = analyse(p)
        except Exception as e:                                   # noqa: BLE001
            print(f"  !! {p.name}: {e}")
            continue
        if r:
            rows.append(r)
            print(f"  {p.name}  window {r['i0']/FS:7.2f}-{r['i1']/FS:7.2f} s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "channel_lags.csv", index=False)

    print("\n" + "=" * 78)
    print("CROSS-CORRELATION LAG vs ch01, per channel (samples @500 Hz)")
    print("=" * 78)
    print(f"{'ch':>5} {'median lag':>11} {'IQR':>8} {'median r':>9} "
          f"{'median amp pm':>14}")
    med = {}
    for ch in CH_COLS:
        lags = df[f"{ch}_lag"]
        rs = df[f"{ch}_r"]
        amps = df[f"{ch}_amp"]
        med[ch] = lags.median()
        q = lags.quantile(0.75) - lags.quantile(0.25)
        print(f"{ch:>5} {lags.median():11.1f} {q:8.1f} {rs.median():9.3f} "
              f"{amps.median():14.1f}")

    print("\nlag in metres, assuming the three groups are the three heads:")
    print("  (cannot convert without v; see head-ratio test below)")

    print("\n" + "=" * 78)
    print("PER-FILE LAG PATTERN (samples) — grouping should be stable per file")
    print("=" * 78)
    print(f"{'file':<28}" + "".join(f"{c[2:]:>6}" for c in CH_COLS))
    for _, r in df.iterrows():
        print(f"{r['name'][:26]:<28}" + "".join(f"{int(r[f'{c}_lag']):>6}" for c in CH_COLS))

    print("\n" + "=" * 78)
    print("HEAD-RATIO SELF-CHECK")
    print("=" * 78)
    print("For each file, cluster the 12 lags into 3 groups and test whether the")
    print(f"two inter-group separations are in the ratio {D12}:{D23} = "
          f"{D12/D23:.2f} (speed-independent).")
    ratios = []
    for _, r in df.iterrows():
        lags = np.array([r[f"{c}_lag"] for c in CH_COLS], dtype=float)
        order = np.sort(lags)
        # 3 groups of 4 in sorted order
        g = [order[0:4], order[4:8], order[8:12]]
        c = [np.median(x) for x in g]
        s1, s2 = c[1] - c[0], c[2] - c[1]
        if min(abs(s1), abs(s2)) > 1e-9:
            ratios.append(abs(s1) / abs(s2) if abs(s1) > abs(s2) else abs(s2) / abs(s1))
        print(f"  {r['name'][:26]:<28} centroids {c[0]:8.1f} {c[1]:8.1f} {c[2]:8.1f}"
              f"   sep {s1:7.1f} {s2:7.1f}")
    if ratios:
        ratios = np.array(ratios)
        print(f"\n  separation ratio: median {np.median(ratios):.2f}  "
              f"(expected {D12/D23:.2f} if grouping is real)")
    print(f"\nwrote {OUT/'channel_lags.csv'}")


if __name__ == "__main__":
    main()

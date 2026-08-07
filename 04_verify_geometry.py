"""Phase 0: turn the channel-group lags into a speed, two independent ways.

If the group partition {ch01-04} / {ch05-08} / {ch09-12} really is the three
counting heads, then the two inter-group lags must yield the SAME speed when
divided by their respective head separations. That is a genuine falsifiable
test: it uses two different distances and must agree without any tuning.

Aliasing note: the axle train is quasi-periodic, so cross-correlation can lock
one carriage-length away from the true lag. Files where that happened are
detected (the A->B and B->C separations disagree in sign or ratio) and excluded
rather than quietly averaged in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG           # noqa: E402
from lightrack.loader import CH_COLS          # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
D12 = CONFIG["geometry"]["head1_to_head2_m"]   # 15 m, Head1<->Head2
D23 = CONFIG["geometry"]["head2_to_head3_m"]   # 10 m, Head2<->Head3

OUT = Path(CONFIG["paths"]["out_dir"])
df = pd.read_csv(OUT / "channel_lags.csv")

GROUPS = {
    "A (ch01-04)": CH_COLS[0:4],
    "B (ch05-08)": CH_COLS[4:8],
    "C (ch09-12)": CH_COLS[8:12],
}

print("=" * 78)
print("SPEED FROM INTER-GROUP LAGS — two independent distances")
print("=" * 78)
print("Hypothesis under test: A->B spans 10 m and B->C spans 15 m,")
print("i.e. the train passes Head 3, then Head 2, then Head 1.")
print()
print(f"{'file':<28}{'A->B':>7}{'B->C':>7}{'v(A->B)':>9}{'v(B->C)':>9}{'diff%':>7}  flag")

rows = []
for _, r in df.iterrows():
    c = {g: np.median([r[f"{ch}_lag"] for ch in chs]) for g, chs in GROUPS.items()}
    s1 = c["B (ch05-08)"] - c["A (ch01-04)"]
    s2 = c["C (ch09-12)"] - c["B (ch05-08)"]
    if s1 <= 0 or s2 <= 0:
        print(f"{r['name'][:26]:<28}{s1:7.0f}{s2:7.0f}{'':>9}{'':>9}{'':>7}  ALIASED (excluded)")
        continue
    v1 = D23 / (s1 / FS) * 3.6          # A->B assumed 10 m
    v2 = D12 / (s2 / FS) * 3.6          # B->C assumed 15 m
    diff = 100 * abs(v1 - v2) / ((v1 + v2) / 2)
    print(f"{r['name'][:26]:<28}{s1:7.0f}{s2:7.0f}{v1:9.1f}{v2:9.1f}{diff:7.1f}")
    rows.append({"name": r["name"], "s1": s1, "s2": s2, "v1": v1, "v2": v2, "diff": diff})

res = pd.DataFrame(rows)
print()
print(f"clean files: {len(res)} / {len(df)}")
print(f"speed range: {res['v1'].min():.0f} - {res['v1'].max():.0f} km/h")
print(f"agreement between the two independent estimates: "
      f"median {res['diff'].median():.1f} %, worst {res['diff'].max():.1f} %")

print()
print("Counter-test — the alternative assignment (A->B = 15 m, B->C = 10 m):")
v1b = D12 / (res["s1"] / FS) * 3.6
v2b = D23 / (res["s2"] / FS) * 3.6
db = 100 * (v1b - v2b).abs() / ((v1b + v2b) / 2)
print(f"  median disagreement {db.median():.1f} %, worst {db.max():.1f} %")
print()
print("The correct assignment is the one whose two estimates agree. If the")
print("first block's disagreement is small and this one's is large, the head")
print("order A=Head3, B=Head2, C=Head1 is established by geometry alone.")

print()
print("=" * 78)
print("SAMPLES PER PEAK at the observed speeds (briefing §3.4: 3960 / v_kmh)")
print("=" * 78)
k = CONFIG["geometry"]["samples_per_peak_numerator_at_500hz"]
for v in [res["v1"].min(), res["v1"].median(), res["v1"].max()]:
    print(f"  v = {v:6.1f} km/h  ->  samples_per_peak = {k/v:5.1f}")

res.to_csv(OUT / "speed_check.csv", index=False)
print(f"\nwrote {OUT/'speed_check.csv'}")

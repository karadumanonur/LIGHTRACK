"""Phase 0: detail on 2026-05-19 (the outlier day) and per-channel time trend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack.config import CONFIG  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
idx = pd.read_csv(OUT / "file_index.csv", parse_dates=["utc", "local"]).sort_values("utc")
ch = pd.read_csv(OUT / "channel_stats.csv", parse_dates=["utc"])

print("=" * 78)
print("ALL FILES ON 2026-05-19 (31 files — the anomalous day)")
print("=" * 78)
d19 = idx[idx["utc"].dt.date == pd.Timestamp("2026-05-19").date()]
for _, r in d19.iterrows():
    print(f"  UTC {r['utc'].strftime('%H:%M:%S')}   local {r['local'].strftime('%H:%M:%S')}   {r['name']}")

print("\n" + "=" * 78)
print("TIME-OF-DAY DISTRIBUTION (UTC hour), all files")
print("=" * 78)
print(idx["utc"].dt.hour.value_counts().sort_index().to_string())

print("\n" + "=" * 78)
print("PER-CHANNEL DAILY TREND — median within-file p2p, in pm")
print("(descriptive only: p2p still contains thermal drift, so read the SHAPE")
print(" of each column over days, not its absolute level)")
print("=" * 78)
ch["date"] = ch["utc"].dt.date
piv = ch.pivot_table(index="date", columns="channel", values="p2p_pm", aggfunc="median")
pd.set_option("display.width", 200)
print(piv.round(0).to_string())

print("\n" + "=" * 78)
print("PER-CHANNEL DAILY TREND — median within-file std, in pm")
print("=" * 78)
piv2 = ch.pivot_table(index="date", columns="channel", values="std_pm", aggfunc="median")
print(piv2.round(1).to_string())

print("\n" + "=" * 78)
print("NORMALISED TO EACH CHANNEL'S FIRST DAY (p2p) — the §8 durability view")
print("=" * 78)
norm = piv.div(piv.iloc[0], axis=1)
print(norm.round(2).to_string())

piv.to_csv(OUT / "daily_p2p_pm.csv")
piv2.to_csv(OUT / "daily_std_pm.csv")
norm.to_csv(OUT / "daily_p2p_normalised.csv")
print(f"\nwrote daily_*.csv to {OUT}")

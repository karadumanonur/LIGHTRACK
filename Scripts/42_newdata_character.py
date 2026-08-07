"""Items 1-2 — is the wavelength scatter thermal, and what is this set's character?

ITEM 1. The +-223 pm scatter across 11 channels was read as possible rework. At
10 pm/degC that band is about +-22 degC, entirely consistent with early June vs
late July. Two tests distinguish thermal from mechanical:
   (a) COMMON MODE. A thermal shift moves every channel TOGETHER. Per-sensor
       strain changes do not. Measured as the variance explained by the first
       principal component of the 12 centre time series.
   (b) TIME OF DAY. If thermal, centres track the diurnal cycle within the set.
If the scatter is thermal it is NOT evidence of rework, and PAD 9's -553 pm
stands out more sharply as the one non-thermal shift.

ITEM 2. 609 records over 11 days is 20-90/day against POC1's 3-6. Either the
filter was relaxed or the criterion changed. Slot arithmetic (A-01) settles it,
and the minimum duration of 124.9 s against POC1's 143.7 s floor needs an
explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, load_file, parse_filename_utc, to_local  # noqa: E402
from lightrack.loader import CH_COLS                                    # noqa: E402

NEW = Path(r"C:\Users\onurk\Desktop\data\Valid_data_new")
OUT = Path(CONFIG["paths"]["out_dir"])
FS = CONFIG["acquisition"]["fs_hz"]
SLOT = CONFIG["acquisition"]["slot_s"]
pd.set_option("display.width", 250)
LINE = "=" * 112

inv = pd.read_csv(OUT / "newdata_inventory.csv")
inv["utc"] = pd.to_datetime(inv["utc"])
inv["local"] = pd.to_datetime(inv["local"])
inv["date"] = inv["local"].dt.date
inv = inv.sort_values("utc").reset_index(drop=True)

# ============================================================== ITEM 2
print(f"{LINE}\nITEM 2a — GAP STRUCTURE AND RETENTION PER DAY\n{LINE}")
print(f"{'date':<13}{'dow':>5}{'files':>7}{'span h':>9}{'slots':>8}"
      f"{'retention':>11}{'gaps = k x slot':>18}")
tot_f = tot_s = 0
for d, g in inv.groupby("date"):
    g = g.sort_values("utc")
    if len(g) < 2:
        continue
    gaps = g["utc"].diff().dt.total_seconds().dropna()
    k = np.round(gaps / SLOT)
    resid = (gaps - k * SLOT).abs()
    span = (g["utc"].iloc[-1] - g["utc"].iloc[0]).total_seconds()
    slots = int(round(span / SLOT)) + 1
    tot_f += len(g)
    tot_s += slots
    dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][pd.Timestamp(d).weekday()]
    print(f"{str(d):<13}{dow:>5}{len(g):>7}{span/3600:>9.2f}{slots:>8}"
          f"{100*len(g)/slots:>10.1f}%{100*(resid < 3.0).mean():>17.0f}%")
print(f"\noverall: {tot_f} files / {tot_s} slots = {100*tot_f/tot_s:.1f} % retention")
print(f"POC1 for comparison: 20.1 % (05-19 contiguous block), 7.8 % campaign-wide")

print(f"\n{LINE}\nITEM 2c — THE SHORT FILES\n{LINE}")
inv["slot_mult"] = inv["duration_s"] / SLOT
short = inv[inv["duration_s"] < 140]
print(f"records shorter than 140 s: {len(short)}/{len(inv)}")
print(f"duration distribution:")
h, e = np.histogram(inv["duration_s"], bins=np.arange(120, 152, 2))
for c, lo in zip(h, e):
    if c:
        print(f"  {lo:5.0f}-{lo+2:5.0f} s  {'#'*min(60,int(60*c/h.max())):<60} {c}")
print(f"\nPOC1 range was 143.658 - 147.258 s; this set is "
      f"{inv['duration_s'].min():.1f} - {inv['duration_s'].max():.1f} s")
if len(short):
    print(f"\nshortest 10, with the gap to the NEXT record:")
    idx = short.nsmallest(10, "duration_s").index
    for i in idx:
        nxt = inv["utc"].iloc[i + 1] - inv["utc"].iloc[i] if i + 1 < len(inv) else None
        print(f"  {inv['local'].iloc[i]:%Y-%m-%d %H:%M:%S}  "
              f"{inv['duration_s'].iloc[i]:7.3f} s   next record +"
              f"{nxt.total_seconds() if nxt is not None else float('nan'):.1f} s")
    print("\nIf a short file is followed by the next at ~144.5 s, the RECORDER")
    print("kept its slot cadence and the file itself was truncated on write.")
    print("If the next follows at ~the short duration, the cadence changed.")

# ============================================================== ITEM 1
print(f"\n{LINE}\nITEM 1 — IS THE WAVELENGTH SCATTER THERMAL?\n{LINE}")
rng = np.random.default_rng(0)
samp = (inv.groupby(inv["local"].dt.hour, group_keys=False)
        .apply(lambda g: g.sample(min(3, len(g)), random_state=0)))
samp = samp.sort_values("utc")
print(f"sampling {len(samp)} records across "
      f"{samp['local'].dt.hour.nunique()} hours of day and "
      f"{samp['date'].nunique()} days")

rows = []
for _, r in samp.iterrows():
    df = load_file(NEW / r["rel"])
    med = df[CH_COLS].median()
    rows.append({"utc": r["utc"], "local": r["local"], "hour": r["local"].hour,
                 "date": r["date"], **{c: med[c] for c in CH_COLS}})
c = pd.DataFrame(rows).sort_values("utc").reset_index(drop=True)
M = c[CH_COLS].to_numpy()
Mc = M - M.mean(axis=0)

# (a) common mode
u, s, vt = np.linalg.svd(Mc, full_matrices=False)
pc1 = s[0] ** 2 / (s ** 2).sum()
load1 = vt[0]
print(f"\n(a) COMMON MODE")
print(f"    variance explained by PC1: {100*pc1:.1f} %")
print(f"    PC1 loadings (all same sign => common mode):")
print("      " + "  ".join(f"{v:+.2f}" for v in load1))
print(f"    all loadings same sign: {bool(np.all(load1 > 0) or np.all(load1 < 0))}")
print(f"    per-channel sd over the set (pm): " +
      "  ".join(f"{v*1000:.0f}" for v in M.std(axis=0)))

# (b) time of day
print(f"\n(b) TIME OF DAY")
cm = Mc.mean(axis=1) * 1000        # common-mode component, pm
c["cm_pm"] = cm
byh = c.groupby("hour")["cm_pm"].agg(["mean", "size"])
print(f"    common-mode centre shift by local hour (pm, relative to set mean):")
for h, r in byh.iterrows():
    print(f"      {h:02d}:00  {r['mean']:+7.1f} pm   (n={int(r['size'])})")
rho = c["hour"].corr(c["cm_pm"])
print(f"    correlation(local hour, common-mode shift) = {rho:+.3f}")
print(f"    peak-to-peak of the hourly means: "
      f"{byh['mean'].max()-byh['mean'].min():.0f} pm "
      f"= {(byh['mean'].max()-byh['mean'].min())/10:.1f} degC at 10 pm/degC")

# (c) residual per channel after removing common mode
res = (Mc - cm[:, None] / 1000) * 1000
print(f"\n(c) PER-CHANNEL RESIDUAL after removing the common mode (pm sd):")
for i, ch in enumerate(CH_COLS):
    print(f"      {ch}  {res[:, i].std():6.1f}")

old_c = pd.read_csv(OUT / "channel_centres.csv", index_col=0).iloc[:, 0]
delta = (c[CH_COLS].median() - old_c) * 1000
print(f"\n(d) SHIFT vs POC1, after removing this set's own common mode:")
d_cm = delta.mean()
print(f"    common-mode offset POC1 -> new: {d_cm:+.0f} pm "
      f"= {d_cm/10:+.1f} degC at 10 pm/degC")
print(f"{'ch':>6}{'raw delta pm':>14}{'minus common mode':>20}")
for ch in CH_COLS:
    print(f"{ch:>6}{delta[ch]:>+14.0f}{delta[ch]-d_cm:>+20.0f}")
r = (delta - d_cm).abs()
print(f"\n    residual after common mode: max {r.max():.0f} pm on "
      f"{r.idxmax()}, next {r.nlargest(2).iloc[1]:.0f} pm, median {r.median():.0f} pm")
c.to_csv(OUT / "newdata_thermal.csv", index=False)
print(f"\nwrote newdata_thermal.csv")

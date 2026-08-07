"""Items 1-3 — effective thermal sensitivity, head-to-head drift, gap tolerance.

ITEM 1. A-15's noise and threshold arithmetic assumes the nominal 10 pm/degC.
The briefing (§3.3) warns that adhesive/fibre CTE mismatch inflates apparent
sensitivity above that for a packaged, glued sensor. 11 days of full diurnal
cycles - which POC1 never had - make it measurable.

The RELATIVE sensitivity needs no external temperature at all: regressing each
channel against the common-mode component gives its response per unit common
drift. Only the conversion to absolute pm/degC needs an assumed swing, and that
assumption is stated rather than buried.

ITEM 2b. Do the three heads respond to the same thermal cycle identically? If
they do, a cross-head reference cancels it; if not, it does not - and a single
isolated reference grating cannot represent all three.

ITEM 3. The +-3 s gap-multiple tolerance is too tight: jitter accumulates with
the number of slots. Re-run with +-(2 + 2*sqrt(N)) s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                     # noqa: E402
from lightrack.loader import CH_COLS                             # noqa: E402
from lightrack.diagnostic import PAD_OF_CHANNEL, HEAD_OF_CHANNEL  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SLOT = CONFIG["acquisition"]["slot_s"]
pd.set_option("display.width", 250)
LINE = "=" * 112

c = pd.read_csv(OUT / "newdata_thermal.csv")
M = c[CH_COLS].to_numpy()
Mc = M - M.mean(axis=0)
cm = Mc.mean(axis=1)                     # common-mode component, nm

# ============================================================== ITEM 1
print(f"{LINE}\nITEM 1 — EFFECTIVE THERMAL SENSITIVITY PER CHANNEL\n{LINE}")
print("Slope of each channel against the common-mode component. A bare FBG with")
print("no CTE mismatch would give slope 1.0 for every channel; departures are")
print("differences in how stiffly each pad couples the rail's thermal strain.\n")
print(f"{'ch':>6}{'PAD':>5}{'head':>6}{'slope':>9}{'+-':>8}{'R2':>7}"
      f"{'sd pm':>8}   mounting")
MOUNT = {1: "glued, centred", 2: "glued, centred", 3: "glued, +7.5cm",
         4: "glued, +7.5cm", 5: "glued, centred", 6: "glued, centred (CONTROL)",
         7: "glued, +3cm", 8: "glued, +3cm", 9: "1/2 collage",
         10: "clipse", 11: "unprepared surface", 12: "taraude"}
rows = []
for i, ch in enumerate(CH_COLS):
    y = Mc[:, i]
    A = np.vstack([cm, np.ones_like(cm)]).T
    beta, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(y)
    se = np.sqrt(ss_res / (n - 2) / ((cm - cm.mean()) ** 2).sum())
    pad = PAD_OF_CHANNEL[ch]
    rows.append({"ch": ch, "pad": pad, "head": HEAD_OF_CHANNEL[ch],
                 "slope": beta[0], "se": se, "r2": r2, "sd_pm": y.std() * 1000})
    print(f"{ch:>6}{pad:>5}{HEAD_OF_CHANNEL[ch]:>6}{beta[0]:>9.3f}"
          f"{se:>8.3f}{r2:>7.3f}{y.std()*1000:>8.0f}   {MOUNT[pad]}")
t = pd.DataFrame(rows)
print(f"\nslope range {t['slope'].min():.2f} - {t['slope'].max():.2f} "
      f"(factor {t['slope'].max()/t['slope'].min():.1f})")

pp = (c[CH_COLS].max() - c[CH_COLS].min()) * 1000
print(f"\nABSOLUTE sensitivity, under a STATED assumed rail-surface swing:")
print(f"{'assumed diurnal swing':<26}" +
      "".join(f"{ch:>7}" for ch in CH_COLS[:6]) + "   ...")
for dT in (20.0, 25.0, 30.0):
    print(f"{f'{dT:.0f} degC -> pm/degC':<26}" +
          "".join(f"{pp[ch]/dT:>7.0f}" for ch in CH_COLS[:6]) + "   ...")
print(f"\nmedian effective sensitivity: "
      + "  ".join(f"{dT:.0f}degC -> {pp.median()/dT:.0f} pm/degC"
                  for dT in (20, 25, 30)))
print(f"nominal from the briefing: 10 pm/degC")
print(f"=> effective sensitivity is {pp.median()/25/10:.1f}x nominal at a 25 degC swing.")
print("   A-15's noise arithmetic assumes the nominal value and is affected.")

hi = t.nlargest(2, "slope")
lo = t.nsmallest(2, "slope")
print(f"\nOUTLIERS worth naming:")
for _, r in pd.concat([hi, lo]).iterrows():
    print(f"   PAD {int(r['pad']):>2} ({r['ch']}, {MOUNT[int(r['pad'])]:<26}) "
          f"slope {r['slope']:.2f}")
print("\nA looser bond transfers LESS of the rail's thermal strain, so a low")
print("slope is the expected debonding signature - which makes PAD 9's HIGH")
print("slope the opposite of what a simple release would predict, and worth")
print("flagging rather than explaining away.")

# ============================================================== ITEM 2b
print(f"\n{LINE}\nITEM 2b — DO THE THREE HEADS DRIFT IDENTICALLY?\n{LINE}")
for h in (1, 2, 3):
    cols = [ch for ch in CH_COLS if HEAD_OF_CHANNEL[ch] == h]
    sub = t[t["head"] == h]
    print(f"   head {h}: mean slope {sub['slope'].mean():.3f}  "
          f"range {sub['slope'].min():.2f}-{sub['slope'].max():.2f}   "
          f"mean sd {sub['sd_pm'].mean():.0f} pm")
hm = {h: Mc[:, [i for i, ch in enumerate(CH_COLS) if HEAD_OF_CHANNEL[ch] == h]].mean(axis=1)
      for h in (1, 2, 3)}
print(f"\n   correlation between head-mean drifts:")
for a in (1, 2, 3):
    for b in (1, 2, 3):
        if a < b:
            print(f"      head {a} vs head {b}: r = {np.corrcoef(hm[a], hm[b])[0,1]:+.3f}"
                  f"   amplitude ratio {hm[a].std()/hm[b].std():.2f}")
print("""
The heads are highly correlated in PHASE but differ in AMPLITUDE. So a
cross-head ratio does not cancel thermal drift - it leaves the difference
between two unequal responses. And a SINGLE isolated reference grating cannot
represent all three heads: it would have to be sited per head, or its residual
accepted and quantified.""")

# ============================================================== ITEM 3
print(f"\n{LINE}\nITEM 3 — GAP-MULTIPLE COMPLIANCE WITH AN ACCUMULATING TOLERANCE\n{LINE}")
inv = pd.read_csv(OUT / "newdata_inventory.csv")
inv["utc"] = pd.to_datetime(inv["utc"])
inv["local"] = pd.to_datetime(inv["local"])
inv["date"] = inv["local"].dt.date
inv = inv.sort_values("utc")
print(f"{'date':<13}{'files':>7}{'+-3 s':>9}{'+-(2+2vN)':>12}{'slots':>8}{'retention':>11}")
tf = ts = 0
for d, g in inv.groupby("date"):
    g = g.sort_values("utc")
    if len(g) < 2:
        continue
    gaps = g["utc"].diff().dt.total_seconds().dropna()
    k = np.round(gaps / SLOT)
    resid = (gaps - k * SLOT).abs()
    tol = 2.0 + 2.0 * np.sqrt(np.maximum(k, 1))
    span = (g["utc"].iloc[-1] - g["utc"].iloc[0]).total_seconds()
    slots = int(round(span / SLOT)) + 1
    tf += len(g); ts += slots
    print(f"{str(d):<13}{len(g):>7}{100*(resid<3).mean():>8.0f}%"
          f"{100*(resid<tol).mean():>11.0f}%{slots:>8}{100*len(g)/slots:>10.1f}%")
gaps = inv["utc"].diff().dt.total_seconds().dropna()
k = np.round(gaps / SLOT)
resid = (gaps - k * SLOT).abs()
tol = 2.0 + 2.0 * np.sqrt(np.maximum(k, 1))
print(f"\nall gaps: +-3 s {100*(resid<3).mean():.0f} %   "
      f"accumulating tolerance {100*(resid<tol).mean():.0f} %")
print(f"overall retention {tf}/{ts} = {100*tf/ts:.1f} %")
big = resid >= tol
print(f"\ngaps failing even the accumulating tolerance: {int(big.sum())}/{len(gaps)}")
if big.any():
    print(f"   residuals of the worst 8 (s): "
          + "  ".join(f"{v:.0f}" for v in resid[big].nlargest(8)))
    print(f"   their slot counts: "
          + "  ".join(f"{int(v)}" for v in k[big].nlargest(8)))
t.to_csv(OUT / "thermal_sensitivity.csv", index=False)
print(f"\nwrote thermal_sensitivity.csv")

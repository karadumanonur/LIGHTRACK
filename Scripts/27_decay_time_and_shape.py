"""Additions 1 & 2 — the 6 % floor as a TIME, and can shape recover common mode?

PART A. "Everything above 94 % retained signal is invisible" is an amplitude
bound; its practical weight is a time. PAD 6 is the one real gradual case, so
its decay is fitted and the crossing time read off, then generalised to a
detection-time-versus-degradation-rate curve.

PART B. The proposal is that a shape feature normalised to each sensor's own
commissioning value removes the position confound (static) while keeping the
degradation signal (dynamic), and might fire below the trend rule's amplitude
floor under common mode.

That rests on a premise: **that debonding actually changes peak shape as it
progresses.** PAD 9 proves shape can be anomalous at commissioning, which is a
different claim. So the premise is tested against PAD 6 and PAD 11 FIRST, and
the simulation is only calibrated from whatever relationship the data shows.
Assuming the coefficient would be assuming the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, parse_filename_utc, to_local   # noqa: E402
from lightrack.health import DEGRADING, BORDERLINE, HEALTHY  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
K_MON, TREND_WIN, CONFIRM_N, CONFIRM_K = 3.0, 5, 3, 2
HEAD_OF = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
           **{p: 3 for p in (9, 10, 11, 12)}}

pd.set_option("display.width", 240)
LINE = "=" * 106

ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
cls = dict(zip(ebp["date"], ebp["day_class"]))

df = pd.read_csv(OUT / "phase4_features_scored.csv")
df["date"] = df["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
df["day_class"] = df["date"].map(cls)
df["utc"] = df["name"].map(parse_filename_utc)
df = df[df["day_class"] == "normal_weekday"].copy()
o = (df[["name", "utc"]].drop_duplicates().sort_values("utc")
     .reset_index(drop=True).reset_index().rename(columns={"index": "qi"}))
df = df.merge(o[["name", "qi"]], on="name").sort_values(["pad", "qi"])
T = o.set_index("qi")["utc"]
DAY = (T - T.iloc[0]).dt.total_seconds() / 86400.0

P2P = df.pivot_table(index="qi", columns="pad", values="p2p_pm")
SNR = df.pivot_table(index="qi", columns="pad", values="snr")
SHP = df.pivot_table(index="qi", columns="pad", values="neg_over_pos")
S1 = sorted(df.loc[df["session"] == 1, "qi"].unique())

# ============================================================ PART A
print(f"{LINE}\nPART A — the 6 % floor expressed as a time\n{LINE}")
amp = (P2P / P2P.loc[S1].median())
p6 = amp[6]
d = DAY.reindex(p6.index)
mask = p6 > 0
# exponential fit A(t) = exp(-t/tau)
tau = float(-np.polyfit(d[mask], np.log(p6[mask]), 1)[0]) ** -1
c = np.polyfit(d[mask], np.log(p6[mask]), 1)
tau = -1.0 / c[0]
a0 = float(np.exp(c[1]))
pred = a0 * np.exp(-d / tau)
resid = float(np.std(p6 - pred))
print(f"PAD 6 exponential fit  A(t) = {a0:.3f} * exp(-t/{tau:.1f} d)   "
      f"residual sd {resid:.3f}")
print(f"  observed: {p6.iloc[0]:.2f} at day 0 -> {p6.iloc[-1]:.2f} at day {d.iloc[-1]:.1f}")
for frac, lab in ((0.94, "6 % loss  (trend-rule floor)"),
                  (0.90, "10 % loss"),
                  (0.70, "30 % loss (onset definition)"),
                  (0.50, "50 % loss")):
    t = tau * np.log(a0 / frac)
    print(f"  crosses {frac:.2f}  ({lab:<28}) at t = {t:6.1f} days")
t6 = tau * np.log(a0 / 0.94)
print(f"\nPAD 6 crossed the 6 % floor after {t6:.1f} days.")
print(f"The detector confirmed at passage 28 = {T.loc[28]:%Y-%m-%d %H:%M} "
      f"= day {DAY.loc[28]:.1f}.")

print(f"\n{LINE}\nDETECTION TIME vs DEGRADATION RATE — the general form\n{LINE}")
print("For exponential decay with time constant tau, a 6 % loss is reached at")
print("t = tau * ln(1/0.94) = 0.0619 * tau, so detection time scales LINEARLY")
print("with the degradation timescale. There is no rate at which a monotonic")
print("decay escapes the floor - only rates at which it takes a long time.\n")
print(f"{'degradation':<26}{'tau (days)':>12}{'time to 6 % loss':>19}"
      f"{'time to 50 % loss':>20}")
for lab, tau_d in (("PAD 6 (measured)", tau),
                   ("half in 1 month", 30 / np.log(2)),
                   ("half in 6 months", 182 / np.log(2)),
                   ("half in 1 year", 365 / np.log(2)),
                   ("half in 5 years", 1826 / np.log(2)),
                   ("half in 20 years", 7305 / np.log(2))):
    t6_ = tau_d * np.log(1 / 0.94)
    t50 = tau_d * np.log(2)
    print(f"{lab:<26}{tau_d:>12.0f}{t6_:>15.1f} d{t50:>16.0f} d")

print("\nSo the amplitude floor is NOT the binding limitation for a monotonic")
print("decay - it is crossed early in every case above. The real long-horizon")
print("risk is different and is NOT tested by POC1: the trend rule compares")
print("against a COMMISSIONING baseline with a threshold calibrated on 3 weeks")
print("of dispersion. Over months, seasonal thermal cycling and interrogator")
print("calibration drift add variance that is absent from that calibration, so")
print("the operational threshold would have to widen and the effective floor")
print("would rise above 6 %. That is the question POC2 must answer, not the")
print("floor measured here.")

# ============================================================ PART B
print(f"\n{LINE}\nPART B, STEP 1 — does shape actually change as bonding degrades?\n{LINE}")
shp0 = SHP.loc[S1].median()
shape_delta = np.log2(SHP / shp0)
print(f"{'PAD':>4}{'mode':<26}{'shape at commissioning':>24}"
      f"{'shape at end':>14}{'corr(amp, shape)':>19}")
for pad in sorted(DEGRADING | BORDERLINE):
    mode = {6: "progressive decline", 9: "over-respond then collapse",
            11: "abrupt collapse", 7: "mild decline"}[pad]
    a_, s_ = amp[pad], shape_delta[pad]
    ok = a_.notna() & s_.notna()
    r = float(np.corrcoef(a_[ok], s_[ok])[0, 1])
    print(f"{pad:>4}{mode:<26}{shp0[pad]:>24.2f}"
          f"{SHP[pad].iloc[-5:].median():>14.2f}{r:>19.3f}")
print(f"\n{'':>4}{'healthy sensors (control)':<26}{shp0[list(HEALTHY)].median():>24.2f}"
      f"{SHP[list(HEALTHY)].iloc[-5:].median().median():>14.2f}")

print("\nPer-sensor shape drift over the campaign, log2 vs commissioning:")
print(f"{'PAD':>4}{'label':<14}{'median':>9}{'final 5':>10}{'max |drift|':>13}")
for pad in range(1, 13):
    lab = ("DEGRADING" if pad in DEGRADING else
           "borderline" if pad in BORDERLINE else "healthy")
    s_ = shape_delta[pad]
    print(f"{pad:>4}{lab:<14}{s_.median():>9.2f}{s_.iloc[-5:].median():>10.2f}"
          f"{s_.abs().max():>13.2f}")

# calibrate k: shape multiplier per unit amplitude loss, from the data
rows = []
for pad in sorted(DEGRADING):
    a_, s_ = amp[pad], SHP[pad] / shp0[pad]
    ok = a_.notna() & s_.notna() & (a_ < 0.95)
    if ok.sum() < 5:
        continue
    # s = 1 + k(1-a)  ->  k = (s-1)/(1-a)
    k = float(np.median((s_[ok] - 1) / (1 - a_[ok])))
    rows.append({"pad": pad, "k": k, "n": int(ok.sum())})
kt = pd.DataFrame(rows)
print(f"\nempirical shape-vs-amplitude coefficient  s = 1 + k(1-a):")
print(kt.to_string(index=False))
K_EMP = float(kt["k"].median()) if len(kt) else 0.0
print(f"  median k = {K_EMP:.2f}")

# ============================================================ PART B step 2
print(f"\n{LINE}\nPART B, STEP 2 — common mode WITH shape perturbation\n{LINE}")
START = 20


def trend_of(snr):
    return np.log2(snr / snr.loc[S1].median()).rolling(TREND_WIN, min_periods=2).median()


def shape_of(shp):
    return np.log2(shp / shp.loc[S1].median()).rolling(TREND_WIN, min_periods=2).median()


def make_ratio(p2p, kind):
    out = {}
    for pad in range(1, 13):
        mates = ([q for q in range(1, 13) if HEAD_OF[q] == HEAD_OF[pad] and q != pad]
                 if kind == "within" else
                 [q for h, ps in {1: [1, 2], 2: [5, 6], 3: [9, 10]}.items()
                  if h != HEAD_OF[pad] for q in ps])
        out[pad] = np.log2(p2p[pad] / p2p[mates].median(axis=1).replace(0, np.nan))
    w = pd.DataFrame(out)
    return w - w.loc[S1].median()


VICTIM = 1


def synth(att: float, k: float) -> dict:
    p2p, snr, shp = P2P.copy(), SNR.copy(), SHP.copy()
    idx = p2p.index[START:]
    p2p.loc[idx, :] *= att
    snr.loc[idx, :] *= att
    shp.loc[idx, :] *= (1 + k * (1 - att))          # shape perturbs with the loss
    pre = slice(0, START)
    res = {}
    for kind, nm in (("within", "pair_within"), ("cross", "pair_cross")):
        w = make_ratio(p2p, kind)
        v = pd.concat([w[q].iloc[pre] for q in HEALTHY - {VICTIM}]).dropna()
        thr = v.median() - K_MON * float(np.median(np.abs(v - v.median())) * 1.4826)
        c = (w[VICTIM] < thr).fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        res[nm] = bool(c.iloc[START:].any())
    for mat, fn, nm, up in ((snr, trend_of, "trend", False),
                            (shp, shape_of, "shape", True)):
        m = fn(mat)
        v = pd.concat([m[q].iloc[pre] for q in HEALTHY - {VICTIM}]).dropna()
        s = K_MON * float(np.median(np.abs(v - v.median())) * 1.4826)
        thr = v.median() + s if up else v.median() - s
        f = (m[VICTIM] > thr) if up else (m[VICTIM] < thr)
        c = f.fillna(False).rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K
        res[nm] = bool(c.iloc[START:].any())
    return res


for k, klab in ((K_EMP, f"k = {K_EMP:.2f} (empirical)"), (2.0, "k = 2.0 (strong)")):
    print(f"\nALL HEADS attenuated, shape perturbed with {klab}")
    print(f"{'attenuation':>12}{'loss %':>8}{'within':>10}{'cross':>9}"
          f"{'trend':>9}{'SHAPE':>9}")
    for att in (1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80, 0.70, 0.50):
        r = synth(att, k)
        print(f"{att:>12.2f}{100*(1-att):>8.0f}"
              + "".join(f"{'FIRES' if r[x] else '.':>10}" if x == 'pair_within'
                        else f"{'FIRES' if r[x] else '.':>9}"
                        for x in ("pair_within", "pair_cross", "trend", "shape")))
    fired = [a for a in np.arange(0.995, 0.80, -0.005) if synth(round(a, 3), k)["shape"]]
    tfire = [a for a in np.arange(0.995, 0.80, -0.005) if synth(round(a, 3), k)["trend"]]
    print(f"  shape fires for attenuation <= {max(fired):.3f} "
          f"({100*(1-max(fired)):.1f} % loss)" if fired else "  shape never fires")
    print(f"  trend fires for attenuation <= {max(tfire):.3f} "
          f"({100*(1-max(tfire)):.1f} % loss)" if tfire else "  trend never fires")

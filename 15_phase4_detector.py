"""Phase 4 step 2 — build and validate the self-diagnosis detector.

The question WP3 actually asks is not "can we tell a dead sensor from a live
one" — that is trivial once it is dead. It is **"how early can a sensor flag
itself, and does the answer survive using only what a 2-FBG head has?"**

So every feature is scored on two axes:
  * separability of degrading vs healthy sensors, and
  * DETECTION LATENCY — the first session at which the sensor trips a threshold
    calibrated purely on the healthy population, measured against the session
    at which its amplitude actually collapsed.

Errors are reported separately and never merged, following the briefing's
asymmetry rule adapted to diagnostics:
  * MISSED degradation = you keep trusting a failed sensor  -> safety
  * FALSE alarm        = needless intervention              -> availability
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD                    # noqa: E402
from lightrack.health import DEGRADING, BORDERLINE, HEALTHY  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
df = pd.read_csv(OUT / "phase4_features.csv")
pd.set_option("display.width", 230)
LINE = "=" * 104

FEATURES = {
    "neg_over_pos":   ("|trough|/peak", "self / single-passage"),
    "fwhm_norm":      ("FWHM / peak width", "self / single-passage"),
    "asymmetry":      ("rise-fall asymmetry", "self / single-passage"),
    "snr":            ("peak / noise", "self / single-passage"),
    "noise_pm":       ("baseline noise", "self / single-passage"),
    "count_disagree": ("head axle-count surplus", "head-level"),
    "pair_log2":      ("log2(self / rest of head)", "pair"),
}

# Session at which each degrading sensor's amplitude actually collapsed:
# the first session whose Phase 1 normalised peak-to-peak falls to <= 0.70,
# i.e. a 30 % loss of sensitivity. Used ONLY to measure how early a flag
# arrives — never as an input to the detector.
COLLAPSE = {11: 2, 6: 5, 9: 13}

sess = df.groupby(["pad", "session"]).median(numeric_only=True).reset_index()

# --- commissioning baseline -------------------------------------------------
# The raw pair ratio penalises any sensor that is simply less sensitive than its
# head-mates, which is a fixed property of its mounting, not a fault. Referencing
# each sensor to its OWN session-1 pair ratio removes that constant offset and
# leaves only the change. Operationally this is a commissioning calibration:
# one baseline recorded at install, which a 2-FBG head can certainly afford.
base = (df[df["session"] == 1].groupby("pad")["pair_log2"].median()
        .rename("pair_base"))
df = df.merge(base, on="pad", how="left")
df["pair_delta"] = df["pair_log2"] - df["pair_base"]
sess = df.groupby(["pad", "session"]).median(numeric_only=True).reset_index()

# ------------------------------------------------- feature separability
print(f"{LINE}\nFEATURE VALUES — median per group, whole campaign\n{LINE}")
print(f"{'feature':<16}{'kind':<24}{'healthy':>12}{'PAD 6':>10}{'PAD 9':>10}"
      f"{'PAD 11':>10}{'PAD 7':>10}")
for f, (desc, kind) in FEATURES.items():
    h = df[df["pad"].isin(HEALTHY)][f].median()
    r = [df[df["pad"] == p][f].median() for p in (6, 9, 11, 7)]
    print(f"{f:<16}{kind:<24}{h:12.3f}" + "".join(f"{x:10.3f}" for x in r))

# ------------------------------------------- SESSION 1 — no history at all
print(f"\n{LINE}\nSESSION 1 ONLY — what a single passage reveals with NO history\n{LINE}")
s1 = df[df["session"] == 1]
print(f"{'PAD':>4}{'ch':>6}  {'mounting':<24}{'|trough|/peak':>15}{'FWHM':>8}"
      f"{'asym':>8}{'SNR':>7}  fate")
mnt = {1: "glued, centred", 2: "glued, centred", 3: "glued, +7.5cm",
       4: "glued, +7.5cm", 5: "glued, centred", 6: "glued (CONTROL)",
       7: "glued, +3cm", 8: "glued, +3cm", 9: "1/2 collage",
       10: "clipse", 11: "unprepared surface", 12: "taraude"}
fate = {6: "degrades from s5", 9: "COLLAPSES at s13",
        11: "COLLAPSES at s2", 7: "mild decline"}
for pad in range(1, 13):
    r = s1[s1["pad"] == pad]
    print(f"{pad:>4}{CH_OF_PAD[pad]:>6}  {mnt[pad]:<24}{r['neg_over_pos'].median():15.2f}"
          f"{r['fwhm_norm'].median():8.2f}{r['asymmetry'].median():8.2f}"
          f"{r['snr'].median():7.0f}  {fate.get(pad,'stable')}")

hs1 = s1[s1["pad"].isin(HEALTHY)]["neg_over_pos"]
mu, sd = hs1.median(), hs1.std()
n9 = s1[s1["pad"] == 9]["neg_over_pos"].median()
print(f"\nhealthy |trough|/peak on session 1: median {mu:.2f}, sd {sd:.2f}, "
      f"max {hs1.max():.2f}")
print(f"PAD 9 on session 1               : {n9:.2f}   "
      f"= {(n9-mu)/sd:.1f} sd above the healthy median")
print(f"\n=> PAD 9 is flagged on DAY ONE, {COLLAPSE[9]-1} sessions before it collapses,")
print("   from a single passage, with no history and no neighbour.")
print("   PAD 11 and PAD 6 look entirely normal on session 1 — they are NOT")
print("   detectable in advance by shape. Different failure modes.")

# --------------------------------------------------- detector + latency
print(f"\n{LINE}\nDETECTOR — thresholds calibrated on the HEALTHY population only\n{LINE}")

def calibrate(feat: str, k: float = 4.0, upper: bool = True) -> float:
    """Robust threshold from healthy sensors: median +/- k * MAD-sigma."""
    v = df[df["pad"].isin(HEALTHY)][feat].dropna()
    m = v.median()
    s = np.median(np.abs(v - m)) * 1.4826
    return m + k * s if upper else m - k * s

RULES = [
    ("shape", "neg_over_pos", True,  "|trough|/peak abnormally high — no history needed"),
    ("pair",  "pair_delta",   False, "amplitude fallen vs the rest of its head, "
                                     "relative to commissioning"),
]
thr = {}
for tag, feat, upper, desc in RULES:
    thr[tag] = calibrate(feat, upper=upper)
    print(f"  {tag:<7} {feat:<14} {'>' if upper else '<'} {thr[tag]:+.3f}   ({desc})")

def trips(row) -> bool:
    return (row["neg_over_pos"] > thr["shape"]) or (row["pair_delta"] < thr["pair"])

df["flag"] = df.apply(trips, axis=1)

# Same detector without the commissioning baseline, for comparison.
thr_raw = calibrate("pair_log2", upper=False)
df["flag_nobase"] = ((df["neg_over_pos"] > thr["shape"]) |
                     (df["pair_log2"] < thr_raw))
fs_raw = df.groupby(["pad", "session"])["flag_nobase"].mean().unstack(0) > 0.5
# A sensor is flagged in a session if the majority of that session's passages trip.
fs = df.groupby(["pad", "session"])["flag"].mean().unstack(0) > 0.5

print(f"\n{LINE}\nPER-SESSION FLAG MATRIX  (X = flagged)\n{LINE}")
print("sess " + "".join(f"{p:>5}" for p in range(1, 13)))
for s in fs.index:
    print(f"{s:>4} " + "".join(f"{'    X' if fs.loc[s, p] else '    .'}"
                               for p in range(1, 13)))

print(f"\n{LINE}\nDETECTION LATENCY — how early, relative to actual collapse\n{LINE}")
print(f"{'PAD':>4}{'mounting':<24}{'collapsed at':>14}{'first flagged':>15}"
      f"{'lead time':>11}  verdict")
for pad in sorted(DEGRADING):
    f = fs[pad]
    first = int(f[f].index[0]) if f.any() else None
    coll = COLLAPSE[pad]
    if first is None:
        print(f"{pad:>4}{mnt[pad]:<24}{coll:>14}{'never':>15}{'-':>11}  MISSED")
        continue
    lead = coll - first
    v = ("EARLY WARNING" if lead > 0 else
         "concurrent" if lead == 0 else "after the fact")
    print(f"{pad:>4}{mnt[pad]:<24}{coll:>14}{first:>15}"
          f"{lead:>+11}  {v}")

print(f"\n{LINE}\nERROR RATES — reported separately, never merged\n{LINE}")
h_cols = [p for p in HEALTHY]
fa_sessions = int(fs[h_cols].sum().sum())
tot_sessions = int(fs[h_cols].size)
print(f"FALSE ALARM (availability cost): {fa_sessions}/{tot_sessions} "
      f"healthy sensor-sessions flagged = {100*fa_sessions/tot_sessions:.1f} %")
for p in h_cols:
    n = int(fs[p].sum())
    if n:
        print(f"    PAD {p:>2} ({mnt[p]}): {n} session(s) — "
              f"{list(fs.index[fs[p]])}")
missed = [p for p in DEGRADING if not fs[p].any()]
print(f"\nMISSED degradation (safety cost): {len(missed)}/{len(DEGRADING)}  "
      f"{missed if missed else '- all three detected'}")
print(f"\nBORDERLINE PAD 7 (deck: mild decline to ~0.75): "
      f"{'flagged at session ' + str(int(fs[7][fs[7]].index[0])) if fs[7].any() else 'not flagged'}")

print(f"\n{LINE}\nEFFECT OF THE COMMISSIONING BASELINE\n{LINE}")
fa_raw = int(fs_raw[h_cols].sum().sum())
print(f"{'variant':<34}{'false alarms':>14}{'missed':>9}   first-flag session")
for tag, m in [("raw pair ratio", fs_raw), ("commissioning-baselined", fs)]:
    fa = int(m[h_cols].sum().sum())
    ms = [p for p in DEGRADING if not m[p].any()]
    firsts = {p: (int(m[p][m[p]].index[0]) if m[p].any() else None)
              for p in sorted(DEGRADING)}
    print(f"{tag:<34}{fa:>6}/{tot_sessions:<7}{len(ms):>9}   "
          + "  ".join(f"PAD{p}:{v}" for p, v in firsts.items()))
print("\nThe raw ratio penalises PAD 10 for being half as sensitive as its")
print("head-mates — a fixed property of clamping, not a fault. Baselining each")
print("sensor against its own commissioning ratio removes that offset.")
print("\nNote Head 3 is a deliberately HETEROGENEOUS test head (four different")
print("mounting methods). An operational 2-FBG head carries two identically")
print("mounted gratings, so the pair feature would be better behaved there than")
print("this false-alarm rate suggests.")

# --------------------------------------------------- operating point sweep
print(f"\n{LINE}\nOPERATING POINT — threshold sweep, and why persistence is needed\n{LINE}")
print("Above, PAD 6 flags only in sessions 5,6,7,8,14: it sits ON the threshold")
print("(a drop to ~0.50 of its head-mates is log2 = -1.00 against a -0.975 gate).")
print("An intermittent health flag is useless operationally, so the sweep below")
print("reports flag PERSISTENCE, not just first detection.\n")
print(f"{'k·σ':>5}{'shape thr':>11}{'pair thr':>10}{'false alarm':>13}{'missed':>8}"
      f"   first flag (PAD6/9/11)   post-onset flagged (PAD6/9/11)")

def sweep(k: float):
    ts = calibrate("neg_over_pos", k=k, upper=True)
    tp = calibrate("pair_delta", k=k, upper=False)
    fl = (df["neg_over_pos"] > ts) | (df["pair_delta"] < tp)
    m = df.assign(f=fl).groupby(["pad", "session"])["f"].mean().unstack(0) > 0.5
    fa = int(m[h_cols].sum().sum())
    firsts, pers = {}, {}
    for p in sorted(DEGRADING):
        col = m[p]
        firsts[p] = int(col[col].index[0]) if col.any() else None
        after = col[col.index >= COLLAPSE[p]]
        pers[p] = after.mean() if len(after) else float("nan")
    ms = sum(1 for p in DEGRADING if firsts[p] is None)
    return ts, tp, fa, ms, firsts, pers

FA_BUDGET = 0.02          # 2 % of healthy sensor-sessions
best = None
for k in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    ts, tp, fa, ms, firsts, pers = sweep(k)
    print(f"{k:>5.1f}{ts:>11.2f}{tp:>10.2f}{fa:>7}/{tot_sessions:<5}{ms:>8}"
          f"   {'/'.join(str(firsts[p]) for p in (6,9,11)):>21}"
          f"   {'/'.join(f'{pers[p]:.0%}' for p in (6,9,11)):>28}")
    # Selection follows the briefing's asymmetry rule, adapted to diagnostics:
    # a flag that keeps dropping out means an untrustworthy sensor gets silently
    # re-trusted, which is the SAFETY error. A false alarm costs a maintenance
    # visit. So maximise persistence subject to a false-alarm BUDGET, rather
    # than driving false alarms to zero and accepting a flaky flag.
    if ms == 0 and fa / tot_sessions <= FA_BUDGET:
        if best is None or min(pers.values()) > best[1]:
            best = (k, min(pers.values()), fa)

print(f"\nrecommended operating point: k = {best[0]:.1f}σ  "
      f"— worst-case persistence {best[1]:.0%}, "
      f"{best[2]}/{tot_sessions} false alarms ({100*best[2]/tot_sessions:.1f} %)")
print("Chosen by maximising flag persistence subject to a "
      f"{FA_BUDGET:.0%} false-alarm budget, NOT by minimising false alarms:")
print("a flag that drops out lets a failed sensor be silently re-trusted.")

# persistence-confirmed variant at the recommended k
ts, tp, *_ = sweep(best[0])
fl = (df["neg_over_pos"] > ts) | (df["pair_delta"] < tp)
m = df.assign(f=fl).groupby(["pad", "session"])["f"].mean().unstack(0) > 0.5
conf = m.rolling(3, min_periods=1).sum() >= 2      # 2 of the last 3 sessions
print(f"\nwith '2 of the last 3 sessions' confirmation (latching):")
print(f"{'PAD':>4}{'collapsed':>11}{'confirmed':>11}{'lead':>7}   sessions flagged after onset")
for p in sorted(DEGRADING):
    c = conf[p]
    first = int(c[c].index[0]) if c.any() else None
    after = c[c.index >= COLLAPSE[p]]
    print(f"{p:>4}{COLLAPSE[p]:>11}{str(first):>11}"
          f"{(COLLAPSE[p]-first) if first else 0:>+7}   {after.mean():.0%}")
fa_conf = int(conf[h_cols].sum().sum())
print(f"false alarms with confirmation: {fa_conf}/{tot_sessions}")
conf.to_csv(OUT / "phase4_flag_confirmed.csv")

# ----------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3))
fig.subplots_adjust(wspace=0.24, top=0.80, bottom=0.16, left=0.055, right=0.99)
s = sorted(df["session"].unique())
COL = {"degrading": "#b2182b", "borderline": "#e08214", "healthy": "#c9ccd1"}

for ax_, (feat, ttl) in zip(axes, [
        ("neg_over_pos", "Shape: |trough| / peak\n(self, single passage, no history)"),
        ("pair_delta", "Pair: log₂(self ÷ rest of head), vs commissioning\n"
                       "(within the 2-FBG budget)"),
        ("snr", "Peak / noise\n(self, single passage)")]):
    for pad in range(1, 13):
        v = sess[sess["pad"] == pad].set_index("session")[feat].reindex(s)
        lab = ("degrading" if pad in DEGRADING else
               "borderline" if pad in BORDERLINE else "healthy")
        ax_.plot(s, v, color=COL[lab], lw=2.2 if lab != "healthy" else 1.0,
                 marker="o" if lab != "healthy" else None, ms=4,
                 zorder=3 if lab != "healthy" else 1,
                 label=f"PAD {pad}" if lab != "healthy" else None)
    if feat in ("neg_over_pos", "pair_delta"):
        # draw the RECOMMENDED operating point, not the default k
        t = ts if feat == "neg_over_pos" else tp
        ax_.axhline(t, color="#2166ac", ls="--", lw=1.4,
                    label="threshold (healthy-calibrated)")
    if feat == "snr":
        ax_.set_yscale("log")
    ax_.set_title(ttl, fontsize=10)
    ax_.set_xlabel("measurement session", fontsize=9)
    ax_.set_xticks(s[::2])
    ax_.grid(alpha=0.3, lw=0.5)
    ax_.legend(fontsize=7.5, ncol=2, framealpha=0.92)
    ax_.tick_params(labelsize=8)
axes[0].set_ylabel("feature value", fontsize=9)

fig.suptitle("Lightrack POC1 — Phase 4: sensor self-diagnosis features\n"
             "grey = the 8 healthy PADs · red = PADs 6, 9, 11 (degrading) · "
             f"orange = PAD 7 (borderline) · thresholds at k = {best[0]:.1f}σ "
             "calibrated on healthy sensors only",
             fontsize=12, y=0.955)
fig.text(0.055, 0.015,
         "The two features are complementary: PAD 9 leaves the shape gate at session 14 "
         "once it has collapsed far enough that peak and trough shrink together, "
         "but by then the pair feature holds it firmly. "
         "PAD 11 is never anomalous in shape — it simply goes quiet.",
         fontsize=8.2, style="italic", color="#444")
fig.subplots_adjust(bottom=0.20)
png = OUT / "phase4_health_features.png"
fig.savefig(png, dpi=145)
fs.to_csv(OUT / "phase4_flag_matrix.csv")
print(f"\nwrote {png}")

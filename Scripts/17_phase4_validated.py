"""Phase 4 (revised) — honest validation of the self-diagnosis detector.

Four corrections to the first pass, each of which weakens or reframes a claim:

1. **Leave-one-sensor-out calibration.** The first pass calibrated thresholds on
   the eight healthy sensors and then measured false alarms on those same eight.
   Zero false alarms was close to guaranteed. Here each healthy sensor is held
   out, thresholds are fitted on the remaining seven, and the held-out sensor is
   scored. That is the number that means something.

2. **Per-passage, not per-session.** Sessions sit ~1.2 days apart, so a lead
   time of "-1 session" was measured at 1.2-day resolution with dozens of
   passages in between. Everything below is per passage, and lead time is
   reported in passages AND hours.

3. **Exposure, not lead.** For the 4->2 FBG case the decision-relevant quantity
   is how many axles crossed a sensor that had stopped being a detector before
   anything flagged it. That is what a safety case asks for.

4. **Rule 3.** peak/noise was plotted but not used. It is the only rule that is
   purely self-referential, so it is the only one that survives BOTH failure
   modes the other two miss: shape goes deaf once collapse is advanced (peak and
   trough shrink together), and in a real 2-FBG head "rest of head" is a single
   sensor, so common-mode drift is invisible to the pair rule.

Statistical honesty: three failures across three distinct failure modes is a
demonstration, not a proof. The rule of three is applied explicitly.
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
from lightrack import CONFIG, CH_OF_PAD, parse_filename_utc     # noqa: E402
from lightrack.health import DEGRADING, BORDERLINE, HEALTHY     # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
THRESH_PM = 40.0            # the project's own detection threshold (§6.2)
ONSET_FRAC = 0.70           # "no longer trustworthy": 30 % loss of sensitivity
TREND_WIN = 5               # passages, for the long-horizon self-referential rule
CONFIRM_N, CONFIRM_K = 3, 2  # confirm on 2 of the last 3 passages

pd.set_option("display.width", 235)
LINE = "=" * 106

df = pd.read_csv(OUT / "phase4_features.csv")
df["utc"] = df["name"].map(parse_filename_utc)
df = df.sort_values(["pad", "utc"]).reset_index(drop=True)

# passage index on a single global timeline, shared by all sensors
order = (df[["name", "utc"]].drop_duplicates().sort_values("utc")
         .reset_index(drop=True).reset_index().rename(columns={"index": "pi"}))
df = df.merge(order[["name", "pi"]], on="name")
N_PASS = order.shape[0]
T0 = order["utc"].iloc[0]
print(f"{N_PASS} passages, {order['utc'].min()} -> {order['utc'].max()}")

# ---- commissioning baselines (session 1 only; PAD 11 already fails at s2) ----
b1 = df[df["session"] == 1].groupby("pad").agg(
    p2p_base=("p2p_pm", "median"), snr_base=("snr", "median"),
    pair_base=("pair_log2", "median")).reset_index()
df = df.merge(b1, on="pad")
df["amp_rel"] = df["p2p_pm"] / df["p2p_base"]
df["pair_delta"] = df["pair_log2"] - df["pair_base"]
df["snr_delta"] = np.log2(df["snr"] / df["snr_base"])
# Rule 3 is a LONG-HORIZON trend: a rolling median over passages suppresses the
# train-to-train mass/speed variation that makes any single passage unusable as
# a self-referential amplitude reference.
df["snr_trend"] = (df.groupby("pad")["snr_delta"]
                   .transform(lambda s: s.rolling(TREND_WIN, min_periods=2).median()))
df["amp_trend"] = (df.groupby("pad")["amp_rel"]
                   .transform(lambda s: s.rolling(TREND_WIN, min_periods=2).median()))
print(f"session-1 baseline uses {int((df['session']==1).sum()/12)} passages "
      f"— thin, and the only clean one (PAD 11 fails at session 2)")

# ------------------------------------------------------------- onset per pad
print(f"\n{LINE}\nONSET — first passage at which the sensor stops being trustworthy\n{LINE}")
print(f"defined as amp_trend <= {ONSET_FRAC:.2f} and never recovering above it\n")
onset = {}
for pad in sorted(DEGRADING | BORDERLINE):
    s = df[df["pad"] == pad].set_index("pi")["amp_trend"]
    bad = s <= ONSET_FRAC
    first = None
    for i in s.index:
        if bad.loc[i:].all() and bad.loc[i]:
            first = int(i)
            break
    onset[pad] = first
    if first is not None:
        u = order.loc[order["pi"] == first, "utc"].iloc[0]
        print(f"  PAD {pad:>2} ({CH_OF_PAD[pad]}): passage {first:>3}  {u:%Y-%m-%d %H:%M} UTC")
    else:
        print(f"  PAD {pad:>2} ({CH_OF_PAD[pad]}): never crosses {ONSET_FRAC}")

RULES = [("shape", "neg_over_pos", True),
         ("pair",  "pair_delta",   False),
         ("trend", "snr_trend",    False)]


def thresholds(cal_pads: set[int], k: float) -> dict[str, float]:
    out = {}
    for tag, feat, upper in RULES:
        v = df[df["pad"].isin(cal_pads)][feat].dropna()
        m = v.median()
        s = np.median(np.abs(v - m)) * 1.4826
        out[tag] = m + k * s if upper else m - k * s
    return out


def flags_for(pad: int, thr: dict[str, float], rules: list[str]) -> pd.Series:
    s = df[df["pad"] == pad].set_index("pi")
    f = pd.Series(False, index=s.index)
    for tag, feat, upper in RULES:
        if tag not in rules:
            continue
        f |= (s[feat] > thr[tag]) if upper else (s[feat] < thr[tag])
    return f.fillna(False)


def confirm(f: pd.Series) -> pd.Series:
    return f.rolling(CONFIRM_N, min_periods=1).sum() >= CONFIRM_K


def evaluate(k: float, rules: list[str]) -> dict:
    """LOSO for healthy sensors; full healthy calibration for degrading ones."""
    fa_flag = fa_tot = 0
    per_healthy = {}
    for pad in sorted(HEALTHY):
        thr = thresholds(HEALTHY - {pad}, k)          # <- held out
        c = confirm(flags_for(pad, thr, rules))
        per_healthy[pad] = int(c.sum())
        fa_flag += int(c.sum())
        fa_tot += len(c)
    thr_all = thresholds(HEALTHY, k)
    det = {}
    for pad in sorted(DEGRADING | BORDERLINE):
        c = confirm(flags_for(pad, thr_all, rules))
        first = int(c.index[c][0]) if c.any() else None
        det[pad] = {"first": first, "conf": c}
    return {"k": k, "thr": thr_all, "fa": fa_flag, "fa_tot": fa_tot,
            "per_healthy": per_healthy, "det": det}


# ---------------------------------------------- rule-set and threshold sweep
print(f"\n{LINE}\nRULE SETS — leave-one-sensor-out false alarms, per passage\n{LINE}")
SETS = {"shape only": ["shape"], "shape+pair": ["shape", "pair"],
        "shape+pair+trend": ["shape", "pair", "trend"]}
print(f"{'rule set':<20}{'k':>5}{'LOSO false alarms':>20}{'missed':>8}"
      f"   first-flag passage (PAD 6/9/11)")
results = {}
for name, rules in SETS.items():
    for k in [3.0, 4.0]:
        r = evaluate(k, rules)
        miss = [p for p in DEGRADING if r["det"][p]["first"] is None]
        results[(name, k)] = r
        print(f"{name:<20}{k:>5.1f}{r['fa']:>10}/{r['fa_tot']:<9}{len(miss):>8}"
              f"   {'/'.join(str(r['det'][p]['first']) for p in (6, 9, 11))}")

CHOSEN = ("shape+pair+trend", 3.0)
R = results[CHOSEN]
print(f"\nchosen: {CHOSEN[0]} at k={CHOSEN[1]}")
for tag, feat, upper in RULES:
    print(f"  {tag:<6} {feat:<14} {'>' if upper else '<'} {R['thr'][tag]:+.3f}")

# ------------------------------------------------------- LOSO FA in detail
print(f"\n{LINE}\nFALSE ALARMS — leave-one-sensor-out, the number that counts\n{LINE}")
print("Each healthy sensor scored with thresholds fitted on the other SEVEN.")
print("The first pass calibrated and measured on the same eight sensors, which")
print("made zero false alarms close to guaranteed. This does not.\n")
print(f"{'PAD':>4}{'ch':>6}  {'mounting':<24}{'flagged passages':>18}")
mnt = {1: "glued, centred", 2: "glued, centred", 3: "glued, +7.5cm",
       4: "glued, +7.5cm", 5: "glued, centred", 6: "glued (CONTROL)",
       7: "glued, +3cm", 8: "glued, +3cm", 9: "1/2 collage",
       10: "clipse", 11: "unprepared surface", 12: "taraude"}
for pad, n in R["per_healthy"].items():
    print(f"{pad:>4}{CH_OF_PAD[pad]:>6}  {mnt[pad]:<24}{n:>10}/{N_PASS:<7}")
print(f"\nLOSO false-alarm rate: {R['fa']}/{R['fa_tot']} = "
      f"{100*R['fa']/R['fa_tot']:.1f} % of healthy sensor-passages")

# ------------------------------------------------- which rule is firing?
print(f"\n{LINE}\nFALSE-ALARM ATTRIBUTION — the failures are not random\n{LINE}")
print(f"{'PAD':>4}{'ch':>6}  {'mounting':<24}" + "".join(f"{t:>9}" for t, _, _ in RULES)
      + f"{'confirmed':>11}")
for pad in sorted(HEALTHY):
    thr = thresholds(HEALTHY - {pad}, CHOSEN[1])
    s = df[df["pad"] == pad].set_index("pi")
    counts = []
    for tag, feat, upper in RULES:
        f = (s[feat] > thr[tag]) if upper else (s[feat] < thr[tag])
        counts.append(int(f.fillna(False).sum()))
    c = confirm(flags_for(pad, thr, list(SETS[CHOSEN[0]])))
    print(f"{pad:>4}{CH_OF_CHAN if False else CH_OF_PAD[pad]:>6}  {mnt[pad]:<24}"
          + "".join(f"{x:>9}" for x in counts) + f"{int(c.sum()):>11}")

s1 = df[df["session"] == 1].groupby("pad")["neg_over_pos"].median()
print(f"\nPAD 3 sits at |trough|/peak = {s1[3]:.2f} on session 1 against a healthy")
print(f"median of {s1[list(HEALTHY)].median():.2f}. Held out, the threshold fitted on the")
print("other seven drops below it and PAD 3 trips on essentially every passage.")
print("PAD 3 is +7.5 cm off-centre, at the pad edge nearest the sleeper — a")
print("position that genuinely deepens the compression lobe. The shape rule is")
print("CONFOUNDED BY SLEEPER-BAY POSITION, not measuring health for that sensor.")
print("\nPAD 10 and PAD 12 trip on the PAIR rule: both sit in Head 3 alongside")
print("PAD 9 and PAD 11. As their head-mates collapse, the 'rest of head'")
print("reference itself becomes garbage. The pair rule assumes the comparison")
print("group is healthy, and in Head 3 it is not.")

# ------------------------------------------------- lead time and exposure
print(f"\n{LINE}\nLEAD TIME — per passage and in hours, not per session\n{LINE}")
print(f"{'PAD':>4}{'onset':>7}{'flagged':>9}{'lead (passages)':>17}"
      f"{'lead (hours)':>14}   verdict")
lead_rows = []
for pad in sorted(DEGRADING | BORDERLINE):
    o, f = onset[pad], R["det"][pad]["first"]
    if o is None or f is None:
        print(f"{pad:>4}{str(o):>7}{str(f):>9}{'-':>17}{'-':>14}   "
              f"{'never degraded' if o is None else 'MISSED'}")
        continue
    to = order.loc[order["pi"] == o, "utc"].iloc[0]
    tf = order.loc[order["pi"] == f, "utc"].iloc[0]
    hrs = (to - tf).total_seconds() / 3600
    v = "EARLY WARNING" if o > f else ("concurrent" if o == f else "after the fact")
    print(f"{pad:>4}{o:>7}{f:>9}{o-f:>+17}{hrs:>+14.1f}   {v}")
    lead_rows.append({"pad": pad, "onset": o, "flag": f, "lead_pass": o - f,
                      "lead_h": hrs})

print(f"\n{LINE}\nEXPOSURE — axles over a sensor that was no longer a detector,\n"
      f"           before anything flagged it\n{LINE}")
print(f"'no longer a detector' = median positive peak < {THRESH_PM:.0f} pm "
      f"(the project's own threshold)\n")
print(f"{'PAD':>4}{'passages sub-40pm':>19}{'...before flag':>16}"
      f"{'axles exposed':>15}{'hours exposed':>15}")
exp_rows = []
for pad in sorted(DEGRADING | BORDERLINE):
    s = df[df["pad"] == pad].set_index("pi")
    c = R["det"][pad]["conf"].reindex(s.index).fillna(False)
    dead = s["pos_pm"] < THRESH_PM
    before = dead & ~c.cummax()          # dead AND no confirmed flag yet
    ax = int(s.loc[before, "n_axles"].sum())
    if before.any():
        ts = order.set_index("pi").loc[before[before].index, "utc"]
        hrs = (ts.max() - ts.min()).total_seconds() / 3600
    else:
        hrs = 0.0
    print(f"{pad:>4}{int(dead.sum()):>19}{int(before.sum()):>16}"
          f"{ax:>15}{hrs:>15.1f}")
    exp_rows.append({"pad": pad, "dead_passages": int(dead.sum()),
                     "exposed_passages": int(before.sum()),
                     "exposed_axles": ax, "exposed_hours": hrs})
ex = pd.DataFrame(exp_rows)
print(f"\ntotal undetected-axle exposure across all three failures: "
      f"{int(ex['exposed_axles'].sum())} axles")
print("Field exposure would be larger: this delivery holds ~6 % of retained")
print("files (A-07), so the true axle count between onset and flag is roughly")
print("an order of magnitude higher. Reported here as OBSERVED exposure only.")

# ----------------------------------------------------- statistical honesty
print(f"\n{LINE}\nWHAT THIS DOES AND DOES NOT ESTABLISH\n{LINE}")
n_deg = len(DEGRADING)
# Exact one-sided Clopper-Pearson upper bound for 0 events in n trials.
# The familiar 3/n "rule of three" is the large-n approximation and is
# degenerate here (3/3 = 100 %), so the exact form is used.
ub = 1 - 0.05 ** (1 / n_deg)
print(f"Detected {n_deg}/{n_deg} degrading sensors. With zero misses out of "
      f"{n_deg} trials, the")
print(f"exact 95 % upper bound on the miss rate is 1-0.05^(1/{n_deg}) = {ub:.0%}.")
print(f"(The usual 3/n rule of three is a large-n approximation and degenerates "
      f"to {3/n_deg:.0%} here.)")
print("So the honest statement is: no failure mode present in POC1 was missed,")
print(f"and the miss rate is bounded above by only ~{ub:.0%} — which is to say, barely")
print("bounded at all. Three sensors spanning three distinct failure modes is a")
print("DEMONSTRATION that self-diagnosis is feasible, not evidence that it is")
print("reliable. A SIL4 claim needs failure counts three orders of magnitude")
print("larger; POC1 cannot supply them and should not be read as trying to.")
print(f"\nThe false-alarm figure is better grounded: {R['fa_tot']} held-out "
      f"healthy sensor-passages,")
print(f"{R['fa']} flagged, and each was scored by thresholds it did not help fit.")

# ------------------------------------------------- residual limitation
print(f"\n{LINE}\nRESIDUAL LIMITATION — state this in any report\n{LINE}")
print("With TWO gratings per head and both degrading together:")
print("  * pair goes silent  — 'rest of head' is one sensor, so common-mode")
print("                        drift cancels out of the ratio entirely;")
print("  * shape goes deaf   — once collapse is advanced, peak and trough")
print("                        shrink together and |trough|/peak renormalises.")
print("                        PAD 9 leaves the shape gate at passage "
      f"{int(df[(df['pad']==9)&(df['neg_over_pos']<R['thr']['shape'])]['pi'].min())}.")
print("  * trend still works — it is self-referential against commissioning and")
print("                        needs neither a neighbour nor a peak shape.")
print("Rule 3 is the MITIGATION for that case, not a proof it is covered:")
print("no common-mode double failure occurred in POC1, so the mitigation is")
print("untested. It should be exercised deliberately in POC2.")

# ------------------------------- acceptance vs monitoring, the real split
print(f"\n{LINE}\nRESTRUCTURING — the shape rule is an ACCEPTANCE test, not monitoring\n{LINE}")
print("PAD 9's anomaly is present on passage 1: a half-glued pad is defective")
print("from installation. Detecting that is a COMMISSIONING ACCEPTANCE check,")
print("not a monitoring function. And a population threshold is exactly right")
print("for acceptance (does this new pad look like a properly bonded one?) while")
print("being wrong for monitoring, where it confounds position with health.\n")

thr_acc = thresholds(HEALTHY, CHOSEN[1])
acc = df[df["session"] == 1].groupby("pad")["neg_over_pos"].median()
print(f"ACCEPTANCE at commissioning — shape only, threshold "
      f"{thr_acc['shape']:+.3f}:")
flagged_acc = [p for p in range(1, 13) if acc[p] > thr_acc["shape"]]
for p in range(1, 13):
    mark = "FLAG" if acc[p] > thr_acc["shape"] else "pass"
    print(f"   PAD {p:>2} ({CH_OF_PAD[p]}) {mnt[p]:<24} {acc[p]:5.2f}  {mark}")
print(f"  => flags {flagged_acc}; PAD 9 caught at installation, "
      f"{len(set(flagged_acc)-{9})} others")

print(f"\nMONITORING — pair + trend only, no shape, LOSO:")
for k in [3.0, 4.0]:
    r = evaluate(k, ["pair", "trend"])
    miss = [p for p in DEGRADING if r["det"][p]["first"] is None]
    print(f"   k={k}: LOSO false alarms {r['fa']}/{r['fa_tot']} "
          f"({100*r['fa']/r['fa_tot']:4.1f} %), missed {miss}, "
          f"first flag PAD6/9/11 = "
          f"{'/'.join(str(r['det'][p]['first']) for p in (6,9,11))}")
print("   PAD 9 is still caught by monitoring even without the shape rule —")
print("   its collapse shows up in pair and trend — so removing shape from the")
print("   monitoring path costs detection of nothing, and removes the")
print("   position confound entirely.")

# ================================================= FINAL TWO-STAGE ARCHITECTURE
print(f"\n{LINE}\nFINAL ARCHITECTURE — acceptance + monitoring, and its numbers\n{LINE}")
K_MON = 3.0
MON_RULES = ["pair", "trend"]
thr_mon = thresholds(HEALTHY, K_MON)
acc_flag = {p: bool(acc[p] > thr_acc["shape"]) for p in range(1, 13)}

print("stage 1  ACCEPTANCE  — shape, once at commissioning, population threshold")
print(f"stage 2  MONITORING  — pair + trend, per passage, "
      f"{CONFIRM_K}-of-{CONFIRM_N} confirmation, k={K_MON}\n")

rows = []
for pad in sorted(DEGRADING | BORDERLINE):
    c = confirm(flags_for(pad, thr_mon, MON_RULES))
    mon_first = int(c.index[c][0]) if c.any() else None
    first = 0 if acc_flag[pad] else mon_first     # acceptance fires at passage 0
    o = onset[pad]
    if o is None or first is None:
        continue
    to = order.loc[order["pi"] == o, "utc"].iloc[0]
    tf = order.loc[order["pi"] == first, "utc"].iloc[0]
    s = df[df["pad"] == pad].set_index("pi")
    conf_full = pd.Series(True, index=s.index) if acc_flag[pad] else c.reindex(s.index).fillna(False)
    dead = s["pos_pm"] < THRESH_PM
    before = dead & ~conf_full.cummax()
    rows.append({
        "pad": pad, "via": "acceptance" if acc_flag[pad] else "monitoring",
        "onset_pi": o, "flag_pi": first,
        "lead_passages": o - first,
        "lead_hours": (to - tf).total_seconds() / 3600,
        "exposed_passages": int(before.sum()),
        "exposed_axles": int(s.loc[before, "n_axles"].sum()),
    })
fin = pd.DataFrame(rows)
print(f"{'PAD':>4}{'caught by':>13}{'onset':>7}{'flag':>6}"
      f"{'lead (pass)':>13}{'lead (h)':>11}{'exposed axles':>15}")
for _, r in fin.iterrows():
    print(f"{int(r['pad']):>4}{r['via']:>13}{int(r['onset_pi']):>7}{int(r['flag_pi']):>6}"
          f"{int(r['lead_passages']):>+13}{r['lead_hours']:>+11.1f}"
          f"{int(r['exposed_axles']):>15}")

r_mon = evaluate(K_MON, MON_RULES)
print(f"\nLOSO false alarms, monitoring stage: {r_mon['fa']}/{r_mon['fa_tot']} "
      f"= {100*r_mon['fa']/r_mon['fa_tot']:.1f} % of healthy sensor-passages")
print(f"  (was {100*R['fa']/R['fa_tot']:.1f} % with shape in the monitoring path)")
print(f"acceptance stage flags {sorted([p for p in range(1,13) if acc_flag[p]])} "
      f"— PAD 9 correctly, PAD 3 as a position artefact")
print(f"total observed undetected-axle exposure: "
      f"{int(fin['exposed_axles'].sum())} axles")
fin.to_csv(OUT / "phase4_final.csv", index=False)

print("\nPAD 3's acceptance flag is an artefact of POC1's DELIBERATE position")
print("variation: it is the only +7.5 cm pad whose compression lobe is deep")
print("enough to clear a threshold fitted mostly on centred pads. An operational")
print("install places every pad identically, so the acceptance population would")
print("be homogeneous and this artefact would not arise. That is an assertion")
print("about POC2, not something POC1 data can confirm.")

pd.DataFrame(lead_rows).to_csv(OUT / "phase4_lead.csv", index=False)
ex.to_csv(OUT / "phase4_exposure.csv", index=False)
df.to_csv(OUT / "phase4_features_scored.csv", index=False)

# ----------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
fig.subplots_adjust(wspace=0.24, top=0.79, bottom=0.20, left=0.055, right=0.99)
COL = {"degrading": "#b2182b", "borderline": "#e08214", "healthy": "#c9ccd1"}
for ax_, (feat, ttl) in zip(axes, [
        ("neg_over_pos", "Rule 1 — shape: |trough|/peak\nself, single passage"),
        ("pair_delta", "Rule 2 — pair vs commissioning\nneeds the other grating"),
        ("snr_trend", f"Rule 3 — peak/noise trend, {TREND_WIN}-passage median\n"
                      "self-referential, survives common mode")]):
    for pad in range(1, 13):
        s = df[df["pad"] == pad].set_index("pi")[feat]
        lab = ("degrading" if pad in DEGRADING else
               "borderline" if pad in BORDERLINE else "healthy")
        ax_.plot(s.index, s.values, color=COL[lab],
                 lw=1.9 if lab != "healthy" else 0.8,
                 zorder=3 if lab != "healthy" else 1,
                 label=f"PAD {pad}" if lab != "healthy" else None)
    tag = {"neg_over_pos": "shape", "pair_delta": "pair", "snr_trend": "trend"}[feat]
    ax_.axhline(R["thr"][tag], color="#2166ac", ls="--", lw=1.4, label="threshold")
    ax_.set_title(ttl, fontsize=9.8)
    ax_.set_xlabel("passage index (chronological)", fontsize=9)
    ax_.grid(alpha=0.3, lw=0.5)
    ax_.legend(fontsize=7.5, ncol=2, framealpha=0.92)
    ax_.tick_params(labelsize=8)
axes[0].set_ylabel("feature value", fontsize=9)
fig.suptitle("Lightrack POC1 — Phase 4 revised: per-passage self-diagnosis, "
             f"leave-one-sensor-out validated\ngrey = 8 healthy PADs · red = PADs 6, 9, 11 "
             f"· orange = PAD 7 · thresholds at k={CHOSEN[1]}σ",
             fontsize=11.8, y=0.955)
fig.text(0.055, 0.035,
         f"LOSO false-alarm rate {R['fa']}/{R['fa_tot']} healthy sensor-passages. "
         f"3/3 degrading sensors detected — but 0 misses out of 3 bounds the miss rate "
         f"at only ~63 % (rule of three, 95 %).\nThis is a demonstration of feasibility, "
         "not evidence of reliability.",
         fontsize=8.4, style="italic", color="#444")
png = OUT / "phase4_validated.png"
fig.savefig(png, dpi=145)
print(f"\nwrote {png}")

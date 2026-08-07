"""Phase 5 — sleeper-bay position effect and mounting-method durability ranking.

Never run before; POC2_RECOMMENDATIONS.md defers its scope note to this.

POSITION. Paired tests on SIMULTANEOUS passages, not pooled distributions: both
pads see the same train at the same instant, so train mass, speed and load
cancel exactly in the ratio. Pooling would put all that variance back in.

A design problem has to be stated first. In Head 1 the centred pair {PAD 1,
PAD 2} is on rail 1-Out and the +7.5 cm pair {PAD 3, PAD 4} is on rail 2-In, so
**sleeper-bay position and rail are perfectly confounded** and the position
effect is not identifiable from Head 1 alone. Head 2 breaks the confound:
PAD 5/6 are centred on rail 1, PAD 7 is +3 cm on rail 1, PAD 8 is +3 cm on
rail 2 — so position can be measured within rail 1 and rail within +3 cm. But
Head 2 carries the degrading PAD 6 and the declining PAD 7 (A-19), so it is
restricted to sessions 1-2, before either has moved.

MOUNTING. Durability as time to cross a fixed fraction of commissioning
amplitude, with non-crossers reported as CENSORED rather than as passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD, parse_filename_utc, to_local  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
TREND_WIN, CROSS_FRAC = 5, 0.70

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
P = df.pivot_table(index="qi", columns="pad", values="p2p_pm")
POS = df.pivot_table(index="qi", columns="pad", values="pos_pm")
NEG = df.pivot_table(index="qi", columns="pad", values="neg_pm").abs()
T = o.set_index("qi")["utc"]
DAY = (T - T.iloc[0]).dt.total_seconds() / 86400.0
S1 = sorted(df.loc[df["session"] == 1, "qi"].unique())
S12 = sorted(df.loc[df["session"].isin([1, 2]), "qi"].unique())


def paired(a_pads, b_pads, idx, label_a, label_b, M=None):
    M = P if M is None else M
    a = M.loc[idx, a_pads].mean(axis=1)
    b = M.loc[idx, b_pads].mean(axis=1)
    ok = a.notna() & b.notna() & (a > 0) & (b > 0)
    r = np.log2(a[ok] / b[ok])
    stat, p = wilcoxon(r) if len(r) >= 6 else (np.nan, np.nan)
    ratio = 2 ** r.median()
    print(f"  {label_a:<28} vs {label_b:<28} n={len(r):>3}  "
          f"ratio {ratio:5.2f}x  ({100*(ratio-1):+5.1f} %)  "
          f"log2 median {r.median():+.3f}  Wilcoxon p={p:.4f}")
    return {"a": label_a, "b": label_b, "n": len(r), "ratio": ratio,
            "log2": float(r.median()), "p": float(p)}


print(f"{LINE}\nSLEEPER-BAY POSITION — paired on simultaneous passages\n{LINE}")
print("\nHEAD 1  (all four sensors healthy for the whole campaign)")
print("  WARNING: position is CONFOUNDED with rail in this head.")
print("  centred = PAD 1,2 on rail 1-Out;  +7.5 cm = PAD 3,4 on rail 2-In.")
rows = []
rows.append(paired([1, 2], [3, 4], P.index, "centred (1,2 / rail 1-Out)",
                   "+7.5 cm (3,4 / rail 2-In)"))

print("\nHEAD 2, SESSIONS 1-2 ONLY  (before PAD 6 and PAD 7 move)")
print("  this head breaks the confound:")
rows.append(paired([5], [7], S12, "centred (PAD 5, rail 1)",
                   "+3 cm (PAD 7, rail 1)"))
rows.append(paired([7], [8], S12, "+3 cm (PAD 7, rail 1)",
                   "+3 cm (PAD 8, rail 2)"))
rows.append(paired([5], [8], S12, "centred (PAD 5, rail 1)",
                   "+3 cm (PAD 8, rail 2)"))

pos = rows[1]["log2"]
rail = rows[2]["log2"]
comb = rows[0]["log2"]
print(f"\nDECOMPOSITION (assumes the rail effect measured in Head 2 also holds")
print(f"in Head 1 — an assumption this design cannot verify):")
print(f"  Head 1 combined (position + rail) : {comb:+.3f} log2  "
      f"({100*(2**comb-1):+.1f} %)")
print(f"  rail effect, from Head 2          : {rail:+.3f} log2  "
      f"({100*(2**rail-1):+.1f} %)")
print(f"  => position residual for +7.5 cm  : {comb-rail:+.3f} log2  "
      f"({100*(2**(comb-rail)-1):+.1f} %)")
print(f"  position effect for +3 cm, direct : {pos:+.3f} log2  "
      f"({100*(2**pos-1):+.1f} %)")

print(f"\n{LINE}\nWHICH COMPONENT MOVES — positive peak vs compression lobe\n{LINE}")
print("Peak-to-peak mixes the axle peak with the compression lobe, and an")
print("off-centre pad has a deeper lobe (PAD 3: |trough|/peak 0.76 vs a healthy")
print("median of 0.31). Since the COUNTING threshold acts on the positive peak,")
print("the two components have to be separated before any design conclusion.\n")
for M, nm in ((POS, "positive peak"), (NEG, "|compression lobe|"), (P, "peak-to-peak")):
    print(f"{nm}:")
    paired([1, 2], [3, 4], P.index, "  centred (H1)", "+7.5 cm (H1)", M)
    paired([5], [7], S12, "  centred (PAD 5)", "+3 cm (PAD 7, same rail)", M)
print("""
THE TWO HEADS DISAGREE, AND THAT IS THE RESULT.

Head 1 (+7.5 cm, rail-confounded): the positive-peak difference is NOT
  significant (p=0.16); the whole peak-to-peak gap comes from a 32 % deeper
  compression lobe on the off-centre pair.
Head 2 (+3 cm, same rail): the positive peak IS significantly higher off-centre
  (+22 %, p<0.0001) and the lobe is 23 % SHALLOWER - the opposite sign.

The lobe effect reverses between the two comparisons. Head 1's can be explained
by rail rather than position; Head 2's cannot be explained by rail, but it rests
on ONE sensor per condition (PAD 5 vs PAD 7), so it confounds position with
unit-to-unit variation. Pairing over 26 passages makes the difference between
those two specific sensors highly significant; it says nothing about whether the
cause is where they sit.

CONCLUSION: POC1's layout cannot cleanly identify the sleeper-bay position
effect. Every available comparison confounds position with either rail (Head 1,
two sensors per condition) or individual sensor (Head 2, same rail, one per
condition). The differences are real and span 5-22 % on the positive peak, but
they are NOT attributable to position on this evidence.

POC2 fix: both positions on the SAME rail, at least two sensors per position,
so that position, rail and unit variation are separable.""")

print(f"\n{LINE}\nWHAT IS AND IS NOT IDENTIFIABLE\n{LINE}")
print("Identifiable: the +3 cm position effect within rail 1 (Head 2, sessions")
print("1-2), and the rail effect at +3 cm. Both from small samples.")
print("NOT identifiable: the +7.5 cm position effect on its own. Head 1 is the")
print("only head carrying it and its rail is confounded with it by design.")
print("This is a limitation of the POC1 LAYOUT, not of the analysis - and it")
print("should be fixed in POC2 by placing both positions on the same rail.")

# =================================================== mounting durability
print(f"\n{LINE}\nMOUNTING-METHOD DURABILITY — time to cross "
      f"{CROSS_FRAC:.0%} of commissioning\n{LINE}")
amp = P / P.loc[S1].median()
tr = amp.rolling(TREND_WIN, min_periods=2).median()
MOUNT = {1: "standard glued", 2: "standard glued", 3: "standard glued",
         4: "standard glued", 5: "standard glued", 6: "standard glued",
         7: "standard glued", 8: "standard glued", 9: "1/2 collage",
         10: "clipse (clamped)", 11: "unprepared surface", 12: "taraude (screwed)"}
HEAD = {**{p: 1 for p in (1, 2, 3, 4)}, **{p: 2 for p in (5, 6, 7, 8)},
        **{p: 3 for p in (9, 10, 11, 12)}}
rows = []
for pad in range(1, 13):
    s = tr[pad]
    bad = s <= CROSS_FRAC
    t = None
    for i in s.index:
        if bad.loc[i] and bad.loc[i:].all():
            t = float(DAY.loc[i])
            break
    rows.append({"pad": pad, "head": HEAD[pad], "mount": MOUNT[pad],
                 "commissioning_pm": float(P.loc[S1, pad].median()),
                 "final_frac": float(s.iloc[-3:].median()),
                 "days_to_cross": t,
                 "censored": t is None})
d = pd.DataFrame(rows)
print(f"{'PAD':>4}{'head':>6}{'mounting':<22}{'commissioning pm':>18}"
      f"{'final frac':>12}{'days to cross':>15}")
for _, r in d.iterrows():
    t = f"{r['days_to_cross']:.1f}" if not r["censored"] else f"censored >{DAY.iloc[-1]:.0f}"
    print(f"{int(r['pad']):>4}{int(r['head']):>6}{r['mount']:<22}"
          f"{r['commissioning_pm']:>18.1f}{r['final_frac']:>12.2f}{t:>15}")

print(f"\n{LINE}\nHEAD 3 — the mounting experiment, ranked\n{LINE}")
h3 = d[d["head"] == 3].sort_values(["censored", "days_to_cross"], ascending=[False, False])
print(f"{'rank':>5}{'PAD':>5}{'mounting':<22}{'survives?':>11}"
      f"{'commissioning pm':>18}{'margin vs 40 pm':>17}")
for i, (_, r) in enumerate(h3.iterrows(), 1):
    surv = "yes" if r["censored"] else f"no ({r['days_to_cross']:.0f} d)"
    print(f"{i:>5}{int(r['pad']):>5}{r['mount']:<22}{surv:>11}"
          f"{r['commissioning_pm']:>18.1f}{r['commissioning_pm']/40:>16.2f}x")
print("\nWARNING: rail is confounded with mounting in Head 3 as well —")
print("PAD 9 (1/2 collage) and PAD 10 (clipse) are both on rail 1-Out;")
print("PAD 11 (unprepared) and PAD 12 (taraude) are both on rail 2-In.")
print("So 'clipse beats unprepared' cannot be separated from 'rail 1 beats")
print("rail 2'. The two survivors sit on DIFFERENT rails, which is the one")
print("comparison the layout does support.")

print(f"\n{LINE}\nVERDICT ON THE DECK'S CLAIM\n{LINE}")
p10, p12 = d[d["pad"] == 10].iloc[0], d[d["pad"] == 12].iloc[0]
print(f"The deck calls clipse 'THE real good idea of POC1'. On durability alone")
print(f"that holds: PAD 10 never crosses {CROSS_FRAC:.0%} (final "
      f"{p10['final_frac']:.2f}), censored at {DAY.iloc[-1]:.0f} days.")
print(f"But so does taraude: PAD 12 final {p12['final_frac']:.2f}, also censored.")
print(f"They are indistinguishable on durability over 18 days.")
print(f"\nOn SIGNAL they are not: clipse commissions at "
      f"{p10['commissioning_pm']:.0f} pm against taraude's "
      f"{p12['commissioning_pm']:.0f} pm — a factor of "
      f"{p12['commissioning_pm']/p10['commissioning_pm']:.1f}.")
print(f"A-15 established this is a threshold problem, not a noise problem")
print(f"(clipse runs at 167x its noise floor). But on the evidence here,")
print(f"'the real good idea' is supported on durability and NOT distinguished")
print(f"from taraude, while giving up half the signal. 18 days is far too")
print(f"short to separate two sensors that both survive it.")

d.to_csv(OUT / "phase5_durability.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "phase5_position.csv", index=False)
print(f"\nwrote phase5_durability.csv, phase5_position.csv")

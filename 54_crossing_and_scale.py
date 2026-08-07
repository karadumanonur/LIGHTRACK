"""Two checks that the previous round left open.

ITEM 1 — CROSSING. The two confirmed-reverse records are also the only matched
passenger passages carrying inter-axle gaps under 2.0 m. Two trains crossing at
the site would produce exactly that: interleaved axle sequences whose apparent
gaps belong to neither train. The 7-window multi-train list was built from
timetabled trains falling in the SAME record; this asks the sharper question -
does the timetable put any train on the opposite track within a few seconds of
each detected passage, whether or not it lands in the same file.

ITEM 2 — SCALE. Every spatial quantity here is v * dt, and v comes from a solver
that IMPOSES 15 m / 10 m head spacings. The ratio was tested (A-21); the absolute
scale never was, because the constrained solver cannot see it. The through-train
speed comparison (A-52) is the first measurement that can, and it says the sensor
reads low - which means every distance in this project is short by the same
factor. A-41's mechanism rests on the freight short-gap mode being BELOW the
2.2 m influence extent, so the margin has to be re-examined at the corrected
scale rather than assumed to survive.

NOT AN INDEPENDENT ANCHOR. `speed_crosscheck.csv`'s `v_bogie` looks like a second
absolute scale but is not: WHEELBASE_M = 2.84 m in 33_classify_15.py is itself the
class-0 template median, computed as v_tof * dt. Feeding it back would confirm the
scale it was derived from. It is excluded here for that reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG                                   # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
EXTENT_M = CONFIG["geometry"].get("axle_influence_extent_m", 2.2)
FOCUS = ("20260523T121014,087Z.txt", "20260525T120238,857Z.txt")
pd.set_option("display.width", 250)
L = "=" * 116

d = pd.read_csv(OUT / "passage_times.csv")
d["train_local"] = pd.to_datetime(d["train_local"]).dt.tz_localize(None)
d["train_end_local"] = pd.to_datetime(d["train_end_local"]).dt.tz_localize(None)
t = pd.read_csv(OUT / "timetable_expected.csv", low_memory=False)
t["site_kin"] = pd.to_datetime(t["site_kin"])
m = pd.read_csv(OUT / "timetable_matches.csv")

# =============================================================== ITEM 1
print(f"{L}\nITEM 1 — WAS A TRAIN CROSSING?\n{L}")
print(f"""Each detected passage is compared against every timetabled train, not
only those landing in the same record. `own` is the train the passage matched;
`other` is the nearest train travelling the OTHER way. The passage occupies
train_local .. train_end_local, so overlap is what matters, not centre distance.""")

mt = m.set_index(["name", "sub"])


def nearest_other(row):
    """Nearest timetabled train running opposite to this passage's own match."""
    key = (row["name"], row["sub"])
    own_dir = mt.loc[key, "toward_charleroi"] if key in mt.index else None
    if own_dir is None:
        return pd.Series({"other_dt": np.nan, "other_rel": None,
                          "other_dir": None, "own_dir": None})
    o = t[t["toward_charleroi"] != own_dir]
    dt = (o["site_kin"] - row["train_local"]).dt.total_seconds()
    if dt.empty:
        return pd.Series({"other_dt": np.nan, "other_rel": None,
                          "other_dir": None, "own_dir": own_dir})
    i = dt.abs().idxmin()
    return pd.Series({"other_dt": dt.loc[i], "other_rel": o.loc[i, "RELATION"],
                      "other_dir": o.loc[i, "toward_charleroi"],
                      "own_dir": own_dir})


mm = d[d.set_index(["name", "sub"]).index.isin(mt.index)].copy()
mm = pd.concat([mm, mm.apply(nearest_other, axis=1)], axis=1)
mm["dur_s"] = (mm["train_end_local"] - mm["train_local"]).dt.total_seconds()
# an opposite-direction train overlaps if it arrives while this one is passing,
# with a margin for the transit-model error measured at the knee (A-50)
mm["overlap"] = mm["other_dt"].abs() <= (mm["dur_s"] + 20.0)

print(f"\nmatched passages: {len(mm)}")
print(f"passage duration: median {mm['dur_s'].median():.1f} s, "
      f"max {mm['dur_s'].max():.1f} s")
print(f"with an opposite-direction train inside "
      f"(duration + 20 s): {int(mm['overlap'].sum())}")

print(f"\n{'record':<28}{'axles':>6}{'gapmin':>8}{'<2.5':>6}{'dur s':>7}"
      f"{'own':>12}{'nearest opposite dt':>21}{'rel':>10}{'overlap':>9}")
show = mm.sort_values("gap_min_m")
for _, r in show.head(8).iterrows():
    print(f"{r['name'][:26]:<28}{int(r['n_axles']):>6}{r['gap_min_m']:>8.2f}"
          f"{int(r['n_gap_lt_2m5']):>6}{r['dur_s']:>7.1f}"
          f"{('->Charleroi' if r['own_dir'] else '->Namur'):>12}"
          f"{r['other_dt']:>+20.1f}s{str(r['other_rel']):>10}"
          f"{str(r['overlap']):>9}")

print(f"\n{L}\nTHE TWO IN QUESTION\n{L}")
for nm in FOCUS:
    r = mm[mm["name"] == nm]
    if r.empty:
        print(f"{nm}: not in the matched set")
        continue
    r = r.iloc[0]
    print(f"\n{nm}")
    print(f"   detected {r['train_local']:%Y-%m-%d %H:%M:%S} .. "
          f"{r['train_end_local']:%H:%M:%S}  ({r['dur_s']:.1f} s), "
          f"{int(r['n_axles'])} axles, v {r['v_kmh']:.1f} km/h")
    print(f"   own match: {('->Charleroi' if r['own_dir'] else '->Namur')}")
    lo = r["train_local"] - pd.Timedelta(seconds=180)
    hi = r["train_end_local"] + pd.Timedelta(seconds=180)
    nb = t[(t["site_kin"] >= lo) & (t["site_kin"] <= hi)].sort_values("site_kin")
    print(f"   every timetabled train within +-180 s: {len(nb)}")
    for _, q in nb.iterrows():
        dt = (q["site_kin"] - r["train_local"]).total_seconds()
        print(f"      {q['site_kin']:%H:%M:%S}  {dt:+7.1f} s  "
              f"{q['RELATION']:<10} "
              f"{'->Charleroi' if q['toward_charleroi'] else '->Namur':<12} "
              f"{'<== the match' if abs(dt) < 30 and q['toward_charleroi'] == r['own_dir'] else ''}")

print(f"""
VERDICT RULE, stated before reading the answer. If an opposite-direction train is
placed inside the detected passage (or within the transit model's own error of
it), interleaving explains the short gaps and these two become a worked example
of the one-train-per-record limitation. If the nearest opposite-direction train
is minutes away, the crossing hypothesis is excluded and the anomaly stays open.""")

# =============================================================== ITEM 2
print(f"\n{L}\nITEM 2 — SPEED SCALE, AND WHETHER A-41 SURVIVES IT\n{L}")
mmatch = m.copy()
th = mmatch[mmatch["family"] == "IC through"].copy()
th["v_leg"] = 2.726 / (th["leg_s"] / 3600)
th["ratio"] = th["v_kmh"] / th["v_leg"]
r_med = th["ratio"].median()
boot = np.random.default_rng(0).choice(th["ratio"].to_numpy(),
                                       (4000, len(th)), replace=True)
lo_r, hi_r = np.percentile(np.median(boot, axis=1), [2.5, 97.5])
print(f"through-train records: {len(th)}")
print(f"sensor / station-to-station speed ratio: median {r_med:.4f} "
      f"[95 % bootstrap {lo_r:.4f}-{hi_r:.4f}]")
print(f"=> implied SCALE CORRECTION on every distance: "
      f"x{1/r_med:.4f}  [{1/hi_r:.4f}-{1/lo_r:.4f}]")
print(f"=> implied true head spacings if the reference is right: "
      f"{15/r_med:.2f} m / {10/r_med:.2f} m  (nominal 15 / 10)")

gp = pd.read_csv(OUT / "passage_gaps.csv")
gp = gp.merge(d[["name", "sub", "n_axles", "n_gap_lt_2m5"]], on=["name", "sub"])
gp["freight"] = gp["n_gap_lt_2m5"] > 22            # the empty-interval split, A-51


# A-41's statistic, reproduced EXACTLY: 37_long_consists_and_filter.py defines
# "short-gap mode" as median(gaps < 5 m). Using a histogram mode instead would
# change the number and make the sensitivity incomparable with the entry it is
# meant to test, so the original definition is kept and a histogram mode is
# reported beside it rather than in place of it.
def short_mode(x: np.ndarray) -> float:
    x = x[x < 5.0]
    return float(np.median(x)) if x.size else np.nan


def hist_mode(x: np.ndarray, lo: float, hi: float, bw: float = 0.05) -> float:
    x = x[(x >= lo) & (x <= hi)]
    if x.size < 5:
        return np.nan
    edges = np.arange(lo, hi + bw, bw)
    h, _ = np.histogram(x, bins=edges)
    return float(edges[int(np.argmax(h))] + bw / 2)


# A-41's own figures, from 37_long_consists_and_filter.py. They come from the
# SAFETY chain's X-crossing detector on head 1, not from the diagnostic
# `axle_peaks` used everywhere in this script - which is correct for A-41, since
# the mechanism it documents is a safety-chain counting failure. Re-deriving them
# from a different detector would test a different claim, so A-41's values are
# scaled as recorded and the diagnostic-chain population is reported beside them.
A41 = {"freight_mode": 2.01, "freight_p5": 1.75, "freight_min": 1.68,
       "passenger_mode": 2.90, "passenger_p5": 2.63, "passenger_min": 2.29}
SCALES = [("nominal 15 / 10", 1.0),
          ("+8 % (16.2 / 10.8)", 1.08),
          (f"implied x{1/r_med:.3f}", 1 / r_med),
          (f"implied low x{1/hi_r:.3f}", 1 / hi_r),
          (f"implied high x{1/lo_r:.3f}", 1 / lo_r)]

print(f"\nA-41's short-gap figures under each scale "
      f"(axle influence extent = {EXTENT_M} m)")
print(f"{'scale':<24}{'freight mode':>13}{'freight p5':>12}"
      f"{'passenger mode':>16}{'margin':>9}{'A-41':>10}")
rows = []
for lab, s in SCALES:
    fm = A41["freight_mode"] * s
    marg = 100 * (EXTENT_M - fm) / EXTENT_M
    verdict = "holds" if fm < EXTENT_M else "FAILS"
    print(f"{lab:<24}{fm:>13.3f}{A41['freight_p5']*s:>12.3f}"
          f"{A41['passenger_mode']*s:>16.3f}{marg:>8.1f}%{verdict:>10}")
    rows.append(dict(label=lab, scale=s, freight_short_mode=fm,
                     passenger_short_mode=A41["passenger_mode"] * s,
                     margin_pct=marg, verdict=verdict))

s_cross = EXTENT_M / A41["freight_mode"]
print(f"\nthe freight short-gap mode reaches {EXTENT_M} m at scale "
      f"x{s_cross:.4f}  ({100*(s_cross-1):+.1f} %)")
print(f"the measured scale correction is         x{1/r_med:.4f}  "
      f"({100*(1/r_med-1):+.1f} %), 95 % interval "
      f"{100*(1/hi_r-1):+.1f} % to {100*(1/lo_r-1):+.1f} %")
inside = (1 / hi_r) <= s_cross <= (1 / lo_r)
print(f"crossing scale INSIDE the measured interval: {inside}"
      f"   <= if True, A-41's margin cannot be asserted on this evidence alone")
print(f"passenger stock never approaches the extent at any scale tested "
      f"({A41['passenger_mode']/lo_r:.2f} m at the extreme).")

gf = gp.loc[gp["freight"], "gap_m"].to_numpy()
gc = gp.loc[~gp["freight"], "gap_m"].to_numpy()
print(f"\nDIAGNOSTIC-CHAIN gap population, same statistic, for comparison:")
print(f"   freight {short_mode(gf):.2f} m ({len(gf)} gaps), "
      f"passenger {short_mode(gc):.2f} m ({len(gc)} gaps)")
print(f"   histogram modes: freight {hist_mode(gf, 1.0, 3.5):.2f} m, "
      f"passenger {hist_mode(gc, 2.0, 4.5):.2f} m")
print(f"""   Both sit ABOVE A-41's safety-chain figures, which is the expected
   direction: the diagnostic detector enforces a speed-adaptive minimum
   separation and merges the closest pairs, so the gaps that survive into this
   population are the longer ones. It cannot be substituted for A-41's number.""")

# consist length, the other spatial quantity that carries the same scale
frnames = set(gp.loc[gp["freight"], "name"])
cls0 = d[(d["n_axles"] == 12) & (~d["name"].isin(frnames))]
span = cls0["length_m"].median()
print(f"\n12-axle consists: n={len(cls0)}, first-to-last-axle span median "
      f"{span:.2f} m -> {span/r_med:.2f} m corrected")
VEH = 79.91          # AM08 / MS08 three-car set, published overall length
print(f"   AM08 three-car set, published overall length: {VEH} m "
      f"(axle arrangement Bo'Bo'+2'2'+Bo'Bo' = 12 axles)")
print(f"      implied end overhang, nominal scale : {(VEH - span)/2:+.2f} m per end")
print(f"      implied end overhang, corrected     : "
      f"{(VEH - span/r_med)/2:+.2f} m per end")
s_max = VEH / span
print(f"\n   The overhang cannot be negative, so this HARD-BOUNDS the scale:")
print(f"      s <= {VEH:.2f} / {span:.2f} = x{s_max:.4f}  "
      f"({100*(s_max-1):+.1f} %)   even with zero overhang")
print(f"      measured correction x{1/r_med:.4f} ({100*(1/r_med-1):+.1f} %) "
      f"{'EXCEEDS' if 1/r_med > s_max else 'is within'} that bound")
for oh in (2.5, 3.0, 3.5):
    print(f"      at a {oh:.1f} m/end overhang the scale would be "
          f"x{(VEH - 2*oh)/span:.4f}  ({100*((VEH-2*oh)/span-1):+.1f} %)")
print("""
HOW TO READ THAT. `length_m` is the FIRST AXLE to LAST AXLE span, not the length
over couplers, so it MUST come out shorter than the vehicle by the two end
overhangs. That makes the catalogue figure a one-sided test rather than a
calibration: a scale that drives the implied overhang NEGATIVE is impossible,
whatever the true overhang is. The sign is what carries the information here, not
the magnitude.

The AM08 axle arrangement is Bo'Bo'+2'2'+Bo'Bo' - twelve axles - which matches
the 12-axle class-0 consists. Its overall length is a published figure; its
bogie-pivot pitch, which would turn this into a two-sided calibration, is not in
any source consulted.""")

print(f"{L}\nWHERE THE 8 % LIVES, IF NOT IN THE HEAD SPACING\n{L}")
need_km = 2.726 / r_med
print(f"For the sensor speed to be right, the distance the punctuality")
print(f"timestamps actually span would have to be "
      f"{need_km:.3f} km, not {2.726:.3f} km — "
      f"{1000*(need_km-2.726):.0f} m longer.")
print(f"""
That is not absurd: Aiseau and Farciennes are operational POINTS with physical
extent, and a timestamp recorded at a point boundary rather than at the platform
centre would lengthen the effective leg. Chainage between platform centres is
what was measured (A-46), validated to 39-73 m against the stated PKs - good
enough to place the site, not good enough to certify a leg to 8 %.

Two candidates remain and this project cannot separate them:
  (a) the punctuality timestamps span a longer distance than platform-to-platform;
  (b) something in the leg is not what it is assumed to be (a speed restriction
      profile, a systematic detection offset differing between the two points).
What the AM08 geometry rules out is the third candidate - that our spatial scale
is wrong - and that is the one that mattered, because it is the only one that
would have propagated into A-41.""")

pd.DataFrame(rows).to_csv(OUT / "scale_sensitivity.csv", index=False)
mm.to_csv(OUT / "crossing_check.csv", index=False)
print(f"\nwrote scale_sensitivity.csv, crossing_check.csv")

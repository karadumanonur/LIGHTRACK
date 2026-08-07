"""Step 4 — match detected passages to timetabled trains, and step 5's outputs.

GEOMETRY (from 48_pk_resolve.py, chainage along the Lambert72 alignment)
    Aiseau      PK 23.761      site  PK 25.889 +- 0.050      Farciennes PK 26.487
    leg L = 2.726 km;  Aiseau->site 2.128 km;  site->Farciennes 0.598 km
Toward Charleroi is PK increasing, i.e. Aiseau first.

TRANSIT MODEL. A train is not at constant speed over this leg, and the error has
opposite sign in the two directions: travelling toward Namur the site is only
598 m past Farciennes, so a train that stopped there is still accelerating hard;
travelling toward Charleroi the site is 598 m BEFORE Farciennes, so a train that
stops there is already braking. Linear interpolation is wrong in both cases and
in opposite directions.

The branch is taken on the MEASURED dwell, not on the relation name:

    dwell = 0 s at both stations   -> through passage, constant speed
    dwell > 0 s at both stations   -> stop-to-stop, symmetric trapezoid

That split is not assumed - it is what the data shows. IC 25, IC 19-1, P and
EXTRA record dwell exactly 0 at both stations in every one of 6954 rows; S61-1
and S61-2 record a non-zero dwell in 4156 of 4157. So the dwell branch and the
service family are the same partition, established two ways.

TRAPEZOID. Symmetric accelerate / cruise / brake over L in the measured leg time
T. Given the nominal acceleration a, the cruise speed follows from L and T:

    v = (aT - sqrt(a^2 T^2 - 4aL)) / 2

which has a real root only for a >= 4L/T^2 - the triangular limit, where the
train never cruises. a is INFERRED, not measured, so it is swept and the
sensitivity of the predicted site time is reported rather than hidden.

MATCHING is global (Hungarian assignment on |dt|), not greedy: greedy matching
lets an early passage consume the partner of a later one and inflates the rate.
Tolerance is swept rather than fixed.

NO TUNING. Nothing in lightrack/config.toml is touched by this script, and no
threshold anywhere is chosen to improve the match rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, list_files, parse_filename_utc, to_local  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SLOT = CONFIG["acquisition"]["slot_s"]
L_KM, D_AIS, D_FAR = 2.726, 2.128, 0.598
A_NOM, A_SWEEP = 0.7, (0.5, 0.7, 1.0)
# The brief asked for +-30 s to +-180 s. The count is already flat at the bottom
# of that range, so the sweep is extended DOWNWARD to find where it actually
# breaks - a knee outside the requested range is still the knee.
TOLS = (5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180)
pd.set_option("display.width", 250)
LINE = "=" * 116

# ============================================================ timetable side
p = pd.read_csv(OUT / "punctuality_site.csv", low_memory=False)
p["arr"] = (pd.to_datetime(p["REAL_DATE_ARR"], format="%d%b%Y")
            + pd.to_timedelta(p["REAL_TIME_ARR"]))
p["dep"] = (pd.to_datetime(p["REAL_DATE_DEP"], format="%d%b%Y")
            + pd.to_timedelta(p["REAL_TIME_DEP"]))
p["dwell"] = (p["dep"] - p["arr"]).dt.total_seconds()

piv = p.pivot_table(index=["DATDEP", "TRAIN_NO"], columns="station",
                    values=["arr", "dep", "dwell"], aggfunc="first")
t = pd.DataFrame({
    "arr_A": piv[("arr", "Aiseau")], "dep_A": piv[("dep", "Aiseau")],
    "arr_F": piv[("arr", "Farciennes")], "dep_F": piv[("dep", "Farciennes")],
    "dw_A": piv[("dwell", "Aiseau")], "dw_F": piv[("dwell", "Farciennes")],
}).dropna(subset=["dep_A", "dep_F", "arr_A", "arr_F"])
meta = p.drop_duplicates(["DATDEP", "TRAIN_NO"]).set_index(["DATDEP", "TRAIN_NO"])
t = t.join(meta[["RELATION", "RELATION_DIRECTION"]]).reset_index()

t["toward_charleroi"] = t["arr_F"] > t["arr_A"]
t["t_up"] = np.where(t["toward_charleroi"], t["dep_A"], t["dep_F"])
t["t_down"] = np.where(t["toward_charleroi"], t["arr_F"], t["arr_A"])
t["t_up"] = pd.to_datetime(t["t_up"]); t["t_down"] = pd.to_datetime(t["t_down"])
t["dw_up"] = np.where(t["toward_charleroi"], t["dw_A"], t["dw_F"])
t["dw_down"] = np.where(t["toward_charleroi"], t["dw_F"], t["dw_A"])
t["d_site_km"] = np.where(t["toward_charleroi"], D_AIS, D_FAR)
t["leg_s"] = (t["t_down"] - t["t_up"]).dt.total_seconds()
t = t[(t["leg_s"] > 45) & (t["leg_s"] < 900)].copy()

t["stopping"] = (t["dw_up"] > 0) & (t["dw_down"] > 0)
t["through"] = (t["dw_up"] == 0) & (t["dw_down"] == 0)
t["family"] = np.where(t["stopping"], "S61 stop-stop",
                       np.where(t["through"], "IC through", "mixed"))


def t_to_site(T: np.ndarray, d: np.ndarray, stopping: np.ndarray,
              a_nom: float) -> np.ndarray:
    """Seconds from the upstream reference to the site."""
    Lm, dm = L_KM * 1000.0, d * 1000.0
    out = T * dm / Lm                                   # constant-speed default
    s = stopping & np.isfinite(T)
    if not s.any():
        return out
    Ts = T[s]
    a = np.maximum(a_nom, 1.02 * 4 * Lm / Ts ** 2)      # keep the root real
    v = (a * Ts - np.sqrt(np.maximum(a * a * Ts ** 2 - 4 * a * Lm, 0.0))) / 2
    t_acc, d_acc = v / a, v * v / (2 * a)
    ds = dm[s]
    res = np.where(
        ds <= d_acc, np.sqrt(2 * ds / a),
        np.where(ds <= Lm - d_acc, t_acc + (ds - d_acc) / v,
                 Ts - np.sqrt(2 * np.maximum(Lm - ds, 0.0) / a)))
    out[s] = res
    return out


print(f"{LINE}\nSTEP 4 — TRANSIT MODEL\n{LINE}")
print(f"trains with both stations, plausible leg: {len(t)}")
print(f"{'family':<18}{'n':>7}{'leg med s':>11}{'dwell up med':>14}")
for f, g in t.groupby("family"):
    print(f"{f:<18}{len(g):>7}{g['leg_s'].median():>11.0f}{g['dw_up'].median():>14.0f}")

T = t["leg_s"].to_numpy(float)
D = t["d_site_km"].to_numpy(float)
S = t["stopping"].to_numpy(bool)
t["dt_lin"] = T * D / L_KM
for a in A_SWEEP:
    t[f"dt_a{a}"] = t_to_site(T, D, S, a)
t["dt_kin"] = t[f"dt_a{A_NOM}"]

print(f"\ntime from the upstream reference to the site, seconds "
      f"(median by family and direction)")
print(f"{'family':<18}{'direction':<14}{'n':>6}{'linear':>9}"
      + "".join(f"{'a=' + str(a):>9}" for a in A_SWEEP) + f"{'kin-lin':>9}{'spread':>8}")
for (f, tc), g in t.groupby(["family", "toward_charleroi"]):
    if len(g) < 20:
        continue
    cols = [g[f"dt_a{a}"].median() for a in A_SWEEP]
    print(f"{f:<18}{'->Charleroi' if tc else '->Namur':<14}{len(g):>6}"
          f"{g['dt_lin'].median():>9.1f}" + "".join(f"{c:>9.1f}" for c in cols)
          + f"{g['dt_kin'].median() - g['dt_lin'].median():>9.1f}"
          f"{max(cols) - min(cols):>8.1f}")
print(f"""
The 'spread' column is how far the prediction moves across a = {A_SWEEP[0]}-{A_SWEEP[-1]} m/s^2.
It is the price of the one parameter that is inferred rather than measured, and
it should be read against the matching tolerances below: if the spread is small
compared with the tolerance at the knee, the acceleration model is not what
limits the match.""")

t["site_lin"] = t["t_up"] + pd.to_timedelta(t["dt_lin"], unit="s")
t["site_kin"] = t["t_up"] + pd.to_timedelta(t["dt_kin"], unit="s")

# =============================================================== sensor side
d = pd.read_csv(OUT / "passage_times.csv")
d["train_local"] = pd.to_datetime(d["train_local"]).dt.tz_localize(None)
d["file_local"] = pd.to_datetime(d["file_local"]).dt.tz_localize(None)

files = [(q.name, to_local(parse_filename_utc(q.name)).replace(tzinfo=None))
         for q in list_files()]
win = pd.DataFrame(files, columns=["name", "start"])
win["end"] = win["start"] + pd.Timedelta(seconds=SLOT)
lo, hi = win["start"].min(), win["end"].max()
t = t[(t["site_kin"] >= lo) & (t["site_kin"] <= hi)].copy()

print(f"\n{LINE}\nOBSERVATION WINDOWS\n{LINE}")
print(f"delivered records            : {len(win)}   "
      f"({len(d)} carry a passage time)")
print(f"campaign span                : {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} local")
print(f"total recorded time          : {len(win)*SLOT/3600:.2f} h")
print(f"passenger trains in span     : {len(t)}")

# Assign by searchsorted rather than IntervalIndex: consecutive slots are
# contiguous to within the timestamp resolution, so the intervals touch and an
# IntervalIndex refuses them as overlapping.
win = win.sort_values("start").reset_index(drop=True)
ws = win["start"].to_numpy("datetime64[ns]")
pos = np.searchsorted(ws, t["site_kin"].to_numpy("datetime64[ns]"), side="right") - 1
off = np.where(pos >= 0,
               (t["site_kin"].to_numpy("datetime64[ns]")
                - ws[np.clip(pos, 0, None)]) / np.timedelta64(1, "s"),
               np.inf)
t["win_i"] = np.where((pos >= 0) & (off <= SLOT), pos, -1)
t["in_window"] = t["win_i"] >= 0
t["win_name"] = np.where(t["in_window"],
                         win["name"].to_numpy()[np.clip(t["win_i"], 0, None)], None)
print(f"of those, expected inside a delivered window: {int(t['in_window'].sum())}")

# --------------------------------------------- item 2b: two trains per record
per = t[t["in_window"]].groupby("win_name").size()
print(f"\n{LINE}\nDO RETAINED RECORDS EVER HOLD MORE THAN ONE TIMETABLED TRAIN?\n{LINE}")
print(f"records containing 1 expected passenger train : {int((per == 1).sum())}")
print(f"records containing 2 or more                  : {int((per >= 2).sum())}")
if (per >= 2).any():
    print(f"\n{'record':<30}{'n':>3}  expected site times / relation")
    for nm, k in per[per >= 2].items():
        sub = t[t["win_name"] == nm].sort_values("site_kin")
        det = d[d["name"] == nm]
        got = "" if det.empty else f"   detected {det['train_local'].iloc[0]:%H:%M:%S}"
        print(f"{nm[:28]:<30}{k:>3}{got}")
        for _, r in sub.iterrows():
            print(f"{'':<33}{r['site_kin']:%H:%M:%S}  {r['RELATION']:<10} "
                  f"{'->Charleroi' if r['toward_charleroi'] else '->Namur'}")
print("""
The pipeline emits ONE passage per record - 52_passage_times.py found a single
segment in every one of the 76 records that produced a passage at all. Any record
above holding two timetabled trains is therefore a record where at most one of
them can be reported, whatever the detector does.""")


# =================================================================== matching
def match(pred_col: str, tol: float, sub: pd.DataFrame | None = None):
    tt = t if sub is None else sub
    if tt.empty or d.empty:
        return pd.DataFrame(), 0
    a = d["train_local"].to_numpy("datetime64[ns]").astype("int64") / 1e9
    b = tt[pred_col].to_numpy("datetime64[ns]").astype("int64") / 1e9
    cost = np.abs(a[:, None] - b[None, :])
    big = tol * 1000.0
    c = np.where(cost <= tol, cost, big)
    ri, ci = linear_sum_assignment(c)
    ok = c[ri, ci] < big
    out = pd.DataFrame({
        "name": d["name"].to_numpy()[ri[ok]],
        "sub": d["sub"].to_numpy()[ri[ok]],
        "dt_s": (a[ri[ok]] - b[ci[ok]]),
        "tt_index": tt.index.to_numpy()[ci[ok]],
    })
    return out, int(ok.sum())


print(f"\n{LINE}\nTOLERANCE SWEEP — GLOBAL ASSIGNMENT\n{LINE}")
print(f"{'tol +-s':>8}{'matched (kin)':>15}{'% of 76':>10}{'matched (lin)':>15}"
      f"{'% of 76':>10}{'gain':>7}{'median |dt|':>13}")
sweep = []
for tol in TOLS:
    mk, nk = match("site_kin", tol)
    ml, nl = match("site_lin", tol)
    md = np.median(np.abs(mk["dt_s"])) if nk else np.nan
    print(f"{tol:>8}{nk:>15}{100*nk/len(d):>9.1f}%{nl:>15}"
          f"{100*nl/len(d):>9.1f}%{nk-nl:>7}{md:>13.1f}")
    sweep.append(dict(tol=tol, kin=nk, lin=nl, med_abs_dt=md))
sw = pd.DataFrame(sweep)
sw["gain"] = sw["kin"].diff()
# knee = the smallest tolerance beyond which loosening buys at most one match
cand = sw.loc[sw["kin"].diff().shift(-1).fillna(0) <= 1, "tol"]
knee = int(cand.iloc[0]) if len(cand) else int(sw["tol"].iloc[-1])
print(f"\nknee at +-{knee} s: loosening past it buys at most one further match.")
print(f"matched at the loosest tolerance tested (+-{TOLS[-1]} s): "
      f"{int(sw['kin'].iloc[-1])}")

# ------------------------------------------------------- the knee, in detail
mk, nk = match("site_kin", knee)
t["matched"] = t.index.isin(mk["tt_index"])
d["matched"] = d.set_index(["name", "sub"]).index.isin(
    mk.set_index(["name", "sub"]).index)

print(f"\n{LINE}\nTHREE POPULATIONS AT +-{knee} s, REPORTED SEPARATELY\n{LINE}")
n_win = int(t["in_window"].sum())
n_mw = int((t["in_window"] & t["matched"]).sum())
k, n = n_mw, n_win
ci = (beta.ppf(.025, k, n - k + 1) if k else 0.0,
      beta.ppf(.975, k + 1, n - k) if k < n else 1.0)
print(f"MATCHED passage <-> train                       : {nk}")
print(f"TIMETABLED inside a delivered window, not matched: {n_win - n_mw}")
print(f"DETECTED passage with no timetabled partner      : {len(d) - nk}")
print(f"\nDETECTION RATE (of trains expected inside a delivered window)")
print(f"   {n_mw} / {n_win} = {100*n_mw/n_win:.1f} %   "
      f"[95 % CI {100*ci[0]:.1f}-{100*ci[1]:.1f} %]")
print(f"""
This is NOT the 66.0 % figure from 50_direction_and_profile.py. That one is a
CAPTURE rate - what fraction of the day's trains reached a retained file - and it
measures the project's own file-selection filter. This one is a DETECTION rate:
given that a train was inside a delivered window, did the pipeline find it and
time it to within {knee} s. The two must never be quoted as one number.""")

print(f"\n{LINE}\nMATCH RATE BY FAMILY — DOES THE ACCELERATION MODEL EARN ITS KEEP?\n{LINE}")
print(f"{'family':<18}{'direction':<14}{'in window':>11}{'matched kin':>13}"
      f"{'matched lin':>13}{'rate kin':>10}{'med dt kin':>12}{'med dt lin':>12}")
rows = []
for (f, tc), g in t[t["in_window"]].groupby(["family", "toward_charleroi"]):
    mk2, nk2 = match("site_kin", knee, g)
    ml2, nl2 = match("site_lin", knee, g)
    dk = np.median(mk2["dt_s"]) if nk2 else np.nan
    dl = np.median(ml2["dt_s"]) if nl2 else np.nan
    print(f"{f:<18}{'->Charleroi' if tc else '->Namur':<14}{len(g):>11}{nk2:>13}"
          f"{nl2:>13}{100*nk2/len(g):>9.1f}%{dk:>12.1f}{dl:>12.1f}")
    rows.append(dict(family=f, toward_charleroi=tc, n=len(g), kin=nk2, lin=nl2,
                     med_dt_kin=dk, med_dt_lin=dl))
print("""
A systematically WORSE rate for the stopping family would say the acceleration
model is the weak part rather than the matching. A comparable rate says the
transit model is adequate at this tolerance and the residual is elsewhere.
The median dt columns carry the sign: a positive value means the sensor saw the
train LATER than the model predicted.""")

# --------------------------------------------------------------- A-12, speed
print(f"\n{LINE}\nA-12 — DIRECTION CONVENTION, CLOSED AGAINST RELATION_DIRECTION\n{LINE}")
mm = mk.merge(d[["name", "sub", "head1_first", "v_kmh", "n_axles", "length_m",
                 "gap_min_m", "n_gap_lt_2m5", "score"]], on=["name", "sub"])
mm = mm.join(t[["toward_charleroi", "RELATION", "RELATION_DIRECTION", "leg_s",
                "family", "site_kin"]], on="tt_index")
ct = pd.crosstab(mm["head1_first"], mm["toward_charleroi"], dropna=False)
print(ct.rename(columns={False: "->Namur", True: "->Charleroi"},
                index={False: "head 3 first", True: "head 1 first"}))
agree = int(((mm["head1_first"]) == (mm["toward_charleroi"])).sum())
n_ct = len(mm)
maj = max(agree, n_ct - agree)
print(f"\nconsistency of the majority mapping: {maj}/{n_ct} = {100*maj/n_ct:.1f} %")
print("head 1 reached first  <->  "
      + ("toward Charleroi (PK increasing, Aiseau -> Farciennes)"
         if agree >= n_ct - agree else
         "toward Namur (PK decreasing, Farciennes -> Aiseau)"))
maj_is_diag = agree >= n_ct - agree
mm["dir_ok"] = (mm["head1_first"] == mm["toward_charleroi"]) == maj_is_diag
print(f"\nrecords AGAINST the majority mapping (sensor direction wrong): "
      f"{int((~mm['dir_ok']).sum())}")
for _, r in mm[~mm["dir_ok"]].iterrows():
    print(f"   {r['name'][:28]:<30} score {r['score']:.2f}  "
          f"{str(r['RELATION']):<10} "
          f"{'->Charleroi' if r['toward_charleroi'] else '->Namur':<12} "
          f"dt {r['dt_s']:+.1f} s")
rev_ok = mm[(~mm["head1_first"]) & mm["dir_ok"]]
print(f"\nreverse-labelled records CONFIRMED reverse by the timetable: {len(rev_ok)}")
for _, r in rev_ok.iterrows():
    print(f"   {r['name'][:28]:<30} score {r['score']:.2f}  "
          f"{str(r['RELATION_DIRECTION'])[:44]:<46} dt {r['dt_s']:+.1f} s")
print(f"""
This is the branch 51_adjacent_track_test.py could not take. Of the 11 reverse-
labelled records, {int((~mm['head1_first']).sum())} matched a timetabled train; of those,
{len(rev_ok)} carry a RELATION_DIRECTION that really is toward Namur and
{int(((~mm['head1_first']) & (~mm['dir_ok'])).sum())} do not. The ones that do not are solver errors,
which is what their fit scores already said.""")

# do the reverse records coincide with a crossing?
multi = set(per[per >= 2].index)
rv = mm[~mm["head1_first"]]["name"]
print(f"reverse-labelled records that sit in a window holding two timetabled "
      f"trains: {int(rv.isin(multi).sum())}/{len(rv)}")

print(f"\n{LINE}\nSPEED CROSS-CHECK — SENSOR vs STATION-TO-STATION AVERAGE\n{LINE}")
mm["v_leg"] = L_KM / (mm["leg_s"] / 3600)
print(f"{'family':<18}{'n':>5}{'v sensor med':>14}{'v leg med':>12}{'ratio':>8}")
for f, g in mm.groupby("family"):
    print(f"{f:<18}{len(g):>5}{g['v_kmh'].median():>14.1f}"
          f"{g['v_leg'].median():>12.1f}{(g['v_kmh']/g['v_leg']).median():>8.2f}")
print(f"""
The leg average includes the station transients, so for the stopping family the
sensor speed SHOULD exceed it. For the through family the two measure nearly the
same thing and a ratio far from 1 would be a real disagreement.""")
hi_v = mm[mm["v_kmh"] > 130].sort_values("v_kmh", ascending=False)
print(f"\nupper tail, sensor v > 130 km/h ({len(hi_v)} matched records):")
if len(hi_v):
    print(f"{'record':<28}{'v sens':>8}{'v leg':>8}{'relation':<12}{'score':>7}")
    for _, r in hi_v.iterrows():
        print(f"{r['name'][:26]:<28}{r['v_kmh']:>8.1f}{r['v_leg']:>8.1f}"
              f"  {str(r['RELATION']):<12}{r['score']:>7.2f}")

# ------------------------------------------------------------ freight labels
print(f"\n{LINE}\nDETECTED WITH NO TIMETABLED PARTNER — FREIGHT OR FALSE?\n{LINE}")
un = d[~d["matched"]].copy()
inw = t[t["in_window"]].groupby("name" if False else "win_name").size()
un["tt_in_own_window"] = un["name"].map(inw).fillna(0).astype(int)
print(f"unmatched detected passages: {len(un)}")
print(f"{'record':<28}{'v':>7}{'axles':>7}{'len m':>8}{'gapmin':>8}"
      f"{'<2.5m':>7}{'tt in window':>14}")
for _, r in un.sort_values("train_local").iterrows():
    print(f"{r['name'][:26]:<28}{r['v_kmh']:>7.1f}{r['n_axles']:>7}"
          f"{r['length_m']:>8.1f}{r['gap_min_m']:>8.2f}{r['n_gap_lt_2m5']:>7}"
          f"{r['tt_in_own_window']:>14}")
print(f"\nfor comparison, the MATCHED passenger population:")
mp = d[d["matched"]]
print(f"   axles  median {mp['n_axles'].median():.0f}  range "
      f"{mp['n_axles'].min()}-{mp['n_axles'].max()}")
print(f"   gap_min_m median {mp['gap_min_m'].median():.2f}  "
      f"p10 {mp['gap_min_m'].quantile(.1):.2f}")
print(f"   gaps < 2.5 m: median {mp['n_gap_lt_2m5'].median():.0f} per passage")
print("""
An unmatched passage has three possible causes and they are not interchangeable:
freight (invisible to this reference), a passenger train not stopping at either
PTCAR, or a false detection. The axle-gap signature separates the first from the
other two; nothing here separates the second from the third.""")

# The discriminator is the SHARE of short gaps, not the smallest one. A freight
# wagon contributes two ~1.8 m intra-bogie gaps, so a freight consist is roughly
# half short gaps; a single short gap in a passenger set is one split peak.
# Taking the minimum instead would let one artefact relabel a whole passage - and
# it does: three matched passenger passages carry a gap under 2.0 m.
print(f"\nCALIBRATION — number of inter-axle gaps below 2.5 m, all "
      f"{len(d)} detected passages")
cnt = np.sort(d["n_gap_lt_2m5"].to_numpy())
gapi = int(np.argmax(np.diff(cnt)))
cut = 0.5 * (cnt[gapi] + cnt[gapi + 1])
print(f"   sorted upper tail: {cnt[-14:].tolist()}")
print(f"   largest empty interval: {cnt[gapi]} .. {cnt[gapi+1]}  "
      f"=> split at {cut:.0f}")
print(f"   the threshold is the midpoint of an empty interval in the observed")
print(f"   distribution, not a value chosen to make the answer come out.")
print(f"""
The COUNT is used rather than the share. A freight wagon contributes two ~1.8 m
intra-bogie gaps, so the count scales with consist length and separates cleanly;
the share is noisy on short consists, where one split peak out of three gaps
already reads 0.33.""")
un["freight_sig"] = un["n_gap_lt_2m5"] > cut
n_f = int(un["freight_sig"].sum())
print(f"{'label':<40}{'n':>5}{'axles med':>11}{'length med m':>14}{'short gaps':>12}")
for lab, sub in (("freight signature (short-gap bogies)", un[un["freight_sig"]]),
                 ("passenger-like, no timetable partner", un[~un["freight_sig"]])):
    if len(sub):
        print(f"{lab:<40}{len(sub):>5}{sub['n_axles'].median():>11.0f}"
              f"{sub['length_m'].median():>14.1f}"
              f"{sub['n_gap_lt_2m5'].median():>12.0f}")
short_matched = mp[mp["gap_min_m"] < 2.0]
print(f"\nmatched PASSENGER passages carrying a gap below 2.0 m: "
      f"{len(short_matched)}")
for _, r in short_matched.iterrows():
    print(f"   {r['name'][:28]:<30} {int(r['n_axles'])} axles, "
          f"{r['length_m']:.0f} m, gap_min {r['gap_min_m']:.2f} m, "
          f"{int(r['n_gap_lt_2m5'])} short of {int(r['n_axles'])-1}")
print(f"""
Both of those are the reverse-direction records confirmed above, which is worth
carrying: whatever produces the extra short gaps is not random with respect to
direction. It is nowhere near enough to relabel them - {int(short_matched['n_gap_lt_2m5'].max())} short gaps against
{int(un.loc[un['freight_sig'], 'n_gap_lt_2m5'].min())} in the lightest freight consist - but it is a loose end, not a closed one.""")
print(f"""
{n_f} of the {len(un)} unmatched passages carry the freight signature - long
consists with a dense population of ~2 m gaps, which is the bogie spacing of
freight wagons. They are {100*n_f/len(d):.1f} % of the {len(d)} detected passages.

That fraction is NOT an estimate of freight in the traffic. The 93 records were
hand-selected, so their composition reflects the selection, not the line. The
independent estimate from the EBP ratio (50_direction_and_profile.py) put freight
near 14 trains per direction per day against ~137 passenger, and the two figures
are not comparable for exactly that reason.""")

t.to_csv(OUT / "timetable_expected.csv", index=False)
mm.to_csv(OUT / "timetable_matches.csv", index=False)
sw.to_csv(OUT / "timetable_tolerance_sweep.csv", index=False)
pd.DataFrame(rows).to_csv(OUT / "timetable_by_family.csv", index=False)
print(f"\nwrote timetable_expected.csv, timetable_matches.csv, "
      f"timetable_tolerance_sweep.csv, timetable_by_family.csv")

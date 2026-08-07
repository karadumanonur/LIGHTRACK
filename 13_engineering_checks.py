"""Engineering checks: SNR, head spacing, timing-transfer robustness, health feature.

Four questions that all bear on decisions rather than on reproducing the deck:

  A. SNR, not threshold margin. 40 pm is not a physical constant — it was chosen
     for glued pads. If clamping is adopted the threshold can move with it. So
     the decision-relevant quantity for PAD 10 is peak / noise sigma. High SNR
     means "lower the threshold"; low SNR means the clamp preload genuinely
     needs redesign for POC2.
  B. Head spacing. `speed.estimate` IMPOSES lag_bc/lag_ab = 15/10. The freely
     refined lags do not, so their ratio measures the true spacing.
  C. Timing-transfer robustness vs speed. At 66 km/h a peak is ~60 samples wide
     and a few samples of lag error are harmless; at 141 km/h it is ~28 and the
     same error lands on the flank.
  D. Cross-head axle-count disagreement as a health feature for Phase 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import CONFIG, CH_OF_PAD, samples_per_peak  # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
D12 = CONFIG["geometry"]["head1_to_head2_m"]
D23 = CONFIG["geometry"]["head2_to_head3_m"]
THRESH = 40.0

a = pd.read_csv(OUT / "phase1_amplitudes_raw.csv")
a = a[a["speed_ok"]].copy()
rec = a.drop_duplicates("name").copy()
pd.set_option("display.width", 220)
LINE = "=" * 100

# ---------------------------------------------------------------- A. SNR
print(f"{LINE}\nA. SIGNAL-TO-NOISE — the decision-relevant quantity, not threshold margin\n{LINE}")
print("40 pm was chosen for glued pads. If clamping is adopted the threshold")
print("moves with it. What cannot be fixed by moving a threshold is poor SNR.\n")
g = a.groupby("pad")
mnt = {1: "glued, centred", 2: "glued, centred", 3: "glued, +7.5cm",
       4: "glued, +7.5cm", 5: "glued, centred", 6: "glued, centred (CONTROL)",
       7: "glued, +3cm", 8: "glued, +3cm", 9: "1/2 collage",
       10: "clipse (clamped)", 11: "unprepared surface", 12: "taraude (screwed)"}
print(f"{'PAD':>4}{'ch':>6}  {'mounting':<26}{'noise pm':>10}{'peak pm':>9}"
      f"{'SNR med':>9}{'SNR min':>9}{'40pm marg':>10}")
rows = []
for pad in range(1, 13):
    s = g.get_group(pad)
    sess = s.groupby("session").agg(pos=("pos_pm", "median"), noise=("noise_pm", "median"))
    snr = sess["pos"] / sess["noise"]
    rows.append({"pad": pad, "noise": sess["noise"].median(), "pos": sess["pos"].median(),
                 "snr_med": snr.median(), "snr_min": snr.min(),
                 "margin": sess["pos"].min() / THRESH})
    print(f"{pad:>4}{CH_OF_PAD[pad]:>6}  {mnt[pad]:<26}{sess['noise'].median():10.2f}"
          f"{sess['pos'].median():9.1f}{snr.median():9.0f}{snr.min():9.0f}"
          f"{sess['pos'].min()/THRESH:9.2f}x")
r = pd.DataFrame(rows).set_index("pad")

p10, p12 = r.loc[10], r.loc[12]
print(f"\nPAD 10 (clipse)  : {p10['pos']:.0f} pm at {p10['noise']:.2f} pm noise "
      f"-> SNR {p10['snr_med']:.0f}x  (worst session {p10['snr_min']:.0f}x)")
print(f"PAD 12 (taraude) : {p12['pos']:.0f} pm at {p12['noise']:.2f} pm noise "
      f"-> SNR {p12['snr_med']:.0f}x  (worst session {p12['snr_min']:.0f}x)")
healthy = r.drop(index=[6, 9, 11])
print(f"\nSNR across all non-degrading sensors: "
      f"{healthy['snr_med'].min():.0f}x - {healthy['snr_med'].max():.0f}x")
verdict = ("threshold, not hardware" if p10["snr_min"] >= 20
           else "clamp preload genuinely needs redesign")
print(f"\n=> PAD 10's limitation is a {verdict.upper()} problem.")
print(f"   At SNR {p10['snr_med']:.0f}x, a detection threshold set at "
      f"{5*p10['noise']:.1f} pm (5 sigma) would give PAD 10 a "
      f"{p10['pos']/(5*p10['noise']):.0f}x margin.")
r.to_csv(OUT / "phase1_snr.csv")

# ------------------------------------------------------- B. head spacing
print(f"\n{LINE}\nB. TRUE HEAD SPACING — from freely refined lags\n{LINE}")
rec = rec[(rec["ref_lag_ab"] != 0) & (rec["ref_lag_bc"] != 0)]
rec["ratio_imposed"] = rec["lag_bc"] / rec["lag_ab"]
rec["ratio_free"] = rec["ref_lag_bc"] / rec["ref_lag_ab"]
print(f"records: {len(rec)}")
print(f"imposed ratio (by construction) : {rec['ratio_imposed'].median():.3f}")
q = rec["ratio_free"].quantile([.1, .25, .5, .75, .9])
print(f"freely refined ratio            : median {rec['ratio_free'].median():.3f}")
print("  " + "  ".join(f"p{int(k*100)}={v:.3f}" for k, v in q.items()))
print(f"  IQR width {q[.75]-q[.25]:.3f}   sd {rec['ratio_free'].std():.3f}")
med = float(rec["ratio_free"].median())
print(f"\nnominal 15:10 = 1.500   measured = {med:.3f}   "
      f"({100*(med-1.5)/1.5:+.1f} %)")
print(f"taking Head1<->Head2 = {D12} m as exact, Head2<->Head3 = "
      f"{D12/med:.2f} m  (nominal {D23} m)")
print(f"taking Head2<->Head3 = {D23} m as exact, Head1<->Head2 = "
      f"{D23*med:.2f} m  (nominal {D12} m)")
print(f"\nsystematic speed error if the nominal 1.500 is kept: "
      f"~{100*abs(med-1.5)/1.5/2:.1f} % on the TOF estimate")

# --------------------------------------------- C. transfer robustness
print(f"\n{LINE}\nC. TIMING-TRANSFER ROBUSTNESS vs SPEED\n{LINE}")
print("offset = where the true local max actually sat, relative to the")
print("transferred index. The search half-window is 0.5 x samples_per_peak, so")
print("|offset| approaching search_half means the peak is escaping the window.\n")
a["v_bin"] = pd.cut(a["v_kmh"], [30, 60, 80, 100, 120, 145])
b = a.groupby(["v_bin", "head"], observed=True).agg(
    n=("name", "nunique"), spp=("v_kmh", lambda x: samples_per_peak(x.median())),
    half=("search_half", "median"),
    off_med=("offset_med", "median"), off_p90=("offset_abs_p90", "median")).reset_index()
b["escape_frac"] = b["off_p90"] / b["half"]
print(f"{'speed km/h':<14}{'head':>5}{'recs':>6}{'peak w':>8}{'half':>6}"
      f"{'median off':>12}{'|off| p90':>11}{'p90/half':>10}")
for _, x in b.iterrows():
    print(f"{str(x['v_bin']):<14}{int(x['head']):>5}{int(x['n']):>6}{x['spp']:8.0f}"
          f"{x['half']:6.0f}{x['off_med']:12.1f}{x['off_p90']:11.1f}"
          f"{x['escape_frac']:10.2f}")
worst = b.loc[b["escape_frac"].idxmax()]
print(f"\nworst case: head {int(worst['head'])} at {worst['v_bin']} km/h, "
      f"|offset| p90 = {worst['off_p90']:.1f} of a {worst['half']:.0f}-sample "
      f"half-window ({worst['escape_frac']:.0%})")
print("Head 1 is the reference-transfer origin in almost every record, so its")
print("offsets are near zero by construction; Heads 2 and 3 carry the real test.")
b.to_csv(OUT / "phase1_transfer_offsets.csv", index=False)

# --------------------------------------------- ref-head exceptions
print(f"\n{LINE}\nREFERENCE-HEAD EXCEPTIONS — the records where Head 2 did not win\n{LINE}")
rh = a.drop_duplicates("name")
print(rh["ref_head"].value_counts().to_string())
odd = rh[rh["ref_head"] != 2]
for _, x in odd.iterrows():
    sub = a[a["name"] == x["name"]]
    print(f"\n  {x['name']}  session {x['session']}  v={x['v_kmh']:.1f} "
          f"score={x['score']:.2f}  ref=H{x['ref_head']}")
    for h in (1, 2, 3):
        s = sub[sub["head"] == h]
        print(f"    head {h}: n_indep={s['n_indep'].iloc[0]:3d} "
              f"axles_used={s['n_axles'].iloc[0]:3d} "
              f"median pos {s['pos_pm'].median():7.1f} pm  "
              f"|off| p90 {s['offset_abs_p90'].median():.1f}")

# ------------------------------------- D. count disagreement feature
print(f"\n{LINE}\nD. CROSS-HEAD AXLE-COUNT DISAGREEMENT as a Phase 4 health feature\n{LINE}")
d = a.groupby(["head", "session"])["count_disagree"].median().unstack(0)
print("median (this head's own count) - (reference head's count), per session:")
print(d.to_string())
print("\nper head, over all records:")
for h in (1, 2, 3):
    s = a[a["head"] == h]["count_disagree"]
    print(f"  head {h}: median {s.median():+.0f}  mean {s.mean():+.1f}  "
          f"p90 {s.quantile(0.9):+.0f}  frac non-zero {(s != 0).mean():.0%}")
print("\nHead 3 carries PAD 9 and PAD 11. Its surplus count tracks the degraded")
print("sensors injecting false peaks into the head mean — i.e. the disagreement")
print("is itself a degradation signal, available with no extra computation.")
d.to_csv(OUT / "phase1_count_disagreement.csv")

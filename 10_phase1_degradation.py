"""Phase 1 — reproduce the briefing §8 durability curves. The correctness gate.

For every record: baseline-remove, estimate speed (score-gated), derive axle
times per head, measure per-channel peak amplitude. Aggregate to the 15
measurement sessions, normalise each sensor to session 1, and compare against
the deck's published behaviour.

Two normalisations, deliberately:
  (a) ABSOLUTE   — each sensor's own amplitude / its session-1 amplitude.
                   This is what the deck plots. It inherits all the
                   train-to-train variation in mass and speed.
  (b) REFERENCED — each sensor's amplitude / PAD 1's amplitude in the SAME
                   passage, then normalised to session 1. Common-mode load and
                   speed cancel. Lower variance, but blind to a fault that hits
                   the whole head at once.

Disagreement between (a) and (b) is informative, not a bug: it separates
"this sensor changed" from "the traffic changed".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                     # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc,
    rolling_median_baseline, estimate_speed, channel_amplitudes,
    head_axle_times, refine_lags, to_frame, PAD_OF_CHANNEL, CH_OF_PAD,
)

OUT = Path(CONFIG["paths"]["out_dir"])
SESSIONS = [pd.Timestamp(d).date() for d in CONFIG["campaign"]["measurement_dates"]]
SESSION_IDX = {d: i + 1 for i, d in enumerate(SESSIONS)}
REF_PAD = 1
THRESH_PM = 40.0            # the deck's own detection threshold (briefing §6.2)


def main() -> None:
    rows = []
    for n, p in enumerate(list_files(), 1):
        utc = parse_filename_utc(p.name)
        sess = SESSION_IDX.get(utc.date())
        if sess is None:
            continue                      # 05-14 install day, 05-18 off-hours
        df = load_file(p)
        y = rolling_median_baseline(df[CH_COLS])
        spd = estimate_speed(y)
        amps = channel_amplitudes(y, spd)
        # Free re-measurement of both inter-head lags, to test the assumed
        # 15:10 head spacing rather than imposing it.
        r_ab, r_bc = refine_lags(y, spd)
        t = to_frame(amps, name=p.name, session=sess, date=str(utc.date()),
                     v_kmh=spd.v_kmh, score=spd.score, speed_ok=spd.ok,
                     head1_first=spd.head1_first,
                     lag_ab=spd.lag_ab, lag_bc=spd.lag_bc,
                     ref_lag_ab=r_ab, ref_lag_bc=r_bc)
        rows.append(t)
        print(f"  [{n:3d}] s{sess:02d} {p.name}  v={spd.v_kmh:6.1f} "
              f"score={spd.score:.2f} {'ok ' if spd.ok else 'REJ'} "
              f"ref=H{amps[0].ref_head} axles={t['n_axles'].max():3d} "
              f"indep={'/'.join(str(t[t['head'] == h]['n_indep'].iloc[0]) for h in (1, 2, 3))} "
              f"lag {spd.lag_ab}->{r_ab} {spd.lag_bc}->{r_bc}", flush=True)

    a = pd.concat(rows, ignore_index=True)
    a.to_csv(OUT / "phase1_amplitudes_raw.csv", index=False)

    # ---- speed gate ------------------------------------------------------
    n_all = a["name"].nunique()
    a = a[a["speed_ok"]]
    n_ok = a["name"].nunique()
    print(f"\nrecords: {n_all} in sessions, {n_ok} pass the speed-fit gate "
          f"(score >= {CONFIG['speed']['min_fit_score']})")

    # ---- (b) referenced amplitude ---------------------------------------
    ref_ch = CH_OF_PAD[REF_PAD]
    ref = a[a["channel"] == ref_ch][["name", "pos_pm"]].rename(
        columns={"pos_pm": "ref_pm"})
    a = a.merge(ref, on="name", how="left")
    a["rel"] = a["pos_pm"] / a["ref_pm"]

    # ---- aggregate to sessions ------------------------------------------
    g = a.groupby(["pad", "session"])
    per = g.agg(pos_pm=("pos_pm", "median"),
                rel=("rel", "median"),
                noise_pm=("noise_pm", "median"),
                snr=("snr", "median"),
                pos_max_pm=("pos_max_pm", "max"),
                n=("name", "nunique")).reset_index()

    def norm(col: str) -> pd.DataFrame:
        w = per.pivot(index="session", columns="pad", values=col).sort_index()
        return w / w.iloc[0]

    abs_n = norm("pos_pm")
    rel_n = norm("rel")
    raw = per.pivot(index="session", columns="pad", values="pos_pm").sort_index()

    pd.set_option("display.width", 220)
    LINE = "=" * 100

    print(f"\n{LINE}\nMEDIAN POSITIVE PEAK AMPLITUDE per PAD per session [pm]\n{LINE}")
    print(raw.round(1).to_string())

    print(f"\n{LINE}\n(a) ABSOLUTE, normalised to session 1  <- the deck's view\n{LINE}")
    print(abs_n.round(2).to_string())

    print(f"\n{LINE}\n(b) REFERENCED to PAD {REF_PAD} in the same passage, "
          f"normalised to session 1\n{LINE}")
    print(rel_n.round(2).to_string())

    # ---- the §8 validation targets --------------------------------------
    print(f"\n{LINE}\nREGRESSION AGAINST BRIEFING §8\n{LINE}")
    targets = {
        6:  ("progressive degradation to ~0.5",        lambda s: 0.35 <= s <= 0.65),
        9:  ("collapse to ~0.05 by measurement 13-15", lambda s: s <= 0.15),
        10: ("stable at ~1.0",                         lambda s: s >= 0.80),
        11: ("collapse to ~0.1 by measurement 3",      lambda s: s <= 0.25),
    }
    print(f"{'PAD':>4} {'channel':>8} {'mounting':<26} {'deck expectation':<38}"
          f"{'final(a)':>9}{'final(b)':>9}  verdict")
    mounting = {6: "standard glued (control)", 9: "1/2 collage",
                10: "clipse (clamped)", 11: "glued, unprepared surface"}
    verdicts = {}
    for pad, (desc, test) in targets.items():
        fa = float(abs_n[pad].iloc[-1])
        fb = float(rel_n[pad].iloc[-1])
        ok = test(fa)
        verdicts[pad] = ok
        print(f"{pad:>4} {CH_OF_PAD[pad]:>8} {mounting[pad]:<26} {desc:<38}"
              f"{fa:9.2f}{fb:9.2f}  {'PASS' if ok else 'FAIL'}")

    # PAD 11's collapse must happen BY session 3, not merely by the end.
    s3 = float(abs_n[11].loc[3])
    print(f"\nPAD 11 at session 3: {s3:.2f}  "
          f"(deck: 'collapse to ~0.1 by measurement 3')  "
          f"{'PASS' if s3 <= 0.30 else 'FAIL'}")

    # PAD 9's threshold claim is a hard, checkable number.
    p9 = per[(per["pad"] == 9)].set_index("session")
    last = p9.loc[15]
    print(f"PAD 9 at session 15: median peak {last['pos_pm']:.1f} pm, "
          f"max peak {last['pos_max_pm']:.1f} pm, threshold {THRESH_PM:.0f} pm")
    print(f"  deck: 'by J15 it never crosses the 40 pm threshold'  -> "
          f"{'PASS' if last['pos_max_pm'] < THRESH_PM else 'FAIL (it does cross)'}")

    # ---- the mounting-method question -----------------------------------
    print(f"\n{LINE}\nPAD 10 (clipse) vs PAD 12 (taraude) — the deck's verdict\n{LINE}")
    for pad, label in [(10, "clipse  "), (12, "taraude ")]:
        c = abs_n[pad]
        print(f"  PAD {pad:>2} {label} ch={CH_OF_PAD[pad]}  "
              f"raw session1 {raw[pad].iloc[0]:6.1f} pm  final {c.iloc[-1]:.2f}  "
              f"min {c.min():.2f}  mean {c.mean():.2f}  sd {c.std():.3f}")
    print("  deck: PAD 10 is 'THE real good idea of POC1'; PAD 12 merely stable.")

    # ---- stable-sensor control group ------------------------------------
    stable = [1, 2, 3, 4, 5, 8, 12]
    print(f"\n{LINE}\nCONTROL: sensors the deck calls stable {stable}\n{LINE}")
    for pad in stable:
        c = abs_n[pad]
        print(f"  PAD {pad:>2} ch={CH_OF_PAD[pad]}  final {c.iloc[-1]:.2f}  "
              f"min {c.min():.2f}  max {c.max():.2f}")
    print(f"  PAD  7 ch={CH_OF_PAD[7]}  final {abs_n[7].iloc[-1]:.2f}  "
          f"(deck: mild decline to ~0.75)")

    for name, d in [("phase1_abs_normalised.csv", abs_n),
                    ("phase1_rel_normalised.csv", rel_n),
                    ("phase1_raw_pm.csv", raw)]:
        d.to_csv(OUT / name)
    per.to_csv(OUT / "phase1_per_session.csv", index=False)
    print(f"\nwrote phase1_*.csv to {OUT}")

    n_pass = sum(verdicts.values())
    print(f"\n{LINE}\nGATE: {n_pass}/4 §8 targets reproduced\n{LINE}")


if __name__ == "__main__":
    main()

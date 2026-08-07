"""Phase 2 — run the safety chain and measure what can be measured without logs.

Infrabel's ground-truth logs do not exist here (briefing §8, open item 1), so
absolute counting accuracy CANNOT be computed. Saying otherwise would be
inventing a number.

What can be measured is the axle counter's own safety principle: **Head 1
("Comptage") and Head 2 ("Décomptage") see the same train, so their counts must
agree.** A section is declared clear when in-count equals out-count, so
head-to-head agreement is not a proxy for accuracy — it is the actual quantity
the interlocking acts on.

Head 3 is excluded throughout: it is the mounting-method test zone, not an
operational counting head (briefing §10).

Errors are reported separately and never merged:
  * OVER-count (phantom axle)  -> can make an occupied block look clear. SAFETY.
  * UNDER-count (missed axle)  -> leaves a clear block occupied. AVAILABILITY.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc,
    rolling_median_baseline, estimate_speed, PAD_OF_CHANNEL, CH_OF_PAD,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.safety import count_axles, lowpass_cutoff_hz, samples_per_peak  # noqa: E402
from lightrack.speed import head_signal                        # noqa: E402

OUT = Path(CONFIG["paths"]["out_dir"])
SESSIONS = {pd.Timestamp(d).date(): i + 1
            for i, d in enumerate(CONFIG["campaign"]["measurement_dates"])}
FS = CONFIG["acquisition"]["fs_hz"]

# Operational sensors only. Head 3 (PADs 9-12) is the mounting-method test zone
# and is excluded (briefing §10).
# NB PAD numbers, not channel numbers: Head 1 = PADs 1-4 = ch09-ch12. An earlier
# version listed PADs 9-12 here, which is Head 3 - exactly the sensor set the
# briefing says must never appear in counting statistics.
OPERATIONAL = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}      # head -> PADs
CH_OF = {p: CH_OF_PAD[p] for p in range(1, 13)}


def main() -> None:
    rows = []
    for n, p in enumerate(list_files(), 1):
        utc = parse_filename_utc(p.name)
        sess = SESSIONS.get(utc.date())
        if sess is None:
            continue
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        if not spd.ok:
            continue

        # Passage bracket from the head mean; counting then runs per sensor.
        for head, pads in OPERATIONAL.items():
            hs = head_signal(y, head)
            w = passage_window(hs, spd.v_kmh)
            if w is None:
                continue
            for pad in pads:
                sig = y[CH_OF[pad]].to_numpy()[w[0]:w[1]]
                r = count_axles(sig, spd.v_kmh)
                rows.append({
                    "name": p.name, "session": sess, "v_kmh": spd.v_kmh,
                    "head": head, "pad": pad, "channel": CH_OF[pad],
                    "n_x": r.n_x, "n_d": r.n_d, "agree": r.agree,
                    "count": r.count, "X_pm": r.X_pm, "cutoff_hz": r.cutoff_hz,
                })
        if n % 10 == 0:
            print(f"  [{n:3d}/93] {p.name}  v={spd.v_kmh:.0f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase2_counts.csv", index=False)

    pd.set_option("display.width", 230)
    LINE = "=" * 104
    print(f"\n{LINE}\nSPEED-ADAPTIVE FILTER — what the cutoff actually does\n{LINE}")
    for v in [32.5, 60, 84, 120, 141]:
        print(f"  v = {v:5.1f} km/h -> peak {samples_per_peak(v):5.1f} samples, "
              f"LPF cutoff {lowpass_cutoff_hz(v):5.1f} Hz")
    print("  (a fixed 10 Hz, as in the reference implementation tuned at 60 km/h,")
    print("   would attenuate the peak by roughly half at 141 km/h)")

    print(f"\n{LINE}\nX-CROSSING vs D-CROSSING — per-sensor agreement\n{LINE}")
    print(f"{'PAD':>4}{'ch':>6}{'head':>6}{'records':>9}{'agree':>8}"
          f"{'X>D (D under)':>15}{'D>X (X under)':>15}{'median X pm':>13}")
    for head, pads in OPERATIONAL.items():
        for pad in pads:
            s = df[df["pad"] == pad]
            if s.empty:
                continue
            print(f"{pad:>4}{CH_OF[pad]:>6}{head:>6}{len(s):>9}"
                  f"{s['agree'].mean():>8.0%}{(s['n_x']>s['n_d']).mean():>15.0%}"
                  f"{(s['n_d']>s['n_x']).mean():>15.0%}{s['X_pm'].median():>13.1f}")

    print(f"\n{LINE}\nHEAD 1 vs HEAD 2 — the quantity the interlocking acts on\n{LINE}")
    print("in-count must equal out-count. Per record, the head count is the")
    print("median across its four sensors (a deterministic vote, no ML).\n")
    hv = (df.groupby(["name", "session", "head"])["count"]
          .median().unstack("head").dropna())
    hv.columns = ["head1", "head2"]
    hv["diff"] = hv["head1"] - hv["head2"]
    n = len(hv)
    print(f"records with both heads reporting : {n}")
    print(f"exact agreement                   : {(hv['diff']==0).mean():.1%}")
    print(f"within +/-1 axle                  : {(hv['diff'].abs()<=1).mean():.1%}")
    print(f"\nHead 1 OVER Head 2 (phantom risk) : {(hv['diff']>0).mean():.1%}")
    print(f"Head 1 UNDER Head 2               : {(hv['diff']<0).mean():.1%}")
    print("\ndisagreement distribution:")
    print(hv["diff"].value_counts().sort_index().to_string())

    print(f"\n{LINE}\nWITHIN-HEAD SENSOR VOTE — 4-of-4 unanimity per record\n{LINE}")
    for head, pads in OPERATIONAL.items():
        s = df[df["head"] == head]
        u = s.groupby("name")["count"].nunique()
        print(f"  head {head}: unanimous across all 4 sensors in "
              f"{(u==1).mean():.0%} of records")
        sp = s.groupby("name")["count"].agg(lambda x: x.max() - x.min())
        print(f"           spread 0/1/2+ axles: "
              f"{(sp==0).mean():.0%} / {(sp==1).mean():.0%} / {(sp>=2).mean():.0%}")

    print(f"\n{LINE}\nDEGRADED SENSORS IN THE COUNTING PATH\n{LINE}")
    print("PAD 6 is in Head 2 and degrades. This is what a counting head loses")
    print("when a sensor fails and nothing flags it.\n")
    print(f"{'PAD':>4}{'sessions 1-4':>15}{'sessions 12-15':>16}   note")
    for pad in OPERATIONAL[2]:
        s = df[df["pad"] == pad]
        e = s[s["session"] <= 4]["agree"].mean()
        l = s[s["session"] >= 12]["agree"].mean()
        note = "degrading control" if pad == 6 else ""
        print(f"{pad:>4}{e:>15.0%}{l:>16.0%}   {note}")

    print(f"\n{LINE}\nWHAT THIS IS NOT\n{LINE}")
    print("This is NOT a counting-accuracy figure. No ground truth exists in this")
    print("delivery: Infrabel's logs are the open item the deck itself flags")
    print("(briefing §8). Head-to-head agreement bounds INTERNAL consistency and")
    print("nothing more — both heads could be wrong in the same direction and")
    print("this statistic would not notice.")
    print("\nNo SIL4 claim is made or implied. The published work states extensive")
    print("further testing is required, and 70 passages is not that.")


if __name__ == "__main__":
    main()

"""Why do weekend and holiday records produce almost no false alarms?

The LOSO count held at 29->30 while the denominator fell 560->464, so weekend
and holiday records contribute near zero. Before item 4 is run, establish
whether that is because the detector generalises or because those records are
simply easier. Three candidate causes, all testable here:

  * traffic falls 84 -> 28 trains/day, so the quiet window between passages is
    longer and cleaner;
  * fewer adjacent-track movements inside the recording window;
  * CVT diversion active on some of those days, so composition differs.

This also probes A-15 on a second axis. That entry logged that the ~0.3 pm
quiet-baseline sigma is the optical/electronic floor and NOT the operational
disturbance floor. If weekend records have a systematically lower disturbance
floor, it is the same finding seen from a different direction and the two
belong in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lightrack import (                                        # noqa: E402
    CONFIG, CH_COLS, list_files, load_file, parse_filename_utc, to_local,
    rolling_median_baseline, estimate_speed, head_signal, quiet_noise_floor,
    PAD_OF_CHANNEL,
)
from lightrack.detect import passage_window                    # noqa: E402
from lightrack.health import HEALTHY                           # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
OUT = Path(CONFIG["paths"]["out_dir"])
REF = Path(CONFIG["paths"]["data_dir"]) / "reference" / "ebp_daily_train_counts.csv"
SESSIONS = {pd.Timestamp(d).date(): i + 1
            for i, d in enumerate(CONFIG["campaign"]["measurement_dates"])}
DIST_SIGMAS = 10.0        # an excursion this far above the quiet floor is an event
SECONDARY_PM = 30.0       # ... and this big is plausibly another train

pd.set_option("display.width", 240)
LINE = "=" * 106

ebp = pd.read_csv(REF, comment="#")
ebp["date"] = pd.to_datetime(ebp["date"]).dt.date
cls = dict(zip(ebp["date"], ebp["day_class"]))
tot = dict(zip(ebp["date"], ebp["total"]))


def main() -> None:
    rows = []
    for n, p in enumerate(list_files(), 1):
        utc = parse_filename_utc(p.name)
        loc = to_local(utc)
        d = loc.date()
        if d not in SESSIONS:
            continue
        y = rolling_median_baseline(load_file(p)[CH_COLS])
        spd = estimate_speed(y)
        if not spd.ok:
            continue
        nrec = len(y)
        hs = head_signal(y, 1)
        w = passage_window(hs, spd.v_kmh)
        if w is None:
            continue
        quiet_mask = np.ones(nrec, dtype=bool)
        quiet_mask[w[0]:w[1]] = False

        for ch in CH_COLS:
            sig = y[ch].to_numpy()
            q = sig[quiet_mask]
            sd = quiet_noise_floor(q)
            # events in the quiet region: contiguous excursions above 10 sigma
            hot = np.abs(q) > DIST_SIGMAS * max(sd, 1e-9)
            n_ev = int(np.count_nonzero(np.diff(hot.astype(int)) == 1))
            rows.append({
                "name": p.name, "date": d, "session": SESSIONS[d],
                "day_class": cls[d], "ebp_total": tot[d],
                "channel": ch, "pad": PAD_OF_CHANNEL[ch],
                "v_kmh": spd.v_kmh,
                "quiet_frac": float(quiet_mask.mean()),
                "passage_s": (w[1] - w[0]) / FS,
                "noise_pm": sd,
                "quiet_max_pm": float(np.abs(q).max()),
                "quiet_p999_pm": float(np.percentile(np.abs(q), 99.9)),
                "n_events": n_ev,
                "secondary": bool(np.abs(q).max() > SECONDARY_PM),
            })
        if n % 15 == 0:
            print(f"  [{n:3d}/93] {p.name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "daytype_disturbance.csv", index=False)

    print(f"\n{LINE}\nRECORD-LEVEL: is the quiet region actually quieter?\n{LINE}")
    rec = df.drop_duplicates("name")
    g = df.groupby("day_class")
    print(f"{'day class':<18}{'records':>9}{'EBP trains/day':>16}{'passage s':>11}"
          f"{'quiet frac':>12}")
    for k, s in g:
        r = s.drop_duplicates("name")
        print(f"{k:<18}{len(r):>9}{r['ebp_total'].median():>16.0f}"
              f"{r['passage_s'].median():>11.2f}{r['quiet_frac'].median():>12.3f}")

    print(f"\n{LINE}\nSENSOR-LEVEL, HEALTHY SENSORS ONLY: the disturbance floor\n{LINE}")
    h = df[df["pad"].isin(HEALTHY)]
    print(f"{'day class':<18}{'noise sigma pm':>16}{'quiet p99.9 pm':>16}"
          f"{'quiet max pm':>14}{'events/record':>15}{'secondary %':>13}")
    for k, s in h.groupby("day_class"):
        print(f"{k:<18}{s['noise_pm'].median():>16.3f}{s['quiet_p999_pm'].median():>16.2f}"
              f"{s['quiet_max_pm'].median():>14.1f}{s['n_events'].mean():>15.2f}"
              f"{100*s['secondary'].mean():>13.1f}")

    nw = h[h["day_class"] == "normal_weekday"]
    we = h[h["day_class"] == "weekend"]
    print(f"\nweekend vs normal weekday, healthy sensors:")
    for col, lab in [("noise_pm", "quiet noise sigma"),
                     ("quiet_p999_pm", "quiet p99.9"),
                     ("quiet_max_pm", "quiet max"),
                     ("n_events", f"events >{DIST_SIGMAS:.0f} sigma")]:
        a, b = nw[col].median(), we[col].median()
        print(f"  {lab:<22} normal {a:8.3f}   weekend {b:8.3f}   "
              f"ratio {b/a if a else float('nan'):.2f}")

    print(f"\n{LINE}\nFEATURE DISPERSION ON HEALTHY SENSORS — the direct test\n{LINE}")
    fe = pd.read_csv(OUT / "phase4_features_scored.csv")
    fe["date"] = fe["name"].map(lambda s: to_local(parse_filename_utc(s)).date())
    fe["day_class"] = fe["date"].map(cls)
    fh = fe[fe["pad"].isin(HEALTHY)]
    print(f"{'feature':<14}{'day class':<18}{'n':>6}{'median':>10}{'MAD sigma':>12}"
          f"{'p1':>9}{'p99':>9}")
    for feat in ("pair_delta", "snr_trend", "neg_over_pos"):
        for k in ("normal_weekday", "weekend", "holiday"):
            s = fh[fh["day_class"] == k][feat].dropna()
            if s.empty:
                continue
            mad = float(np.median(np.abs(s - s.median())) * 1.4826)
            print(f"{feat:<14}{k:<18}{len(s):>6}{s.median():>10.3f}{mad:>12.3f}"
                  f"{s.quantile(0.01):>9.3f}{s.quantile(0.99):>9.3f}")
        print()

    print(f"{LINE}\nVERDICT\n{LINE}")
    a = fh[fh["day_class"] == "normal_weekday"]["pair_delta"].dropna()
    b = fh[fh["day_class"] == "weekend"]["pair_delta"].dropna()
    ma = float(np.median(np.abs(a - a.median())) * 1.4826)
    mb = float(np.median(np.abs(b - b.median())) * 1.4826)
    print(f"pair_delta dispersion on healthy sensors: normal {ma:.3f}, "
          f"weekend {mb:.3f}  (ratio {mb/ma:.2f})")
    print("If weekend dispersion is materially tighter, the weekend records are")
    print("EASIER, and item 4 is a regression check - 'nothing broke' - not a")
    print("generalisation test. Report it that way.")


if __name__ == "__main__":
    main()

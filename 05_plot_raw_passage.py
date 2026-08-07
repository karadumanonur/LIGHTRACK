"""Phase 0 deliverable: one raw train passage, all 12 channels.

Deliberately RAW: what is plotted is the stored quantity, absolute Bragg
wavelength in nm, exactly as it sits on disk. No filter, no baseline removal,
no detection. Each panel is auto-scaled because the twelve gratings sit ~4.6 nm
apart and share no useful common axis.

The right-hand annotation gives the same trace's peak-to-peak in pm so the
amplitudes can be compared against briefing §6.1 without rescaling the plot.
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
from lightrack.config import CONFIG                        # noqa: E402
from lightrack.loader import CH_COLS, load_file, parse_filename_utc, to_local  # noqa: E402

FS = CONFIG["acquisition"]["fs_hz"]
WIN = CONFIG["baseline"]["rolling_window_samples"]
OUT = Path(CONFIG["paths"]["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)

FILE = Path(CONFIG["paths"]["data_dir"]) / "20260515T122333,377Z.txt"

# Provisional grouping established in Phase 0 (scripts 03 + 04).
# Head assignment: high confidence. Within-head PAD number: NOT resolved.
HEAD_OF = {**{c: 3 for c in CH_COLS[0:4]},
           **{c: 2 for c in CH_COLS[4:8]},
           **{c: 1 for c in CH_COLS[8:12]}}
HEAD_COLOUR = {3: "#b2182b", 2: "#2166ac", 1: "#1a7f37"}
PAD_SET = {3: "PAD 9-12", 2: "PAD 5-8", 1: "PAD 1-4"}


def find_passage(df: pd.DataFrame) -> tuple[int, int]:
    """Locate the passage. Baseline removal is used HERE ONLY, to choose the
    x-limits of the plot. The plotted data itself is untouched."""
    x = df[CH_COLS]
    y = ((x - x.rolling(WIN, center=True, min_periods=1).median()) * 1000.0).to_numpy()
    s = np.abs(y) / (np.median(np.abs(y), axis=0, keepdims=True) + 1e-9)
    env = pd.Series(s.mean(axis=1)).rolling(int(0.2 * FS), center=True,
                                            min_periods=1).mean().to_numpy()
    c = int(np.argmax(env))
    half = int(6.0 * FS)
    return max(0, c - half), min(len(env), c + half)


def main() -> None:
    df = load_file(FILE)
    i0, i1 = find_passage(df)
    seg = df.iloc[i0:i1]
    t = seg["time_s"].to_numpy()
    t = t - t[0]

    utc = parse_filename_utc(FILE.name)
    loc = to_local(utc)

    fig, axes = plt.subplots(12, 1, figsize=(13, 19), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.80, top=0.945, bottom=0.045, hspace=0.14)

    for k, ch in enumerate(CH_COLS):
        ax = axes[k]
        v = seg[ch].to_numpy()
        head = HEAD_OF[ch]
        col = HEAD_COLOUR[head]
        ax.plot(t, v, lw=0.6, color=col)
        ax.set_ylabel(f"{ch}\n[nm]", fontsize=8.5, rotation=0, ha="right",
                      va="center", labelpad=26)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.margins(x=0)

        centre = float(np.median(v))
        p2p_pm = (v.max() - v.min()) * 1000.0
        pos_pm = (v.max() - centre) * 1000.0
        neg_pm = (v.min() - centre) * 1000.0
        ax.text(1.012, 0.5,
                f"$\\lambda_B\\approx${centre:.2f} nm\n"
                f"Head {head} ({PAD_SET[head]})\n"
                f"p2p {p2p_pm:.0f} pm\n"
                f"+{pos_pm:.0f} / {neg_pm:.0f} pm",
                transform=ax.transAxes, fontsize=7.6, va="center", ha="left",
                color=col,
                bbox=dict(fc="white", ec=col, lw=0.6, alpha=0.9, pad=2.6))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.locator_params(axis="y", nbins=4)

    axes[-1].set_xlabel("time within record [s]  (raw 500 Hz samples, nothing removed)",
                        fontsize=9.5)

    fig.suptitle(
        f"Lightrack POC1 — single raw train passage, all 12 channels\n"
        f"{FILE.name}   |   {utc:%Y-%m-%d %H:%M:%S} UTC   =   {loc:%H:%M:%S} local (UTC+2)   "
        f"|   window {i0/FS:.1f}–{i1/FS:.1f} s of {len(df)/FS:.1f} s\n"
        f"stored quantity = ABSOLUTE Bragg wavelength [nm], unfiltered, no baseline removal",
        fontsize=11.5, y=0.986)

    fig.text(0.115, 0.012,
             "Head grouping from cross-correlation + 15 m/10 m geometry (confidence: high). "
             "PAD numbers within each head are NOT yet resolved — see ASSUMPTIONS.md A-04.",
             fontsize=8, style="italic", color="#444444")

    png = OUT / "raw_passage_12ch.png"
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")

    # amplitude table for the report
    print(f"\n{'ch':>5} {'centre_nm':>11} {'p2p_pm':>9} {'pos_pm':>8} {'neg_pm':>8}  head")
    for ch in CH_COLS:
        v = seg[ch].to_numpy()
        c = float(np.median(v))
        print(f"{ch:>5} {c:11.4f} {(v.max()-v.min())*1000:9.1f} "
              f"{(v.max()-c)*1000:8.1f} {(v.min()-c)*1000:8.1f}  {HEAD_OF[ch]}")


if __name__ == "__main__":
    main()

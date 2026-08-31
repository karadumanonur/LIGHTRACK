"""Research utility for finding recordings with unusual axle counts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensor12_peak_tracker import (
    butterworth_process,
    detect_peaks,
    load_fbg_file,
    make_peak_table,
    plot_results,
    sampling_rate,
    vmd_process,
)


def timestamp_from_name(path: Path) -> str:
    """Convert names such as 20260514T145753,187Z.txt to an ISO timestamp."""
    try:
        value = datetime.strptime(path.stem, "%Y%m%dT%H%M%S,%fZ")
        return value.isoformat(timespec="milliseconds") + "Z"
    except ValueError:
        return ""


def process_recording(path: Path, args: argparse.Namespace) -> tuple[dict, dict | None]:
    """Return summary information and data needed for an optional plot."""
    time_s, raw_nm = load_fbg_file(path, args.sensor)
    fs_hz = sampling_rate(time_s)

    if args.method == "vmd":
        processed_nm = vmd_process(
            raw_nm,
            fs_hz,
            args.highpass,
            args.lowpass,
            args.vmd_modes,
            args.vmd_alpha,
        )
    else:
        processed_nm = butterworth_process(
            raw_nm, fs_hz, args.highpass, args.lowpass
        )

    indices, signs, prominence = detect_peaks(
        processed_nm,
        fs_hz,
        args.polarity,
        args.prominence_sigma,
        args.prominence_nm,
        args.height_fraction,
        args.minimum_distance,
    )
    peaks = make_peak_table(time_s, raw_nm, processed_nm, indices, signs)

    event_start = float(peaks["elapsed_s"].min()) if len(peaks) else np.nan
    event_end = float(peaks["elapsed_s"].max()) if len(peaks) else np.nan
    result = {
        "file": path.name,
        "recording_utc": timestamp_from_name(path),
        "peak_count": len(peaks),
        "event_start_s": event_start,
        "event_end_s": event_end,
        "event_duration_s": event_end - event_start if len(peaks) > 1 else np.nan,
        "maximum_amplitude_nm": float(np.max(np.abs(processed_nm))),
        "prominence_nm": prominence,
        "sampling_rate_hz": fs_hz,
        "error": "",
    }
    plot_data = {
        "time_s": time_s,
        "raw_nm": raw_nm,
        "processed_nm": processed_nm,
        "peaks": peaks,
    }
    return result, plot_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find FBG recordings whose detected axle/peak count differs."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory containing recordings (default: current directory)",
    )
    parser.add_argument(
        "--pattern", default="20*.txt", help="Recording filename pattern"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N recordings after sorting by filename",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=None,
        help="Expected count; if omitted, use the most common count of at least --min-train-peaks",
    )
    parser.add_argument(
        "--min-train-peaks",
        type=int,
        default=4,
        help="Minimum count treated as a train when inferring the reference (default: 4)",
    )
    parser.add_argument("--sensor", type=int, default=12)
    parser.add_argument("--method", choices=("filter", "vmd"), default="filter")
    parser.add_argument(
        "--polarity", choices=("positive", "negative", "both"), default="positive"
    )
    parser.add_argument("--highpass", type=float, default=0.5)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--prominence-sigma", type=float, default=8.0)
    parser.add_argument("--prominence-nm", type=float, default=None)
    parser.add_argument("--height-fraction", type=float, default=0.06)
    parser.add_argument(
        "--minimum-distance",
        type=float,
        default=0.04,
        help="Minimum time between axle peaks [s] (default: 0.04)",
    )
    parser.add_argument("--vmd-modes", type=int, default=5)
    parser.add_argument("--vmd-alpha", type=float, default=2000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("axle_count_summary.csv"),
        help="Summary CSV path",
    )
    parser.add_argument(
        "--plot-different",
        action="store_true",
        help="Create labelled plots for recordings with a different count",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Create a raw-signal and numbered-peak plot for every recording",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("different_axle_plots")
    )
    parser.add_argument(
        "--all-plot-dir", type=Path, default=Path("all_axle_plots")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    files = sorted(directory.glob(args.pattern))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No files match {args.pattern!r} in {directory}")

    results: list[dict] = []
    cached_plot_data: dict[Path, dict] = {}
    all_plot_dir = args.all_plot_dir.expanduser().resolve()
    if args.plot_all:
        all_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {len(files)} recordings from {directory}")

    for number, path in enumerate(files, start=1):
        try:
            result, plot_data = process_recording(path, args)
            if args.plot_different:
                cached_plot_data[path] = plot_data
            print(f"[{number:>3}/{len(files)}] {path.name}: {result['peak_count']} peaks")
            if args.plot_all:
                plot_results(
                    path,
                    plot_data["time_s"],
                    plot_data["raw_nm"],
                    plot_data["processed_nm"],
                    plot_data["peaks"],
                    all_plot_dir / f"{path.stem}_sensor{args.sensor}_peaks.png",
                    args.sensor,
                    args.method,
                    False,
                )
        except Exception as exc:  # Keep scanning when one recording is malformed.
            result = {
                "file": path.name,
                "recording_utc": timestamp_from_name(path),
                "peak_count": np.nan,
                "event_start_s": np.nan,
                "event_end_s": np.nan,
                "event_duration_s": np.nan,
                "maximum_amplitude_nm": np.nan,
                "prominence_nm": np.nan,
                "sampling_rate_hz": np.nan,
                "error": str(exc),
            }
            print(f"[{number:>3}/{len(files)}] {path.name}: ERROR - {exc}")
        results.append(result)

    summary = pd.DataFrame(results)
    valid_counts = summary.loc[summary["error"].eq(""), "peak_count"].astype(int)
    if args.reference_count is None:
        train_counts = valid_counts[valid_counts >= args.min_train_peaks]
        if train_counts.empty:
            raise RuntimeError(
                "No recording met --min-train-peaks; provide --reference-count explicitly"
            )
        reference_count = Counter(train_counts).most_common(1)[0][0]
        reference_source = "most common detected train count"
    else:
        reference_count = args.reference_count
        reference_source = "command-line reference"

    summary["classification"] = np.select(
        [
            summary["error"].ne(""),
            summary["peak_count"].lt(args.min_train_peaks),
            summary["peak_count"].eq(reference_count),
        ],
        ["error", "no_train_or_uncertain", "reference_count"],
        default="different_count",
    )
    summary["count_difference"] = summary["peak_count"] - reference_count
    summary = summary.sort_values(
        ["classification", "peak_count", "file"], na_position="last"
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, float_format="%.8f")

    different = summary[summary["classification"].eq("different_count")]
    print(f"\nReference count: {reference_count} ({reference_source})")
    print(f"Different-count candidates: {len(different)}")
    if len(different):
        print(
            different[
                ["file", "recording_utc", "peak_count", "count_difference"]
            ].to_string(index=False)
        )
    else:
        print("No different-count train was found with the current settings.")
    print(f"Summary: {output_path}")
    if args.plot_all:
        print(f"All numbered peak plots: {all_plot_dir}")

    if args.plot_different and len(different):
        plot_dir = args.plot_dir.expanduser().resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
        for file_name in different["file"]:
            path = directory / file_name
            data = cached_plot_data[path]
            plot_results(
                path,
                data["time_s"],
                data["raw_nm"],
                data["processed_nm"],
                data["peaks"],
                plot_dir / f"{path.stem}_sensor{args.sensor}_peaks.png",
                args.sensor,
                args.method,
                False,
            )
        print(f"Different-count plots: {plot_dir}")


if __name__ == "__main__":
    main()

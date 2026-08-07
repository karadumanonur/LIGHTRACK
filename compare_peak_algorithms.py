"""Benchmark several FBG axle-peak detectors against independent time labels.

Ground-truth labels are taken only from FBG recordings having exactly one
time-window Infrabel candidate, a complete ACAT counter cycle, and one FBG
frame.  Axle-count agreement is deliberately not used when selecting labels.
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


ALGORITHM_LABELS = {
    "current_robust": "Current: MAD prominence + relative height",
    "relative_height": "Relative height only",
    "std_adaptive": "Standard-deviation adaptive",
    "fixed_absolute": "Fixed absolute threshold",
    "mad_prominence_only": "MAD prominence only",
    "horizontal_hysteresis": "Horizontal threshold hysteresis",
    "combined_hysteresis": "Combined derivative + horizontal hysteresis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing raw FBG text files")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("peak_algorithm_benchmark"))
    parser.add_argument("--detector-dir", type=Path, default=None,
                        help="Directory containing sensor12_peak_tracker.py")
    parser.add_argument("--sensor", type=int, default=12, help="File/data channel (default: 12)")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--max-examples", type=int, default=6)
    parser.add_argument("--fixed-height", type=float, default=0.02)
    parser.add_argument("--fixed-prominence", type=float, default=0.005)
    parser.add_argument("--horizontal-upper-fraction", type=float, default=0.10)
    parser.add_argument("--horizontal-lower-fraction", type=float, default=0.05)
    parser.add_argument("--horizontal-minimum-distance", type=float, default=0.04)
    parser.add_argument("--combined-derivative-up-sigma", type=float, default=6.0)
    parser.add_argument("--combined-derivative-down-sigma", type=float, default=4.0)
    parser.add_argument("--combined-confirmation-window", type=float, default=0.25)
    return parser.parse_args()


def load_detector_functions(detector_dir: Path):
    sys.path.insert(0, str(detector_dir.resolve()))
    from sensor12_peak_tracker import (  # pylint: disable=import-outside-toplevel
        butterworth_process,
        detect_peaks,
        load_fbg_file,
        robust_sigma,
        sampling_rate,
    )
    return butterworth_process, detect_peaks, load_fbg_file, robust_sigma, sampling_rate


def independent_labels(candidate_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(candidate_csv)
    candidates = frame[frame["match_status"].eq("candidate")].copy()
    candidate_counts = candidates.groupby("fbg_file").size()
    unique_files = candidate_counts[candidate_counts.eq(1)].index
    labels = candidates[candidates["fbg_file"].isin(unique_files)].copy()
    labels = labels[
        labels["complete_counter_cycle"].eq(True)
        & labels["split_frame_count"].eq(1)
        & ~labels["fbg_file"].str.contains("|", regex=False)
    ]
    labels["ground_truth_axles"] = labels["infrabel_axles"].astype(int)
    return labels[["fbg_file", "ground_truth_axles", "infrabel_event_id"]].sort_values("fbg_file")


def horizontal_preprocess(raw_signal: np.ndarray, fs_hz: float) -> np.ndarray:
    """Low-pass without high-pass filtering so a stopped-axle plateau remains."""
    sos = butter(4, 40.0, btype="lowpass", fs=fs_hz, output="sos")
    filtered = sosfiltfilt(sos, raw_signal)
    return filtered - np.median(filtered)


def horizontal_hysteresis_details(
    signal: np.ndarray,
    fs_hz: float,
    upper_fraction: float,
    lower_fraction: float,
    minimum_distance_s: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray, float, float]:
    """Schmitt trigger using two horizontal wavelength-change thresholds."""
    if not 0 <= lower_fraction < upper_fraction:
        raise ValueError("Require 0 <= lower_fraction < upper_fraction")
    level = signal - np.percentile(signal, 20.0)
    maximum = max(float(np.max(level)), np.finfo(float).eps)
    upper_threshold = upper_fraction * maximum
    lower_threshold = lower_fraction * maximum
    minimum_distance = max(1, int(round(minimum_distance_s * fs_hz)))

    peaks: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    active_start: int | None = None
    rearm_after = 0
    for index, value in enumerate(level):
        if active_start is None:
            if index >= rearm_after and value >= upper_threshold:
                active_start = index
                starts.append(index)
        elif value <= lower_threshold:
            peak = active_start + int(np.argmax(level[active_start : index + 1]))
            peaks.append(peak)
            ends.append(index)
            active_start = None
            rearm_after = index + minimum_distance

    remains_occupied = active_start is not None
    if remains_occupied:
        peak = active_start + int(np.argmax(level[active_start:]))
        peaks.append(peak)
    return (
        np.asarray(peaks, dtype=int),
        np.asarray(starts, dtype=int),
        np.asarray(ends, dtype=int),
        remains_occupied,
        level,
        upper_threshold,
        lower_threshold,
    )


def combined_hysteresis_details(
    signal: np.ndarray,
    fs_hz: float,
    robust_sigma,
    upper_fraction: float,
    lower_fraction: float,
    derivative_up_sigma: float,
    derivative_down_sigma: float,
    confirmation_window_s: float,
    minimum_distance_s: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray, float, float, float, float]:
    """Confirm transitions by derivative while horizontal levels retain state."""
    if not 0 <= lower_fraction < upper_fraction:
        raise ValueError("Require 0 <= lower_fraction < upper_fraction")
    (
        horizontal_peaks,
        horizontal_starts,
        horizontal_ends,
        horizontal_remains_occupied,
        level,
        upper_threshold,
        lower_threshold,
    ) = horizontal_hysteresis_details(
        signal,
        fs_hz,
        upper_fraction,
        lower_fraction,
        minimum_distance_s,
    )
    slope = np.gradient(level) * fs_hz
    slope_sigma = robust_sigma(slope)
    derivative_up_threshold = max(derivative_up_sigma * slope_sigma, np.finfo(float).eps)
    derivative_down_threshold = -max(derivative_down_sigma * slope_sigma, np.finfo(float).eps)
    confirmation_samples = max(1, int(round(confirmation_window_s * fs_hz)))
    accepted: list[bool] = []
    derivative_starts: list[int] = []
    for start in horizontal_starts:
        left = max(0, int(start) - confirmation_samples)
        window = slope[left : int(start) + 1]
        local = int(np.argmax(window))
        accepted.append(bool(window[local] >= derivative_up_threshold))
        derivative_starts.append(left + local)
    accepted_mask = np.asarray(accepted, dtype=bool)
    peaks = horizontal_peaks[accepted_mask]
    starts = np.asarray(derivative_starts, dtype=int)[accepted_mask]

    derivative_ends: list[int] = []
    completed_acceptance = accepted_mask[: len(horizontal_ends)]
    for end, keep in zip(horizontal_ends, completed_acceptance):
        if not keep:
            continue
        left = max(0, int(end) - confirmation_samples)
        window = slope[left : int(end) + 1]
        local = int(np.argmin(window))
        derivative_ends.append(
            left + local if window[local] <= derivative_down_threshold else int(end)
        )
    ends = np.asarray(derivative_ends, dtype=int)
    remains_occupied = bool(
        horizontal_remains_occupied
        and len(accepted_mask)
        and accepted_mask[-1]
    )
    return (
        np.asarray(peaks, dtype=int),
        np.asarray(starts, dtype=int),
        ends,
        remains_occupied,
        level,
        slope,
        upper_threshold,
        lower_threshold,
        derivative_up_threshold,
        derivative_down_threshold,
    )


def run_detectors(
    signal: np.ndarray,
    horizontal_signal: np.ndarray,
    fs_hz: float,
    detect_peaks,
    robust_sigma,
    fixed_height: float,
    fixed_prominence: float,
    horizontal_upper_fraction: float,
    horizontal_lower_fraction: float,
    horizontal_minimum_distance: float,
    combined_derivative_up_sigma: float,
    combined_derivative_down_sigma: float,
    combined_confirmation_window: float,
) -> dict[str, np.ndarray]:
    distance = max(1, int(round(0.04 * fs_hz)))
    mad_sigma = robust_sigma(signal)
    ordinary_sigma = float(np.std(signal))
    maximum = max(float(np.max(signal)), np.finfo(float).eps)

    return {
        "current_robust": detect_peaks(
            signal, fs_hz, "positive", 8.0, None, 0.06, 0.04
        )[0],
        "relative_height": find_peaks(
            signal, height=0.06 * maximum, distance=distance
        )[0],
        "std_adaptive": find_peaks(
            signal,
            height=3.0 * ordinary_sigma,
            prominence=3.0 * ordinary_sigma,
            distance=distance,
        )[0],
        "fixed_absolute": find_peaks(
            signal,
            height=fixed_height,
            prominence=fixed_prominence,
            distance=distance,
        )[0],
        "mad_prominence_only": find_peaks(
            signal, prominence=8.0 * mad_sigma, distance=distance
        )[0],
        "horizontal_hysteresis": horizontal_hysteresis_details(
            horizontal_signal,
            fs_hz,
            horizontal_upper_fraction,
            horizontal_lower_fraction,
            horizontal_minimum_distance,
        )[0],
        "combined_hysteresis": combined_hysteresis_details(
            horizontal_signal,
            fs_hz,
            robust_sigma,
            horizontal_upper_fraction,
            horizontal_lower_fraction,
            combined_derivative_up_sigma,
            combined_derivative_down_sigma,
            combined_confirmation_window,
            horizontal_minimum_distance,
        )[0],
    }


def score_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algorithm, scenario), group in results.groupby(["algorithm", "scenario"]):
        error = group["error"].to_numpy(float)
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "scenario": scenario,
                "recordings": len(group),
                "exact_accuracy_percent": 100.0 * float(np.mean(error == 0)),
                "mae_axles": float(np.mean(np.abs(error))),
                "rmse_axles": float(np.sqrt(np.mean(error**2))),
                "bias_axles": float(np.mean(error)),
                "runtime_ms": 1000.0 * float(group["runtime_s"].sum()),
            }
        )
    metrics = pd.DataFrame(rows)
    scenario_order = ["clean", "snr_30_db", "snr_20_db", "snr_15_db", "amplitude_0.5x", "amplitude_2.0x"]
    metrics["scenario_order"] = metrics["scenario"].map({v: i for i, v in enumerate(scenario_order)})
    return metrics.sort_values(["scenario_order", "algorithm"]).drop(columns="scenario_order")


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    scenario_order = ["clean", "snr_30_db", "snr_20_db", "snr_15_db", "amplitude_0.5x", "amplitude_2.0x"]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for algorithm, group in metrics.groupby("algorithm"):
        indexed = group.set_index("scenario").reindex(scenario_order)
        axes[0].plot(scenario_order, indexed["exact_accuracy_percent"], marker="o", label=ALGORITHM_LABELS[algorithm])
        axes[1].plot(scenario_order, indexed["mae_axles"], marker="o", label=ALGORITHM_LABELS[algorithm])
    axes[0].set_ylabel("Exact axle-count accuracy [%]")
    axes[0].set_ylim(bottom=0, top=102)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, ncol=2)
    axes[1].set_ylabel("Mean absolute error [axles]")
    axes[1].set_xlabel("Test condition")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("FBG peak-detector comparison on independent Infrabel time labels")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_example(
    file_name: str,
    truth: int,
    elapsed: np.ndarray,
    signal: np.ndarray,
    detections: dict[str, np.ndarray],
    output: Path,
) -> None:
    all_indices = np.concatenate(list(detections.values())) if detections else np.array([], dtype=int)
    if len(all_indices):
        left = max(0.0, float(elapsed[all_indices].min()) - 0.6)
        right = min(float(elapsed[-1]), float(elapsed[all_indices].max()) + 0.6)
    else:
        left, right = float(elapsed[0]), float(elapsed[-1])
    figure, axes = plt.subplots(len(detections), 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for axis, (algorithm, indices) in zip(axes, detections.items()):
        axis.plot(elapsed, signal, color="black", linewidth=0.75)
        axis.scatter(elapsed[indices], signal[indices], s=18, color="tab:red", zorder=3)
        axis.set_xlim(left, right)
        axis.set_ylabel(ALGORITHM_LABELS[algorithm], fontsize=8)
        axis.grid(alpha=0.2)
        axis.text(0.995, 0.9, f"count={len(indices)}", transform=axis.transAxes, ha="right", va="top")
    axes[-1].set_xlabel("Elapsed time [s]")
    figure.suptitle(f"{file_name} — independent Infrabel count: {truth}")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_stopped_axle_demo(
    output: Path,
    detect_peaks,
    robust_sigma,
    butterworth_process,
    upper_fraction: float,
    lower_fraction: float,
    minimum_distance_s: float,
    combined_derivative_up_sigma: float,
    combined_derivative_down_sigma: float,
    combined_confirmation_window: float,
    seed: int,
) -> None:
    """Show why horizontal hysteresis is useful when an axle remains on a pad."""
    fs_hz = 500.0
    time_s = np.arange(0.0, 12.0, 1.0 / fs_hz)
    rng = np.random.default_rng(seed + 1)
    edge_width = 0.035
    load = 0.08 * (
        1.0 / (1.0 + np.exp(-(time_s - 3.0) / edge_width))
        - 1.0 / (1.0 + np.exp(-(time_s - 9.0) / edge_width))
    )
    raw = 1515.0 + 0.0003 * time_s + load + rng.normal(0.0, 0.00035, len(time_s))
    current_signal = butterworth_process(raw, fs_hz, 0.5, 40.0)
    current_indices = detect_peaks(
        current_signal, fs_hz, "positive", 8.0, None, 0.06, 0.04
    )[0]
    horizontal_signal = horizontal_preprocess(raw, fs_hz)
    (
        horizontal_indices,
        starts,
        ends,
        remains_occupied,
        level,
        upper_threshold,
        lower_threshold,
    ) = horizontal_hysteresis_details(
        horizontal_signal, fs_hz, upper_fraction, lower_fraction, minimum_distance_s
    )
    (
        combined_indices,
        combined_starts,
        combined_ends,
        combined_remains_occupied,
        _,
        combined_slope,
        _,
        _,
        derivative_up_threshold,
        derivative_down_threshold,
    ) = combined_hysteresis_details(
        horizontal_signal,
        fs_hz,
        robust_sigma,
        upper_fraction,
        lower_fraction,
        combined_derivative_up_sigma,
        combined_derivative_down_sigma,
        combined_confirmation_window,
        minimum_distance_s,
    )

    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    axes[0].plot(time_s, raw - np.median(raw[:500]), linewidth=0.9)
    axes[0].axvspan(3.0, 9.0, color="tab:orange", alpha=0.15, label="Axle remains on sensor")
    axes[0].set_ylabel("Wavelength change [nm]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time_s, current_signal, color="black", linewidth=0.9)
    axes[1].scatter(time_s[current_indices], current_signal[current_indices], color="tab:red", s=28)
    axes[1].set_ylabel("Current band-pass signal")
    axes[1].set_title(f"Current detector: {len(current_indices)} positive peak(s), but no occupancy state")

    axes[2].plot(time_s, level, color="tab:blue", linewidth=0.8, label="Low-pass wavelength change")
    axes[2].axhline(upper_threshold, color="tab:green", linewidth=1.1, label="Upper horizontal threshold")
    axes[2].axhline(lower_threshold, color="tab:red", linewidth=1.1, label="Lower horizontal threshold")
    for start in starts:
        axes[2].axvline(time_s[start], color="tab:green", alpha=0.7)
    for end in ends:
        axes[2].axvline(time_s[end], color="tab:red", alpha=0.7)
    axes[2].set_ylabel("Wavelength change [nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title(
        f"Horizontal hysteresis: {len(horizontal_indices)} event(s); "
        f"occupied at file end={remains_occupied}"
    )
    axes[2].legend(loc="upper right", ncol=3)
    axes[3].plot(time_s, combined_slope, color="tab:blue", linewidth=0.8, label="Derivative")
    axes[3].axhline(derivative_up_threshold, color="tab:green", linewidth=1.1, label="Derivative up threshold")
    axes[3].axhline(derivative_down_threshold, color="tab:red", linewidth=1.1, label="Derivative down threshold")
    for start in combined_starts:
        axes[3].axvline(time_s[start], color="tab:green", alpha=0.7)
    for end in combined_ends:
        axes[3].axvline(time_s[end], color="tab:red", alpha=0.7)
    axes[3].set_ylabel("dλ/dt [nm/s]")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title(
        f"Combined: {len(combined_indices)} event(s); "
        f"occupied at file end={combined_remains_occupied}"
    )
    axes[3].legend(loc="upper right", ncol=3)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(metrics: pd.DataFrame, examples: list[Path], output: Path, label_count: int) -> None:
    clean = metrics[metrics["scenario"].eq("clean")].sort_values(
        ["exact_accuracy_percent", "mae_axles"], ascending=[False, True]
    )
    average = metrics.groupby(["algorithm", "algorithm_label"], as_index=False).agg(
        average_exact_accuracy_percent=("exact_accuracy_percent", "mean"),
        average_mae_axles=("mae_axles", "mean"),
    ).sort_values(["average_exact_accuracy_percent", "average_mae_axles"], ascending=[False, True])
    table = clean.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    average_table = average.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    images = "\n".join(
        f'<h3>{html.escape(path.stem)}</h3><img src="{html.escape(path.name)}" style="max-width:100%;height:auto">'
        for path in examples
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>FBG peak algorithm benchmark</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ccc;text-align:right}}
th:nth-child(2),td:nth-child(2){{text-align:left}}img{{margin-bottom:2rem}}</style></head><body>
<h1>FBG peak algorithm benchmark</h1>
<p>Evaluation set: {label_count} recordings having exactly one time-selected, complete-cycle Infrabel label. Axle agreement was not used to select labels.</p>
<img src="algorithm_comparison.png" style="max-width:100%;height:auto" alt="Accuracy and error comparison">
<h2>Stopped-axle simulation</h2>
<img src="stopped_axle_comparison.png" style="max-width:100%;height:auto" alt="Current and horizontal-threshold detector response to a stopped axle">
<h2>Clean-signal results</h2>{table}
<h2>Average across clean, noise, and amplitude tests</h2>{average_table}
<h2>Selected disagreements</h2>{images}
</body></html>"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_dir = (args.detector_dir or Path(__file__).resolve().parents[1] / "fbg_matching_update").resolve()
    butterworth_process, detect_peaks, load_fbg_file, robust_sigma, sampling_rate = load_detector_functions(detector_dir)
    labels = independent_labels(args.candidate_csv.expanduser().resolve())
    labels.to_csv(output_dir / "benchmark_ground_truth.csv", index=False)

    rng = np.random.default_rng(args.seed)
    result_rows: list[dict] = []
    clean_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]] = {}
    scenarios = {
        "clean": (None, 1.0),
        "snr_30_db": (30.0, 1.0),
        "snr_20_db": (20.0, 1.0),
        "snr_15_db": (15.0, 1.0),
        "amplitude_0.5x": (None, 0.5),
        "amplitude_2.0x": (None, 2.0),
    }

    for number, label in enumerate(labels.itertuples(index=False), 1):
        path = data_dir / label.fbg_file
        time_s, raw = load_fbg_file(path, args.sensor)
        fs_hz = sampling_rate(time_s)
        base = butterworth_process(raw, fs_hz, 0.5, 40.0)
        horizontal_base = horizontal_preprocess(raw, fs_hz)
        signal_rms = float(np.sqrt(np.mean(base**2)))
        horizontal_rms = float(np.sqrt(np.mean(horizontal_base**2)))
        shared_noise = rng.normal(0.0, 1.0, len(base))
        for scenario, (snr_db, scale) in scenarios.items():
            signal = base * scale
            horizontal_signal = horizontal_base * scale
            if snr_db is not None:
                noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
                signal = signal + shared_noise * noise_rms
                horizontal_noise_rms = horizontal_rms / (10.0 ** (snr_db / 20.0))
                horizontal_signal = horizontal_signal + shared_noise * horizontal_noise_rms
            started = time.perf_counter()
            detections = run_detectors(
                signal, horizontal_signal, fs_hz, detect_peaks, robust_sigma,
                args.fixed_height, args.fixed_prominence,
                args.horizontal_upper_fraction, args.horizontal_lower_fraction,
                args.horizontal_minimum_distance,
                args.combined_derivative_up_sigma,
                args.combined_derivative_down_sigma,
                args.combined_confirmation_window,
            )
            total_runtime = time.perf_counter() - started
            per_algorithm_runtime = total_runtime / len(detections)
            for algorithm, indices in detections.items():
                count = len(indices)
                result_rows.append(
                    {
                        "file": label.fbg_file,
                        "ground_truth_axles": label.ground_truth_axles,
                        "infrabel_event_id": label.infrabel_event_id,
                        "scenario": scenario,
                        "algorithm": algorithm,
                        "algorithm_label": ALGORITHM_LABELS[algorithm],
                        "detected_axles": count,
                        "error": count - label.ground_truth_axles,
                        "exact": count == label.ground_truth_axles,
                        "runtime_s": per_algorithm_runtime,
                    }
                )
            if scenario == "clean":
                clean_cache[label.fbg_file] = (time_s - time_s[0], signal, detections, label.ground_truth_axles)
        print(f"[{number:>2}/{len(labels)}] {label.fbg_file}")

    results = pd.DataFrame(result_rows)
    metrics = score_results(results)
    results.to_csv(output_dir / "algorithm_file_results.csv", index=False, float_format="%.8f")
    metrics.to_csv(output_dir / "algorithm_comparison_metrics.csv", index=False, float_format="%.8f")
    plot_metrics(metrics, output_dir / "algorithm_comparison.png")
    plot_stopped_axle_demo(
        output_dir / "stopped_axle_comparison.png",
        detect_peaks,
        robust_sigma,
        butterworth_process,
        args.horizontal_upper_fraction,
        args.horizontal_lower_fraction,
        args.horizontal_minimum_distance,
        args.combined_derivative_up_sigma,
        args.combined_derivative_down_sigma,
        args.combined_confirmation_window,
        args.seed,
    )

    clean_results = results[results["scenario"].eq("clean")]
    spreads = clean_results.groupby("file")["detected_axles"].agg(lambda x: x.max() - x.min())
    current_exact = clean_results[clean_results["algorithm"].eq("current_robust") & clean_results["exact"]]
    preferred = list(current_exact.sort_values("file")["file"])
    ordered_examples = list(spreads.sort_values(ascending=False).index)
    ordered_examples = [name for name in ordered_examples if name in preferred] + ordered_examples
    chosen = list(dict.fromkeys(ordered_examples))[: args.max_examples]
    example_paths = []
    for file_name in chosen:
        elapsed, signal, detections, truth = clean_cache[file_name]
        path = output_dir / f"{Path(file_name).stem}_algorithm_overlay.png"
        plot_example(file_name, truth, elapsed, signal, detections, path)
        example_paths.append(path)

    write_report(metrics, example_paths, output_dir / "index.html", len(labels))
    print(f"Labels: {len(labels)}")
    print(metrics[metrics['scenario'].eq('clean')][
        ['algorithm_label', 'exact_accuracy_percent', 'mae_axles', 'rmse_axles']
    ].sort_values(['exact_accuracy_percent', 'mae_axles'], ascending=[False, True]).to_string(index=False))
    print(f"Report: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()

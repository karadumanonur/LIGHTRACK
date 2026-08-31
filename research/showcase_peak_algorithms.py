"""Research utility showing how every axle detector reacts to one FBG file."""

from __future__ import annotations

import argparse
import html
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("peak_algorithm_showcases"))
    parser.add_argument("--detector-dir", type=Path, default=None)
    parser.add_argument("--benchmark-script", type=Path, default=None)
    parser.add_argument("--sensor", type=int, default=12, help="File/data channel (default: 12)")
    parser.add_argument("--expected-axles", type=int, default=None)
    parser.add_argument("--fixed-height", type=float, default=0.02)
    parser.add_argument("--fixed-prominence", type=float, default=0.005)
    parser.add_argument("--horizontal-upper-fraction", type=float, default=0.10)
    parser.add_argument("--horizontal-lower-fraction", type=float, default=0.05)
    parser.add_argument("--horizontal-minimum-distance", type=float, default=0.04)
    parser.add_argument("--combined-derivative-up-sigma", type=float, default=6.0)
    parser.add_argument("--combined-derivative-down-sigma", type=float, default=4.0)
    parser.add_argument("--combined-confirmation-window", type=float, default=0.25)
    return parser.parse_args()


def load_benchmark_module(path: Path):
    specification = importlib.util.spec_from_file_location("peak_benchmark", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load benchmark functions from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def event_limits(elapsed: np.ndarray, detections: dict[str, np.ndarray]) -> tuple[float, float]:
    nonempty = [indices for indices in detections.values() if len(indices)]
    if not nonempty:
        return float(elapsed[0]), float(elapsed[-1])
    indices = np.concatenate(nonempty)
    start = float(elapsed[indices].min())
    end = float(elapsed[indices].max())
    padding = max(0.6, 0.08 * max(end - start, 1.0))
    return max(float(elapsed[0]), start - padding), min(float(elapsed[-1]), end + padding)


def algorithm_plot_data(
    algorithm: str,
    bandpass_signal: np.ndarray,
    horizontal_signal: np.ndarray,
    robust_sigma,
    fixed_height: float,
    fixed_prominence: float,
    upper_fraction: float,
    lower_fraction: float,
) -> tuple[np.ndarray, list[tuple[float, str, str]], str]:
    if algorithm in {"horizontal_hysteresis", "combined_hysteresis"}:
        level = horizontal_signal - np.percentile(horizontal_signal, 20.0)
        maximum = max(float(np.max(level)), np.finfo(float).eps)
        return level, [
            (upper_fraction * maximum, "upper horizontal threshold", "tab:green"),
            (lower_fraction * maximum, "lower horizontal threshold", "tab:red"),
        ], "Low-pass wavelength change [nm]"

    maximum = max(float(np.max(bandpass_signal)), np.finfo(float).eps)
    mad_prominence = 8.0 * robust_sigma(bandpass_signal)
    ordinary_sigma = float(np.std(bandpass_signal))
    thresholds: list[tuple[float, str, str]] = []
    if algorithm == "current_robust":
        thresholds.append((max(mad_prominence, 0.06 * maximum), "height floor", "tab:green"))
    elif algorithm == "relative_height":
        thresholds.append((0.06 * maximum, "6% relative height", "tab:green"))
    elif algorithm == "std_adaptive":
        thresholds.append((3.0 * ordinary_sigma, "3σ height", "tab:green"))
    elif algorithm == "fixed_absolute":
        thresholds.extend([
            (fixed_height, "fixed height", "tab:green"),
            (fixed_prominence, "fixed prominence value", "tab:orange"),
        ])
    return bandpass_signal, thresholds, "Band-pass amplitude [nm]"


def draw_algorithm_axis(
    axis,
    elapsed: np.ndarray,
    signal: np.ndarray,
    indices: np.ndarray,
    thresholds: list[tuple[float, str, str]],
    title: str,
    ylabel: str,
    limits: tuple[float, float],
) -> None:
    axis.plot(elapsed, signal, color="black", linewidth=0.8)
    axis.scatter(elapsed[indices], signal[indices], color="tab:red", s=25, zorder=3, label="detections")
    for value, label, color in thresholds:
        axis.axhline(value, color=color, linewidth=1.0, alpha=0.85, label=label)
    axis.set_xlim(*limits)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title} — {len(indices)} detections")
    axis.grid(alpha=0.2)
    if thresholds or len(indices):
        axis.legend(loc="upper right", fontsize=8, ncol=min(3, 1 + len(thresholds)))


def write_index(
    output_dir: Path,
    input_file: Path,
    expected: int | None,
    counts: pd.DataFrame,
    individual_images: list[tuple[str, Path]],
) -> None:
    expected_text = "not supplied" if expected is None else str(expected)
    table = counts.to_html(index=False, float_format=lambda value: f"{value:.3f}")
    sections = "\n".join(
        f'<h2>{html.escape(label)}</h2><img src="{html.escape(path.name)}" alt="{html.escape(label)} result">'
        for label, path in individual_images
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Peak algorithm showcase</title>
<style>body{{font-family:Arial,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ccc;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto;margin-bottom:2rem}}</style></head><body>
<h1>{html.escape(input_file.name)}</h1><p>Expected axle count: {expected_text}</p>
{table}<h2>All algorithms</h2><img src="all_algorithms_comparison.png" alt="All algorithm detections">
{sections}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = args.input_file.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    detector_dir = (args.detector_dir or script_dir.parent).resolve()
    benchmark_script = (args.benchmark_script or script_dir / "compare_peak_algorithms.py").resolve()
    sys.path.insert(0, str(detector_dir))
    from sensor12_peak_tracker import (  # pylint: disable=import-outside-toplevel
        butterworth_process,
        detect_peaks,
        load_fbg_file,
        robust_sigma,
        sampling_rate,
    )

    benchmark = load_benchmark_module(benchmark_script)
    output_dir = args.output_root.expanduser().resolve() / input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    time_s, raw = load_fbg_file(input_file, args.sensor)
    elapsed = time_s - time_s[0]
    fs_hz = sampling_rate(time_s)
    bandpass = butterworth_process(raw, fs_hz, 0.5, 40.0)
    horizontal = benchmark.horizontal_preprocess(raw, fs_hz)
    detections = benchmark.run_detectors(
        bandpass,
        horizontal,
        fs_hz,
        detect_peaks,
        robust_sigma,
        args.fixed_height,
        args.fixed_prominence,
        args.horizontal_upper_fraction,
        args.horizontal_lower_fraction,
        args.horizontal_minimum_distance,
        args.combined_derivative_up_sigma,
        args.combined_derivative_down_sigma,
        args.combined_confirmation_window,
    )
    limits = event_limits(elapsed, detections)

    rows = []
    for algorithm, indices in detections.items():
        detected = len(indices)
        rows.append({
            "algorithm": algorithm,
            "algorithm_label": benchmark.ALGORITHM_LABELS[algorithm],
            "detected_axles": detected,
            "expected_axles": args.expected_axles,
            "error": np.nan if args.expected_axles is None else detected - args.expected_axles,
            "exact": False if args.expected_axles is None else detected == args.expected_axles,
        })
    counts = pd.DataFrame(rows).sort_values(["detected_axles", "algorithm_label"])
    counts.to_csv(output_dir / "algorithm_counts.csv", index=False)

    figure, axes = plt.subplots(len(detections) + 1, 1, figsize=(15, 3.0 * (len(detections) + 1)), sharex=False, constrained_layout=True)
    axes[0].plot(elapsed, raw, color="tab:blue", linewidth=0.75)
    axes[0].set_title(f"Raw file channel {args.sensor}: {input_file.name}")
    axes[0].set_ylabel("Wavelength [nm]")
    axes[0].grid(alpha=0.2)
    for axis, (algorithm, indices) in zip(axes[1:], detections.items()):
        signal, thresholds, ylabel = algorithm_plot_data(
            algorithm, bandpass, horizontal, robust_sigma,
            args.fixed_height, args.fixed_prominence,
            args.horizontal_upper_fraction, args.horizontal_lower_fraction,
        )
        draw_algorithm_axis(
            axis, elapsed, signal, indices, thresholds,
            benchmark.ALGORITHM_LABELS[algorithm], ylabel, limits,
        )
    axes[-1].set_xlabel("Elapsed time [s]")
    figure.savefig(output_dir / "all_algorithms_comparison.png", dpi=175)
    plt.close(figure)

    individual_images: list[tuple[str, Path]] = []
    for algorithm, indices in detections.items():
        signal, thresholds, ylabel = algorithm_plot_data(
            algorithm, bandpass, horizontal, robust_sigma,
            args.fixed_height, args.fixed_prominence,
            args.horizontal_upper_fraction, args.horizontal_lower_fraction,
        )
        panel_count = 3 if algorithm == "combined_hysteresis" else 2
        figure, axes = plt.subplots(panel_count, 1, figsize=(15, 4 * panel_count), constrained_layout=True)
        axes[0].plot(elapsed, raw, color="tab:blue", linewidth=0.75)
        axes[0].set_ylabel("Raw wavelength [nm]")
        axes[0].set_title(input_file.name)
        axes[0].grid(alpha=0.2)
        draw_algorithm_axis(
            axes[1], elapsed, signal, indices, thresholds,
            benchmark.ALGORITHM_LABELS[algorithm], ylabel, limits,
        )
        if algorithm == "combined_hysteresis":
            details = benchmark.combined_hysteresis_details(
                horizontal,
                fs_hz,
                robust_sigma,
                args.horizontal_upper_fraction,
                args.horizontal_lower_fraction,
                args.combined_derivative_up_sigma,
                args.combined_derivative_down_sigma,
                args.combined_confirmation_window,
                args.horizontal_minimum_distance,
            )
            slope = details[5]
            derivative_up_threshold = details[8]
            derivative_down_threshold = details[9]
            axes[2].plot(elapsed, slope, color="tab:blue", linewidth=0.8, label="derivative")
            axes[2].axhline(derivative_up_threshold, color="tab:green", label="derivative up threshold")
            axes[2].axhline(derivative_down_threshold, color="tab:red", label="derivative down threshold")
            axes[2].set_xlim(*limits)
            axes[2].set_ylabel("dλ/dt [nm/s]")
            axes[2].set_title("Derivative transition confirmation")
            axes[2].grid(alpha=0.2)
            axes[2].legend(loc="upper right")
        axes[-1].set_xlabel("Elapsed time [s]")
        image_path = output_dir / f"{algorithm}.png"
        figure.savefig(image_path, dpi=175)
        plt.close(figure)
        individual_images.append((benchmark.ALGORITHM_LABELS[algorithm], image_path))

    write_index(output_dir, input_file, args.expected_axles, counts, individual_images)
    print(counts[["algorithm_label", "detected_axles", "expected_axles", "error", "exact"]].to_string(index=False))
    print(f"Showcase: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()

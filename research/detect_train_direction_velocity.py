"""Research utility for estimating direction and velocity from separated FBGs.

Physical layout (positive coordinate points left in the supplied drawing):

* physical FBG 1--4: counting zone, x = 0 m (most reliable zone)
* physical FBG 5--8: decounting zone, x = 15 m
* physical FBG 9--12: test zone, x = 25 m

Dataset wavelength channels are reversed: physical FBG p is stored as
``Wavelength (13 - p)``. Each zone is represented by the median of four
normalized, smoothed signal envelopes. Cross-correlation estimates the delay
between zones, and velocity is distance / signed delay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, correlate, correlation_lags, detrend, hilbert, sosfiltfilt


ZONE_DEFINITIONS = {
    "counting": {"physical_sensors": (1, 2, 3, 4), "position_m": 0.0},
    "decounting": {"physical_sensors": (5, 6, 7, 8), "position_m": 15.0},
    "test": {"physical_sensors": (9, 10, 11, 12), "position_m": 25.0},
}


def robust_sigma(values: np.ndarray, axis=0) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(values - median), axis=axis)


def physical_to_data_channel(physical_sensor: int) -> int:
    """Map physical sensor 1 -> wavelength 12, ..., sensor 12 -> wavelength 1."""
    if not 1 <= physical_sensor <= 12:
        raise ValueError("Physical FBG number must be between 1 and 12")
    return 13 - physical_sensor


def load_all_fbg_channels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        usecols=list(range(13)),
        decimal=",",
        engine="c",
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.sort_values(0).drop_duplicates(0)
    if len(frame) < 100:
        raise ValueError(f"{path} does not contain enough valid FBG samples")
    return frame.iloc[:, 0].to_numpy(float), frame.iloc[:, 1:13].to_numpy(float)


def sampling_rate(time_s: np.ndarray) -> float:
    steps = np.diff(time_s)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if not len(steps):
        raise ValueError("Cannot determine the sampling rate")
    return float(1.0 / np.median(steps))


def process_all_channels(
    wavelengths_nm: np.ndarray,
    fs_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
) -> np.ndarray:
    nyquist = fs_hz / 2.0
    if not 0 < highpass_hz < lowpass_hz < nyquist:
        raise ValueError(f"Filter limits must be below Nyquist ({nyquist:.2f} Hz)")
    centered = detrend(wavelengths_nm, axis=0, type="linear")
    sos = butter(
        4,
        [highpass_hz, lowpass_hz],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )
    return sosfiltfilt(sos, centered, axis=0)


def zone_envelope(
    processed: np.ndarray,
    physical_sensors: tuple[int, ...],
    fs_hz: float,
    smoothing_s: float,
) -> np.ndarray:
    # Wavelength array index 0 is data channel 1.
    indices = [physical_to_data_channel(sensor) - 1 for sensor in physical_sensors]
    signals = processed[:, indices]
    scales = robust_sigma(signals, axis=0)
    scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
    normalized = signals / scales
    envelopes = np.abs(hilbert(normalized, axis=0))
    window = max(1, int(round(smoothing_s * fs_hz)))
    envelopes = uniform_filter1d(envelopes, size=window, axis=0, mode="nearest")
    # Median aggregation prevents one unreliable sensor from dominating a zone.
    composite = np.median(envelopes, axis=1)
    baseline = np.median(composite)
    return np.maximum(composite - baseline, 0.0)


def crop_to_train_event(
    envelopes: dict[str, np.ndarray], fs_hz: float, padding_s: float
) -> slice:
    combined = np.max(np.column_stack(list(envelopes.values())), axis=1)
    edge = min(int(round(fs_hz)), max(0, len(combined) // 10))
    usable = combined.copy()
    if edge:
        usable[:edge] = 0
        usable[-edge:] = 0
    noise = float(robust_sigma(usable, axis=0))
    threshold = np.median(usable) + 6.0 * noise
    active = np.flatnonzero(usable > threshold)
    if not len(active):
        return slice(0, len(combined))
    padding = int(round(padding_s * fs_hz))
    return slice(max(0, active[0] - padding), min(len(combined), active[-1] + padding + 1))


def correlation_coefficient_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson correlation when b is shifted by lag samples relative to a."""
    if lag > 0:
        a_overlap, b_overlap = a[:-lag], b[lag:]
    elif lag < 0:
        a_overlap, b_overlap = a[-lag:], b[:lag]
    else:
        a_overlap, b_overlap = a, b
    if len(a_overlap) < 10 or np.std(a_overlap) == 0 or np.std(b_overlap) == 0:
        return np.nan
    return float(np.corrcoef(a_overlap, b_overlap)[0, 1])


def estimate_pair_delay(
    envelope_a: np.ndarray,
    envelope_b: np.ndarray,
    fs_hz: float,
    distance_m: float,
    minimum_speed_kmh: float,
    maximum_speed_kmh: float,
) -> tuple[float, float]:
    """Return signed delay t_B-t_A and its envelope correlation coefficient."""
    a = envelope_a - np.mean(envelope_a)
    b = envelope_b - np.mean(envelope_b)
    a_std, b_std = np.std(a), np.std(b)
    if a_std == 0 or b_std == 0:
        raise ValueError("A zone envelope has zero variance")
    a /= a_std
    b /= b_std

    min_delay_s = distance_m / (maximum_speed_kmh / 3.6)
    max_delay_s = distance_m / (minimum_speed_kmh / 3.6)
    min_lag = max(1, int(np.floor(min_delay_s * fs_hz)))
    max_lag = min(len(a) - 10, int(np.ceil(max_delay_s * fs_hz)))
    if max_lag <= min_lag:
        raise ValueError("Event window is too short for the selected speed range")

    correlation = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")
    allowed = (np.abs(lags) >= min_lag) & (np.abs(lags) <= max_lag)
    if not np.any(allowed):
        raise ValueError("No cross-correlation lags satisfy the speed limits")
    allowed_indices = np.flatnonzero(allowed)
    best_index = allowed_indices[np.argmax(correlation[allowed])]
    best_lag = int(lags[best_index])
    delay_s = best_lag / fs_hz
    coefficient = correlation_coefficient_at_lag(a, b, best_lag)
    return delay_s, coefficient


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def summarize_pairs(pair_frame: pd.DataFrame) -> dict:
    valid = pair_frame.dropna(subset=["signed_speed_kmh", "correlation"]).copy()
    if valid.empty:
        return {
            "direction": "unknown",
            "speed_kmh": np.nan,
            "direction_confidence": "low",
            "agreeing_pairs": 0,
            "total_pairs": 0,
            "direction_vote_strength": np.nan,
            "max_speed_deviation_fraction": np.nan,
            "delay_closure_error_fraction": np.nan,
        }
    weights = np.maximum(valid["correlation"].to_numpy(float), 0.05)
    vote = np.sum(np.sign(valid["signed_speed_kmh"]) * weights)
    vote_strength = abs(float(vote)) / float(np.sum(weights))
    direction_sign = 1 if vote >= 0 else -1
    agreeing = valid[np.sign(valid["signed_speed_kmh"]).eq(direction_sign)]
    agreeing_weights = np.maximum(agreeing["correlation"].to_numpy(float), 0.05)
    speed = weighted_median(
        np.abs(agreeing["signed_speed_kmh"].to_numpy(float)), agreeing_weights
    )
    relative_spread = (
        np.median(np.abs(np.abs(agreeing["signed_speed_kmh"]) - speed)) / speed
        if speed > 0
        else np.inf
    )
    median_corr = float(agreeing["correlation"].median())
    maximum_speed_deviation = (
        float(np.max(np.abs(np.abs(agreeing["signed_speed_kmh"]) - speed)) / speed)
        if speed > 0
        else np.inf
    )

    # For the three ordered zones, the long-baseline delay must equal the sum
    # of the two adjacent delays: t(counting,test) = t(counting,decounting)
    # + t(decounting,test). This detects an axle-period correlation alias.
    delay_lookup = {
        (row.zone_a, row.zone_b): float(row.delay_b_minus_a_s)
        for row in valid.itertuples()
    }
    closure_error = np.nan
    closure_keys = (
        ("counting", "decounting"),
        ("counting", "test"),
        ("decounting", "test"),
    )
    if all(key in delay_lookup for key in closure_keys):
        delay_cd = delay_lookup[("counting", "decounting")]
        delay_ct = delay_lookup[("counting", "test")]
        delay_dt = delay_lookup[("decounting", "test")]
        denominator = max(abs(delay_ct), abs(delay_cd) + abs(delay_dt), 1e-12)
        closure_error = abs(delay_ct - (delay_cd + delay_dt)) / denominator

    all_pairs_agree = len(agreeing) == len(valid) == 3
    closure_is_good = np.isfinite(closure_error) and closure_error <= 0.15
    if (
        all_pairs_agree
        and valid["correlation"].min() >= 0.75
        and maximum_speed_deviation <= 0.15
        and closure_is_good
    ):
        confidence = "high"
    elif len(agreeing) >= 2 and median_corr >= 0.75 and vote_strength >= 0.25:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "direction": "right_to_left" if direction_sign > 0 else "left_to_right",
        "speed_kmh": speed,
        "direction_confidence": confidence,
        "agreeing_pairs": len(agreeing),
        "total_pairs": len(valid),
        "direction_vote_strength": vote_strength,
        "max_speed_deviation_fraction": maximum_speed_deviation,
        "delay_closure_error_fraction": closure_error,
    }


def analyze_recording(path: Path, args: argparse.Namespace) -> tuple[dict, pd.DataFrame, dict]:
    time_s, wavelengths = load_all_fbg_channels(path)
    fs_hz = sampling_rate(time_s)
    processed = process_all_channels(wavelengths, fs_hz, args.highpass, args.lowpass)
    envelopes = {
        name: zone_envelope(
            processed,
            definition["physical_sensors"],
            fs_hz,
            args.smoothing,
        )
        for name, definition in ZONE_DEFINITIONS.items()
    }
    event_slice = crop_to_train_event(envelopes, fs_hz, args.event_padding)
    cropped = {name: values[event_slice] for name, values in envelopes.items()}

    pair_rows = []
    zone_names = list(ZONE_DEFINITIONS)
    for i, zone_a in enumerate(zone_names):
        for zone_b in zone_names[i + 1 :]:
            position_a = ZONE_DEFINITIONS[zone_a]["position_m"]
            position_b = ZONE_DEFINITIONS[zone_b]["position_m"]
            distance = abs(position_b - position_a)
            try:
                delay_s, coefficient = estimate_pair_delay(
                    cropped[zone_a],
                    cropped[zone_b],
                    fs_hz,
                    distance,
                    args.min_speed,
                    args.max_speed,
                )
                signed_speed_mps = (position_b - position_a) / delay_s
                signed_speed_kmh = signed_speed_mps * 3.6
                error = ""
            except Exception as exc:
                delay_s = coefficient = signed_speed_kmh = np.nan
                error = str(exc)
            pair_rows.append(
                {
                    "file": path.name,
                    "zone_a": zone_a,
                    "zone_b": zone_b,
                    "distance_m": distance,
                    "delay_b_minus_a_s": delay_s,
                    "signed_speed_kmh": signed_speed_kmh,
                    "speed_kmh": abs(signed_speed_kmh),
                    "direction": (
                        "right_to_left"
                        if signed_speed_kmh > 0
                        else "left_to_right" if signed_speed_kmh < 0 else "unknown"
                    ),
                    "correlation": coefficient,
                    "error": error,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    summary = summarize_pairs(pairs)
    summary.update(
        {
            "file": path.name,
            "sampling_rate_hz": fs_hz,
            "event_start_s": time_s[event_slice.start] - time_s[0],
            "event_end_s": time_s[event_slice.stop - 1] - time_s[0],
        }
    )
    plot_data = {
        "time_s": time_s[event_slice] - time_s[0],
        "envelopes": cropped,
    }
    return summary, pairs, plot_data


def plot_analysis(path: Path, summary: dict, pairs: pd.DataFrame, plot_data: dict, output: Path, show: bool) -> None:
    fig, (axis, table_axis) = plt.subplots(
        2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
    time_s = plot_data["time_s"]
    for zone, envelope in plot_data["envelopes"].items():
        scaled = envelope / max(float(np.max(envelope)), np.finfo(float).eps)
        axis.plot(time_s, scaled, linewidth=1.2, label=zone)
    axis.set_title(
        f"{path.name} — {summary['direction']} — {summary['speed_kmh']:.1f} km/h "
        f"({summary['direction_confidence']} confidence)"
    )
    axis.set_xlabel("Elapsed time from recording start [s]")
    axis.set_ylabel("Normalized zone envelope")
    axis.grid(alpha=0.25)
    axis.legend()

    table_axis.axis("off")
    display = pairs[["zone_a", "zone_b", "distance_m", "delay_b_minus_a_s", "speed_kmh", "direction", "correlation"]].copy()
    for column in ["distance_m", "delay_b_minus_a_s", "speed_kmh", "correlation"]:
        display[column] = display[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    table = table_axis.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate train direction and velocity from the 12 reversed-order FBG channels."
    )
    parser.add_argument("input", type=Path, help="One FBG text file or a directory")
    parser.add_argument("--pattern", default="20*.txt")
    parser.add_argument("--output-dir", type=Path, default=Path("direction_velocity_results"))
    parser.add_argument("--highpass", type=float, default=0.5)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--smoothing", type=float, default=0.04, help="Envelope smoothing [s]")
    parser.add_argument("--event-padding", type=float, default=12.0)
    parser.add_argument("--min-speed", type=float, default=5.0, help="Plausible minimum [km/h]")
    parser.add_argument("--max-speed", type=float, default=200.0, help="Plausible maximum [km/h]")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    files = [input_path] if input_path.is_file() else sorted(input_path.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found at {input_path} with pattern {args.pattern!r}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_pairs = []
    for number, path in enumerate(files, 1):
        try:
            summary, pairs, plot_data = analyze_recording(path, args)
            summaries.append(summary)
            all_pairs.append(pairs)
            print(
                f"[{number}/{len(files)}] {path.name}: {summary['direction']}, "
                f"{summary['speed_kmh']:.1f} km/h, {summary['direction_confidence']}"
            )
            if not args.no_plots:
                plot_analysis(
                    path,
                    summary,
                    pairs,
                    plot_data,
                    output_dir / f"{path.stem}_direction_velocity.png",
                    args.show and len(files) == 1,
                )
        except Exception as exc:
            summaries.append(
                {
                    "file": path.name,
                    "direction": "error",
                    "speed_kmh": np.nan,
                    "direction_confidence": "low",
                    "error": str(exc),
                }
            )
            print(f"[{number}/{len(files)}] {path.name}: ERROR - {exc}")

    pd.DataFrame(summaries).to_csv(
        output_dir / "direction_velocity_summary.csv", index=False, float_format="%.6f"
    )
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(
            output_dir / "direction_velocity_pairs.csv", index=False, float_format="%.6f"
        )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

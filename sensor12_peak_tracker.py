"""Track and number peaks in FBG sensor 12 recordings.

The default processing path removes baseline drift, applies a Butterworth
band-pass filter, detects prominent positive peaks, writes the peak table to
CSV, and creates a labelled plot. VMD is available as an optional denoising
method when the ``vmdpy`` package is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt


def load_fbg_file(path: Path, sensor_number: int) -> tuple[np.ndarray, np.ndarray]:
    """Read time and one sensor column from the whitespace-separated FBG file."""
    if not 1 <= sensor_number <= 12:
        raise ValueError("sensor_number must be between 1 and 12")

    # The file header only names the first wavelength, while every data row has
    # time followed by all 12 wavelengths. Therefore column positions are used.
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        usecols=[0, sensor_number],
        names=["time_s", "wavelength_nm"],
        decimal=",",
        engine="c",
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.sort_values("time_s").drop_duplicates("time_s")

    if len(frame) < 20:
        raise ValueError(f"{path} does not contain enough valid samples")

    return (
        frame["time_s"].to_numpy(dtype=float),
        frame["wavelength_nm"].to_numpy(dtype=float),
    )


def sampling_rate(time_s: np.ndarray) -> float:
    steps = np.diff(time_s)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if not len(steps):
        raise ValueError("Cannot determine sampling rate from the time column")
    return float(1.0 / np.median(steps))


def butterworth_process(
    wavelength_nm: np.ndarray,
    fs_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
) -> np.ndarray:
    """Remove linear drift and retain frequencies useful for transient peaks."""
    nyquist = fs_hz / 2.0
    if not 0 < highpass_hz < lowpass_hz < nyquist:
        raise ValueError(
            f"Filter limits must satisfy 0 < highpass < lowpass < {nyquist:.2f} Hz"
        )
    centered = detrend(wavelength_nm, type="linear")
    sos = butter(
        4,
        [highpass_hz, lowpass_hz],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )
    return sosfiltfilt(sos, centered)


def vmd_process(
    wavelength_nm: np.ndarray,
    fs_hz: float,
    highpass_hz: float,
    lowpass_hz: float,
    mode_count: int,
    alpha: float,
) -> np.ndarray:
    """Use VMD and reconstruct only modes inside the requested frequency band."""
    try:
        from vmdpy import VMD
    except ImportError as exc:
        raise RuntimeError(
            "VMD processing requires vmdpy. Install it with: pip install vmdpy"
        ) from exc

    centered = detrend(wavelength_nm, type="linear")
    scale = float(np.std(centered))
    if scale == 0:
        return centered

    modes, _, omega_history = VMD(
        centered / scale,
        alpha,
        0.0,  # noise tolerance (tau)
        mode_count,
        0,  # do not force a DC mode
        1,  # initialize center frequencies uniformly
        1e-7,
    )
    center_hz = np.asarray(omega_history[-1]) * fs_hz
    selected = (center_hz >= highpass_hz) & (center_hz <= lowpass_hz)
    if not np.any(selected):
        raise RuntimeError(
            "No VMD modes fell inside the selected band. "
            f"Mode centers were {np.round(center_hz, 2).tolist()} Hz."
        )
    print(
        "VMD mode centers [Hz]:",
        np.round(center_hz, 2).tolist(),
        "| retained:",
        np.round(center_hz[selected], 2).tolist(),
    )
    return np.sum(modes[selected], axis=0) * scale


def robust_sigma(values: np.ndarray) -> float:
    """Estimate background scale without being dominated by large train peaks."""
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def detect_peaks(
    signal_nm: np.ndarray,
    fs_hz: float,
    polarity: str,
    prominence_sigma: float,
    prominence_nm: float | None,
    height_fraction: float,
    minimum_distance_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    sigma = robust_sigma(signal_nm)
    prominence = (
        float(prominence_nm)
        if prominence_nm is not None
        else float(prominence_sigma * sigma)
    )
    distance_samples = max(1, int(round(minimum_distance_s * fs_hz)))

    candidates: list[tuple[int, int]] = []
    if polarity in {"positive", "both"}:
        positive_height = max(prominence, height_fraction * float(np.max(signal_nm)))
        indices, _ = find_peaks(
            signal_nm,
            height=positive_height,
            prominence=prominence,
            distance=distance_samples,
        )
        candidates.extend((int(index), 1) for index in indices)
    if polarity in {"negative", "both"}:
        negative_height = max(prominence, height_fraction * float(np.max(-signal_nm)))
        indices, _ = find_peaks(
            -signal_nm,
            height=negative_height,
            prominence=prominence,
            distance=distance_samples,
        )
        candidates.extend((int(index), -1) for index in indices)

    candidates.sort(key=lambda item: item[0])
    return (
        np.asarray([item[0] for item in candidates], dtype=int),
        np.asarray([item[1] for item in candidates], dtype=int),
        prominence,
    )


def make_peak_table(
    time_s: np.ndarray,
    raw_nm: np.ndarray,
    processed_nm: np.ndarray,
    indices: np.ndarray,
    signs: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "peak_number": np.arange(1, len(indices) + 1),
            "time_s": time_s[indices],
            "elapsed_s": time_s[indices] - time_s[0],
            "raw_wavelength_nm": raw_nm[indices],
            "processed_amplitude_nm": processed_nm[indices],
            "polarity": np.where(signs > 0, "positive", "negative"),
        }
    )


def plot_results(
    input_path: Path,
    time_s: np.ndarray,
    raw_nm: np.ndarray,
    processed_nm: np.ndarray,
    peaks: pd.DataFrame,
    output_path: Path,
    sensor_number: int,
    method: str,
    show: bool,
) -> None:
    elapsed_s = time_s - time_s[0]
    fig, (raw_axis, peak_axis) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True, constrained_layout=True
    )

    raw_axis.plot(elapsed_s, raw_nm, color="tab:blue", linewidth=0.8)
    raw_axis.set_ylabel("Wavelength [nm]")
    raw_axis.set_title(f"FBG sensor {sensor_number}: {input_path.name}")
    raw_axis.grid(alpha=0.25)

    peak_axis.plot(elapsed_s, processed_nm, color="black", linewidth=0.8)
    if not peaks.empty:
        colors = np.where(peaks["polarity"].eq("positive"), "tab:red", "tab:green")
        peak_axis.scatter(
            peaks["elapsed_s"],
            peaks["processed_amplitude_nm"],
            c=colors,
            s=28,
            zorder=3,
            label="Detected peaks",
        )
        for row in peaks.itertuples(index=False):
            offset = 6 if row.processed_amplitude_nm >= 0 else -13
            peak_axis.annotate(
                str(row.peak_number),
                (row.elapsed_s, row.processed_amplitude_nm),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=7,
                color="tab:red" if row.polarity == "positive" else "tab:green",
            )
        peak_axis.margins(y=min(0.25, 0.03 + 0.001 * len(peaks)))

        # Keep the raw panel as the full-recording overview, but zoom the
        # processed panel so sequential peak labels remain readable.
        event_start = float(peaks["elapsed_s"].min())
        event_end = float(peaks["elapsed_s"].max())
        event_span = max(event_end - event_start, 1.0)
        padding = max(0.5, 0.08 * event_span)
        peak_axis.set_xlim(
            max(float(elapsed_s[0]), event_start - padding),
            min(float(elapsed_s[-1]), event_end + padding),
        )

    peak_axis.set_title(f"{method.upper()} processed signal — {len(peaks)} peaks")
    peak_axis.set_xlabel("Elapsed time [s]")
    peak_axis.set_ylabel("Amplitude change [nm]")
    peak_axis.grid(alpha=0.25)
    peak_axis.legend(loc="upper right") if not peaks.empty else None
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an FBG channel and number its detected peaks."
    )
    parser.add_argument("input_file", type=Path, help="FBG .txt recording")
    parser.add_argument("--sensor", type=int, default=12, help="Sensor number (default: 12)")
    parser.add_argument(
        "--method", choices=("filter", "vmd"), default="filter", help="Denoising method"
    )
    parser.add_argument(
        "--polarity",
        choices=("positive", "negative", "both"),
        default="positive",
        help="Which peaks to count (default: positive)",
    )
    parser.add_argument("--highpass", type=float, default=0.5, help="High-pass frequency [Hz]")
    parser.add_argument("--lowpass", type=float, default=40.0, help="Low-pass frequency [Hz]")
    parser.add_argument(
        "--prominence-sigma",
        type=float,
        default=8.0,
        help="Adaptive prominence multiplier (default: 8)",
    )
    parser.add_argument(
        "--prominence-nm",
        type=float,
        default=None,
        help="Fixed prominence in nm; overrides --prominence-sigma",
    )
    parser.add_argument(
        "--height-fraction",
        type=float,
        default=0.06,
        help="Minimum peak height as a fraction of the largest peak (default: 0.06)",
    )
    parser.add_argument(
        "--minimum-distance",
        type=float,
        default=0.04,
        help=(
            "Minimum time between same-polarity axle peaks [s] (default: 0.04). "
            "Keep this below the time separation of the two axles in a bogie."
        ),
    )
    parser.add_argument("--vmd-modes", type=int, default=5, help="Number of VMD modes")
    parser.add_argument("--vmd-alpha", type=float, default=2000.0, help="VMD bandwidth penalty")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="Open the plot interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_file.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    time_s, raw_nm = load_fbg_file(input_path, args.sensor)
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

    output_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{input_path.stem}_sensor{args.sensor}_peaks"
    csv_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"
    peaks.to_csv(csv_path, index=False, float_format="%.8f")
    plot_results(
        input_path,
        time_s,
        raw_nm,
        processed_nm,
        peaks,
        plot_path,
        args.sensor,
        args.method,
        args.show,
    )

    print(f"Sampling rate: {fs_hz:.2f} Hz")
    print(f"Prominence threshold: {prominence:.8f} nm")
    print(f"Detected peaks: {len(peaks)}")
    if not peaks.empty:
        print(peaks.to_string(index=False))
    print(f"Peak table: {csv_path}")
    print(f"Labelled plot: {plot_path}")


if __name__ == "__main__":
    main()

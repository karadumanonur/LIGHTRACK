"""Interactive dashboard for inspecting FBG recordings and axle detections.

Run with:
    streamlit run legacy/fbg_analysis_dashboard.py

The dashboard deliberately separates physical FBG identity from stored column
position. It detects ascending versus descending wavelength order in every
recording, so both the May/June and July deliveries are mapped correctly.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date
from pathlib import Path

import altair as alt
import matplotlib

# Streamlit renders figures into the browser.  A desktop GUI backend such as
# macOS Cocoa cannot create figures from Streamlit's worker thread.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException
from scipy.interpolate import CubicSpline
from scipy.signal import detrend, fftconvolve, find_peaks, resample, welch

try:
    from .self_calibrating_axle_detector import (
        AdaptiveSettings,
        detect_adaptive_peaks,
    )
    from .sensor12_peak_tracker import butterworth_process, sampling_rate
except ImportError:  # Support direct Streamlit execution of this legacy file.
    from self_calibrating_axle_detector import (
        AdaptiveSettings,
        detect_adaptive_peaks,
    )
    from sensor12_peak_tracker import butterworth_process, sampling_rate


TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6},\d{3})Z")
HEAD_SENSORS = {1: (1, 2, 3, 4), 2: (5, 6, 7, 8), 3: (9, 10, 11, 12)}
FIXED_HIGHPASS_HZ = 0.5
FIXED_LOWPASS_HZ = 40.0
MAX_PLOT_POINTS = 12_000
SUMMARY_FILENAME = "two_period_peak_change_mean_std_pm.csv"
AXLE_HEIGHT_CACHE_VERSION = "raw-linear-detrend-polarity-v2"


def default_two_period_summary() -> Path:
    configured = os.environ.get("LIGHTRACK_TWO_PERIOD_SUMMARY")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(__file__).resolve().parent / "assets" / SUMMARY_FILENAME,
        (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "fbg1_4_5_9_10_two_periods_detrended_axle_peak_pm"
            / SUMMARY_FILENAME
        ),
    ]
    return next(
        (candidate.resolve() for candidate in candidates if candidate and candidate.is_file()),
        Path(__file__).resolve().parent / "assets" / SUMMARY_FILENAME,
    )


TWO_PERIOD_SUMMARY = default_two_period_summary()


def timestamp_from_name(path: Path) -> pd.Timestamp | None:
    match = TIMESTAMP_RE.search(path.name)
    if match is None:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%dT%H%M%S,%f")


def title_with_recording_timestamp(
    title: str, timestamp: pd.Timestamp | None
) -> str:
    """Append the source recording's UTC timestamp to a plot title."""
    if timestamp is None:
        return title
    formatted = timestamp.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]
    return f"{title} · {formatted} UTC"


def default_data_root() -> str:
    configured = os.environ.get("LIGHTRACK_DATA_DIR")
    if configured:
        return configured
    candidates = (
        Path.home() / "OneDrive" / "Desktop" / "Railway Monitoring" / "data" / "data" / "raw_data",
        Path.home() / "Desktop" / "Railway Monitoring" / "data" / "data" / "raw_data",
        Path.cwd() / "data" / "raw_data",
        Path.cwd().parent / "data" / "raw_data",
        Path.cwd().parent.parent / "data" / "raw_data",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path.cwd() / "data" / "raw_data").resolve())


def persist_uploaded_recordings(uploaded_files: list[object]) -> str:
    """Store uploaded recordings in this Streamlit session's temporary folder."""
    signature = tuple(
        (Path(uploaded.name).name, int(uploaded.size))
        for uploaded in uploaded_files
    )
    if st.session_state.get("uploaded_recording_signature") != signature:
        upload_root = Path(tempfile.mkdtemp(prefix="lightrack_fbg_upload_"))
        for uploaded in uploaded_files:
            safe_name = Path(uploaded.name).name
            if TIMESTAMP_RE.search(safe_name) is None:
                continue
            (upload_root / safe_name).write_bytes(uploaded.getvalue())
        st.session_state["uploaded_recording_root"] = str(upload_root)
        st.session_state["uploaded_recording_signature"] = signature
        index_recordings.clear()
    return str(st.session_state["uploaded_recording_root"])


@st.cache_data(show_spinner=False)
def index_recordings(root_text: str) -> pd.DataFrame:
    cleaned = root_text.strip().strip('"').strip("'")
    root = Path(cleaned).expanduser()
    candidates = (
        root,
        root / "raw_data",
        root / "data" / "raw_data",
        root / "data" / "data" / "raw_data",
    )
    search_root = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if search_root is None:
        return pd.DataFrame(columns=["path", "timestamp", "date", "time", "period"])
    rows: list[dict[str, object]] = []
    for path in search_root.rglob("*.txt"):
        timestamp = timestamp_from_name(path)
        if timestamp is None:
            continue
        rows.append({
            "path": str(path.resolve()),
            "timestamp": timestamp,
            "date": timestamp.date(),
            "time": timestamp.strftime("%H:%M:%S.%f")[:-3],
            "period": path.parent.name,
        })
    if not rows:
        return pd.DataFrame(columns=["path", "timestamp", "date", "time", "period"])
    indexed = pd.DataFrame(rows)
    # The local archive contains 89 recordings in both ``Detected`` and the
    # explicit first-period folder. Prefer the named period folders and show
    # each physical recording only once in the selectors.
    indexed["period_priority"] = indexed["period"].map(
        lambda value: 0 if str(value).startswith("period_") else 1
    )
    indexed = (
        indexed.sort_values(["timestamp", "period_priority", "path"])
        .drop_duplicates(subset=["timestamp"], keep="first")
        .drop(columns="period_priority")
    )
    return indexed.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_two_period_summary(
    path_text: str,
    calculation_version: str = AXLE_HEIGHT_CACHE_VERSION,
    file_signature: tuple[int, int] | None = None,
) -> pd.DataFrame:
    # Both values are included in Streamlit's cache key.  The file signature
    # makes a newly copied or replaced summary visible without a server restart.
    del calculation_version, file_signature
    path = Path(path_text)
    if not path.is_file():
        return pd.DataFrame()
    data = pd.read_csv(path, parse_dates=["recording_time_utc"])
    required = {
        "file", "recording_time_utc", "physical_fbg", "positive_peak_count",
        "mean_peak_change_pm", "mean_minus_std_pm", "mean_plus_std_pm",
        "measurement_period",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Two-period summary is missing columns: {sorted(missing)}")
    return data.loc[
        (data["positive_peak_count"] > 0)
        & np.isfinite(data["mean_peak_change_pm"])
    ].copy()


@st.cache_resource(show_spinner=False, max_entries=2)
def load_recording_frame(path_text: str) -> pd.DataFrame:
    """Read one recording once and keep only the two most recent files in RAM."""
    path = Path(path_text)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        usecols=range(13),
        decimal=",",
        engine="c",
    ).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=[0]).sort_values(0).drop_duplicates(0)
    if len(frame) < 20:
        raise ValueError("The recording contains fewer than 20 valid samples")
    return frame


def load_physical_fbg_from_frame(
    frame: pd.DataFrame, physical_fbg: int
) -> dict[str, object]:
    """Select one physical FBG after detecting this recording's stored order."""
    if physical_fbg not in range(1, 13):
        raise ValueError("Physical FBG must be between 1 and 12")
    medians = frame.loc[:, 1:12].median(axis=0, skipna=True)
    if not np.isfinite(medians.loc[1]) or not np.isfinite(medians.loc[12]):
        raise ValueError("Stored wavelength order cannot be identified")
    ascending = bool(medians.loc[1] < medians.loc[12])
    stored_by_physical_fbg = {
        sensor: sensor if ascending else 13 - sensor for sensor in range(1, 13)
    }
    stored_column = stored_by_physical_fbg[physical_fbg]
    selected = frame.loc[:, [0, stored_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(selected) < 20:
        raise ValueError(f"Physical FBG {physical_fbg} has too few valid samples")
    return {
        "time_s": selected[0].to_numpy(dtype=float),
        "raw_nm": selected[stored_column].to_numpy(dtype=float),
        "physical_fbg": physical_fbg,
        "stored_column": stored_column,
        "stored_order": "ascending" if ascending else "descending",
    }


def load_physical_fbg(path_text: str, physical_fbg: int) -> dict[str, object]:
    # All physical FBGs in a recording share the same source table. Reading
    # that ~10 MB text file separately for every sensor caused large temporary
    # DataFrames and could terminate the packaged Streamlit server.
    frame = load_recording_frame(path_text)

    return load_physical_fbg_from_frame(frame, physical_fbg)


def calculate_axle_heights(
    frame: pd.DataFrame,
    physical_fbg: int,
    detector=detect_adaptive_peaks,
) -> dict[str, object]:
    """Calculate polarity-oriented heights from detrended raw axle samples."""
    loaded = load_physical_fbg_from_frame(frame, physical_fbg)
    time_s = np.asarray(loaded["time_s"], dtype=float)
    raw_nm = np.asarray(loaded["raw_nm"], dtype=float)
    fs_hz = sampling_rate(time_s)
    processed_nm = butterworth_process(
        raw_nm, fs_hz, FIXED_HIGHPASS_HZ, FIXED_LOWPASS_HZ
    )
    detection = detector(
        processed_nm,
        time_s,
        fs_hz,
        polarity="auto",
        settings=AdaptiveSettings(),
    )
    accepted = np.asarray(detection.indices, dtype=int)
    accepted = accepted[(accepted >= 0) & (accepted < len(raw_nm))]
    selected_sign = 1 if detection.selected_polarity == "positive" else -1
    baseline_removed_pm = detrend(raw_nm, type="linear") * 1000.0
    axle_heights_pm = selected_sign * baseline_removed_pm[accepted]
    finite = np.isfinite(axle_heights_pm)
    accepted = accepted[finite]
    axle_heights_pm = axle_heights_pm[finite]
    peak_count = int(len(axle_heights_pm))
    if peak_count:
        mean_pm = float(np.mean(axle_heights_pm))
        std_pm = float(np.std(axle_heights_pm, ddof=1)) if peak_count > 1 else 0.0
        lower_pm = mean_pm - std_pm / 2.0
        upper_pm = mean_pm + std_pm / 2.0
    else:
        mean_pm = std_pm = lower_pm = upper_pm = float("nan")
    return {
        **loaded,
        "fs_hz": fs_hz,
        "processed_nm": processed_nm,
        "baseline_removed_pm": baseline_removed_pm,
        "peak_indices": accepted,
        "peak_signs": np.full(peak_count, selected_sign, dtype=int),
        "peak_count": peak_count,
        "axle_heights_pm": axle_heights_pm,
        "mean_pm": mean_pm,
        "std_pm": std_pm,
        "lower_pm": lower_pm,
        "upper_pm": upper_pm,
        "selected_sign": selected_sign,
        "detection": detection,
    }


@st.cache_resource(show_spinner=False, max_entries=12)
def analyse_recording(
    path_text: str,
    physical_fbg: int,
    calculation_version: str = AXLE_HEIGHT_CACHE_VERSION,
) -> dict[str, object]:
    del calculation_version  # Included in the Streamlit cache key.
    calculated = calculate_axle_heights(load_recording_frame(path_text), physical_fbg)
    time_s = np.asarray(calculated["time_s"], dtype=float)
    processed_nm = np.asarray(calculated["processed_nm"], dtype=float)
    detection = calculated.pop("detection")
    derivative_nm_s = np.gradient(processed_nm, time_s)
    return {
        **calculated,
        "derivative_nm_s": derivative_nm_s,
        "polarity": detection.selected_polarity,
        "polarity_confidence": detection.polarity_confidence,
        "background_sigma_nm": detection.background_sigma_nm,
        "correlation_time_s": detection.correlation_time_s,
        "quality_score": detection.quality_score,
        "quality_label": detection.quality_label,
        "positive_height_sigma": detection.positive.height_threshold_sigma,
        "positive_prominence_sigma": detection.positive.prominence_threshold_sigma,
        "negative_height_sigma": detection.negative.height_threshold_sigma,
        "negative_prominence_sigma": detection.negative.prominence_threshold_sigma,
        "positive_axle_count": len(detection.positive.accepted_indices),
        "negative_axle_count": len(detection.negative.accepted_indices),
    }


def axle_height_summary_row(
    path_text: str, physical_fbg: int, timestamp: pd.Timestamp | None = None
) -> dict[str, object] | None:
    """Return the canonical graph/export row, or None when no peaks exist."""
    result = analyse_recording(path_text, physical_fbg)
    if int(result["peak_count"]) == 0 or not np.isfinite(result["mean_pm"]):
        return None
    stamp = timestamp if timestamp is not None else timestamp_from_name(Path(path_text))
    return {
        "file": Path(path_text).name,
        "recording_time_utc": stamp,
        "physical_fbg": int(result["physical_fbg"]),
        "stored_column": int(result["stored_column"]),
        "positive_peak_count": int(result["peak_count"]),
        "peak_count": int(result["peak_count"]),
        "selected_sign": int(result["selected_sign"]),
        "mean_peak_change_pm": float(result["mean_pm"]),
        "std_peak_change_pm": float(result["std_pm"]),
        "mean_minus_std_pm": float(result["lower_pm"]),
        "mean_plus_std_pm": float(result["upper_pm"]),
    }


def summary_to_csv(data: pd.DataFrame) -> bytes:
    """Serialize the exact canonical rows used by the graph."""
    return data.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=12)
def calculate_dataset_summary(
    recording_rows: tuple[tuple[str, pd.Timestamp, str], ...],
    physical_fbg: int,
    calculation_version: str = AXLE_HEIGHT_CACHE_VERSION,
) -> pd.DataFrame:
    """Calculate canonical graph/export rows for one FBG across all recordings."""
    del calculation_version
    rows: list[dict[str, object]] = []
    for path_text, timestamp, period in recording_rows:
        try:
            row = axle_height_summary_row(path_text, physical_fbg, timestamp)
        except Exception:
            continue
        if row is None:
            continue
        row["measurement_period"] = (
            "Period 2 (Jul)" if "period_2" in str(period).lower()
            else "Period 1 (May-Jun)"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reduced_indices(length: int, protected: np.ndarray | None = None) -> np.ndarray:
    if length <= MAX_PLOT_POINTS:
        return np.arange(length)
    base = np.linspace(0, length - 1, MAX_PLOT_POINTS, dtype=int)
    if protected is None or not len(protected):
        return np.unique(base)
    return np.unique(np.r_[base, protected.astype(int)])


def signal_figure(
    time_s: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    peaks: np.ndarray | None = None,
    x_limits: tuple[float, float] | None = None,
    recording_timestamp: pd.Timestamp | None = None,
    show_full_y_values: bool = False,
) -> plt.Figure:
    elapsed = time_s - time_s[0]
    if x_limits is None:
        display = reduced_indices(len(values), peaks)
    else:
        visible = np.flatnonzero(
            (elapsed >= x_limits[0]) & (elapsed <= x_limits[1])
        )
        if not len(visible):
            visible = np.arange(len(values))
        display = visible[reduced_indices(len(visible))]
        if peaks is not None and len(peaks):
            visible_peaks = peaks[
                (elapsed[peaks] >= x_limits[0])
                & (elapsed[peaks] <= x_limits[1])
            ]
            display = np.unique(np.r_[display, visible_peaks])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(elapsed[display], values[display], linewidth=0.8, color="#1261a0")
    if peaks is not None and len(peaks):
        ax.scatter(
            elapsed[peaks], values[peaks], s=19, color="#d62728",
            zorder=4, label=f"Accepted axle candidates ({len(peaks)})",
        )
        ax.legend(loc="upper right")
    ax.set_title(title_with_recording_timestamp(title, recording_timestamp))
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel(ylabel)
    if show_full_y_values:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    return fig


def axle_count_overlay_figure(
    results: list[tuple[int, dict[str, object]]],
    recording_timestamp: pd.Timestamp | None = None,
) -> plt.Figure:
    """Overlay selected physical FBG detector responses on one time axis."""
    fbg_colors = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#5f5f5f",
        "#9a9d00", "#00a6c8", "#000000", "#ff1493",
    )
    all_peak_times: list[float] = []
    prepared: list[tuple[int, dict[str, object], np.ndarray, np.ndarray, np.ndarray]] = []
    for physical_fbg, result in results:
        time_s = np.asarray(result["time_s"], dtype=float)
        elapsed = time_s - time_s[0]
        values_pm = np.asarray(result["processed_nm"], dtype=float) * 1000.0
        peaks = np.asarray(result["peak_indices"], dtype=int)
        prepared.append((physical_fbg, result, elapsed, values_pm, peaks))
        if len(peaks):
            all_peak_times.extend(elapsed[peaks].tolist())

    x_limits: tuple[float, float] | None = None
    if all_peak_times:
        x_limits = (
            max(0.0, min(all_peak_times) - 2.0),
            max(all_peak_times) + 2.0,
        )

    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    for physical_fbg, _result, elapsed, values_pm, peaks in prepared:
        color = fbg_colors[(physical_fbg - 1) % len(fbg_colors)]
        if x_limits is None:
            display = reduced_indices(len(values_pm), peaks)
        else:
            visible = np.flatnonzero(
                (elapsed >= x_limits[0]) & (elapsed <= x_limits[1])
            )
            if not len(visible):
                visible = np.arange(len(values_pm))
            display = visible[reduced_indices(len(visible))]
            if len(peaks):
                visible_peaks = peaks[
                    (elapsed[peaks] >= x_limits[0])
                    & (elapsed[peaks] <= x_limits[1])
                ]
                display = np.unique(np.r_[display, visible_peaks])

        ax.plot(
            elapsed[display], values_pm[display], color=color,
            linewidth=1.35, alpha=0.96,
            label=f"Physical FBG {physical_fbg} — {len(peaks)} axles",
        )
        if len(peaks):
            ax.scatter(
                elapsed[peaks], values_pm[peaks], s=24, color=color,
                edgecolors="white", linewidths=0.55, zorder=4,
            )

    ax.axhline(0.0, color="#64748b", linewidth=0.7, alpha=0.45)
    ax.set_title(
        title_with_recording_timestamp(
            "Self-calibrating axle counts", recording_timestamp
        )
    )
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Filtered response [pm]")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.grid(alpha=0.26)
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.94)
    fig.tight_layout()
    return fig


def passage_limits(result: dict[str, object], context_s: float = 2.0) -> tuple[float, float] | None:
    time_s = np.asarray(result["time_s"], dtype=float)
    elapsed = time_s - time_s[0]
    peaks = np.asarray(result["peak_indices"], dtype=int)
    if not len(peaks):
        return None

    # The strict detector already identifies one contiguous train-event
    # interval.  Use that interval for the detail view instead of allowing an
    # isolated accepted candidate near either file boundary to expand the plot
    # to the entire recording.
    active_mask_value = result.get("active_mask")
    if active_mask_value is not None:
        active_mask = np.asarray(active_mask_value, dtype=bool)
        if active_mask.shape == time_s.shape and np.any(active_mask):
            active = np.flatnonzero(active_mask)
            return (
                max(float(elapsed[0]), float(elapsed[active[0]]) - context_s),
                min(float(elapsed[-1]), float(elapsed[active[-1]]) + context_s),
            )

    return (
        max(float(elapsed[0]), float(elapsed[peaks[0]]) - context_s),
        min(float(elapsed[-1]), float(elapsed[peaks[-1]]) + context_s),
    )


def analysis_window(result: dict[str, object], passage_only: bool) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(result["time_s"], dtype=float)
    raw_nm = np.asarray(result["raw_nm"], dtype=float)
    peaks = np.asarray(result["peak_indices"], dtype=int)
    if passage_only and len(peaks):
        fs_hz = float(result["fs_hz"])
        margin = int(round(fs_hz))
        left = max(0, int(peaks[0]) - margin)
        right = min(len(raw_nm), int(peaks[-1]) + margin + 1)
        time_s = time_s[left:right]
        raw_nm = raw_nm[left:right]
    return time_s, detrend(raw_nm, type="linear")


def fft_plot(
    values: np.ndarray, fs_hz: float, recording_timestamp: pd.Timestamp | None
) -> plt.Figure:
    window = np.hanning(len(values))
    frequency = np.fft.rfftfreq(len(values), d=1.0 / fs_hz)
    power = np.abs(np.fft.rfft(values * window)) ** 2 / max(
        fs_hz * np.sum(window ** 2), np.finfo(float).tiny
    )
    keep = (frequency >= 0.1) & (frequency <= min(100.0, fs_hz / 2.0))
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.plot(frequency[keep], power[keep], linewidth=1.0)
    ax.set(
        title=title_with_recording_timestamp("FFT power spectrum", recording_timestamp),
        xlabel="Frequency [Hz]",
        ylabel="Power [nm²/Hz]",
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    return fig


def psd_plot(
    values: np.ndarray, fs_hz: float, recording_timestamp: pd.Timestamp | None
) -> plt.Figure:
    nperseg = min(2048, len(values))
    frequency, power = welch(
        values,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    keep = (frequency >= 0.1) & (frequency <= min(100.0, fs_hz / 2.0))
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.semilogy(frequency[keep], np.maximum(power[keep], np.finfo(float).tiny))
    ax.set(
        title=title_with_recording_timestamp(
            "Welch power spectral density", recording_timestamp
        ),
        xlabel="Frequency [Hz]",
        ylabel="PSD [nm²/Hz]",
    )
    ax.grid(alpha=0.22)
    fig.tight_layout()
    return fig


def wavelet_plot(
    time_s: np.ndarray,
    values: np.ndarray,
    fs_hz: float,
    recording_timestamp: pd.Timestamp | None,
) -> plt.Figure:
    maximum_samples = 6000
    if len(values) > maximum_samples:
        reduced = resample(values, maximum_samples)
        reduced_time = np.linspace(time_s[0], time_s[-1], maximum_samples)
        reduced_fs = fs_hz * maximum_samples / len(values)
    else:
        reduced, reduced_time, reduced_fs = values, time_s, fs_hz
    frequencies = np.geomspace(0.5, min(80.0, reduced_fs * 0.45), 38)
    energy = np.empty((len(frequencies), len(reduced)), dtype=float)
    for row, frequency in enumerate(frequencies):
        sigma_t = 6.0 / (2.0 * np.pi * frequency)
        half_width = max(2, int(np.ceil(4.0 * sigma_t * reduced_fs)))
        kernel_time = np.arange(-half_width, half_width + 1) / reduced_fs
        kernel = np.exp(2j * np.pi * frequency * kernel_time) * np.exp(
            -(kernel_time ** 2) / (2.0 * sigma_t ** 2)
        )
        kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2))
        coefficient = fftconvolve(reduced, np.conjugate(kernel[::-1]), mode="same")
        energy[row] = np.abs(coefficient) ** 2
    fig, ax = plt.subplots(figsize=(13, 5.0))
    mesh = ax.pcolormesh(
        reduced_time - reduced_time[0], frequencies, energy,
        shading="auto", cmap="viridis",
    )
    ax.set_yscale("log")
    ax.set(
        title=title_with_recording_timestamp("Morlet wavelet energy", recording_timestamp),
        xlabel="Elapsed time [s]",
        ylabel="Frequency [Hz]",
    )
    fig.colorbar(mesh, ax=ax, label="Wavelet energy [nm²]")
    fig.tight_layout()
    return fig


def extrema_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxima, _ = find_peaks(values)
    minima, _ = find_peaks(-values)
    return maxima, minima


def emd_decompose(
    values: np.ndarray,
    maximum_imfs: int = 4,
    maximum_sifts: int = 8,
    tolerance: float = 0.08,
) -> tuple[list[np.ndarray], np.ndarray]:
    residue = values.astype(float).copy()
    positions = np.arange(len(values), dtype=float)
    imfs: list[np.ndarray] = []
    for _ in range(maximum_imfs):
        maxima, minima = extrema_indices(residue)
        if min(len(maxima), len(minima)) < 2:
            break
        component = residue.copy()
        for _ in range(maximum_sifts):
            maxima, minima = extrema_indices(component)
            if min(len(maxima), len(minima)) < 2:
                break
            upper_x = np.r_[0, maxima, len(component) - 1]
            lower_x = np.r_[0, minima, len(component) - 1]
            upper_y = np.r_[component[0], component[maxima], component[-1]]
            lower_y = np.r_[component[0], component[minima], component[-1]]
            upper = CubicSpline(upper_x, upper_y, bc_type="natural")(positions)
            lower = CubicSpline(lower_x, lower_y, bc_type="natural")(positions)
            candidate = component - 0.5 * (upper + lower)
            relative_change = np.sum((component - candidate) ** 2) / (
                np.sum(component ** 2) + np.finfo(float).tiny
            )
            component = candidate
            if relative_change < tolerance:
                break
        imfs.append(component)
        residue -= component
    return imfs, residue


def decomposition_input(
    time_s: np.ndarray,
    values: np.ndarray,
    fs_hz: float,
    maximum_samples: int = 3500,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(values) <= maximum_samples:
        return time_s, values, fs_hz
    reduced = resample(values, maximum_samples)
    reduced_time = np.linspace(time_s[0], time_s[-1], maximum_samples)
    reduced_fs = fs_hz * maximum_samples / len(values)
    return reduced_time, reduced, reduced_fs


def emd_plot(
    time_s: np.ndarray,
    values: np.ndarray,
    fs_hz: float,
    recording_timestamp: pd.Timestamp | None,
) -> plt.Figure:
    reduced_time, reduced, _ = decomposition_input(time_s, values, fs_hz)
    imfs, residue = emd_decompose(reduced, maximum_imfs=4)
    series = [*imfs, residue]
    labels = [f"IMF {index}" for index in range(1, len(imfs) + 1)] + ["Residue"]
    fig, axes = plt.subplots(len(series), 1, figsize=(13, 1.75 * len(series)), sharex=True)
    axes = np.atleast_1d(axes)
    elapsed = reduced_time - reduced_time[0]
    for ax, component, label in zip(axes, series, labels):
        ax.plot(elapsed, component, linewidth=0.75)
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Elapsed time [s]")
    axes[0].set_title(
        title_with_recording_timestamp(
            "Empirical mode decomposition", recording_timestamp
        )
    )
    fig.tight_layout()
    return fig


def vmd_plot(
    time_s: np.ndarray,
    values: np.ndarray,
    fs_hz: float,
    recording_timestamp: pd.Timestamp | None,
) -> plt.Figure:
    try:
        from vmdpy import VMD
    except ImportError as exc:
        raise RuntimeError("VMD requires vmdpy: pip install vmdpy") from exc
    reduced_time, reduced, _ = decomposition_input(time_s, values, fs_hz)
    modes, _, omega = VMD(reduced, 2000.0, 0.0, 4, 0, 1, 1e-7)
    fig, axes = plt.subplots(len(modes), 1, figsize=(13, 1.8 * len(modes)), sharex=True)
    axes = np.atleast_1d(axes)
    elapsed = reduced_time - reduced_time[0]
    final_omega = omega[-1] if np.ndim(omega) > 1 else omega
    for index, (ax, mode) in enumerate(zip(axes, modes), start=1):
        center_hz = float(final_omega[index - 1] * fs_hz)
        # vmdpy can return one fewer sample for an odd-length input window.
        # Keep the plot dimensions consistent without changing the signal data.
        sample_count = min(len(elapsed), len(mode))
        ax.plot(elapsed[:sample_count], mode[:sample_count], linewidth=0.75)
        ax.set_ylabel(f"Mode {index}\n{center_hz:.1f} Hz")
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("Elapsed time [s]")
    axes[0].set_title(
        title_with_recording_timestamp(
            "Variational mode decomposition", recording_timestamp
        )
    )
    fig.tight_layout()
    return fig


def show_frequency_analysis(
    method: str,
    result: dict[str, object],
    passage_only: bool,
    recording_timestamp: pd.Timestamp | None,
) -> None:
    time_s, values = analysis_window(result, passage_only)
    fs_hz = float(result["fs_hz"])
    if len(values) < 32:
        st.warning("The selected analysis window is too short for frequency analysis.")
        return
    with st.spinner(f"Calculating {method}…"):
        if method == "FFT":
            figure = fft_plot(values, fs_hz, recording_timestamp)
        elif method == "Welch PSD":
            figure = psd_plot(values, fs_hz, recording_timestamp)
        elif method == "Morlet wavelet":
            figure = wavelet_plot(time_s, values, fs_hz, recording_timestamp)
        elif method == "EMD":
            figure = emd_plot(time_s, values, fs_hz, recording_timestamp)
        else:
            figure = vmd_plot(time_s, values, fs_hz, recording_timestamp)
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def two_period_panel(
    data: pd.DataFrame,
    period: str,
    title: str,
    width: int,
    y_domain: tuple[float, float],
) -> alt.LayerChart:
    selected = data.loc[data["measurement_period"] == period]
    colors = alt.Scale(
        domain=["Period 1 (May-Jun)", "Period 2 (Jul)"],
        range=["#1261a0", "#1b8a5a"],
    )
    base = alt.Chart(selected).encode(
        x=alt.X(
            "recording_time_utc:T",
            title=title,
            axis=alt.Axis(format="%d %b", labelAngle=-45),
        ),
        color=alt.Color(
            "measurement_period:N",
            scale=colors,
            legend=alt.Legend(title="Measurement period"),
        ),
    )
    error_bars = base.mark_rule(opacity=0.34, clip=True).encode(
        y=alt.Y(
            "mean_minus_std_pm:Q",
            title="Detrended axle-peak amplitude [pm]",
            scale=alt.Scale(domain=list(y_domain), zero=False),
        ),
        y2="mean_plus_std_pm:Q",
    )
    points = base.mark_circle(size=43, clip=True).encode(
        y=alt.Y("mean_peak_change_pm:Q", title="Detrended axle-peak amplitude [pm]"),
        opacity=alt.value(0.72),
        tooltip=[
            alt.Tooltip("recording_time_utc:T", title="Recording", format="%d %b %Y %H:%M:%S"),
            alt.Tooltip("file:N", title="File"),
            alt.Tooltip("peak_count:Q", title="Detected axles"),
            alt.Tooltip("mean_peak_change_pm:Q", title="Mean", format=".2f"),
            alt.Tooltip("std_peak_change_pm:Q", title="Standard deviation", format=".2f"),
        ],
    )
    return (error_bars + points).properties(width=width, height=430)


def render_two_period_overview(recordings: pd.DataFrame) -> None:
    st.subheader("Detrended axle-amplitude comparison")
    st.caption(
        "Click a dot to inspect that recording."
    )
    sensor = st.selectbox(
        "Physical FBG",
        list(range(1, 13)),
        key="trend_physical_fbg",
    )
    if not TWO_PERIOD_SUMMARY.is_file():
        st.error(
            "The required Two Period summary CSV was not found: "
            f"{TWO_PERIOD_SUMMARY}"
        )
        return
    try:
        summary_stat = TWO_PERIOD_SUMMARY.stat()
        summary = load_two_period_summary(
            str(TWO_PERIOD_SUMMARY),
            file_signature=(summary_stat.st_mtime_ns, summary_stat.st_size),
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(
            "The Two Period summary CSV could not be loaded: "
            f"{error}"
        )
        return
    sensor_data = summary.loc[summary["physical_fbg"] == sensor].copy()
    if sensor_data.empty:
        st.error("No recordings with accepted axle peaks were available for this physical FBG.")
        return
    def period_y_domain(period: str) -> tuple[float, float]:
        period_data = sensor_data.loc[sensor_data["measurement_period"] == period]
        lower_values = period_data["mean_minus_std_pm"].dropna()
        upper_values = period_data["mean_plus_std_pm"].dropna()
        if min(len(lower_values), len(upper_values)) >= 20:
            lower = float(lower_values.quantile(0.01))
            upper = float(upper_values.quantile(0.99))
        else:
            lower = float(lower_values.min())
            upper = float(upper_values.max())
        padding = max(1.0, 0.04 * (upper - lower))
        return lower - padding, upper + padding

    # Use Period 1 as the fixed reference range so both periods are compared
    # against exactly the same amplitude scale.
    period_1_domain = period_y_domain("Period 1 (May-Jun)")
    pick = alt.selection_point(
        name="trend_pick",
        fields=["file", "physical_fbg"],
        on="click",
        clear="dblclick",
    )
    period_1 = two_period_panel(
        sensor_data, "Period 1 (May-Jun)", "Period 1: May–June", 690,
        period_1_domain,
    )
    period_2 = two_period_panel(
        sensor_data, "Period 2 (Jul)", "Period 2: July", 430,
        period_1_domain,
    )
    chart = (
        alt.hconcat(period_1, period_2, spacing=22)
        .resolve_scale(y="shared")
        .add_params(pick)
    )
    try:
        event = st.altair_chart(
            chart,
            key=f"two_period_trend_fbg{sensor}",
            on_select="rerun",
            selection_mode="trend_pick",
            width="stretch",
        )
        chosen = event.selection.get("trend_pick", [])
    except StreamlitAPIException as exc:
        if "Selections are not yet supported for multi-view charts" not in str(exc):
            raise
        # Streamlit <1.55 cannot return selections from a composed chart. Keep
        # the mean points clickable in that environment with two simple views.
        simple_events = []
        for period, title, width in (
            ("Period 1 (May-Jun)", "Period 1: May–June", 690),
            ("Period 2 (Jul)", "Period 2: July", 430),
        ):
            selected = sensor_data.loc[
                sensor_data["measurement_period"] == period
            ].copy()
            y_domain = period_1_domain
            selected["span_low"] = (
                selected["mean_peak_change_pm"]
                - selected["std_peak_change_pm"] / 2
            )
            selected["span_high"] = (
                selected["mean_peak_change_pm"]
                + selected["std_peak_change_pm"] / 2
            )
            # A single line mark keeps the chart selectable on Streamlit 1.52.
            # Three ordered rows per recording draw the thin low-to-high span
            # and expose the mean row as a visible point on that line.
            span_rows = pd.concat(
                [
                    selected.assign(span_value=selected["span_low"], span_order=0),
                    selected.assign(
                        span_value=selected["mean_peak_change_pm"], span_order=1
                    ),
                    selected.assign(span_value=selected["span_high"], span_order=2),
                ],
                ignore_index=True,
            )
            simple_chart = alt.Chart(span_rows).mark_line(
                point=alt.OverlayMarkDef(filled=True, strokeWidth=0),
                strokeWidth=1,
                clip=True,
            ).encode(
                x=alt.X(
                    "recording_time_utc:T",
                    title=title,
                    axis=alt.Axis(format="%d %b", labelAngle=-45),
                ),
                y=alt.Y(
                    "span_value:Q",
                    title="Detrended axle-peak amplitude [pm]",
                    scale=alt.Scale(domain=list(y_domain), zero=False),
                ),
                detail="file:N",
                order=alt.Order("span_order:Q"),
                color=alt.Color(
                    "measurement_period:N",
                    scale=alt.Scale(
                        domain=["Period 1 (May-Jun)", "Period 2 (Jul)"],
                        range=["#1261a0", "#1b8a5a"],
                    ),
                    legend=alt.Legend(title="Measurement period"),
                ),
                fill=alt.condition(
                    alt.datum.span_order == 1,
                    alt.Color(
                        "measurement_period:N",
                        scale=alt.Scale(
                            domain=["Period 1 (May-Jun)", "Period 2 (Jul)"],
                            range=["#1261a0", "#1b8a5a"],
                        ),
                    ),
                    alt.value("transparent"),
                ),
                tooltip=[
                    "file:N",
                    "mean_peak_change_pm:Q",
                    "std_peak_change_pm:Q",
                ],
            ).add_params(pick).properties(width=width, height=430)
            simple_events.append(
                st.altair_chart(
                    simple_chart,
                    key=f"two_period_trend_legacy_{sensor}_{period}",
                    on_select="rerun",
                    selection_mode="trend_pick",
                    width="stretch",
                )
            )
        chosen = simple_events[0].selection.get("trend_pick", [])
        if not chosen:
            chosen = simple_events[1].selection.get("trend_pick", [])
    if not chosen:
        st.info("Click a mean dot to open its raw signal and latest axle detection.")
        return
    if isinstance(chosen, dict):
        chosen = [chosen]
    selected_file = str(chosen[0].get("file", ""))
    selected_rows = sensor_data.loc[sensor_data["file"] == selected_file]
    path_rows = recordings.loc[
        recordings["path"].map(lambda value: Path(value).name == selected_file)
    ]
    if selected_rows.empty or path_rows.empty:
        st.warning(f"The source recording for {selected_file} was not found in the indexed data.")
        return

    selected_row = selected_rows.iloc[0]
    selected_path = str(path_rows.iloc[0]["path"])
    selected_timestamp = timestamp_from_name(Path(selected_path))
    with st.spinner("Opening the selected recording and running the latest detector…"):
        result = analyse_recording(selected_path, sensor)
    latest_indices = np.asarray(result["peak_indices"], dtype=int)

    st.markdown(f"**Selected recording:** `{selected_file}`")
    with st.container(horizontal=True):
        st.metric("Detrended axle mean", f"{selected_row['mean_peak_change_pm']:.2f} pm", border=True)
        st.metric("Axle standard deviation", f"{selected_row['std_peak_change_pm']:.2f} pm", border=True)
        st.metric("Detected axle count used by dot", int(selected_row["peak_count"]), border=True)
        st.metric("Latest self-calibrating count", len(latest_indices), border=True)
        if sensor == 9:
            st.metric("Latest polarity", str(result["polarity"]).title(), border=True)

    if len(latest_indices):
        detail_view = st.segmented_control(
            "Detail time range",
            ("Detected passage", "Whole recording"),
            default="Detected passage",
            key=f"trend_detail_view_{sensor}",
        )
    else:
        detail_view = "Whole recording"

    if detail_view == "Detected passage":
        # All physical FBGs observe the same train passage in a recording.
        # FBG 1 provides the stable reference interval used by the overview,
        # while the selected sensor keeps its own signal and axle locations.
        passage_reference = (
            result if sensor == 1 else analyse_recording(selected_path, 1)
        )
        limits = passage_limits(passage_reference)
    else:
        limits = None
    raw_figure = signal_figure(
        np.asarray(result["time_s"]),
        np.asarray(result["raw_nm"]),
        "Raw wavelength with latest accepted axle locations",
        "Raw wavelength [nm]",
        latest_indices,
        limits,
        selected_timestamp,
        show_full_y_values=True,
    )
    st.pyplot(raw_figure, width="stretch")
    plt.close(raw_figure)

    axle_figure = signal_figure(
        np.asarray(result["time_s"]),
        np.asarray(result["processed_nm"]) * 1000.0,
        "Latest self-calibrating axle-count result",
        "Filtered response [pm]",
        latest_indices,
        limits,
        selected_timestamp,
    )
    st.pyplot(axle_figure, width="stretch")
    plt.close(axle_figure)

    st.caption(
        "The overview dot and opened axle-count plot use the same accepted axle locations."
    )


def main() -> None:
    st.set_page_config(page_title="Train Monitoring", layout="wide")
    st.title("Train Monitoring — FBG recording analysis")

    with st.sidebar:
        st.header("Data source")
        local_default = default_data_root()
        source_mode = st.radio(
            "Recording source",
            ("Local/server folder", "Upload recordings"),
            index=0 if Path(local_default).is_dir() else 1,
            help=(
                "Use the folder option on your own computer. Use uploads when the "
                "app is hosted and cannot access your local drive."
            ),
        )
        if source_mode == "Local/server folder":
            data_root = st.text_input(
                "Raw-data directory",
                value=local_default,
                help="Choose the folder containing both measurement-period subfolders.",
            )
            if st.button("Refresh recording index"):
                index_recordings.clear()
        else:
            uploaded_recordings = st.file_uploader(
                "Upload timestamped FBG text recordings",
                type="txt",
                accept_multiple_files=True,
                help=(
                    "Files must retain names such as 20260520T123416,727Z.txt. "
                    "Uploaded data is temporary and disappears when the hosted session ends."
                ),
            )
            if not uploaded_recordings:
                st.info("Upload one or more timestamped FBG recordings to begin.")
                st.stop()
            data_root = persist_uploaded_recordings(uploaded_recordings)

    recordings = index_recordings(data_root)
    if recordings.empty:
        st.error(
            "No timestamped FBG text recordings were found. Enter the raw_data folder "
            "or the Railway Monitoring project folder in the sidebar, without needing quotes."
        )
        st.code(
            str(
                Path.home()
                / "OneDrive"
                / "Desktop"
                / "Railway Monitoring"
                / "data"
                / "data"
                / "raw_data"
            )
        )
        st.stop()
    with st.sidebar:
        st.success(f"Found {len(recordings):,} unique timestamped recordings")

    view_options = ["Recording explorer"]
    view_options.append("Two-period amplitude overview")
    app_view = st.segmented_control(
        "Analysis view",
        view_options,
        default="Recording explorer",
        key="analysis_view",
    )
    if app_view == "Two-period amplitude overview":
        render_two_period_overview(recordings)
        return

    controls = st.columns([1.05, 1.65, 0.75, 0.95])
    available_dates = sorted(recordings["date"].unique())
    selected_date = controls[0].selectbox(
        "1. Date", available_dates,
        format_func=lambda value: value.strftime("%d %B %Y"),
    )
    day_rows = recordings.loc[recordings["date"] == selected_date].copy()
    recording_labels = {
        row.path: f"{row.time} UTC — {Path(row.path).name}"
        for row in day_rows.itertuples()
    }
    selected_path = controls[1].selectbox(
        "2. Recording", list(recording_labels), format_func=recording_labels.get
    )
    selected_timestamp = timestamp_from_name(Path(selected_path))
    selected_head = controls[2].selectbox("3. Head", ["All", *sorted(HEAD_SENSORS)])
    available_fbgs = (
        list(range(1, 13))
        if selected_head == "All"
        else list(HEAD_SENSORS[int(selected_head)])
    )
    selected_fbg = controls[3].selectbox(
        "4. Physical FBG",
        ["All", *available_fbgs],
        # Keep the comparison option, but do not analyse all 12 large signals
        # automatically whenever the date or recording changes.
        index=1,
    )

    selected_sensors = (
        available_fbgs if selected_fbg == "All" else [int(selected_fbg)]
    )
    results: list[tuple[int, dict[str, object]]] = []
    failures: list[str] = []
    with st.spinner(
        f"Counting axles with {len(selected_sensors)} physical FBG"
        f"{'s' if len(selected_sensors) != 1 else ''}…"
    ):
        for physical_fbg in selected_sensors:
            try:
                result = analyse_recording(selected_path, physical_fbg)
                results.append((physical_fbg, result))
            except Exception as exc:
                failures.append(f"Physical FBG {physical_fbg}: {exc}")

    if not results:
        st.error("None of the selected physical FBG signals could be analysed.")
        for failure in failures:
            st.warning(failure)
        st.stop()

    count_rows = []
    for physical_fbg, result in results:
        head = next(
            head_number
            for head_number, sensors in HEAD_SENSORS.items()
            if physical_fbg in sensors
        )
        count_rows.append({
            "Head": head,
            "Physical FBG": physical_fbg,
            "Axle count": len(np.asarray(result["peak_indices"], dtype=int)),
            "Polarity": str(result["polarity"]).title(),
        })

    if selected_fbg == "All":
        st.subheader("Axle count")
        count_table = pd.DataFrame(count_rows)
        selection_scope = (
            "overlay_fbg_selection_"
            f"{Path(selected_path).stem}_{selected_head}"
        )
        selection_values_key = f"{selection_scope}_values"
        select_all_key = f"{selection_scope}_all"
        editor_revision_key = f"{selection_scope}_revision"
        available_sensor_ids = count_table["Physical FBG"].astype(int).tolist()

        stored_selection = st.session_state.get(selection_values_key)
        if (
            not isinstance(stored_selection, dict)
            or set(stored_selection) != set(available_sensor_ids)
        ):
            stored_selection = {sensor_id: True for sensor_id in available_sensor_ids}
            st.session_state[selection_values_key] = stored_selection
            st.session_state[editor_revision_key] = 0

        all_selected = all(stored_selection.values())
        if st.session_state.get(select_all_key) != all_selected:
            st.session_state[select_all_key] = all_selected

        def apply_select_all() -> None:
            selected = bool(st.session_state[select_all_key])
            st.session_state[selection_values_key] = {
                sensor_id: selected for sensor_id in available_sensor_ids
            }
            st.session_state[editor_revision_key] = (
                int(st.session_state.get(editor_revision_key, 0)) + 1
            )

        st.checkbox(
            "Select all",
            key=select_all_key,
            on_change=apply_select_all,
        )
        count_table.insert(
            0,
            "Plot",
            count_table["Physical FBG"].astype(int).map(stored_selection),
        )
        edited_count_table = st.data_editor(
            count_table,
            hide_index=True,
            width="stretch",
            disabled=["Head", "Physical FBG", "Axle count", "Polarity"],
            column_config={
                "Plot": st.column_config.CheckboxColumn(required=True),
            },
            key=(
                f"{selection_scope}_editor_"
                f"{st.session_state.get(editor_revision_key, 0)}"
            ),
        )
        updated_selection = {
            int(row["Physical FBG"]): bool(row["Plot"])
            for _, row in edited_count_table.iterrows()
        }
        if updated_selection != stored_selection:
            st.session_state[selection_values_key] = updated_selection
            st.rerun()

        plotted_fbgs = {
            sensor_id for sensor_id, selected in updated_selection.items() if selected
        }
        plotted_results = [
            (physical_fbg, result)
            for physical_fbg, result in results
            if physical_fbg in plotted_fbgs
        ]
        if plotted_results:
            figure = axle_count_overlay_figure(plotted_results, selected_timestamp)
            st.pyplot(figure, width="stretch")
            plt.close(figure)

    for failure in failures:
        st.warning(failure)

    # Keep the multi-sensor view focused on comparison. The heavier raw,
    # derivative, frequency and diagnostic panels appear only after the user
    # explicitly chooses one physical FBG in the main selector.
    if selected_fbg == "All":
        return

    detail_sensor = int(selected_fbg)
    detail_result = next(
        result for physical_fbg, result in results
        if physical_fbg == detail_sensor
    )
    detail_peaks = np.asarray(detail_result["peak_indices"], dtype=int)
    detail_head = next(
        head_number
        for head_number, sensors in HEAD_SENSORS.items()
        if detail_sensor in sensors
    )

    st.subheader(f"Detailed analysis · Head {detail_head} · Physical FBG {detail_sensor}")
    with st.container(horizontal=True):
        st.metric("Axle count", len(detail_peaks))
        if detail_sensor == 9:
            st.metric("Selected polarity", str(detail_result["polarity"]).title())

    if len(detail_peaks):
        plot_view = st.segmented_control(
            "Plot time range",
            ("Detected passage", "Whole recording"),
            default="Detected passage",
            help="Detected passage adds two seconds before the first and after the last accepted axle.",
        )
    else:
        plot_view = "Whole recording"
    plot_limits = (
        passage_limits(detail_result)
        if plot_view == "Detected passage" else None
    )

    raw_figure = signal_figure(
        np.asarray(detail_result["time_s"]),
        np.asarray(detail_result["raw_nm"]),
        "Raw wavelength with accepted axle locations",
        "Raw wavelength [nm]",
        detail_peaks,
        plot_limits,
        selected_timestamp,
        show_full_y_values=True,
    )
    st.pyplot(raw_figure, width="stretch")
    plt.close(raw_figure)

    processed_figure = signal_figure(
        np.asarray(detail_result["time_s"]),
        np.asarray(detail_result["processed_nm"]) * 1000.0,
        "Latest self-calibrating axle detector",
        "Filtered response [pm]",
        detail_peaks,
        plot_limits,
        selected_timestamp,
    )
    st.pyplot(processed_figure, width="stretch")
    plt.close(processed_figure)

    derivative_figure = signal_figure(
        np.asarray(detail_result["time_s"]),
        np.asarray(detail_result["derivative_nm_s"]) * 1000.0,
        "Derivative of the filtered response",
        "Derivative [pm/s]",
        x_limits=plot_limits,
        recording_timestamp=selected_timestamp,
    )
    st.pyplot(derivative_figure, width="stretch")
    plt.close(derivative_figure)

    st.subheader("Frequency-domain analysis")
    frequency_controls = st.columns(2)
    method = frequency_controls[0].selectbox(
        "Method", ("FFT", "Welch PSD", "Morlet wavelet", "EMD", "VMD")
    )
    passage_only = False
    try:
        show_frequency_analysis(
            method, detail_result, passage_only, selected_timestamp
        )
    except Exception as exc:
        st.error(str(exc))

    with st.expander("Detector diagnostics"):
        diagnostics = pd.DataFrame({
            "Value": {
                "Selected polarity": str(detail_result["polarity"]),
                "Background sigma [pm]": float(detail_result["background_sigma_nm"]) * 1000.0,
                "Correlation time [s]": float(detail_result["correlation_time_s"]),
                "Polarity confidence": float(detail_result["polarity_confidence"]),
                "Positive axle count": int(detail_result["positive_axle_count"]),
                "Negative axle count": int(detail_result["negative_axle_count"]),
                "Quality score": float(detail_result["quality_score"]),
                "Positive height [sigma]": float(detail_result["positive_height_sigma"]),
                "Positive prominence [sigma]": float(detail_result["positive_prominence_sigma"]),
                "Negative height [sigma]": float(detail_result["negative_height_sigma"]),
                "Negative prominence [sigma]": float(detail_result["negative_prominence_sigma"]),
            }
        })
        diagnostics["Value"] = diagnostics["Value"].astype(str)
        st.dataframe(diagnostics, width="stretch")


if __name__ == "__main__":
    main()

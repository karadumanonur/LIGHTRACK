"""Experimental axle detector with fixed installation knowledge.

This module is intentionally separate from ``self_calibrating_axle_detector``.
It leaves the working detector unchanged while testing three simplifications:

* the acquisition rate is fixed at 500 Hz, with timestamps used for validation;
* response polarity is configured by physical FBG number, while FBG 9 is
  evaluated in both directions per recording; and
* the active mask is one strict contiguous interval supported by filtered
  excursions at or above the existing 20 pm empty-record gate.

The height, prominence, evidence, width, valley, ringing, and broad-response
rules are reused from the working self-calibrating detector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

try:
    from .self_calibrating_axle_detector import (
        AdaptiveSettings,
        PolarityResult,
        analyse_polarity,
        estimate_correlation_samples,
        merge_ringing_candidates,
        polarity_impulse_sharpness,
        quiet_background_mask,
        robust_location_scale,
        separated_broad_width_cluster,
    )
    from .sensor12_peak_tracker import butterworth_process
except ImportError:  # Support direct execution from the legacy folder.
    from self_calibrating_axle_detector import (
        AdaptiveSettings,
        PolarityResult,
        analyse_polarity,
        estimate_correlation_samples,
        merge_ringing_candidates,
        polarity_impulse_sharpness,
        quiet_background_mask,
        robust_location_scale,
        separated_broad_width_cluster,
    )
    from sensor12_peak_tracker import butterworth_process


EXPECTED_SAMPLING_RATE_HZ = 500.0
FIXED_HIGHPASS_HZ = 0.5
FIXED_LOWPASS_HZ = 40.0

POLARITY_BY_PHYSICAL_FBG = {
    physical_fbg: ("negative" if physical_fbg == 9 else "positive")
    for physical_fbg in range(1, 13)
}


@dataclass(frozen=True)
class StrictDetectorSettings:
    """Settings specific to the experimental wrapper."""

    sampling_rate_hz: float = EXPECTED_SAMPLING_RATE_HZ
    maximum_sampling_rate_error_hz: float = 0.5
    highpass_hz: float = FIXED_HIGHPASS_HZ
    lowpass_hz: float = FIXED_LOWPASS_HZ
    event_padding_blocks: int = 1
    adaptive: AdaptiveSettings = AdaptiveSettings()


@dataclass
class StrictAxleDetection:
    """Result and diagnostics for one physical FBG recording."""

    indices: np.ndarray
    selected_polarity: str
    physical_fbg: int
    stored_column: int | None
    stored_order: str | None
    filtered_signal_nm: np.ndarray
    active_mask: np.ndarray
    background_mask: np.ndarray
    gate_applied: bool
    maximum_excursion_nm: float
    background_location_nm: float
    background_sigma_nm: float
    correlation_samples: int
    quality_score: float
    quality_label: str
    polarity_result: PolarityResult | None
    positive_result: PolarityResult | None
    negative_result: PolarityResult | None
    polarity_confidence: float

    @property
    def axle_count(self) -> int:
        return int(len(self.indices))


def validate_fixed_sampling_rate(
    time_s: np.ndarray,
    settings: StrictDetectorSettings,
) -> None:
    """Validate timestamps while keeping the processing rate fixed at 500 Hz."""
    time_s = np.asarray(time_s, dtype=float)
    steps = np.diff(time_s)
    valid = steps[np.isfinite(steps) & (steps > 0)]
    if len(valid) != len(steps):
        raise ValueError("Timestamps must be finite and strictly increasing")
    observed_hz = float(1.0 / np.median(valid))
    if abs(observed_hz - settings.sampling_rate_hz) > settings.maximum_sampling_rate_error_hz:
        raise ValueError(
            f"Expected {settings.sampling_rate_hz:.1f} Hz data but timestamps imply "
            f"{observed_hz:.3f} Hz"
        )


def strict_active_event_mask(
    filtered_nm: np.ndarray,
    correlation_samples: int,
    settings: StrictDetectorSettings,
) -> np.ndarray:
    """Return one padded event interval supported by absolute 20 pm crossings.

    The current recordings contain at most one train event per physical file.
    The first and last samples whose centered filtered amplitude reaches the
    absolute gate define the event. One learned block is added on each side so
    the mask includes response onset, decay, and boundary axles.
    """
    values = np.asarray(filtered_nm, dtype=float)
    location, _ = robust_location_scale(values)
    gate_crossings = np.flatnonzero(
        np.abs(values - location) >= settings.adaptive.absolute_gate_nm
    )
    mask = np.zeros(len(values), dtype=bool)
    if not len(gate_crossings):
        return mask

    block_size = max(64, 20 * int(correlation_samples))
    padding = settings.event_padding_blocks * block_size
    start = max(0, int(gate_crossings[0]) - padding)
    stop = min(len(values), int(gate_crossings[-1]) + padding + 1)
    mask[start:stop] = True
    return mask


def _quality_label(result: PolarityResult) -> str:
    if not len(result.accepted_indices):
        return "no statistically detectable event"
    if result.quality_score >= 0.65:
        return "strong"
    if result.quality_score >= 0.30:
        return "usable with review"
    return "uncertain"


def preserve_physically_strong_candidates(
    result: PolarityResult,
    signal_nm: np.ndarray,
    time_s: np.ndarray,
    sign: int,
    background_location_nm: float,
    background_sigma_nm: float,
    settings: StrictDetectorSettings,
) -> PolarityResult:
    """Keep clear axle responses even when relative clustering splits amplitudes.

    FBG 6 can produce alternating strong and weaker axle responses. Both groups
    are far above the installation's 20 pm physical event floor, but relative
    evidence clustering can label the weaker group as noise. This universal
    safeguard restores a preliminary candidate only when both its height and
    prominence independently reach that same physical floor. The normal
    ringing and broad-response filters are then reapplied to the combined set.
    """
    if not len(result.preliminary_indices):
        return result

    oriented = sign * (np.asarray(signal_nm, dtype=float) - background_location_nm)
    local_indices, _ = find_peaks(oriented)
    prominences = peak_prominences(oriented, local_indices)[0]
    widths_s = peak_widths(oriented, local_indices, rel_height=0.5)[0] / settings.sampling_rate_hz
    local_position = {int(index): position for position, index in enumerate(local_indices)}

    preliminary_positions = np.asarray(
        [local_position[int(index)] for index in result.preliminary_indices], dtype=int
    )
    physical_floor = settings.adaptive.absolute_gate_nm
    strong_mask = (
        (oriented[result.preliminary_indices] >= physical_floor)
        & (prominences[preliminary_positions] >= physical_floor)
    )
    strong_indices = result.preliminary_indices[strong_mask]
    restored = np.setdiff1d(strong_indices, result.accepted_indices, assume_unique=False)
    if not len(restored):
        return result

    combined_indices = np.union1d(result.accepted_indices, strong_indices).astype(int)
    combined_positions = np.asarray(
        [local_position[int(index)] for index in combined_indices], dtype=int
    )
    combined_widths = widths_s[combined_positions]
    combined_prominences = prominences[combined_positions]

    keep = merge_ringing_candidates(
        oriented,
        combined_indices,
        combined_widths,
        combined_prominences,
        time_s,
        background_sigma_nm,
        result.valley_threshold_sigma,
    )
    accepted_indices = combined_indices[keep]
    accepted_widths = combined_widths[keep]
    accepted_prominences = combined_prominences[keep]

    if len(accepted_indices):
        broad_keep = separated_broad_width_cluster(
            accepted_widths,
            oriented[accepted_indices]
            / max(result.height_threshold_nm, np.finfo(float).eps),
            accepted_prominences
            / max(result.prominence_threshold_nm, np.finfo(float).eps),
        )
        accepted_indices = accepted_indices[broad_keep]
        accepted_widths = accepted_widths[broad_keep]
        accepted_prominences = accepted_prominences[broad_keep]

    result.accepted_indices = accepted_indices.astype(int)
    result.accepted_width_s = accepted_widths.astype(float)
    result.accepted_prominence_nm = accepted_prominences.astype(float)
    result.accepted_height_sigma = (
        oriented[accepted_indices] / background_sigma_nm
        if len(accepted_indices)
        else np.array([], dtype=float)
    )
    retained_count = len(np.intersect1d(restored, accepted_indices))
    if retained_count:
        result.warnings.append(
            f"preserved {retained_count} candidate"
            f"{'s' if retained_count != 1 else ''} above the 20 pm physical floor"
        )
    return result


def detect_strict_axles(
    raw_wavelength_nm: np.ndarray,
    time_s: np.ndarray,
    physical_fbg: int,
    *,
    stored_column: int | None = None,
    stored_order: str | None = None,
    settings: StrictDetectorSettings | None = None,
) -> StrictAxleDetection:
    """Filter and count one physical FBG without changing the working detector."""
    settings = settings or StrictDetectorSettings()
    if physical_fbg not in POLARITY_BY_PHYSICAL_FBG:
        raise ValueError("physical_fbg must be between 1 and 12")

    raw = np.asarray(raw_wavelength_nm, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if raw.ndim != 1 or time_s.ndim != 1 or len(raw) != len(time_s) or len(raw) < 64:
        raise ValueError("Signal and time must be equal-length one-dimensional arrays")
    if np.any(~np.isfinite(raw)):
        raise ValueError("Signal contains non-finite values")

    validate_fixed_sampling_rate(time_s, settings)
    filtered = butterworth_process(
        raw,
        settings.sampling_rate_hz,
        settings.highpass_hz,
        settings.lowpass_hz,
    )
    initial_location, _ = robust_location_scale(filtered)
    maximum_excursion = float(np.max(np.abs(filtered - initial_location)))
    polarity = POLARITY_BY_PHYSICAL_FBG[physical_fbg]
    polarity_confidence = 1.0
    if physical_fbg == 9:
        centered = filtered - initial_location
        tail_count = max(1, int(np.ceil(np.sqrt(len(centered)))))
        ordered = np.sort(centered)
        initial_positive_evidence = float(np.mean(ordered[-tail_count:]))
        initial_negative_evidence = float(np.mean(-ordered[:tail_count]))
        polarity = (
            "positive"
            if initial_positive_evidence >= initial_negative_evidence
            else "negative"
        )
        initial_evidence_total = (
            abs(initial_positive_evidence) + abs(initial_negative_evidence)
        )
        polarity_confidence = (
            abs(initial_positive_evidence - initial_negative_evidence)
            / initial_evidence_total
            if initial_evidence_total > 0
            else 0.0
        )

    if maximum_excursion < settings.adaptive.absolute_gate_nm:
        empty = np.zeros(len(filtered), dtype=bool)
        return StrictAxleDetection(
            indices=np.array([], dtype=int),
            selected_polarity=polarity,
            physical_fbg=physical_fbg,
            stored_column=stored_column,
            stored_order=stored_order,
            filtered_signal_nm=filtered,
            active_mask=empty.copy(),
            background_mask=np.ones(len(filtered), dtype=bool),
            gate_applied=True,
            maximum_excursion_nm=maximum_excursion,
            background_location_nm=initial_location,
            background_sigma_nm=float("nan"),
            correlation_samples=0,
            quality_score=0.0,
            quality_label="no statistically detectable event",
            polarity_result=None,
            positive_result=None,
            negative_result=None,
            polarity_confidence=float(polarity_confidence),
        )

    correlation_samples = estimate_correlation_samples(filtered, settings.sampling_rate_hz)
    active_mask = strict_active_event_mask(filtered, correlation_samples, settings)

    learned_quiet = quiet_background_mask(
        filtered - initial_location,
        correlation_samples,
        settings.adaptive,
    )
    background_mask = learned_quiet & ~active_mask
    minimum_background_samples = max(64, 20 * correlation_samples)
    if int(np.sum(background_mask)) < minimum_background_samples:
        background_mask = learned_quiet

    background_location, background_sigma = robust_location_scale(filtered[background_mask])
    def evaluate_polarity(sign: int) -> PolarityResult:
        evaluated = analyse_polarity(
            filtered,
            time_s,
            settings.sampling_rate_hz,
            sign,
            background_mask,
            active_mask,
            background_location,
            background_sigma,
            settings.adaptive,
        )
        return preserve_physically_strong_candidates(
            evaluated,
            filtered,
            time_s,
            sign,
            background_location,
            background_sigma,
            settings,
        )

    positive_result: PolarityResult | None = None
    negative_result: PolarityResult | None = None
    polarity_confidence = 1.0
    if physical_fbg == 9:
        positive_result = evaluate_polarity(1)
        negative_result = evaluate_polarity(-1)
        positive_impulse = polarity_impulse_sharpness(positive_result)
        negative_impulse = polarity_impulse_sharpness(negative_result)
        if np.isclose(positive_impulse, negative_impulse):
            active_centered = filtered[active_mask] - background_location
            if not len(active_centered):
                active_centered = filtered - background_location
            tail_count = max(1, int(np.ceil(np.sqrt(len(active_centered)))))
            ordered = np.sort(active_centered)
            positive_evidence = float(np.mean(ordered[-tail_count:]))
            negative_evidence = float(np.mean(-ordered[:tail_count]))
            result = (
                positive_result
                if positive_evidence >= negative_evidence
                else negative_result
            )
        else:
            result = (
                positive_result
                if positive_impulse > negative_impulse
                else negative_result
            )
        polarity = "positive" if result.sign > 0 else "negative"
        impulse_total = abs(positive_impulse) + abs(negative_impulse)
        polarity_confidence = (
            abs(positive_impulse - negative_impulse) / impulse_total
            if impulse_total > 0
            else 0.0
        )
    else:
        sign = 1 if polarity == "positive" else -1
        result = evaluate_polarity(sign)
        if sign > 0:
            positive_result = result
        else:
            negative_result = result

    return StrictAxleDetection(
        indices=result.accepted_indices,
        selected_polarity=polarity,
        physical_fbg=physical_fbg,
        stored_column=stored_column,
        stored_order=stored_order,
        filtered_signal_nm=filtered,
        active_mask=active_mask,
        background_mask=background_mask,
        gate_applied=False,
        maximum_excursion_nm=maximum_excursion,
        background_location_nm=background_location,
        background_sigma_nm=background_sigma,
        correlation_samples=correlation_samples,
        quality_score=result.quality_score,
        quality_label=_quality_label(result),
        polarity_result=result,
        positive_result=positive_result,
        negative_result=negative_result,
        polarity_confidence=float(polarity_confidence),
    )


def load_physical_fbg(
    path: Path,
    physical_fbg: int,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Load one physical FBG while handling both stored column orders."""
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        decimal=",",
        engine="c",
    )
    frame = frame.apply(pd.to_numeric, errors="coerce")
    medians = frame.loc[:, [1, 12]].median()
    ascending = bool(medians.loc[1] < medians.loc[12])
    stored_column = physical_fbg if ascending else 13 - physical_fbg
    selected = (
        frame.loc[:, [0, stored_column]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(0)
        .drop_duplicates(0)
    )
    if len(selected) < 64:
        raise ValueError(f"{path} does not contain enough valid samples")
    return (
        selected[0].to_numpy(dtype=float),
        selected[stored_column].to_numpy(dtype=float),
        int(stored_column),
        "ascending" if ascending else "descending",
    )


def diagnostic_dict(detection: StrictAxleDetection) -> dict[str, object]:
    """Return compact JSON-safe diagnostics."""
    return {
        "axle_count": detection.axle_count,
        "physical_fbg": detection.physical_fbg,
        "stored_column": detection.stored_column,
        "stored_order": detection.stored_order,
        "selected_polarity": detection.selected_polarity,
        "polarity_confidence": detection.polarity_confidence,
        "positive_axle_count": (
            len(detection.positive_result.accepted_indices)
            if detection.positive_result is not None else 0
        ),
        "negative_axle_count": (
            len(detection.negative_result.accepted_indices)
            if detection.negative_result is not None else 0
        ),
        "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
        "maximum_excursion_pm": detection.maximum_excursion_nm * 1000.0,
        "absolute_gate_pm": AdaptiveSettings().absolute_gate_nm * 1000.0,
        "gate_applied": detection.gate_applied,
        "active_fraction": float(np.mean(detection.active_mask)),
        "background_fraction": float(np.mean(detection.background_mask)),
        "correlation_samples": detection.correlation_samples,
        "background_sigma_pm": detection.background_sigma_nm * 1000.0,
        "quality_score": detection.quality_score,
        "quality_label": detection.quality_label,
    }


def plot_detection(
    time_s: np.ndarray,
    detection: StrictAxleDetection,
    output_path: Path | None = None,
) -> None:
    """Plot the strict active interval and accepted axle peaks."""
    import matplotlib.pyplot as plt

    elapsed = np.asarray(time_s) - float(time_s[0])
    sign = 1 if detection.selected_polarity == "positive" else -1
    oriented_pm = sign * (
        detection.filtered_signal_nm - detection.background_location_nm
    ) * 1000.0
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(elapsed, oriented_pm, color="black", linewidth=0.75, label="filtered signal")
    if np.any(detection.active_mask):
        ax.fill_between(
            elapsed,
            np.min(oriented_pm),
            np.max(oriented_pm),
            where=detection.active_mask,
            color="tab:green",
            alpha=0.12,
            label="strict active interval",
        )
    if detection.axle_count:
        ax.scatter(
            elapsed[detection.indices],
            oriented_pm[detection.indices],
            color="tab:red",
            s=28,
            zorder=3,
            label=f"accepted axles: {detection.axle_count}",
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Polarity-oriented filtered amplitude [pm]")
    ax.set_title(
        f"Physical FBG {detection.physical_fbg} | {detection.selected_polarity} | "
        f"count {detection.axle_count}"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--physical-fbg", type=int, default=1, choices=range(1, 13))
    parser.add_argument("--plot", type=Path, help="Optional output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_s, raw_nm, stored_column, stored_order = load_physical_fbg(
        args.recording,
        args.physical_fbg,
    )
    detection = detect_strict_axles(
        raw_nm,
        time_s,
        args.physical_fbg,
        stored_column=stored_column,
        stored_order=stored_order,
    )
    print(json.dumps(diagnostic_dict(detection), indent=2))
    if args.plot:
        plot_detection(time_s, detection, args.plot)


if __name__ == "__main__":
    main()

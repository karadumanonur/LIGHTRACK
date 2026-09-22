"""Self-calibrating axle-peak detector for FBG recordings.

Uses fixed Butterworth filtering (0.5-40 Hz by default), followed by
recording-specific background, height, prominence, peak-width, and valley
estimation. The strict detector and dashboard reuse the statistical helpers.
Height, prominence, and valley thresholds use the maximum quiet-background
measurement; no adaptive spectral passband or tail-probability setting is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import (
    find_peaks,
    peak_prominences,
    peak_widths,
)
from scipy.stats import norm

from sensor12_peak_tracker import (
    butterworth_process,
    load_fbg_file,
    sampling_rate,
)


PHYSICAL_TO_STORED = {physical: 13 - physical for physical in range(1, 13)}


@dataclass(frozen=True)
class AdaptiveSettings:
    """Sensor-independent statistical design settings."""

    quiet_fraction: float = 0.5
    minimum_quiet_blocks: int = 4
    minimum_background_peaks: int = 30
    minimum_width_model_peaks: int = 3
    width_outlier_probability: float = 0.01
    absolute_gate_nm: float = 0.020  # 0.020 nm = 20 pm mutlak gürültü eşiği


@dataclass
class PolarityResult:
    sign: int
    local_peak_count: int
    background_peak_count: int
    height_threshold_nm: float
    prominence_threshold_nm: float
    height_threshold_sigma: float
    prominence_threshold_sigma: float
    valley_threshold_sigma: float
    typical_width_s: float
    width_log_sigma: float
    evidence_threshold: float
    evidence_log_gap: float
    preliminary_indices: np.ndarray
    accepted_indices: np.ndarray
    accepted_width_s: np.ndarray
    accepted_prominence_nm: np.ndarray
    accepted_height_sigma: np.ndarray
    quality_score: float
    warnings: list[str]


@dataclass
class AdaptiveDetection:
    indices: np.ndarray
    signs: np.ndarray
    selected_polarity: str
    polarity_confidence: float
    background_location_nm: float
    background_sigma_nm: float
    correlation_time_s: float
    quiet_fraction_observed: float
    quality_score: float
    quality_label: str
    positive: PolarityResult
    negative: PolarityResult


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Return median location and Gaussian-calibrated MAD scale."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("No finite values are available for noise estimation")
    location = float(np.median(finite))
    sigma = float(1.4826 * np.median(np.abs(finite - location)))
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        raise ValueError("The processed signal has no measurable variation")
    return location, sigma


def estimate_correlation_samples(values: np.ndarray, fs_hz: float) -> int:
    """Estimate the noise correlation length at the first 1/e ACF crossing."""
    centered = np.asarray(values, dtype=float) - np.median(values)
    n = len(centered)
    maximum_lag = min(max(8, int(round(2.0 * fs_hz))), max(8, n // 20))
    if maximum_lag >= n:
        return 1
    variance = float(np.dot(centered, centered))
    if variance <= 0:
        return 1
    acf = np.array(
        [np.dot(centered[:-lag], centered[lag:]) / variance for lag in range(1, maximum_lag + 1)]
    )
    crossings = np.flatnonzero(acf <= np.exp(-1.0))
    return int(crossings[0] + 1) if len(crossings) else int(maximum_lag)


def quiet_background_mask(
    signal_nm: np.ndarray,
    correlation_samples: int,
    settings: AdaptiveSettings,
) -> np.ndarray:
    """Select low-energy blocks as background without locating the train."""
    n = len(signal_nm)
    block_size = max(64, 20 * correlation_samples)
    block_count = max(1, int(np.ceil(n / block_size)))
    blocks = np.array_split(np.arange(n), block_count)
    energies = np.array(
        [np.median(np.square(signal_nm[block] - np.median(signal_nm[block]))) for block in blocks]
    )
    keep_count = min(
        block_count,
        max(settings.minimum_quiet_blocks, int(np.ceil(settings.quiet_fraction * block_count))),
    )
    selected = np.argsort(energies)[:keep_count]
    mask = np.zeros(n, dtype=bool)
    for block_index in selected:
        mask[blocks[int(block_index)]] = True
    return mask


def active_event_mask(
    signal_nm: np.ndarray,
    correlation_samples: int,
    background_mask: np.ndarray,
    settings: AdaptiveSettings,
) -> np.ndarray:
    """Locate energetic train blocks relative to the learned quiet blocks."""
    n = len(signal_nm)
    block_size = max(64, 20 * correlation_samples)
    blocks = np.array_split(np.arange(n), max(1, int(np.ceil(n / block_size))))
    energies = np.array(
        [np.median(np.square(signal_nm[block] - np.median(signal_nm[block]))) for block in blocks]
    )
    quiet_block = np.array([np.mean(background_mask[block]) > 0.99 for block in blocks])
    quiet_energies = energies[quiet_block]
    if not len(quiet_energies):
        return ~background_mask
    threshold = maximum_background_threshold(quiet_energies)
    active_blocks = energies > threshold
    quiet_median = float(np.median(quiet_energies))
    expanded = active_blocks.copy()
    for index in np.flatnonzero(active_blocks):
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(blocks) and energies[neighbour] > quiet_median:
                expanded[neighbour] = True
    mask = np.zeros(n, dtype=bool)
    for block_index in np.flatnonzero(expanded):
        mask[blocks[int(block_index)]] = True
    return mask


def maximum_background_threshold(background_values: np.ndarray) -> float:
    """Use the largest finite background measurement as the threshold."""
    values = np.asarray(background_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No background peak features are available")
    return float(np.max(values))


def robust_log_width_model(width_s: np.ndarray) -> tuple[float, float]:
    positive = np.asarray(width_s, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if not len(positive):
        return float("nan"), float("nan")
    log_width = np.log(positive)
    location = float(np.median(log_width))
    scale = float(1.4826 * np.median(np.abs(log_width - location)))
    if not np.isfinite(scale):
        scale = float("nan")
    return float(np.exp(location)), float(scale)


def separated_broad_width_cluster(
    width_s: np.ndarray,
    height_ratio: np.ndarray,
    prominence_ratio: np.ndarray,
) -> np.ndarray:
    """Reject a distinct broad, weak response population."""
    widths = np.asarray(width_s, dtype=float)
    keep = np.ones(len(widths), dtype=bool)
    if len(widths) < 4 or np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        return keep

    evidence = np.minimum(
        np.asarray(height_ratio, dtype=float),
        np.asarray(prominence_ratio, dtype=float),
    )
    order = np.argsort(widths)
    values = np.log(widths[order])
    count = len(values)
    epsilon = np.finfo(float).eps
    variance_one = max(float(np.var(values)), epsilon)
    likelihood_one = -0.5 * count * (np.log(2.0 * np.pi * variance_one) + 1.0)
    bic_one = -2.0 * likelihood_one + 2.0 * np.log(count)

    best_bic_two = float("inf")
    best_split: int | None = None
    for split_candidate in range(1, count):
        lower_values = values[:split_candidate]
        upper_values = values[split_candidate:]
        pooled_squared_error = float(
            np.sum((lower_values - np.mean(lower_values)) ** 2)
            + np.sum((upper_values - np.mean(upper_values)) ** 2)
        )
        shared_variance = max(pooled_squared_error / count, epsilon)
        likelihood_two = (
            len(lower_values) * np.log(len(lower_values) / count)
            + len(upper_values) * np.log(len(upper_values) / count)
            - 0.5 * count * (np.log(2.0 * np.pi * shared_variance) + 1.0)
        )
        bic_two = -2.0 * likelihood_two + 4.0 * np.log(count)
        if bic_two < best_bic_two:
            best_bic_two = float(bic_two)
            best_split = split_candidate

    if best_split is None or best_bic_two >= bic_one:
        return keep

    lower = order[:best_split]
    broad = order[best_split:]
    if len(broad) > len(lower):
        return keep
    width_ratio = float(np.median(widths[broad]) / np.median(widths[lower]))
    evidence_ratio = float(
        np.median(evidence[broad]) / max(np.median(evidence[lower]), epsilon)
    )
    if width_ratio >= 2.0 and evidence_ratio <= 0.5:
        keep[broad] = False
    return keep


def separated_evidence_cluster(
    height_ratio: np.ndarray,
    prominence_ratio: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Separate coherent responses from marginal background exceedances."""
    evidence = np.minimum(height_ratio, prominence_ratio)
    if len(evidence) < 3:
        return np.ones(len(evidence), dtype=bool), 1.0, 0.0
    order = np.argsort(evidence)
    sorted_evidence = evidence[order]
    log_evidence = np.log(np.maximum(sorted_evidence, np.finfo(float).tiny))
    gaps = np.diff(log_evidence)
    largest_gap = int(np.argmax(gaps))

    epsilon = np.finfo(float).eps

    def best_bic_split(values: np.ndarray) -> int | None:
        count = len(values)
        if count < 4:
            return None
        variance_one = max(float(np.var(values)), epsilon)
        likelihood_one = -0.5 * count * (np.log(2.0 * np.pi * variance_one) + 1.0)
        bic_one = -2.0 * likelihood_one + 2.0 * np.log(count)
        best_bic_two = float("inf")
        best_split: int | None = None
        for split_candidate in range(1, count):
            lower = values[:split_candidate]
            upper = values[split_candidate:]
            pooled_squared_error = float(
                np.sum((lower - np.mean(lower)) ** 2)
                + np.sum((upper - np.mean(upper)) ** 2)
            )
            shared_variance = max(pooled_squared_error / count, epsilon)
            likelihood_two = (
                len(lower) * np.log(len(lower) / count)
                + len(upper) * np.log(len(upper) / count)
                - 0.5 * count * (np.log(2.0 * np.pi * shared_variance) + 1.0)
            )
            bic_two = -2.0 * likelihood_two + 4.0 * np.log(count)
            if bic_two < best_bic_two:
                best_bic_two = float(bic_two)
                best_split = split_candidate
        return best_split if best_split is not None and best_bic_two < bic_one else None

    relative_split = best_bic_split(log_evidence)
    first_kept = relative_split if relative_split is not None else 0
    if first_kept == 0:
        proposed = largest_gap + 1
        sample_adjusted_gap = np.log1p(np.sqrt(len(evidence)))
        if (
            gaps[largest_gap] > sample_adjusted_gap
            and proposed >= 2
            and len(evidence) - proposed >= 2
        ):
            first_kept = proposed
    if first_kept == 0:
        return np.ones(len(evidence), dtype=bool), 1.0, float(gaps[largest_gap])
    split = first_kept - 1
    threshold = float(np.sqrt(sorted_evidence[split] * sorted_evidence[split + 1]))
    return evidence >= threshold, threshold, float(gaps[split])


def background_valley_depths(
    oriented_nm: np.ndarray,
    background_peaks: np.ndarray,
    background_mask: np.ndarray,
    sigma_nm: float,
) -> np.ndarray:
    """Measure valleys between adjacent maxima fully contained in quiet data."""
    depths: list[float] = []
    for left, right in zip(background_peaks[:-1], background_peaks[1:]):
        if right <= left + 1 or not np.all(background_mask[left : right + 1]):
            continue
        valley = float(np.min(oriented_nm[left : right + 1]))
        depth = min(float(oriented_nm[left]), float(oriented_nm[right])) - valley
        if depth >= 0:
            depths.append(depth / sigma_nm)
    return np.asarray(depths, dtype=float)


def merge_ringing_candidates(
    oriented_nm: np.ndarray,
    indices: np.ndarray,
    widths_s: np.ndarray,
    prominences_nm: np.ndarray,
    time_s: np.ndarray,
    sigma_nm: float,
    valley_threshold_sigma: float,
) -> np.ndarray:
    """Merge close, shallow-valley maxima without a fixed time-distance rule."""
    if len(indices) < 2:
        return np.arange(len(indices), dtype=int)
    kept: list[int] = [0]
    for current in range(1, len(indices)):
        previous = kept[-1]
        left = int(indices[previous])
        right = int(indices[current])
        gap_s = float(time_s[right] - time_s[left])
        learned_response_span = max(float(widths_s[previous]), float(widths_s[current]))
        valley = float(np.min(oriented_nm[left : right + 1]))
        valley_depth_sigma = (
            min(float(oriented_nm[left]), float(oriented_nm[right])) - valley
        ) / sigma_nm
        same_response = (
            gap_s < learned_response_span
            and valley_depth_sigma < valley_threshold_sigma
        )
        if same_response:
            if prominences_nm[current] > prominences_nm[previous]:
                kept[-1] = current
        else:
            kept.append(current)
    return np.asarray(kept, dtype=int)


def analyse_polarity(
    signal_nm: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    sign: int,
    background_mask: np.ndarray,
    activity_mask: np.ndarray,
    background_location_nm: float,
    background_sigma_nm: float,
    settings: AdaptiveSettings,
) -> PolarityResult:
    oriented = sign * (signal_nm - background_location_nm)
    local_indices, _ = find_peaks(oriented)
    warnings: list[str] = []
    if not len(local_indices):
        return PolarityResult(
            sign=sign,
            local_peak_count=0,
            background_peak_count=0,
            height_threshold_nm=float("nan"),
            prominence_threshold_nm=float("nan"),
            height_threshold_sigma=float("nan"),
            prominence_threshold_sigma=float("nan"),
            valley_threshold_sigma=float("nan"),
            typical_width_s=float("nan"),
            width_log_sigma=float("nan"),
            evidence_threshold=float("nan"),
            evidence_log_gap=float("nan"),
            preliminary_indices=np.array([], dtype=int),
            accepted_indices=np.array([], dtype=int),
            accepted_width_s=np.array([], dtype=float),
            accepted_prominence_nm=np.array([], dtype=float),
            accepted_height_sigma=np.array([], dtype=float),
            quality_score=0.0,
            warnings=["no local maxima"],
        )

    prominences = peak_prominences(oriented, local_indices)[0]
    width_samples = peak_widths(oriented, local_indices, rel_height=0.5)[0]
    width_s = width_samples / fs_hz
    height = oriented[local_indices]

    background_selector = background_mask[local_indices]
    background_indices = local_indices[background_selector]
    background_heights = height[background_selector]
    background_prominences = prominences[background_selector]
    if len(background_indices) < settings.minimum_background_peaks:
        warnings.append(
            f"only {len(background_indices)} quiet-background maxima; thresholds may be unstable"
        )
        order = np.argsort(height)
        fallback_count = min(len(order), max(settings.minimum_background_peaks, len(order) // 2))
        selected = order[:fallback_count]
        background_indices = local_indices[selected]
        background_heights = height[selected]
        background_prominences = prominences[selected]

    height_threshold = maximum_background_threshold(background_heights)
    prominence_threshold = maximum_background_threshold(background_prominences)
    preliminary_mask = (
        (height > height_threshold)
        & (prominences > prominence_threshold)
        & activity_mask[local_indices]
    )
    preliminary_positions = np.flatnonzero(preliminary_mask)
    preliminary_indices = local_indices[preliminary_positions]

    evidence_keep, evidence_threshold, evidence_log_gap = separated_evidence_cluster(
        height[preliminary_positions] / max(height_threshold, np.finfo(float).eps),
        prominences[preliminary_positions] / max(prominence_threshold, np.finfo(float).eps),
    )
    evidence_positions = preliminary_positions[evidence_keep]

    if len(evidence_positions):
        preliminary_widths = width_s[evidence_positions]
        typical_width_s, width_log_sigma = robust_log_width_model(preliminary_widths)
    else:
        preliminary_widths = np.array([], dtype=float)
        typical_width_s, width_log_sigma = float("nan"), float("nan")

    width_keep = np.ones(len(evidence_positions), dtype=bool)
    if (
        len(evidence_positions) >= settings.minimum_width_model_peaks
        and np.isfinite(width_log_sigma)
        and width_log_sigma > np.finfo(float).eps
    ):
        width_z = np.abs(np.log(preliminary_widths / typical_width_s)) / width_log_sigma
        width_keep = width_z <= float(1.0 / np.sqrt(settings.width_outlier_probability))

        per_tail_probability = settings.width_outlier_probability / (
            2.0 * len(evidence_positions)
        )
        joint_width_limit = float(norm.isf(per_tail_probability))
        second_evidence_keep, _, second_evidence_gap = separated_evidence_cluster(
            height[evidence_positions] / max(height_threshold, np.finfo(float).eps),
            prominences[evidence_positions] / max(prominence_threshold, np.finfo(float).eps),
        )
        weak_group = ~second_evidence_keep
        if (
            np.any(weak_group)
            and second_evidence_gap > evidence_log_gap
            and np.all(width_z[weak_group] > joint_width_limit)
        ):
            width_keep &= second_evidence_keep

    positions = evidence_positions[width_keep]
    candidate_indices = local_indices[positions]
    candidate_widths = width_s[positions]
    candidate_prominences = prominences[positions]

    valley_depths = background_valley_depths(
        oriented, background_indices, background_mask, background_sigma_nm
    )
    if len(valley_depths):
        valley_threshold_sigma = maximum_background_threshold(valley_depths)
    else:
        valley_threshold_sigma = prominence_threshold / background_sigma_nm
        warnings.append("no quiet valley distribution; prominence threshold used for separation")

    keep_positions = merge_ringing_candidates(
        oriented,
        candidate_indices,
        candidate_widths,
        candidate_prominences,
        time_s,
        background_sigma_nm,
        valley_threshold_sigma,
    )
    accepted_indices = candidate_indices[keep_positions]
    accepted_widths = candidate_widths[keep_positions]
    accepted_prominences = candidate_prominences[keep_positions]
    accepted_height_sigma = (
        oriented[accepted_indices] / background_sigma_nm
        if len(accepted_indices)
        else np.array([], dtype=float)
    )

    count_before_broad_filter = len(accepted_indices)
    if count_before_broad_filter:
        broad_width_keep = separated_broad_width_cluster(
            accepted_widths,
            oriented[accepted_indices] / max(height_threshold, np.finfo(float).eps),
            accepted_prominences / max(prominence_threshold, np.finfo(float).eps),
        )
        broad_rejected = int(np.sum(~broad_width_keep))
        if broad_rejected:
            accepted_indices = accepted_indices[broad_width_keep]
            accepted_widths = accepted_widths[broad_width_keep]
            accepted_prominences = accepted_prominences[broad_width_keep]
            accepted_height_sigma = accepted_height_sigma[broad_width_keep]
            warnings.append(
                f"removed {broad_rejected} broad weak response"
                f"{'s' if broad_rejected != 1 else ''}"
            )

    if len(accepted_indices):
        evidence = np.median(
            np.minimum(
                oriented[accepted_indices] / max(height_threshold, np.finfo(float).eps),
                accepted_prominences / max(prominence_threshold, np.finfo(float).eps),
            )
        )
        preliminary_width_consistency = (
            float(np.mean(width_keep)) if len(width_keep) else 0.0
        )
        broad_width_consistency = len(accepted_indices) / count_before_broad_filter
        width_consistency = preliminary_width_consistency * broad_width_consistency
        separation_strength = float(1.0 - np.exp(-max(0.0, evidence_log_gap)))
        quality_score = float(
            np.clip(
                (1.0 - np.exp(-max(0.0, evidence - 1.0)))
                * width_consistency
                * separation_strength,
                0.0,
                1.0,
            )
        )
    else:
        quality_score = 0.0

    return PolarityResult(
        sign=sign,
        local_peak_count=len(local_indices),
        background_peak_count=len(background_indices),
        height_threshold_nm=float(height_threshold),
        prominence_threshold_nm=float(prominence_threshold),
        height_threshold_sigma=float(height_threshold / background_sigma_nm),
        prominence_threshold_sigma=float(prominence_threshold / background_sigma_nm),
        valley_threshold_sigma=float(valley_threshold_sigma),
        typical_width_s=float(typical_width_s),
        width_log_sigma=float(width_log_sigma),
        evidence_threshold=float(evidence_threshold),
        evidence_log_gap=float(evidence_log_gap),
        preliminary_indices=preliminary_indices.astype(int),
        accepted_indices=accepted_indices.astype(int),
        accepted_width_s=accepted_widths.astype(float),
        accepted_prominence_nm=accepted_prominences.astype(float),
        accepted_height_sigma=accepted_height_sigma.astype(float),
        quality_score=quality_score,
        warnings=warnings,
    )


def polarity_impulse_sharpness(result: PolarityResult) -> float:
    """Score the direct axle impulse without favouring rebound multiplicity."""
    if not len(result.accepted_indices):
        return 0.0
    widths = np.maximum(result.accepted_width_s, np.finfo(float).eps)
    sharpness = np.sort(result.accepted_prominence_nm / widths)
    sharper_half = sharpness[len(sharpness) // 2 :]
    return float(np.median(sharper_half))


def detect_adaptive_peaks(
    signal_nm: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    polarity: str = "auto",
    settings: AdaptiveSettings | None = None,
) -> AdaptiveDetection:
    """Detect peaks using parameters learned from this sensor recording."""
    settings = settings or AdaptiveSettings()
    if polarity not in {"auto", "positive", "negative"}:
        raise ValueError("polarity must be auto, positive, or negative")

    initial_location, _ = robust_location_scale(signal_nm)

    # -------------------------------------------------------------------------
    # ABSOLUTE NOISE GATE / INPUT VALIDATION
    # Tren yoksa doğrudan 0 aks döndür, istatistiksel hesaplamalara girme.
    # -------------------------------------------------------------------------
    max_excursion_nm = float(np.max(np.abs(signal_nm - initial_location)))
    if max_excursion_nm < settings.absolute_gate_nm:
        empty_polarity = PolarityResult(
            sign=1,
            local_peak_count=0,
            background_peak_count=0,
            height_threshold_nm=settings.absolute_gate_nm,
            prominence_threshold_nm=settings.absolute_gate_nm,
            height_threshold_sigma=float("nan"),
            prominence_threshold_sigma=float("nan"),
            valley_threshold_sigma=float("nan"),
            typical_width_s=float("nan"),
            width_log_sigma=float("nan"),
            evidence_threshold=float("nan"),
            evidence_log_gap=float("nan"),
            preliminary_indices=np.array([], dtype=int),
            accepted_indices=np.array([], dtype=int),
            accepted_width_s=np.array([], dtype=float),
            accepted_prominence_nm=np.array([], dtype=float),
            accepted_height_sigma=np.array([], dtype=float),
            quality_score=0.0,
            warnings=["Signal maximum excursion below absolute gate (empty record)"],
        )
        empty_neg = PolarityResult(
            sign=-1,
            local_peak_count=0,
            background_peak_count=0,
            height_threshold_nm=settings.absolute_gate_nm,
            prominence_threshold_nm=settings.absolute_gate_nm,
            height_threshold_sigma=float("nan"),
            prominence_threshold_sigma=float("nan"),
            valley_threshold_sigma=float("nan"),
            typical_width_s=float("nan"),
            width_log_sigma=float("nan"),
            evidence_threshold=float("nan"),
            evidence_log_gap=float("nan"),
            preliminary_indices=np.array([], dtype=int),
            accepted_indices=np.array([], dtype=int),
            accepted_width_s=np.array([], dtype=float),
            accepted_prominence_nm=np.array([], dtype=float),
            accepted_height_sigma=np.array([], dtype=float),
            quality_score=0.0,
            warnings=["Signal maximum excursion below absolute gate (empty record)"],
        )
        return AdaptiveDetection(
            indices=np.array([], dtype=int),
            signs=np.array([], dtype=int),
            selected_polarity=polarity if polarity != "auto" else "positive",
            polarity_confidence=0.0,
            background_location_nm=float(initial_location),
            background_sigma_nm=float("nan"),
            correlation_time_s=0.0,
            quiet_fraction_observed=1.0,
            quality_score=0.0,
            quality_label="no statistically detectable event",
            positive=empty_polarity,
            negative=empty_neg,
        )
    # -------------------------------------------------------------------------

    correlation_samples = estimate_correlation_samples(signal_nm, fs_hz)
    background_mask = quiet_background_mask(
        signal_nm - initial_location, correlation_samples, settings
    )
    activity_mask = active_event_mask(
        signal_nm - initial_location, correlation_samples, background_mask, settings
    )
    background_location, background_sigma = robust_location_scale(signal_nm[background_mask])

    positive = analyse_polarity(
        signal_nm, time_s, fs_hz, 1, background_mask, activity_mask,
        background_location, background_sigma, settings,
    )
    negative = analyse_polarity(
        signal_nm, time_s, fs_hz, -1, background_mask, activity_mask,
        background_location, background_sigma, settings,
    )
    active_centered = signal_nm[activity_mask] - background_location
    if not len(active_centered):
        active_centered = signal_nm - background_location

    tail_count = max(1, int(np.ceil(np.sqrt(len(active_centered)))))
    ordered = np.sort(active_centered)
    evidence_positive = float(np.mean(ordered[-tail_count:]))
    evidence_negative = float(np.mean(-ordered[:tail_count]))

    impulse_positive = polarity_impulse_sharpness(positive)
    impulse_negative = polarity_impulse_sharpness(negative)

    if polarity == "positive":
        selected = positive
    elif polarity == "negative":
        selected = negative
    else:
        if np.isclose(impulse_positive, impulse_negative):
            selected = positive if evidence_positive >= evidence_negative else negative
        else:
            selected = positive if impulse_positive > impulse_negative else negative
    denominator = abs(impulse_positive) + abs(impulse_negative)
    polarity_confidence = (
        abs(impulse_positive - impulse_negative) / denominator if denominator > 0 else 0.0
    )

    count = len(selected.accepted_indices)
    if count == 0:
        quality_label = "no statistically detectable event"
    elif selected.quality_score >= 0.65 and polarity_confidence >= 0.15:
        quality_label = "strong"
    elif selected.quality_score >= 0.30:
        quality_label = "usable with review"
    else:
        quality_label = "uncertain"

    return AdaptiveDetection(
        indices=selected.accepted_indices,
        signs=np.full(count, selected.sign, dtype=int),
        selected_polarity="positive" if selected.sign > 0 else "negative",
        polarity_confidence=float(polarity_confidence),
        background_location_nm=float(background_location),
        background_sigma_nm=float(background_sigma),
        correlation_time_s=float(correlation_samples / fs_hz),
        quiet_fraction_observed=float(np.mean(background_mask)),
        quality_score=float(selected.quality_score),
        quality_label=quality_label,
        positive=positive,
        negative=negative,
    )


def peak_table(
    time_s: np.ndarray,
    raw_nm: np.ndarray,
    processed_nm: np.ndarray,
    detection: AdaptiveDetection,
) -> pd.DataFrame:
    selected = detection.positive if detection.selected_polarity == "positive" else detection.negative
    return pd.DataFrame(
        {
            "peak_number": np.arange(1, len(detection.indices) + 1),
            "time_s": time_s[detection.indices],
            "elapsed_s": time_s[detection.indices] - time_s[0],
            "raw_wavelength_nm": raw_nm[detection.indices],
            "processed_amplitude_nm": processed_nm[detection.indices],
            "polarity": detection.selected_polarity,
            "height_sigma": selected.accepted_height_sigma,
            "prominence_nm": selected.accepted_prominence_nm,
            "width_s": selected.accepted_width_s,
        }
    )


def diagnostic_dict(detection: AdaptiveDetection) -> dict:
    def compact(result: PolarityResult) -> dict:
        data = asdict(result)
        for key in (
            "preliminary_indices", "accepted_indices", "accepted_width_s",
            "accepted_prominence_nm", "accepted_height_sigma",
        ):
            data[key] = np.asarray(data[key]).tolist()
        return data

    return {
        "selected_polarity": detection.selected_polarity,
        "polarity_confidence": detection.polarity_confidence,
        "background_location_nm": detection.background_location_nm,
        "background_sigma_nm": detection.background_sigma_nm,
        "correlation_time_s": detection.correlation_time_s,
        "quiet_fraction_observed": detection.quiet_fraction_observed,
        "quality_score": detection.quality_score,
        "quality_label": detection.quality_label,
        "positive": compact(detection.positive),
        "negative": compact(detection.negative),
    }


def plot_detection(
    input_path: Path,
    stored_sensor: int,
    time_s: np.ndarray,
    raw_nm: np.ndarray,
    processed_nm: np.ndarray,
    detection: AdaptiveDetection,
    highpass_hz: float,
    lowpass_hz: float,
    output_path: Path,
) -> None:
    physical_sensor = 13 - stored_sensor
    elapsed = time_s - time_s[0]
    table = peak_table(time_s, raw_nm, processed_nm, detection)
    selected = detection.positive if detection.selected_polarity == "positive" else detection.negative
    sign = 1 if detection.selected_polarity == "positive" else -1
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    axes[0].plot(elapsed, raw_nm, linewidth=0.75, color="tab:blue")
    axes[0].set_title(
        f"Physical FBG {physical_sensor} (stored column {stored_sensor}): {input_path.name}"
    )
    axes[0].set_ylabel("Wavelength [nm]")
    axes[0].grid(alpha=0.25)

    axes[1].plot(elapsed, processed_nm, linewidth=0.75, color="black")
    height_line = detection.background_location_nm + sign * selected.height_threshold_nm
    axes[1].axhline(
        height_line, color="tab:orange", linestyle="--", linewidth=1,
        label=(
            f"learned height = {selected.height_threshold_sigma:.2f} sigma "
            f"({selected.height_threshold_nm:.6f} nm)"
        ),
    )
    if not table.empty:
        axes[1].scatter(
            table["elapsed_s"], table["processed_amplitude_nm"],
            color="tab:red", s=24, zorder=3, label="accepted axle candidates",
        )
        for row in table.itertuples(index=False):
            axes[1].annotate(
                str(row.peak_number),
                (row.elapsed_s, row.processed_amplitude_nm),
                xytext=(0, 6 if sign > 0 else -12), textcoords="offset points",
                ha="center", fontsize=6.5, color="tab:red",
            )
        span = max(1.0, float(table.elapsed_s.max() - table.elapsed_s.min()))
        pad = max(0.5, 0.08 * span)
        axes[1].set_xlim(max(elapsed[0], table.elapsed_s.min() - pad), min(elapsed[-1], table.elapsed_s.max() + pad))
    axes[1].set_title(
        f"Self-calibrating detector — {len(table)} peaks — {detection.selected_polarity} — "
        f"quality: {detection.quality_label}"
    )
    axes[1].set_xlabel("Elapsed time [s]")
    axes[1].set_ylabel("Filtered response [nm]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")
    axes[1].text(
        0.01,
        0.98,
        (
            f"Noise sigma = {detection.background_sigma_nm:.6f} nm\n"
            f"Filter = {highpass_hz:.3g}–{lowpass_hz:.3g} Hz\n"
            f"Height = {selected.height_threshold_sigma:.2f} sigma\n"
            f"Prominence = {selected.prominence_threshold_sigma:.2f} sigma"
        ),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_sensor_list(value: str, physical: bool) -> list[int]:
    if value.strip().lower() == "all":
        requested = list(range(1, 13))
    else:
        requested = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not requested or any(sensor < 1 or sensor > 12 for sensor in requested):
        raise argparse.ArgumentTypeError("sensors must be a comma list from 1 to 12, or 'all'")
    return [PHYSICAL_TO_STORED[sensor] for sensor in requested] if physical else requested


def iter_files(path: Path, pattern: str) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob(pattern))
    else:
        raise FileNotFoundError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-calibrating FBG axle detector for one file or a directory."
    )
    parser.add_argument("input", type=Path, help="FBG .txt file or directory")
    parser.add_argument("--pattern", default="20*.txt")
    parser.add_argument("--sensors", default="12", help="Stored columns, comma list or 'all'")
    parser.add_argument(
        "--physical-sensors", action="store_true",
        help="Interpret --sensors as physical FBG numbers using the reversed mapping",
    )
    parser.add_argument("--polarity", choices=("auto", "positive", "negative"), default="auto")
    parser.add_argument("--highpass", type=float, default=0.5, help="Fixed high-pass cutoff in Hz")
    parser.add_argument("--lowpass", type=float, default=40.0, help="Fixed low-pass cutoff in Hz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("self_calibrating_results"))
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    sensors = parse_sensor_list(args.sensors, args.physical_sensors)
    files = list(iter_files(input_path, args.pattern))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} in {input_path}")

    settings = AdaptiveSettings()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for file_number, file_path in enumerate(files, start=1):
        for stored_sensor in sensors:
            try:
                physical_sensor = 13 - stored_sensor
                time_s, raw_nm = load_fbg_file(file_path, stored_sensor)
                fs_hz = sampling_rate(time_s)
                processed_nm = butterworth_process(
                    raw_nm, fs_hz, args.highpass, args.lowpass
                )
                detection = detect_adaptive_peaks(
                    processed_nm, time_s, fs_hz, args.polarity, settings
                )
                table = peak_table(time_s, raw_nm, processed_nm, detection)
                base = f"{file_path.stem}_fbg{physical_sensor}_adaptive"
                table.to_csv(output_dir / f"{base}_peaks.csv", index=False, float_format="%.8f")
                diagnostics = diagnostic_dict(detection)
                diagnostics["physical_sensor"] = physical_sensor
                diagnostics["stored_column"] = stored_sensor
                diagnostics["filter"] = {
                    "mode": "fixed",
                    "highpass_hz": args.highpass,
                    "lowpass_hz": args.lowpass,
                }
                (output_dir / f"{base}_diagnostics.json").write_text(
                    json.dumps(diagnostics, indent=2, allow_nan=True),
                    encoding="utf-8",
                )
                if args.plot:
                    plot_detection(
                        file_path, stored_sensor, time_s, raw_nm, processed_nm,
                        detection, args.highpass, args.lowpass, output_dir / f"{base}.png",
                    )
                selected = detection.positive if detection.selected_polarity == "positive" else detection.negative
                summary_rows.append(
                    {
                        "file": file_path.name,
                        "stored_sensor": stored_sensor,
                        "physical_sensor": physical_sensor,
                        "peak_count": len(table),
                        "selected_polarity": detection.selected_polarity,
                        "polarity_confidence": detection.polarity_confidence,
                        "quality_score": detection.quality_score,
                        "quality_label": detection.quality_label,
                        "background_sigma_nm": detection.background_sigma_nm,
                        "height_threshold_nm": selected.height_threshold_nm,
                        "prominence_threshold_nm": selected.prominence_threshold_nm,
                        "typical_width_s": selected.typical_width_s,
                        "valley_threshold_sigma": selected.valley_threshold_sigma,
                        "correlation_time_s": detection.correlation_time_s,
                        "sampling_rate_hz": fs_hz,
                        "filter_mode": "fixed",
                        "highpass_hz": args.highpass,
                        "lowpass_hz": args.lowpass,
                        "error": "",
                    }
                )
                print(
                    f"[{file_number}/{len(files)}] {file_path.name} FBG {physical_sensor} "
                    f"(stored column {stored_sensor}): "
                    f"{len(table)} peaks, {detection.selected_polarity}, {detection.quality_label}; "
                    f"band {args.highpass:.3g}-{args.lowpass:.3g} Hz"
                )
            except Exception as exc:
                physical_sensor = 13 - stored_sensor
                summary_rows.append(
                    {
                        "file": file_path.name,
                        "stored_sensor": stored_sensor,
                        "physical_sensor": physical_sensor,
                        "peak_count": np.nan,
                        "error": str(exc),
                    }
                )
                print(
                    f"[{file_number}/{len(files)}] {file_path.name} FBG {physical_sensor} "
                    f"(stored column {stored_sensor}): ERROR - {exc}"
                )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "self_calibrating_axle_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

"""Final strict axle detector: all counting calculations in one module.

Start with detect_axles(). Inputs are a selected physical FBG's raw wavelength
signal in nm and timestamps in seconds. Filtering, the 20 pm gate, calibration,
polarity, and peak acceptance are implemented below without UI or file access.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, detrend, find_peaks, peak_prominences, peak_widths, sosfiltfilt
from scipy.stats import norm

EXPECTED_SAMPLING_RATE_HZ = 500.0
FIXED_HIGHPASS_HZ = 0.5
FIXED_LOWPASS_HZ = 40.0
POLARITY_BY_PHYSICAL_FBG = {
    physical_fbg: ("negative" if physical_fbg == 9 else "positive")
    for physical_fbg in range(1, 13)
}

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


# Main pipeline: read this function first.
def detect_axles(
    raw_wavelength_nm: np.ndarray,
    time_s: np.ndarray,
    physical_fbg: int,
    *,
    stored_column: int | None = None,
    stored_order: str | None = None,
    settings: StrictDetectorSettings | None = None,
) -> StrictAxleDetection:
    """Filter and count one physical FBG without changing the working detector."""
    # 1. Validate the selected physical sensor and input arrays.
    settings = settings or StrictDetectorSettings()
    if physical_fbg not in POLARITY_BY_PHYSICAL_FBG:
        raise ValueError("physical_fbg must be between 1 and 12")

    raw = np.asarray(raw_wavelength_nm, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if raw.ndim != 1 or time_s.ndim != 1 or len(raw) != len(time_s) or len(raw) < 64:
        raise ValueError("Signal and time must be equal-length one-dimensional arrays")
    if np.any(~np.isfinite(raw)):
        raise ValueError("Signal contains non-finite values")

    # 2. Detrend and apply the fixed Butterworth filter at 500 Hz.
    validate_fixed_sampling_rate(time_s, settings)
    filtered = butterworth_process(
        raw,
        settings.sampling_rate_hz,
        settings.highpass_hz,
        settings.lowpass_hz,
    )
    # 3. Measure the excursion from the recording median for the 20 pm gate.
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

    # Below the physical event floor, return an empty count and diagnostics.
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

    # 4. Define the active interval and learn the quiet background.
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
    # 5. Apply the core peak rules, then preserve physically strong candidates.
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

    # 6. Select polarity; FBG 9 evaluates both directions.
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

    # 7. Return accepted axle positions and the exact signals used for plotting.
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


# Filtering and event boundaries.
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


# Quiet background and learned thresholds.
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


def maximum_background_threshold(background_values: np.ndarray) -> float:
    """Use the largest finite background measurement as the threshold."""
    values = np.asarray(background_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No background peak features are available")
    return float(np.max(values))


# Axle candidates, evidence, and response separation.
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


def polarity_impulse_sharpness(result: PolarityResult) -> float:
    """Score the direct axle impulse without favouring rebound multiplicity."""
    if not len(result.accepted_indices):
        return 0.0
    widths = np.maximum(result.accepted_width_s, np.finfo(float).eps)
    sharpness = np.sort(result.accepted_prominence_nm / widths)
    sharper_half = sharpness[len(sharpness) // 2 :]
    return float(np.median(sharper_half))


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


def _quality_label(result: PolarityResult) -> str:
    if not len(result.accepted_indices):
        return "no statistically detectable event"
    if result.quality_score >= 0.65:
        return "strong"
    if result.quality_score >= 0.30:
        return "usable with review"
    return "uncertain"

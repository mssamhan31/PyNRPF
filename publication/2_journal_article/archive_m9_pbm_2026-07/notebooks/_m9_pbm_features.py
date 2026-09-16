"""Physical candidate-window features and correction metrics for m9_pbm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "F1_bridge_improvement",
    "F2_roughness_improvement",
    "F3_slope_continuity_improvement",
    "F4_duration_plausibility",
    "F5_n_height_ratio",
    "F6_solar_strength_ratio",
    "F7_solar_peak_alignment",
    "F8_substation_centered_core_score",
    "F9_substation_rank_core_score",
]
COMPACT_FEATURE_COLUMNS = [
    "F1_bridge_improvement",
    "F3_slope_continuity_improvement",
    "F4_duration_plausibility",
]


@dataclass(frozen=True)
class CandidateSpec:
    """Label-free candidate geometry and feature constants."""

    slots_per_day: int = 96
    slot_minutes: int = 15
    scan_start_slot: int = 24
    scan_end_slot: int = 72
    min_duration_slots: int = 2
    max_duration_slots: int = 32
    shoulder_slots: int = 3
    anchor_offset_slots: int = 1
    solar_peak_radius_slots: int = 14
    max_internal_gap_slots: int = 4
    duration_saturation_hours: float = 1.5
    robust_bound_scale: float = 3.0
    epsilon: float = 1e-9

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CandidateSpec":
        windows = config["m9_pbm"]["candidate_windows"]
        features = config["m9_pbm"]["features"]
        return cls(
            **{key: windows[key] for key in cls.__dataclass_fields__ if key in windows},
            duration_saturation_hours=features["duration_saturation_hours"],
            robust_bound_scale=features["robust_bound_scale"],
            epsilon=features["epsilon"],
        )


def prepare_day_values(
    values: np.ndarray,
    *,
    max_internal_gap_slots: int,
    nonnegative: bool = False,
) -> tuple[np.ndarray, int]:
    """Interpolate short internal gaps, then replace unresolved values with zero."""

    series = pd.Series(np.asarray(values, dtype=float))
    filled = series.interpolate(
        method="linear",
        limit=max_internal_gap_slots,
        limit_area="inside",
    )
    unresolved = int(filled.isna().sum())
    result = filled.fillna(0.0).to_numpy(dtype=float)
    if nonnegative:
        result = np.maximum(result, 0.0)
    return result, unresolved


def reconstruct_demand(
    net_load: np.ndarray,
    solar: np.ndarray,
    left_slot: int,
    right_slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return no-correction and candidate-corrected underlying demand curves."""

    y = np.asarray(net_load, dtype=float)
    s = np.asarray(solar, dtype=float)
    if y.shape != s.shape:
        raise ValueError("net_load and solar must have the same shape.")
    if not 0 <= left_slot <= right_slot < len(y):
        raise ValueError("Candidate bounds fall outside the day array.")
    uncorrected = s + y
    corrected = uncorrected.copy()
    corrected[left_slot : right_slot + 1] = (
        s[left_slot : right_slot + 1] - y[left_slot : right_slot + 1]
    )
    return uncorrected, corrected


def candidate_windows(solar: np.ndarray, spec: CandidateSpec) -> pd.DataFrame:
    """Generate all valid inclusive candidate windows in deterministic order."""

    solar_values = np.asarray(solar, dtype=float)
    if len(solar_values) != spec.slots_per_day:
        raise ValueError(f"Expected {spec.slots_per_day} solar slots, got {len(solar_values)}.")
    daytime = solar_values[spec.scan_start_slot : spec.scan_end_slot + 1]
    solar_peak_slot = int(np.argmax(daytime)) + spec.scan_start_slot
    rows: list[dict[str, int | float]] = []
    candidate_id = 0
    for left in range(spec.scan_start_slot, spec.scan_end_slot):
        minimum_right = left + spec.min_duration_slots - 1
        maximum_right = min(
            spec.scan_end_slot,
            left + spec.max_duration_slots - 1,
        )
        for right in range(minimum_right, maximum_right + 1):
            midpoint = (left + right) / 2.0
            if abs(midpoint - solar_peak_slot) > spec.solar_peak_radius_slots:
                continue
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "left_slot": left,
                    "right_slot": right,
                    "duration_slots": right - left + 1,
                    "duration_hours": (right - left + 1) * spec.slot_minutes / 60,
                    "solar_peak_slot": solar_peak_slot,
                }
            )
            candidate_id += 1
    return pd.DataFrame(rows)


def bridge_line(
    values: np.ndarray,
    left_slot: int,
    right_slot: int,
    spec: CandidateSpec,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the linear bridge inside W and its two outside anchor slots."""

    series = np.asarray(values, dtype=float)
    left_anchor = max(0, left_slot - spec.anchor_offset_slots)
    right_anchor = min(len(series) - 1, right_slot + spec.anchor_offset_slots)
    window_slots = np.arange(left_slot, right_slot + 1)
    if right_anchor == left_anchor:
        line = np.full(len(window_slots), series[left_anchor], dtype=float)
    else:
        line = np.interp(
            window_slots,
            [left_anchor, right_anchor],
            [series[left_anchor], series[right_anchor]],
        )
    return line, (left_anchor, right_anchor)


def bridge_error(
    values: np.ndarray,
    anchor_values: np.ndarray,
    left_slot: int,
    right_slot: int,
    spec: CandidateSpec,
) -> float:
    """Median absolute deviation from the line joining the outside anchors."""

    line, _ = bridge_line(anchor_values, left_slot, right_slot, spec)
    window = np.asarray(values, dtype=float)[left_slot : right_slot + 1]
    return float(np.median(np.abs(window - line)))


def total_variation(
    values: np.ndarray,
    left_slot: int,
    right_slot: int,
    shoulder_slots: int,
) -> float:
    """Total variation over W plus symmetric local shoulders."""

    series = np.asarray(values, dtype=float)
    start = max(0, left_slot - shoulder_slots)
    end = min(len(series) - 1, right_slot + shoulder_slots)
    return float(np.abs(np.diff(series[start : end + 1])).sum())


def _median_slope(values: np.ndarray, start: int, end: int) -> float:
    series = np.asarray(values, dtype=float)
    start = max(start, 0)
    end = min(end, len(series) - 1)
    if end <= start:
        return 0.0
    return float(np.median(np.diff(series[start : end + 1])))


def slope_jump(
    values: np.ndarray,
    left_slot: int,
    right_slot: int,
    shoulder_slots: int,
) -> float:
    """Sum robust inside/outside slope mismatches at both boundaries."""

    outside_left = _median_slope(values, left_slot - shoulder_slots, left_slot)
    inside_left = _median_slope(values, left_slot, min(right_slot, left_slot + shoulder_slots))
    inside_right = _median_slope(values, max(left_slot, right_slot - shoulder_slots), right_slot)
    outside_right = _median_slope(values, right_slot, right_slot + shoulder_slots)
    return abs(outside_left - inside_left) + abs(inside_right - outside_right)


def _normalised_improvement(before: float, after: float, epsilon: float) -> float:
    return float((before - after) / (before + after + epsilon))


def _row_range_median(
    differences: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Median row value over half-open, candidate-specific difference ranges."""

    lengths = ends - starts
    width = int(lengths.max())
    offsets = np.arange(width)[None, :]
    indices = np.clip(starts[:, None] + offsets, 0, differences.shape[1] - 1)
    gathered = np.take_along_axis(differences, indices, axis=1)
    return np.nanmedian(np.where(offsets < lengths[:, None], gathered, np.nan), axis=1)


def compute_candidate_features(
    net_load: np.ndarray,
    solar: np.ndarray,
    *,
    substation_solar_scale: float,
    spec: CandidateSpec,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compute F1-F7 for every valid candidate on one substation-day."""

    y, unresolved_net = prepare_day_values(
        net_load,
        max_internal_gap_slots=spec.max_internal_gap_slots,
    )
    s, unresolved_solar = prepare_day_values(
        solar,
        max_internal_gap_slots=spec.max_internal_gap_slots,
        nonnegative=True,
    )
    candidates = candidate_windows(s, spec)
    uncorrected = s + y
    corrected_inside = s - y
    daytime = slice(spec.scan_start_slot, spec.scan_end_slot + 1)
    day_scale = max(
        float(np.percentile(np.abs(y[daytime]), 95)),
        float(np.percentile(s[daytime], 95)),
        spec.epsilon,
    )
    left = candidates["left_slot"].to_numpy(dtype=int)
    right = candidates["right_slot"].to_numpy(dtype=int)
    slots = np.arange(spec.slots_per_day)[None, :]
    inside = (slots >= left[:, None]) & (slots <= right[:, None])
    corrected = np.where(inside, corrected_inside[None, :], uncorrected[None, :])

    # F1: compare both reconstructions with the same line through outside anchors.
    left_anchor = np.maximum(0, left - spec.anchor_offset_slots)
    right_anchor = np.minimum(spec.slots_per_day - 1, right + spec.anchor_offset_slots)
    anchor_span = np.maximum(right_anchor - left_anchor, 1)
    bridge_slope = (
        uncorrected[right_anchor] - uncorrected[left_anchor]
    ) / anchor_span
    window_offsets = np.arange(spec.max_duration_slots)[None, :]
    window_lengths = right - left + 1
    window_valid = window_offsets < window_lengths[:, None]
    window_indices = np.clip(
        left[:, None] + window_offsets, 0, spec.slots_per_day - 1
    )
    bridge = uncorrected[left_anchor, None] + bridge_slope[:, None] * (
        window_indices - left_anchor[:, None]
    )
    bridge_before = np.nanmedian(
        np.where(
            window_valid,
            np.abs(uncorrected[window_indices] - bridge),
            np.nan,
        ),
        axis=1,
    )
    bridge_after = np.nanmedian(
        np.where(
            window_valid,
            np.abs(corrected_inside[window_indices] - bridge),
            np.nan,
        ),
        axis=1,
    )
    f1 = (bridge_before - bridge_after) / (
        bridge_before + bridge_after + spec.epsilon
    )

    # F2: total variation over each window plus configured shoulders.
    diff_slots = np.arange(spec.slots_per_day - 1)[None, :]
    context_start = np.maximum(0, left - spec.shoulder_slots)
    context_end = np.minimum(spec.slots_per_day - 1, right + spec.shoulder_slots)
    context = (diff_slots >= context_start[:, None]) & (
        diff_slots < context_end[:, None]
    )
    uncorrected_differences = np.abs(np.diff(uncorrected))[None, :]
    corrected_differences = np.abs(np.diff(corrected, axis=1))
    roughness_before = np.where(context, uncorrected_differences, 0.0).sum(axis=1)
    roughness_after = np.where(context, corrected_differences, 0.0).sum(axis=1)
    f2 = (roughness_before - roughness_after) / (
        roughness_before + roughness_after + spec.epsilon
    )

    # F3: robust outside/inside slope mismatch at each boundary.
    uncorrected_slopes = np.broadcast_to(
        np.diff(uncorrected)[None, :], corrected_differences.shape
    )
    corrected_slopes = np.diff(corrected, axis=1)

    def boundary_jump(differences: np.ndarray) -> np.ndarray:
        outside_left = _row_range_median(
            differences,
            np.maximum(0, left - spec.shoulder_slots),
            left,
        )
        inside_left = _row_range_median(
            differences,
            left,
            np.minimum(right, left + spec.shoulder_slots),
        )
        inside_right = _row_range_median(
            differences,
            np.maximum(left, right - spec.shoulder_slots),
            right,
        )
        outside_right = _row_range_median(
            differences,
            right,
            np.minimum(spec.slots_per_day - 1, right + spec.shoulder_slots),
        )
        return np.abs(outside_left - inside_left) + np.abs(inside_right - outside_right)

    slope_before = boundary_jump(uncorrected_slopes)
    slope_after = boundary_jump(corrected_slopes)
    f3 = (slope_before - slope_after) / (slope_before + slope_after + spec.epsilon)

    # F4-F7: duration, N-height, solar strength, and solar-peak alignment.
    duration_score = np.clip(
        candidates["duration_hours"].to_numpy(dtype=float)
        / spec.duration_saturation_hours,
        0.0,
        1.0,
    )
    net_peak = np.where(inside, y[None, :], -np.inf).max(axis=1)
    net_edge = np.maximum(y[left], y[right])
    n_height = np.clip((net_peak - net_edge) / day_scale, 0.0, 1.0)
    window_solar = np.where(window_valid, s[window_indices], np.nan)
    solar_p95 = np.nanpercentile(window_solar, 95, axis=1)
    solar_strength = np.clip(
        solar_p95 / max(substation_solar_scale, spec.epsilon), 0.0, 1.0
    )
    midpoint = (left + right) / 2
    peak_alignment = np.clip(
        1
        - np.abs(midpoint - candidates["solar_peak_slot"].to_numpy(dtype=float))
        / spec.solar_peak_radius_slots,
        0.0,
        1.0,
    )

    result = candidates.copy()
    result["F1_bridge_improvement"] = f1
    result["F2_roughness_improvement"] = f2
    result["F3_slope_continuity_improvement"] = f3
    result["F4_duration_plausibility"] = duration_score
    result["F5_n_height_ratio"] = n_height
    result["F6_solar_strength_ratio"] = solar_strength
    result["F7_solar_peak_alignment"] = peak_alignment
    result["core_score"] = f1 + f2 + f3
    feature_values = result.filter(regex=r"^F[1-7]_").to_numpy(dtype=float)
    nonfinite = int((~np.isfinite(feature_values)).sum())
    if nonfinite:
        result.loc[:, result.columns.str.match(r"^F[1-7]_")] = result.filter(
            regex=r"^F[1-7]_"
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    audit = {
        "unresolved_net_slots_replaced_with_zero": unresolved_net,
        "unresolved_solar_slots_replaced_with_zero": unresolved_solar,
        "nonfinite_feature_values_replaced_with_zero": nonfinite,
        "candidate_count": len(result),
    }
    return result, audit


def substation_solar_scale(solar_days: np.ndarray, spec: CandidateSpec) -> float:
    """Median daily daytime P95 solar, computed without labels."""

    prepared = [
        prepare_day_values(
            row,
            max_internal_gap_slots=spec.max_internal_gap_slots,
            nonnegative=True,
        )[0]
        for row in np.asarray(solar_days)
    ]
    matrix = np.vstack(prepared)
    daily_p95 = np.percentile(
        matrix[:, spec.scan_start_slot : spec.scan_end_slot + 1],
        95,
        axis=1,
    )
    return max(float(np.median(daily_p95)), spec.epsilon)


def add_substation_relative_features(
    candidates: pd.DataFrame,
    *,
    spec: CandidateSpec,
) -> pd.DataFrame:
    """Add label-free F8/F9 using each substation's daily best core score."""

    required = {"substation_id", "date", "core_score"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate cache is missing columns: {sorted(missing)}")
    result = candidates.copy()
    groups = ["dataset", "substation_id"] if "dataset" in result else ["substation_id"]
    day_groups = [*groups, "date"]
    daily_best = (
        result.groupby(day_groups, as_index=False)["core_score"]
        .max()
        .rename(columns={"core_score": "daily_best_core_score"})
    )
    daily_best["substation_median_daily_best_core_score"] = daily_best.groupby(groups)[
        "daily_best_core_score"
    ].transform("median")
    daily_best["substation_daily_best_core_rank_pct"] = daily_best.groupby(groups)[
        "daily_best_core_score"
    ].rank(method="average", pct=True)
    result = result.merge(daily_best, on=day_groups, how="left", validate="many_to_one")
    centered = (
        result["core_score"] - result["substation_median_daily_best_core_score"]
    )
    result["F8_substation_centered_core_score"] = np.clip(
        centered,
        -spec.robust_bound_scale,
        spec.robust_bound_scale,
    ) / spec.robust_bound_scale
    result["F9_substation_rank_core_score"] = (
        2 * result["substation_daily_best_core_rank_pct"] - 1
    )
    return result


def build_substation_candidate_features(
    *,
    dataset: str,
    keys: pd.DataFrame,
    net_load_days: np.ndarray,
    solar_days: np.ndarray,
    spec: CandidateSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one resumable substation partition and its day-level audits."""

    if len(keys) != len(net_load_days) or len(keys) != len(solar_days):
        raise ValueError("Day keys, net-load rows, and solar rows must align.")
    substations = keys["substation_id"].astype(str).unique().tolist()
    if len(substations) != 1:
        raise ValueError(f"Expected one substation partition, found {substations}.")

    solar_scale = substation_solar_scale(solar_days, spec)
    candidate_parts = []
    audit_rows = []
    for day_index, key in keys.reset_index(drop=True).iterrows():
        candidates, audit = compute_candidate_features(
            net_load_days[day_index],
            solar_days[day_index],
            substation_solar_scale=solar_scale,
            spec=spec,
        )
        candidates.insert(0, "dataset", dataset)
        candidates.insert(1, "substation_id", str(key["substation_id"]))
        candidates.insert(2, "date", str(key["date"]))
        candidate_parts.append(candidates)
        audit_rows.append(
            {
                "dataset": dataset,
                "substation_id": str(key["substation_id"]),
                "date": str(key["date"]),
                "true_day": bool(key["true_day"]),
                "true_interval_count": int(key["true_interval_count"]),
                "confidence": str(key["confidence"]),
                "input_rows": int(key["n_rows"]),
                "input_missing_net_slots": int(key["n_missing_net"]),
                "input_missing_solar_slots": int(key["n_missing_solar"]),
                "substation_solar_scale_MW": solar_scale,
                **audit,
            }
        )

    candidate_frame = pd.concat(candidate_parts, ignore_index=True)
    candidate_frame = add_substation_relative_features(candidate_frame, spec=spec)
    audit_frame = pd.DataFrame(audit_rows)
    daily_best = (
        candidate_frame.sort_values(
            ["date", "core_score", "left_slot", "duration_slots"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("date", keep="first")
        [
            [
                "dataset",
                "substation_id",
                "date",
                "candidate_id",
                "left_slot",
                "right_slot",
                "core_score",
                "daily_best_core_score",
                "substation_median_daily_best_core_score",
                "substation_daily_best_core_rank_pct",
            ]
        ]
        .reset_index(drop=True)
    )

    integer_columns = [
        "candidate_id",
        "left_slot",
        "right_slot",
        "duration_slots",
        "solar_peak_slot",
    ]
    float_columns = [
        "duration_hours",
        *FEATURE_COLUMNS,
        "core_score",
        "daily_best_core_score",
        "substation_median_daily_best_core_score",
        "substation_daily_best_core_rank_pct",
    ]
    candidate_frame[integer_columns] = candidate_frame[integer_columns].astype("int16")
    candidate_frame[float_columns] = candidate_frame[float_columns].astype("float32")
    return candidate_frame, audit_frame, daily_best


def maximum_subset_scores(
    candidates: pd.DataFrame,
    *,
    feature_columns: list[str],
    weight_matrix: np.ndarray,
    batch_days: int = 32,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return each model's maximum candidate score for every day in a partition."""

    if weight_matrix.ndim != 2 or weight_matrix.shape[1] != len(feature_columns):
        raise ValueError("weight_matrix must have one column per feature.")
    required = {"dataset", "substation_id", "date", "candidate_id", *feature_columns}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate partition is missing columns: {sorted(missing)}")

    ordered = candidates.sort_values(["date", "candidate_id"], kind="mergesort")
    group_sizes = ordered.groupby("date", sort=False).size().to_numpy(dtype=int)
    group_starts = np.r_[0, np.cumsum(group_sizes)[:-1]]
    group_ends = np.cumsum(group_sizes)
    day_keys = ordered.loc[group_starts, ["dataset", "substation_id", "date"]].reset_index(
        drop=True
    )
    features = ordered[feature_columns].to_numpy(dtype=np.float32)
    weights = np.asarray(weight_matrix, dtype=np.float32)
    maxima = np.empty((len(group_sizes), len(weights)), dtype=np.float32)

    for first_day in range(0, len(group_sizes), batch_days):
        last_day = min(first_day + batch_days, len(group_sizes))
        row_start = group_starts[first_day]
        row_end = group_ends[last_day - 1]
        scores = features[row_start:row_end] @ weights.T
        local_starts = group_starts[first_day:last_day] - row_start
        maxima[first_day:last_day] = np.maximum.reduceat(scores, local_starts, axis=0)
    return day_keys, maxima


def score_candidates(
    candidates: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Score candidates with explicit weights and verify all active features exist."""

    active = [feature for feature, weight in weights.items() if weight != 0]
    missing = set(active) - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate frame is missing active features: {sorted(missing)}")
    result = candidates.copy()
    result["score"] = sum(
        float(weights[feature]) * pd.to_numeric(result[feature], errors="coerce").fillna(0)
        for feature in active
    )
    return result


def select_best_candidates(
    candidates: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Choose one candidate per day; ties prefer earlier then shorter windows."""

    scored = score_candidates(candidates, weights)
    group_columns = ["dataset", "substation_id", "date"]
    group_columns = [column for column in group_columns if column in scored]
    ordered = scored.sort_values(
        [*group_columns, "score", "left_slot", "duration_slots", "candidate_id"],
        ascending=[*[True] * len(group_columns), False, True, True, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates(group_columns, keep="first").reset_index(drop=True)


def corrected_net_load(
    net_load: np.ndarray,
    *,
    predicted_day: bool,
    left_slot: int | None,
    right_slot: int | None,
) -> np.ndarray:
    """Flip observed net-load signs only inside a predicted-positive window."""

    result = np.asarray(net_load, dtype=float).copy()
    if predicted_day:
        if left_slot is None or right_slot is None:
            raise ValueError("A predicted-positive day requires candidate bounds.")
        result[left_slot : right_slot + 1] *= -1
    return result


def slots_for_window(left_slot: int | None, right_slot: int | None) -> set[int]:
    if left_slot is None or right_slot is None:
        return set()
    return set(range(int(left_slot), int(right_slot) + 1))


def window_iou(true_slots: Iterable[int], predicted_slots: Iterable[int]) -> float:
    """Set IoU; one-sided empty windows score zero and two empty windows are undefined."""

    truth = set(true_slots)
    prediction = set(predicted_slots)
    union = truth | prediction
    if not union:
        return float("nan")
    return len(truth & prediction) / len(union)


def correction_energy_components(
    net_load: np.ndarray,
    true_interval: np.ndarray,
    predicted_interval: np.ndarray,
    *,
    slot_hours: float = 0.25,
) -> dict[str, float]:
    """Correction-energy precision, recall, F1, and IoU for aligned intervals."""

    energy = 2 * np.maximum(np.asarray(net_load, dtype=float), 0.0) * slot_hours
    truth = np.asarray(true_interval, dtype=bool)
    prediction = np.asarray(predicted_interval, dtype=bool)
    if not (energy.shape == truth.shape == prediction.shape):
        raise ValueError("Energy and interval arrays must have the same shape.")
    manual = float(energy[truth].sum())
    predicted = float(energy[prediction].sum())
    overlap = float(energy[truth & prediction].sum())
    union = float(energy[truth | prediction].sum())
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / manual if manual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "manual_correction_MWh": manual,
        "predicted_correction_MWh": predicted,
        "overlap_correction_MWh": overlap,
        "union_correction_MWh": union,
        "energy_precision": precision,
        "energy_recall": recall,
        "energy_f1": f1,
        "energy_iou": overlap / union if union else float("nan"),
    }

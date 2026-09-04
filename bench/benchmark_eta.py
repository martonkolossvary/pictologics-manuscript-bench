"""Empirical wall-clock ETA estimation for monotone benchmark plans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
import statistics
import time
from typing import Any, Mapping, Sequence

from bench.benchmark_models import STATUS_MEASURED, STATUS_TIMED_OUT, TERMINAL_STATUSES


# Known benchmark kernels range from approximately linear voxel traversal to
# Spatial autocorrelation's dense pairwise ROI work is quadratic in ROI
# voxel count; a larger exponent would imply growth beyond any reviewed
# implementation and would amplify sparse early observations unrealistically.
MAX_VOXEL_EXPONENT = 2.0


@dataclass(frozen=True)
class EtaEstimate:
    seconds: float
    lower_seconds: float
    upper_seconds: float
    observed_task_count: int
    pending_task_count: int
    exact_repeat_predictions: int
    scoped_scaling_predictions: int
    adapter_workload_scaling_predictions: int
    fallback_predictions: int
    timeout_capped_predictions: int
    timeout_cutoff_predictions: int
    timeout_repeat_predictions: int

    @property
    def basis(self) -> str:
        basis = (
            "empirical full-task turnaround scaled by ROI voxels from "
            f"{self.observed_task_count} measured tasks"
        )
        if self.timeout_capped_predictions:
            basis += (
                f"; {self.timeout_capped_predictions} extrapolations capped at "
                "the two-phase timeout bound"
            )
        if self.timeout_cutoff_predictions:
            basis += (
                f"; {self.timeout_cutoff_predictions} pending tasks are expected "
                "to be skipped by observed timeout cutoffs"
            )
        if self.timeout_repeat_predictions:
            basis += (
                f"; {self.timeout_repeat_predictions} exact repeats use observed "
                "timeout turnaround"
            )
        return basis


@dataclass(frozen=True)
class _Observation:
    case_id: str
    adapter: str
    workload: str
    scope: str
    voxels: int
    seconds: float


@dataclass(frozen=True)
class _Prediction:
    seconds: float
    lower_seconds: float
    upper_seconds: float


@dataclass(frozen=True)
class _PowerModel:
    exponent: float
    fixed_seconds: float
    coefficient: float
    voxel_scale: float
    residual_margin: float
    minimum_voxels: int
    maximum_voxels: int
    observed_span: float


def _field(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None else value


def _spec(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = _field(row, "spec_json", "{}")
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _predictor_voxels(row: Mapping[str, Any], spec: Mapping[str, Any]) -> int:
    """Use ROI voxels, falling back to total image voxels/plan complexity."""

    for value in (
        spec.get("mask_voxels"),
        spec.get("image_voxels"),
        spec.get("complexity"),
        _field(row, "complexity"),
    ):
        parsed = _positive_int(value)
        if parsed is not None:
            return parsed
    return 1


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _median_prediction(seconds: Sequence[float]) -> _Prediction:
    center = statistics.median(seconds)
    if len(seconds) == 1:
        return _Prediction(center, center / 2.0, center * 2.0)
    lower = _quantile(seconds, 0.10)
    upper = _quantile(seconds, 0.90)
    return _Prediction(center, min(lower, center), max(upper, center))


def _fit_power_model(
    observations: Sequence[_Observation],
) -> _PowerModel | None:
    """Fit one bounded power model for reuse across every pending task."""

    by_voxels: dict[int, list[float]] = defaultdict(list)
    for observation in observations:
        by_voxels[observation.voxels].append(observation.seconds)
    points = sorted(
        (value, statistics.median(samples))
        for value, samples in by_voxels.items()
    )
    if len(points) < 2:
        return None

    log_points = [(math.log(value), math.log(seconds)) for value, seconds in points]
    if len(log_points) <= 32:
        slopes = [
            (right_y - left_y) / (right_x - left_x)
            for index, (left_x, left_y) in enumerate(log_points)
            for right_x, right_y in log_points[index + 1 :]
            if right_x > left_x
        ]
        if not slopes:
            return None
        raw_exponent = statistics.median(slopes)
    else:
        # A bounded least-squares slope keeps per-progress work linear for
        # heterogeneous real-world cohorts with hundreds of distinct ROIs.
        mean_x = statistics.fmean(value for value, _ in log_points)
        mean_y = statistics.fmean(value for _, value in log_points)
        denominator = sum((value - mean_x) ** 2 for value, _ in log_points)
        if denominator <= 0:
            return None
        raw_exponent = sum(
            (value - mean_x) * (seconds - mean_y)
            for value, seconds in log_points
        ) / denominator
    exponent = min(MAX_VOXEL_EXPONENT, max(0.0, raw_exponent))
    voxel_scale = float(points[-1][0])
    fixed_seconds = 0.0
    coefficient = math.exp(
        statistics.median(
            log_seconds - exponent * log_voxels
            for log_voxels, log_seconds in log_points
        )
    ) * voxel_scale**exponent

    # Once three size levels exist, separate fixed process/import overhead from
    # voxel-dependent work. A pure log-log fit otherwise forces startup cost to
    # scale with the image and systematically overstates or distorts the curve.
    if len(points) >= 3:
        best: tuple[float, float, float, float] | None = None
        for candidate_exponent in (
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
        ):
            transformed = [
                (value / voxel_scale) ** candidate_exponent
                for value, _ in points
            ]
            if candidate_exponent == 0.0:
                candidate_coefficient = 0.0
                candidate_fixed = statistics.median(value for _, value in points)
            else:
                slopes = [
                    (points[right][1] - points[left][1])
                    / (transformed[right] - transformed[left])
                    for left in range(len(points))
                    for right in range(left + 1, len(points))
                    if transformed[right] > transformed[left]
                ]
                candidate_coefficient = max(0.0, statistics.median(slopes))
                candidate_fixed = max(
                    0.0,
                    statistics.median(
                        seconds - candidate_coefficient * transformed[index]
                        for index, (_, seconds) in enumerate(points)
                    ),
                )
            predictions = [
                max(
                    1e-12,
                    candidate_fixed + candidate_coefficient * transformed[index],
                )
                for index in range(len(points))
            ]
            score = _quantile(
                [
                    abs(math.log(seconds / predictions[index]))
                    for index, (_, seconds) in enumerate(points)
                ],
                0.90,
            )
            candidate = (
                score,
                candidate_exponent,
                candidate_fixed,
                candidate_coefficient,
            )
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        _, exponent, fixed_seconds, coefficient = best
    residuals = [
        abs(
            math.log(seconds)
            - math.log(
                max(
                    1e-12,
                    fixed_seconds
                    + coefficient * (voxels / voxel_scale) ** exponent,
                )
            )
        )
        for voxels, seconds in points
    ]
    # With only two size levels a power law interpolates perfectly but its
    # extrapolation uncertainty is not zero. Retain an explicit conservative
    # floor that narrows only after more distinct voxel levels are observed.
    if len(points) == 2:
        minimum_margin = math.log(2.0)
    elif len(points) == 3:
        minimum_margin = math.log(1.5)
    else:
        minimum_margin = math.log(1.25)
    minimum_voxels = points[0][0]
    maximum_voxels = points[-1][0]
    observed_span = max(
        math.log(maximum_voxels / minimum_voxels),
        math.log(2.0),
    )
    return _PowerModel(
        exponent=exponent,
        fixed_seconds=fixed_seconds,
        coefficient=coefficient,
        voxel_scale=voxel_scale,
        residual_margin=max(minimum_margin, _quantile(residuals, 0.90)),
        minimum_voxels=minimum_voxels,
        maximum_voxels=maximum_voxels,
        observed_span=observed_span,
    )


def _power_prediction(model: _PowerModel | None, voxels: int) -> _Prediction | None:
    if model is None:
        return None
    prediction = max(
        1e-12,
        model.fixed_seconds
        + model.coefficient * (max(1, voxels) / model.voxel_scale) ** model.exponent,
    )
    margin = model.residual_margin
    minimum_voxels = model.minimum_voxels
    maximum_voxels = model.maximum_voxels
    if voxels < minimum_voxels or voxels > maximum_voxels:
        boundary = minimum_voxels if voxels < minimum_voxels else maximum_voxels
        extrapolation_distance = abs(math.log(voxels / boundary))
        margin += min(
            math.log(4.0),
            math.log(2.0) * extrapolation_distance / model.observed_span,
        )

    return _Prediction(
        prediction,
        prediction / math.exp(margin),
        prediction * math.exp(margin),
    )


def _observation(row: Mapping[str, Any]) -> _Observation | None:
    if str(_field(row, "status", "")) != STATUS_MEASURED:
        return None
    try:
        started = float(_field(row, "started_at"))
        finished = float(_field(row, "finished_at"))
    except (TypeError, ValueError):
        return None
    seconds = finished - started
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    spec = _spec(row)
    return _Observation(
        case_id=str(_field(row, "case_id", spec.get("case_id") or "")),
        adapter=str(_field(row, "adapter", spec.get("adapter") or "")),
        workload=str(
            _field(row, "workload_key", spec.get("workload") or "")
        ),
        scope=str(spec.get("guardrail_group") or ""),
        voxels=_predictor_voxels(row, spec),
        seconds=seconds,
    )


def _timeout_observation(row: Mapping[str, Any]) -> _Observation | None:
    """Return controller turnaround for an exact-repeat ETA, not a runtime fit."""

    if str(_field(row, "status", "")) != STATUS_TIMED_OUT:
        return None
    try:
        started = float(_field(row, "started_at"))
        finished = float(_field(row, "finished_at"))
    except (TypeError, ValueError):
        return None
    seconds = finished - started
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    spec = _spec(row)
    return _Observation(
        case_id=str(_field(row, "case_id", spec.get("case_id") or "")),
        adapter=str(_field(row, "adapter", spec.get("adapter") or "")),
        workload=str(_field(row, "workload_key", spec.get("workload") or "")),
        scope=str(spec.get("guardrail_group") or ""),
        voxels=_predictor_voxels(row, spec),
        seconds=seconds,
    )


def estimate_pending_turnaround(
    rows: Sequence[Mapping[str, Any]],
    *,
    now: float | None = None,
    maximum_task_seconds: float | None = None,
    timeout_cutoffs: Sequence[Mapping[str, Any]] = (),
) -> EtaEstimate | None:
    """Estimate current-ledger wall time without borrowing incompatible runs."""

    observations = [
        observation
        for row in rows
        if (observation := _observation(row)) is not None
    ]
    if not observations:
        return None
    timeout_observations = [
        observation
        for row in rows
        if (observation := _timeout_observation(row)) is not None
    ]

    exact: dict[tuple[str, str, str], list[_Observation]] = defaultdict(list)
    scoped: dict[tuple[str, str, str], list[_Observation]] = defaultdict(list)
    grouped: dict[tuple[str, str], list[_Observation]] = defaultdict(list)
    timeout_exact: dict[tuple[str, str, str], list[_Observation]] = defaultdict(list)
    for observation in observations:
        exact[
            (observation.case_id, observation.adapter, observation.workload)
        ].append(observation)
        scoped[
            (observation.adapter, observation.workload, observation.scope)
        ].append(observation)
        grouped[(observation.adapter, observation.workload)].append(observation)
    for observation in timeout_observations:
        timeout_exact[
            (observation.case_id, observation.adapter, observation.workload)
        ].append(observation)

    scoped_models = {
        key: _fit_power_model(items) for key, items in scoped.items()
    }
    grouped_models = {
        key: _fit_power_model(items) for key, items in grouped.items()
    }

    global_seconds = [observation.seconds for observation in observations]
    estimate = lower = upper = 0.0
    pending_count = exact_count = scoped_count = grouped_count = fallback_count = 0
    timeout_capped_count = 0
    timeout_cutoff_count = 0
    timeout_repeat_count = 0
    cutoff_by_scope = {
        (
            str(item.get("adapter") or ""),
            str(item.get("workload_key") or ""),
            str(item.get("guardrail_group") or ""),
        ): (
            str(item.get("complexity_metric") or "plan_complexity"),
            int(item["cutoff_complexity"]),
        )
        for item in timeout_cutoffs
        if item.get("cutoff_complexity") is not None
    }
    observed_now = time.time() if now is None else float(now)

    for row in rows:
        status = str(_field(row, "status", ""))
        if status in TERMINAL_STATUSES:
            continue
        spec = _spec(row)
        expected_count = spec.get("expected_feature_count")
        if expected_count is not None and int(expected_count) == 0:
            continue
        pending_count += 1
        case_id = str(_field(row, "case_id", spec.get("case_id") or ""))
        adapter = str(_field(row, "adapter", spec.get("adapter") or ""))
        workload = str(
            _field(row, "workload_key", spec.get("workload") or "")
        )
        scope = str(spec.get("guardrail_group") or "")
        cutoff = cutoff_by_scope.get((adapter, workload, scope))
        complexity = None
        if cutoff is not None:
            metric, cutoff_value = cutoff
            complexity = _positive_int(
                spec.get("image_voxels")
                if metric == "image_voxels"
                else spec.get("complexity") or _field(row, "complexity")
            )
        if (
            cutoff is not None
            and complexity is not None
            and complexity > cutoff_value
        ):
            timeout_cutoff_count += 1
            continue
        voxels = _predictor_voxels(row, spec)

        timeout_repeats = timeout_exact.get((case_id, adapter, workload), [])
        exact_observations = exact.get((case_id, adapter, workload), [])
        if timeout_repeats:
            prediction = _median_prediction(
                [item.seconds for item in timeout_repeats]
            )
            timeout_repeat_count += 1
        elif exact_observations:
            prediction = _median_prediction(
                [item.seconds for item in exact_observations]
            )
            exact_count += 1
        else:
            prediction = _power_prediction(
                scoped_models.get((adapter, workload, scope)), voxels
            )
            if prediction is not None:
                scoped_count += 1
            else:
                prediction = _power_prediction(
                    grouped_models.get((adapter, workload)), voxels
                )
                if prediction is not None:
                    grouped_count += 1
                else:
                    group_observations = grouped.get((adapter, workload), [])
                    prediction = _median_prediction(
                        [item.seconds for item in group_observations]
                        or global_seconds
                    )
                    fallback_count += 1

        task_seconds = prediction.seconds
        task_lower = prediction.lower_seconds
        task_upper = prediction.upper_seconds
        if (
            maximum_task_seconds is not None
            and math.isfinite(float(maximum_task_seconds))
            and float(maximum_task_seconds) > 0
            and task_upper > float(maximum_task_seconds)
        ):
            cap = float(maximum_task_seconds)
            task_seconds = min(task_seconds, cap)
            task_lower = min(task_lower, cap)
            task_upper = cap
            timeout_capped_count += 1
        if status == "running":
            try:
                already_elapsed = max(
                    0.0,
                    observed_now - float(_field(row, "started_at")),
                )
            except (TypeError, ValueError):
                already_elapsed = 0.0
            task_seconds = max(0.0, task_seconds - already_elapsed)
            task_lower = max(0.0, task_lower - already_elapsed)
            task_upper = max(task_seconds, task_upper - already_elapsed)

        estimate += task_seconds
        lower += task_lower
        upper += task_upper

    return EtaEstimate(
        seconds=estimate,
        lower_seconds=lower,
        upper_seconds=upper,
        observed_task_count=len(observations),
        pending_task_count=pending_count,
        exact_repeat_predictions=exact_count,
        scoped_scaling_predictions=scoped_count,
        adapter_workload_scaling_predictions=grouped_count,
        fallback_predictions=fallback_count,
        timeout_capped_predictions=timeout_capped_count,
        timeout_cutoff_predictions=timeout_cutoff_count,
        timeout_repeat_predictions=timeout_repeat_count,
    )


__all__ = ["EtaEstimate", "estimate_pending_turnaround"]

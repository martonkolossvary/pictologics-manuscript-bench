from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable


BENCHMARK_EVENT_PREFIX = "BENCH_EVENT "
DEFAULT_RESULT_RTOL = 1e-9
DEFAULT_RESULT_ATOL = 1e-12
DEFAULT_CALIBRATION_MIN_ROUNDS = 3
DEFAULT_CALIBRATION_MAX_ROUNDS = 24
DEFAULT_CALIBRATION_CV_THRESHOLD = 0.05
DEFAULT_CALIBRATION_SPAN_RATIO = 1.10


class ResultEquivalenceError(RuntimeError):
    """Raised when repeated native calculations return different results."""


class TimingStabilityError(RuntimeError):
    """Raised when a task cannot satisfy the reviewed timing-window policy."""


def assert_numerically_equivalent(
    reference: Any,
    observed: Any,
    *,
    path: str = "result",
    rtol: float = DEFAULT_RESULT_RTOL,
    atol: float = DEFAULT_RESULT_ATOL,
) -> None:
    """Require identical names/structure and numerically equivalent values."""

    import numpy as np

    if isinstance(reference, Mapping) or isinstance(observed, Mapping):
        if not isinstance(reference, Mapping) or not isinstance(observed, Mapping):
            raise ResultEquivalenceError(f"{path} changed result type")
        reference_keys = set(reference)
        observed_keys = set(observed)
        if reference_keys != observed_keys:
            missing = sorted(str(value) for value in reference_keys - observed_keys)
            extra = sorted(str(value) for value in observed_keys - reference_keys)
            raise ResultEquivalenceError(
                f"{path} changed feature names; missing={missing[:5]}, extra={extra[:5]}"
            )
        for key in reference:
            assert_numerically_equivalent(
                reference[key],
                observed[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
            )
        return

    sequence_types = (list, tuple)
    if isinstance(reference, sequence_types) or isinstance(observed, sequence_types):
        if not isinstance(reference, sequence_types) or not isinstance(
            observed, sequence_types
        ):
            raise ResultEquivalenceError(f"{path} changed result type")
        if len(reference) != len(observed):
            raise ResultEquivalenceError(f"{path} changed sequence length")
        for index, (left, right) in enumerate(zip(reference, observed)):
            assert_numerically_equivalent(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return

    try:
        reference_array = np.asarray(reference)
        observed_array = np.asarray(observed)
    except Exception as exc:
        if reference != observed:
            raise ResultEquivalenceError(f"{path} changed value") from exc
        return

    if reference_array.shape != observed_array.shape:
        raise ResultEquivalenceError(f"{path} changed array shape")
    if reference_array.dtype.kind in "biufc" and observed_array.dtype.kind in "biufc":
        if not np.isfinite(reference_array).all() or not np.isfinite(
            observed_array
        ).all():
            raise ResultEquivalenceError(f"{path} contains a non-finite value")
        if not np.allclose(
            reference_array,
            observed_array,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            maximum_delta = float(
                np.max(np.abs(reference_array.astype(float) - observed_array.astype(float)))
            )
            raise ResultEquivalenceError(
                f"{path} changed numeric value (maximum absolute delta {maximum_delta:.6g})"
            )
        return
    if not np.array_equal(reference_array, observed_array):
        raise ResultEquivalenceError(f"{path} changed value")


def _calibration_is_stable(
    samples: Sequence[float],
    *,
    minimum_rounds: int,
    cv_threshold: float,
    span_ratio: float,
) -> tuple[bool, float | None, float | None]:
    if len(samples) < minimum_rounds:
        return False, None, None
    import numpy as np

    recent = np.asarray(samples[-minimum_rounds:], dtype=float)
    mean = float(np.mean(recent))
    coefficient_of_variation = (
        float(np.std(recent)) / mean if mean > 0 else float("inf")
    )
    minimum = float(np.min(recent))
    observed_span = float(np.max(recent)) / minimum if minimum > 0 else float("inf")
    return (
        coefficient_of_variation <= cv_threshold and observed_span <= span_ratio,
        coefficient_of_variation,
        observed_span,
    )


def _self_peak_rss_bytes() -> int | None:
    """Return a process-owned peak RSS observation using only the standard library."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    if value <= 0:
        return None
    # Darwin reports bytes; Linux and the BSDs exposed by our supported CI
    # images report KiB.  The controller still performs process-tree polling;
    # this worker value makes short-lived event boundaries race-free.
    return value if sys.platform == "darwin" else value * 1024


def write_json(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def write_event(event: str, **details: Any) -> None:
    """Emit one machine-readable worker lifecycle event outside timed regions."""

    payload = {"event": str(event), **details}
    worker_peak = _self_peak_rss_bytes()
    if worker_peak is not None:
        payload["worker_self_peak_rss_bytes"] = worker_peak
    sys.stdout.write(
        BENCHMARK_EVENT_PREFIX
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    sys.stdout.flush()


def _seconds(start_ns: int, stop_ns: int) -> float:
    return (stop_ns - start_ns) / 1_000_000_000.0


def run_timed_computation(
    compute_fn: Callable[[Any], Any],
    iterations: int = 3,
    *,
    prepare_fn: Callable[[], Any] | None = None,
    finalize_fn: Callable[[Any, Any], Any] | None = None,
    event_fn: Callable[..., None] | None = None,
    target_observation_window_sec: float = 0.05,
    maximum_calls_per_observation: int = 4096,
    calibration_headroom_factor: float = 2.0,
    calibration_minimum_rounds: int = DEFAULT_CALIBRATION_MIN_ROUNDS,
    calibration_maximum_rounds: int = DEFAULT_CALIBRATION_MAX_ROUNDS,
    calibration_cv_threshold: float = DEFAULT_CALIBRATION_CV_THRESHOLD,
    calibration_span_ratio: float = DEFAULT_CALIBRATION_SPAN_RATIO,
    result_rtol: float = DEFAULT_RESULT_RTOL,
    result_atol: float = DEFAULT_RESULT_ATOL,
):
    """Measure calculation only, with preparation/finalisation outside the clock.

    ``prepare_fn`` may build package-native ROI objects, validate/cast masks,
    reset caches, or otherwise prepare one fresh calculation state.  The
    returned state is passed to ``compute_fn``.  ``finalize_fn`` converts the
    raw result to the adapter payload after the calculation clock has stopped.
    All three stages are measured separately so the primary calculation scope
    remains narrow without discarding operational evidence.  Very short
    calculations are repeated within each observation window.  Only calculation
    intervals are accumulated; preparation and finalisation remain outside the
    clock between every call.  Reported calculation samples are normalized to
    one native grouped-workload call.
    """

    import numpy as np

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be an integer >= 1")
    if not math.isfinite(float(target_observation_window_sec)) or float(
        target_observation_window_sec
    ) <= 0:
        raise ValueError("target observation window must be finite and positive")
    if (
        isinstance(maximum_calls_per_observation, bool)
        or not isinstance(maximum_calls_per_observation, int)
        or maximum_calls_per_observation < 1
    ):
        raise ValueError("maximum calls per observation must be an integer >= 1")
    if not math.isfinite(float(calibration_headroom_factor)) or float(
        calibration_headroom_factor
    ) < 1.0:
        raise ValueError("calibration headroom factor must be finite and >= 1")
    if calibration_minimum_rounds < 2:
        raise ValueError("calibration minimum rounds must be >= 2")
    if calibration_maximum_rounds < calibration_minimum_rounds:
        raise ValueError("calibration maximum rounds must be >= minimum rounds")
    if not 0 < float(calibration_cv_threshold) < 1:
        raise ValueError("calibration CV threshold must be between 0 and 1")
    if float(calibration_span_ratio) < 1:
        raise ValueError("calibration span ratio must be >= 1")
    if not math.isfinite(float(result_rtol)) or float(result_rtol) < 0:
        raise ValueError("result rtol must be finite and non-negative")
    if not math.isfinite(float(result_atol)) or float(result_atol) < 0:
        raise ValueError("result atol must be finite and non-negative")

    prepare = prepare_fn or (lambda: None)
    finalize = finalize_fn or (lambda raw, _state: raw)
    invoke = (
        (lambda state: compute_fn(state))
        if prepare_fn is not None
        else (lambda _state: compute_fn())
    )
    emit = event_fn or (lambda *_args, **_kwargs: None)

    if prepare_fn is None:
        state = None
        warmup_prepare_sec = 0.0
    else:
        start = time.perf_counter_ns()
        state = prepare()
        warmup_prepare_sec = _seconds(start, time.perf_counter_ns())
    sw = time.perf_counter_ns()
    sc = time.process_time_ns()
    raw = invoke(state)
    ew = time.perf_counter_ns()
    ec = time.process_time_ns()
    warmup_duration_sec = _seconds(sw, ew)
    warmup_cpu_time_sec = _seconds(sc, ec)
    if finalize_fn is None:
        reference_result = raw
        warmup_finalize_sec = 0.0
    else:
        start = time.perf_counter_ns()
        reference_result = finalize(raw, state)
        warmup_finalize_sec = _seconds(start, time.perf_counter_ns())
    emit(
        "warmup_complete",
        calculation_sec=warmup_duration_sec,
        preparation_sec=warmup_prepare_sec,
        finalization_sec=warmup_finalize_sec,
    )

    headroom_target_sec = float(target_observation_window_sec) * float(
        calibration_headroom_factor
    )
    initial_calls_per_observation = min(
        maximum_calls_per_observation,
        max(
            1,
            int(
                math.ceil(
                    headroom_target_sec
                    / max(warmup_duration_sec, 1e-12)
                )
            ),
        ),
    )

    # Import/JIT/cache work is intentionally paid by the warmup. Every task then
    # receives at least one independent post-warmup verification call. This is
    # essential for JIT-backed adapters: the warmup can be seconds long while
    # the first steady-state call is only milliseconds. Short verification calls
    # receive multiple untimed calibration windows. Batching is sized from the
    # fastest observed call so measured windows cannot inherit an optimistic
    # transient state.
    calibration_calls = 0
    calibration_rounds = 0
    calibration_duration_sec = 0.0
    calibration_cpu_time_sec = 0.0
    calibration_window_samples = []
    calibration_per_call_samples = []
    calibration_calls_per_round = []
    equivalence_checks = 0
    calibration_stable = False
    calibration_stability_cv = None
    calibration_stability_span = None
    if not calibration_stable:
        round_calls = initial_calls_per_observation
        for _round in range(calibration_maximum_rounds):
            window_duration = 0.0
            window_cpu = 0.0
            for _ in range(round_calls):
                state = prepare() if prepare_fn is not None else None
                sw = time.perf_counter_ns()
                sc = time.process_time_ns()
                raw = invoke(state)
                ew = time.perf_counter_ns()
                ec = time.process_time_ns()
                window_duration += _seconds(sw, ew)
                window_cpu += _seconds(sc, ec)
                current_result = (
                    raw if finalize_fn is None else finalize(raw, state)
                )
                assert_numerically_equivalent(
                    reference_result,
                    current_result,
                    rtol=result_rtol,
                    atol=result_atol,
                )
                equivalence_checks += 1
            calibration_rounds += 1
            calibration_calls += round_calls
            calibration_calls_per_round.append(round_calls)
            calibration_duration_sec += window_duration
            calibration_cpu_time_sec += window_cpu
            per_call = window_duration / round_calls
            calibration_window_samples.append(window_duration)
            calibration_per_call_samples.append(per_call)
            if calibration_rounds == 1 and per_call >= headroom_target_sec:
                # The first independent post-warmup window already provides
                # the full reviewed timing headroom. A later transiently slow
                # window must not bypass the multi-window stability test.
                # This still avoids repeating a genuinely minutes-long call.
                calibration_stable = True
                calibration_stability_cv = 0.0
                calibration_stability_span = 1.0
            else:
                (
                    calibration_stable,
                    calibration_stability_cv,
                    calibration_stability_span,
                ) = _calibration_is_stable(
                    calibration_per_call_samples,
                    minimum_rounds=calibration_minimum_rounds,
                    cv_threshold=calibration_cv_threshold,
                    span_ratio=calibration_span_ratio,
                )
            fastest = min(calibration_per_call_samples)
            round_calls = min(
                maximum_calls_per_observation,
                max(1, int(math.ceil(headroom_target_sec / max(fastest, 1e-12)))),
            )
            if calibration_stable:
                break
        if not calibration_stable:
            raise TimingStabilityError(
                "untimed calibration did not reach steady state within "
                f"{calibration_maximum_rounds} windows "
                f"(CV={calibration_stability_cv}, span={calibration_stability_span})"
            )
        calibration_per_call_sec = calibration_duration_sec / calibration_calls
        calls_per_observation = round_calls
    else:
        calibration_per_call_sec = None
        calls_per_observation = 1
    emit(
        "calibration_complete",
        calibration_calls=calibration_calls,
        calibration_rounds=calibration_rounds,
        calculation_sec=calibration_duration_sec,
        cpu_time_sec=calibration_cpu_time_sec,
        calls_per_observation=calls_per_observation,
        stable=calibration_stable,
    )

    n_measured = iterations

    durations = []
    cpu_times = []
    preparation_times = []
    finalization_times = []
    observation_windows = []
    cpu_observation_windows = []

    res = None
    emit(
        "worker_ready",
        measured_observations=n_measured,
        calls_per_observation=calls_per_observation,
    )
    for index in range(n_measured):
        preparation_total = 0.0
        finalization_total = 0.0
        calculation_total = 0.0
        cpu_total = 0.0
        for call_index in range(calls_per_observation):
            if prepare_fn is None:
                state = None
            else:
                start = time.perf_counter_ns()
                state = prepare()
                preparation_total += _seconds(start, time.perf_counter_ns())
            if call_index == 0:
                emit(
                    "calculation_start",
                    iteration=index + 1,
                    calls_per_observation=calls_per_observation,
                )
            sw = time.perf_counter_ns()
            sc = time.process_time_ns()
            raw = invoke(state)
            ew = time.perf_counter_ns()
            ec = time.process_time_ns()
            calculation_total += _seconds(sw, ew)
            cpu_total += _seconds(sc, ec)
            if finalize_fn is None:
                current_result = raw
            else:
                start = time.perf_counter_ns()
                current_result = finalize(raw, state)
                finalization_total += _seconds(start, time.perf_counter_ns())
            assert_numerically_equivalent(
                reference_result,
                current_result,
                rtol=result_rtol,
                atol=result_atol,
            )
            equivalence_checks += 1
            res = current_result

        duration = calculation_total / calls_per_observation
        cpu_time = cpu_total / calls_per_observation
        durations.append(duration)
        cpu_times.append(cpu_time)
        observation_windows.append(calculation_total)
        cpu_observation_windows.append(cpu_total)
        preparation_times.append(preparation_total / calls_per_observation)
        finalization_times.append(finalization_total / calls_per_observation)
        emit(
            "calculation_complete",
            iteration=index + 1,
            calculation_sec=duration,
            cpu_time_sec=cpu_time,
            observation_window_sec=calculation_total,
            cpu_observation_window_sec=cpu_total,
            calls_per_observation=calls_per_observation,
        )
        if calculation_total < float(target_observation_window_sec):
            raise TimingStabilityError(
                "measured calculation window fell below the required minimum: "
                f"{calculation_total:.9g}s < {float(target_observation_window_sec):.9g}s"
            )

    stats = {
        # Median is the primary observation. Minima remain available as a
        # diagnostic but are too optimistic for adapter comparisons.
        "duration_sec": float(np.median(durations)),
        "duration_min_sec": float(np.min(durations)),
        "duration_mean_sec": float(np.mean(durations)),
        "duration_median_sec": float(np.median(durations)),
        "duration_std_sec": float(np.std(durations)) if len(durations) > 1 else 0.0,
        "duration_max_sec": float(np.max(durations)),
        "cpu_time_sec": float(np.median(cpu_times)),
        "cpu_time_min_sec": float(np.min(cpu_times)),
        "cpu_time_mean_sec": float(np.mean(cpu_times)),
        "cpu_time_median_sec": float(np.median(cpu_times)),
        "cpu_time_std_sec": float(np.std(cpu_times)) if len(cpu_times) > 1 else 0.0,
        "cpu_time_max_sec": float(np.max(cpu_times)),
        "measured_iterations": n_measured,
        "measured_observations": n_measured,
        "warmup_iterations": 1,
        "total_iterations": n_measured + 1,
        "calls_per_observation": calls_per_observation,
        "calibration_calls": calibration_calls,
        "calibration_rounds": calibration_rounds,
        "calibration_duration_sec": calibration_duration_sec,
        "calibration_cpu_time_sec": calibration_cpu_time_sec,
        "calibration_per_call_sec": calibration_per_call_sec,
        "calibration_window_samples_sec": [
            float(value) for value in calibration_window_samples
        ],
        "calibration_per_call_samples_sec": [
            float(value) for value in calibration_per_call_samples
        ],
        "calibration_calls_per_round": [
            int(value) for value in calibration_calls_per_round
        ],
        "calibration_stability_cv": calibration_stability_cv,
        "calibration_stability_span": calibration_stability_span,
        "calibration_stable": bool(calibration_stable),
        "calibration_minimum_rounds": calibration_minimum_rounds,
        "calibration_maximum_rounds": calibration_maximum_rounds,
        "calibration_cv_threshold": float(calibration_cv_threshold),
        "calibration_span_ratio": float(calibration_span_ratio),
        "calibration_headroom_factor": float(calibration_headroom_factor),
        "measured_calculation_calls": n_measured * calls_per_observation,
        "total_calculation_calls": (
            1 + calibration_calls + n_measured * calls_per_observation
        ),
        "target_observation_window_sec": float(target_observation_window_sec),
        "maximum_calls_per_observation": maximum_calls_per_observation,
        "minimum_observation_window_sec": float(min(observation_windows)),
        "result_equivalence_checks": equivalence_checks,
        "result_equivalence_passed": True,
        "result_equivalence_rtol": float(result_rtol),
        "result_equivalence_atol": float(result_atol),
        "warmup_duration_sec": warmup_duration_sec,
        "warmup_cpu_time_sec": warmup_cpu_time_sec,
        "warmup_preparation_sec": warmup_prepare_sec,
        "warmup_finalization_sec": warmup_finalize_sec,
        "duration_samples_sec": [float(value) for value in durations],
        "cpu_time_samples_sec": [float(value) for value in cpu_times],
        "observation_window_samples_sec": [
            float(value) for value in observation_windows
        ],
        "cpu_observation_window_samples_sec": [
            float(value) for value in cpu_observation_windows
        ],
        "preparation_samples_sec": [float(value) for value in preparation_times],
        "finalization_samples_sec": [float(value) for value in finalization_times],
    }
    return res, stats

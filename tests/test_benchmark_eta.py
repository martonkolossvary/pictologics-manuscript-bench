from __future__ import annotations

import json

import pytest

from bench.benchmark_eta import estimate_pending_turnaround


def _row(
    *,
    ordinal: int,
    case_id: str,
    voxels: int,
    status: str,
    started: float | None = None,
    finished: float | None = None,
    repeat: int = 1,
    expected_feature_count: int = 10,
) -> dict[str, object]:
    return {
        "task_id": f"task-{ordinal}",
        "ordinal": ordinal,
        "case_id": case_id,
        "adapter": "adapter",
        "workload_key": "texture",
        "complexity": voxels * 2,
        "status": status,
        "started_at": started,
        "finished_at": finished,
        # Deliberately unrelated to turnaround: ETA must not use the internal
        # calculation-only duration as wall-clock task time.
        "duration_sec": 0.001 if status == "measured" else None,
        "spec_json": json.dumps(
            {
                "case_id": case_id,
                "adapter": "adapter",
                "workload": "texture",
                "repeat": repeat,
                "mask_voxels": voxels,
                "image_voxels": voxels * 2,
                "complexity": voxels * 2,
                "guardrail_group": "synthetic-series",
                "expected_feature_count": expected_feature_count,
            }
        ),
    }


def test_eta_learns_voxel_power_from_full_task_turnaround() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="small",
            voxels=1_000,
            status="measured",
            started=10.0,
            finished=20.0,
        ),
        _row(
            ordinal=2,
            case_id="medium",
            voxels=8_000,
            status="measured",
            started=30.0,
            finished=110.0,
        ),
        _row(
            ordinal=3,
            case_id="large",
            voxels=27_000,
            status="pending",
        ),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    # Turnaround scales linearly with total ROI voxels here. The estimator must
    # not cube a voxel count that already represents a 3-D volume.
    assert estimate.seconds == pytest.approx(270.0)
    assert estimate.scoped_scaling_predictions == 1
    assert estimate.observed_task_count == 2
    assert estimate.lower_seconds < estimate.seconds < estimate.upper_seconds


def test_eta_prefers_same_case_from_an_earlier_fresh_process_repeat() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="same-case",
            voxels=10_000,
            status="measured",
            started=100.0,
            finished=112.0,
            repeat=1,
        ),
        _row(
            ordinal=2,
            case_id="same-case",
            voxels=10_000,
            status="pending",
            repeat=2,
        ),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    assert estimate.seconds == pytest.approx(12.0)
    assert estimate.exact_repeat_predictions == 1
    assert estimate.scoped_scaling_predictions == 0


def test_eta_does_not_extrapolate_from_near_identical_roi_sizes() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="same-image-mask-a",
            voxels=1_000,
            status="measured",
            started=10.0,
            finished=11.0,
        ),
        _row(
            ordinal=2,
            case_id="same-image-mask-b",
            voxels=1_100,
            status="measured",
            started=20.0,
            finished=28.0,
        ),
        _row(
            ordinal=3,
            case_id="future-large-image",
            voxels=1_000_000,
            status="pending",
        ),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    assert estimate.seconds == pytest.approx(4.5)
    assert estimate.scoped_scaling_predictions == 0
    assert estimate.adapter_workload_scaling_predictions == 0
    assert estimate.fallback_predictions == 1
    assert "1 median-fallback" in estimate.basis
    assert "voxel-scaled" not in estimate.basis


def test_eta_subtracts_elapsed_time_for_the_active_task() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="prior",
            voxels=10_000,
            status="measured",
            started=100.0,
            finished=120.0,
        ),
        _row(
            ordinal=2,
            case_id="active",
            voxels=10_000,
            status="running",
            started=200.0,
        ),
    ]

    estimate = estimate_pending_turnaround(rows, now=207.0)

    assert estimate is not None
    assert estimate.seconds == pytest.approx(13.0)


def test_eta_ignores_declared_unsupported_pending_tasks() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="prior",
            voxels=10_000,
            status="measured",
            started=100.0,
            finished=120.0,
        ),
        _row(
            ordinal=2,
            case_id="unsupported",
            voxels=10_000,
            status="pending",
            expected_feature_count=0,
        ),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    assert estimate.pending_task_count == 0
    assert estimate.seconds == 0.0


def test_eta_is_unavailable_without_completed_measurements() -> None:
    assert (
        estimate_pending_turnaround(
            [
                _row(
                    ordinal=1,
                    case_id="pending",
                    voxels=1_000,
                    status="pending",
                )
            ]
        )
        is None
    )


def test_eta_does_not_extrapolate_faster_than_reviewed_quadratic_work() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="small",
            voxels=1_000,
            status="measured",
            started=0.0,
            finished=1.0,
        ),
        _row(
            ordinal=2,
            case_id="medium",
            voxels=2_000,
            status="measured",
            started=10.0,
            finished=18.0,
        ),
        _row(
            ordinal=3,
            case_id="large",
            voxels=4_000,
            status="pending",
        ),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    assert 0.0 < estimate.seconds <= 32.0


def test_eta_caps_extreme_extrapolation_at_two_phase_timeout_bound() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="small",
            voxels=1_000,
            status="measured",
            started=0.0,
            finished=1.0,
        ),
        _row(
            ordinal=2,
            case_id="medium",
            voxels=2_000,
            status="measured",
            started=10.0,
            finished=18.0,
        ),
        _row(
            ordinal=3,
            case_id="large",
            voxels=1_000_000,
            status="pending",
        ),
    ]

    estimate = estimate_pending_turnaround(rows, maximum_task_seconds=60.0)

    assert estimate is not None
    assert estimate.seconds == 60.0
    assert estimate.upper_seconds == 60.0
    assert estimate.timeout_capped_predictions == 1


def test_eta_separates_fixed_turnaround_from_voxel_scaling() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="size-1",
            voxels=1_000,
            status="measured",
            started=0.0,
            finished=11.0,
        ),
        _row(
            ordinal=2,
            case_id="size-2",
            voxels=2_000,
            status="measured",
            started=20.0,
            finished=32.0,
        ),
        _row(
            ordinal=3,
            case_id="size-4",
            voxels=4_000,
            status="measured",
            started=40.0,
            finished=54.0,
        ),
        _row(ordinal=4, case_id="size-8", voxels=8_000, status="pending"),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    # 10 seconds fixed process overhead plus one second per 1,000 ROI voxels.
    assert estimate.seconds == pytest.approx(18.0)


def test_eta_excludes_tasks_covered_by_a_timeout_cutoff() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="small",
            voxels=1_000,
            status="measured",
            started=0.0,
            finished=10.0,
        ),
        _row(ordinal=2, case_id="large", voxels=8_000, status="pending"),
    ]

    estimate = estimate_pending_turnaround(
        rows,
        timeout_cutoffs=[
            {
                "adapter": "adapter",
                "workload_key": "texture",
                "guardrail_group": "synthetic-series",
                "complexity_metric": "image_voxels",
                "cutoff_complexity": 1_000,
            }
        ],
    )

    assert estimate is not None
    assert estimate.seconds == 0.0
    assert estimate.timeout_cutoff_predictions == 1


def test_eta_uses_observed_timeout_turnaround_for_an_exact_repeat() -> None:
    rows = [
        _row(
            ordinal=1,
            case_id="measured-reference",
            voxels=1_000,
            status="measured",
            started=0.0,
            finished=10.0,
        ),
        _row(
            ordinal=2,
            case_id="timeout-case",
            voxels=8_000,
            status="timed_out_censored",
            started=20.0,
            finished=80.0,
            repeat=1,
        ),
        _row(
            ordinal=3,
            case_id="timeout-case",
            voxels=8_000,
            status="pending",
            repeat=2,
        ),
    ]

    estimate = estimate_pending_turnaround(rows)

    assert estimate is not None
    assert estimate.seconds == pytest.approx(60.0)
    assert estimate.timeout_repeat_predictions == 1

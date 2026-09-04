from __future__ import annotations

from bench.benchmark_memory import MemoryPreflightPolicy, evaluate_memory_preflight


def test_preflight_does_not_scale_fixed_process_overhead() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=1.0,
        fixed_overhead_bytes=100,
        safety_factor=1.5,
        adapter_input_multipliers=(("pictologics", 2.0),),
    )
    prior = [
        {
            "task_status": "measured",
            "adapter": "pictologics",
            "workload": "texture",
            "guardrail_group": "series",
            "input_uncompressed_bytes": 100,
            "host_peak_rss_bytes": 500,
        }
    ]
    result = evaluate_memory_preflight(
        adapter="pictologics",
        workload_key="texture",
        guardrail_group="series",
        input_uncompressed_bytes=1000,
        available_bytes=100_000,
        total_bytes=100_000,
        prior_records=prior,
        policy=policy,
    )
    assert result["memory_static_estimate_bytes"] == 2100
    assert result["memory_empirical_baseline_bytes"] == 500
    assert result["memory_empirical_projected_increment_bytes"] == 0
    assert result["memory_empirical_estimate_bytes"] == 500
    assert result["memory_estimate_bytes"] == 2100
    assert result["memory_empirical_observation_count"] == 1
    assert result["memory_empirical_same_scope_observation_count"] == 1
    assert result["memory_preflight_decision"] == "launch"


def test_preflight_scales_only_empirical_growth_above_fixed_overhead() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=1.0,
        fixed_overhead_bytes=100,
        safety_factor=1.5,
        adapter_input_multipliers=(("pictologics", 2.0),),
    )
    prior = [
        {
            "task_status": "measured",
            "adapter": "pictologics",
            "workload": "texture",
            "guardrail_group": "series",
            "input_uncompressed_bytes": 100,
            "host_peak_rss_bytes": 500,
        },
        {
            "task_status": "measured",
            "adapter": "pictologics",
            "workload": "texture",
            "guardrail_group": "series",
            "input_uncompressed_bytes": 200,
            "host_peak_rss_bytes": 700,
        },
    ]
    result = evaluate_memory_preflight(
        adapter="pictologics",
        workload_key="texture",
        guardrail_group="series",
        input_uncompressed_bytes=1000,
        available_bytes=100_000,
        total_bytes=100_000,
        prior_records=prior,
        policy=policy,
    )
    assert result["memory_empirical_baseline_bytes"] == 500
    assert result["memory_empirical_projected_increment_bytes"] == 2236
    assert result["memory_empirical_estimate_bytes"] == 3854
    assert result["memory_estimate_bytes"] == 3854


def test_estimate_above_budget_is_advisory_and_still_launches() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=1.0,
        user_cap_bytes=500,
        fixed_overhead_bytes=100,
        adapter_input_multipliers=(("pictologics", 2.0),),
    )
    result = evaluate_memory_preflight(
        adapter="pictologics",
        workload_key="texture",
        guardrail_group="series",
        input_uncompressed_bytes=1000,
        available_bytes=10_000,
        total_bytes=10_000,
        prior_records=[],
        policy=policy,
    )
    assert result["memory_budget_bytes"] == 500
    assert result["memory_estimate_exceeds_budget"] is True
    assert result["memory_preflight_decision"] == "launch"


def test_zrad_spatial_autocorrelation_has_safe_first_launch_quadratic_bound() -> None:
    result = evaluate_memory_preflight(
        adapter="zrad",
        workload_key="spatial_autocorrelation",
        guardrail_group="series",
        input_uncompressed_bytes=331_776,
        mask_voxels=10_665,
        available_bytes=8 * 1024**3,
        total_bytes=8 * 1024**3,
        prior_records=[],
        policy=MemoryPreflightPolicy(),
    )
    assert result["memory_quadratic_static_estimate_bytes"] > 7 * 1024**3
    assert result["memory_estimate_exceeds_budget"] is True
    assert result["memory_preflight_decision"] == "launch"

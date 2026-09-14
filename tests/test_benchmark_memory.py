from __future__ import annotations

from bench.benchmark_memory import MemoryPreflightPolicy, evaluate_memory_preflight


def observation(size, peak, scope="series"):
    return dict(task_status="measured", adapter="pictologics", workload="texture",
                guardrail_group=scope, input_uncompressed_bytes=size,
                host_peak_rss_bytes=peak)


def forecast(target, history):
    return evaluate_memory_preflight(
        adapter="pictologics", workload_key="texture", guardrail_group="series",
        input_uncompressed_bytes=target, available_bytes=20*2**30,
        total_bytes=32*2**30, prior_records=history,
        policy=MemoryPreflightPolicy(user_cap_bytes=12*2**30))


def test_small_worker_noise_does_not_reject_measured_feasible_size():
    # Reproduces the production false-positive at 256^3: ~1 MiB startup noise.
    history=[observation(2*32**3, 283*2**20), observation(2*32**3, 285*2**20),
             observation(2*192**3, 439*2**20)]
    result=forecast(2*256**3, history)
    assert result['memory_preflight_decision']=='launch'
    assert result['memory_estimate_bytes'] < 2*2**30
    assert result['memory_empirical_anchor_input_bytes']==2*192**3


def test_same_size_replicate_noise_is_not_a_growth_slope():
    result=forecast(2*256**3, [observation(2*32**3, 283*2**20),
                              observation(2*32**3, 285*2**20)])
    assert result['memory_empirical_projected_increment_bytes']==0
    assert result['memory_empirical_estimate_bytes']==285*2**20


def test_unrelated_patient_or_mask_cannot_censor_this_scope():
    result=forecast(2*256**3, [observation(2*32**3, 270*2**20),
        observation(2*48**3, 10*2**30, scope='other_patient')])
    assert result['memory_preflight_decision']=='launch'
    assert result['memory_empirical_observation_count']==1


def test_larger_completed_size_does_not_censor_a_smaller_repeat():
    result=forecast(2*32**3, [observation(2*32**3, 270*2**20),
        observation(2*512**3, 10*2**30)])
    assert result['memory_empirical_anchor_input_bytes']==2*32**3
    assert result['memory_estimate_bytes'] < 2**30


def test_genuine_growth_still_blocks_an_oversized_worker():
    result=forecast(2*256**3, [observation(2*32**3, 270*2**20),
        observation(2*192**3, 8*2**30)])
    assert result['memory_preflight_decision']=='skip'


def test_forecast_does_not_depend_on_record_order():
    history=[observation(2*32**3, 283*2**20), observation(2*32**3, 285*2**20),
             observation(2*192**3, 439*2**20)]
    assert forecast(2*256**3,history)==forecast(2*256**3,list(reversed(history)))


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


def test_estimate_above_budget_prevents_launch() -> None:
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
    assert result["memory_preflight_decision"] == "skip"


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
    assert result["memory_preflight_decision"] == "skip"

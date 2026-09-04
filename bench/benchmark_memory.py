"""Conservative, auditable memory preflight for isolated benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


GIB = 1024**3


@dataclass(frozen=True)
class MemoryPreflightPolicy:
    budget_fraction: float = 0.80
    reserve_bytes: int = 4 * GIB
    user_cap_bytes: int | None = None
    safety_factor: float = 1.50
    empirical_growth_exponent: float = 1.50
    fixed_overhead_bytes: int = 512 * 1024**2
    quadratic_mask_terms: tuple[tuple[str, str, float], ...] = (
        # Z-Rad spatial autocorrelation materializes pairwise ROI-coordinate
        # distances.
        # Sixty-four bytes per voxel pair conservatively bounds the observed
        # process-tree growth while preventing unsafe first launches on small
        # memory hosts before an empirical observation exists.
        ("zrad", "spatial_autocorrelation", 64.0),
    )
    # These are conservative feasibility multipliers, not fitted peak-RSS
    # claims. Empirical growth is applied only to RSS above the observed
    # fixed-process baseline; scaling the interpreter and imported-library RSS
    # with image volume grossly overpredicts otherwise lightweight adapters.
    adapter_input_multipliers: tuple[tuple[str, float], ...] = (
        ("pictologics", 10.0),
        ("pyradiomics", 12.0),
        ("mirp", 14.0),
        ("medimage", 18.0),
        ("zrad", 14.0),
    )

    def __post_init__(self) -> None:
        if not 0.0 < float(self.budget_fraction) <= 1.0:
            raise ValueError("memory budget fraction must be in (0, 1]")
        if int(self.reserve_bytes) < 0:
            raise ValueError("memory reserve cannot be negative")
        if self.user_cap_bytes is not None and int(self.user_cap_bytes) <= 0:
            raise ValueError("memory cap must be positive")
        if float(self.safety_factor) < 1.0:
            raise ValueError("memory safety factor must be >= 1")
        if float(self.empirical_growth_exponent) < 1.0:
            raise ValueError("empirical growth exponent must be >= 1")

    def multiplier(self, adapter: str) -> float:
        values = dict(self.adapter_input_multipliers)
        # External adapters are permitted by the runner. Use the most
        # conservative declared built-in multiplier until they have empirical
        # observations, rather than silently disabling their preflight.
        return float(values.get(adapter, max(values.values())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": "fixed_overhead_incremental_empirical_upper",
            "mode": "advisory_only",
            "budget_fraction": self.budget_fraction,
            "reserve_bytes": self.reserve_bytes,
            "user_cap_bytes": self.user_cap_bytes,
            "safety_factor": self.safety_factor,
            "empirical_growth_exponent": self.empirical_growth_exponent,
            "fixed_overhead_bytes": self.fixed_overhead_bytes,
            "quadratic_mask_terms": [
                {
                    "adapter": adapter,
                    "workload": workload,
                    "bytes_per_mask_voxel_pair": coefficient,
                }
                for adapter, workload, coefficient in self.quadratic_mask_terms
            ],
            "adapter_input_multipliers": dict(self.adapter_input_multipliers),
            "empirical_scope": {
                "primary": ["adapter", "workload", "guardrail_group"],
                "cross_stratum_upper": ["adapter", "workload"],
            },
            "interpretation": (
                "advisory estimate only; never gates execution and is never "
                "reported as measured memory"
            ),
        }


def evaluate_memory_preflight(
    *,
    adapter: str,
    workload_key: str,
    guardrail_group: str,
    input_uncompressed_bytes: int,
    mask_voxels: int | None = None,
    available_bytes: int,
    total_bytes: int,
    prior_records: Iterable[Mapping[str, Any]],
    policy: MemoryPreflightPolicy,
) -> dict[str, Any]:
    """Return a complete launch/skip decision without mutating run state."""

    if input_uncompressed_bytes <= 0:
        raise ValueError("task input byte count must be positive")
    reserve = max(int(policy.reserve_bytes), int(0.20 * total_bytes))
    dynamic_budget = max(0, int(available_bytes) - reserve)
    fractional_budget = max(0, int(float(policy.budget_fraction) * available_bytes))
    budget = min(dynamic_budget, fractional_budget)
    if policy.user_cap_bytes is not None:
        budget = min(budget, int(policy.user_cap_bytes))

    linear_static_estimate = int(
        policy.fixed_overhead_bytes
        + policy.multiplier(adapter) * int(input_uncompressed_bytes)
    )
    quadratic_static_estimate = None
    for term_adapter, term_workload, coefficient in policy.quadratic_mask_terms:
        if adapter != term_adapter or workload_key != term_workload:
            continue
        if mask_voxels is None or int(mask_voxels) <= 0:
            raise ValueError(
                f"{adapter}/{workload_key} memory preflight requires mask_voxels"
            )
        quadratic_static_estimate = int(
            policy.fixed_overhead_bytes + float(coefficient) * int(mask_voxels) ** 2
        )
    static_estimate = max(
        linear_static_estimate,
        quadratic_static_estimate or 0,
    )
    observations: list[tuple[int, int, bool]] = []
    for record in prior_records:
        if record.get("task_status") != "measured":
            continue
        if record.get("adapter") != adapter or record.get("workload") != workload_key:
            continue
        previous_input = record.get("input_uncompressed_bytes")
        observed_peak = record.get("host_peak_rss_bytes") or record.get(
            "peak_rss_bytes"
        )
        try:
            previous_input = int(previous_input)
            observed_peak = int(observed_peak)
        except (TypeError, ValueError):
            continue
        if previous_input <= 0 or observed_peak <= 0:
            continue
        observations.append(
            (
                previous_input,
                observed_peak,
                record.get("guardrail_group") == guardrail_group,
            )
        )

    empirical_estimate = None
    empirical_baseline = None
    empirical_projected_increment = None
    if observations:
        empirical_baseline = min(peak for _, peak, _ in observations)
        projected_increments = []
        for previous_input, observed_peak, _ in observations:
            scale = max(1.0, int(input_uncompressed_bytes) / previous_input)
            observed_increment = max(0, observed_peak - empirical_baseline)
            projected_increments.append(
                observed_increment
                * scale ** float(policy.empirical_growth_exponent)
            )
        empirical_projected_increment = int(max(projected_increments))
        empirical_estimate = int(
            empirical_baseline
            + policy.safety_factor * empirical_projected_increment
        )
    estimate = max(
        static_estimate,
        empirical_estimate if empirical_estimate is not None else 0,
    )
    return {
        "memory_preflight_policy_id": "fixed_overhead_incremental_empirical_upper",
        "memory_preflight_enabled": False,
        "input_uncompressed_bytes": int(input_uncompressed_bytes),
        "memory_static_estimate_bytes": static_estimate,
        "memory_linear_static_estimate_bytes": linear_static_estimate,
        "memory_quadratic_static_estimate_bytes": quadratic_static_estimate,
        "memory_empirical_estimate_bytes": empirical_estimate,
        "memory_empirical_baseline_bytes": empirical_baseline,
        "memory_empirical_projected_increment_bytes": empirical_projected_increment,
        "memory_empirical_observation_count": len(observations),
        "memory_empirical_same_scope_observation_count": sum(
            1 for _, _, same_scope in observations if same_scope
        ),
        "memory_empirical_growth_exponent": float(policy.empirical_growth_exponent),
        "memory_estimate_bytes": estimate,
        "memory_available_bytes": int(available_bytes),
        "memory_total_bytes": int(total_bytes),
        "memory_reserve_bytes": reserve,
        "memory_budget_bytes": budget,
        "memory_preflight_decision": "launch",
        "memory_estimate_exceeds_budget": estimate > budget,
    }


__all__ = ["GIB", "MemoryPreflightPolicy", "evaluate_memory_preflight"]

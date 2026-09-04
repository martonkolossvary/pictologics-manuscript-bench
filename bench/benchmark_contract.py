from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bench.benchmark_ledger import sha256_file
from bench.benchmark_workloads import WORKLOADS, WORKLOAD_ORDER
from bench.ibsi_families import FAMILY_ORDER


@dataclass(frozen=True)
class BenchmarkContract:
    contract_id: str
    path: Path
    sha256: str
    payload: Mapping[str, Any]

    def expected_feature_count(self, adapter: str, family: str) -> int:
        inventories = self.payload["adapter_inventories"]
        try:
            value = inventories[adapter]["family_output_counts"][family]
        except KeyError as exc:
            raise ValueError(
                f"endpoint contract has no feature count for {adapter}/{family}"
            ) from exc
        return int(value)

    def expected_workload_count(self, adapter: str, workload: str) -> int:
        inventories = self.payload["adapter_inventories"]
        try:
            value = inventories[adapter]["workload_output_counts"][workload]
        except KeyError as exc:
            raise ValueError(
                f"endpoint contract has no workload count for {adapter}/{workload}"
            ) from exc
        return int(value)


def default_contract_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "benchmark"
        / "calculation_only_workload.json"
    )


def load_benchmark_contract(path: str | Path | None = None) -> BenchmarkContract:
    contract_path = Path(path or default_contract_path()).resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"benchmark endpoint contract not found: {contract_path}"
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 4:
        raise ValueError("benchmark endpoint contract must use schema version 4")
    if payload.get("aggregation") != "3d_merge":
        raise ValueError("benchmark endpoint contract must require 3d_merge")
    if payload.get("workload_order") != list(WORKLOAD_ORDER):
        raise ValueError("benchmark endpoint contract workload order is not canonical")
    expected_workloads = {
        workload.name: {
            "families": list(workload.families),
            "feature_partition": workload.feature_partition,
        }
        for workload in WORKLOADS
    }
    observed_workloads = payload.get("workloads")
    if not isinstance(observed_workloads, dict):
        raise ValueError("benchmark endpoint contract has no grouped workloads")
    for name, expected in expected_workloads.items():
        observed = observed_workloads.get(name)
        if not isinstance(observed, dict) or any(
            observed.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"invalid workload definition for {name}")
    if observed_workloads["morphology"].get("excluded_ibsi_codes") != [
        "N365",
        "NPT7",
    ] or observed_workloads["spatial_autocorrelation"].get(
        "included_ibsi_codes"
    ) != ["N365", "NPT7"]:
        raise ValueError("invalid spatial-autocorrelation feature partition")
    timing = payload.get("timing")
    if not isinstance(timing, dict) or timing != {
        "contract_version": 8,
        "scope": "prepared_workload_inputs_to_radiomic_calculations",
        "file_io_included": False,
        "mask_preparation_included": False,
        "resegmentation_included": False,
        "discretization_included": False,
        "result_normalization_or_serialization_included": False,
        "matrix_mesh_neighborhood_construction_included": True,
        "untimed_warmup_calls_per_process": 1,
        "measured_observations_per_process": 3,
        "adaptive_calls_per_observation": True,
        "untimed_steady_state_calibration": True,
        "multi_window_calibration_convergence": True,
        "calibration_headroom_factor": 2.0,
        "calibration_minimum_rounds": 3,
        "calibration_maximum_rounds": 12,
        "calibration_cv_threshold": 0.05,
        "calibration_span_ratio": 1.1,
        "post_warmup_verification_calls_minimum": 1,
        "single_call_calibration_accepted_above_headroom": True,
        "target_observation_window_seconds": 0.05,
        "measured_window_minimum_enforced": True,
        "maximum_calls_per_observation": 4096,
        "reported_samples_are_per_call": True,
        "within_process_result_equivalence_required": True,
        "fresh_process_result_equivalence_required": True,
        "result_equivalence_rtol": 1e-9,
        "result_equivalence_atol": 1e-12,
        "fresh_process_repeats": 3,
        "within_process_primary_statistic": "median",
        "between_process_primary_statistic": "median",
        "retain_all_raw_samples": True,
        "runtime_normalization": "none",
    }:
        raise ValueError(
            "benchmark endpoint timing contract is not the reviewed policy"
        )
    inventories = payload.get("adapter_inventories")
    if not isinstance(inventories, dict) or not inventories:
        raise ValueError("benchmark endpoint contract has no adapter inventories")
    for adapter, inventory in inventories.items():
        counts = inventory.get("family_output_counts")
        if not isinstance(counts, dict) or set(counts) != set(FAMILY_ORDER):
            raise ValueError(f"invalid family count inventory for {adapter}")
        normalized = [int(counts[family]) for family in FAMILY_ORDER]
        if any(value < 0 for value in normalized):
            raise ValueError(f"negative family feature count for {adapter}")
        if sum(normalized) != int(inventory.get("native_output_count", -1)):
            raise ValueError(f"native feature count does not sum for {adapter}")
        workload_counts = inventory.get("workload_output_counts")
        if not isinstance(workload_counts, dict) or set(workload_counts) != set(
            WORKLOAD_ORDER
        ):
            raise ValueError(f"invalid workload count inventory for {adapter}")
        normalized_workload_counts = {
            name: int(workload_counts[name]) for name in WORKLOAD_ORDER
        }
        if any(value < 0 for value in normalized_workload_counts.values()):
            raise ValueError(f"negative workload feature count for {adapter}")
        expected_complete_counts = {
            "local_intensity": int(counts["local_intensity"]),
            "intensity": int(counts["intensity"]),
            "texture": sum(
                int(counts[family])
                for family in (
                    "histogram",
                    "glcm",
                    "glrlm",
                    "glszm",
                    "gldzm",
                    "ngtdm",
                    "ngldm",
                )
            ),
            "ivh": int(counts["ivh"]),
        }
        for workload, expected in expected_complete_counts.items():
            if normalized_workload_counts[workload] != expected:
                raise ValueError(
                    f"workload feature count does not sum for {adapter}/{workload}"
                )
        if (
            normalized_workload_counts["morphology"]
            + normalized_workload_counts["spatial_autocorrelation"]
            != int(counts["morphology"])
        ):
            raise ValueError(
                f"morphology partitions do not sum for {adapter}"
            )
        if sum(normalized_workload_counts.values()) != int(
            inventory.get("native_output_count", -1)
        ):
            raise ValueError(f"native workload count does not sum for {adapter}")
    evidence = payload.get("evidence", {}).get("feature_surface_contract", {})
    evidence_path = Path(__file__).resolve().parents[1] / str(evidence.get("path", ""))
    if not evidence_path.is_file() or sha256_file(evidence_path) != evidence.get(
        "sha256"
    ):
        raise ValueError(
            "benchmark endpoint feature-surface contract is missing or changed"
        )
    contract_id = str(payload.get("contract_id") or "").strip()
    if not contract_id:
        raise ValueError("benchmark endpoint contract_id is missing")
    return BenchmarkContract(
        contract_id=contract_id,
        path=contract_path,
        sha256=sha256_file(contract_path),
        payload=payload,
    )

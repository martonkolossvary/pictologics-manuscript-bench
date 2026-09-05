"""Build and attest the three-pillar benchmark input workspace.

This module never imports an adapter and never calculates radiomic features.
It serially prepares the two synthetic pillars and the fixed IBSI 2 Phase 3
cohort, validates every committed representation, and writes one immutable
workspace manifest used by later timing runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from bench.adapters.registry import get_adapter
from bench.benchmark_contract import BenchmarkContract, load_benchmark_contract
from bench.benchmark_ledger import sha256_file
from bench.benchmark_workloads import WORKLOADS
from bench.dataset_manifest import DatasetValidationError, atomic_write_json
from bench.ibsi2_phase3_dataset import (
    prepare_ibsi2_phase3_dataset,
    validate_ibsi2_phase3_dataset,
)
from bench.pillar1_dataset import build_pillar1_dataset, validate_pillar1_dataset
from bench.pillar2_dataset import build_pillar2_dataset, validate_pillar2_dataset


WORKSPACE_ID = "pictologics_three_pillar_benchmark_inputs"
WORKSPACE_MANIFEST_SCHEMA_VERSION = 4
ADAPTERS = ("pictologics", "pyradiomics", "mirp", "medimage", "zrad")


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DatasetValidationError(
            f"benchmark input must be inside the workspace root: {path}"
        ) from exc


def _dataset_record(
    root: Path,
    dataset_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": _relative(root, dataset_dir),
        "dataset": manifest["dataset"],
        "dataset_kind": manifest["dataset_kind"],
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "case_count": len(manifest["cases"]),
        "file_count": len(manifest["files"]),
        "validation": dict(validation),
    }


def _task_inventory(
    datasets: Mapping[str, Mapping[str, Any]],
    contract: BenchmarkContract,
) -> dict[str, Any]:
    repeats = int(contract.payload["timing"]["fresh_process_repeats"])
    measured_observations = int(
        contract.payload["timing"]["measured_observations_per_process"]
    )
    warmups = int(contract.payload["timing"]["untimed_warmup_calls_per_process"])
    verification_calls = int(
        contract.payload["timing"]["post_warmup_verification_calls_minimum"]
    )
    workloads = [workload.name for workload in WORKLOADS]
    scheduled_per_case = len(ADAPTERS) * len(workloads) * repeats
    supported_workloads = {
        adapter: [
            workload
            for workload in workloads
            if contract.expected_workload_count(adapter, workload) > 0
        ]
        for adapter in ADAPTERS
    }
    eligible_per_case = (
        sum(len(values) for values in supported_workloads.values()) * repeats
    )
    rows: dict[str, Any] = {}
    total_cases = 0
    for pillar, dataset in datasets.items():
        cases = int(dataset["case_count"])
        total_cases += cases
        scheduled = cases * scheduled_per_case
        eligible = cases * eligible_per_case
        rows[pillar] = {
            "case_count": cases,
            "scheduled_task_records": scheduled,
            "eligible_calculation_tasks": eligible,
            "preempted_unsupported_tasks": scheduled - eligible,
            "discarded_warmup_calls_if_all_eligible_complete": eligible * warmups,
            "minimum_post_warmup_verification_calls_if_all_eligible_complete": (
                eligible * verification_calls
            ),
            "measured_observations_if_all_eligible_complete": (
                eligible * measured_observations
            ),
        }
    scheduled = total_cases * scheduled_per_case
    eligible = total_cases * eligible_per_case
    return {
        "workload_count": len(workloads),
        "adapter_count": len(ADAPTERS),
        "fresh_process_repeats": repeats,
        "measured_observations_per_process": measured_observations,
        "adaptive_calls_per_observation": True,
        "untimed_steady_state_calibration": True,
        "multi_window_calibration_convergence": True,
        "calibration_headroom_factor": 2.0,
        "calibration_minimum_rounds": 3,
        "calibration_maximum_rounds": 24,
        "within_process_result_equivalence_required": True,
        "fresh_process_result_equivalence_required": True,
        "untimed_warmup_calls_per_process": warmups,
        "post_warmup_verification_calls_minimum": verification_calls,
        "supported_workloads_by_adapter": supported_workloads,
        "pillars": rows,
        "totals": {
            "case_count": total_cases,
            "scheduled_task_records": scheduled,
            "eligible_calculation_tasks": eligible,
            "preempted_unsupported_tasks": scheduled - eligible,
            "discarded_warmup_calls_if_all_eligible_complete": eligible * warmups,
            "minimum_post_warmup_verification_calls_if_all_eligible_complete": (
                eligible * verification_calls
            ),
            "measured_observations_if_all_eligible_complete": (
                eligible * measured_observations
            ),
        },
    }


def write_workspace_manifest(
    root: Path,
    *,
    endpoint_contract_path: Path | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    """Deep-validate all pillars and atomically bind the release workspace."""

    root = root.expanduser().resolve()
    pillar1 = root / "pillar1"
    pillar2 = root / "pillar2_a1"
    pillar3 = root / "ibsi2_phase3"
    contract = load_benchmark_contract(endpoint_contract_path)
    validations = {
        "pillar1_morphology": validate_pillar1_dataset(pillar1, deep=deep),
        "pillar2_whole_anatomy": validate_pillar2_dataset(pillar2),
        "pillar3_ibsi2_phase3": validate_ibsi2_phase3_dataset(pillar3),
    }
    datasets = {
        "pillar1_morphology": _dataset_record(
            root, pillar1, validations["pillar1_morphology"]
        ),
        "pillar2_whole_anatomy": _dataset_record(
            root, pillar2, validations["pillar2_whole_anatomy"]
        ),
        "pillar3_ibsi2_phase3": _dataset_record(
            root, pillar3, validations["pillar3_ibsi2_phase3"]
        ),
    }
    repository = Path(__file__).resolve().parents[1]
    workspace_sources = [
        repository / "bench" / "benchmark_workspace.py",
        repository / "bench" / "benchmark_workloads.py",
        repository / "scripts" / "prepare_benchmark_workspace.py",
        repository / "scripts" / "launch_benchmark.py",
        repository / "poetry.lock",
    ]
    record = {
        "schema_version": WORKSPACE_MANIFEST_SCHEMA_VERSION,
        "workspace_id": WORKSPACE_ID,
        "status": "inputs_validated_calculations_not_started",
        "benchmark_timing_executed": False,
        "adapter_order": list(ADAPTERS),
        "endpoint_contract": {
            "contract_id": contract.contract_id,
            "path": contract.path.relative_to(
                Path(__file__).resolve().parents[1]
            ).as_posix(),
            "sha256": contract.sha256,
        },
        "workspace_sources": [
            {
                "path": path.relative_to(repository).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in workspace_sources
        ],
        "datasets": datasets,
        "task_inventory": _task_inventory(datasets, contract),
        "launch_policy": {
            "aggregation": "3d_merge",
            "native_workload_tasks_are_independent": True,
            "reported_workloads": [workload.name for workload in WORKLOADS],
            "post_hoc_runtime_aggregation": "none",
            "runtime_normalization": "none",
            "warmup_is_untimed": True,
            "within_process_primary_statistic": "median",
            "between_process_primary_statistic": "median",
            "timeout_seconds": 1800.0,
            "checkpoint_interval_tasks": 25,
            "progress_interval_seconds": 30.0,
            "speed_truncation_enabled": False,
            "runtime_limit_policy": (
                "per-task timeout censoring, then skip strictly larger images "
                "for the same adapter, workload, mask, and input configuration"
            ),
            "timeout_cutoff_enabled": True,
            "timeout_cutoff_scope": ["adapter", "workload", "guardrail_group"],
            "timeout_cutoff_complexity_metric": "image_voxels",
            "memory_preflight": "advisory_estimate_only_no_skip",
            "unsupported_is_not_zero": True,
            "censored_is_not_zero": True,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "workspace_manifest.json", record)
    return record


def prepare_workspace(
    root: Path,
    *,
    ibsi2_source: Path,
    pillar1_profile: Path,
    pillar2_profile: Path,
    endpoint_contract_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Serially prepare all inputs; no adapter process is invoked."""

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    build_pillar1_dataset(root / "pillar1", profile_path=pillar1_profile, resume=resume)
    build_pillar2_dataset(
        root / "pillar2_a1", profile_path=pillar2_profile, resume=resume
    )
    prepare_ibsi2_phase3_dataset(
        ibsi2_source,
        root / "ibsi2_phase3",
        expected_subjects=51,
        resume=resume,
    )
    return write_workspace_manifest(
        root, endpoint_contract_path=endpoint_contract_path, deep=True
    )


def validate_workspace(root: Path, *, deep: bool = True) -> dict[str, Any]:
    """Recreate the manifest and require its calculation-free status."""

    record = write_workspace_manifest(root, deep=deep)
    for adapter in ADAPTERS:
        get_adapter(adapter)
    return record


__all__ = [
    "ADAPTERS",
    "WORKSPACE_ID",
    "prepare_workspace",
    "validate_workspace",
    "write_workspace_manifest",
]

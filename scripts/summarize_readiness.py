"""Summarize existing untimed IBSI evidence without running adapters or timings.

This is an integrity/coverage snapshot, not authorization to start final timing.
IBSI 1 and IBSI 2 Phase 2 use the same checksummed comparison-table schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.benchmark_ledger import sha256_file
from bench.benchmark_models import fingerprint
from bench.compliance.models import ComparisonRecord
from bench.compliance.references import (
    load_reference_csv,
    select_ibsi1_digital_phantom_profile,
    validate_reference_table_manifest,
)
from bench.compliance.report import load_comparison_csv
from bench.compliance.tolerance import compare_absolute, compare_ibsi1

KINDS = {"ibsi1_digital_phantom", "ibsi2_phase2_from_response_maps"}
UNSUPPORTED = {"unsupported", "native_unsupported"}
MISSING = {
    "missing", "candidate_not_supplied", "configuration_unsupported_or_not_supplied",
    "filter_unsupported_or_not_supplied",
}
ERRORS = {"error", "nonfinite", "ambiguous", "unmapped_reference"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _checked_file(path: Path, expected: Any) -> str:
    observed = sha256_file(path)
    if not expected or observed != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return observed


def _reference_rows(run: dict[str, Any]):
    inputs = run.get("inputs", run)
    references = Path(inputs["references"])
    manifest = Path(inputs["reference_manifest"])
    for name in ("references", "reference_manifest"):
        _checked_file(Path(inputs[name]), inputs.get(f"{name}_sha256"))
    is_ibsi1 = run["kind"] == "ibsi1_digital_phantom"
    validate_reference_table_manifest(
        references, manifest,
        expected_specification="IBSI 1" if is_ibsi1 else "IBSI 2",
        expected_phase=None if is_ibsi1 else "phase2",
    )
    rows = load_reference_csv(references)
    if is_ibsi1:
        for name in ("image", "mask"):
            _checked_file(Path(inputs[name]), inputs.get(f"{name}_sha256"))
        return select_ibsi1_digital_phantom_profile(rows)
    _checked_file(Path(run["candidate_manifest"]), run.get("candidate_manifest_sha256"))
    return [row for row in rows if row.specification == "IBSI 2" and row.phase == "phase2"]


def _key(row) -> tuple[str, ...]:
    return (row.specification, row.phase, row.configuration, row.aggregation, row.feature_tag)


def _check_row(row: ComparisonRecord, reference) -> None:
    allowed = UNSUPPORTED | MISSING | ERRORS | {"pass", "fail", "not_standardized", "reference_unavailable"}
    if row.status not in allowed:
        raise ValueError(f"Unrecognized comparison outcome: {row.status}")
    if (row.standardized != reference.standardized
            or row.semantic_key != reference.semantic_key
            or row.reference_value != reference.reference_value
            or row.tolerance != reference.tolerance):
        raise ValueError(f"Comparison/reference identity mismatch: {row.adapter}/{row.feature_tag}")
    referencable = (reference.standardized and reference.reference_value is not None
                   and reference.tolerance is not None)
    if row.referencable != referencable:
        raise ValueError("Comparison reference availability is inconsistent")
    if row.finite and (row.value is None or not math.isfinite(row.value)):
        raise ValueError("Finite comparison lacks a finite value")
    if row.evaluated:
        if not (row.standardized and row.referencable and row.finite and row.observed_supported):
            raise ValueError("Evaluated comparison lacks standardized, finite supported evidence")
        compare = compare_ibsi1 if row.specification == "IBSI 1" else compare_absolute
        expected = compare(row.reference_value, row.value, row.tolerance)
        if row.passed != expected.passed or row.status != ("pass" if expected.passed else "fail"):
            raise ValueError("Comparison tolerance outcome is inconsistent")
    elif row.passed is not None or row.status in {"pass", "fail"}:
        raise ValueError("Unevaluated comparison cannot be a pass or failure")


def _environment_identity(adapter: str, profile: dict, environment: dict) -> dict:
    if (environment.get("profile") != adapter
            or environment.get("version") != profile.get("version")
            or environment.get("distribution") != profile.get("distribution")
            or environment.get("distribution_metadata_version")
            != profile.get("distribution_metadata_version")):
        raise ValueError(f"Environment/profile identity mismatch: {adapter}")
    freeze = environment.get("freeze")
    if not isinstance(freeze, list) or not freeze:
        raise ValueError(f"Missing recorded package freeze: {adapter}")
    digest = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()
    lock = environment.get("environment_lock") or {}
    if digest != environment.get("freeze_sha256") or digest != lock.get("freeze_sha256"):
        raise ValueError(f"Recorded environment freeze/lock mismatch: {adapter}")
    return {
        "adapter": adapter, "version": profile["version"],
        "distribution": profile["distribution"],
        "distribution_metadata_version": profile["distribution_metadata_version"],
        "platform": environment.get("platform"), "machine": environment.get("machine"),
        "environment_freeze_sha256": digest, "environment_lock": lock,
    }


def _counts(rows: list[ComparisonRecord]) -> dict[str, Any]:
    standardized = [row for row in rows if row.standardized]
    evaluated = sum(row.evaluated for row in standardized)
    passed = sum(row.passed is True for row in standardized)
    failed = sum(row.passed is False for row in standardized)
    return {
        "defined_instances": len(rows),
        "native_supported_instances": sum(row.observed_supported for row in rows),
        "finite_instances": sum(row.finite for row in rows),
        "standardized_instances": len(standardized),
        "reference_available_standardized": sum(row.referencable for row in standardized),
        "evaluated_standardized": evaluated,
        "passed_standardized": passed, "failed_standardized": failed,
        "unevaluated_standardized": len(standardized) - evaluated,
        "unsupported_standardized": sum(row.status in UNSUPPORTED for row in standardized),
        "missing_instances": sum(row.status in MISSING for row in rows),
        "error_instances": sum(row.status in ERRORS for row in rows),
        "coverage_only_instances": sum(not row.standardized for row in rows),
        "finite_coverage_only_instances": sum(not row.standardized and row.finite for row in rows),
        "reference_unavailable_standardized": sum(not row.referencable for row in standardized),
        "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
        "conditional_accuracy": {"numerator": passed, "denominator": evaluated,
                                 "value": passed / evaluated if evaluated else None},
        "overall_standardized_success": {
            "numerator": passed, "denominator": len(standardized),
            "value": passed / len(standardized) if standardized else None,
        },
    }


def summarize_run(directory: Path) -> dict[str, Any]:
    """Read one evidence bundle. Missing/changed evidence can never qualify."""
    root = directory.expanduser().resolve()
    result: dict[str, Any] = {
        "directory": str(root), "status": "missing", "integrity": "unverified",
        "issues": [], "adapters": [],
    }
    required = ("run_manifest.json", "result_manifest.json", "run_state.json", "comparisons.csv")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        result["status"] = "incomplete" if root.exists() else "missing"
        result["issues"] = [f"Missing evidence: {name}" for name in missing]
        return result
    try:
        before = {name: sha256_file(root / name) for name in required}
        run = _json(root / "run_manifest.json")
        manifest = _json(root / "result_manifest.json")
        state = _json(root / "run_state.json")
        if run.get("schema_version") != 1 or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported compliance evidence schema")
        if run.get("kind") not in KINDS or manifest.get("kind") != run["kind"]:
            raise ValueError("Expected IBSI 1 or IBSI 2 Phase 2 comparison-table evidence")
        identity = dict(run)
        stored_fingerprint = identity.pop("fingerprint", None)
        if not stored_fingerprint or fingerprint(identity) != stored_fingerprint:
            raise ValueError("Run manifest fingerprint mismatch")
        if not run.get("controller_source_sha256"):
            raise ValueError("Missing recorded controller source identity")
        _checked_file(root / "comparisons.csv", manifest.get("comparison_table_sha256"))
        references = {_key(row): row for row in _reference_rows(run)}
        rows = load_comparison_csv(root / "comparisons.csv")
        if manifest.get("record_count") != len(rows):
            raise ValueError("Comparison row count differs from result manifest")
        adapters = run["adapters"]
        if not adapters or len(adapters) != len(set(adapters)):
            raise ValueError("Missing or duplicate requested adapter identity")
        if set(row.adapter for row in rows) != set(adapters):
            raise ValueError("Comparison adapter set differs from run manifest")
        tasks = state.get("tasks", {})
        if state.get("schema_version") != 1 or set(tasks) != set(run["tasks"]):
            raise ValueError("Run-state task grid differs from run manifest")
        for task in tasks.values():
            if task.get("status") == "completed":
                path = (root / task["payload_path"]).resolve()
                if root not in path.parents:
                    raise ValueError("Payload path escapes evidence directory")
                _checked_file(path, task.get("payload_sha256"))
        processing_complete = (manifest.get("complete") is True if run["kind"] == "ibsi1_digital_phantom"
                               else manifest.get("publication_complete") is True)
        if any(task.get("status") != "completed" for task in tasks.values()):
            processing_complete = False
        for adapter in adapters:
            profile = run["configured_adapter_profiles"][adapter]
            item = _environment_identity(adapter, profile, run["verified_adapter_environments"][adapter])
            selected = [row for row in rows if row.adapter == adapter]
            if Counter(_key(row) for row in selected) != Counter(references.keys()):
                raise ValueError(f"Comparison grid is incomplete or duplicated: {adapter}")
            for row in selected:
                if row.software_version != profile["version"]:
                    raise ValueError(f"Comparison version differs from frozen profile: {adapter}")
                _check_row(row, references[_key(row)])
            counts = _counts(selected)
            if counts["error_instances"]:
                status = "execution_or_mapping_errors"
            elif counts["missing_instances"] or not processing_complete:
                status = "incomplete"
            elif counts["failed_standardized"]:
                status = "numerical_failures"
            elif not counts["evaluated_standardized"]:
                status = "no_evaluated_accuracy_evidence"
            elif counts["unevaluated_standardized"] or counts["coverage_only_instances"]:
                status = "qualified_with_coverage_limits"
            else:
                status = "qualified"
            item.update(status=status, counts=counts)
            result["adapters"].append(item)
        after = {name: sha256_file(root / name) for name in required}
        if before != after:
            raise ValueError("Evidence changed during snapshot; retry after the run stops")
        result.update(
            kind=run["kind"], integrity="recorded_evidence_verified",
            evidence_sha256=before, run_fingerprint=stored_fingerprint,
            controller_source_sha256=run["controller_source_sha256"],
            processing_complete=processing_complete,
            status="qualified_with_coverage_limits" if all(
                item["status"] in {"qualified", "qualified_with_coverage_limits"}
                for item in result["adapters"]
            ) else "needs_review",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(status="invalid_evidence", integrity="unverified", adapters=[])
        result["issues"].append(str(error))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON snapshot outside evidence directories")
    args = parser.parse_args(argv)
    if args.output:
        target = args.output.resolve()
        if any(target == root.resolve() or root.resolve() in target.parents for root in args.run_dir):
            parser.error("Snapshot output must be outside input evidence directories")
    runs = [summarize_run(path) for path in args.run_dir]
    snapshot = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing untimed IBSI 1 / IBSI 2 Phase 2 evidence only",
        "final_timing_status": "deferred_until_pictologics_finalized",
        "limitations": [
            "Recorded source/environment identities are preserved; current installations are not reverified.",
            "IBSI 2 Phase 1, native smoke, workspace and host qualification are separate evidence.",
            "Coverage limits and numerical failures must accompany any accuracy claim.",
        ],
        "runs": runs,
    }
    output = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if all(run["status"] == "qualified_with_coverage_limits" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

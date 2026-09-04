from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
from importlib import metadata as importlib_metadata
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import re
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench.benchmark_ledger import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from bench.ibsi_mapping import classify_feature
from bench.power_provenance import summarize_task_power_records


BASELINE_ADAPTER = "pictologics"
MEASURED_TASK_STATUS = "measured"

PUBLIC_TASK_COLUMNS = (
    "run_id", "task_id", "task_status", "attempt", "case_id", "dataset",
    "modality", "subject_id", "size", "variant", "mask_id", "mask_label",
    "image_sha256", "source_image_sha256", "mask_sha256", "input_contract",
    "input_representation_id", "representation_derivation_sha256",
    "configured_levels", "occupied_levels", "shape", "spacing", "image_voxels",
    "mask_voxels", "mask_fraction", "complexity", "adapter",
    "adapter_distribution", "adapter_version", "workload",
    "requested_families",
    "guardrail_group", "repeat", "discretization", "effective_bins",
    "effective_bin_width", "intensity_min", "intensity_max",
    "requested_timing_observations", "endpoint_contract_id",
    "endpoint_contract_sha256", "expected_feature_count",
    "input_uncompressed_bytes", "success", "duration_sec", "duration_min_sec",
    "duration_mean_sec", "duration_median_sec", "duration_std_sec",
    "duration_max_sec", "duration_samples_sec", "cpu_time_sec",
    "cpu_time_min_sec", "cpu_time_mean_sec", "cpu_time_median_sec",
    "cpu_time_std_sec", "cpu_time_max_sec", "cpu_time_samples_sec",
    "preparation_samples_sec", "finalization_samples_sec", "warmup_duration_sec",
    "warmup_cpu_time_sec", "warmup_preparation_sec", "warmup_finalization_sec",
    "measured_iterations", "measured_observations", "warmup_iterations",
    "total_iterations", "calls_per_observation", "measured_calculation_calls",
    "calibration_calls", "calibration_duration_sec",
    "calibration_cpu_time_sec", "calibration_per_call_sec",
    "calibration_headroom_factor", "calibration_rounds",
    "calibration_window_samples_sec", "calibration_per_call_samples_sec",
    "calibration_calls_per_round", "calibration_stability_cv",
    "calibration_stability_span", "calibration_stable",
    "minimum_observation_window_sec", "result_equivalence_checks",
    "result_equivalence_passed", "result_equivalence_rtol",
    "result_equivalence_atol",
    "fresh_process_reference_task_id", "fresh_process_reference_repeat",
    "fresh_process_result_equivalence_passed",
    "fresh_process_result_equivalence_rtol",
    "fresh_process_result_equivalence_atol",
    "total_calculation_calls", "target_observation_window_sec",
    "maximum_calls_per_observation", "observation_window_samples_sec",
    "cpu_observation_window_samples_sec",
    "peak_rss_bytes", "host_peak_rss_bytes", "worker_ready_rss_bytes",
    "calculation_peak_rss_bytes", "incremental_calculation_peak_rss_bytes",
    "host_wall_time_sec", "adapter_event_count",
    "host_session_index", "host_session_observed_at_utc",
    "host_power_observation_scope", "host_power_start_observed_at_utc",
    "host_power_end_observed_at_utc", "host_power_start_mode_tag",
    "host_power_end_mode_tag", "host_power_start_probe_source",
    "host_power_end_probe_source", "host_power_start_probe_attempts",
    "host_power_end_probe_attempts", "host_power_mode_changed_during_task",
    "host_power_mode_tag", "host_energy_mode",
    "host_energy_mode_observation_status", "host_power_source",
    "host_pmset_lowpowermode", "host_power_probe_errors",
    "host_power_probe_diagnostics",
    "memory_phase_observation_status", "timing_source", "timing_scope",
    "memory_scope", "feature_count", "attempted_feature_count",
    "finite_feature_count", "censor_lower_bound_sec",
    "partial_duration_samples_sec", "partial_cpu_time_samples_sec",
    "partial_completed_iterations", "policy_reason", "memory_preflight_policy_id",
    "timeout_cutoff_complexity", "timeout_cutoff_complexity_metric",
    "timeout_cutoff_evidence_task_id",
    "memory_preflight_enabled", "memory_estimate_exceeds_budget",
    "memory_static_estimate_bytes",
    "memory_linear_static_estimate_bytes",
    "memory_quadratic_static_estimate_bytes",
    "memory_empirical_estimate_bytes", "memory_empirical_observation_count",
    "memory_empirical_baseline_bytes",
    "memory_empirical_projected_increment_bytes",
    "memory_empirical_growth_exponent",
    "memory_empirical_same_scope_observation_count",
    "memory_estimate_bytes", "memory_available_bytes", "memory_total_bytes",
    "memory_reserve_bytes", "memory_budget_bytes", "memory_preflight_decision",
)

PUBLIC_QC_ISSUE_COLUMNS = (
    "run_id", "case_id", "dataset", "size", "variant", "mask_id",
    "adapter", "workload", "requested_families", "repeat", "family",
    "host_session_index", "host_power_observation_scope",
    "host_power_start_mode_tag", "host_power_end_mode_tag",
    "host_power_mode_changed_during_task", "host_power_mode_tag",
    "host_energy_mode", "host_energy_mode_observation_status",
    "issue_type", "severity", "metric",
)

# Every series differs by color, marker, and line pattern. This keeps the
# performance figures interpretable in grayscale and for common color-vision
# deficiencies.
ADAPTER_STYLES = {
    "pictologics": {
        "color": "#0F4C81",
        "marker": "o",
        "linestyle": "-",
        "label": "Pictologics",
    },
    "pyradiomics": {
        "color": "#E66C00",
        "marker": "s",
        "linestyle": "--",
        "label": "PyRadiomics",
    },
    "mirp": {
        "color": "#6F3FA0",
        "marker": "^",
        "linestyle": ":",
        "label": "MIRP",
    },
    "medimage": {
        "color": "#007A3D",
        "marker": "D",
        "linestyle": "-.",
        "label": "MEDimage",
    },
    "zrad": {
        "color": "#B61C4A",
        "marker": "P",
        "linestyle": (0, (5, 2, 1, 2)),
        "label": "Z-Rad",
    },
}

ReportRecordLoader = Callable[
    [Path], pd.DataFrame | tuple[pd.DataFrame, Mapping[str, Any]]
]


def _report_generator_provenance() -> Dict[str, Any]:
    """Describe the code and rendering environment used for this report."""

    bench_root = Path(__file__).resolve().parent
    sources = {
        path.relative_to(bench_root).as_posix(): sha256_file(path)
        for path in sorted(bench_root.rglob("*.py"))
    }
    source_payload = json.dumps(
        sources,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dependencies = {
        "matplotlib": str(matplotlib.__version__),
        "numpy": str(np.__version__),
        "openpyxl": importlib_metadata.version("openpyxl"),
        "pandas": str(pd.__version__),
    }
    provenance: Dict[str, Any] = {
        "schema_version": 1,
        "source_tree_sha256": hashlib.sha256(source_payload).hexdigest(),
        "python_version": platform.python_version(),
        "dependencies": dependencies,
    }
    provenance_payload = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance["provenance_sha256"] = hashlib.sha256(provenance_payload).hexdigest()
    return provenance


def _atomic_write_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def publication_task_observations(
    records: pd.DataFrame,
    verified_payloads: list[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a public, path-free task table including every raw timing sample."""

    if "task_id" not in records.columns:
        raise ValueError("Report records must include task_id")
    if records["task_id"].astype(str).duplicated().any():
        raise ValueError("Report records contain duplicate task_id values")

    payload_timings: dict[str, Mapping[str, Any]] = {}
    for payload in verified_payloads:
        benchmark = payload.get("benchmark")
        timing = payload.get("timing")
        if not isinstance(benchmark, Mapping) or not isinstance(timing, Mapping):
            raise ValueError("Verified measured payload is missing benchmark/timing data")
        task_id = str(benchmark.get("task_id") or "")
        if not task_id or task_id in payload_timings:
            raise ValueError("Verified measured payload has an invalid task identity")
        payload_timings[task_id] = timing

    timing_fields = {
        "duration_samples_sec",
        "cpu_time_samples_sec",
        "preparation_samples_sec",
        "finalization_samples_sec",
        "observation_window_samples_sec",
        "cpu_observation_window_samples_sec",
        "warmup_duration_sec",
        "warmup_cpu_time_sec",
        "warmup_preparation_sec",
        "warmup_finalization_sec",
        "calibration_calls",
        "calibration_rounds",
        "calibration_duration_sec",
        "calibration_cpu_time_sec",
        "calibration_per_call_sec",
        "calibration_window_samples_sec",
        "calibration_per_call_samples_sec",
        "calibration_calls_per_round",
        "calibration_stability_cv",
        "calibration_stability_span",
        "calibration_stable",
        "calibration_headroom_factor",
        "minimum_observation_window_sec",
        "result_equivalence_checks",
        "result_equivalence_passed",
        "result_equivalence_rtol",
        "result_equivalence_atol",
        "target_observation_window_sec",
        "maximum_calls_per_observation",
    }
    rows: list[dict[str, Any]] = []
    for raw_record in records.to_dict(orient="records"):
        task_id = str(raw_record.get("task_id") or "")
        row = {
            column: raw_record.get(column)
            for column in PUBLIC_TASK_COLUMNS
            if column not in timing_fields
        }
        timing = payload_timings.get(task_id, {})
        for field in timing_fields:
            row[field] = timing.get(field, raw_record.get(field))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=PUBLIC_TASK_COLUMNS)
    for column in frame.columns:
        if frame[column].dtype != "object":
            continue
        frame[column] = frame[column].map(
            lambda value: json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if isinstance(value, (dict, list, tuple))
            else value
        )
    return frame


def _atomic_save_figure(figure: plt.Figure, path: Path) -> None:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format=path.suffix.lstrip("."),
        bbox_inches="tight",
    )
    atomic_write_bytes(path, buffer.getvalue())


def timing_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid, uncensored timing observations.

    ``task_status`` is authoritative. Timeout bounds, failures, unsupported
    tasks, and policy skips are never converted into measured durations.
    """

    if "task_status" not in df.columns:
        raise ValueError("Report records must include authoritative task_status values")
    observations = df.copy()
    status = observations["task_status"].fillna("").astype(str).str.strip().str.lower()
    observations = observations.loc[status == MEASURED_TASK_STATUS].copy()

    if "duration_sec" not in observations.columns:
        return observations.iloc[0:0].copy()

    observations["duration_sec"] = pd.to_numeric(
        observations["duration_sec"], errors="coerce"
    )
    valid = np.isfinite(observations["duration_sec"]) & (
        observations["duration_sec"] > 0.0
    )
    return observations.loc[valid].copy()


def with_report_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the explicit native workload used by the benchmark controller."""

    out = df.copy()
    if "workload" not in out.columns:
        raise ValueError("Report records must include explicit workload values")
    group = out["workload"].fillna("").astype(str).str.strip()
    if group.eq("").any() or group.str.lower().eq("nan").any():
        raise ValueError("Report records contain a missing workload value")
    source = pd.Series("workload", index=out.index, dtype="object")

    out["_report_workload"] = group
    out["_report_grouping_dimension"] = source
    return out


def _safe_report_suffix(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token or "all"


def _normalise_dataset_kind(value: object) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"synthetic", "synthetic_data"}:
        return "synthetic"
    if token in {"real", "real_world", "realworld", "validation"}:
        return "real_world"
    return token or "unknown"


def _resolve_dataset_kind(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        return _normalise_dataset_kind(explicit)

    for column in ("_dataset_kind", "dataset_kind"):
        if column not in df.columns:
            continue
        values = {
            _normalise_dataset_kind(value)
            for value in df[column].dropna().tolist()
            if str(value).strip()
        }
        if len(values) > 1:
            raise ValueError(
                f"Report input mixes dataset kinds in {column}: {sorted(values)}"
            )
        if values:
            return next(iter(values))

    if "modality" in df.columns:
        modalities = {
            str(value).strip().lower()
            for value in df["modality"].dropna().tolist()
            if str(value).strip()
        }
        if modalities == {"synthetic"}:
            return "synthetic"
        if modalities and "synthetic" not in modalities:
            return "real_world"
    return "unknown"


def _ledger_record_loader(
    input_dir: Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Load authoritative records and verify every committed measured payload."""

    from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError
    from bench.benchmark_models import fingerprint, run_spec_identity

    ledger_path = input_dir / "benchmark.sqlite3"
    with BenchmarkLedger(ledger_path) as ledger:
        records, payloads = ledger.verified_records_and_payloads()
        status_counts = ledger.status_counts()
        run_status = ledger.metadata_value("run_status")
        run_fingerprint = ledger.metadata_value("run_fingerprint")
        run_spec_json = ledger.metadata_value("run_spec_json")

    if not run_spec_json:
        raise RunIntegrityError("benchmark ledger has no run_spec_json")
    try:
        decoded = json.loads(run_spec_json)
    except json.JSONDecodeError as exc:
        raise RunIntegrityError(
            "benchmark ledger contains malformed run_spec_json"
        ) from exc
    if not isinstance(decoded, dict):
        raise RunIntegrityError("benchmark ledger run_spec_json must contain an object")
    run_spec: Dict[str, Any] = decoded
    if (
        not run_fingerprint
        or fingerprint(run_spec_identity(run_spec)) != run_fingerprint
    ):
        raise RunIntegrityError(
            "benchmark ledger run specification differs from its stored fingerprint"
        )

    unfinished_statuses = {"pending", "running", "interrupted"}
    unfinished_n = sum(
        count
        for status, count in status_counts.items()
        if status in unfinished_statuses
    )
    execution_complete = (
        run_status in {"completed", "completed_with_failures"} and unfinished_n == 0
    )
    return pd.DataFrame(records), {
        "record_source": "benchmark.sqlite3",
        "source_attested": True,
        "verified_payloads": payloads,
        "verified_payload_count": len(payloads),
        "authoritative_record_count": len(records),
        "status_counts": status_counts,
        "task_count": sum(status_counts.values()),
        "unfinished_task_count": unfinished_n,
        "run_status": run_status or "unknown",
        "execution_complete": execution_complete,
        "run_fingerprint": run_fingerprint,
        "run_spec": run_spec,
    }


def load_report_records(
    input_dir: Path | str,
    *,
    record_loader: ReportRecordLoader | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load report rows through an explicit input trust boundary.

    The default loader requires ``benchmark.sqlite3`` and verifies every
    committed measured payload. A custom loader can be injected for testing by
    returning either a DataFrame or ``(DataFrame, metadata)``.
    """

    directory = Path(input_dir)
    loader = record_loader or _ledger_record_loader
    loaded = loader(directory)
    loader_metadata: Dict[str, Any] = {}
    if isinstance(loaded, tuple):
        frame, raw_metadata = loaded
        loader_metadata.update(dict(raw_metadata))
    else:
        frame = loaded

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Report record loader must return a pandas DataFrame")

    metadata: Dict[str, Any] = {
        "record_source": (
            str(directory / "benchmark.sqlite3")
            if record_loader is None
            else getattr(loader, "__name__", loader.__class__.__name__)
        ),
        "source_attested": False,
    }
    metadata.update(loader_metadata)

    run_spec = metadata.get("run_spec")
    if not isinstance(run_spec, dict) or not run_spec:
        run_spec_path = directory / "run_spec.json"
        run_spec = None
        if run_spec_path.is_file():
            try:
                run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid run specification: {run_spec_path}") from exc
            if not isinstance(run_spec, dict):
                raise ValueError(
                    f"Run specification must be a JSON object: {run_spec_path}"
                )
            metadata["run_spec"] = run_spec

    if isinstance(run_spec, dict) and run_spec:
        metadata["run_id"] = run_spec.get("run_id")
        metadata["dataset_kind"] = _normalise_dataset_kind(run_spec.get("dataset_kind"))
        metadata["dataset"] = run_spec.get("dataset")
    else:
        metadata["dataset_kind"] = _resolve_dataset_kind(frame)

    observation_path = directory / "host_observations.json"
    host_observations: list[dict[str, Any]] = []
    if observation_path.is_file():
        try:
            decoded = json.loads(observation_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid host observation history") from exc
        if not isinstance(decoded, list) or not all(
            isinstance(item, dict) for item in decoded
        ):
            raise ValueError("Host observation history must be a list of objects")
        host_observations = [dict(item) for item in decoded]
        run_fingerprint = metadata.get("run_fingerprint")
        if run_fingerprint and any(
            item.get("run_fingerprint") != run_fingerprint
            for item in host_observations
        ):
            raise ValueError("Host observation history has a run fingerprint mismatch")
    metadata["host_observations"] = host_observations
    metadata["power_mode_summary"] = summarize_task_power_records(
        frame.to_dict(orient="records")
    )

    out = frame.copy()
    out["_dataset_kind"] = metadata["dataset_kind"]
    return out, metadata


def load_timing_csv(path: Path | str) -> pd.DataFrame:
    """Load a summary CSV and retain only measured timing rows."""

    return timing_observations(pd.read_csv(path))


def _baseline_adapter_from_metadata(metadata: Mapping[str, Any]) -> str:
    run_spec = metadata.get("run_spec")
    if isinstance(run_spec, dict):
        guardrail = run_spec.get("guardrail")
        if isinstance(guardrail, dict):
            baseline = str(guardrail.get("baseline_adapter") or "").strip()
            if baseline:
                return baseline
    return BASELINE_ADAPTER


def _metadata_cell(value: Any) -> Any:
    """Return a stable, spreadsheet-safe representation of protocol metadata."""

    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _markdown_cell(value: Any) -> str:
    return (
        str(_metadata_cell(value))
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def _protocol_summary(run_spec: Any) -> pd.DataFrame:
    """Flatten publication-relevant RunSpec fields without local file paths."""

    columns = ["field", "value"]
    if not isinstance(run_spec, Mapping):
        return pd.DataFrame(columns=columns)

    fields = (
        "schema_version",
        "run_id",
        "dataset",
        "dataset_kind",
        "dataset_manifest_schema_version",
        "manifest_sha256",
        "dataset_hashes_verified",
        "dataset_values_inspected",
        "selected_case_ids",
        "adapters",
        "workloads",
        "repeats",
        "aggregation",
        "timing_observations",
        "capture_values",
        "timeout_seconds",
        "keep_going",
        "task_plan_sha256",
        "runtime_profiles_sha256",
        "benchmark_sources_sha256",
        "thread_policy",
        "initialization_policy",
        "guardrail",
        "memory_preflight",
        "endpoint_contract_id",
        "endpoint_contract_sha256",
    )
    rows = [
        {"field": field, "value": _metadata_cell(run_spec.get(field))}
        for field in fields
        if field in run_spec
    ]
    machine = run_spec.get("benchmark_machine")
    if isinstance(machine, Mapping):
        rows.extend(
            {
                "field": f"machine:{field}",
                "value": _metadata_cell(value),
            }
            for field, value in sorted(machine.items())
            if field != "hostname"
        )
    return pd.DataFrame(rows, columns=columns)


def _adapter_environment_summary(run_spec: Any) -> pd.DataFrame:
    """Expose reviewed release and numerical-runtime provenance per adapter."""

    columns = [
        "adapter",
        "distribution",
        "configured_release_version",
        "distribution_metadata_version",
        "python_version",
        "profile_python",
        "numpy_version",
        "installed_package_count",
        "installed_packages_sha256",
        "environment_freeze_sha256",
        "environment_metadata_sha256",
        "numpy_build_configuration",
    ]
    if not isinstance(run_spec, Mapping):
        return pd.DataFrame(columns=columns)
    environments = run_spec.get("adapter_environments")
    if not isinstance(environments, Mapping):
        return pd.DataFrame(columns=columns)

    configured_order = run_spec.get("adapters")
    adapter_names = (
        [str(value) for value in configured_order]
        if isinstance(configured_order, list)
        else sorted(str(value) for value in environments)
    )
    rows: list[Dict[str, Any]] = []
    for adapter in adapter_names:
        snapshot = environments.get(adapter)
        if not isinstance(snapshot, Mapping):
            continue
        packages = snapshot.get("packages")
        package_list = packages if isinstance(packages, list) else []
        package_payload = json.dumps(
            package_list,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        numpy_config = snapshot.get("numpy_config")
        if not isinstance(numpy_config, Mapping):
            numpy_config = {}
        rows.append(
            {
                "adapter": adapter,
                "distribution": snapshot.get("distribution"),
                "configured_release_version": snapshot.get(
                    "configured_release_version"
                ),
                "distribution_metadata_version": snapshot.get("distribution_version"),
                "python_version": snapshot.get("python_version"),
                "profile_python": snapshot.get("profile_python"),
                "numpy_version": numpy_config.get("version"),
                "installed_package_count": len(package_list),
                "installed_packages_sha256": hashlib.sha256(
                    package_payload
                ).hexdigest(),
                "environment_freeze_sha256": snapshot.get("environment_freeze_sha256"),
                "environment_metadata_sha256": snapshot.get(
                    "environment_metadata_sha256"
                ),
                "numpy_build_configuration": numpy_config.get("show_config"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def non_timing_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Return terminal outcomes excluded from performance estimates."""

    if "task_status" not in df.columns:
        raise ValueError("Report records must include authoritative task_status values")
    outcomes = df.copy()
    status = outcomes["task_status"].fillna("").astype(str).str.strip().str.lower()
    outcomes = outcomes.loc[status != MEASURED_TASK_STATUS].copy()
    outcomes["_report_status"] = status.loc[outcomes.index].replace("", "unknown")
    return with_report_workload(outcomes)


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required column(s): {', '.join(missing)}"
        )


def _canonical_triplet(
    value: object,
    *,
    integer: bool,
    field: str,
    context: str,
) -> str:
    """Return a stable JSON triplet for grouping ledger and CSV records."""

    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} has an invalid {field} triplet: {value!r}"
            ) from exc
    if isinstance(candidate, np.ndarray):
        candidate = candidate.tolist()
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
        raise ValueError(f"{context} requires {field} as a three-element sequence")

    normalized: list[int | float] = []
    for item in candidate:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{context} has a non-numeric {field} value")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{context} requires positive finite {field} values")
        if integer:
            if not numeric.is_integer():
                raise ValueError(f"{context} requires integer {field} values")
            normalized.append(int(numeric))
        else:
            normalized.append(numeric)
    return json.dumps(normalized, separators=(",", ":"))


def _normalise_real_world_geometry(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Validate and canonicalise the case geometry used for report grouping."""

    required = ["case_id", "modality", "shape", "spacing", "image_voxels"]
    _require_columns(frame, required, context)
    normalized = frame.copy()
    normalized["shape"] = normalized["shape"].map(
        lambda value: _canonical_triplet(
            value,
            integer=True,
            field="shape",
            context=context,
        )
    )
    normalized["spacing"] = normalized["spacing"].map(
        lambda value: _canonical_triplet(
            value,
            integer=False,
            field="spacing",
            context=context,
        )
    )
    case_ids = normalized["case_id"].fillna("").astype(str).str.strip()
    modalities = normalized["modality"].fillna("").astype(str).str.strip()
    if case_ids.eq("").any() or modalities.eq("").any():
        raise ValueError(f"{context} requires non-empty case_id and modality")
    normalized["case_id"] = case_ids
    normalized["modality"] = modalities
    if "subject_id" in normalized.columns:
        subject_ids = normalized["subject_id"].fillna("").astype(str).str.strip()
        if subject_ids.eq("").any():
            raise ValueError(f"{context} requires non-empty subject_id when present")
        normalized["subject_id"] = subject_ids

    voxels = pd.to_numeric(normalized["image_voxels"], errors="coerce")
    valid_voxels = np.isfinite(voxels) & (voxels > 0) & (voxels % 1 == 0)
    if not valid_voxels.all():
        raise ValueError(f"{context} requires positive integer image_voxels")
    normalized["image_voxels"] = voxels.astype(np.int64)
    shape_products = normalized["shape"].map(lambda value: math.prod(json.loads(value)))
    if not shape_products.eq(normalized["image_voxels"]).all():
        raise ValueError(f"{context} shape does not match image_voxels")

    geometry_columns = [
        "case_id",
        "modality",
        "shape",
        "spacing",
        "image_voxels",
    ]
    if "subject_id" in normalized.columns:
        geometry_columns.insert(1, "subject_id")
    geometry_variants = (
        normalized[geometry_columns]
        .drop_duplicates()
        .groupby("case_id", dropna=False)
        .size()
    )
    if (geometry_variants > 1).any():
        raise ValueError(
            f"{context} contains inconsistent geometry for the same case_id"
        )
    return normalized


def _performance_group_columns(
    observations: pd.DataFrame, dataset_kind: str
) -> list[str]:
    columns = ["_report_workload", "adapter"]
    if dataset_kind == "synthetic":
        _require_columns(observations, ["size"], "Synthetic timing report")
        if "subject_id" in observations.columns:
            columns.append("subject_id")
        columns.append("size")
        if "mask_id" in observations.columns:
            columns.append("mask_id")
    elif dataset_kind == "real_world":
        _require_columns(
            observations,
            ["case_id", "modality", "shape", "spacing", "image_voxels"],
            "Real-world timing report",
        )
        columns.append("case_id")
        if "subject_id" in observations.columns:
            columns.append("subject_id")
        columns.extend(["modality", "shape", "spacing", "image_voxels"])
    elif "size" in observations.columns:
        columns.append("size")
    elif "case_id" in observations.columns:
        columns.append("case_id")
    return columns


def aggregate_timing_observations(
    df: pd.DataFrame, *, dataset_kind: str | None = None
) -> pd.DataFrame:
    """Summarize absolute runtime using measured observations only.

    Synthetic rows retain profile/subject, cube edge length, and mask.
    Real-world rows retain modality and the true image voxel count; cube edge
    length is used only for synthetic scaling cases.
    """

    observations = with_report_workload(timing_observations(df))
    if observations.empty:
        return pd.DataFrame()
    _require_columns(observations, ["adapter"], "Timing report")

    kind = _resolve_dataset_kind(observations, dataset_kind)
    if kind == "real_world":
        observations = _normalise_real_world_geometry(
            observations, "Real-world timing report"
        )
    group_columns = _performance_group_columns(observations, kind)

    memory_columns = (
        "peak_rss_bytes",
        "worker_ready_rss_bytes",
        "calculation_peak_rss_bytes",
        "incremental_calculation_peak_rss_bytes",
    )
    for column in memory_columns:
        if column not in observations.columns:
            observations[column] = np.nan
        observations[column] = pd.to_numeric(observations[column], errors="coerce")

    grouped = (
        observations.groupby(group_columns, dropna=False)
        .agg(
            grouping_dimension=("_report_grouping_dimension", "first"),
            duration_steady_sec=("duration_sec", "median"),
            duration_q1_sec=("duration_sec", lambda values: values.quantile(0.25)),
            duration_q3_sec=("duration_sec", lambda values: values.quantile(0.75)),
            duration_mean_sec=("duration_sec", "mean"),
            duration_std_sec=("duration_sec", "std"),
            peak_rss_mb=(
                "peak_rss_bytes",
                lambda values: (
                    float(values.max()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_rss_mb=(
                "peak_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            calculation_peak_rss_mb=(
                "calculation_peak_rss_bytes",
                lambda values: (
                    float(values.max()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_calculation_peak_rss_mb=(
                "calculation_peak_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_worker_ready_rss_mb=(
                "worker_ready_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_incremental_calculation_rss_mb=(
                "incremental_calculation_peak_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            timing_observations=("duration_sec", "count"),
        )
        .reset_index()
        .rename(columns={"_report_workload": "workload"})
    )
    grouped.insert(0, "dataset_kind", kind)
    return grouped


def feature_workload_contract(records: pd.DataFrame) -> pd.DataFrame:
    """Summarise planned and observed workload-level feature denominators.

    Runtime is deliberately not divided by either denominator.  The table
    instead exposes the exact frozen native-output count beside task coverage
    so readers can interpret unequal package workloads without inventing a
    linear per-feature cost model.
    """

    columns = [
        "adapter",
        "workload",
        "expected_native_outputs",
        "observed_native_outputs_min",
        "observed_native_outputs_max",
        "planned_tasks",
        "measured_tasks",
        "terminal_tasks",
        "measured_fraction_of_planned",
        "runtime_normalization",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    frame = with_report_workload(records.copy())
    _require_columns(
        frame,
        ["adapter", "_report_workload", "task_status"],
        "Feature workload",
    )
    for field in ("expected_feature_count", "feature_count"):
        if field not in frame.columns:
            frame[field] = np.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (adapter, workload), group in frame.groupby(
        ["adapter", "_report_workload"], dropna=False, sort=True
    ):
        expected_values = group["expected_feature_count"].dropna().unique()
        if len(expected_values) > 1:
            raise ValueError(
                f"Feature workload has inconsistent frozen counts for "
                f"{adapter}/{workload}"
            )
        measured = group.loc[group["task_status"] == MEASURED_TASK_STATUS]
        observed = measured["feature_count"].dropna()
        terminal = group.loc[
            ~group["task_status"].isin(["pending", "running", "interrupted"])
        ]
        planned_n = int(len(group))
        measured_n = int(len(measured))
        rows.append(
            {
                "adapter": adapter,
                "workload": workload,
                "expected_native_outputs": (
                    int(expected_values[0]) if len(expected_values) else np.nan
                ),
                "observed_native_outputs_min": (
                    int(observed.min()) if not observed.empty else np.nan
                ),
                "observed_native_outputs_max": (
                    int(observed.max()) if not observed.empty else np.nan
                ),
                "planned_tasks": planned_n,
                "measured_tasks": measured_n,
                "terminal_tasks": int(len(terminal)),
                "measured_fraction_of_planned": (
                    measured_n / planned_n if planned_n else np.nan
                ),
                "runtime_normalization": "none",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _duplicate_key_message(frame: pd.DataFrame, keys: list[str], label: str) -> str:
    duplicate = frame.loc[frame.duplicated(keys, keep=False), keys].head(5)
    examples = duplicate.astype(str).agg(" | ".join, axis=1).tolist()
    return (
        f"{label} has duplicate measured rows for exact comparison key "
        f"{tuple(keys)}; examples: {examples}"
    )


def matched_runtime_observations(
    df: pd.DataFrame,
    *,
    baseline_adapter: str = BASELINE_ADAPTER,
) -> pd.DataFrame:
    """Create exact candidate/baseline runtime pairs.

    Rows are matched only on ``(case_id, workload, repeat)`` after filtering
    to measured observations. The ratio is candidate duration
    divided by the configured baseline duration; values above one mean the
    baseline ran faster. Unmatched or censored rows never enter the result.
    Duplicate comparison identities are rejected instead of being multiplied
    by a many-to-many merge.
    """

    observations = with_report_workload(timing_observations(df))
    match_keys = ["case_id", "_report_workload", "repeat"]
    required = [*match_keys, "adapter", "duration_sec"]
    _require_columns(observations, required, "Matched runtime report")

    baseline_name = baseline_adapter.strip().lower()
    adapter_key = observations["adapter"].fillna("").astype(str).str.strip().str.lower()
    baseline = observations.loc[adapter_key == baseline_name].copy()
    candidates = observations.loc[adapter_key != baseline_name].copy()

    if baseline.duplicated(match_keys, keep=False).any():
        raise ValueError(
            _duplicate_key_message(baseline, match_keys, "Baseline adapter")
        )
    candidate_keys = [*match_keys, "adapter"]
    if candidates.duplicated(candidate_keys, keep=False).any():
        raise ValueError(
            _duplicate_key_message(candidates, candidate_keys, "Candidate adapter")
        )

    output_columns = [
        "case_id",
        "subject_id",
        "workload",
        "repeat",
        "adapter",
        "candidate_duration_sec",
        "baseline_duration_sec",
        "runtime_ratio",
        "baseline_adapter",
        "dataset",
        "modality",
        "size",
        "mask_id",
        "mask_label",
        "shape",
        "spacing",
        "image_voxels",
        "mask_voxels",
        "_dataset_kind",
        "candidate_task_id",
        "baseline_task_id",
    ]
    if baseline.empty or candidates.empty:
        return pd.DataFrame(columns=output_columns)

    baseline_columns = [*match_keys, "duration_sec"]
    if "task_id" in baseline.columns:
        baseline_columns.append("task_id")
    baseline_for_join = baseline[baseline_columns].rename(
        columns={
            "duration_sec": "baseline_duration_sec",
            "task_id": "baseline_task_id",
        }
    )

    candidate_frame = candidates.copy()
    candidate_frame = candidate_frame.rename(
        columns={
            "duration_sec": "candidate_duration_sec",
            "task_id": "candidate_task_id",
        }
    )
    matched = candidate_frame.merge(
        baseline_for_join,
        on=match_keys,
        how="inner",
        validate="many_to_one",
    )
    matched["runtime_ratio"] = (
        matched["candidate_duration_sec"] / matched["baseline_duration_sec"]
    )
    matched["baseline_adapter"] = baseline_adapter
    # ``workload`` may already exist in modern summaries. Overwrite it with the
    # normalized report key instead of creating a duplicate column name.
    matched["workload"] = matched["_report_workload"]

    for column in output_columns:
        if column not in matched.columns:
            matched[column] = np.nan
    return (
        matched[output_columns]
        .sort_values(
            ["workload", "adapter", "case_id", "repeat"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def matched_runtime_summary(
    matched: pd.DataFrame, *, dataset_kind: str | None = None
) -> pd.DataFrame:
    """Summarize exact runtime pairs and expose the matched sample size."""

    if matched.empty:
        kind = _normalise_dataset_kind(dataset_kind)
        dimensions = ["size", "mask_id"] if kind == "synthetic" else []
        if kind == "synthetic" and "subject_id" in matched.columns:
            dimensions.insert(0, "subject_id")
        if kind == "real_world":
            dimensions = [
                "case_id",
                "modality",
                "shape",
                "spacing",
                "image_voxels",
            ]
            if "subject_id" in matched.columns:
                dimensions.insert(1, "subject_id")
        return pd.DataFrame(
            columns=[
                "dataset_kind",
                "workload",
                "adapter",
                "baseline_adapter",
                *dimensions,
                "matched_n",
                "runtime_ratio_median",
                "runtime_ratio_q1",
                "runtime_ratio_q3",
                "candidate_duration_median_sec",
                "baseline_duration_median_sec",
                "ratio_definition",
            ]
        )
    required = [
        "workload",
        "adapter",
        "candidate_duration_sec",
        "baseline_duration_sec",
        "runtime_ratio",
        "baseline_adapter",
    ]
    _require_columns(matched, required, "Matched runtime summary")
    kind = _resolve_dataset_kind(matched, dataset_kind)
    if kind == "real_world":
        matched = _normalise_real_world_geometry(
            matched, "Real-world matched runtime summary"
        )

    group_columns = ["workload", "adapter", "baseline_adapter"]
    if kind == "synthetic":
        _require_columns(matched, ["size"], "Synthetic matched runtime summary")
        if "subject_id" in matched.columns:
            group_columns.append("subject_id")
        group_columns.append("size")
        if "mask_id" in matched.columns:
            group_columns.append("mask_id")
    elif kind == "real_world":
        _require_columns(
            matched,
            ["case_id", "modality", "shape", "spacing", "image_voxels"],
            "Real-world matched runtime summary",
        )
        group_columns.append("case_id")
        if "subject_id" in matched.columns:
            group_columns.append("subject_id")
        group_columns.extend(["modality", "shape", "spacing", "image_voxels"])
    elif "size" in matched.columns:
        group_columns.append("size")

    summary = (
        matched.groupby(group_columns, dropna=False)
        .agg(
            matched_n=("runtime_ratio", "count"),
            runtime_ratio_median=("runtime_ratio", "median"),
            runtime_ratio_q1=(
                "runtime_ratio",
                lambda values: values.quantile(0.25),
            ),
            runtime_ratio_q3=(
                "runtime_ratio",
                lambda values: values.quantile(0.75),
            ),
            candidate_duration_median_sec=(
                "candidate_duration_sec",
                "median",
            ),
            baseline_duration_median_sec=(
                "baseline_duration_sec",
                "median",
            ),
        )
        .reset_index()
    )
    summary.insert(0, "dataset_kind", kind)
    summary["ratio_definition"] = "candidate_duration / baseline_duration"
    return summary


def apply_scientific_style(ax: plt.Axes) -> None:
    """Apply high-contrast, publication-oriented axis styling."""

    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#D7D7D7", linestyle=":", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.tick_params(colors="#222222", labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)


def _adapter_style(adapter: object) -> Dict[str, Any]:
    name = str(adapter)
    if name in ADAPTER_STYLES:
        return ADAPTER_STYLES[name]
    fallback_markers = ("X", "v", "<", ">")
    index = sum(ord(char) for char in name) % len(fallback_markers)
    return {
        "color": "#444444",
        "marker": fallback_markers[index],
        "linestyle": "--",
        "label": name,
    }


def _panel_spec(dataset_kind: str, data: pd.DataFrame) -> tuple[str | None, str, str]:
    if dataset_kind == "synthetic":
        return (
            "_synthetic_stratum"
            if "_synthetic_stratum" in data.columns
            else "mask_id"
            if "mask_id" in data.columns
            else None,
            "size",
            "Cube edge length (voxels)",
        )
    if dataset_kind == "real_world":
        return "modality", "image_voxels", "Image voxels"
    if "size" in data.columns:
        return None, "size", "Reported size index"
    if "image_voxels" in data.columns:
        return None, "image_voxels", "Image voxels"
    raise ValueError("Performance plot has no valid horizontal-axis field")


def _plot_panels(
    data: pd.DataFrame,
    *,
    dataset_kind: str,
    value_column: str,
    q1_column: str,
    q3_column: str,
    ylabel: str,
    title: str,
    ratio_reference: bool,
) -> plt.Figure:
    data = data.copy()
    if dataset_kind == "synthetic" and "subject_id" in data.columns:
        profiles = data["subject_id"].fillna("unknown").astype(str)
        if "mask_id" in data.columns:
            masks = data["mask_id"].fillna("unknown").astype(str)
            data["_synthetic_stratum"] = (
                "Profile " + profiles + " / mask " + masks
            )
        else:
            data["_synthetic_stratum"] = "Profile " + profiles
    panel_column, x_column, xlabel = _panel_spec(dataset_kind, data)
    if panel_column:
        panel_values = list(data[panel_column].drop_duplicates())
    else:
        panel_values = [None]

    column_count = min(3, max(1, len(panel_values)))
    row_count = int(math.ceil(len(panel_values) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.3 * column_count, 4.3 * row_count),
        dpi=200,
        squeeze=False,
    )
    all_handles: Dict[str, Any] = {}

    for panel_index, panel_value in enumerate(panel_values):
        ax = axes.flat[panel_index]
        panel = (
            data.loc[data[panel_column] == panel_value].copy()
            if panel_column
            else data.copy()
        )
        apply_scientific_style(ax)

        for adapter in sorted(panel["adapter"].astype(str).unique()):
            adapter_data = panel.loc[panel["adapter"].astype(str) == adapter].copy()
            adapter_data[x_column] = pd.to_numeric(
                adapter_data[x_column], errors="coerce"
            )
            adapter_data = adapter_data.dropna(
                subset=[x_column, value_column]
            ).sort_values(x_column)
            if adapter_data.empty:
                continue
            style = _adapter_style(adapter)
            line = ax.plot(
                adapter_data[x_column],
                adapter_data[value_column],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=6,
                label=style["label"],
            )[0]
            all_handles[style["label"]] = line

            lower = pd.to_numeric(adapter_data[q1_column], errors="coerce")
            upper = pd.to_numeric(adapter_data[q3_column], errors="coerce")
            if lower.notna().any() and upper.notna().any():
                ax.fill_between(
                    adapter_data[x_column].to_numpy(dtype=float),
                    lower.to_numpy(dtype=float),
                    upper.to_numpy(dtype=float),
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0,
                )

        if ratio_reference:
            ax.axhline(
                1.0,
                color="#333333",
                linestyle=(0, (3, 2)),
                linewidth=1.2,
                label="Equal runtime",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        positive_values = pd.to_numeric(panel[value_column], errors="coerce").dropna()
        if not positive_values.empty and (positive_values > 0).all():
            ax.set_yscale("log")
        if dataset_kind == "real_world":
            positive_x = pd.to_numeric(panel[x_column], errors="coerce").dropna()
            if not positive_x.empty and (positive_x > 0).all():
                ax.set_xscale("log")
        if panel_column:
            ax.set_title(
                f"{str(panel_column).replace('_', ' ').title()}: {panel_value}"
            )

    for unused_index in range(len(panel_values), len(axes.flat)):
        axes.flat[unused_index].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#111111")
    if all_handles:
        fig.legend(
            list(all_handles.values()),
            list(all_handles.keys()),
            loc="lower center",
            ncol=min(5, len(all_handles)),
            frameon=True,
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig


def generate_plots(
    df: pd.DataFrame,
    report_dir: Path,
    *,
    dataset_kind: str | None = None,
    baseline_adapter: str = BASELINE_ADAPTER,
    matched: pd.DataFrame | None = None,
) -> list[Dict[str, str]]:
    """Generate a compact runtime figure set.

    Absolute runtime includes every measured observation. Relative runtime is
    calculated only from exact candidate/baseline pairs.
    """

    observations = timing_observations(df)
    if observations.empty:
        return []
    kind = _resolve_dataset_kind(observations, dataset_kind)
    absolute = aggregate_timing_observations(observations, dataset_kind=kind)
    exact_pairs = (
        matched_runtime_observations(observations, baseline_adapter=baseline_adapter)
        if matched is None
        else matched.copy()
    )
    relative = matched_runtime_summary(exact_pairs, dataset_kind=kind)
    report_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Dict[str, str]] = []
    for workload in sorted(absolute["workload"].astype(str).unique()):
        workload_data = absolute.loc[absolute["workload"].astype(str) == workload]
        suffix = _safe_report_suffix(workload)
        figure = _plot_panels(
            workload_data,
            dataset_kind=kind,
            value_column="duration_steady_sec",
            q1_column="duration_q1_sec",
            q3_column="duration_q3_sec",
            ylabel="Median runtime (seconds)",
            title=f"Measured runtime — {workload}",
            ratio_reference=False,
        )
        stem = f"plot_runtime_{suffix}"
        for extension in ("pdf", "svg"):
            _atomic_save_figure(
                figure,
                report_dir / f"{stem}.{extension}",
            )
        plt.close(figure)
        figures.append(
            {
                "artifact": f"{stem}.pdf",
                "format": "PDF and SVG",
                "alt_text": (
                    f"Median measured runtime by adapter for workload {workload}, "
                    f"stratified using {kind.replace('_', ' ')} dataset dimensions; "
                    "shaded intervals show the interquartile range."
                ),
            }
        )

        if relative.empty:
            continue
        ratio_data = relative.loc[relative["workload"].astype(str) == workload]
        if ratio_data.empty:
            continue
        ratio_figure = _plot_panels(
            ratio_data,
            dataset_kind=kind,
            value_column="runtime_ratio_median",
            q1_column="runtime_ratio_q1",
            q3_column="runtime_ratio_q3",
            ylabel=f"Runtime ratio (candidate / {baseline_adapter})",
            title=f"Matched runtime ratio — {workload}",
            ratio_reference=True,
        )
        ratio_stem = f"plot_matched_runtime_ratio_{suffix}"
        for extension in ("pdf", "svg"):
            _atomic_save_figure(
                ratio_figure,
                report_dir / f"{ratio_stem}.{extension}",
            )
        plt.close(ratio_figure)
        figures.append(
            {
                "artifact": f"{ratio_stem}.pdf",
                "format": "PDF and SVG",
                "alt_text": (
                    f"Median candidate-to-{baseline_adapter} runtime ratio for "
                    f"workload {workload} using exact case, workload, and repeat "
                    "matches; the reference line denotes equal runtime."
                ),
            }
        )
    return figures


def generate_excel_report(
    timing_summary: pd.DataFrame,
    matched: pd.DataFrame,
    matched_summary: pd.DataFrame,
    feature_contract: pd.DataFrame,
    outcomes: pd.DataFrame,
    qc_issues: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    report_dir: Path,
) -> Path:
    """Write the minimal auditable performance workbook."""

    path = report_dir / "academic_performance_report.xlsx"
    metadata_rows = [
        {"field": "run_id", "value": run_metadata.get("run_id")},
        {"field": "run_status", "value": run_metadata.get("run_status")},
        {
            "field": "execution_complete",
            "value": bool(run_metadata.get("execution_complete")),
        },
        {"field": "task_count", "value": run_metadata.get("task_count")},
        {
            "field": "unfinished_task_count",
            "value": run_metadata.get("unfinished_task_count"),
        },
        {
            "field": "verified_payload_count",
            "value": run_metadata.get("verified_payload_count"),
        },
        {"field": "record_source", "value": run_metadata.get("record_source")},
        {
            "field": "source_attested",
            "value": bool(run_metadata.get("source_attested")),
        },
        {
            "field": "publication_attested",
            "value": bool(run_metadata.get("publication_attested")),
        },
        {
            "field": "run_fingerprint",
            "value": run_metadata.get("run_fingerprint"),
        },
        {
            "field": "power_mode_classification",
            "value": dict(run_metadata.get("power_mode_summary") or {}).get(
                "classification"
            ),
        },
        {
            "field": "power_mode_tags",
            "value": json.dumps(
                dict(run_metadata.get("power_mode_summary") or {}).get("tags", []),
                separators=(",", ":"),
            ),
        },
    ]
    report_generator = run_metadata.get("report_generator")
    if isinstance(report_generator, Mapping):
        metadata_rows.extend(
            [
                {
                    "field": "report_generator:provenance_sha256",
                    "value": report_generator.get("provenance_sha256"),
                },
                {
                    "field": "report_generator:source_tree_sha256",
                    "value": report_generator.get("source_tree_sha256"),
                },
                {
                    "field": "report_generator:python_version",
                    "value": report_generator.get("python_version"),
                },
            ]
        )
        dependencies = report_generator.get("dependencies")
        if isinstance(dependencies, Mapping):
            metadata_rows.extend(
                {
                    "field": f"report_generator:dependency:{name}",
                    "value": version,
                }
                for name, version in sorted(dependencies.items())
            )
    for status, count in sorted(dict(run_metadata.get("status_counts") or {}).items()):
        metadata_rows.append({"field": f"status_count:{status}", "value": count})
    protocol = _protocol_summary(run_metadata.get("run_spec"))
    environments = _adapter_environment_summary(run_metadata.get("run_spec"))
    sheets = {
        "Run metadata": pd.DataFrame(metadata_rows),
        "Protocol": protocol,
        "Adapter environments": environments,
        "Measured runtime": timing_summary,
        "Feature workload": feature_contract,
        "Matched pairs": matched,
        "Matched summary": matched_summary,
        "Excluded outcomes": outcomes,
        "QC issues": qc_issues,
    }
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [
                    "" if cell.value is None else str(cell.value)
                    for cell in column_cells[:200]
                ]
                width = min(48, max(11, max(map(len, values), default=0) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    atomic_write_bytes(path, buffer.getvalue())
    return path


def _denominator_names(payload: Mapping[str, Any], name: str) -> Optional[set[str]]:
    containers = [payload.get("feature_denominators")]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        containers.append(metadata.get("feature_denominators"))
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(name)
        if isinstance(value, list):
            return {str(item) for item in value}
        if isinstance(value, dict):
            return {str(item) for item in value}
    return None


def _summarize_adapter_payloads(
    adapter: str, payloads: list[Mapping[str, Any]]
) -> Dict[str, Any]:
    attempted: set[str] = set()
    finite: set[str] = set()
    supported: set[str] = set()
    referenced: set[str] = set()
    passing: set[str] = set()
    finite_observed = False
    supported_observed = False
    referenced_observed = False
    passing_observed = False

    for payload in payloads:
        benchmark = payload.get("benchmark")
        if isinstance(benchmark, dict) and benchmark.get("status") not in {
            None,
            MEASURED_TASK_STATUS,
        }:
            continue
        features = payload.get("features")
        if isinstance(features, dict):
            for feature in features.get("all", []):
                attempted.add(str(feature))

        values_container = payload.get("values")
        values = (
            values_container.get("all") if isinstance(values_container, dict) else None
        )
        if isinstance(values, dict):
            finite_observed = True
            for feature_name, value in values.items():
                try:
                    if math.isfinite(float(value)):
                        finite.add(str(feature_name))
                except (TypeError, ValueError):
                    continue

        for name, target in (
            ("supported", supported),
            ("referenced", referenced),
            ("passing", passing),
        ):
            names = _denominator_names(payload, name)
            if names is None:
                continue
            target.update(names)
            if name == "supported":
                supported_observed = True
            elif name == "referenced":
                referenced_observed = True
            else:
                passing_observed = True

    from bench.ibsi_mapping import documented_semantic_aliases

    ibsi_codes: set[str] = set()
    for feature_name in attempted:
        code, status = classify_feature(adapter, feature_name)
        if status == "mapped" and code:
            ibsi_codes.add(code)
        ibsi_codes.update(documented_semantic_aliases(adapter, feature_name))

    from bench.ibsi_families import CODE_TO_FAMILY

    family_counts: Dict[str, int] = {}
    for code in ibsi_codes:
        family = CODE_TO_FAMILY.get(code)
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "denominators": {
            "supported": len(supported) if supported_observed else None,
            "attempted": len(attempted),
            "finite": len(finite) if finite_observed else None,
            "referenced": len(referenced) if referenced_observed else None,
            "passing": len(passing) if passing_observed else None,
        },
        "uniquely_mapped_ibsi_from_attempted": len(ibsi_codes),
        "uniquely_mapped_features_by_family": family_counts,
    }


def summarize_features_from_payloads(
    payloads: list[Mapping[str, Any]], adapters: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Summarize feature denominators from ledger-verified payload objects."""

    by_adapter: Dict[str, list[Mapping[str, Any]]] = {
        adapter: [] for adapter in adapters
    }
    for payload in payloads:
        adapter = str(payload.get("adapter") or "").strip()
        if adapter in by_adapter:
            by_adapter[adapter].append(payload)
    return {
        adapter: _summarize_adapter_payloads(adapter, by_adapter[adapter])
        for adapter in adapters
    }


def generate_feature_coverage_plot(
    coverage: Dict[str, Dict[str, Any]], report_dir: Path
) -> Optional[Path]:
    """Plot explicitly observed feature denominators with redundant hatching."""

    adapters = list(coverage)
    if not adapters:
        return None

    fields = (
        ("supported", "Supported", "#777777", "//"),
        ("attempted", "Attempted", "#A6CEE3", "\\\\"),
        ("finite", "Finite", "#1F78B4", "xx"),
        ("referenced", "Referenced", "#FDBF6F", ".."),
        ("passing", "Passing", "#33A02C", "++"),
    )
    available = [
        item
        for item in fields
        if any(
            coverage[adapter]["denominators"].get(item[0]) is not None
            for adapter in adapters
        )
    ]
    if not available:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    apply_scientific_style(ax)
    x = np.arange(len(adapters))
    width = min(0.72 / len(available), 0.3)
    for index, (key, label, color, hatch) in enumerate(available):
        offset = (index - (len(available) - 1) / 2.0) * width
        values = [
            coverage[adapter]["denominators"].get(key, np.nan)
            if coverage[adapter]["denominators"].get(key) is not None
            else np.nan
            for adapter in adapters
        ]
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#111111",
            linewidth=0.8,
            hatch=hatch,
        )
    ax.set_ylabel("Feature count")
    ax.set_title("Observed feature denominators")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_adapter_style(adapter)["label"] for adapter in adapters],
        rotation=15,
        ha="right",
    )
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    path = report_dir / "plot_feature_coverage.pdf"
    _atomic_save_figure(fig, path)
    _atomic_save_figure(
        fig,
        report_dir / "plot_feature_coverage.svg",
    )
    plt.close(fig)
    return path


def _diagnostic_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "subject_id",
                "adapter",
                "workload",
                "repeat",
                "status",
                "modality",
                "shape",
                "spacing",
                "image_voxels",
                "size",
                "mask_id",
                "reason",
                "censor_lower_bound_sec",
            ]
        )

    result = pd.DataFrame(index=outcomes.index)
    for column in (
        "case_id",
        "subject_id",
        "adapter",
        "repeat",
        "modality",
        "shape",
        "spacing",
        "image_voxels",
        "size",
        "mask_id",
        "censor_lower_bound_sec",
    ):
        result[column] = outcomes[column] if column in outcomes.columns else np.nan
    result["workload"] = outcomes["_report_workload"]
    result["status"] = outcomes["_report_status"]

    result["reason"] = (
        "Diagnostic detail retained in the local checksum-attested ledger"
    )
    return result[
        [
            "case_id",
            "subject_id",
            "adapter",
            "workload",
            "repeat",
            "status",
            "modality",
            "shape",
            "spacing",
            "image_voxels",
            "size",
            "mask_id",
            "reason",
            "censor_lower_bound_sec",
        ]
    ]


def _write_markdown_summary(
    path: Path,
    *,
    input_dir: Path,
    metadata: Mapping[str, Any],
    measured_n: int,
    matched: pd.DataFrame,
    matched_summary: pd.DataFrame,
    feature_contract: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: Dict[str, Dict[str, Any]],
    figures: list[Dict[str, str]],
) -> None:
    kind = _normalise_dataset_kind(metadata.get("dataset_kind"))
    run_id = metadata.get("run_id") or input_dir.name
    baseline_adapter = str(metadata.get("baseline_adapter") or BASELINE_ADAPTER)
    lines = [
        "# Benchmark performance report",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset kind: `{kind}`",
        f"- Run status: `{metadata.get('run_status', 'unknown')}`",
        f"- Execution complete: "
        f"{'yes' if metadata.get('execution_complete') else 'no'}",
        f"- Planned tasks: {metadata.get('task_count', 'unknown')}",
        f"- Unfinished tasks: {metadata.get('unfinished_task_count', 'unknown')}",
        f"- Measured timing observations: {measured_n}",
        f"- Checksum-verified measured payloads: "
        f"{metadata.get('verified_payload_count', 'unknown')}",
        f"- Comparison baseline adapter: `{baseline_adapter}`",
        f"- Power-mode provenance: "
        f"`{dict(metadata.get('power_mode_summary') or {}).get('classification', 'unavailable')}` "
        f"{dict(metadata.get('power_mode_summary') or {}).get('tags', [])}",
        f"- QC issues: "
        f"{dict(metadata.get('qc_summary') or {}).get('issue_count_total', 'unknown')}",
        f"- Exact candidate/baseline pairs: {len(matched)}",
        f"- Input record source: `{metadata.get('record_source', 'unknown')}`",
        f"- Report generator provenance: "
        f"`{dict(metadata.get('report_generator') or {}).get('provenance_sha256', 'unknown')}`",
        f"- Source attested against ledger payload hashes: "
        f"{'yes' if metadata.get('source_attested') else 'no'}",
        f"- Publication-attested complete report: "
        f"{'yes' if metadata.get('publication_attested') else 'no'}",
        "",
    ]
    status_counts = dict(metadata.get("status_counts") or {})
    if status_counts:
        lines.extend(
            [
                "## Authoritative task status counts",
                "",
                "| Task status | n |",
                "|---|---:|",
            ]
        )
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} |")
        lines.append("")
    if not metadata.get("execution_complete"):
        lines.extend(
            [
                "> **Incomplete run:** this report is an explicitly partial "
                "snapshot and must not be described as a completed benchmark.",
                "",
            ]
        )

    protocol = _protocol_summary(metadata.get("run_spec"))
    if not protocol.empty:
        lines.extend(
            [
                "## Executed protocol",
                "",
                "| Field | Value |",
                "|---|---|",
            ]
        )
        for row in protocol.itertuples(index=False):
            lines.append(
                f"| {_markdown_cell(row.field)} | {_markdown_cell(row.value)} |"
            )
        lines.append("")

    environments = _adapter_environment_summary(metadata.get("run_spec"))
    if not environments.empty:
        lines.extend(
            [
                "## Adapter environments",
                "",
                "| Adapter | Distribution | Reviewed release | Installed metadata | Python | NumPy | Environment record SHA-256 |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in environments.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        row.adapter,
                        row.distribution,
                        row.configured_release_version,
                        row.distribution_metadata_version,
                        row.python_version,
                        row.numpy_version,
                        row.environment_metadata_sha256,
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "The separate `adapter_environment_summary.csv` retains the "
                "complete NumPy/BLAS build configuration and a digest of the "
                "installed package set, plus the exact recorded `pip freeze` "
                "digest. Reviewed release and installed "
                "distribution metadata are intentionally separate; this is "
                "material for PyRadiomics 3.1.0, whose official tag exposes "
                "upstream metadata 3.0.1a1.",
                "",
            ]
        )

    lines.extend(
        [
            "## Runtime comparison contract",
            "",
            f"Relative runtime is candidate duration divided by {baseline_adapter} "
            "duration. Values above 1 mean the baseline was faster. Each ratio uses "
            "the same `case_id`, workload, and process repeat. "
            "Unmatched, timed-out, "
            "failed, unsupported, skipped, and timed-out rows are excluded from ratios.",
            "",
            "The six timed workloads preserve comparable scaling classes and "
            "native shared calculation paths: morphology excluding spatial "
            "autocorrelation; Moran's I plus Geary's C; local-intensity peaks; "
            "first-order intensity; all histogram/texture families; and IVH. IVH "
            "is a separate timed component because it uses a different frozen "
            "representation. The six workloads remain separate in "
            "every report; their runtimes are not summed after measurement. These "
            "timings represent each package's reviewed "
            "maximum native surface; they are package "
            "workloads, not a common-feature microbenchmark. Observed feature "
            "denominators are reported separately and must accompany speed claims.",
            "",
            "Adapter timing policies are explicit: package import, NIfTI loading, "
            "mask/resegmentation preparation, stored-representation validation, "
            "discretisation, warm-up, and output normalisation are outside the "
            "timer. Matrix/mesh/neighbourhood construction and feature arithmetic "
            "are inside. Pictologics JIT initialization is outside the measured "
            "region, and Z-Rad local-intensity cache state is cleared before and "
            "after every calculation.",
            "",
            "Synthetic and real-world performance runs benchmark radiomic feature "
            "extraction only. IBSI 2 filter results establish conformity, not filter "
            "runtime; no filter-performance claim is made.",
            "",
        ]
    )
    if kind == "synthetic":
        lines.extend(
            [
                "Synthetic summaries retain profile/subject, cube edge length, "
                "and mask identity.",
                "",
            ]
        )
    elif kind == "real_world":
        lines.extend(
            [
                "Real-world summaries retain modality and the true image voxel "
                "count. No cube geometry is inferred from a maximum dimension.",
                "",
            ]
        )

    lines.extend(
        [
            "## Output tables",
            "",
            "- [Measured runtime summary](adapter_performance_table.csv)",
            "- [Public task observations and raw samples](task_observations.csv)",
            "- [Frozen and observed feature workloads](feature_workload_contract.csv)",
            "- [Exact matched runtime pairs](matched_runtime_pairs.csv)",
            "- [Matched runtime summary with matched n](matched_runtime_summary.csv)",
            "- [Diagnostic-only excluded outcomes](excluded_outcomes.csv)",
            "- [QC issue identities](qc_issues.csv)",
            "- [QC summary counts](qc_summary.json)",
            "- [Executed protocol and machine](protocol_summary.csv)",
            "- [Adapter software and numerical runtimes](adapter_environment_summary.csv)",
            "- [Auditable workbook](academic_performance_report.xlsx)",
            "- [Run-bound artifact checksums](report_manifest.json)",
            "",
        ]
    )
    if not matched_summary.empty:
        lines.extend(
            [
                "The matched summary reports medians and interquartile ranges. "
                "`matched_n` is the number of exact pairs in each displayed cell.",
                "",
            ]
        )
    if not feature_contract.empty:
        lines.extend(
            [
                "## Feature workload denominators",
                "",
                "Runtime is not divided by feature count. Matrix construction "
                "and feature evaluation share costs, so a per-feature quotient "
                "would assert a linear cost model that the benchmark does not "
                "establish. The separate workload table reports frozen native "
                "outputs and measured/planned task coverage beside runtime.",
                "",
            ]
        )

    lines.extend(["## Figures", ""])
    if figures:
        for figure in figures:
            lines.append(
                f"- [{figure['artifact']}]({figure['artifact']}): {figure['alt_text']}"
            )
    else:
        lines.append("- No runtime figure could be generated.")
    if coverage:
        lines.append(
            "- [Observed feature denominators](plot_feature_coverage.pdf): "
            "Feature counts reported by measured adapter payloads, separated "
            "into supported, attempted, finite, referenced, and passing sets."
        )
    lines.append("")

    lines.extend(["## Excluded outcomes", ""])
    if diagnostics.empty:
        lines.append("No non-measured terminal outcomes were recorded.")
    else:
        counts = (
            diagnostics.groupby(["status", "adapter"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        lines.extend(
            [
                "| Status | Adapter | n |",
                "|---|---|---:|",
            ]
        )
        for row in counts.itertuples(index=False):
            lines.append(f"| {row.status} | {row.adapter} | {row.n} |")
    lines.append("")

    if not metadata.get("source_attested"):
        lines.extend(
            [
                "## Input integrity note",
                "",
                "This report used a custom unattested source and is not "
                "publication-attested. A ledger-backed run is required to verify "
                "payload hashes and task completeness.",
                "",
            ]
        )
    elif not metadata.get("publication_attested"):
        lines.extend(
            [
                "## Publication integrity note",
                "",
                "Ledger records and measured payloads were integrity-checked, "
                "but this report is not publication-attested because execution "
                "is incomplete, a calculation failed, or dataset hashes were not "
                "verified.",
                "",
            ]
        )

    atomic_write_text(path, "\n".join(lines) + "\n")


def _publication_attested(metadata: Mapping[str, Any]) -> bool:
    run_spec = metadata.get("run_spec")
    hashes_verified = (
        bool(run_spec.get("dataset_hashes_verified"))
        if isinstance(run_spec, dict)
        else False
    )
    status_counts = metadata.get("status_counts")
    failed_tasks = (
        int(status_counts.get("failed") or 0)
        if isinstance(status_counts, Mapping)
        else 0
    )
    return bool(
        metadata.get("source_attested")
        and metadata.get("execution_complete")
        and metadata.get("run_fingerprint")
        and hashes_verified
        and failed_tasks == 0
    )


def _write_report_manifest(
    path: Path,
    *,
    output_dir: Path,
    artifacts: list[Path],
    metadata: Mapping[str, Any],
    dataset_kind: str,
    baseline_adapter: str,
) -> None:
    """Bind every report artifact to its run provenance and file digest.

    The manifest itself is excluded because a file cannot contain its own
    stable checksum. It is written last and atomically.
    """

    entries: list[Dict[str, Any]] = []
    unique_paths = sorted(
        {artifact.resolve() for artifact in artifacts},
        key=lambda artifact: str(artifact),
    )
    for artifact in unique_paths:
        if artifact == path.resolve():
            continue
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Report artifact is missing before manifest commit: {artifact}"
            )
        try:
            relative = artifact.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Report artifact is outside the output directory: {artifact}"
            ) from exc
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        )

    run_spec = metadata.get("run_spec")
    hashes_verified = (
        bool(run_spec.get("dataset_hashes_verified"))
        if isinstance(run_spec, dict)
        else False
    )
    manifest = {
        "schema_version": 1,
        "run_id": metadata.get("run_id"),
        "run_fingerprint": metadata.get("run_fingerprint"),
        "report_generator": dict(metadata.get("report_generator") or {}),
        "run_status": metadata.get("run_status"),
        "execution_complete": bool(metadata.get("execution_complete")),
        "source_attested": bool(metadata.get("source_attested")),
        "publication_attested": _publication_attested(metadata),
        "dataset_hashes_verified": hashes_verified,
        "dataset_kind": dataset_kind,
        "baseline_adapter": baseline_adapter,
        "power_mode_summary": dict(metadata.get("power_mode_summary") or {}),
        "qc_summary": dict(metadata.get("qc_summary") or {}),
        "record_source": metadata.get("record_source"),
        "status_counts": dict(metadata.get("status_counts") or {}),
        "verified_payload_count": metadata.get("verified_payload_count"),
        "artifact_count": len(entries),
        "manifest_self_excluded": True,
        "artifacts": entries,
    }
    atomic_write_json(path, manifest)


def generate_report(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    record_loader: ReportRecordLoader | None = None,
) -> Dict[str, Path]:
    """Generate the performance report without executing benchmark tasks."""

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    records, metadata = load_report_records(input_path, record_loader=record_loader)
    metadata["report_generator"] = _report_generator_provenance()
    from bench.run import run_qc_checks

    qc = run_qc_checks(
        str(metadata.get("run_id") or input_path.name),
        records.to_dict(orient="records"),
    )
    metadata["qc_summary"] = dict(qc["summary"])
    qc_issues = pd.DataFrame(
        [
            {column: issue.get(column) for column in PUBLIC_QC_ISSUE_COLUMNS}
            for issue in qc["issues"]
        ],
        columns=PUBLIC_QC_ISSUE_COLUMNS,
    )
    qc_issues["requested_families"] = qc_issues["requested_families"].map(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if isinstance(value, (dict, list, tuple))
        else value
    )
    measured = timing_observations(records)
    if measured.empty:
        raise ValueError(
            "No measured timing observations found in report input."
        )
    kind = _resolve_dataset_kind(measured, metadata.get("dataset_kind"))
    baseline_adapter = _baseline_adapter_from_metadata(metadata)
    metadata["baseline_adapter"] = baseline_adapter
    metadata["publication_attested"] = _publication_attested(metadata)

    timing_summary = aggregate_timing_observations(measured, dataset_kind=kind)
    matched = matched_runtime_observations(measured, baseline_adapter=baseline_adapter)
    matched_summary = matched_runtime_summary(matched, dataset_kind=kind)
    diagnostics = _diagnostic_table(non_timing_outcomes(records))
    feature_contract = feature_workload_contract(records)

    table_path = output_path / "adapter_performance_table.csv"
    pair_path = output_path / "matched_runtime_pairs.csv"
    matched_summary_path = output_path / "matched_runtime_summary.csv"
    diagnostic_path = output_path / "excluded_outcomes.csv"
    feature_contract_path = output_path / "feature_workload_contract.csv"
    protocol_path = output_path / "protocol_summary.csv"
    environment_path = output_path / "adapter_environment_summary.csv"
    observations_path = output_path / "task_observations.csv"
    qc_issues_path = output_path / "qc_issues.csv"
    qc_summary_path = output_path / "qc_summary.json"
    _atomic_write_dataframe_csv(timing_summary, table_path)
    _atomic_write_dataframe_csv(matched, pair_path)
    _atomic_write_dataframe_csv(matched_summary, matched_summary_path)
    _atomic_write_dataframe_csv(diagnostics, diagnostic_path)
    _atomic_write_dataframe_csv(qc_issues, qc_issues_path)
    atomic_write_json(qc_summary_path, qc["summary"])
    _atomic_write_dataframe_csv(feature_contract, feature_contract_path)
    _atomic_write_dataframe_csv(
        _protocol_summary(metadata.get("run_spec")), protocol_path
    )
    _atomic_write_dataframe_csv(
        _adapter_environment_summary(metadata.get("run_spec")),
        environment_path,
    )

    figures = generate_plots(
        measured,
        output_path,
        dataset_kind=kind,
        baseline_adapter=baseline_adapter,
        matched=matched,
    )
    figure_manifest_path = output_path / "figure_manifest.csv"
    _atomic_write_dataframe_csv(
        pd.DataFrame(
            figures,
            columns=["artifact", "format", "alt_text"],
        ),
        figure_manifest_path,
    )

    adapters = sorted(measured["adapter"].dropna().astype(str).unique())
    verified_payloads = metadata.get("verified_payloads")
    if not isinstance(verified_payloads, list):
        raise ValueError(
            "Report generation requires checksum-verified payloads from the ledger"
        )
    _atomic_write_dataframe_csv(
        publication_task_observations(records, verified_payloads),
        observations_path,
    )
    coverage = summarize_features_from_payloads(verified_payloads, adapters)
    coverage_path = output_path / "feature_coverage_report.json"
    atomic_write_json(coverage_path, coverage)
    coverage_plot_path = generate_feature_coverage_plot(coverage, output_path)

    workbook_path = generate_excel_report(
        timing_summary,
        matched,
        matched_summary,
        feature_contract,
        diagnostics,
        qc_issues,
        metadata,
        output_path,
    )
    summary_path = output_path / "academic_report_summary.md"
    _write_markdown_summary(
        summary_path,
        input_dir=input_path,
        metadata=metadata,
        measured_n=len(measured),
        matched=matched,
        matched_summary=matched_summary,
        feature_contract=feature_contract,
        diagnostics=diagnostics,
        coverage=coverage,
        figures=figures,
    )
    artifact_paths = [
        table_path,
        pair_path,
        matched_summary_path,
        diagnostic_path,
        qc_issues_path,
        qc_summary_path,
        feature_contract_path,
        protocol_path,
        environment_path,
        observations_path,
        figure_manifest_path,
        coverage_path,
        workbook_path,
        summary_path,
    ]
    for figure in figures:
        pdf_path = output_path / figure["artifact"]
        artifact_paths.extend([pdf_path, pdf_path.with_suffix(".svg")])
    if coverage_plot_path is not None:
        artifact_paths.extend(
            [coverage_plot_path, coverage_plot_path.with_suffix(".svg")]
        )
    report_manifest_path = output_path / "report_manifest.json"
    _write_report_manifest(
        report_manifest_path,
        output_dir=output_path,
        artifacts=artifact_paths,
        metadata=metadata,
        dataset_kind=kind,
        baseline_adapter=baseline_adapter,
    )
    return {
        "timing_summary": table_path,
        "matched_pairs": pair_path,
        "matched_summary": matched_summary_path,
        "excluded_outcomes": diagnostic_path,
        "qc_issues": qc_issues_path,
        "qc_summary": qc_summary_path,
        "feature_workload_contract": feature_contract_path,
        "protocol_summary": protocol_path,
        "adapter_environments": environment_path,
        "task_observations": observations_path,
        "figure_manifest": figure_manifest_path,
        "feature_coverage": coverage_path,
        "workbook": workbook_path,
        "markdown_summary": summary_path,
        "report_manifest": report_manifest_path,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench report")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Run directory containing the authoritative benchmark.sqlite3 ledger",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for report tables and figures",
    )
    args = parser.parse_args(argv)
    outputs = generate_report(args.input_dir, args.output_dir)
    print(f"Report generated: {outputs['markdown_summary'].parent}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

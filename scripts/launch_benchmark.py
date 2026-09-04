#!/usr/bin/env python3
"""Print, dry-validate, or explicitly execute the frozen three-pillar runs.

The default action only prints the commands.  Radiomic calculation requires
both ``--execute`` and the literal acknowledgement ``--confirm CALCULATE``.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import psutil

from bench.benchmark_ledger import atomic_write_json, sha256_file
from bench.benchmark_models import RUN_SPEC_SCHEMA_VERSION
from bench.benchmark_workspace import WORKSPACE_MANIFEST_SCHEMA_VERSION
from bench.power_provenance import observe_task_power_state


PILLARS = (
    ("pillar1_morphology", "pillar1"),
    ("pillar2_whole_anatomy", "pillar2_a1"),
    ("pillar3_ibsi2_phase3", "ibsi2_phase3"),
)


MACHINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
WINDOWS_PATH_BUDGET = 240
RESULT_BYTES_PER_ELIGIBLE_TASK = 64 * 1024
RESULT_FIXED_RESERVE_BYTES = 2 * 1024**3
PMSET_ENERGY_MODES = {0: "automatic", 1: "low_power", 2: "high_power"}
PMSET_MODE_ATTEMPTS = 30
PMSET_MODE_RETRY_SECONDS = 0.5
ES_SYSTEM_REQUIRED = 0x00000001
ES_CONTINUOUS = 0x80000000


def _portable_command(
    command: list[str],
    repository: Path,
    result_root: Path | None = None,
) -> list[str]:
    roots = [(repository.resolve(), None)]
    if result_root is not None:
        roots.insert(0, (result_root.resolve(), "{RESULT_ROOT}"))
    portable: list[str] = []
    for value in command:
        path = Path(value)
        if not path.is_absolute():
            portable.append(value)
            continue
        replacement = value
        for root, token in roots:
            try:
                # Preserve a repository-local interpreter path lexically. Its
                # executable may be a symlink to a system Python outside the
                # repository, but the invocation itself remains portable.
                relative = path.relative_to(root)
            except ValueError:
                continue
            replacement = (
                relative.as_posix()
                if token is None
                else f"{token}/{relative.as_posix()}"
            )
            break
        portable.append(replacement)
    return portable


def _default_machine_id() -> str:
    private_identity = socket.gethostname().encode("utf-8")
    return "anonymous-" + hashlib.sha256(private_identity).hexdigest()[:16]


def _validate_machine_id(value: str | None) -> str:
    resolved = str(value or "").strip() or _default_machine_id()
    if not MACHINE_ID_PATTERN.fullmatch(resolved):
        raise ValueError(
            "machine ID must be 1-64 characters using only letters, numbers, "
            "periods, underscores, or hyphens"
        )
    return resolved


def _load_host_profile(path: Path) -> dict[str, Any]:
    """Load a public-safe, checksum-bound benchmark host profile."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported benchmark host profile")
    required_strings = ("profile_id", "machine_id", "machine_label", "cpu_model")
    for field in required_strings:
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"host profile is missing {field}")
    if payload["profile_id"] != payload["machine_id"]:
        raise ValueError("host profile ID and machine ID must match")
    _validate_machine_id(str(payload["machine_id"]))
    for field in ("expected_hardware", "required_runtime_state", "benchmark_settings"):
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"host profile {field} must be a JSON object")
    forbidden_keys = {"serial", "serial_number", "hardware_uuid", "hostname", "udid"}
    pending: list[Any] = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise ValueError(f"host profile contains private field: {key}")
                pending.append(item)
        elif isinstance(value, list):
            pending.extend(value)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "payload": payload,
    }


def _profile_value(
    supplied: Any,
    profile: dict[str, Any] | None,
    field: str,
    *,
    flag: str,
) -> Any:
    if profile is None:
        return supplied
    frozen = profile["payload"].get(field)
    if supplied is not None and str(supplied) != str(frozen):
        raise ValueError(f"{flag} conflicts with the frozen host profile")
    return frozen


def _darwin_power_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "power_source": None,
        "low_power_mode": None,
        "energy_mode": None,
        "sleep_assertion": None,
        "probe_errors": [],
    }
    try:
        battery = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        match = re.search(r"Now drawing from '([^']+)'", battery.stdout or "")
        if battery.returncode == 0 and match:
            state["power_source"] = match.group(1)
        else:
            state["probe_errors"].append("power_source_unavailable")
    except (OSError, subprocess.SubprocessError):
        state["probe_errors"].append("power_source_probe_failed")
    custom_probe_failed = False
    # macOS may briefly omit the updated AC profile while Energy Mode is
    # changing. Retry with a delay, but never infer an absent mode.
    for attempt in range(PMSET_MODE_ATTEMPTS):
        try:
            custom = subprocess.run(
                ["/usr/bin/pmset", "-g", "custom"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            custom_probe_failed = True
            custom = None
        if custom is not None:
            ac_block = re.search(
                r"(?:^|\n)AC Power:\s*(.*?)(?=\n\S[^\n]*:\s*$|\Z)",
                custom.stdout or "",
                flags=re.DOTALL,
            )
            low_power = (
                re.search(
                    r"^\s*lowpowermode\s+(\d+)\s*$",
                    ac_block.group(1),
                    re.MULTILINE,
                )
                if custom.returncode == 0 and ac_block
                else None
            )
            if low_power:
                value = int(low_power.group(1))
                state["low_power_mode"] = value
                state["energy_mode"] = PMSET_ENERGY_MODES.get(value, f"unknown_{value}")
                break
        if attempt + 1 < PMSET_MODE_ATTEMPTS:
            time.sleep(PMSET_MODE_RETRY_SECONDS)

    # The active-profile view is an independent fallback for hosts where the
    # custom-profile view transiently omits the lowpowermode setting.
    if state["low_power_mode"] is None:
        try:
            current = subprocess.run(
                ["/usr/bin/pmset", "-g"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            low_power = (
                re.search(
                    r"^\s*lowpowermode\s+(\d+)\s*$",
                    current.stdout or "",
                    re.MULTILINE,
                )
                if current.returncode == 0
                else None
            )
            if low_power:
                value = int(low_power.group(1))
                state["low_power_mode"] = value
                state["energy_mode"] = PMSET_ENERGY_MODES.get(value, f"unknown_{value}")
        except (OSError, subprocess.SubprocessError):
            custom_probe_failed = True
    if state["low_power_mode"] is None:
        state["probe_errors"].append(
            "low_power_mode_probe_failed"
            if custom_probe_failed
            else "low_power_mode_unavailable"
        )
    try:
        assertions = subprocess.run(
            ["/usr/bin/pmset", "-g", "assertions"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        values = [
            int(value)
            for value in re.findall(
                r"^\s*(?:PreventSystemSleep|PreventUserIdleSystemSleep)\s+(\d+)\s*$",
                assertions.stdout or "",
                flags=re.MULTILINE,
            )
        ]
        if assertions.returncode == 0 and values:
            state["sleep_assertion"] = any(value > 0 for value in values)
        else:
            state["probe_errors"].append("sleep_assertion_unavailable")
    except (OSError, subprocess.SubprocessError):
        state["probe_errors"].append("sleep_assertion_probe_failed")
    return state


def _set_windows_execution_state(flags: int) -> int:
    try:
        return int(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except (AttributeError, OSError) as exc:
        raise OSError("SetThreadExecutionState is unavailable") from exc


def _enable_windows_sleep_prevention() -> bool:
    """Hold a system-sleep assertion for the executing launcher thread."""

    if platform.system() != "Windows":
        return False
    if not _set_windows_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        raise OSError("Windows refused the benchmark sleep-prevention request")
    return True


def _disable_windows_sleep_prevention() -> None:
    if platform.system() == "Windows":
        _set_windows_execution_state(ES_CONTINUOUS)


def _host_profile_preflight(
    profile: dict[str, Any] | None,
    *,
    require_sleep_assertion: bool,
    windows_sleep_prevention_active: bool = False,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    payload = profile["payload"]
    expected = payload["expected_hardware"]
    logical_cores = psutil.cpu_count(logical=True)
    if logical_cores is None:
        logical_cores = os.cpu_count()
    observed = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": logical_cores,
        "memory_total_bytes": psutil.virtual_memory().total,
    }
    errors: list[str] = []
    for field, frozen in expected.items():
        if observed.get(field) != frozen:
            errors.append(
                f"host hardware mismatch for {field}: expected {frozen!r}, "
                f"observed {observed.get(field)!r}"
            )
    runtime_state: dict[str, Any] | None = None
    required_state = payload["required_runtime_state"]
    if expected.get("platform") == "Darwin":
        runtime_state = _darwin_power_state()
        required_power = required_state.get("power_source")
        if required_power and runtime_state["power_source"] != required_power:
            errors.append(
                f"power source must be {required_power!r}; observed "
                f"{runtime_state['power_source']!r}"
            )
        if (
            require_sleep_assertion
            and required_state.get("sleep_assertion_during_calculation") is True
            and runtime_state["sleep_assertion"] is not True
        ):
            errors.append(
                "a macOS sleep-prevention assertion is required during calculation; "
                "use scripts/run_benchmark.sh"
            )
    elif expected.get("platform") == "Windows":
        runtime_state = observe_task_power_state("Windows")
        runtime_state["sleep_prevention_active"] = bool(windows_sleep_prevention_active)
        required_power = required_state.get("power_source")
        if required_power and runtime_state["power_source"] != required_power:
            errors.append(
                f"power source must be {required_power!r}; observed "
                f"{runtime_state['power_source']!r}"
            )
        required_battery_saver = required_state.get("battery_saver")
        if (
            required_battery_saver is not None
            and runtime_state["battery_saver"] != required_battery_saver
        ):
            errors.append(
                f"battery saver must be {required_battery_saver!r}; observed "
                f"{runtime_state['battery_saver']!r}"
            )
        if (
            require_sleep_assertion
            and required_state.get("sleep_assertion_during_calculation") is True
            and not windows_sleep_prevention_active
        ):
            errors.append(
                "a Windows sleep-prevention assertion is required during calculation; "
                "use scripts/run_benchmark.ps1"
            )
    return {
        "status": "pass" if not errors else "fail",
        "profile_id": payload["profile_id"],
        "profile_path": profile["path"],
        "profile_sha256": profile["sha256"],
        "expected_hardware": expected,
        "observed_hardware": observed,
        "required_runtime_state": required_state,
        "observed_runtime_state": runtime_state,
        "benchmark_settings": payload["benchmark_settings"],
        "operator_confirmation_required": [
            "background_load",
            "operating_system_updates",
        ],
        "errors": errors,
    }


def _known_sync_root(path: Path) -> str | None:
    """Return evidence that *path* is managed by a desktop sync client."""

    resolved = path.expanduser().resolve()
    folded = str(resolved).casefold().replace("\\", "/")
    markers = (
        "/library/cloudstorage/",
        "/onedrive",
        "/dropbox/",
        "/google drive/",
        "/google/drivefs/",
    )
    if any(marker in folded for marker in markers):
        return "path_marker"
    for variable in (
        "OneDrive",
        "OneDriveConsumer",
        "OneDriveCommercial",
        "Dropbox",
        "GoogleDrive",
    ):
        raw = str(os.environ.get(variable) or "").strip()
        if not raw:
            continue
        try:
            resolved.relative_to(Path(raw).expanduser().resolve())
        except ValueError:
            continue
        return f"environment:{variable}"
    return None


def _existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise ValueError(f"no existing parent for result root: {path}")
    return candidate


def _staged_input_bytes(root: Path, workspace: dict[str, Any]) -> int:
    total = 0
    for pillar, _ in PILLARS:
        dataset = workspace["datasets"][pillar]
        manifest = json.loads(
            (root / dataset["path"] / "manifest.json").read_text(encoding="utf-8")
        )
        unique_files: dict[tuple[str, str], int] = {}
        for item in manifest.get("files", []):
            checksum = str(item.get("sha256") or "").lower()
            suffix = (
                ".nii.gz"
                if str(item.get("path") or "").lower().endswith(".nii.gz")
                else Path(str(item.get("path") or "")).suffix.lower()
            )
            unique_files[(checksum, suffix)] = int(item.get("bytes") or 0)
        total += sum(unique_files.values())
    return total


def _git_source_state(repository: Path) -> dict[str, Any]:
    """Resolve the exact clean source commit used for benchmark execution."""

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "commit": None, "dirty_entries": []}
    dirty_entries = [
        line for line in (status.stdout or "").splitlines() if line.strip()
    ]
    resolved_commit = (commit.stdout or "").strip()
    available = (
        status.returncode == 0
        and commit.returncode == 0
        and bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", resolved_commit))
    )
    return {
        "status": (
            "unavailable" if not available else "dirty" if dirty_entries else "clean"
        ),
        "commit": resolved_commit if available else None,
        "dirty_entries": dirty_entries,
    }


def _result_preflight(
    root: Path,
    workspace: dict[str, Any],
    *,
    result_root: Path,
    machine_id: str,
    allow_synced_results: bool,
    repeats: int | None = None,
    host_profile: dict[str, Any] | None = None,
    require_sleep_assertion: bool = False,
    require_clean_source: bool = False,
    windows_sleep_prevention_active: bool = False,
) -> dict[str, Any]:
    """Check storage isolation without creating a result directory."""

    result_root = result_root.expanduser().resolve()
    machine_root = result_root / machine_id
    existing_parent = _existing_parent(result_root)
    disk = shutil.disk_usage(existing_parent)
    staged_bytes = _staged_input_bytes(root, workspace)
    reviewed_repeats = int(workspace["task_inventory"]["fresh_process_repeats"])
    requested_repeats = reviewed_repeats if repeats is None else int(repeats)
    if requested_repeats < reviewed_repeats:
        raise ValueError(f"repeats cannot be below reviewed minimum {reviewed_repeats}")
    reviewed_eligible_tasks = int(
        workspace["task_inventory"]["totals"]["eligible_calculation_tasks"]
    )
    eligible_tasks = reviewed_eligible_tasks * requested_repeats // reviewed_repeats
    estimated_payload_bytes = eligible_tasks * RESULT_BYTES_PER_ELIGIBLE_TASK
    minimum_free_bytes = (
        staged_bytes + estimated_payload_bytes + RESULT_FIXED_RESERVE_BYTES
    )
    sync_evidence = _known_sync_root(result_root)
    repository_sync_evidence = _known_sync_root(root.resolve().parents[1])
    longest_pillar = max(pillar for pillar, _ in PILLARS)
    representative_record = (
        machine_root
        / longest_pillar
        / "records"
        / "p1_high_contrast_m4_n512"
        / "pictologics_local_intensity_rep3_0123456789ab.json"
    )
    representative_input = (
        machine_root
        / longest_pillar
        / "inputs"
        / "ff"
        / ("f" * 64)
        / "STS_051_pet_ivh_fbn1000.nii.gz"
    )
    estimated_max_path_chars = max(
        len(str(representative_record)), len(str(representative_input))
    )
    errors: list[str] = []
    warnings: list[str] = []
    source_state = (
        _git_source_state(root.resolve().parents[1]) if require_clean_source else None
    )
    if require_clean_source and source_state["status"] != "clean":
        errors.append(
            "source repository must be a clean Git commit before controller dry-runs "
            "or calculations"
        )
    errors.extend(
        _existing_result_resume_errors(
            machine_root,
            workspace,
            source_commit=(
                source_state.get("commit")
                if source_state and source_state.get("status") == "clean"
                else None
            ),
        )
    )
    host_preflight = _host_profile_preflight(
        host_profile,
        require_sleep_assertion=require_sleep_assertion,
        windows_sleep_prevention_active=windows_sleep_prevention_active,
    )
    if host_preflight and host_preflight["status"] != "pass":
        errors.extend(host_preflight["errors"])
    if sync_evidence and not allow_synced_results:
        errors.append(
            "result root is managed by a desktop sync client; SQLite WAL files, "
            "locks, and atomic payload commits require a local unsynchronised path"
        )
    if disk.free < minimum_free_bytes:
        errors.append(
            "insufficient free space for staged immutable inputs and a conservative "
            "result-payload reserve"
        )
    if os.name == "nt" and estimated_max_path_chars > WINDOWS_PATH_BUDGET:
        errors.append(
            f"estimated result path length {estimated_max_path_chars} exceeds the "
            f"{WINDOWS_PATH_BUDGET}-character native-library safety budget"
        )
    elif estimated_max_path_chars > WINDOWS_PATH_BUDGET:
        warnings.append(
            f"this result root would exceed the {WINDOWS_PATH_BUDGET}-character "
            "Windows path budget if reused there"
        )
    if repository_sync_evidence:
        warnings.append(
            "the source/input workspace is cloud-synchronised; fully hydrate it and "
            "prefer a short local execution copy, especially on Windows"
        )
    temp_roots = {Path(tempfile.gettempdir()).resolve()}
    if os.name != "nt":
        temp_roots.add(Path("/tmp").resolve())
    if any(
        machine_root == temporary or temporary in machine_root.parents
        for temporary in temp_roots
    ):
        warnings.append(
            "result root is temporary storage; copy the complete machine directory "
            "to durable storage only after the run is stopped and SQLite is closed"
        )
    return {
        "status": "pass" if not errors else "fail",
        "machine_id": machine_id,
        "workspace_root": str(root),
        "result_root": str(result_root),
        "machine_result_root": str(machine_root),
        "result_root_sync_evidence": sync_evidence,
        "workspace_sync_evidence": repository_sync_evidence,
        "staged_input_bytes": staged_bytes,
        "estimated_payload_bytes": estimated_payload_bytes,
        "fresh_process_repeats": requested_repeats,
        "fixed_reserve_bytes": RESULT_FIXED_RESERVE_BYTES,
        "minimum_free_bytes": minimum_free_bytes,
        "available_bytes": disk.free,
        "estimated_max_path_chars": estimated_max_path_chars,
        "windows_path_budget_chars": WINDOWS_PATH_BUDGET,
        "host_profile": host_preflight,
        "source_state": source_state,
        "errors": errors,
        "warnings": warnings,
    }


def _effective_host_settings(
    host_profile: dict[str, Any] | None,
    preflight: dict[str, Any],
) -> dict[str, Any] | None:
    """Return fixed host settings plus non-gating session power provenance."""

    if host_profile is None:
        return None
    settings = dict(host_profile["payload"]["benchmark_settings"])
    host_preflight = preflight.get("host_profile") or {}
    runtime_state = host_preflight.get("observed_runtime_state") or {}
    if runtime_state:
        is_windows = (
            runtime_state.get("platform") == "Windows"
            or "power_scheme_guid" in runtime_state
        )
        for field in (
            "battery_life_percent",
            "battery_saver",
            "energy_mode",
            "energy_mode_observation_status",
            "pmset_lowpowermode",
            "power_mode_tag",
            "power_scheme_guid",
            "power_scheme_name",
            "power_source",
            "sleep_prevention_active",
        ):
            if field in runtime_state:
                settings[field] = runtime_state[field]
        settings["energy_mode"] = runtime_state.get("energy_mode") or "unavailable"
        mode_value = runtime_state.get("low_power_mode")
        if (
            not is_windows
            and "pmset_lowpowermode" not in settings
            and "low_power_mode" in runtime_state
        ):
            settings["pmset_lowpowermode"] = mode_value
        if not settings.get("power_mode_tag"):
            mode_label = runtime_state.get("energy_mode") or "unavailable"
            if is_windows:
                settings["power_mode_tag"] = "windows-power-scheme-unavailable"
            else:
                settings["power_mode_tag"] = (
                    f"macos-{str(mode_label).replace('_', '-')}-pmset-{mode_value}"
                    if mode_value is not None
                    else "macos-energy-mode-unavailable"
                )
        if not settings.get("energy_mode_observation_status"):
            settings["energy_mode_observation_status"] = (
                "observed"
                if (settings["energy_mode"] != "unavailable")
                else "unavailable"
            )
        settings["power_state_probe_errors"] = list(
            runtime_state.get("probe_errors") or []
        )
    return settings


def _load_workspace(root: Path) -> dict[str, Any]:
    path = root / "workspace_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    observed_schema = value.get("schema_version")
    if observed_schema != WORKSPACE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark workspace manifest: expected schema "
            f"{WORKSPACE_MANIFEST_SCHEMA_VERSION}, observed {observed_schema!r}; "
            "refresh it without calculations using `poetry run python "
            "scripts/prepare_benchmark_workspace.py --output-root "
            f"{root} --validate-only`"
        )
    if value.get("benchmark_timing_executed") is not False:
        raise ValueError("workspace manifest is not a pristine input-only record")
    repository = root.resolve().parents[1]
    for source in value.get("workspace_sources", []):
        source_path = repository / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError(
                f"workspace preparation/launch source changed: {source['path']}"
            )
    for pillar, _ in PILLARS:
        dataset = value["datasets"][pillar]
        manifest_path = root / dataset["path"] / "manifest.json"
        if sha256_file(manifest_path) != dataset["manifest_sha256"]:
            raise ValueError(f"dataset manifest changed after validation: {pillar}")
    return value


def _existing_result_resume_errors(
    machine_root: Path,
    workspace: dict[str, Any],
    *,
    source_commit: str | None,
) -> list[str]:
    """Reject prior ledgers that cannot be resumed by the current protocol."""

    expected_contract_sha256 = str(workspace["endpoint_contract"]["sha256"])
    expected_workloads = list(workspace["launch_policy"]["reported_workloads"])
    expected_adapters = list(workspace["adapter_order"])
    errors: list[str] = []
    for pillar, _ in PILLARS:
        report_dir = machine_root / pillar
        ledger_path = report_dir / "benchmark.sqlite3"
        run_spec_path = report_dir / "run_spec.json"
        if not ledger_path.exists() and not run_spec_path.exists():
            continue
        if not ledger_path.is_file() or not run_spec_path.is_file():
            errors.append(
                f"existing result data for {pillar} is incomplete and cannot be resumed"
            )
            continue
        try:
            run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(
                f"existing result run specification for {pillar} is unreadable"
            )
            continue
        mismatches: list[str] = []
        if not isinstance(run_spec, dict):
            mismatches.append("run specification is not a JSON object")
        else:
            if run_spec.get("schema_version") != RUN_SPEC_SCHEMA_VERSION:
                mismatches.append(
                    "run-spec schema "
                    f"{run_spec.get('schema_version')!r} != {RUN_SPEC_SCHEMA_VERSION}"
                )
            if run_spec.get("endpoint_contract_sha256") != expected_contract_sha256:
                mismatches.append("endpoint contract differs")
            if run_spec.get("workloads") != expected_workloads:
                mismatches.append("workload set differs")
            if run_spec.get("adapters") != expected_adapters:
                mismatches.append("adapter set differs")
            expected_manifest_sha256 = workspace["datasets"][pillar]["manifest_sha256"]
            if run_spec.get("manifest_sha256") != expected_manifest_sha256:
                mismatches.append("dataset manifest differs")
        run_meta_path = report_dir / "run_meta.json"
        if source_commit and run_meta_path.is_file():
            try:
                run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                mismatches.append("run metadata is unreadable")
            else:
                if (
                    not isinstance(run_meta, dict)
                    or run_meta.get("git_commit") != source_commit
                ):
                    mismatches.append("source commit differs")
        if mismatches:
            errors.append(
                f"existing result data for {pillar} is incompatible with the "
                f"current benchmark ({'; '.join(mismatches)}); use a new empty "
                "--result-root or archive/remove the prior run"
            )
    return errors


def build_commands(
    root: Path,
    workspace: dict[str, Any],
    *,
    result_root: Path,
    dry_run: bool,
    machine_id: str | None = None,
    machine_label: str | None = None,
    cpu_model: str | None = None,
    cpu_base_ghz: float | None = None,
    host_profile_id: str | None = None,
    host_profile_sha256: str | None = None,
    host_settings: dict[str, Any] | None = None,
    repeats: int | None = None,
) -> list[list[str]]:
    policy = workspace["launch_policy"]
    endpoint = workspace["endpoint_contract"]
    repository = root.resolve().parents[1]
    resolved_machine_id = _validate_machine_id(machine_id)
    machine_result_root = result_root.expanduser().resolve() / resolved_machine_id
    reviewed_repeats = int(workspace["task_inventory"]["fresh_process_repeats"])
    requested_repeats = reviewed_repeats if repeats is None else int(repeats)
    if requested_repeats < reviewed_repeats:
        raise ValueError(f"repeats cannot be below reviewed minimum {reviewed_repeats}")
    commands: list[list[str]] = []
    pillar_task_counts = {
        pillar: int(workspace["datasets"][pillar]["case_count"])
        * int(workspace["task_inventory"]["adapter_count"])
        * int(workspace["task_inventory"]["workload_count"])
        * requested_repeats
        for pillar, _ in PILLARS
    }
    project_total_tasks = sum(pillar_task_counts.values())
    project_task_offset = 0
    for pillar, directory in PILLARS:
        command = [
            sys.executable,
            "-m",
            "bench.cli",
            "run",
            "--dataset-dir",
            str((root / directory).resolve()),
            "--run-id",
            f"{pillar}_calculation_only",
            "--report-dir",
            str((machine_result_root / pillar).resolve()),
            "--endpoint-contract",
            str((repository / endpoint["path"]).resolve()),
            "--input-contract",
            "manifest_harmonized",
            "--adapters",
            ",".join(workspace["adapter_order"]),
            "--workloads",
            "all",
            "--repeats",
            str(requested_repeats),
            "--timing-observations",
            str(workspace["task_inventory"]["measured_observations_per_process"]),
            "--timeout",
            str(policy["timeout_seconds"]),
            "--checkpoint-interval",
            str(policy["checkpoint_interval_tasks"]),
            "--progress-interval",
            str(policy.get("progress_interval_seconds", 30.0)),
            "--project-total-tasks",
            str(project_total_tasks),
            "--project-task-offset",
            str(project_task_offset),
            "--machine-id",
            resolved_machine_id,
            "--keep-going",
            "--resume",
        ]
        if machine_label:
            command.extend(["--machine-label", str(machine_label)])
        if cpu_model:
            command.extend(["--cpu-model", str(cpu_model)])
        if cpu_base_ghz is not None:
            command.extend(["--cpu-base-ghz", str(float(cpu_base_ghz))])
        if host_profile_id:
            command.extend(["--host-profile-id", host_profile_id])
        if host_profile_sha256:
            command.extend(["--host-profile-sha256", host_profile_sha256])
        if host_settings is not None:
            command.extend(
                [
                    "--host-settings-json",
                    json.dumps(host_settings, sort_keys=True, separators=(",", ":")),
                ]
            )
        if requested_repeats > reviewed_repeats:
            command.append("--extend-repeats")
        if dry_run:
            command.append("--dry-run")
        commands.append(command)
        project_task_offset += pillar_task_counts[pillar]
    return commands


def _run_controller_command(command: list[str]) -> int:
    """Forward stop signals and wait for the controller's final checkpoint."""

    process: subprocess.Popen[str] | None = None
    pending_signals: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def send_to_controller(signum: int) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                # The controller is in its own process group.  CTRL_BREAK is
                # the reliable group-targeted console event on Windows; the
                # controller registers SIGBREAK as an orderly-stop request.
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signum)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def forward_signal(signum, frame) -> None:
        del frame
        if process is None:
            pending_signals.append(int(signum))
        else:
            send_to_controller(int(signum))

    if threading.current_thread() is threading.main_thread():
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for signum in signals:
            previous_handlers[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
    try:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        for signum in pending_signals:
            send_to_controller(signum)
        return int(process.wait())
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default="data/benchmark")
    parser.add_argument("--result-root", default="results/benchmark")
    parser.add_argument(
        "--machine-id",
        default=None,
        help="Stable public-safe ID; defaults to an anonymous hostname hash",
    )
    parser.add_argument("--machine-label", default=None)
    parser.add_argument("--cpu-model", default=None)
    parser.add_argument("--cpu-base-ghz", type=float, default=None)
    parser.add_argument(
        "--host-profile",
        type=Path,
        default=None,
        help="Frozen public-safe host identity and operating-settings profile",
    )
    parser.add_argument(
        "--allow-synced-results",
        action="store_true",
        help="Explicitly override the fail-closed cloud-synchronised result check",
    )
    parser.add_argument(
        "--validate-plans",
        action="store_true",
        help="Invoke each controller with --dry-run; no adapter calculation occurs",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Absolute fresh-process repeat horizon; use 5 to append repeats 4 and 5",
    )
    args = parser.parse_args()
    if args.execute and args.confirm != "CALCULATE":
        parser.error("--execute requires --confirm CALCULATE")
    if args.execute and args.validate_plans:
        parser.error("choose either --execute or --validate-plans")

    windows_sleep_prevention_active = (
        _enable_windows_sleep_prevention() if args.execute else False
    )
    if windows_sleep_prevention_active:
        atexit.register(_disable_windows_sleep_prevention)

    root = Path(args.workspace_root).expanduser().resolve()
    workspace = _load_workspace(root)
    host_profile = _load_host_profile(args.host_profile) if args.host_profile else None
    machine_id = _validate_machine_id(
        _profile_value(
            args.machine_id,
            host_profile,
            "machine_id",
            flag="--machine-id",
        )
    )
    machine_label = _profile_value(
        args.machine_label,
        host_profile,
        "machine_label",
        flag="--machine-label",
    )
    cpu_model = _profile_value(
        args.cpu_model,
        host_profile,
        "cpu_model",
        flag="--cpu-model",
    )
    cpu_base_ghz = _profile_value(
        args.cpu_base_ghz,
        host_profile,
        "cpu_base_ghz",
        flag="--cpu-base-ghz",
    )
    result_root = Path(args.result_root).expanduser().resolve()
    preflight = _result_preflight(
        root,
        workspace,
        result_root=result_root,
        machine_id=machine_id,
        allow_synced_results=args.allow_synced_results,
        repeats=args.repeats,
        host_profile=host_profile,
        require_sleep_assertion=args.execute,
        require_clean_source=args.execute or args.validate_plans,
        windows_sleep_prevention_active=windows_sleep_prevention_active,
    )
    if (args.execute or args.validate_plans) and preflight["status"] != "pass":
        parser.error("execution preflight failed: " + "; ".join(preflight["errors"]))
    host_settings = _effective_host_settings(host_profile, preflight)
    commands = build_commands(
        root,
        workspace,
        result_root=result_root,
        dry_run=args.validate_plans,
        machine_id=machine_id,
        machine_label=machine_label,
        cpu_model=cpu_model,
        cpu_base_ghz=cpu_base_ghz,
        host_profile_id=(
            str(host_profile["payload"]["profile_id"]) if host_profile else None
        ),
        host_profile_sha256=(str(host_profile["sha256"]) if host_profile else None),
        host_settings=host_settings,
        repeats=args.repeats,
    )
    plan = {
        "workspace_manifest_sha256": sha256_file(root / "workspace_manifest.json"),
        "mode": (
            "execute"
            if args.execute
            else "controller_dry_run"
            if args.validate_plans
            else "print_only"
        ),
        "radiomic_calculation_started": bool(args.execute),
        "machine": {
            "machine_id": machine_id,
            "machine_label": machine_label,
            "host_profile_id": (
                host_profile["payload"]["profile_id"] if host_profile else None
            ),
            "host_profile_sha256": host_profile["sha256"] if host_profile else None,
        },
        "execution_preflight": preflight,
        "commands": commands,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute and not args.validate_plans:
        return 0
    if args.validate_plans:
        validated: list[dict[str, Any]] = []
        repository = root.resolve().parents[1]
        for (pillar, _), command in zip(PILLARS, commands):
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            fingerprint_line = next(
                (
                    line
                    for line in output.splitlines()
                    if line.startswith("Immutable run fingerprint: ")
                ),
                None,
            )
            validated_line = next(
                (line for line in output.splitlines() if line.startswith("Validated ")),
                None,
            )
            if fingerprint_line is None or validated_line is None:
                raise RuntimeError(
                    f"controller dry-run did not emit its attestation for {pillar}"
                )
            validated.append(
                {
                    "pillar": pillar,
                    "summary": validated_line,
                    "run_fingerprint": fingerprint_line.split(": ", 1)[1],
                    "command": _portable_command(command, repository, result_root),
                }
            )
            print(output)
        record = {
            "schema_version": 1,
            "status": "controller_plans_validated_calculations_not_started",
            "radiomic_calculation_started": False,
            "workspace_manifest_sha256": plan["workspace_manifest_sha256"],
            "machine": plan["machine"],
            "execution_preflight": preflight,
            "plans": validated,
        }
        attestation_dir = root / "host_attestations" / machine_id
        atomic_write_json(attestation_dir / "controller_dry_run.json", record)
        print(json.dumps(record, indent=2))
        return 0
    try:
        for command in commands:
            returncode = _run_controller_command(command)
            if returncode != 0:
                return returncode
        return 0
    finally:
        if windows_sleep_prevention_active:
            _disable_windows_sleep_prevention()
            atexit.unregister(_disable_windows_sleep_prevention)


if __name__ == "__main__":
    raise SystemExit(main())

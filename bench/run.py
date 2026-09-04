from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from functools import lru_cache
from numbers import Real
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import psutil

from bench.adapters.base import (
    ResultEquivalenceError,
    assert_numerically_equivalent,
)
from bench.adapters.protocol import (
    ADAPTER_PROTOCOL_VERSION,
    CALIBRATION_CV_THRESHOLD,
    CALIBRATION_HEADROOM_FACTOR,
    CALIBRATION_MAXIMUM_ROUNDS,
    CALIBRATION_MINIMUM_ROUNDS,
    CALIBRATION_SPAN_RATIO,
    REQUIRED_AGGREGATION,
    RESULT_EQUIVALENCE_ATOL,
    RESULT_EQUIVALENCE_RTOL,
    TARGET_OBSERVATION_WINDOW_SEC,
    TIMING_CONTRACT_VERSION,
    supports_aggregation,
)
from bench.benchmark_workloads import (
    BenchmarkWorkload,
    families_for_workloads,
    parse_workloads,
)
from bench.benchmark_guardrails import GuardrailPolicy
from bench.benchmark_eta import estimate_pending_turnaround
from bench.benchmark_memory import GIB, MemoryPreflightPolicy, evaluate_memory_preflight
from bench.benchmark_contract import BenchmarkContract, load_benchmark_contract
from bench.benchmark_ledger import (
    BenchmarkLedger,
    RunAlreadyExists,
    RunLock,
    RunIntegrityError,
    RunSpecMismatch,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from bench.benchmark_models import (
    RECOVERABLE_STATUSES,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_MEASURED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SKIPPED_TIMEOUT,
    STATUS_TIMED_OUT,
    STATUS_UNSUPPORTED,
    TERMINAL_STATUSES,
    RunSpec,
    TaskSpec,
    fingerprint,
    run_spec_identity,
    task_plan_fingerprint,
)
from bench.power_provenance import (
    observe_task_power_state,
    summarize_task_power_records,
)
from bench.benchmark_representations import (
    HARMONIZED_INPUT_CONTRACT,
    select_representation,
)
from bench.dataset_manifest import contained_path, load_and_validate_manifest


BENCHMARK_EVENT_PREFIX = "BENCH_EVENT "


BENCHMARK_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    "BLIS_NUM_THREADS",
)


def _benchmark_thread_count(physical_cores: Any = None) -> int:
    """Resolve the all-core allowance used by every isolated adapter task."""

    try:
        count = int(physical_cores)
    except (TypeError, ValueError):
        count = int(
            psutil.cpu_count(logical=False) or psutil.cpu_count() or os.cpu_count() or 1
        )
    return max(1, count)


def _benchmark_thread_environment(physical_cores: Any = None) -> Dict[str, str]:
    count = str(_benchmark_thread_count(physical_cores))
    return {variable: count for variable in BENCHMARK_THREAD_VARIABLES}


def _benchmark_thread_policy(physical_cores: Any = None) -> Dict[str, Any]:
    count = _benchmark_thread_count(physical_cores)
    return {
        "mode": "all_physical_cores_per_isolated_task",
        "requested_threads": count,
        "concurrent_adapter_processes": 1,
        "environment": _benchmark_thread_environment(count),
    }


BENCHMARK_INITIALIZATION_ENV = {
    "PICTOLOGICS_DISABLE_WARMUP": "1",
}
MEDIMAGE_BENCHMARK_INTENSITY_TYPE = "definite"
BENCHMARK_INITIALIZATION_POLICY = {
    "pictologics_automatic_import_warmup": "disabled_before_package_import",
    "scheduled_jit_warmup": "explicit_adapter_call_outside_measured_region",
    "calculation_boundary": "prepared_workload_inputs_to_radiomic_calculations",
    "mask_resegmentation_and_discretization": "outside_measured_region",
    "mirp_feature_objects_and_result_table": "outside_measured_region",
    "medimage_intensity_type": MEDIMAGE_BENCHMARK_INTENSITY_TYPE,
    "zrad_local_intensity_cache": "cleared_before_and_after_each_calculation",
    "environment": BENCHMARK_INITIALIZATION_ENV,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


@lru_cache(maxsize=None)
def _adapter_env_dir(adapter: str) -> Path:
    """Resolve the isolated environment declared by the environment manager."""
    try:
        from bench import env as benchmark_env

        profile = benchmark_env.load_runtime_profiles().get(adapter)
        if profile is not None:
            return benchmark_env.env_dir_for_profile(profile)
    except Exception:
        pass
    return repo_root() / ".venvs" / "adapters" / adapter


def _safe_float(value: Any) -> Optional[float]:
    try:
        converted = float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None
    if converted is not None and not math.isfinite(converted):
        return None
    return converted


def _finite_real_scalar(value: Any) -> Optional[float]:
    """Accept only an actual finite, non-boolean real scalar from JSON."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _auto_cpu_model() -> Optional[str]:
    model = str(platform.processor() or "").strip()
    generic_models = {"", "arm", "arm64", "i386", "x86_64", "amd64"}
    if platform.system() == "Darwin":
        # ``platform.processor()`` is commonly just ``arm`` on Apple Silicon.
        # Capture only the non-identifying chip line from the hardware report;
        # never persist the report, which also contains serial identifiers.
        try:
            process = subprocess.run(
                ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if process.returncode == 0:
                for line in str(process.stdout or "").splitlines():
                    key, separator, value = line.strip().partition(":")
                    if separator and key == "Chip" and value.strip():
                        return value.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            process = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            value = str(process.stdout or "").strip()
            if process.returncode == 0 and value:
                return value
        except (OSError, subprocess.SubprocessError):
            pass
    if model.casefold() not in generic_models:
        return model
    fallback = str(platform.machine() or "").strip()
    return fallback or None


def _cpu_frequency_ghz(value_mhz: Any) -> Optional[float]:
    """Return a plausible GHz value or None for unavailable/bogus probes."""

    value = _safe_float(value_mhz)
    if value is None or value <= 0:
        return None
    value /= 1000.0
    return value if 0.1 <= value <= 10.0 else None


def _machine_info(
    *,
    machine_id: Optional[str] = None,
    machine_label: Optional[str] = None,
    cpu_model: Optional[str] = None,
    cpu_base_ghz: Optional[float] = None,
    host_profile_id: Optional[str] = None,
    host_profile_sha256: Optional[str] = None,
    host_settings_json: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        cpu_freq = psutil.cpu_freq()
    except (OSError, RuntimeError):
        # Frequency probes are unavailable on some virtualized macOS runners
        # and on otherwise supported hosts. Hardware identity remains valid
        # without this optional observation.
        cpu_freq = None
    cpu_current_ghz = None
    cpu_max_ghz = None
    if cpu_freq:
        cpu_current_ghz = _cpu_frequency_ghz(cpu_freq.current)
        cpu_max_ghz = _cpu_frequency_ghz(cpu_freq.max)

    resolved_cpu_base_ghz = _safe_float(cpu_base_ghz)
    if resolved_cpu_base_ghz is not None and not (0.1 <= resolved_cpu_base_ghz <= 10.0):
        raise ValueError("cpu_base_ghz must be between 0.1 and 10.0 GHz")
    if resolved_cpu_base_ghz is None:
        resolved_cpu_base_ghz = cpu_max_ghz or cpu_current_ghz

    resolved_cpu_model = str(cpu_model or "").strip() or _auto_cpu_model()
    raw_machine_id = str(machine_id or "").strip()
    if raw_machine_id:
        resolved_machine_id = raw_machine_id
    else:
        # Preserve cross-run identity without publishing the local hostname.
        private_identity = socket.gethostname().encode("utf-8")
        resolved_machine_id = (
            "anonymous-" + hashlib.sha256(private_identity).hexdigest()[:16]
        )
    resolved_machine_label = (
        str(machine_label or "").strip()
        or str(resolved_cpu_model or "").strip()
        or str(platform.machine() or "unknown")
    )
    logical_cores = psutil.cpu_count(logical=True)
    if logical_cores is None:
        logical_cores = os.cpu_count()

    resolved_profile_id = str(host_profile_id or "").strip() or None
    resolved_profile_sha256 = str(host_profile_sha256 or "").strip().lower() or None
    if resolved_profile_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", resolved_profile_sha256
    ):
        raise ValueError("host_profile_sha256 must be a lowercase SHA-256 value")
    if (resolved_profile_id is None) != (resolved_profile_sha256 is None):
        raise ValueError("host profile ID and SHA-256 must be provided together")
    host_settings: Dict[str, Any] | None = None
    if host_settings_json is not None:
        try:
            decoded_settings = json.loads(host_settings_json)
        except json.JSONDecodeError as exc:
            raise ValueError("host_settings_json must contain valid JSON") from exc
        if not isinstance(decoded_settings, dict):
            raise ValueError("host_settings_json must contain a JSON object")
        host_settings = dict(decoded_settings)
    if resolved_profile_id is not None and host_settings is None:
        raise ValueError("a frozen host profile requires host settings")

    return {
        "machine_id": resolved_machine_id,
        "machine_label": resolved_machine_label,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": resolved_cpu_model,
        "cpu_base_ghz": resolved_cpu_base_ghz,
        "cpu_current_ghz": cpu_current_ghz,
        "cpu_max_ghz": cpu_max_ghz,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": logical_cores,
        "memory_total_bytes": psutil.virtual_memory().total,
        "host_profile_id": resolved_profile_id,
        "host_profile_sha256": resolved_profile_sha256,
        "host_settings": host_settings,
    }


def _benchmark_machine_identity(machine: Mapping[str, Any]) -> Dict[str, Any]:
    """Select stable machine properties that make performance results comparable."""
    fields = (
        "machine_id",
        "machine_label",
        "platform",
        "platform_release",
        "machine",
        "python_version",
        "cpu_model",
        "cpu_count_physical",
        "cpu_count_logical",
        "memory_total_bytes",
        "host_profile_id",
        "host_profile_sha256",
        "host_settings",
    )
    identity = {field: machine.get(field) for field in fields}
    host_settings = identity.get("host_settings")
    if isinstance(host_settings, Mapping):
        # Power observations are session provenance, not protocol identity.  A
        # mode change must be visible in results without making a safe resume
        # impossible.  Operator-frozen settings remain fingerprint-bound.
        dynamic_observations = {
            "battery_life_percent",
            "battery_saver",
            "energy_mode",
            "pmset_lowpowermode",
            "energy_mode_observation_status",
            "power_scheme_guid",
            "power_scheme_name",
            "power_mode_tag",
            "power_state_probe_errors",
            "sleep_prevention_active",
        }
        identity["host_settings"] = {
            key: value
            for key, value in host_settings.items()
            if key not in dynamic_observations
        }
    return identity


def _record_host_observation(
    report_dir: Path,
    *,
    machine: Mapping[str, Any],
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    """Append one non-gating host/power observation for this controller session."""

    path = report_dir / "host_observations.json"
    observations: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunIntegrityError("host observation history is malformed") from exc
        if not isinstance(existing, list):
            raise RunIntegrityError("host observation history must be a JSON list")
        observations = [dict(item) for item in existing if isinstance(item, Mapping)]
        if len(observations) != len(existing):
            raise RunIntegrityError("host observation history contains a non-object")
    observation = {
        "session_index": len(observations) + 1,
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_fingerprint": run_fingerprint,
        "machine_id": machine.get("machine_id"),
        "host_profile_id": machine.get("host_profile_id"),
        "host_settings": machine.get("host_settings"),
    }
    observations.append(observation)
    atomic_write_json(path, observations)
    return observations


def _task_power_provenance(
    session_observation: Mapping[str, Any],
    start: Mapping[str, Any],
    end: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Bind live before/after power observations to one terminal task."""

    end = start if end is None else end
    start_tag = str(start.get("power_mode_tag") or "power-mode-unavailable")
    end_tag = str(end.get("power_mode_tag") or "power-mode-unavailable")
    changed = start_tag != end_tag

    def stable_value(field: str) -> Any:
        left = start.get(field)
        right = end.get(field)
        return left if left == right else "mixed"

    probe_errors = list(
        dict.fromkeys(
            [f"start:{value}" for value in list(start.get("probe_errors") or [])]
            + [f"end:{value}" for value in list(end.get("probe_errors") or [])]
        )
    )
    return {
        "host_session_index": _safe_int(session_observation.get("session_index")),
        "host_session_observed_at_utc": session_observation.get("observed_at_utc"),
        "host_power_observation_scope": "immediately_before_and_after_task",
        "host_power_start_observed_at_utc": start.get("observed_at_utc"),
        "host_power_end_observed_at_utc": end.get("observed_at_utc"),
        "host_power_start_mode_tag": start_tag,
        "host_power_end_mode_tag": end_tag,
        "host_power_start_probe_source": start.get("probe_source"),
        "host_power_end_probe_source": end.get("probe_source"),
        "host_power_start_probe_attempts": start.get("probe_attempts"),
        "host_power_end_probe_attempts": end.get("probe_attempts"),
        "host_power_mode_changed_during_task": changed,
        "host_power_mode_tag": "mixed-within-task" if changed else start_tag,
        "host_energy_mode": stable_value("energy_mode"),
        "host_energy_mode_observation_status": stable_value(
            "energy_mode_observation_status"
        ),
        "host_power_source": stable_value("power_source"),
        "host_pmset_lowpowermode": stable_value("pmset_lowpowermode"),
        "host_power_scheme_guid": stable_value("power_scheme_guid"),
        "host_power_scheme_name": stable_value("power_scheme_name"),
        "host_battery_saver": stable_value("battery_saver"),
        "host_battery_life_percent": stable_value("battery_life_percent"),
        "host_power_probe_errors": probe_errors,
        "host_power_probe_diagnostics": {
            "start": start.get("probe_diagnostics"),
            "end": end.get("probe_diagnostics"),
        },
    }


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _rss_bytes(process: psutil.Process) -> int:
    try:
        rss = process.memory_info().rss
    except Exception:
        rss = 0
    try:
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except Exception:
                continue
    except Exception:
        pass
    return int(rss)


def _safe_cpu_time(process: psutil.Process) -> Optional[float]:
    """Return sampled CPU time for the adapter process and its descendants."""

    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.Error, OSError):
        pass
    total = 0.0
    observed = False
    seen: set[int] = set()
    for member in processes:
        try:
            if member.pid in seen:
                continue
            seen.add(member.pid)
            times = member.cpu_times()
        except (psutil.Error, OSError):
            continue
        total += float(times.user + times.system)
        observed = True
    return total if observed else None


class AdapterProcessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        returncode: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class AdapterTimeout(TimeoutError):
    def __init__(
        self,
        adapter: str,
        timeout_seconds: float,
        elapsed_seconds: float,
        pid: int,
        phase: str = "calculation",
        partial_duration_samples_sec: Sequence[float] = (),
        partial_cpu_time_samples_sec: Sequence[float] = (),
    ):
        super().__init__(
            f"Adapter {adapter} timed out during {phase} after "
            f"{timeout_seconds} seconds"
        )
        self.adapter = adapter
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.pid = pid
        self.phase = phase
        self.partial_duration_samples_sec = tuple(partial_duration_samples_sec)
        self.partial_cpu_time_samples_sec = tuple(partial_cpu_time_samples_sec)


class AdapterInterrupted(InterruptedError):
    def __init__(
        self,
        adapter: str,
        elapsed_seconds: float,
        pid: int,
        *,
        partial_duration_samples_sec: Sequence[float] = (),
        partial_cpu_time_samples_sec: Sequence[float] = (),
    ):
        super().__init__(f"Adapter {adapter} was interrupted")
        self.adapter = adapter
        self.elapsed_seconds = elapsed_seconds
        self.pid = pid
        self.partial_duration_samples_sec = tuple(partial_duration_samples_sec)
        self.partial_cpu_time_samples_sec = tuple(partial_cpu_time_samples_sec)


class UnsupportedTaskError(RuntimeError):
    pass


def _terminate_process_tree(
    process: subprocess.Popen, grace_seconds: float = 3.0
) -> None:
    """Terminate an adapter's entire isolated process group and reap its parent."""
    if process.poll() is not None:
        try:
            process.wait(timeout=0)
        except Exception:
            pass
        return

    try:
        ps_process = psutil.Process(process.pid)
        descendants = ps_process.children(recursive=True)
    except (psutil.Error, OSError):
        ps_process = None
        descendants = []

    signalled_group = False
    if os.name != "nt":
        try:
            process_group = os.getpgid(process.pid)
            if process_group != os.getpgrp():
                os.killpg(process_group, signal.SIGTERM)
                signalled_group = True
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if not signalled_group:
        for child in reversed(descendants):
            try:
                child.terminate()
            except (psutil.Error, OSError):
                pass
        try:
            process.terminate()
        except (ProcessLookupError, OSError):
            pass

    try:
        process.wait(timeout=max(0.0, grace_seconds))
    except subprocess.TimeoutExpired:
        if signalled_group and os.name != "nt":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        for child in reversed(descendants):
            try:
                child.kill()
            except (psutil.Error, OSError):
                pass
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=max(1.0, grace_seconds))
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass

    if ps_process is not None:
        try:
            remaining = [child for child in descendants if child.is_running()]
            _, alive = psutil.wait_procs(remaining, timeout=max(0.0, grace_seconds))
            for child in alive:
                try:
                    child.kill()
                except (psutil.Error, OSError):
                    pass
        except (psutil.Error, OSError):
            pass


def _run_process_command(
    command: Sequence[str],
    *,
    adapter_name: str,
    environment: Optional[Mapping[str, str]] = None,
    sample_interval: float = 0.05,
    timeout: Optional[float] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    termination_grace: float = 3.0,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    progress_interval: float = 30.0,
) -> Tuple[str, str, Dict[str, Any]]:
    creation_flags = 0
    popen_kwargs: Dict[str, Any] = {"start_new_session": True}
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        popen_kwargs = {"creationflags": creation_flags}

    started = time.perf_counter()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(environment) if environment is not None else None,
        **popen_kwargs,
    )
    try:
        ps_process = psutil.Process(process.pid)
    except psutil.Error:
        ps_process = None

    peak_rss = 0
    calculation_peak_rss = 0
    worker_ready_rss = None
    event_count = 0
    in_calculation = False
    worker_ready_at: Optional[float] = None
    calculation_started_at: Optional[float] = None
    current_iteration: Optional[int] = None
    completed_iterations = 0
    partial_duration_samples: List[float] = []
    partial_cpu_time_samples: List[float] = []
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    state_lock = threading.Lock()

    def sample_rss() -> int:
        return _rss_bytes(ps_process) if ps_process is not None else 0

    def read_stdout() -> None:
        nonlocal calculation_peak_rss, event_count, in_calculation, peak_rss
        nonlocal worker_ready_rss
        nonlocal worker_ready_at, calculation_started_at, current_iteration
        nonlocal completed_iterations
        assert process.stdout is not None
        for line in process.stdout:
            stdout_lines.append(line)
            stripped = line.strip()
            if not stripped.startswith(BENCHMARK_EVENT_PREFIX):
                continue
            try:
                event = json.loads(stripped[len(BENCHMARK_EVENT_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            controller_rss = sample_rss()
            worker_rss = _safe_int(event.get("worker_self_peak_rss_bytes")) or 0
            rss = max(controller_rss, worker_rss)
            with state_lock:
                # Event-boundary samples can observe a short-lived RSS peak
                # that the controller's periodic polling thread misses. Keep
                # the overall peak consistent with every accepted sample.
                peak_rss = max(peak_rss, rss)
                event_count += 1
                event_name = str(event.get("event") or "")
                if event_name == "worker_ready":
                    worker_ready_rss = rss or worker_ready_rss
                    worker_ready_at = time.perf_counter()
                elif event_name == "calculation_start":
                    in_calculation = True
                    calculation_started_at = time.perf_counter()
                    current_iteration = int(event.get("iteration") or 0) or None
                    calculation_peak_rss = max(calculation_peak_rss, rss)
                elif event_name == "calculation_complete":
                    calculation_peak_rss = max(calculation_peak_rss, rss)
                    in_calculation = False
                    completed_iterations = max(
                        completed_iterations, int(event.get("iteration") or 0)
                    )
                    duration_sample = _safe_float(event.get("calculation_sec"))
                    cpu_sample = _safe_float(event.get("cpu_time_sec"))
                    if duration_sample is not None:
                        partial_duration_samples.append(duration_sample)
                    if cpu_sample is not None:
                        partial_cpu_time_samples.append(cpu_sample)
                    calculation_started_at = None

    def partial_samples() -> tuple[tuple[float, ...], tuple[float, ...]]:
        with state_lock:
            return tuple(partial_duration_samples), tuple(partial_cpu_time_samples)

    def read_stderr() -> None:
        assert process.stderr is not None
        stderr_lines.extend(process.stderr)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    last_cpu = None
    last_progress = started
    try:
        while process.poll() is None:
            if ps_process is not None:
                sampled_rss = _rss_bytes(ps_process)
                with state_lock:
                    peak_rss = max(peak_rss, sampled_rss)
                    if in_calculation:
                        calculation_peak_rss = max(calculation_peak_rss, sampled_rss)
                sampled_cpu = _safe_cpu_time(ps_process)
                if sampled_cpu is not None:
                    last_cpu = sampled_cpu

            now = time.perf_counter()
            elapsed = now - started
            if stop_requested is not None and stop_requested():
                partial_wall, partial_cpu = partial_samples()
                raise AdapterInterrupted(
                    adapter_name,
                    elapsed,
                    process.pid,
                    partial_duration_samples_sec=partial_wall,
                    partial_cpu_time_samples_sec=partial_cpu,
                )
            with state_lock:
                ready_elapsed = (
                    None if worker_ready_at is None else now - worker_ready_at
                )
                iteration_elapsed = (
                    None
                    if calculation_started_at is None
                    else now - calculation_started_at
                )
                snapshot = {
                    "host_elapsed_sec": elapsed,
                    "ready_elapsed_sec": ready_elapsed,
                    "iteration_elapsed_sec": iteration_elapsed,
                    "current_iteration": current_iteration,
                    "completed_iterations": completed_iterations,
                    "phase": (
                        "calculation"
                        if in_calculation
                        else "prepared_between_calls"
                        if worker_ready_at is not None
                        else "startup_or_warmup"
                    ),
                }
            if (
                timeout is not None
                and ready_elapsed is not None
                and ready_elapsed >= timeout
            ):
                partial_wall, partial_cpu = partial_samples()
                raise AdapterTimeout(
                    adapter_name,
                    timeout,
                    ready_elapsed,
                    process.pid,
                    phase="prepared_calculation_region",
                    partial_duration_samples_sec=partial_wall,
                    partial_cpu_time_samples_sec=partial_cpu,
                )
            if timeout is not None and ready_elapsed is None and elapsed >= timeout:
                # Operational safety for import/preparation/warm-up hangs. The
                # progress phase makes clear that this is not a calculation
                # timeout and the record remains censored.
                partial_wall, partial_cpu = partial_samples()
                raise AdapterTimeout(
                    adapter_name,
                    timeout,
                    elapsed,
                    process.pid,
                    phase="startup_or_warmup",
                    partial_duration_samples_sec=partial_wall,
                    partial_cpu_time_samples_sec=partial_cpu,
                )
            if (
                progress_callback is not None
                and progress_interval > 0
                and now - last_progress >= progress_interval
            ):
                progress_callback(snapshot)
                last_progress = now
            time.sleep(max(0.001, sample_interval))

        process.wait()
        stdout_thread.join(timeout=1.0)
        stderr_thread.join(timeout=1.0)
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
    except KeyboardInterrupt as exc:
        elapsed = time.perf_counter() - started
        _terminate_process_tree(process, termination_grace)
        stdout_thread.join(timeout=0.2)
        stderr_thread.join(timeout=0.2)
        partial_wall, partial_cpu = partial_samples()
        raise AdapterInterrupted(
            adapter_name,
            elapsed,
            process.pid,
            partial_duration_samples_sec=partial_wall,
            partial_cpu_time_samples_sec=partial_cpu,
        ) from exc
    except BaseException:
        _terminate_process_tree(process, termination_grace)
        stdout_thread.join(timeout=0.2)
        stderr_thread.join(timeout=0.2)
        raise

    elapsed = time.perf_counter() - started
    phase_observation_status = (
        "complete"
        if worker_ready_rss and calculation_peak_rss
        else "partial"
        if worker_ready_rss or calculation_peak_rss
        else "unavailable"
    )
    host_metrics = {
        "host_wall_time_sec": elapsed,
        "host_peak_rss_bytes": peak_rss,
        "worker_ready_rss_bytes": worker_ready_rss,
        "calculation_peak_rss_bytes": calculation_peak_rss or None,
        "incremental_calculation_peak_rss_bytes": (
            max(0, calculation_peak_rss - worker_ready_rss)
            if worker_ready_rss is not None and calculation_peak_rss > 0
            else None
        ),
        "host_cpu_time_sec": last_cpu,
        "adapter_event_count": event_count,
        "adapter_stderr": stderr,
        "memory_phase_observation_status": phase_observation_status,
    }
    if process.returncode != 0:
        raise AdapterProcessError(
            f"Adapter {adapter_name} failed with code {process.returncode}: {stderr}\n{stdout}",
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return stdout, stderr, host_metrics


def run_adapter_process(
    name: str,
    *,
    image: str,
    mask: str,
    image_sha256: Optional[str] = None,
    source_image_sha256: Optional[str] = None,
    mask_sha256: Optional[str] = None,
    input_contract: str = HARMONIZED_INPUT_CONTRACT,
    input_representation_id: str = "original_continuous_image",
    representation_derivation_sha256: Optional[str] = None,
    configured_levels: Optional[int] = None,
    occupied_levels: Optional[int] = None,
    modality: Optional[str] = None,
    discretization: str = "fbn",
    aggregation: str = REQUIRED_AGGREGATION,
    bins: int = 32,
    bin_width: float = 32.0,
    intensity_min: Optional[float] = None,
    intensity_max: Optional[float] = None,
    sample_interval: float = 0.05,
    timeout: Optional[float] = None,
    families: Optional[List[str]] = None,
    benchmark_workload: Optional[str] = None,
    iterations: int = 5,
    include_values: bool = False,
    include_ibsi_codes: Optional[Sequence[str]] = None,
    timed: bool = True,
    stop_requested: Optional[Callable[[], bool]] = None,
    termination_grace: float = 3.0,
    progress_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
    progress_interval: float = 30.0,
    thread_environment: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if families:
        try:
            from bench.adapters.registry import get_adapter

            capabilities = get_adapter(name)
            if benchmark_workload and not capabilities.supports_workload(
                benchmark_workload
            ):
                raise UnsupportedTaskError(
                    f"{name} does not declare support for workload {benchmark_workload}"
                )
            unsupported = [
                family for family in families if not capabilities.supports(family)
            ]
            supported = [family for family in families if family not in unsupported]
            if not supported:
                raise UnsupportedTaskError(
                    f"{name} does not declare support for: {', '.join(unsupported)}"
                )
            unsupported_aggregation = [
                family
                for family in supported
                if not supports_aggregation(name, aggregation, [family])
            ]
            if unsupported_aggregation:
                raise UnsupportedTaskError(
                    f"{name} cannot calculate {aggregation} for: "
                    + ", ".join(unsupported_aggregation)
                )
        except ValueError:
            # Allow external/custom adapters that are not in the built-in registry.
            pass
    env_dir = _adapter_env_dir(name)
    python = _env_python(env_dir)
    if not python.exists():
        raise RuntimeError(
            f"Venv python not found: {python}. Run environment setup first."
        )

    command = [
        str(python),
        "-m",
        f"bench.adapters.{name}_adapter",
        "--image",
        image,
        "--mask",
        mask,
        "--discretization",
        discretization,
        "--aggregation",
        aggregation,
        "--bins",
        str(bins),
        "--bin-width",
        str(bin_width),
        "--iterations",
        str(iterations),
    ]
    if timed:
        command.append("--timed")
    if name == "medimage":
        # MEDimage suppresses complete supported feature families in its
        # modality-derived ``arbitrary`` mode. The performance protocol fixes
        # the calculation mode so each case exercises the same native surface.
        command.extend(["--intensity-type", MEDIMAGE_BENCHMARK_INTENSITY_TYPE])
    if modality:
        command.extend(["--modality", modality])
    if image_sha256:
        command.extend(["--image-sha256", image_sha256])
    if source_image_sha256:
        command.extend(["--source-image-sha256", source_image_sha256])
    if mask_sha256:
        command.extend(["--mask-sha256", mask_sha256])
    command.extend(
        [
            "--input-contract",
            input_contract,
            "--input-representation-id",
            input_representation_id,
        ]
    )
    if representation_derivation_sha256:
        command.extend(
            ["--representation-derivation-sha256", representation_derivation_sha256]
        )
    if configured_levels is not None:
        command.extend(["--configured-levels", str(configured_levels)])
    if occupied_levels is not None:
        command.extend(["--occupied-levels", str(occupied_levels)])
    if intensity_min is not None and intensity_max is not None:
        command.extend(
            [
                "--intensity-min",
                str(intensity_min),
                "--intensity-max",
                str(intensity_max),
            ]
        )
    if include_values:
        command.append("--include-values")
    if include_ibsi_codes:
        command.extend(["--include-ibsi-codes", ",".join(include_ibsi_codes)])
    if families:
        command.extend(["--families", ",".join(families)])
    if benchmark_workload:
        command.extend(["--benchmark-workload", benchmark_workload])

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root())
    environment.update(
        dict(thread_environment)
        if thread_environment is not None
        else _benchmark_thread_environment()
    )
    environment.update(BENCHMARK_INITIALIZATION_ENV)
    stdout, stderr, host = _run_process_command(
        command,
        adapter_name=name,
        environment=environment,
        sample_interval=sample_interval,
        timeout=timeout,
        stop_requested=stop_requested,
        termination_grace=termination_grace,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )

    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError:
        payload = None
        for line in reversed(
            [line.strip() for line in stdout.splitlines() if line.strip()]
        ):
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise AdapterProcessError(
                f"Parse error from adapter {name}: {stdout}\n{stderr}",
                stdout=stdout,
                stderr=stderr,
            )
    if not isinstance(payload, dict):
        raise AdapterProcessError(f"Adapter {name} returned a non-object JSON payload")

    timing_value = payload.get("timing")
    if timed:
        if not isinstance(timing_value, dict):
            raise AdapterProcessError(
                f"Timed adapter {name} did not return a timing payload"
            )
        timing = timing_value
    else:
        if timing_value is not None:
            raise AdapterProcessError(
                f"Untimed adapter {name} unexpectedly returned a timing payload"
            )
        timing = {}
    host_wall = _safe_float(host.get("host_wall_time_sec")) or 0.0
    duration = _safe_float(timing.get("duration_sec")) if timed else host_wall
    cpu_time = (
        _safe_float(timing.get("cpu_time_sec"))
        if timed
        else (_safe_float(host.get("host_cpu_time_sec")) or host_wall)
    )
    peak_rss = int(host.get("host_peak_rss_bytes") or 0)

    metrics = {
        "duration_sec": duration,
        "duration_min_sec": _safe_float(timing.get("duration_min_sec")) or duration,
        "duration_mean_sec": _safe_float(timing.get("duration_mean_sec")) or duration,
        "duration_median_sec": _safe_float(timing.get("duration_median_sec"))
        or duration,
        "duration_std_sec": _safe_float(timing.get("duration_std_sec")) or 0.0,
        "duration_max_sec": _safe_float(timing.get("duration_max_sec")) or duration,
        "cpu_time_sec": cpu_time,
        "cpu_time_min_sec": _safe_float(timing.get("cpu_time_min_sec")) or cpu_time,
        "cpu_time_mean_sec": _safe_float(timing.get("cpu_time_mean_sec")) or cpu_time,
        "cpu_time_median_sec": _safe_float(timing.get("cpu_time_median_sec"))
        or cpu_time,
        "cpu_time_std_sec": _safe_float(timing.get("cpu_time_std_sec")) or 0.0,
        "cpu_time_max_sec": _safe_float(timing.get("cpu_time_max_sec")) or cpu_time,
        "measured_iterations": _safe_int(timing.get("measured_iterations")) or 0,
        "measured_observations": _safe_int(timing.get("measured_observations")) or 0,
        "warmup_iterations": _safe_int(timing.get("warmup_iterations")) or 0,
        "total_iterations": _safe_int(timing.get("total_iterations")) or 0,
        "calls_per_observation": _safe_int(timing.get("calls_per_observation")) or 0,
        "calibration_calls": _safe_int(timing.get("calibration_calls")) or 0,
        "calibration_rounds": _safe_int(timing.get("calibration_rounds")) or 0,
        "calibration_duration_sec": _safe_float(timing.get("calibration_duration_sec"))
        or 0.0,
        "calibration_cpu_time_sec": _safe_float(timing.get("calibration_cpu_time_sec"))
        or 0.0,
        "calibration_per_call_sec": _safe_float(timing.get("calibration_per_call_sec")),
        "calibration_headroom_factor": _safe_float(
            timing.get("calibration_headroom_factor")
        ),
        "calibration_stability_cv": _safe_float(timing.get("calibration_stability_cv")),
        "calibration_stability_span": _safe_float(
            timing.get("calibration_stability_span")
        ),
        "calibration_stable": timing.get("calibration_stable"),
        "minimum_observation_window_sec": _safe_float(
            timing.get("minimum_observation_window_sec")
        ),
        "result_equivalence_checks": _safe_int(timing.get("result_equivalence_checks"))
        or 0,
        "result_equivalence_passed": timing.get("result_equivalence_passed"),
        "result_equivalence_rtol": _safe_float(timing.get("result_equivalence_rtol")),
        "result_equivalence_atol": _safe_float(timing.get("result_equivalence_atol")),
        "measured_calculation_calls": _safe_int(
            timing.get("measured_calculation_calls")
        )
        or 0,
        "total_calculation_calls": _safe_int(timing.get("total_calculation_calls"))
        or 0,
        "peak_rss_bytes": peak_rss,
        "host_peak_rss_bytes": peak_rss,
        "worker_ready_rss_bytes": _safe_int(host.get("worker_ready_rss_bytes")),
        "calculation_peak_rss_bytes": _safe_int(host.get("calculation_peak_rss_bytes")),
        "incremental_calculation_peak_rss_bytes": _safe_int(
            host.get("incremental_calculation_peak_rss_bytes")
        ),
        "host_wall_time_sec": host_wall,
        "adapter_event_count": _safe_int(host.get("adapter_event_count")) or 0,
        "adapter_stderr": str(host.get("adapter_stderr") or ""),
        "memory_phase_observation_status": str(
            host.get("memory_phase_observation_status") or "unavailable"
        ),
        "timing_source": "adapter_payload" if timed else "not_timed",
        "timing_scope": "adapter_internal_feature_compute"
        if timed
        else "not_applicable",
        "memory_scope": (
            "host_process_tree_polling_with_worker_reported_event_peak_fallback"
        ),
    }
    return payload, metrics


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        raise RuntimeError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _issue_row(
    *,
    run_id: str,
    record: Dict[str, Any],
    issue_type: str,
    severity: str,
    metric: str,
    value: Any,
    details: str,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": record.get("case_id"),
        "dataset": record.get("dataset"),
        "size": record.get("size"),
        "variant": record.get("variant"),
        "mask_id": record.get("mask_id"),
        "adapter": record.get("adapter"),
        "workload": record.get("workload"),
        "requested_families": record.get("requested_families"),
        "repeat": record.get("repeat"),
        "family": family,
        "host_session_index": record.get("host_session_index"),
        "host_power_observation_scope": record.get("host_power_observation_scope"),
        "host_power_start_mode_tag": record.get("host_power_start_mode_tag"),
        "host_power_end_mode_tag": record.get("host_power_end_mode_tag"),
        "host_power_mode_changed_during_task": record.get(
            "host_power_mode_changed_during_task"
        ),
        "host_power_mode_tag": record.get("host_power_mode_tag"),
        "host_energy_mode": record.get("host_energy_mode"),
        "host_energy_mode_observation_status": record.get(
            "host_energy_mode_observation_status"
        ),
        "issue_type": issue_type,
        "severity": severity,
        "metric": metric,
        "value": value,
        "details": details,
    }


def run_qc_checks(run_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    evaluated = [
        record
        for record in records
        if record.get("task_status") == STATUS_MEASURED and record.get("success")
    ]

    for record in records:
        status = str(record.get("task_status") or "").strip().lower()
        if status == STATUS_FAILED:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="calculation_failed",
                    severity="error",
                    metric="task_status",
                    value=status,
                    details=str(record.get("error") or "calculation task failed"),
                    family=str(record.get("family") or "") or None,
                )
            )
        elif status == STATUS_TIMED_OUT:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="calculation_timed_out",
                    severity="warning",
                    metric="censor_lower_bound_sec",
                    value=record.get("censor_lower_bound_sec"),
                    details="Task is censored and is not a measured runtime.",
                    family=str(record.get("family") or "") or None,
                )
            )
        elif status == STATUS_INTERRUPTED:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="calculation_interrupted",
                    severity="warning",
                    metric="task_status",
                    value=status,
                    details=(
                        "Task was interrupted before commit and remains resumable; "
                        "it is not a measured runtime."
                    ),
                    family=str(record.get("family") or "") or None,
                )
            )
        elif status == STATUS_SKIPPED:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="performance_policy_skip",
                    severity="warning",
                    metric="task_status",
                    value=status,
                    details=str(
                        record.get("policy_reason")
                        or "Task was excluded by an explicitly enabled policy."
                    ),
                    family=str(record.get("family") or "") or None,
                )
            )
        elif status == STATUS_SKIPPED_TIMEOUT:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="timeout_cutoff_skip",
                    severity="warning",
                    metric="timeout_cutoff_complexity",
                    value=record.get("timeout_cutoff_complexity"),
                    details=(
                        "Task was not launched because a smaller image in the same "
                        "adapter, workload, mask, and input-configuration scaling "
                        "series reached the timeout."
                    ),
                    family=str(record.get("family") or "") or None,
                )
            )

    for record in evaluated:
        duration = _safe_float(record.get("duration_sec"))
        if duration is None or duration <= 0:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="non_positive_duration",
                    severity="error",
                    metric="duration_sec",
                    value=record.get("duration_sec"),
                    details="Measured duration must be positive and finite.",
                )
            )
        duration_mean = _safe_float(record.get("duration_mean_sec"))
        duration_std = _safe_float(record.get("duration_std_sec"))
        if duration_mean and duration_std is not None:
            coefficient_of_variation = duration_std / duration_mean
            if coefficient_of_variation > 0.10:
                issues.append(
                    _issue_row(
                        run_id=run_id,
                        record=record,
                        issue_type="unstable_within_process_timing",
                        severity="warning",
                        metric="duration_coefficient_of_variation",
                        value=coefficient_of_variation,
                        details=(
                            "Within-process timing CV exceeds the reviewed 10% "
                            "diagnostic threshold."
                        ),
                        family=str(record.get("family") or "") or None,
                    )
                )
        stderr = str(record.get("adapter_stderr") or "").strip()
        if stderr:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="adapter_stderr",
                    severity="warning",
                    metric="adapter_stderr",
                    value=stderr[:500],
                    details="Adapter emitted stderr during a measured task.",
                    family=str(record.get("family") or "") or None,
                )
            )
        phase_status = str(record.get("memory_phase_observation_status") or "").strip()
        if phase_status and phase_status != "complete":
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="incomplete_phase_memory_observation",
                    severity="warning",
                    metric="memory_phase_observation_status",
                    value=phase_status,
                    details=(
                        "Calculation timing is valid, but phase-specific RSS was "
                        "not fully observable; process-tree peak RSS remains available."
                    ),
                    family=str(record.get("family") or "") or None,
                )
            )
        power_status = str(
            record.get("host_energy_mode_observation_status") or ""
        ).strip()
        if power_status != "observed":
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="task_power_mode_unavailable",
                    severity="warning",
                    metric="host_energy_mode_observation_status",
                    value=power_status or "unavailable",
                    details=(
                        "The task remains valid, but its live power mode could not "
                        "be classified at both task boundaries."
                    ),
                )
            )
        if record.get("host_power_mode_changed_during_task") is True:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="task_power_mode_changed",
                    severity="warning",
                    metric="host_power_mode_tag",
                    value=record.get("host_power_mode_tag"),
                    details=(
                        "The observed power-mode tag changed between the task start "
                        "and end; interpret this timing separately."
                    ),
                )
            )
        feature_count = _safe_int(record.get("feature_count"))
        if feature_count is None or feature_count <= 0:
            issues.append(
                _issue_row(
                    run_id=run_id,
                    record=record,
                    issue_type="empty_feature_result",
                    severity="error",
                    metric="feature_count",
                    value=record.get("feature_count"),
                    details="A measured task must contain at least one feature.",
                )
            )
    by_type = Counter(issue["issue_type"] for issue in issues)
    by_severity = Counter(issue["severity"] for issue in issues)
    by_workload = Counter(str(issue.get("workload") or "unknown") for issue in issues)
    by_adapter = Counter(str(issue.get("adapter") or "unknown") for issue in issues)
    return {
        "summary": {
            "record_count_total": len(records),
            "record_count_evaluated": len(evaluated),
            "task_status_counts": dict(
                Counter(str(record.get("task_status") or "") for record in records)
            ),
            "issue_count_total": len(issues),
            "issue_counts_by_type": dict(by_type),
            "issue_counts_by_severity": dict(by_severity),
            "issue_counts_by_workload": dict(by_workload),
            "issue_counts_by_adapter": dict(by_adapter),
            "has_error_issues": bool(by_severity.get("error", 0)),
        },
        "issues": issues,
    }


def _jsonable_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def save_summaries(
    report_dir: Path,
    run_id: str,
    records: List[Dict[str, Any]],
    machine: Dict[str, Any],
    args: argparse.Namespace,
    final: bool = False,
    *,
    ledger: Optional[BenchmarkLedger] = None,
    run_spec: Optional[RunSpec] = None,
    run_status: Optional[str] = None,
) -> None:
    qc = run_qc_checks(run_id, records)
    status_counts = ledger.status_counts() if ledger is not None else {}
    guardrail_decisions = ledger.guardrail_decisions() if ledger is not None else []
    timeout_cutoffs = ledger.timeout_cutoffs() if ledger is not None else []
    host_observations: list[dict[str, Any]] = []
    observation_path = report_dir / "host_observations.json"
    if observation_path.is_file():
        decoded_observations = json.loads(observation_path.read_text(encoding="utf-8"))
        if isinstance(decoded_observations, list):
            host_observations = decoded_observations
    power_mode_summary = summarize_task_power_records(records)
    meta = {
        "run_id": run_id,
        "run_fingerprint": run_spec.run_fingerprint if run_spec is not None else None,
        "run_status": run_status,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": _jsonable_parameters(args),
        "machine": machine,
        "host_observations": host_observations,
        "power_mode_summary": power_mode_summary,
        "status_counts": status_counts,
        "guardrail_decisions": guardrail_decisions,
        "timeout_cutoffs": timeout_cutoffs,
        "qc_summary": qc["summary"],
    }
    atomic_write_json(report_dir / "run_meta.json", meta)
    atomic_write_json(report_dir / "qc_summary.json", qc)

    csv_path = report_dir / "summary.csv"
    if records:
        first_keys = list(records[0])
        extra_keys = sorted(
            {key for record in records for key in record if key not in first_keys}
        )
        fieldnames = first_keys + extra_keys
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})
        atomic_write_text(csv_path, output.getvalue())

    if final:
        print(
            "QC completed: "
            f"{qc['summary']['issue_count_total']} issues flagged "
            f"({qc['summary']['issue_counts_by_severity'].get('error', 0)} errors)."
        )


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    seen = set()
    result = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _validate_and_select_cases(
    dataset_path: Path,
    manifest: Mapping[str, Any],
    *,
    sizes: Optional[str],
    variants: Optional[str],
    masks: Optional[str],
    modalities: Optional[str],
    verify_hashes: bool,
) -> List[Dict[str, Any]]:
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("dataset manifest must contain a non-empty cases list")

    allowed_sizes = {int(value) for value in _parse_csv(sizes)} if sizes else None
    allowed_variants = (
        {int(value) for value in _parse_csv(variants)} if variants else None
    )
    allowed_masks = set(_parse_csv(masks)) if masks else None
    allowed_modalities = (
        {value.lower() for value in _parse_csv(modalities)} if modalities else None
    )
    selected: List[Dict[str, Any]] = []
    case_ids = set()
    verified_hashes: Dict[Path, str] = {}

    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("every manifest case must be an object")
        case_id = str(raw.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("every manifest case requires case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id in manifest: {case_id}")
        case_ids.add(case_id)

        image_voxels = _safe_int(raw.get("image_voxels"))
        explicit_complexity = _safe_int(raw.get("complexity"))
        complexity = explicit_complexity or image_voxels
        size = _safe_int(raw.get("size"))
        if size is None and image_voxels is not None and image_voxels > 0:
            size = max(1, int(round(image_voxels ** (1.0 / 3.0))))
        if size is None or size <= 0:
            raise ValueError(f"case {case_id} has invalid size")
        if image_voxels is None or image_voxels <= 0:
            raise ValueError(f"case {case_id} has invalid image_voxels")
        if complexity is None or complexity <= 0:
            raise ValueError(f"case {case_id} has invalid complexity")
        variant = _safe_int(raw.get("variant")) or 0
        mask_id = str(raw.get("mask_id") or "default")
        modality = str(raw.get("modality") or "").strip().lower()
        if allowed_sizes is not None and size not in allowed_sizes:
            continue
        if allowed_variants is not None and variant not in allowed_variants:
            continue
        if allowed_masks is not None and mask_id not in allowed_masks:
            continue
        if allowed_modalities is not None and modality not in allowed_modalities:
            continue

        image_relative = str(raw.get("image_path") or "")
        mask_relative = str(raw.get("mask_path") or "")
        if not image_relative or not mask_relative:
            raise ValueError(f"case {case_id} requires image_path and mask_path")
        image_path = contained_path(dataset_path, image_relative)
        mask_path = contained_path(dataset_path, mask_relative)
        for kind, path, expected in (
            ("image", image_path, raw.get("image_sha256")),
            ("mask", mask_path, raw.get("mask_sha256")),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"case {case_id} {kind} does not exist: {path}")
            if verify_hashes and expected:
                actual = verified_hashes.get(path)
                if actual is None:
                    actual = sha256_file(path)
                    verified_hashes[path] = actual
                if actual.lower() != str(expected).lower():
                    raise ValueError(
                        f"case {case_id} {kind} hash mismatch: expected {expected}, got {actual}"
                    )

        discrete_relative = str(raw.get("discrete_image_path") or "").strip()
        discrete_path = None
        if discrete_relative:
            discrete_path = contained_path(dataset_path, discrete_relative)
            expected = str(raw.get("discrete_image_sha256") or "").strip().lower()
            if len(expected) != 64:
                raise ValueError(f"case {case_id} has an invalid discrete image hash")
            if not discrete_path.is_file():
                raise FileNotFoundError(
                    f"case {case_id} discrete image does not exist: {discrete_path}"
                )
            if verify_hashes:
                actual = verified_hashes.get(discrete_path)
                if actual is None:
                    actual = sha256_file(discrete_path)
                    verified_hashes[discrete_path] = actual
                if actual.lower() != expected:
                    raise ValueError(
                        f"case {case_id} discrete image hash mismatch: "
                        f"expected {expected}, got {actual}"
                    )

        ivh_relative = str(raw.get("ivh_image_path") or "").strip()
        ivh_path = None
        if ivh_relative:
            ivh_path = contained_path(dataset_path, ivh_relative)
            expected = str(raw.get("ivh_image_sha256") or "").strip().lower()
            if len(expected) != 64:
                raise ValueError(f"case {case_id} has an invalid IVH image hash")
            if not ivh_path.is_file():
                raise FileNotFoundError(
                    f"case {case_id} IVH image does not exist: {ivh_path}"
                )
            if verify_hashes:
                actual = verified_hashes.get(ivh_path)
                if actual is None:
                    actual = sha256_file(ivh_path)
                    verified_hashes[ivh_path] = actual
                if actual.lower() != expected:
                    raise ValueError(
                        f"case {case_id} IVH image hash mismatch: "
                        f"expected {expected}, got {actual}"
                    )

        normalized = dict(raw)
        normalized.update(
            {
                "case_id": case_id,
                "modality": modality or None,
                "size": size,
                "variant": variant,
                "subject_id": str(raw.get("subject_id") or case_id),
                "mask_id": mask_id,
                "mask_label": str(raw.get("mask_label") or mask_id),
                "image_voxels": image_voxels,
                "mask_voxels": _safe_int(raw.get("mask_voxels")),
                "mask_fraction": _safe_float(raw.get("mask_fraction")),
                "image_abs": str(image_path),
                "discrete_image_abs": str(discrete_path) if discrete_path else None,
                "ivh_image_abs": str(ivh_path) if ivh_path else None,
                "mask_abs": str(mask_path),
                "image_sha256": str(raw.get("image_sha256") or "").lower(),
                "discrete_image_sha256": str(
                    raw.get("discrete_image_sha256") or ""
                ).lower(),
                "ivh_image_sha256": str(raw.get("ivh_image_sha256") or "").lower(),
                "mask_sha256": str(raw.get("mask_sha256") or "").lower(),
                "shape": tuple(int(value) for value in raw["shape"]),
                "spacing": tuple(float(value) for value in raw["spacing"]),
                "complexity": complexity,
            }
        )
        selected.append(normalized)

    if not selected:
        raise ValueError("no dataset cases match the requested filters")
    return sorted(
        selected,
        key=lambda case: (
            int(case["complexity"]),
            int(case["size"]),
            str(case["case_id"]),
        ),
    )


def _nifti_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise ValueError(f"benchmark input is not a NIfTI file: {path}")


def _stage_input_file(source: Path, destination: Path, expected_sha256: str) -> None:
    """Commit one content-addressed input snapshot after hashing copied bytes."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64:
        raise ValueError(f"invalid expected input SHA-256 for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected:
            raise RunIntegrityError(
                f"staged input does not match its content address: {destination}"
            )
        os.chmod(destination, 0o444)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=_nifti_suffix(destination),
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with (
            source.open("rb") as input_stream,
            os.fdopen(
                descriptor,
                "wb",
            ) as output_stream,
        ):
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                digest.update(chunk)
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise RunIntegrityError(
                f"source input changed while it was staged: expected {expected}, got {actual}"
            )
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _input_stat_identity(path: Path) -> Tuple[int, int, int, int]:
    status = path.stat()
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_size),
        int(status.st_mtime_ns),
    )


def _stage_selected_inputs(
    cases: Sequence[Mapping[str, Any]],
    stage_dir: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Tuple[int, int, int, int]]]:
    """Snapshot each unique selected NIfTI once and bind tasks to those bytes."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_by_key: Dict[Tuple[str, str], Path] = {}
    source_paths: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    identities: Dict[str, Tuple[int, int, int, int]] = {}
    staged_cases: List[Dict[str, Any]] = []

    for raw_case in cases:
        case = dict(raw_case)
        roles = ["image", "mask"]
        if case.get("discrete_image_abs"):
            roles.append("discrete_image")
        if case.get("ivh_image_abs"):
            roles.append("ivh_image")
        for role in roles:
            source = Path(str(case[f"{role}_abs"])).resolve()
            expected = str(case[f"{role}_sha256"]).lower()
            suffix = _nifti_suffix(source)
            key = (expected, suffix)
            destination = staged_by_key.get(key)
            if destination is None:
                destination = stage_dir / expected[:2] / expected / source.name
                _stage_input_file(source, destination, expected)
                staged_by_key[key] = destination
                identities[str(destination.resolve())] = _input_stat_identity(
                    destination
                )
            source_paths[key].add(str(source))
            case[f"{role}_abs"] = str(destination.resolve())
        staged_cases.append(case)

    files = []
    for key, destination in sorted(staged_by_key.items()):
        expected, _ = key
        files.append(
            {
                "path": str(destination.relative_to(stage_dir)),
                "sha256": expected,
                "bytes": destination.stat().st_size,
                "source_paths": sorted(source_paths[key]),
            }
        )
    atomic_write_json(
        stage_dir / "manifest.json",
        {
            "schema_version": 1,
            "policy": "content_addressed_read_only_run_local_snapshots",
            "files": files,
        },
    )
    return staged_cases, identities


def _verify_staged_task_inputs(
    task: TaskSpec,
    identities: Mapping[str, Tuple[int, int, int, int]],
) -> None:
    """Detect replacement or mutation of a staged input around adapter execution."""

    paths = [("image", task.image_path), ("mask", task.mask_path)]
    if task.source_image_path != task.image_path:
        paths.append(("source_image", task.source_image_path))
    for role, value in paths:
        path = Path(value).resolve()
        expected = identities.get(str(path))
        if expected is None:
            raise RunIntegrityError(
                f"task {role} is not bound to the input stage: {path}"
            )
        try:
            observed = _input_stat_identity(path)
        except OSError as exc:
            raise RunIntegrityError(
                f"staged task {role} is unavailable: {path}"
            ) from exc
        if observed != expected:
            raise RunIntegrityError(
                f"staged task {role} changed during the run: {path}"
            )


@lru_cache(maxsize=None)
def _nifti_uncompressed_bytes(path_value: str) -> int:
    """Return array bytes from a NIfTI header without loading the voxel array."""

    import nibabel as nib
    import numpy as np

    image = nib.load(path_value)
    return int(np.prod(image.shape, dtype=np.int64)) * int(
        np.dtype(image.get_data_dtype()).itemsize
    )


def _task_input_uncompressed_bytes(case: Mapping[str, Any], representation: Any) -> int:
    """Count only the selected image and binary mask read by an adapter."""

    try:
        return _nifti_uncompressed_bytes(
            str(representation.image_path)
        ) + _nifti_uncompressed_bytes(str(case["mask_abs"]))
    except (OSError, ValueError):
        # Direct unit tests may use path placeholders. Executable plans always
        # pass validated, staged NIfTIs and therefore take the header path.
        voxels = int(case["image_voxels"])
        return voxels * 9


def _ordered_adapters(adapters: Sequence[str], baseline_adapter: str) -> List[str]:
    result = list(adapters)
    if baseline_adapter in result:
        result.remove(baseline_adapter)
        result.insert(0, baseline_adapter)
    return result


def _validate_baseline_capabilities(
    baseline_adapter: str,
    selected_adapters: Sequence[str],
    workloads: Sequence[BenchmarkWorkload],
) -> None:
    """Require a selected comparison baseline to cover every scheduled workload."""

    if baseline_adapter not in selected_adapters:
        return
    try:
        from bench.adapters.registry import get_adapter

        capabilities = get_adapter(baseline_adapter)
    except ValueError as exc:
        raise ValueError(
            "the configured baseline adapter must have declared capabilities: "
            f"{baseline_adapter!r}"
        ) from exc

    families = families_for_workloads(workloads)
    unsupported_families = [
        family for family in families if not capabilities.supports(family)
    ]
    unsupported_workloads = [
        workload.name
        for workload in workloads
        if not capabilities.supports_workload(workload.name)
    ]
    unsupported_aggregation = [
        family
        for family in families
        if capabilities.supports(family)
        and not supports_aggregation(baseline_adapter, REQUIRED_AGGREGATION, [family])
    ]
    if unsupported_families or unsupported_workloads or unsupported_aggregation:
        details = []
        if unsupported_families:
            details.append("unsupported families: " + ", ".join(unsupported_families))
        if unsupported_workloads:
            details.append("unsupported workloads: " + ", ".join(unsupported_workloads))
        if unsupported_aggregation:
            details.append(
                f"no {REQUIRED_AGGREGATION} implementation: "
                + ", ".join(unsupported_aggregation)
            )
        raise ValueError(
            f"configured baseline adapter {baseline_adapter!r} cannot provide "
            "every scheduled workload (" + "; ".join(details) + ")"
        )


def _declared_unsupported_reason(task: TaskSpec) -> Optional[str]:
    """Return a deterministic built-in capability failure before preflight."""

    try:
        from bench.adapters.registry import get_adapter

        capabilities = get_adapter(task.adapter)
    except ValueError:
        # External adapters remain allowed and declare support through their
        # own protocol response.
        return None
    if not capabilities.supports_workload(task.workload):
        return f"{task.adapter} does not declare support for workload {task.workload}"
    supported = [
        family for family in task.scheduled_families if capabilities.supports(family)
    ]
    unsupported = [
        family for family in task.scheduled_families if family not in supported
    ]
    if not supported:
        return (
            f"{task.adapter} does not declare support for grouped workload "
            f"{task.workload}: {', '.join(unsupported)}"
        )
    unsupported_aggregation = [
        family
        for family in supported
        if not supports_aggregation(task.adapter, REQUIRED_AGGREGATION, [family])
    ]
    if unsupported_aggregation:
        return (
            f"{task.adapter} cannot calculate {REQUIRED_AGGREGATION} for grouped "
            f"workload {task.workload}: {', '.join(unsupported_aggregation)}"
        )
    return None


def build_task_plan(
    *,
    cases: Sequence[Mapping[str, Any]],
    dataset: str,
    adapters: Sequence[str],
    workloads: Sequence[BenchmarkWorkload],
    repeats: int,
    timing_observations: int,
    input_contract: str = HARMONIZED_INPUT_CONTRACT,
    endpoint_contract: Optional[BenchmarkContract] = None,
) -> List[TaskSpec]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if timing_observations < 1:
        raise ValueError("timing observations must be >= 1")
    if not adapters:
        raise ValueError("at least one adapter is required")

    workload_specs = list(workloads)
    if not workload_specs:
        raise ValueError("at least one grouped workload is required")

    tasks: List[TaskSpec] = []
    ordinal = 0
    adapter_count = len(adapters)
    # Repeat is the outer plan dimension so repeats 1..N keep identical task
    # ordinals when a reviewed run is later extended to N+1 or N+2.
    for repeat in range(1, repeats + 1):
        for case_index, case in enumerate(cases):
            for workload_index, workload in enumerate(workload_specs):
                representation = select_representation(
                    case,
                    workload.representation_family,
                    input_contract=input_contract,
                    default_bins=32,
                    default_bin_width=32.0,
                )
                # Rotate the canonical adapter order across exact comparison
                # blocks. Over successive repeats/cases/workloads, each adapter
                # occupies each execution position instead of permanently
                # assigning the baseline the coolest/earliest position.
                rotation = (
                    case_index * len(workload_specs) + workload_index + repeat - 1
                ) % adapter_count
                block_adapters = list(adapters[rotation:]) + list(adapters[:rotation])
                for adapter in block_adapters:
                    ordinal += 1
                    tasks.append(
                        TaskSpec(
                            ordinal=ordinal,
                            case_id=str(case["case_id"]),
                            dataset=dataset,
                            modality=str(case.get("modality") or "").strip() or None,
                            size=int(case["size"]),
                            variant=int(case["variant"]),
                            mask_id=str(case["mask_id"]),
                            mask_label=str(case["mask_label"]),
                            image_path=representation.image_path,
                            source_image_path=str(case["image_abs"]),
                            mask_path=str(case["mask_abs"]),
                            image_sha256=representation.image_sha256,
                            source_image_sha256=str(case["image_sha256"]),
                            mask_sha256=str(case["mask_sha256"]),
                            shape=tuple(int(value) for value in case["shape"]),
                            spacing=tuple(float(value) for value in case["spacing"]),
                            image_voxels=int(case["image_voxels"]),
                            mask_voxels=_safe_int(case.get("mask_voxels")),
                            mask_fraction=_safe_float(case.get("mask_fraction")),
                            complexity=int(case["complexity"]),
                            subject_id=str(case.get("subject_id") or case["case_id"]),
                            input_contract=input_contract,
                            representation_id=representation.representation_id,
                            representation_derivation_sha256=(
                                representation.derivation_sha256
                            ),
                            configured_levels=representation.configured_levels,
                            occupied_levels=representation.occupied_levels,
                            adapter=adapter,
                            workload=workload.name,
                            requested_families=workload.families,
                            repeat=repeat,
                            discretization=representation.discretization,
                            bins=representation.bins,
                            bin_width=representation.bin_width,
                            intensity_min=representation.intensity_min,
                            intensity_max=representation.intensity_max,
                            timing_observations=timing_observations,
                            endpoint_contract_id=(
                                endpoint_contract.contract_id
                                if endpoint_contract is not None
                                else None
                            ),
                            endpoint_contract_sha256=(
                                endpoint_contract.sha256
                                if endpoint_contract is not None
                                else None
                            ),
                            expected_feature_count=(
                                endpoint_contract.expected_workload_count(
                                    adapter, workload.name
                                )
                                if endpoint_contract is not None
                                else None
                            ),
                            input_uncompressed_bytes=_task_input_uncompressed_bytes(
                                case, representation
                            ),
                        )
                    )
    return tasks


def _adapter_environment_probe_script() -> str:
    """Return the isolated-runtime probe, including NumPy/BLAS build details."""

    return """
import contextlib
import importlib.metadata as metadata
import io
import json
import platform

packages = sorted(
    (str(distribution.metadata.get("Name") or "unknown"), str(distribution.version))
    for distribution in metadata.distributions()
)
numpy_config = {"available": False}
try:
    import numpy

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        numpy.show_config()
    numpy_config = {
        "available": True,
        "version": str(numpy.__version__),
        "show_config": output.getvalue().strip(),
    }
except Exception as exc:
    numpy_config = {
        "available": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }

print(
    json.dumps(
        {
            "python_version": platform.python_version(),
            "packages": packages,
            "numpy_config": numpy_config,
        },
        sort_keys=True,
    )
)
""".strip()


def _adapter_environment_snapshot(adapter: str) -> Dict[str, Any]:
    environment_dir = _adapter_env_dir(adapter)
    python = _env_python(environment_dir)
    if not python.exists():
        raise RuntimeError(
            f"adapter environment is missing for {adapter}: {python}; "
            "run `python -m bench.cli env create` before planning a benchmark"
        )
    script = _adapter_environment_probe_script()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=environment,
        )
        if result.returncode == 0:
            snapshot = json.loads(result.stdout)
            numpy_config = snapshot.get("numpy_config")
            if (
                not isinstance(numpy_config, Mapping)
                or numpy_config.get("available") is not True
                or not str(numpy_config.get("version") or "").strip()
                or not str(numpy_config.get("show_config") or "").strip()
            ):
                raise RuntimeError(
                    f"{adapter} environment did not provide an auditable NumPy/BLAS "
                    "build configuration"
                )
            snapshot["available"] = True
            snapshot["python"] = str(python)
            try:
                from bench import env as benchmark_env

                profile = benchmark_env.load_runtime_profiles()[adapter]
            except (KeyError, OSError, ValueError):
                profile = None
            if profile is not None:
                verified_environment = benchmark_env.verify_profile(
                    profile,
                    smoke=False,
                )
                installed = {
                    re.sub(r"[-_.]+", "-", str(name)).lower(): str(version)
                    for name, version in snapshot.get("packages", [])
                }
                distribution_key = re.sub(r"[-_.]+", "-", profile.distribution).lower()
                actual_version = installed.get(distribution_key)
                expected_metadata_version = profile.metadata_version or profile.version
                if actual_version != expected_metadata_version:
                    raise RuntimeError(
                        f"{adapter} environment contains {profile.distribution} "
                        f"{actual_version!r}; expected {expected_metadata_version!r}"
                    )
                snapshot.update(
                    {
                        "distribution": profile.distribution,
                        "distribution_version": actual_version,
                        "configured_release_version": profile.version,
                        "profile_python": profile.python,
                        "environment_freeze_sha256": verified_environment[
                            "freeze_sha256"
                        ],
                    }
                )
            environment_metadata = environment_dir / "environment.json"
            if environment_metadata.is_file():
                snapshot["environment_metadata_sha256"] = sha256_file(
                    environment_metadata
                )
            return snapshot
        raise RuntimeError(
            f"cannot inspect adapter environment {adapter}: "
            f"{result.stderr.strip() or f'exit {result.returncode}'}"
        )
    except Exception as exc:
        raise RuntimeError(
            f"cannot fingerprint adapter environment {adapter}: {exc}"
        ) from exc


def _adapter_environment_snapshots(adapters: Sequence[str]) -> Dict[str, Any]:
    return {adapter: _adapter_environment_snapshot(adapter) for adapter in adapters}


def _runtime_profiles_sha256() -> Optional[str]:
    profiles_dir = repo_root() / "configs" / "adapters"
    if profiles_dir.is_dir():
        values = {
            str(path.relative_to(repo_root())): sha256_file(path)
            for path in sorted(profiles_dir.glob("*.yaml"))
        }
        return fingerprint(values)
    return None


_EXECUTION_SOURCE_FILES = (
    "pyproject.toml",
    "poetry.lock",
    "bench/__init__.py",
    "bench/cli.py",
    "bench/run.py",
    "bench/env.py",
    "bench/benchmark_contract.py",
    "bench/benchmark_guardrails.py",
    "bench/benchmark_eta.py",
    "bench/benchmark_ledger.py",
    "bench/benchmark_memory.py",
    "bench/benchmark_models.py",
    "bench/benchmark_representations.py",
    "bench/benchmark_workloads.py",
    "bench/dataset_manifest.py",
    "bench/ibsi_families.py",
    "bench/ibsi_codes.py",
    "bench/ibsi_mapping.py",
    "bench/power_provenance.py",
    "bench/adapters/__init__.py",
    "bench/adapters/base.py",
    "bench/adapters/protocol.py",
    "bench/adapters/registry.py",
)


def _benchmark_execution_sources(adapters: Sequence[str]) -> Dict[str, str]:
    """Return the exact controller/adapter dependency hashes bound to a run.

    Reporting, plotting, compliance, and dataset-generation modules are
    deliberately outside this timing-execution boundary. Their independently
    recorded provenance can change without invalidating a resumable timing run.
    """

    relative_paths = set(_EXECUTION_SOURCE_FILES)
    relative_paths.update(
        f"bench/adapters/{adapter}_adapter.py" for adapter in adapters
    )
    sources: Dict[str, str] = {}
    for relative_path in sorted(relative_paths):
        path = repo_root() / relative_path
        if not path.is_file():
            raise RunIntegrityError(
                f"benchmark execution dependency is missing: {relative_path}"
            )
        sources[relative_path] = sha256_file(path)
    return sources


def _benchmark_sources_sha256(adapters: Sequence[str]) -> str:
    controller_environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": sorted(
            (
                str(distribution.metadata.get("Name") or "unknown"),
                str(distribution.version),
            )
            for distribution in importlib_metadata.distributions()
        ),
    }
    return fingerprint(
        {
            "execution_files": _benchmark_execution_sources(adapters),
            "controller_environment": controller_environment,
        }
    )


def _verify_execution_bindings(run_spec: RunSpec) -> None:
    """Detect live code/profile changes before they can mix task payloads."""

    observed_sources = _benchmark_sources_sha256(run_spec.adapters)
    if observed_sources != run_spec.benchmark_sources_sha256:
        raise RunIntegrityError(
            "benchmark execution sources changed after run planning; restore the "
            "bound sources and resume, or start a new run"
        )
    observed_profiles = _runtime_profiles_sha256()
    if observed_profiles != run_spec.runtime_profiles_sha256:
        raise RunIntegrityError(
            "adapter runtime profiles changed after run planning; restore the "
            "bound profiles and resume, or start a new run"
        )

    for adapter, environment in dict(run_spec.adapter_environments).items():
        if not isinstance(environment, Mapping):
            raise RunIntegrityError(
                f"run specification has invalid environment provenance for {adapter}"
            )
        expected = environment.get("environment_metadata_sha256")
        metadata_path = _adapter_env_dir(adapter) / "environment.json"
        if not expected or not metadata_path.is_file():
            raise RunIntegrityError(
                f"recorded adapter environment provenance is unavailable for {adapter}"
            )
        if sha256_file(metadata_path) != expected:
            raise RunIntegrityError(
                f"adapter environment provenance changed after planning for {adapter}; "
                "restore it and resume, or start a new run"
            )


def _record_template(task: TaskSpec, run_id: str, attempt: int = 0) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "task_id": task.task_id,
        "task_status": STATUS_PENDING,
        "attempt": attempt,
        "case_id": task.case_id,
        "dataset": task.dataset,
        "modality": task.modality,
        "subject_id": task.subject_id,
        "size": task.size,
        "variant": task.variant,
        "mask_id": task.mask_id,
        "mask_label": task.mask_label,
        "image_sha256": task.image_sha256,
        "source_image_sha256": task.source_image_sha256,
        "mask_sha256": task.mask_sha256,
        "input_contract": task.input_contract,
        "input_representation_id": task.representation_id,
        "representation_derivation_sha256": task.representation_derivation_sha256,
        "configured_levels": task.configured_levels,
        "occupied_levels": task.occupied_levels,
        "shape": list(task.shape),
        "spacing": list(task.spacing),
        "image_voxels": task.image_voxels,
        "mask_voxels": task.mask_voxels,
        "mask_fraction": task.mask_fraction,
        "complexity": task.complexity,
        "adapter": task.adapter,
        "workload": task.workload,
        "requested_families": list(task.scheduled_families),
        "guardrail_group": task.guardrail_group,
        "repeat": task.repeat,
        "discretization": task.discretization,
        "effective_bins": task.bins,
        "effective_bin_width": task.bin_width,
        "intensity_min": task.intensity_min,
        "intensity_max": task.intensity_max,
        "requested_timing_observations": task.timing_observations,
        "endpoint_contract_id": task.endpoint_contract_id,
        "endpoint_contract_sha256": task.endpoint_contract_sha256,
        "expected_feature_count": task.expected_feature_count,
        "input_uncompressed_bytes": task.input_uncompressed_bytes,
        "success": False,
        "duration_sec": None,
        "duration_min_sec": None,
        "duration_mean_sec": None,
        "duration_median_sec": None,
        "duration_std_sec": None,
        "duration_max_sec": None,
        "cpu_time_sec": None,
        "cpu_time_min_sec": None,
        "cpu_time_mean_sec": None,
        "cpu_time_median_sec": None,
        "cpu_time_std_sec": None,
        "cpu_time_max_sec": None,
        "measured_iterations": None,
        "measured_observations": None,
        "warmup_iterations": None,
        "total_iterations": None,
        "calls_per_observation": None,
        "calibration_calls": None,
        "calibration_rounds": None,
        "calibration_duration_sec": None,
        "calibration_cpu_time_sec": None,
        "calibration_per_call_sec": None,
        "calibration_headroom_factor": None,
        "calibration_stability_cv": None,
        "calibration_stability_span": None,
        "calibration_stable": None,
        "minimum_observation_window_sec": None,
        "result_equivalence_checks": None,
        "result_equivalence_passed": None,
        "result_equivalence_rtol": None,
        "result_equivalence_atol": None,
        "fresh_process_reference_task_id": None,
        "fresh_process_reference_repeat": None,
        "fresh_process_result_equivalence_passed": None,
        "fresh_process_result_equivalence_rtol": None,
        "fresh_process_result_equivalence_atol": None,
        "measured_calculation_calls": None,
        "total_calculation_calls": None,
        "peak_rss_bytes": None,
        "host_peak_rss_bytes": None,
        "worker_ready_rss_bytes": None,
        "calculation_peak_rss_bytes": None,
        "incremental_calculation_peak_rss_bytes": None,
        "host_wall_time_sec": None,
        "adapter_event_count": None,
        "adapter_stderr": None,
        "host_session_index": None,
        "host_session_observed_at_utc": None,
        "host_power_observation_scope": None,
        "host_power_start_observed_at_utc": None,
        "host_power_end_observed_at_utc": None,
        "host_power_start_mode_tag": None,
        "host_power_end_mode_tag": None,
        "host_power_start_probe_source": None,
        "host_power_end_probe_source": None,
        "host_power_start_probe_attempts": None,
        "host_power_end_probe_attempts": None,
        "host_power_mode_changed_during_task": None,
        "host_power_mode_tag": None,
        "host_energy_mode": None,
        "host_energy_mode_observation_status": None,
        "host_power_source": None,
        "host_pmset_lowpowermode": None,
        "host_power_probe_errors": None,
        "host_power_probe_diagnostics": None,
        "timing_source": None,
        "timing_scope": None,
        "memory_scope": None,
        "memory_phase_observation_status": None,
        "feature_count": None,
        "attempted_feature_count": None,
        "finite_feature_count": None,
        "censor_lower_bound_sec": None,
        "policy_reason": None,
        "timeout_cutoff_complexity": None,
        "timeout_cutoff_complexity_metric": None,
        "timeout_cutoff_evidence_task_id": None,
        "memory_preflight_policy_id": None,
        "memory_preflight_enabled": None,
        "memory_estimate_exceeds_budget": None,
        "memory_static_estimate_bytes": None,
        "memory_linear_static_estimate_bytes": None,
        "memory_quadratic_static_estimate_bytes": None,
        "memory_empirical_estimate_bytes": None,
        "memory_empirical_baseline_bytes": None,
        "memory_empirical_projected_increment_bytes": None,
        "memory_empirical_observation_count": None,
        "memory_empirical_same_scope_observation_count": None,
        "memory_empirical_growth_exponent": None,
        "memory_estimate_bytes": None,
        "memory_available_bytes": None,
        "memory_total_bytes": None,
        "memory_reserve_bytes": None,
        "memory_budget_bytes": None,
        "memory_preflight_decision": None,
        "error": None,
    }


def _terminal_record(
    task: TaskSpec,
    run_id: str,
    status: str,
    *,
    attempt: int = 0,
    error: Optional[str] = None,
    policy_reason: Optional[str] = None,
    censor_lower_bound_sec: Optional[float] = None,
) -> Dict[str, Any]:
    record = _record_template(task, run_id, attempt)
    record["task_status"] = status
    record["success"] = status == STATUS_MEASURED
    record["error"] = error
    record["policy_reason"] = policy_reason
    record["censor_lower_bound_sec"] = censor_lower_bound_sec
    return record


def _validate_measured_result(
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    expected_adapter: Optional[str] = None,
    expected_family: Optional[str] = None,
    expected_task: Optional[TaskSpec] = None,
    expected_software: Optional[Mapping[str, Any]] = None,
) -> int:
    if not isinstance(payload, Mapping):
        raise AdapterProcessError("adapter payload must be a JSON object")
    adapter_status = str(payload.get("status") or "").strip().lower()
    if adapter_status in {"unsupported", "not_supported", "not-supported"}:
        raise UnsupportedTaskError(
            str(payload.get("reason") or "adapter reports unsupported workload")
        )

    if payload.get("schema_version") != ADAPTER_PROTOCOL_VERSION:
        raise AdapterProcessError(
            f"adapter payload schema_version must be {ADAPTER_PROTOCOL_VERSION}"
        )
    if expected_task is not None:
        expected_adapter = expected_task.adapter
        expected_family = None

    adapter_name = str(payload.get("adapter") or "").strip()
    if not adapter_name:
        raise AdapterProcessError("adapter payload requires adapter identity")
    if expected_adapter is not None and adapter_name != expected_adapter:
        raise AdapterProcessError(
            f"adapter payload identity mismatch: expected {expected_adapter}, got {adapter_name}"
        )

    software = payload.get("software")
    if not isinstance(software, dict):
        raise AdapterProcessError("adapter payload requires software provenance")
    distribution = str(software.get("distribution") or "").strip()
    version = str(software.get("version") or "").strip()
    if not distribution or not version or version.lower() == "unknown":
        raise AdapterProcessError(
            "adapter payload requires a known software distribution and version"
        )
    capabilities = None
    try:
        from bench.adapters.registry import get_adapter

        capabilities = get_adapter(adapter_name)
        expected_distribution = capabilities.distribution
    except ValueError:
        expected_distribution = None
    if expected_distribution is not None and distribution != expected_distribution:
        raise AdapterProcessError(
            f"adapter distribution mismatch: expected {expected_distribution}, got {distribution}"
        )
    if expected_software is not None:
        snapshot_distribution = str(expected_software.get("distribution") or "").strip()
        snapshot_version = str(
            expected_software.get("distribution_version") or ""
        ).strip()
        if snapshot_distribution and distribution != snapshot_distribution:
            raise AdapterProcessError(
                "adapter payload distribution differs from the immutable "
                f"environment snapshot: expected {snapshot_distribution}, got {distribution}"
            )
        if snapshot_version and version != snapshot_version:
            raise AdapterProcessError(
                "adapter payload version differs from the immutable environment "
                f"snapshot: expected {snapshot_version}, got {version}"
            )

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise AdapterProcessError("adapter payload requires selection metadata")
    requested_raw = selection.get("requested_families")
    unsupported_raw = selection.get("unsupported_families")
    selection_mode = str(selection.get("mode") or "").strip()
    benchmark_workload_raw = selection.get("benchmark_workload")
    if (
        not isinstance(requested_raw, list)
        or any(
            not isinstance(value, str) or not value.strip() for value in requested_raw
        )
        or not isinstance(unsupported_raw, list)
        or any(
            not isinstance(value, str) or not value.strip() for value in unsupported_raw
        )
        or (
            benchmark_workload_raw is not None
            and (
                not isinstance(benchmark_workload_raw, str)
                or not benchmark_workload_raw.strip()
            )
        )
        or not selection_mode
    ):
        raise AdapterProcessError("adapter payload selection metadata is malformed")
    requested = [value.strip().lower() for value in requested_raw]
    unsupported_list = [value.strip().lower() for value in unsupported_raw]
    observed_benchmark_workload = (
        None
        if benchmark_workload_raw is None
        else benchmark_workload_raw.strip().lower()
    )
    unsupported = set(unsupported_list)
    if len(requested) != len(set(requested)) or len(unsupported_list) != len(
        unsupported
    ):
        raise AdapterProcessError(
            "adapter payload selection lists must not contain duplicates"
        )
    if set(requested).intersection(unsupported):
        raise AdapterProcessError(
            "adapter payload cannot report a family as both requested and unsupported"
        )
    if capabilities is not None and selection_mode != capabilities.selection_mode:
        raise AdapterProcessError(
            "adapter payload selection mode differs from the registered capability"
        )
    if expected_task is not None:
        if observed_benchmark_workload != expected_task.workload:
            raise AdapterProcessError(
                "adapter payload benchmark workload mismatch: expected "
                f"{expected_task.workload}, got {observed_benchmark_workload!r}"
            )
        scheduled = list(expected_task.scheduled_families)
        if capabilities is None:
            expected_requested = scheduled
            expected_unsupported: list[str] = []
        else:
            expected_requested = [
                family for family in scheduled if capabilities.supports(family)
            ]
            expected_unsupported = [
                family for family in scheduled if not capabilities.supports(family)
            ]
        if not expected_requested:
            raise UnsupportedTaskError(
                f"adapter reports no supported family for workload {expected_task.workload}"
            )
        if requested != expected_requested or unsupported_list != expected_unsupported:
            raise AdapterProcessError(
                "adapter payload family selection does not exactly match grouped "
                f"workload {expected_task.workload}: expected requested={expected_requested}, "
                f"unsupported={expected_unsupported}; observed requested={requested}, "
                f"unsupported={unsupported_list}"
            )
    elif expected_family:
        normalized_family = expected_family.strip().lower()
        if normalized_family in unsupported:
            raise UnsupportedTaskError(
                f"adapter reports unsupported family: {expected_family}"
            )
        if requested != [normalized_family]:
            raise AdapterProcessError(
                "adapter payload requested_families does not exactly match "
                f"the scheduled family {expected_family}"
            )
        if unsupported_list:
            raise AdapterProcessError(
                "a supported standalone family request cannot report unrelated "
                "unsupported families"
            )

    if expected_task is not None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise AdapterProcessError(
                "adapter payload requires benchmark execution metadata"
            )

        input_metadata = metadata.get("input")
        if not isinstance(input_metadata, Mapping):
            raise AdapterProcessError(
                "adapter payload requires metadata.input provenance"
            )
        expected_input = {
            "image_sha256": expected_task.image_sha256,
            "source_image_sha256": expected_task.source_image_sha256,
            "mask_sha256": expected_task.mask_sha256,
            "modality": expected_task.modality,
            "input_contract": expected_task.input_contract,
            "representation_id": expected_task.representation_id,
            "representation_derivation_sha256": (
                expected_task.representation_derivation_sha256
            ),
            "configured_levels": expected_task.configured_levels,
            "occupied_levels": expected_task.occupied_levels,
        }
        observed_input = {
            "image_sha256": str(input_metadata.get("image_sha256") or "").lower(),
            "source_image_sha256": str(
                input_metadata.get("source_image_sha256") or ""
            ).lower(),
            "mask_sha256": str(input_metadata.get("mask_sha256") or "").lower(),
            "modality": str(input_metadata.get("modality") or "").strip() or None,
            "input_contract": str(input_metadata.get("input_contract") or ""),
            "representation_id": str(input_metadata.get("representation_id") or ""),
            "representation_derivation_sha256": (
                str(input_metadata.get("representation_derivation_sha256")).lower()
                if input_metadata.get("representation_derivation_sha256")
                else None
            ),
            "configured_levels": _safe_int(input_metadata.get("configured_levels")),
            "occupied_levels": _safe_int(input_metadata.get("occupied_levels")),
        }
        if observed_input != expected_input:
            raise AdapterProcessError(
                "adapter payload input provenance does not match the scheduled task"
            )

        preprocessing = metadata.get("preprocessing")
        if not isinstance(preprocessing, Mapping):
            raise AdapterProcessError("adapter payload requires metadata.preprocessing")
        if (
            str(preprocessing.get("discretization") or "").lower()
            != expected_task.discretization
        ):
            raise AdapterProcessError(
                "adapter payload discretization does not match the scheduled task"
            )
        if _safe_int(preprocessing.get("bins")) != expected_task.bins:
            raise AdapterProcessError(
                "adapter payload bin count does not match the scheduled task"
            )
        observed_bin_width = _safe_float(preprocessing.get("bin_width"))
        if observed_bin_width is None or not math.isclose(
            observed_bin_width,
            expected_task.bin_width,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AdapterProcessError(
                "adapter payload bin width does not match the scheduled task"
            )
        expected_range = (
            [expected_task.intensity_min, expected_task.intensity_max]
            if expected_task.intensity_min is not None
            and expected_task.intensity_max is not None
            else None
        )
        observed_range = preprocessing.get("intensity_range")
        if expected_range is None:
            range_matches = observed_range is None
        else:
            range_matches = (
                isinstance(observed_range, list)
                and len(observed_range) == 2
                and all(
                    _safe_float(observed) is not None
                    and math.isclose(
                        float(observed),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for observed, expected in zip(observed_range, expected_range)
                )
            )
        if not range_matches:
            raise AdapterProcessError(
                "adapter payload intensity range does not match the scheduled task"
            )
        if (
            expected_task.adapter == "medimage"
            and preprocessing.get("intensity_type") != MEDIMAGE_BENCHMARK_INTENSITY_TYPE
        ):
            raise AdapterProcessError(
                "MEDimage payload does not attest the fixed benchmark "
                f"intensity type {MEDIMAGE_BENCHMARK_INTENSITY_TYPE!r}"
            )

        aggregation = metadata.get("aggregation")
        if not isinstance(aggregation, Mapping):
            raise AdapterProcessError("adapter payload requires metadata.aggregation")
        if (
            str(aggregation.get("requested") or "").lower() != REQUIRED_AGGREGATION
            or str(aggregation.get("effective_directional") or "").lower()
            != REQUIRED_AGGREGATION
        ):
            raise AdapterProcessError(
                "adapter payload does not attest the required 3D-merged aggregation"
            )

        timing_contract = metadata.get("timing_contract")
        if (
            not isinstance(timing_contract, Mapping)
            or _safe_int(timing_contract.get("version")) != TIMING_CONTRACT_VERSION
            or timing_contract.get("scope")
            != "prepared_workload_inputs_to_radiomic_calculations"
            or timing_contract.get("includes_required_preprocessing") is not False
            or timing_contract.get("excludes_file_io") is not True
            or timing_contract.get("excludes_mask_preparation") is not True
            or timing_contract.get("excludes_resegmentation") is not True
            or timing_contract.get("excludes_discretization") is not True
            or timing_contract.get("excludes_result_serialization") is not True
            or timing_contract.get("includes_matrix_mesh_neighborhood_construction")
            is not True
            or timing_contract.get("iterations_meaning")
            != "measured_observations_excluding_one_required_warmup"
            or timing_contract.get("adaptive_calls_per_observation") is not True
            or timing_contract.get("untimed_steady_state_calibration") is not True
            or timing_contract.get("multi_window_calibration_convergence") is not True
            or not math.isclose(
                float(timing_contract.get("calibration_headroom_factor") or 0.0),
                CALIBRATION_HEADROOM_FACTOR,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or _safe_int(timing_contract.get("calibration_minimum_rounds"))
            != CALIBRATION_MINIMUM_ROUNDS
            or _safe_int(timing_contract.get("calibration_maximum_rounds"))
            != CALIBRATION_MAXIMUM_ROUNDS
            or not math.isclose(
                float(timing_contract.get("calibration_cv_threshold") or 0.0),
                CALIBRATION_CV_THRESHOLD,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(timing_contract.get("calibration_span_ratio") or 0.0),
                CALIBRATION_SPAN_RATIO,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or _safe_int(timing_contract.get("post_warmup_verification_calls_minimum"))
            != 1
            or timing_contract.get("single_call_calibration_accepted_above_headroom")
            is not True
            or not math.isclose(
                float(timing_contract.get("target_observation_window_sec") or 0.0),
                TARGET_OBSERVATION_WINDOW_SEC,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or timing_contract.get("measured_window_minimum_enforced") is not True
            or _safe_int(timing_contract.get("maximum_calls_per_observation")) != 4096
            or timing_contract.get("reported_samples_are_per_call") is not True
            or timing_contract.get("within_process_result_equivalence_required")
            is not True
            or timing_contract.get("fresh_process_result_equivalence_required")
            is not True
            or not math.isclose(
                float(timing_contract.get("result_equivalence_rtol") or 0.0),
                RESULT_EQUIVALENCE_RTOL,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(timing_contract.get("result_equivalence_atol") or 0.0),
                RESULT_EQUIVALENCE_ATOL,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise AdapterProcessError(
                "adapter payload timing contract is missing or incompatible"
            )

        if expected_task.adapter == "pictologics":
            initialization = metadata.get("package_initialization")
            if (
                not isinstance(initialization, Mapping)
                or initialization.get("jit_warmup_performed") is not True
                or initialization.get("outside_measured_region") is not True
            ):
                raise AdapterProcessError(
                    "Pictologics payload does not attest the scheduled JIT "
                    "initialization policy"
                )

        timing = payload.get("timing")
        if not isinstance(timing, Mapping):
            raise AdapterProcessError(
                "timed benchmark tasks require an adapter timing payload"
            )
        expected_warmups = 1
        expected_measured = expected_task.timing_observations
        if (
            _safe_int(timing.get("warmup_iterations")) != expected_warmups
            or _safe_int(timing.get("measured_iterations")) != expected_measured
            or _safe_int(timing.get("measured_observations")) != expected_measured
            or _safe_int(timing.get("total_iterations"))
            != expected_measured + expected_warmups
        ):
            raise AdapterProcessError(
                "adapter timing observation counts do not match the scheduled policy"
            )
        calls_per_observation = _safe_int(timing.get("calls_per_observation"))
        calibration_calls = _safe_int(timing.get("calibration_calls"))
        calibration_rounds = _safe_int(timing.get("calibration_rounds"))
        calibration_duration = _safe_float(timing.get("calibration_duration_sec"))
        calibration_cpu = _safe_float(timing.get("calibration_cpu_time_sec"))
        calibration_per_call = _safe_float(timing.get("calibration_per_call_sec"))
        calibration_headroom = _safe_float(timing.get("calibration_headroom_factor"))
        calibration_stability_cv = _safe_float(timing.get("calibration_stability_cv"))
        calibration_stability_span = _safe_float(
            timing.get("calibration_stability_span")
        )
        calibration_stable = timing.get("calibration_stable")
        calibration_window_samples = timing.get("calibration_window_samples_sec")
        calibration_per_call_samples = timing.get("calibration_per_call_samples_sec")
        calibration_calls_per_round = timing.get("calibration_calls_per_round")
        single_window_per_call = (
            _safe_float(calibration_per_call_samples[0])
            if isinstance(calibration_per_call_samples, list)
            and len(calibration_per_call_samples) == 1
            else None
        )
        single_window_headroom_calibration = (
            calibration_rounds == 1
            and calibration_calls is not None
            and calibration_calls > 0
            and single_window_per_call is not None
            and single_window_per_call
            >= TARGET_OBSERVATION_WINDOW_SEC * CALIBRATION_HEADROOM_FACTOR
        )
        if (
            calls_per_observation is None
            or calls_per_observation < 1
            or calls_per_observation > 4096
            or calibration_calls is None
            or calibration_calls < 0
            or calibration_calls > 4096 * CALIBRATION_MAXIMUM_ROUNDS
            or calibration_rounds is None
            or calibration_rounds < 0
            or calibration_rounds > CALIBRATION_MAXIMUM_ROUNDS
            or calibration_duration is None
            or calibration_duration < 0
            or calibration_cpu is None
            or calibration_cpu < 0
            or calibration_headroom is None
            or not math.isclose(
                calibration_headroom,
                CALIBRATION_HEADROOM_FACTOR,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or calibration_stable is not True
            or _safe_int(timing.get("calibration_minimum_rounds"))
            != CALIBRATION_MINIMUM_ROUNDS
            or _safe_int(timing.get("calibration_maximum_rounds"))
            != CALIBRATION_MAXIMUM_ROUNDS
            or not math.isclose(
                _safe_float(timing.get("calibration_cv_threshold")) or -1.0,
                CALIBRATION_CV_THRESHOLD,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _safe_float(timing.get("calibration_span_ratio")) or -1.0,
                CALIBRATION_SPAN_RATIO,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isinstance(calibration_window_samples, list)
            or not isinstance(calibration_per_call_samples, list)
            or not isinstance(calibration_calls_per_round, list)
            or len(calibration_window_samples) != calibration_rounds
            or len(calibration_per_call_samples) != calibration_rounds
            or len(calibration_calls_per_round) != calibration_rounds
            or (calibration_calls == 0 and calibration_per_call is not None)
            or (calibration_calls == 0 and calibration_rounds != 0)
            or (
                calibration_calls == 0
                and (
                    calibration_stability_cv is not None
                    or calibration_stability_span is not None
                )
            )
            or (
                calibration_calls > 0
                and (
                    (
                        calibration_rounds < CALIBRATION_MINIMUM_ROUNDS
                        and not single_window_headroom_calibration
                    )
                    or calibration_per_call is None
                    or calibration_per_call <= 0
                    or calibration_stability_cv is None
                    or calibration_stability_cv > CALIBRATION_CV_THRESHOLD
                    or calibration_stability_span is None
                    or calibration_stability_span > CALIBRATION_SPAN_RATIO
                    or not math.isclose(
                        calibration_per_call,
                        calibration_duration / calibration_calls,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                )
            )
            or _safe_int(timing.get("measured_calculation_calls"))
            != expected_measured * calls_per_observation
            or _safe_int(timing.get("total_calculation_calls"))
            != (
                expected_warmups
                + calibration_calls
                + expected_measured * calls_per_observation
            )
        ):
            raise AdapterProcessError(
                "adapter timing adaptive batching counts are invalid: "
                f"calls_per_observation={calls_per_observation!r}, "
                f"calibration_calls={calibration_calls!r}, "
                f"calibration_rounds={calibration_rounds!r}, "
                f"calibration_stable={calibration_stable!r}, "
                f"calibration_cv={calibration_stability_cv!r}, "
                f"calibration_span={calibration_stability_span!r}, "
                "measured_calculation_calls="
                f"{timing.get('measured_calculation_calls')!r}, "
                f"total_calculation_calls={timing.get('total_calculation_calls')!r}"
            )
        normalized_calibration_windows = [
            _safe_float(value) for value in calibration_window_samples
        ]
        normalized_calibration_per_call = [
            _safe_float(value) for value in calibration_per_call_samples
        ]
        normalized_calibration_calls = [
            _safe_int(value) for value in calibration_calls_per_round
        ]
        if (
            any(value is None or value <= 0 for value in normalized_calibration_windows)
            or any(
                value is None or value <= 0 for value in normalized_calibration_per_call
            )
            or any(
                value is None or value < 1 or value > 4096
                for value in normalized_calibration_calls
            )
            or sum(int(value) for value in normalized_calibration_calls)
            != calibration_calls
            or not math.isclose(
                sum(float(value) for value in normalized_calibration_windows),
                calibration_duration,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or any(
                not math.isclose(
                    float(window),
                    float(per_call) * int(round_calls),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for window, per_call, round_calls in zip(
                    normalized_calibration_windows,
                    normalized_calibration_per_call,
                    normalized_calibration_calls,
                )
            )
        ):
            raise AdapterProcessError(
                "adapter timing calibration-window samples are inconsistent"
            )
        timing_samples: Dict[str, List[float]] = {}
        for key, positive in (
            ("duration_samples_sec", True),
            ("cpu_time_samples_sec", False),
        ):
            samples = timing.get(key)
            if not isinstance(samples, list) or len(samples) != expected_measured:
                raise AdapterProcessError(
                    f"adapter timing {key} does not match measured observation count"
                )
            normalized_samples = [_safe_float(value) for value in samples]
            if any(value is None for value in normalized_samples):
                raise AdapterProcessError(
                    f"adapter timing {key} contains a non-finite value"
                )
            if positive and any(float(value) <= 0 for value in normalized_samples):
                raise AdapterProcessError(
                    f"adapter timing {key} must contain positive observations"
                )
            if not positive and any(float(value) < 0 for value in normalized_samples):
                raise AdapterProcessError(
                    f"adapter timing {key} must contain non-negative observations"
                )
            timing_samples[key] = [float(value) for value in normalized_samples]
        observation_window_samples: List[float] = []
        for key, per_call_key, positive in (
            ("observation_window_samples_sec", "duration_samples_sec", True),
            (
                "cpu_observation_window_samples_sec",
                "cpu_time_samples_sec",
                False,
            ),
        ):
            samples = timing.get(key)
            if not isinstance(samples, list) or len(samples) != expected_measured:
                raise AdapterProcessError(
                    f"adapter timing {key} does not match measured observation count"
                )
            normalized = [_safe_float(value) for value in samples]
            if any(
                value is None or (float(value) <= 0 if positive else float(value) < 0)
                for value in normalized
            ):
                raise AdapterProcessError(
                    f"adapter timing {key} contains an invalid observation"
                )
            for window, per_call in zip(normalized, timing_samples[per_call_key]):
                assert window is not None
                if not math.isclose(
                    float(window),
                    per_call * calls_per_observation,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise AdapterProcessError(
                        f"adapter timing {key} is inconsistent with per-call samples"
                    )
            if key == "observation_window_samples_sec":
                observation_window_samples = [float(value) for value in normalized]
        minimum_observation_window = _safe_float(
            timing.get("minimum_observation_window_sec")
        )
        total_calculation_calls = _safe_int(timing.get("total_calculation_calls"))
        if (
            not observation_window_samples
            or any(
                value < TARGET_OBSERVATION_WINDOW_SEC
                for value in observation_window_samples
            )
            or minimum_observation_window is None
            or not math.isclose(
                minimum_observation_window,
                min(observation_window_samples),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or timing.get("result_equivalence_passed") is not True
            or total_calculation_calls is None
            or _safe_int(timing.get("result_equivalence_checks"))
            != total_calculation_calls - 1
            or not math.isclose(
                _safe_float(timing.get("result_equivalence_rtol")) or -1.0,
                RESULT_EQUIVALENCE_RTOL,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                _safe_float(timing.get("result_equivalence_atol")) or -1.0,
                RESULT_EQUIVALENCE_ATOL,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise AdapterProcessError(
                "adapter timing minimum-window or result-equivalence attestation is invalid"
            )
        for key in ("preparation_samples_sec", "finalization_samples_sec"):
            samples = timing.get(key)
            if not isinstance(samples, list) or len(samples) != expected_measured:
                raise AdapterProcessError(
                    f"adapter timing {key} does not match measured observation count"
                )
            normalized = [_safe_float(value) for value in samples]
            if any(value is None or float(value) < 0 for value in normalized):
                raise AdapterProcessError(
                    f"adapter timing {key} contains an invalid observation"
                )
        warmup_duration = _safe_float(timing.get("warmup_duration_sec"))
        warmup_cpu = _safe_float(timing.get("warmup_cpu_time_sec"))
        if warmup_duration is None or warmup_duration <= 0:
            raise AdapterProcessError(
                "adapter timing requires a positive untimed warmup duration"
            )
        if warmup_cpu is None or warmup_cpu < 0:
            raise AdapterProcessError(
                "adapter timing requires a non-negative untimed warmup CPU time"
            )

        for prefix, samples_key in (
            ("duration", "duration_samples_sec"),
            ("cpu_time", "cpu_time_samples_sec"),
        ):
            samples = timing_samples[samples_key]
            ordered = sorted(samples)
            midpoint = len(ordered) // 2
            median = (
                ordered[midpoint]
                if len(ordered) % 2
                else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
            )
            mean = sum(samples) / float(len(samples))
            variance = sum((value - mean) ** 2 for value in samples) / float(
                len(samples)
            )
            expected_summaries = {
                f"{prefix}_sec": median,
                f"{prefix}_min_sec": min(samples),
                f"{prefix}_mean_sec": mean,
                f"{prefix}_median_sec": median,
                f"{prefix}_std_sec": math.sqrt(variance),
                f"{prefix}_max_sec": max(samples),
            }
            for key, expected_value in expected_summaries.items():
                observed_value = _safe_float(timing.get(key))
                if observed_value is None or not math.isclose(
                    observed_value,
                    expected_value,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise AdapterProcessError(
                        f"adapter timing summary {key} does not match raw samples"
                    )
        if (
            _safe_int(metrics.get("warmup_iterations")) != expected_warmups
            or _safe_int(metrics.get("measured_iterations")) != expected_measured
            or _safe_int(metrics.get("measured_observations")) != expected_measured
            or _safe_int(metrics.get("total_iterations"))
            != expected_measured + expected_warmups
            or _safe_int(metrics.get("calls_per_observation")) != calls_per_observation
            or _safe_int(metrics.get("measured_calculation_calls"))
            != expected_measured * calls_per_observation
            or _safe_int(metrics.get("total_calculation_calls"))
            != (
                expected_warmups
                + calibration_calls
                + expected_measured * calls_per_observation
            )
            or _safe_int(metrics.get("calibration_calls")) != calibration_calls
        ):
            raise AdapterProcessError(
                "controller metrics do not preserve the adapter timing observation policy"
            )
        metric_duration = _safe_float(metrics.get("duration_sec"))
        timing_duration = _safe_float(timing.get("duration_sec"))
        if (
            metric_duration is None
            or timing_duration is None
            or not math.isclose(
                metric_duration,
                timing_duration,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise AdapterProcessError(
                "controller duration does not match the adapter timing payload"
            )
        host_wall_time = _safe_float(metrics.get("host_wall_time_sec"))
        peak_rss = _safe_float(metrics.get("peak_rss_bytes"))
        host_peak_rss = _safe_float(metrics.get("host_peak_rss_bytes"))
        worker_ready_rss = _safe_float(metrics.get("worker_ready_rss_bytes"))
        calculation_peak_rss = _safe_float(metrics.get("calculation_peak_rss_bytes"))
        incremental_peak_rss = _safe_float(
            metrics.get("incremental_calculation_peak_rss_bytes")
        )
        if host_wall_time is None or host_wall_time <= 0:
            raise AdapterProcessError(
                "controller host wall time must be positive and finite"
            )
        if (
            peak_rss is None
            or host_peak_rss is None
            or peak_rss <= 0
            or host_peak_rss <= 0
            or not peak_rss.is_integer()
            or not host_peak_rss.is_integer()
            or peak_rss != host_peak_rss
        ):
            raise AdapterProcessError(
                "controller process-tree peak RSS must be a matching positive "
                "integer measurement"
            )
        phase_status = str(metrics.get("memory_phase_observation_status") or "").strip()
        if phase_status not in {"complete", "partial", "unavailable"}:
            raise AdapterProcessError(
                "controller memory phase observation status is invalid"
            )
        phase_values = (worker_ready_rss, calculation_peak_rss)
        if any(
            value is not None
            and (value <= 0 or not value.is_integer() or value > peak_rss)
            for value in phase_values
        ):
            raise AdapterProcessError(
                "controller phase RSS contains an invalid observed value"
            )
        if phase_status == "complete":
            if (
                worker_ready_rss is None
                or calculation_peak_rss is None
                or incremental_peak_rss is None
                or incremental_peak_rss < 0
                or not incremental_peak_rss.is_integer()
                or incremental_peak_rss
                != max(0.0, calculation_peak_rss - worker_ready_rss)
            ):
                raise AdapterProcessError(
                    "controller complete phase RSS measurements are inconsistent"
                )
        elif incremental_peak_rss is not None:
            raise AdapterProcessError(
                "incremental calculation RSS requires complete phase observations"
            )
        for metric_key, expected_value in (
            ("calibration_duration_sec", calibration_duration),
            ("calibration_cpu_time_sec", calibration_cpu),
            ("calibration_per_call_sec", calibration_per_call),
            ("calibration_headroom_factor", calibration_headroom),
            ("calibration_stability_cv", calibration_stability_cv),
            ("calibration_stability_span", calibration_stability_span),
            ("minimum_observation_window_sec", minimum_observation_window),
            (
                "result_equivalence_rtol",
                _safe_float(timing.get("result_equivalence_rtol")),
            ),
            (
                "result_equivalence_atol",
                _safe_float(timing.get("result_equivalence_atol")),
            ),
        ):
            observed_value = _safe_float(metrics.get(metric_key))
            if (observed_value is None) != (expected_value is None) or (
                observed_value is not None
                and expected_value is not None
                and not math.isclose(
                    observed_value,
                    expected_value,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise AdapterProcessError(
                    f"controller metric {metric_key} does not match adapter timing"
                )
        for metric_key, expected_value in (
            ("calibration_rounds", calibration_rounds),
            (
                "result_equivalence_checks",
                _safe_int(timing.get("result_equivalence_checks")),
            ),
        ):
            if _safe_int(metrics.get(metric_key)) != expected_value:
                raise AdapterProcessError(
                    f"controller metric {metric_key} does not match adapter timing"
                )
        if (
            metrics.get("calibration_stable") is not True
            or metrics.get("result_equivalence_passed") is not True
        ):
            raise AdapterProcessError(
                "controller result-equivalence or calibration status is invalid"
            )
        if (_safe_int(metrics.get("adapter_event_count")) or 0) < (
            2 + 2 * expected_measured + expected_warmups
        ):
            raise AdapterProcessError(
                "controller did not observe the complete adapter timing event stream"
            )
    features = payload.get("features")
    if not isinstance(features, dict) or not isinstance(features.get("all"), list):
        raise AdapterProcessError("adapter payload requires features.all as a list")
    feature_names = features["all"]
    if (
        not feature_names
        or any(
            not isinstance(value, str) or not value.strip() for value in feature_names
        )
        or len(feature_names) != len(set(feature_names))
    ):
        raise AdapterProcessError(
            "adapter payload requires a non-empty, unique list of feature names"
        )
    if expected_task is not None:
        if (
            expected_task.expected_feature_count is not None
            and len(feature_names) != expected_task.expected_feature_count
        ):
            raise AdapterProcessError(
                f"adapter returned {len(feature_names)} features for "
                f"{expected_task.adapter}/{expected_task.workload}; endpoint contract "
                f"requires {expected_task.expected_feature_count}"
            )
        values = payload.get("values")
        if not isinstance(values, Mapping) or not isinstance(
            values.get("all"),
            Mapping,
        ):
            raise AdapterProcessError("benchmark adapter payload requires values.all")
        value_mapping = values["all"]
        if set(value_mapping) != set(feature_names):
            raise AdapterProcessError(
                "adapter values.all keys must exactly match the reported feature names"
            )
        nonfinite = [
            name
            for name, value in value_mapping.items()
            if _finite_real_scalar(value) is None
        ]
        if nonfinite:
            raise AdapterProcessError(
                "adapter returned non-finite/non-scalar feature values: "
                + ", ".join(sorted(nonfinite)[:5])
            )
    duration = _safe_float(metrics.get("duration_sec"))
    if duration is None or duration <= 0:
        raise AdapterProcessError(
            "adapter returned a non-positive or non-finite duration"
        )
    return len(feature_names)


def _expected_measured_record(
    *,
    task: TaskSpec,
    run_id: str,
    attempt: int,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reconstruct the only valid record for an atomically written payload."""

    features = payload.get("features")
    names = features.get("all") if isinstance(features, Mapping) else None
    if not isinstance(names, list):
        raise AdapterProcessError("cannot reconstruct record without features.all")
    feature_count = len(names)
    expected = _terminal_record(
        task,
        run_id,
        STATUS_MEASURED,
        attempt=attempt,
    )
    expected.update(dict(metrics))
    expected["task_status"] = STATUS_MEASURED
    expected["success"] = True
    expected["feature_count"] = feature_count
    expected["attempted_feature_count"] = feature_count
    expected["finite_feature_count"] = feature_count
    software = payload.get("software")
    if isinstance(software, Mapping):
        expected["adapter_distribution"] = software.get("distribution")
        expected["adapter_version"] = software.get("version")
    return expected


def _validate_measured_record(
    record: Mapping[str, Any],
    *,
    task: TaskSpec,
    run_id: str,
    attempt: int,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    expected = _expected_measured_record(
        task=task,
        run_id=run_id,
        attempt=attempt,
        payload=payload,
        metrics=metrics,
    )
    if dict(record) != expected:
        differing = sorted(
            key
            for key in set(record).union(expected)
            if record.get(key) != expected.get(key)
        )
        detail = ", ".join(differing[:8]) or "unknown"
        raise AdapterProcessError(
            "embedded benchmark record does not match the scheduled task and "
            f"validated measurements (fields: {detail})"
        )


def _fresh_process_equivalence_key(task: TaskSpec) -> Tuple[str, ...]:
    """Identify repeats that must return the same scientific result."""

    return (
        task.case_id,
        task.adapter,
        task.workload_key,
        task.image_sha256,
        task.mask_sha256,
        task.representation_id,
        str(task.representation_derivation_sha256 or ""),
    )


def _validate_or_register_fresh_process_result(
    references: Dict[Tuple[str, ...], Dict[str, Any]],
    *,
    task: TaskSpec,
    payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Require exact names and equivalent values across fresh-process repeats."""

    features = payload.get("features")
    values = payload.get("values")
    names = features.get("all") if isinstance(features, Mapping) else None
    value_mapping = values.get("all") if isinstance(values, Mapping) else None
    if not isinstance(names, list) or not isinstance(value_mapping, Mapping):
        raise AdapterProcessError(
            "fresh-process equivalence requires features.all and values.all"
        )
    key = _fresh_process_equivalence_key(task)
    reference = references.get(key)
    if reference is None:
        references[key] = {
            "task_id": task.task_id,
            "repeat": task.repeat,
            "feature_names": list(names),
            "values": dict(value_mapping),
        }
        return None
    if list(names) != reference["feature_names"]:
        raise AdapterProcessError(
            "fresh-process repeat changed feature names or their deterministic order "
            f"relative to repeat {reference['repeat']}"
        )
    try:
        assert_numerically_equivalent(
            reference["values"],
            value_mapping,
            path="fresh_process.values",
            rtol=RESULT_EQUIVALENCE_RTOL,
            atol=RESULT_EQUIVALENCE_ATOL,
        )
    except ResultEquivalenceError as exc:
        raise AdapterProcessError(
            "fresh-process repeat changed numerical feature values relative to "
            f"repeat {reference['repeat']}: {exc}"
        ) from exc
    return reference


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unnamed"


def _payload_path(records_dir: Path, task: TaskSpec) -> Path:
    workload = task.workload
    filename = (
        f"{_safe_component(task.adapter)}_{_safe_component(workload)}_"
        f"rep{task.repeat}_{task.task_id[:12]}.json"
    )
    return records_dir / _safe_component(task.case_id) / filename


def _write_measured_payload(
    path: Path,
    *,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    task: TaskSpec,
    record: Mapping[str, Any],
    run_fingerprint: str,
) -> str:
    stored = dict(payload)
    stored["host_metrics"] = dict(metrics)
    stored["benchmark"] = {
        "schema_version": 1,
        "task_id": task.task_id,
        "run_fingerprint": run_fingerprint,
        "status": STATUS_MEASURED,
        "record": dict(record),
    }
    return atomic_write_json(path, stored)


def _try_adopt_payload(
    ledger: BenchmarkLedger,
    task: TaskSpec,
    path: Path,
    run_fingerprint: str,
    run_id: str,
    expected_software: Optional[Mapping[str, Any]] = None,
) -> bool:
    if not path.is_file() or ledger.status(task.task_id) not in RECOVERABLE_STATUSES:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark = payload.get("benchmark")
        if not isinstance(benchmark, dict):
            return False
        if benchmark.get("task_id") != task.task_id:
            return False
        if benchmark.get("run_fingerprint") != run_fingerprint:
            return False
        if benchmark.get("status") != STATUS_MEASURED:
            return False
        record = benchmark.get("record")
        if not isinstance(record, dict) or record.get("task_status") != STATUS_MEASURED:
            return False
        metrics = payload.get("host_metrics")
        if not isinstance(metrics, dict):
            return False
        _validate_measured_result(
            payload,
            metrics,
            expected_task=task,
            expected_software=expected_software,
        )
        ledger_row = ledger.task_row(task.task_id)
        _validate_measured_record(
            record,
            task=task,
            run_id=run_id,
            attempt=int(ledger_row["attempt"]),
            payload=payload,
            metrics=metrics,
        )
        digest = sha256_file(path)
        ledger.mark_terminal(
            task.task_id,
            STATUS_MEASURED,
            record,
            duration_sec=_safe_float(record.get("duration_sec")),
            payload_path=ledger.portable_payload_path(path),
            payload_sha256=digest,
            adopted=True,
        )
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
        return False


def _verify_committed_payloads(
    ledger: BenchmarkLedger,
    run_fingerprint: str,
    run_id: str,
    tasks_by_id: Mapping[str, TaskSpec],
    adapter_environments: Mapping[str, Mapping[str, Any]],
) -> int:
    """Reject a resume if an authoritative measured payload is missing or altered."""
    verified = 0
    for row in ledger.task_rows():
        if str(row["status"]) != STATUS_MEASURED:
            continue
        task_id = str(row["task_id"])
        path_value = str(row["payload_path"] or "")
        expected_digest = str(row["payload_sha256"] or "")
        if not path_value or len(expected_digest) != 64:
            raise RunIntegrityError(
                f"measured task {task_id} has incomplete payload metadata"
            )
        path = ledger.resolve_payload_path(path_value)
        if not path.is_file():
            raise RunIntegrityError(
                f"measured task {task_id} payload is missing: {path}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise RunIntegrityError(
                f"measured task {task_id} payload checksum changed: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            benchmark = payload["benchmark"]
            record = benchmark["record"]
            stored_record = json.loads(str(row["record_json"]))
            metrics = payload["host_metrics"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RunIntegrityError(
                f"measured task {task_id} payload is not a valid committed result: {path}"
            ) from exc
        if not isinstance(benchmark, dict) or not isinstance(record, dict):
            raise RunIntegrityError(
                f"measured task {task_id} payload metadata is malformed"
            )
        if (
            benchmark.get("task_id") != task_id
            or benchmark.get("run_fingerprint") != run_fingerprint
            or benchmark.get("status") != STATUS_MEASURED
            or record != stored_record
        ):
            raise RunIntegrityError(
                f"measured task {task_id} payload identity does not match ledger"
            )
        if not isinstance(metrics, dict):
            raise RunIntegrityError(
                f"measured task {task_id} payload metrics are malformed"
            )
        try:
            task = tasks_by_id.get(task_id)
            if task is None:
                raise RunIntegrityError(
                    f"measured task {task_id} is absent from the immutable task plan"
                )
            _validate_measured_result(
                payload,
                metrics,
                expected_task=task,
                expected_software=adapter_environments.get(task.adapter),
            )
            _validate_measured_record(
                record,
                task=task,
                run_id=run_id,
                attempt=int(row["attempt"]),
                payload=payload,
                metrics=metrics,
            )
        except RuntimeError as exc:
            raise RunIntegrityError(
                f"measured task {task_id} payload failed validation"
            ) from exc
        verified += 1
    return verified


def _load_fresh_process_references(
    ledger: BenchmarkLedger,
    tasks_by_id: Mapping[str, TaskSpec],
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """Revalidate cross-process equivalence for every committed repeat."""

    references: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    _, payloads = ledger.verified_records_and_payloads()
    for payload in payloads:
        benchmark = payload.get("benchmark")
        task_id = benchmark.get("task_id") if isinstance(benchmark, Mapping) else None
        task = tasks_by_id.get(str(task_id or ""))
        if task is None:
            raise RunIntegrityError(
                "committed payload is absent from the immutable task plan"
            )
        try:
            reference = _validate_or_register_fresh_process_result(
                references,
                task=task,
                payload=payload,
            )
        except AdapterProcessError as exc:
            raise RunIntegrityError(
                "committed fresh-process repeats are not numerically equivalent"
            ) from exc
        metrics = payload.get("host_metrics")
        if not isinstance(metrics, Mapping):
            raise RunIntegrityError("committed payload host metrics are malformed")
        expected_task_id = None if reference is None else reference["task_id"]
        expected_repeat = None if reference is None else reference["repeat"]
        if (
            metrics.get("fresh_process_reference_task_id") != expected_task_id
            or metrics.get("fresh_process_reference_repeat") != expected_repeat
            or metrics.get("fresh_process_result_equivalence_passed") is not True
            or not math.isclose(
                float(metrics.get("fresh_process_result_equivalence_rtol") or -1.0),
                RESULT_EQUIVALENCE_RTOL,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(metrics.get("fresh_process_result_equivalence_atol") or -1.0),
                RESULT_EQUIVALENCE_ATOL,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RunIntegrityError(
                "committed payload fresh-process equivalence attestation is invalid"
            )
    return references


class _SignalStop:
    def __init__(self):
        self.requested = False
        self.signal_number: Optional[int] = None
        self._previous: Dict[int, Any] = {}

    def _handler(self, signum, frame) -> None:
        self.requested = True
        self.signal_number = int(signum)

    def __enter__(self) -> "_SignalStop":
        if threading.current_thread() is threading.main_thread():
            signals = [signal.SIGINT, signal.SIGTERM]
            if hasattr(signal, "SIGBREAK"):
                signals.append(signal.SIGBREAK)
            for signum in signals:
                self._previous[int(signum)] = signal.getsignal(signum)
                signal.signal(signum, self._handler)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)


def _evaluate_guardrail(
    ledger: BenchmarkLedger,
    policy: GuardrailPolicy,
    task: TaskSpec,
    duration: float,
) -> None:
    if not policy.should_compare(task.adapter):
        return
    baseline = ledger.measured_match(
        case_id=task.case_id,
        adapter=policy.baseline_adapter,
        workload_key=task.workload_key,
        repeat=task.repeat,
    )
    if baseline is None:
        return
    ratio = policy.ratio(duration, _safe_float(baseline["duration_sec"]))
    if ratio is None:
        return
    scope = policy.scope_key(
        task.adapter,
        task.workload_key,
        task.guardrail_group,
    )
    slow_count = ledger.record_guardrail_observation(
        task_id=task.task_id,
        scope_key=scope,
        baseline_task_id=str(baseline["task_id"]),
        complexity=task.complexity,
        ratio=ratio,
        slow_threshold=policy.skip_ratio,
    )
    if ratio >= policy.skip_ratio and slow_count >= policy.minimum_slow_observations:
        ledger.activate_guardrail(
            scope_key=scope,
            adapter=task.adapter,
            workload_key=task.workload_key,
            guardrail_group=task.guardrail_group,
            cutoff_complexity=task.complexity,
            reason=(
                f"measured slowdown {ratio:.6g}x met the configured "
                f"{policy.skip_ratio:.6g}x threshold after {slow_count} observation(s)"
            ),
            evidence_task_id=task.task_id,
        )


def _guardrail_comparison_key(task: TaskSpec) -> Tuple[str, str, int]:
    return task.case_id, task.workload_key, task.repeat


def _reconcile_guardrail_group(
    ledger: BenchmarkLedger,
    policy: GuardrailPolicy,
    tasks: Sequence[TaskSpec],
) -> None:
    """Evaluate every now-complete candidate/baseline pair in one block.

    Adapter order is deliberately rotated. A candidate can therefore finish
    before its exact baseline; reconciliation when either peer arrives, and on
    resume, makes the persisted decision independent of execution order.
    """

    for candidate in tasks:
        if not policy.should_compare(candidate.adapter):
            continue
        measured = ledger.measured_match(
            case_id=candidate.case_id,
            adapter=candidate.adapter,
            workload_key=candidate.workload_key,
            repeat=candidate.repeat,
        )
        if measured is None:
            continue
        duration = _safe_float(measured["duration_sec"])
        if duration is not None:
            _evaluate_guardrail(ledger, policy, candidate, duration)


def _guardrail_comparison_groups(
    tasks: Sequence[TaskSpec],
) -> Dict[Tuple[str, str, int], List[TaskSpec]]:
    groups: Dict[Tuple[str, str, int], List[TaskSpec]] = defaultdict(list)
    for task in tasks:
        groups[_guardrail_comparison_key(task)].append(task)
    return groups


def _reconcile_guardrails(
    ledger: BenchmarkLedger,
    policy: GuardrailPolicy,
    groups: Mapping[Tuple[str, str, int], Sequence[TaskSpec]],
) -> None:
    for tasks in groups.values():
        _reconcile_guardrail_group(ledger, policy, tasks)


def _reconcile_timeout_cutoffs(
    ledger: BenchmarkLedger,
    policy: GuardrailPolicy,
    tasks_by_id: Mapping[str, TaskSpec],
) -> None:
    """Recover timeout cutoffs if interruption followed the terminal commit."""

    if not policy.truncate_on_timeout:
        return
    for row in ledger.task_rows():
        if str(row["status"]) != STATUS_TIMED_OUT:
            continue
        task = tasks_by_id.get(str(row["task_id"]))
        if task is None:
            continue
        reason = str(row["error"] or "task exceeded the configured timeout")
        complexity_metric, complexity = _timeout_cutoff_complexity(task)
        ledger.activate_timeout_cutoff(
            scope_key=policy.timeout_scope_key(
                task.adapter,
                task.workload_key,
                task.guardrail_group,
            ),
            adapter=task.adapter,
            workload_key=task.workload_key,
            guardrail_group=task.guardrail_group,
            complexity_metric=complexity_metric,
            cutoff_complexity=complexity,
            reason=reason,
            evidence_task_id=task.task_id,
        )


def _timeout_cutoff_complexity(task: TaskSpec) -> Tuple[str, int]:
    """Return the monotone work-size proxy used by timeout truncation."""

    return "image_voxels", int(task.image_voxels)


def _checkpoint(
    *,
    ledger: BenchmarkLedger,
    report_dir: Path,
    run_id: str,
    machine: Dict[str, Any],
    args: argparse.Namespace,
    run_spec: RunSpec,
    run_status: str,
    final: bool = False,
) -> None:
    save_summaries(
        report_dir,
        run_id,
        ledger.records(),
        machine,
        args,
        final=final,
        ledger=ledger,
        run_spec=run_spec,
        run_status=run_status,
    )


def _human_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _print_ledger_progress(
    ledger: BenchmarkLedger,
    *,
    total_tasks: int,
    project_total_tasks: Optional[int],
    project_task_offset: int,
    session_started: float,
    task_timeout_sec: Optional[float],
) -> None:
    rows = ledger.task_rows()
    terminal = [row for row in rows if str(row["status"]) in TERMINAL_STATUSES]
    completed = len(terminal)
    remaining = max(0, total_tasks - completed)
    elapsed = time.perf_counter() - session_started
    # Startup/warmup and the prepared calculation region have independent
    # timeout clocks. Cap extrapolated task turnaround at their combined bound.
    eta_estimate = estimate_pending_turnaround(
        rows,
        maximum_task_seconds=(
            None if task_timeout_sec is None else 2.0 * float(task_timeout_sec)
        ),
        timeout_cutoffs=ledger.timeout_cutoffs(),
    )
    estimate_basis = "unavailable"
    eta: Optional[float] = None
    eta_range_text = ""
    if eta_estimate is not None:
        eta = eta_estimate.seconds
        estimate_basis = eta_estimate.basis
        eta_range_text = (
            f" [rough range {_human_duration(eta_estimate.lower_seconds)}–"
            f"{_human_duration(eta_estimate.upper_seconds)}]"
        )
    counts = ledger.status_counts()
    status_text = ", ".join(
        f"{key}={value}" for key, value in sorted(counts.items()) if value
    )
    project_text = ""
    project_eta_text = ""
    if project_total_tasks is not None:
        project_completed = min(project_total_tasks, project_task_offset + completed)
        project_text = (
            f"; project {project_completed}/{project_total_tasks} "
            f"({100.0 * project_completed / project_total_tasks:.2f}%)"
        )
        unseen_future_tasks = max(
            0,
            project_total_tasks - project_task_offset - total_tasks,
        )
        if unseen_future_tasks:
            project_eta_text = (
                "; rough project ETA=unknown "
                f"({unseen_future_tasks} later-pillar tasks are not modeled yet)"
            )
        else:
            project_eta_text = f"; rough project ETA={_human_duration(eta)}"
    print(
        f"PROGRESS run {completed}/{total_tasks} "
        f"({100.0 * completed / max(1, total_tasks):.2f}%){project_text}; "
        f"remaining={remaining}; rough run ETA={_human_duration(eta)}{eta_range_text}"
        f"{project_eta_text} "
        f"({estimate_basis}); elapsed={_human_duration(elapsed)}; {status_text}",
        flush=True,
    )


def _existing_run_id(report_dir: Optional[str]) -> Optional[str]:
    if not report_dir:
        return None
    path = Path(report_dir) / "run_spec.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value.get("run_id") or "") or None
    except Exception:
        return None


def _plan_benchmark_execution(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dataset_validation: Mapping[str, Any],
    dataset: str,
    dataset_kind: str,
    cases: List[Dict[str, Any]],
    adapters: Sequence[str],
    workloads: Sequence[BenchmarkWorkload],
    policy: GuardrailPolicy,
    memory_policy: MemoryPreflightPolicy,
    report_dir: Path,
    records_dir: Path,
    ledger_path: Path,
    run_id: str,
    endpoint_contract: Optional[BenchmarkContract],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Tuple[int, int, int, int]],
    List[TaskSpec],
    Dict[Tuple[str, str, int], List[TaskSpec]],
    Dict[str, Any],
    Dict[str, Any],
    RunSpec,
]:
    """Stage inputs and build the immutable plan under the caller's run lock."""

    input_identities: Dict[str, Tuple[int, int, int, int]] = {}
    if not args.dry_run:
        if args.resume and not ledger_path.exists():
            untracked_records = any(records_dir.rglob("*.json"))
            if untracked_records or (report_dir / "run_spec.json").exists():
                raise RunSpecMismatch(
                    "cannot safely resume a run without its transactional ledger; "
                    "start a new run directory"
                )
        if not args.resume:
            if ledger_path.exists():
                raise RunAlreadyExists(
                    f"benchmark ledger already exists in {report_dir}; pass --resume "
                    "with an identical specification or use a new directory"
                )
            existing_artifacts = (report_dir / "run_spec.json").exists() or any(
                records_dir.rglob("*.json")
            )
            if existing_artifacts:
                raise RunAlreadyExists(
                    f"benchmark artifacts already exist in {report_dir}; "
                    "use a new directory or pass --resume for a transactional run"
                )
        cases, input_identities = _stage_selected_inputs(
            cases,
            report_dir / "inputs",
        )

    tasks = build_task_plan(
        cases=cases,
        dataset=dataset,
        adapters=adapters,
        workloads=workloads,
        repeats=args.repeats,
        timing_observations=args.timing_observations,
        input_contract=args.input_contract,
        endpoint_contract=endpoint_contract,
    )
    guardrail_groups = _guardrail_comparison_groups(tasks)
    adapter_environments = _adapter_environment_snapshots(adapters)
    machine = _machine_info(
        machine_id=args.machine_id,
        machine_label=args.machine_label,
        cpu_model=args.cpu_model,
        cpu_base_ghz=args.cpu_base_ghz,
        host_profile_id=args.host_profile_id,
        host_profile_sha256=args.host_profile_sha256,
        host_settings_json=args.host_settings_json,
    )
    thread_policy = _benchmark_thread_policy(machine.get("cpu_count_physical"))
    run_spec = RunSpec.create(
        run_id=run_id,
        dataset=dataset,
        dataset_kind=dataset_kind,
        dataset_manifest_schema_version=int(manifest["schema_version"]),
        dataset_dir=str(Path(args.dataset_dir).resolve()),
        manifest_sha256=str(dataset_validation["manifest_sha256"]),
        dataset_hashes_verified=bool(dataset_validation["hashes_verified"]),
        dataset_values_inspected=bool(dataset_validation["voxel_values_inspected"]),
        selected_case_ids=tuple(str(case["case_id"]) for case in cases),
        adapters=tuple(adapters),
        workloads=tuple(dict.fromkeys(task.workload_key for task in tasks)),
        repeats=args.repeats,
        aggregation=REQUIRED_AGGREGATION,
        input_contract=args.input_contract,
        timing_observations=args.timing_observations,
        capture_values=True,
        timeout_seconds=args.timeout,
        keep_going=args.keep_going,
        task_plan_sha256=task_plan_fingerprint(tasks),
        runtime_profiles_sha256=_runtime_profiles_sha256(),
        benchmark_sources_sha256=_benchmark_sources_sha256(adapters),
        benchmark_machine=_benchmark_machine_identity(machine),
        adapter_environments=adapter_environments,
        thread_policy=thread_policy,
        initialization_policy=BENCHMARK_INITIALIZATION_POLICY,
        guardrail=policy.to_dict(),
        memory_preflight=memory_policy.to_dict(),
        endpoint_contract_id=(
            endpoint_contract.contract_id if endpoint_contract is not None else None
        ),
        endpoint_contract_path=(
            str(endpoint_contract.path) if endpoint_contract is not None else None
        ),
        endpoint_contract_sha256=(
            endpoint_contract.sha256 if endpoint_contract is not None else None
        ),
    )
    return (
        cases,
        input_identities,
        tasks,
        guardrail_groups,
        adapter_environments,
        machine,
        run_spec,
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench run")
    parser.add_argument(
        "--dataset-dir", required=True, help="Directory containing manifest.json"
    )
    parser.add_argument("--run-id", default=None, help="Identifier of the run")
    parser.add_argument(
        "--report-dir", default=None, help="Directory to save run reports"
    )
    parser.add_argument(
        "--adapters", default="pictologics,pyradiomics,mirp,medimage,zrad"
    )
    parser.add_argument("--sizes", default=None, help="Comma-separated sizes to filter")
    parser.add_argument(
        "--variants", default=None, help="Comma-separated variants to filter"
    )
    parser.add_argument(
        "--masks", default=None, help="Comma-separated mask IDs to filter"
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help="Optional comma-separated modality filter (for example ct,mri,pet)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Fresh adapter processes per eligible case/workload (reviewed default: 3)",
    )
    parser.add_argument(
        "--endpoint-contract",
        default=str(Path("configs/benchmark/calculation_only_workload.json")),
        help="Frozen Benchmark calculation endpoint JSON",
    )
    parser.add_argument(
        "--input-contract",
        choices=[HARMONIZED_INPUT_CONTRACT],
        default=HARMONIZED_INPUT_CONTRACT,
        help="Frozen Benchmark representation-routing contract",
    )
    parser.add_argument(
        "--workloads",
        default="all",
        help=(
            "Comma-separated grouped calculation workloads: morphology, intensity, "
            "texture, ivh; or 'all'"
        ),
    )
    parser.add_argument(
        "--timing-observations",
        type=int,
        default=3,
        help="Measured observation windows per process (warmup is additional)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Per-phase process-tree safety limit: one clock before worker-ready "
            "and a fresh clock after worker-ready"
        ),
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--extend-repeats",
        action="store_true",
        help=(
            "With --resume, append higher absolute repeat numbers without "
            "altering existing payloads or persisted guardrail decisions"
        ),
    )
    parser.add_argument("--guardrail-baseline", default="pictologics")
    parser.add_argument(
        "--guardrail-skip-ratio",
        type=float,
        default=1000.0,
        help=(
            "Editable candidate/baseline slowdown threshold used only for "
            "future larger ranks in the same validated branch (default: 1000)"
        ),
    )
    parser.add_argument("--guardrail-min-observations", type=int, default=1)
    parser.add_argument(
        "--enable-speed-truncation",
        action="store_true",
        help=(
            "Development-only opt-in to extrapolative speed skipping. The "
            "publication protocol leaves this disabled and uses only observed "
            "timeout cutoffs."
        ),
    )
    parser.add_argument(
        "--no-truncate-on-timeout",
        action="store_true",
        help=(
            "Disable the default cutoff for subsequent same-adapter/workload "
            "tasks on strictly larger images in the same mask/configuration series"
        ),
    )
    parser.add_argument("--memory-budget-fraction", type=float, default=0.80)
    parser.add_argument("--memory-reserve-gib", type=float, default=4.0)
    parser.add_argument("--memory-cap-gib", type=float, default=None)
    parser.add_argument("--memory-safety-factor", type=float, default=1.50)
    parser.add_argument("--termination-grace", type=float, default=3.0)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between live status lines while an adapter process is active",
    )
    parser.add_argument(
        "--project-total-tasks",
        type=int,
        default=None,
        help="Optional all-pillar task total used only for live progress reporting",
    )
    parser.add_argument(
        "--project-task-offset",
        type=int,
        default=0,
        help="Terminal task offset from earlier pillars for project progress",
    )
    parser.add_argument(
        "--machine-id",
        default=None,
        help="Optional stable, non-identifying machine ID for multi-host studies",
    )
    parser.add_argument(
        "--machine-label",
        default=None,
        help="Optional reader-facing machine label",
    )
    parser.add_argument(
        "--cpu-model",
        default=None,
        help="Override an unavailable or generic automatic CPU model",
    )
    parser.add_argument(
        "--cpu-base-ghz",
        type=float,
        default=None,
        help="Optional documented CPU base frequency in GHz",
    )
    parser.add_argument(
        "--host-profile-id",
        default=None,
        help="Frozen host-profile identifier forwarded by the reviewed launcher",
    )
    parser.add_argument(
        "--host-profile-sha256",
        default=None,
        help="SHA-256 of the frozen host-profile file",
    )
    parser.add_argument(
        "--host-settings-json",
        default=None,
        help="Frozen host settings encoded as a JSON object",
    )
    parser.add_argument("--no-verify-dataset-hashes", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the plan only"
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    endpoint_contract: Optional[BenchmarkContract]
    contract_path = Path(args.endpoint_contract)
    if not contract_path.is_absolute():
        contract_path = repo_root() / contract_path
    endpoint_contract = load_benchmark_contract(contract_path)
    timing_policy = endpoint_contract.payload["timing"]
    reviewed_repeats = int(timing_policy["fresh_process_repeats"])
    if args.repeats < reviewed_repeats:
        raise ValueError(
            "frozen endpoint contract requires at least "
            f"{reviewed_repeats} fresh process repeats"
        )
    if args.repeats > reviewed_repeats and not args.extend_repeats:
        raise ValueError(
            "adding fresh process repeats requires --extend-repeats so the "
            "append-only plan change is explicit"
        )
    if args.timing_observations != int(
        timing_policy["measured_observations_per_process"]
    ):
        raise ValueError(
            "frozen endpoint contract requires exactly "
            f"{timing_policy['measured_observations_per_process']} measured "
            "observations per process"
        )
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.termination_grace < 0:
        raise ValueError("termination grace cannot be negative")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be >= 1")
    if args.progress_interval <= 0:
        raise ValueError("progress interval must be positive")
    if args.project_total_tasks is not None and args.project_total_tasks < 1:
        raise ValueError("project total tasks must be positive")
    if args.project_task_offset < 0:
        raise ValueError("project task offset cannot be negative")
    if args.no_verify_dataset_hashes and not args.dry_run:
        raise ValueError(
            "--no-verify-dataset-hashes is restricted to dry-run planning; "
            "every executable benchmark must verify the manifest-bound input bytes"
        )
    dataset_path = Path(args.dataset_dir).resolve()
    manifest, dataset_validation = load_and_validate_manifest(
        dataset_path,
        verify_hashes=not args.no_verify_dataset_hashes,
        # Headers are cheap to inspect and make actual shape/voxel complexity
        # authoritative for planning and resource checks. Full-value inspection is deliberate:
        # adapters must receive the same canonical binary ROI and local-intensity
        # implementations may inspect image values outside that ROI.
        inspect_geometry=True,
        inspect_values=True,
    )
    dataset = str(manifest["dataset"])
    dataset_kind = str(manifest["dataset_kind"])

    cases = _validate_and_select_cases(
        dataset_path,
        manifest,
        sizes=args.sizes,
        variants=args.variants,
        masks=args.masks,
        modalities=args.modalities,
        # The schema loader above already checked every declared file exactly once.
        verify_hashes=False,
    )
    if args.input_contract == HARMONIZED_INPUT_CONTRACT:
        missing = [
            str(case["case_id"]) for case in cases if not case.get("discrete_image_abs")
        ]
        if missing:
            raise ValueError(
                "manifest_harmonized requires a frozen discrete image for every "
                "selected case; missing: " + ", ".join(missing[:5])
            )
    if (
        endpoint_contract is not None
        and args.input_contract != endpoint_contract.payload["input_contract"]
    ):
        raise ValueError("frozen endpoint contract requires manifest_harmonized inputs")
    adapters = _parse_csv(args.adapters)
    if not adapters:
        raise ValueError("at least one adapter is required")
    policy = GuardrailPolicy(
        enabled=args.enable_speed_truncation,
        baseline_adapter=args.guardrail_baseline.strip(),
        skip_ratio=args.guardrail_skip_ratio,
        minimum_slow_observations=args.guardrail_min_observations,
        truncate_on_timeout=not args.no_truncate_on_timeout,
    )
    memory_policy = MemoryPreflightPolicy(
        budget_fraction=args.memory_budget_fraction,
        reserve_bytes=int(args.memory_reserve_gib * GIB),
        user_cap_bytes=(
            int(args.memory_cap_gib * GIB) if args.memory_cap_gib is not None else None
        ),
        safety_factor=args.memory_safety_factor,
    )
    if policy.enabled and policy.baseline_adapter not in adapters:
        raise ValueError(
            "speed truncation requires the configured baseline adapter to be selected: "
            f"{policy.baseline_adapter!r}"
        )
    if policy.baseline_adapter in adapters:
        adapters = _ordered_adapters(adapters, policy.baseline_adapter)
    workloads = parse_workloads(args.workloads)
    if args.input_contract == HARMONIZED_INPUT_CONTRACT and any(
        workload.name == "ivh" for workload in workloads
    ):
        missing_ivh = [
            str(case["case_id"])
            for case in cases
            if str(case.get("modality") or "").lower() in {"ct", "synthetic"}
            and not case.get("ivh_image_abs")
        ]
        if missing_ivh:
            raise ValueError(
                "harmonized IVH tasks require a frozen mask-specific IVH image; "
                "missing: " + ", ".join(missing_ivh[:5])
            )
    if policy.enabled:
        _validate_baseline_capabilities(
            policy.baseline_adapter,
            adapters,
            workloads,
        )
    existing_run_id = _existing_run_id(args.report_dir) if args.resume else None
    run_id = args.run_id or existing_run_id or f"run_{int(time.time())}"
    report_dir = Path(
        args.report_dir or repo_root() / "results" / "runs" / run_id
    ).resolve()
    records_dir = report_dir / "records"
    ledger_path = report_dir / "benchmark.sqlite3"
    run_lock: Optional[RunLock] = None
    if not args.dry_run:
        # Acquiring the lock creates the report directory. No executable input
        # staging or other report mutation is allowed before this point.
        run_lock = RunLock(report_dir / ".benchmark.lock").acquire()
    try:
        (
            cases,
            input_identities,
            tasks,
            guardrail_groups,
            adapter_environments,
            machine,
            run_spec,
        ) = _plan_benchmark_execution(
            args=args,
            manifest=manifest,
            dataset_validation=dataset_validation,
            dataset=dataset,
            dataset_kind=dataset_kind,
            cases=cases,
            adapters=adapters,
            workloads=workloads,
            policy=policy,
            memory_policy=memory_policy,
            report_dir=report_dir,
            records_dir=records_dir,
            ledger_path=ledger_path,
            run_id=run_id,
            endpoint_contract=endpoint_contract,
        )

        print(
            f"Validated {len(cases)} selected {dataset_kind} cases "
            f"({dataset_validation['case_count']} total) and planned {len(tasks)} tasks "
            f"for adapters: {', '.join(adapters)}."
        )
        print(f"Immutable run fingerprint: {run_spec.run_fingerprint}")
        if args.dry_run:
            return 0

        records_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Benchmarking machine: {machine['cpu_model']} "
            f"({machine['cpu_count_physical']} physical cores, "
            f"{machine['memory_total_bytes'] / (1024**3):.1f} GB RAM)"
        )
    except BaseException:
        if run_lock is not None:
            run_lock.release()
        raise

    if run_lock is None:
        raise RuntimeError("executable benchmark did not acquire its run lock")
    with run_lock:
        with BenchmarkLedger(ledger_path) as ledger:
            session_started = time.perf_counter()
            ledger.initialize(run_spec, tasks, resume=args.resume)
            recovered = ledger.recover_running() if args.resume else 0
            if recovered:
                print(
                    f"Recovered {recovered} uncommitted running task(s) as interrupted."
                )
            requeued_failures = ledger.recover_failed() if args.resume else 0
            if requeued_failures:
                print(
                    f"Requeued {requeued_failures} failed task(s) for a fresh attempt."
                )
            host_observations = _record_host_observation(
                report_dir,
                machine=machine,
                run_fingerprint=run_spec.run_fingerprint,
            )
            session_observation = host_observations[-1]
            tasks_by_id = {task.task_id: task for task in tasks}
            if args.resume:
                verified = _verify_committed_payloads(
                    ledger,
                    run_spec.run_fingerprint,
                    run_id,
                    tasks_by_id,
                    adapter_environments,
                )
                if verified:
                    print(f"Verified {verified} committed measured payload(s).")

            existing_spec_path = report_dir / "run_spec.json"
            if existing_spec_path.is_file():
                existing_spec = json.loads(
                    existing_spec_path.read_text(encoding="utf-8")
                )
                if (
                    fingerprint(run_spec_identity(existing_spec))
                    != run_spec.run_fingerprint
                ):
                    raise RunSpecMismatch(
                        "run_spec.json differs from the requested immutable protocol"
                    )
            atomic_write_json(existing_spec_path, run_spec.to_dict())
            ledger.set_run_status("running")

            if args.resume:
                for task in tasks:
                    if _try_adopt_payload(
                        ledger,
                        task,
                        _payload_path(records_dir, task),
                        run_spec.run_fingerprint,
                        run_id,
                        adapter_environments.get(task.adapter),
                    ):
                        print(
                            f"Recovered committed payload for {task.case_id} "
                            f"{task.adapter} {task.workload_key}."
                        )

            fresh_process_references = _load_fresh_process_references(
                ledger,
                tasks_by_id,
            )

            # Reconstruct any observation that was interrupted after a measured
            # payload commit, and resolve rotated blocks whose candidate arrived
            # before its baseline.
            _reconcile_guardrails(ledger, policy, guardrail_groups)
            _reconcile_timeout_cutoffs(ledger, policy, tasks_by_id)
            _print_ledger_progress(
                ledger,
                total_tasks=len(tasks),
                project_total_tasks=args.project_total_tasks,
                project_task_offset=args.project_task_offset,
                session_started=session_started,
                task_timeout_sec=args.timeout,
            )

            memory_history: Dict[Tuple[str, str], List[Mapping[str, Any]]] = (
                defaultdict(list)
            )
            for prior_record in ledger.records():
                if prior_record.get("task_status") != STATUS_MEASURED:
                    continue
                memory_history[
                    (
                        str(prior_record.get("adapter") or ""),
                        str(prior_record.get("workload") or ""),
                    )
                ].append(prior_record)

            completed_since_checkpoint = 0
            aborted_for_failure = False
            interrupted = False
            current_task: Optional[TaskSpec] = None
            finish_task_power: Callable[[], Dict[str, Any]] | None = None

            with _SignalStop() as stop:
                try:
                    for task in tasks:
                        status = ledger.status(task.task_id)
                        if status in TERMINAL_STATUSES:
                            continue
                        if status not in RECOVERABLE_STATUSES:
                            raise RuntimeError(
                                f"unexpected task status for {task.task_id}: {status}"
                            )
                        if stop.requested:
                            interrupted = True
                            break

                        task_power_snapshots = [observe_task_power_state()]

                        def finish_task_power() -> Dict[str, Any]:
                            if len(task_power_snapshots) == 1:
                                task_power_snapshots.append(observe_task_power_state())
                            return _task_power_provenance(
                                session_observation,
                                task_power_snapshots[0],
                                task_power_snapshots[1],
                            )

                        unsupported_reason = _declared_unsupported_reason(task)
                        if unsupported_reason is not None:
                            record = _terminal_record(
                                task,
                                run_id,
                                STATUS_UNSUPPORTED,
                                error=unsupported_reason,
                            )
                            record.update(finish_task_power())
                            ledger.mark_terminal(
                                task.task_id,
                                STATUS_UNSUPPORTED,
                                record,
                                error=unsupported_reason,
                            )
                            print(f"UNSUPPORTED: {unsupported_reason}")
                            completed_since_checkpoint += 1
                            if completed_since_checkpoint >= args.checkpoint_interval:
                                _checkpoint(
                                    ledger=ledger,
                                    report_dir=report_dir,
                                    run_id=run_id,
                                    machine=machine,
                                    args=args,
                                    run_spec=run_spec,
                                    run_status="running",
                                )
                                completed_since_checkpoint = 0
                            continue

                        _verify_execution_bindings(run_spec)

                        scope = policy.scope_key(
                            task.adapter,
                            task.workload_key,
                            task.guardrail_group,
                        )
                        # The selected baseline must remain observable at every
                        # complexity because it is the denominator for exact
                        # matched runtime ratios. Optional development-only
                        # truncation can never apply to the baseline itself.
                        decision = (
                            ledger.guardrail_decision(scope)
                            if policy.should_compare(task.adapter)
                            else None
                        )
                        timeout_scope = policy.timeout_scope_key(
                            task.adapter,
                            task.workload_key,
                            task.guardrail_group,
                        )
                        timeout_complexity_metric, timeout_complexity = (
                            _timeout_cutoff_complexity(task)
                        )
                        timeout_decision = (
                            ledger.timeout_cutoff(timeout_scope)
                            if policy.truncate_on_timeout
                            else None
                        )
                        memory_key = (
                            task.adapter,
                            task.workload_key,
                        )
                        virtual_memory = psutil.virtual_memory()
                        memory_preflight = evaluate_memory_preflight(
                            adapter=task.adapter,
                            workload_key=task.workload_key,
                            guardrail_group=task.guardrail_group,
                            input_uncompressed_bytes=int(
                                task.input_uncompressed_bytes or 0
                            ),
                            mask_voxels=int(task.mask_voxels),
                            available_bytes=int(virtual_memory.available),
                            total_bytes=int(virtual_memory.total),
                            prior_records=memory_history[memory_key],
                            policy=memory_policy,
                        )
                        if (
                            timeout_decision is not None
                            and timeout_complexity_metric
                            == str(timeout_decision["complexity_metric"])
                            and timeout_complexity
                            > int(timeout_decision["cutoff_complexity"])
                        ):
                            reason = str(timeout_decision["reason"])
                            record = _terminal_record(
                                task,
                                run_id,
                                STATUS_SKIPPED_TIMEOUT,
                                error=f"Skipped above persisted timeout cutoff: {reason}",
                                policy_reason=reason,
                            )
                            record.update(finish_task_power())
                            record.update(memory_preflight)
                            record.update(
                                {
                                    "timeout_cutoff_complexity": int(
                                        timeout_decision["cutoff_complexity"]
                                    ),
                                    "timeout_cutoff_complexity_metric": str(
                                        timeout_decision["complexity_metric"]
                                    ),
                                    "timeout_cutoff_evidence_task_id": str(
                                        timeout_decision["evidence_task_id"]
                                    ),
                                }
                            )
                            ledger.mark_terminal(
                                task.task_id,
                                STATUS_SKIPPED_TIMEOUT,
                                record,
                                error=str(record["error"]),
                            )
                            print(
                                f"SKIPPED TIMEOUT CUTOFF {task.case_id} "
                                f"{task.adapter} {task.workload_key}: "
                                f"{timeout_complexity_metric}={timeout_complexity} "
                                "is strictly above observed timeout boundary "
                                f"{timeout_decision['cutoff_complexity']}."
                            )
                            completed_since_checkpoint += 1
                        elif decision is not None and task.complexity > int(
                            decision["cutoff_complexity"]
                        ):
                            reason = str(decision["reason"])
                            record = _terminal_record(
                                task,
                                run_id,
                                STATUS_SKIPPED,
                                error=f"Skipped by persisted performance policy: {reason}",
                                policy_reason=reason,
                            )
                            record.update(finish_task_power())
                            record.update(memory_preflight)
                            ledger.mark_terminal(
                                task.task_id,
                                STATUS_SKIPPED,
                                record,
                                error=str(record["error"]),
                            )
                            print(
                                f"SKIPPED {task.case_id} {task.adapter} {task.workload_key}: "
                                "strictly larger than policy cutoff."
                            )
                            completed_since_checkpoint += 1
                        else:
                            current_task = task
                            attempt = ledger.mark_running(task.task_id)
                            print(
                                f"RUN {task.ordinal}/{len(tasks)} {task.case_id} "
                                f"{task.adapter} {task.workload_key} repeat {task.repeat} attempt {attempt}"
                            )
                            try:
                                _verify_staged_task_inputs(task, input_identities)
                                baseline = ledger.measured_match(
                                    case_id=task.case_id,
                                    adapter=policy.baseline_adapter,
                                    workload_key=task.workload_key,
                                    repeat=task.repeat,
                                )
                                baseline_duration = (
                                    None
                                    if baseline is None
                                    else _safe_float(baseline["duration_sec"])
                                )
                                relative_cutoff = (
                                    None
                                    if baseline_duration is None
                                    or not policy.should_compare(task.adapter)
                                    else baseline_duration * policy.skip_ratio
                                )

                                def report_active(snapshot: Mapping[str, Any]) -> None:
                                    ready_elapsed = _safe_float(
                                        snapshot.get("ready_elapsed_sec")
                                    )
                                    iteration_elapsed = _safe_float(
                                        snapshot.get("iteration_elapsed_sec")
                                    )
                                    cutoff_text = "awaiting matched baseline"
                                    if relative_cutoff is not None:
                                        elapsed_for_cutoff = iteration_elapsed or 0.0
                                        cutoff_text = (
                                            f"{policy.skip_ratio:g}x comparison threshold="
                                            f"{_human_duration(relative_cutoff)}, "
                                            f"remaining≈{_human_duration(max(0.0, relative_cutoff - elapsed_for_cutoff))}"
                                            " (current task is not aborted)"
                                        )
                                    timeout_text = "none"
                                    if args.timeout is not None:
                                        timeout_elapsed = (
                                            ready_elapsed
                                            if ready_elapsed is not None
                                            else _safe_float(
                                                snapshot.get("host_elapsed_sec")
                                            )
                                            or 0.0
                                        )
                                        timeout_text = _human_duration(
                                            max(0.0, args.timeout - timeout_elapsed)
                                        )
                                    print(
                                        f"ACTIVE {task.ordinal}/{len(tasks)} "
                                        f"case={task.case_id} adapter={task.adapter} "
                                        f"workload={task.workload} "
                                        f"repeat={task.repeat} attempt={attempt} "
                                        f"phase={snapshot.get('phase')} "
                                        f"iteration={snapshot.get('current_iteration') or '-'} "
                                        f"completed_observations={snapshot.get('completed_iterations', 0)}/"
                                        f"{task.timing_observations}; host="
                                        f"{_human_duration(_safe_float(snapshot.get('host_elapsed_sec')))}; "
                                        f"current_call={_human_duration(iteration_elapsed)}; "
                                        f"timeout_remaining={timeout_text}; {cutoff_text}",
                                        flush=True,
                                    )

                                try:
                                    payload, metrics = run_adapter_process(
                                        task.adapter,
                                        image=task.image_path,
                                        mask=task.mask_path,
                                        image_sha256=task.image_sha256,
                                        source_image_sha256=task.source_image_sha256,
                                        mask_sha256=task.mask_sha256,
                                        input_contract=task.input_contract,
                                        input_representation_id=task.representation_id,
                                        representation_derivation_sha256=(
                                            task.representation_derivation_sha256
                                        ),
                                        configured_levels=task.configured_levels,
                                        occupied_levels=task.occupied_levels,
                                        modality=task.modality,
                                        discretization=task.discretization,
                                        aggregation=REQUIRED_AGGREGATION,
                                        bins=task.bins,
                                        bin_width=task.bin_width,
                                        intensity_min=task.intensity_min,
                                        intensity_max=task.intensity_max,
                                        timeout=args.timeout,
                                        families=list(task.scheduled_families),
                                        benchmark_workload=task.workload,
                                        iterations=task.timing_observations,
                                        include_values=True,
                                        stop_requested=lambda: stop.requested,
                                        termination_grace=args.termination_grace,
                                        progress_callback=report_active,
                                        progress_interval=args.progress_interval,
                                        thread_environment=dict(run_spec.thread_policy)[
                                            "environment"
                                        ],
                                    )
                                finally:
                                    _verify_staged_task_inputs(
                                        task,
                                        input_identities,
                                    )
                                    _verify_execution_bindings(run_spec)
                                feature_count = _validate_measured_result(
                                    payload,
                                    metrics,
                                    expected_task=task,
                                    expected_software=adapter_environments.get(
                                        task.adapter
                                    ),
                                )
                                repeat_reference = (
                                    _validate_or_register_fresh_process_result(
                                        fresh_process_references,
                                        task=task,
                                        payload=payload,
                                    )
                                )
                                metrics.update(
                                    {
                                        "fresh_process_reference_task_id": (
                                            None
                                            if repeat_reference is None
                                            else repeat_reference["task_id"]
                                        ),
                                        "fresh_process_reference_repeat": (
                                            None
                                            if repeat_reference is None
                                            else repeat_reference["repeat"]
                                        ),
                                        "fresh_process_result_equivalence_passed": True,
                                        "fresh_process_result_equivalence_rtol": (
                                            RESULT_EQUIVALENCE_RTOL
                                        ),
                                        "fresh_process_result_equivalence_atol": (
                                            RESULT_EQUIVALENCE_ATOL
                                        ),
                                    }
                                )
                            except RunIntegrityError:
                                raise
                            except UnsupportedTaskError as exc:
                                record = _terminal_record(
                                    task,
                                    run_id,
                                    STATUS_UNSUPPORTED,
                                    attempt=attempt,
                                    error=str(exc),
                                )
                                record.update(finish_task_power())
                                record.update(memory_preflight)
                                ledger.mark_terminal(
                                    task.task_id,
                                    STATUS_UNSUPPORTED,
                                    record,
                                    error=str(exc),
                                )
                                print(f"UNSUPPORTED: {exc}")
                            except AdapterTimeout as exc:
                                record = _terminal_record(
                                    task,
                                    run_id,
                                    STATUS_TIMED_OUT,
                                    attempt=attempt,
                                    error=str(exc),
                                    censor_lower_bound_sec=exc.elapsed_seconds,
                                )
                                record.update(finish_task_power())
                                record.update(memory_preflight)
                                record["timeout_phase"] = exc.phase
                                record["partial_duration_samples_sec"] = list(
                                    exc.partial_duration_samples_sec
                                )
                                record["partial_cpu_time_samples_sec"] = list(
                                    exc.partial_cpu_time_samples_sec
                                )
                                record["partial_completed_iterations"] = len(
                                    exc.partial_duration_samples_sec
                                )
                                if policy.truncate_on_timeout:
                                    record["timeout_cutoff_complexity"] = (
                                        timeout_complexity
                                    )
                                    record["timeout_cutoff_complexity_metric"] = (
                                        timeout_complexity_metric
                                    )
                                    record["timeout_cutoff_evidence_task_id"] = (
                                        task.task_id
                                    )
                                ledger.mark_terminal(
                                    task.task_id,
                                    STATUS_TIMED_OUT,
                                    record,
                                    error=str(exc),
                                )
                                if policy.truncate_on_timeout:
                                    timeout_reason = (
                                        f"task exceeded the configured "
                                        f"{exc.timeout_seconds:.6g}s {exc.phase} "
                                        "process-tree safety limit"
                                    )
                                    ledger.activate_timeout_cutoff(
                                        scope_key=timeout_scope,
                                        adapter=task.adapter,
                                        workload_key=task.workload_key,
                                        guardrail_group=task.guardrail_group,
                                        complexity_metric=timeout_complexity_metric,
                                        cutoff_complexity=timeout_complexity,
                                        reason=timeout_reason,
                                        evidence_task_id=task.task_id,
                                    )
                                print(f"TIMED OUT (censored, not measured): {exc}")
                            except AdapterInterrupted as exc:
                                record = _terminal_record(
                                    task,
                                    run_id,
                                    STATUS_INTERRUPTED,
                                    attempt=attempt,
                                    error=str(exc),
                                    censor_lower_bound_sec=exc.elapsed_seconds,
                                )
                                record.update(finish_task_power())
                                record.update(memory_preflight)
                                record["partial_duration_samples_sec"] = list(
                                    exc.partial_duration_samples_sec
                                )
                                record["partial_cpu_time_samples_sec"] = list(
                                    exc.partial_cpu_time_samples_sec
                                )
                                record["partial_completed_iterations"] = len(
                                    exc.partial_duration_samples_sec
                                )
                                ledger.mark_terminal(
                                    task.task_id,
                                    STATUS_INTERRUPTED,
                                    record,
                                    error=str(exc),
                                )
                                interrupted = True
                                print(
                                    f"INTERRUPTED: {task.case_id} {task.adapter} {task.workload_key}"
                                )
                            except Exception as exc:
                                record = _terminal_record(
                                    task,
                                    run_id,
                                    STATUS_FAILED,
                                    attempt=attempt,
                                    error=str(exc),
                                )
                                record.update(finish_task_power())
                                record.update(memory_preflight)
                                ledger.mark_terminal(
                                    task.task_id,
                                    STATUS_FAILED,
                                    record,
                                    error=str(exc),
                                )
                                print(f"FAILED: {exc}")
                                if not args.keep_going:
                                    aborted_for_failure = True
                            else:
                                metrics.update(memory_preflight)
                                metrics.update(finish_task_power())
                                record = _terminal_record(
                                    task,
                                    run_id,
                                    STATUS_MEASURED,
                                    attempt=attempt,
                                )
                                record.update(metrics)
                                record["task_status"] = STATUS_MEASURED
                                record["success"] = True
                                record["feature_count"] = feature_count
                                record["attempted_feature_count"] = feature_count
                                record["finite_feature_count"] = feature_count
                                software = payload.get("software")
                                if isinstance(software, dict):
                                    record["adapter_distribution"] = software.get(
                                        "distribution"
                                    )
                                    record["adapter_version"] = software.get("version")
                                payload_path = _payload_path(records_dir, task)
                                payload_sha256 = _write_measured_payload(
                                    payload_path,
                                    payload=payload,
                                    metrics=metrics,
                                    task=task,
                                    record=record,
                                    run_fingerprint=run_spec.run_fingerprint,
                                )
                                ledger.mark_terminal(
                                    task.task_id,
                                    STATUS_MEASURED,
                                    record,
                                    duration_sec=float(record["duration_sec"]),
                                    payload_path=ledger.portable_payload_path(
                                        payload_path
                                    ),
                                    payload_sha256=payload_sha256,
                                )
                                memory_history[memory_key].append(record)
                                _reconcile_guardrail_group(
                                    ledger,
                                    policy,
                                    guardrail_groups[_guardrail_comparison_key(task)],
                                )
                                print(
                                    f"MEASURED {record['duration_sec']:.6g}s, "
                                    f"{feature_count} features, "
                                    f"{int(record['peak_rss_bytes'] or 0) / (1024**2):.1f} MiB; "
                                    f"wall samples="
                                    f"{payload.get('timing', {}).get('duration_samples_sec')} "
                                    f"min={payload.get('timing', {}).get('duration_min_sec')} "
                                    f"SD={payload.get('timing', {}).get('duration_std_sec')}"
                                )

                            current_task = None
                            completed_since_checkpoint += 1

                        _print_ledger_progress(
                            ledger,
                            total_tasks=len(tasks),
                            project_total_tasks=args.project_total_tasks,
                            project_task_offset=args.project_task_offset,
                            session_started=session_started,
                            task_timeout_sec=args.timeout,
                        )

                        if completed_since_checkpoint >= args.checkpoint_interval:
                            _checkpoint(
                                ledger=ledger,
                                report_dir=report_dir,
                                run_id=run_id,
                                machine=machine,
                                args=args,
                                run_spec=run_spec,
                                run_status="running",
                            )
                            completed_since_checkpoint = 0
                        if interrupted or aborted_for_failure:
                            break
                except BaseException as exc:
                    if (
                        current_task is not None
                        and ledger.status(current_task.task_id) == STATUS_RUNNING
                    ):
                        row = ledger.task_row(current_task.task_id)
                        record = _terminal_record(
                            current_task,
                            run_id,
                            STATUS_INTERRUPTED,
                            attempt=int(row["attempt"]),
                            error=f"controller error before commit: {exc}",
                        )
                        if finish_task_power is not None:
                            record.update(finish_task_power())
                        ledger.mark_terminal(
                            current_task.task_id,
                            STATUS_INTERRUPTED,
                            record,
                            error=str(record["error"]),
                        )
                    ledger.set_run_status("interrupted")
                    try:
                        _checkpoint(
                            ledger=ledger,
                            report_dir=report_dir,
                            run_id=run_id,
                            machine=machine,
                            args=args,
                            run_spec=run_spec,
                            run_status="interrupted",
                        )
                    except Exception:
                        pass
                    raise

            if interrupted or stop.requested:
                run_status = "interrupted"
            elif aborted_for_failure:
                run_status = "failed"
            else:
                counts = ledger.status_counts()
                incomplete = counts.get(STATUS_PENDING, 0) + counts.get(
                    STATUS_RUNNING, 0
                )
                if incomplete:
                    run_status = "interrupted"
                elif counts.get(STATUS_FAILED, 0) or counts.get(STATUS_TIMED_OUT, 0):
                    run_status = "completed_with_failures"
                else:
                    run_status = "completed"
            ledger.set_run_status(run_status)
            _checkpoint(
                ledger=ledger,
                report_dir=report_dir,
                run_id=run_id,
                machine=machine,
                args=args,
                run_spec=run_spec,
                run_status=run_status,
                final=run_status.startswith("completed"),
            )

            if run_status == "interrupted":
                signum = stop.signal_number or int(signal.SIGINT)
                print(
                    "Benchmark interrupted safely; resume with the identical run specification."
                )
                return 128 + signum
            if run_status == "failed":
                print("Benchmark stopped after a failed task; state was checkpointed.")
                return 1
            print(f"Benchmark {run_status}. Outputs saved to: {report_dir}")
            if run_status == "completed_with_failures" and ledger.status_counts().get(
                STATUS_FAILED, 0
            ):
                print(
                    "One or more adapter tasks failed; resume the identical command "
                    "to requeue them."
                )
                return 1
            return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

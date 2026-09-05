from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import psutil

from bench import run
from bench.adapters.protocol import ADAPTER_PROTOCOL_VERSION
from bench.adapters.protocol import timing_contract_metadata
from bench.adapters.registry import get_adapter
from bench.benchmark_ledger import (
    BenchmarkLedger,
    RunAlreadyExists,
    RunIntegrityError,
    RunSpecMismatch,
)
from bench.benchmark_models import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_MEASURED,
    STATUS_SKIPPED,
    STATUS_SKIPPED_TIMEOUT,
    STATUS_TIMED_OUT,
)
from bench.benchmark_workloads import WORKLOADS, WORKLOAD_BY_NAME, WORKLOAD_ORDER
from bench.ibsi_families import FAMILY_ORDER


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(duration: float) -> dict:
    return {
        "duration_sec": duration,
        "duration_mean_sec": duration,
        "duration_median_sec": duration,
        "duration_std_sec": 0.0,
        "duration_max_sec": duration,
        "cpu_time_sec": duration,
        "cpu_time_mean_sec": duration,
        "cpu_time_median_sec": duration,
        "cpu_time_std_sec": 0.0,
        "cpu_time_max_sec": duration,
        "measured_iterations": 1,
        "measured_observations": 1,
        "total_iterations": 1,
        "calls_per_observation": 1,
        "calibration_calls": 0,
        "calibration_duration_sec": 0.0,
        "calibration_cpu_time_sec": 0.0,
        "calibration_per_call_sec": None,
        "calibration_headroom_factor": 2.0,
        "measured_calculation_calls": 1,
        "total_calculation_calls": 1,
        "peak_rss_bytes": 1024,
        "host_peak_rss_bytes": 1024,
        "worker_ready_rss_bytes": 768,
        "calculation_peak_rss_bytes": 1024,
        "incremental_calculation_peak_rss_bytes": 256,
        "host_wall_time_sec": duration,
        "adapter_event_count": 8,
        "adapter_stderr": "",
        "memory_phase_observation_status": "complete",
        "timing_source": "fake",
        "timing_scope": "fake",
        "memory_scope": "fake",
    }


def contractual_timing(duration: float, measured: int) -> dict:
    total_calls = measured + 1
    return {
        "duration_sec": duration,
        "duration_min_sec": duration,
        "duration_mean_sec": duration,
        "duration_median_sec": duration,
        "duration_std_sec": 0.0,
        "duration_max_sec": duration,
        "cpu_time_sec": duration,
        "cpu_time_min_sec": duration,
        "cpu_time_mean_sec": duration,
        "cpu_time_median_sec": duration,
        "cpu_time_std_sec": 0.0,
        "cpu_time_max_sec": duration,
        "measured_iterations": measured,
        "measured_observations": measured,
        "warmup_iterations": 1,
        "total_iterations": measured + 1,
        "calls_per_observation": 1,
        "calibration_calls": 0,
        "calibration_rounds": 0,
        "calibration_duration_sec": 0.0,
        "calibration_cpu_time_sec": 0.0,
        "calibration_per_call_sec": None,
        "calibration_window_samples_sec": [],
        "calibration_per_call_samples_sec": [],
        "calibration_calls_per_round": [],
        "calibration_stability_cv": None,
        "calibration_stability_span": None,
        "calibration_stable": True,
        "calibration_minimum_rounds": 3,
        "calibration_maximum_rounds": 24,
        "calibration_cv_threshold": 0.05,
        "calibration_span_ratio": 1.1,
        "calibration_headroom_factor": 2.0,
        "measured_calculation_calls": measured,
        "total_calculation_calls": total_calls,
        "target_observation_window_sec": 0.05,
        "maximum_calls_per_observation": 4096,
        "minimum_observation_window_sec": duration,
        "result_equivalence_checks": total_calls - 1,
        "result_equivalence_passed": True,
        "result_equivalence_rtol": 1e-9,
        "result_equivalence_atol": 1e-12,
        "duration_samples_sec": [duration] * measured,
        "cpu_time_samples_sec": [duration] * measured,
        "observation_window_samples_sec": [duration] * measured,
        "cpu_observation_window_samples_sec": [duration] * measured,
        "preparation_samples_sec": [0.0] * measured,
        "finalization_samples_sec": [0.0] * measured,
        "warmup_duration_sec": duration,
        "warmup_cpu_time_sec": duration,
        "warmup_preparation_sec": 0.0,
        "warmup_finalization_sec": 0.0,
    }


def single_window_contractual_timing(duration: float, measured: int) -> dict:
    """Return the valid long-call branch of the adaptive timing contract."""

    timing = contractual_timing(duration, measured)
    total_calls = measured + 2
    timing.update(
        {
            "calibration_calls": 1,
            "calibration_rounds": 1,
            "calibration_duration_sec": duration,
            "calibration_cpu_time_sec": duration,
            "calibration_per_call_sec": duration,
            "calibration_window_samples_sec": [duration],
            "calibration_per_call_samples_sec": [duration],
            "calibration_calls_per_round": [1],
            "calibration_stability_cv": 0.0,
            "calibration_stability_span": 1.0,
            "total_calculation_calls": total_calls,
            "result_equivalence_checks": total_calls - 1,
        }
    )
    return timing


CONTRACTUAL_METRIC_KEYS = (
    "measured_iterations",
    "measured_observations",
    "warmup_iterations",
    "total_iterations",
    "calls_per_observation",
    "calibration_calls",
    "calibration_rounds",
    "calibration_duration_sec",
    "calibration_cpu_time_sec",
    "calibration_per_call_sec",
    "calibration_headroom_factor",
    "calibration_stability_cv",
    "calibration_stability_span",
    "calibration_stable",
    "measured_calculation_calls",
    "total_calculation_calls",
    "minimum_observation_window_sec",
    "result_equivalence_checks",
    "result_equivalence_passed",
    "result_equivalence_rtol",
    "result_equivalence_atol",
)


def metrics_with_contract(duration: float, measured: int) -> dict:
    observed = metrics(duration)
    timing = contractual_timing(duration, measured)
    for key in CONTRACTUAL_METRIC_KEYS:
        observed[key] = timing[key]
    return observed


def adapter_payload(
    adapter: str,
    families: str | list[str],
    feature_names: list[str] | None = None,
    benchmark_workload: str | None = None,
) -> dict:
    """Return a complete current-protocol payload for runner unit tests."""

    capabilities = None
    try:
        capabilities = get_adapter(adapter)
        distribution = capabilities.distribution
        selection_mode = capabilities.selection_mode
    except ValueError:
        distribution = f"test-{adapter}"
        selection_mode = "unit-test"
    requested = [families] if isinstance(families, str) else list(families)
    supported = (
        [family for family in requested if capabilities.supports(family)]
        if capabilities is not None
        else requested
    )
    unsupported = [family for family in requested if family not in supported]
    default_features = [f"{family}-feature" for family in supported]
    return {
        "schema_version": ADAPTER_PROTOCOL_VERSION,
        "adapter": adapter,
        "software": {"distribution": distribution, "version": "test-1.0"},
        "selection": {
            "requested_families": supported,
            "unsupported_families": unsupported,
            "mode": selection_mode,
            "benchmark_workload": benchmark_workload,
        },
        "features": {"all": feature_names or default_features},
    }


def harmonized_case(case: dict) -> dict:
    """Attach the current stored representations to a compact unit-test case."""

    bound = dict(case)
    bound.update(
        {
            "discrete_image_abs": "/tmp/discrete.nii.gz",
            "discrete_image_sha256": "c" * 64,
            "ivh_image_abs": "/tmp/ivh.nii.gz",
            "ivh_image_sha256": "d" * 64,
            "raw_representation": "original_continuous_image",
            "texture_representation": {
                "id": "mask_specific_ibsi_fbn32",
                "configured_levels": 32,
                "occupied_levels": 32,
                "derivation_sha256": "e" * 64,
            },
            "ivh_representation": {
                "id": "mask_specific_ibsi_fbs1_ivh_indices",
                "configured_levels": 32,
                "occupied_levels": 32,
                "derivation_sha256": "f" * 64,
            },
        }
    )
    return bound


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.report = self.root / "run"
        unit_timing = {
            "untimed_warmup_calls_per_process": 1,
            "measured_observations_per_process": 3,
            "target_observation_window_seconds": 0.05,
            "maximum_calls_per_observation": 4096,
            "fresh_process_repeats": 1,
        }
        inventories = {}
        for adapter in (
            "pictologics",
            "pyradiomics",
            "mirp",
            "medimage",
            "zrad",
            "slow",
            "fake",
        ):
            try:
                capabilities = get_adapter(adapter)
                counts = {
                    family: int(capabilities.supports(family))
                    for family in FAMILY_ORDER
                }
            except ValueError:
                counts = {family: 1 for family in FAMILY_ORDER}
            inventories[adapter] = {
                "family_output_counts": counts,
                "workload_output_counts": {
                    workload.name: sum(counts[family] for family in workload.families)
                    for workload in WORKLOADS
                },
            }
        unit_contract = run.BenchmarkContract(
            contract_id="unit_test_benchmark",
            path=self.root / "unit-contract.json",
            sha256="a" * 64,
            payload={
                "input_contract": run.HARMONIZED_INPUT_CONTRACT,
                "timing": unit_timing,
                "adapter_inventories": inventories,
            },
        )
        contract_patch = mock.patch.object(
            run, "load_benchmark_contract", return_value=unit_contract
        )
        contract_patch.start()
        self.addCleanup(contract_patch.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_process_event_stream_delimits_calculation_memory(self) -> None:
        script = "\n".join(
            [
                "import json, time",
                "held = bytearray(8 * 1024 * 1024)",
                "def event(name):",
                "    print('BENCH_EVENT ' + json.dumps({'event': name}), flush=True)",
                "event('worker_ready')",
                "time.sleep(0.05)",
                "event('warmup_complete')",
                "event('calculation_start')",
                "extra = bytearray(8 * 1024 * 1024)",
                "time.sleep(0.10)",
                "event('calculation_complete')",
                "print(json.dumps({'ok': True}), flush=True)",
            ]
        )
        stdout, stderr, observed = run._run_process_command(
            [sys.executable, "-c", script],
            adapter_name="event-fixture",
            sample_interval=0.005,
            timeout=5.0,
        )

        self.assertIn('"ok": true', stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(observed["adapter_event_count"], 4)
        self.assertGreater(observed["worker_ready_rss_bytes"], 0)
        self.assertGreaterEqual(
            observed["calculation_peak_rss_bytes"],
            observed["worker_ready_rss_bytes"],
        )
        self.assertGreaterEqual(
            observed["host_peak_rss_bytes"],
            observed["worker_ready_rss_bytes"],
        )
        self.assertGreaterEqual(
            observed["host_peak_rss_bytes"],
            observed["calculation_peak_rss_bytes"],
        )
        self.assertEqual(
            observed["incremental_calculation_peak_rss_bytes"],
            observed["calculation_peak_rss_bytes"] - observed["worker_ready_rss_bytes"],
        )

    def test_event_samples_are_included_in_host_peak_rss(self) -> None:
        script = "\n".join(
            [
                "import json",
                "def event(name):",
                "    print('BENCH_EVENT ' + json.dumps({'event': name}), flush=True)",
                "event('worker_ready')",
                "event('calculation_start')",
                "event('calculation_complete')",
                "print(json.dumps({'ok': True}), flush=True)",
            ]
        )
        controller_thread = threading.current_thread()

        def sampled_rss(_process) -> int:
            if threading.current_thread() is controller_thread:
                return 1024
            return 8192

        with mock.patch.object(run, "_rss_bytes", side_effect=sampled_rss):
            _, _, observed = run._run_process_command(
                [sys.executable, "-c", script],
                adapter_name="event-peak-fixture",
                sample_interval=0.005,
                timeout=5.0,
            )

        self.assertEqual(observed["worker_ready_rss_bytes"], 8192)
        self.assertEqual(observed["calculation_peak_rss_bytes"], 8192)
        self.assertEqual(observed["host_peak_rss_bytes"], 8192)

    def test_process_progress_reports_phase_and_measured_iteration(self) -> None:
        script = "\n".join(
            [
                "import json, time",
                "def event(name, **extra):",
                "    print('BENCH_EVENT ' + json.dumps({'event': name, **extra}), flush=True)",
                "event('worker_ready')",
                "event('calculation_start', iteration=1)",
                "time.sleep(0.08)",
                "event('calculation_complete', iteration=1)",
                "print(json.dumps({'ok': True}), flush=True)",
            ]
        )
        snapshots = []
        run._run_process_command(
            [sys.executable, "-c", script],
            adapter_name="progress-fixture",
            sample_interval=0.002,
            timeout=5.0,
            progress_callback=snapshots.append,
            progress_interval=0.01,
        )
        self.assertTrue(snapshots)
        self.assertTrue(
            any(snapshot["phase"] == "calculation" for snapshot in snapshots)
        )
        self.assertTrue(
            any(snapshot["current_iteration"] == 1 for snapshot in snapshots)
        )

    def test_declared_unsupported_precedes_resource_preflight(self) -> None:
        task = mock.Mock(
            adapter="pyradiomics",
            workload="ivh",
            scheduled_families=("ivh",),
        )
        self.assertEqual(
            run._declared_unsupported_reason(task),
            "pyradiomics does not declare support for workload ivh",
        )

        spatial_task = mock.Mock(
            adapter="pyradiomics",
            workload="spatial_autocorrelation",
            scheduled_families=("morphology",),
        )
        self.assertEqual(
            run._declared_unsupported_reason(spatial_task),
            "pyradiomics does not declare support for workload spatial_autocorrelation",
        )

    def make_dataset(
        self,
        complexities: list[int],
        *,
        modalities: list[str] | None = None,
        synthetic_series: bool = False,
    ) -> None:
        case_modalities = modalities or (
            ["synthetic"] * len(complexities)
            if synthetic_series
            else ["ct"] * len(complexities)
        )
        cases = []
        files = []
        for index, complexity in enumerate(complexities, 1):
            size = 10 if complexity == 1000 else 20
            case_id = f"case-{index}"
            image = self.dataset / f"{case_id}-image.nii.gz"
            mask = self.dataset / f"{case_id}-mask.nii.gz"
            discrete = self.dataset / f"{case_id}-fbn32.nii.gz"
            ivh = self.dataset / f"{case_id}-ivh.nii.gz"
            image_data = np.arange(complexity, dtype=np.float32).reshape(
                (size, size, size)
            )
            mask_data = np.zeros(complexity, dtype=np.uint8)
            mask_data[: max(1, complexity // 2)] = 1
            nib.save(nib.Nifti1Image(image_data, np.eye(4)), image)
            nib.save(
                nib.Nifti1Image(mask_data.reshape((size, size, size)), np.eye(4)),
                mask,
            )
            indices = np.mod(np.arange(complexity), 32).astype(np.float32) + 1.0
            nib.save(
                nib.Nifti1Image(indices.reshape((size, size, size)), np.eye(4)),
                discrete,
            )
            nib.save(
                nib.Nifti1Image(indices.reshape((size, size, size)), np.eye(4)),
                ivh,
            )
            files.extend(
                [
                    {
                        "path": image.name,
                        "sha256": digest(image),
                        "bytes": image.stat().st_size,
                        "kind": "image",
                    },
                    {
                        "path": mask.name,
                        "sha256": digest(mask),
                        "bytes": mask.stat().st_size,
                        "kind": "mask",
                    },
                    {
                        "path": discrete.name,
                        "sha256": digest(discrete),
                        "bytes": discrete.stat().st_size,
                        "kind": "image",
                    },
                    {
                        "path": ivh.name,
                        "sha256": digest(ivh),
                        "bytes": ivh.stat().st_size,
                        "kind": "image",
                    },
                ]
            )
            cases.append(
                {
                    "case_id": case_id,
                    "modality": case_modalities[index - 1],
                    "size": size,
                    "variant": index,
                    "subject_id": "shared-series" if synthetic_series else case_id,
                    "mask_id": "M1" if synthetic_series else f"M{index}",
                    "mask_label": f"mask-{index}",
                    "image_path": image.name,
                    "discrete_image_path": discrete.name,
                    "ivh_image_path": ivh.name,
                    "mask_path": mask.name,
                    "image_sha256": digest(image),
                    "discrete_image_sha256": digest(discrete),
                    "ivh_image_sha256": digest(ivh),
                    "mask_sha256": digest(mask),
                    "raw_representation": "original_continuous_image",
                    "texture_representation": {
                        "id": "mask_specific_ibsi_fbn32",
                        "configured_levels": 32,
                        "occupied_levels": 32,
                        "derivation_sha256": "e" * 64,
                    },
                    "ivh_representation": {
                        "id": "mask_specific_ibsi_fbs1_ivh_indices",
                        "configured_levels": 32,
                        "occupied_levels": 32,
                        "derivation_sha256": "f" * 64,
                    },
                    "shape": [size, size, size],
                    "spacing": [1.0, 1.0, 1.0],
                    "orientation": ["R", "A", "S"],
                    "affine": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "image_voxels": complexity,
                    "complexity": complexity,
                    "mask_voxels": max(1, complexity // 2),
                    "mask_fraction": 0.5,
                }
            )
        (self.dataset / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "dataset": "unit-benchmark",
                    "dataset_kind": "synthetic" if synthetic_series else "real_world",
                    "files": files,
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )

    def args(self, *extra: str) -> list[str]:
        return [
            "--dataset-dir",
            str(self.dataset),
            "--report-dir",
            str(self.report),
            "--run-id",
            "unit-run",
            "--checkpoint-interval",
            "1000",
            "--repeats",
            "1",
            *extra,
        ]

    def patches(self, fake):
        return mock.patch.multiple(
            run,
            run_adapter_process=mock.DEFAULT,
            _adapter_environment_snapshots=mock.DEFAULT,
            _machine_info=mock.DEFAULT,
            _git_commit=mock.DEFAULT,
        )

    def test_default_workloads_are_all_reviewed_groups(self) -> None:
        parsed = run._make_parser().parse_args(["--dataset-dir", "unused"])
        self.assertEqual(
            [workload.name for workload in run.parse_workloads(parsed.workloads)],
            list(WORKLOAD_ORDER),
        )
        with self.assertRaises(ValueError):
            run.parse_workloads("none")
        with self.assertRaises(ValueError):
            run.parse_workloads("")

    def test_task_plan_rotates_adapter_order_across_exact_comparison_blocks(
        self,
    ) -> None:
        case = harmonized_case(
            {
                "case_id": "case",
                "modality": "ct",
                "size": 10,
                "variant": 1,
                "mask_id": "M1",
                "mask_label": "roi",
                "image_abs": "/tmp/image.nii.gz",
                "mask_abs": "/tmp/mask.nii.gz",
                "image_sha256": "a" * 64,
                "mask_sha256": "b" * 64,
                "shape": [10, 10, 10],
                "spacing": [1.0, 1.0, 1.0],
                "image_voxels": 1000,
                "mask_voxels": 500,
                "mask_fraction": 0.5,
                "complexity": 1000,
            }
        )
        tasks = run.build_task_plan(
            cases=[case],
            dataset="unit",
            adapters=["pictologics", "pyradiomics"],
            workloads=[WORKLOAD_BY_NAME["morphology"], WORKLOAD_BY_NAME["texture"]],
            repeats=2,
            timing_observations=2,
        )

        self.assertEqual(
            [(task.workload, task.repeat, task.adapter) for task in tasks],
            [
                ("morphology", 1, "pictologics"),
                ("morphology", 1, "pyradiomics"),
                ("texture", 1, "pyradiomics"),
                ("texture", 1, "pictologics"),
                ("morphology", 2, "pyradiomics"),
                ("morphology", 2, "pictologics"),
                ("texture", 2, "pictologics"),
                ("texture", 2, "pyradiomics"),
            ],
        )
        self.assertEqual([task.ordinal for task in tasks], list(range(1, 9)))

    def test_rotation_balances_every_adapter_across_execution_positions(self) -> None:
        case = harmonized_case(
            {
                "case_id": "case",
                "modality": "ct",
                "size": 2,
                "variant": 1,
                "mask_id": "M1",
                "mask_label": "roi",
                "image_abs": "/tmp/image.nii.gz",
                "mask_abs": "/tmp/mask.nii.gz",
                "image_sha256": "a" * 64,
                "mask_sha256": "b" * 64,
                "shape": [2, 2, 2],
                "spacing": [1.0, 1.0, 1.0],
                "image_voxels": 8,
                "mask_voxels": 4,
                "mask_fraction": 0.5,
                "complexity": 8,
            }
        )
        adapters = ["pictologics", "mirp", "medimage", "pyradiomics", "zrad"]
        tasks = run.build_task_plan(
            cases=[case],
            dataset="unit",
            adapters=adapters,
            workloads=[WORKLOAD_BY_NAME["texture"]],
            repeats=len(adapters),
            timing_observations=2,
        )
        blocks = [
            tasks[index : index + len(adapters)]
            for index in range(0, len(tasks), len(adapters))
        ]
        for adapter in adapters:
            self.assertEqual(
                sorted(
                    position
                    for block in blocks
                    for position, task in enumerate(block)
                    if task.adapter == adapter
                ),
                list(range(len(adapters))),
            )

    def test_environment_probe_captures_numpy_build_configuration(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", run._adapter_environment_probe_script()],
            capture_output=True,
            text=True,
            check=True,
        )
        snapshot = json.loads(completed.stdout)

        self.assertEqual(snapshot["python_version"], sys.version.split()[0])
        self.assertTrue(snapshot["numpy_config"]["available"])
        self.assertEqual(snapshot["numpy_config"]["version"], np.__version__)
        self.assertTrue(snapshot["numpy_config"]["show_config"].strip())

    def test_execution_source_scope_binds_transitive_and_lock_dependencies(
        self,
    ) -> None:
        sources = run._benchmark_execution_sources(["pyradiomics"])
        for required in (
            "pyproject.toml",
            "poetry.lock",
            "bench/run.py",
            "bench/benchmark_contract.py",
            "bench/benchmark_eta.py",
            "bench/benchmark_memory.py",
            "bench/benchmark_workloads.py",
            "bench/ibsi_mapping.py",
            "bench/ibsi_codes.py",
            "bench/power_provenance.py",
            "bench/adapters/pyradiomics_adapter.py",
        ):
            self.assertIn(required, sources)
        self.assertNotIn("bench/report.py", sources)
        self.assertFalse(any(path.startswith("bench/compliance/") for path in sources))

    def test_darwin_cpu_probe_uses_chip_without_persisting_hardware_report(
        self,
    ) -> None:
        hardware_report = subprocess.CompletedProcess(
            args=["system_profiler"],
            returncode=0,
            stdout=(
                "Hardware:\n"
                "    Chip: Apple M4 Pro\n"
                "    Serial Number (system): must-not-be-recorded\n"
            ),
            stderr="",
        )
        with (
            mock.patch.object(run.platform, "system", return_value="Darwin"),
            mock.patch.object(run.platform, "processor", return_value="arm"),
            mock.patch.object(run.subprocess, "run", return_value=hardware_report),
        ):
            self.assertEqual(run._auto_cpu_model(), "Apple M4 Pro")

    def test_machine_probe_is_anonymous_and_rejects_bogus_frequency(self) -> None:
        frequency = mock.Mock(current=4.0, max=4.0)
        memory = mock.Mock(total=48 * 1024**3)
        with (
            mock.patch.object(run, "_auto_cpu_model", return_value="Apple M4 Pro"),
            mock.patch.object(
                run.socket, "gethostname", return_value="private-device-id"
            ),
            mock.patch.object(run.psutil, "cpu_freq", return_value=frequency),
            mock.patch.object(run.psutil, "cpu_count", side_effect=[14, None]),
            mock.patch.object(run.os, "cpu_count", return_value=14),
            mock.patch.object(run.psutil, "virtual_memory", return_value=memory),
        ):
            machine = run._machine_info()

        self.assertNotIn("hostname", machine)
        self.assertRegex(machine["machine_id"], r"^anonymous-[0-9a-f]{16}$")
        self.assertNotIn("private-device-id", json.dumps(machine))
        self.assertEqual(machine["machine_label"], "Apple M4 Pro")
        self.assertIsNone(machine["cpu_current_ghz"])
        self.assertIsNone(machine["cpu_max_ghz"])
        self.assertEqual(machine["cpu_count_logical"], 14)

        with self.assertRaisesRegex(ValueError, "cpu_base_ghz"):
            run._machine_info(cpu_base_ghz=0.004)

    def test_frozen_host_profile_is_part_of_benchmark_machine_identity(self) -> None:
        machine = run._machine_info(
            machine_id="test-host",
            host_profile_id="test-host",
            host_profile_sha256="a" * 64,
            host_settings_json='{"power_source":"AC Power"}',
        )
        identity = run._benchmark_machine_identity(machine)
        self.assertEqual(identity["host_profile_id"], "test-host")
        self.assertEqual(identity["host_profile_sha256"], "a" * 64)
        self.assertEqual(identity["host_settings"], {"power_source": "AC Power"})

        with self.assertRaisesRegex(ValueError, "valid JSON"):
            run._machine_info(host_settings_json="not-json")
        with self.assertRaisesRegex(ValueError, "provided together"):
            run._machine_info(host_profile_id="test-host")

    def test_task_power_provenance_detects_an_in_task_mode_change(self) -> None:
        provenance = run._task_power_provenance(
            {
                "session_index": 3,
                "observed_at_utc": "2026-09-02T10:00:00Z",
            },
            {
                "observed_at_utc": "2026-09-02T10:01:00Z",
                "power_mode_tag": "macos-high-power-pmset-2",
                "energy_mode": "high_power",
                "energy_mode_observation_status": "observed",
                "power_source": "AC Power",
                "pmset_lowpowermode": 2,
                "probe_errors": [],
                "probe_diagnostics": {"custom_returncodes": [0]},
            },
            {
                "observed_at_utc": "2026-09-02T10:02:00Z",
                "power_mode_tag": "macos-automatic-pmset-0",
                "energy_mode": "automatic",
                "energy_mode_observation_status": "observed",
                "power_source": "AC Power",
                "pmset_lowpowermode": 0,
                "probe_errors": [],
                "probe_diagnostics": {"custom_returncodes": [0, 0]},
            },
        )

        self.assertEqual(provenance["host_session_index"], 3)
        self.assertTrue(provenance["host_power_mode_changed_during_task"])
        self.assertEqual(provenance["host_power_mode_tag"], "mixed-within-task")
        self.assertEqual(
            provenance["host_power_probe_diagnostics"],
            {
                "start": {"custom_returncodes": [0]},
                "end": {"custom_returncodes": [0, 0]},
            },
        )
        self.assertEqual(provenance["host_energy_mode"], "mixed")
        self.assertEqual(
            provenance["host_power_observation_scope"],
            "immediately_before_and_after_task",
        )

    def test_qc_reports_non_measured_outcomes_by_workload_and_adapter(self) -> None:
        common = {
            "case_id": "case-1",
            "dataset": "test",
            "adapter": "pictologics",
            "workload": "texture",
            "requested_families": ["glcm", "glrlm"],
            "repeat": 1,
            "host_session_index": 1,
            "host_power_observation_scope": "immediately_before_and_after_task",
            "host_power_start_mode_tag": "macos-high-power-pmset-2",
            "host_power_end_mode_tag": "macos-high-power-pmset-2",
            "host_power_mode_changed_during_task": False,
            "host_power_mode_tag": "macos-high-power-pmset-2",
            "host_energy_mode": "high_power",
            "host_energy_mode_observation_status": "observed",
        }
        records = [
            {
                **common,
                "task_status": STATUS_MEASURED,
                "success": True,
                "duration_sec": 1.0,
                "duration_mean_sec": 1.0,
                "duration_std_sec": 0.0,
                "memory_phase_observation_status": "complete",
                "feature_count": 2,
            },
            {
                **common,
                "task_status": STATUS_SKIPPED_TIMEOUT,
                "success": False,
                "timeout_cutoff_complexity": 1000,
            },
            {
                **common,
                "task_status": STATUS_INTERRUPTED,
                "success": False,
            },
        ]

        qc = run.run_qc_checks("test-run", records)

        self.assertEqual(qc["summary"]["issue_count_total"], 2)
        self.assertEqual(qc["summary"]["issue_counts_by_workload"], {"texture": 2})
        self.assertEqual(qc["summary"]["issue_counts_by_adapter"], {"pictologics": 2})
        self.assertEqual(
            {issue["issue_type"] for issue in qc["issues"]},
            {"timeout_cutoff_skip", "calculation_interrupted"},
        )

    def test_fresh_process_repeat_with_changed_value_is_rejected(self) -> None:
        self.make_dataset([1000])
        calls = 0

        def fake(adapter, **kwargs):
            nonlocal calls
            calls += 1
            payload = adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            )
            payload["values"] = {
                "all": {
                    name: (1.01 if calls == 2 and index == 0 else 1.0)
                    for index, name in enumerate(payload["features"]["all"])
                }
            }
            return payload, metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            result = run.main(
                self.args(
                    "--adapters",
                    "pictologics",
                    "--workloads",
                    "texture",
                    "--repeats",
                    "2",
                    "--extend-repeats",
                    "--keep-going",
                )
            )

        self.assertEqual(result, 1)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            counts = ledger.status_counts()
            records = ledger.records()
        self.assertEqual(counts[STATUS_MEASURED], 1)
        self.assertEqual(counts[STATUS_FAILED], 1)
        self.assertIn(
            "fresh-process repeat changed numerical feature values",
            next(
                record["error"]
                for record in records
                if record["task_status"] == STATUS_FAILED
            ),
        )

    def test_hash_verification_can_only_be_disabled_for_dry_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "restricted to dry-run"):
            run.main(self.args("--no-verify-dataset-hashes"))

    def test_case_selection_can_filter_modalities(self) -> None:
        self.make_dataset([1000, 1000, 1000], modalities=["ct", "mri", "pet"])
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        selected = run._validate_and_select_cases(
            self.dataset,
            manifest,
            sizes=None,
            variants=None,
            masks=None,
            modalities="ct,pet",
            verify_hashes=False,
        )
        self.assertEqual(
            {case["modality"] for case in selected},
            {"ct", "pet"},
        )

    def test_payload_contract_validates_aggregation_timing_and_values(self) -> None:
        task = run.build_task_plan(
            cases=[
                harmonized_case(
                    {
                        "case_id": "case",
                        "modality": "ct",
                        "size": 10,
                        "variant": 1,
                        "mask_id": "M1",
                        "mask_label": "roi",
                        "image_abs": "/tmp/image.nii.gz",
                        "mask_abs": "/tmp/mask.nii.gz",
                        "image_sha256": "a" * 64,
                        "mask_sha256": "b" * 64,
                        "shape": [10, 10, 10],
                        "spacing": [1.0, 1.0, 1.0],
                        "image_voxels": 1000,
                        "mask_voxels": 500,
                        "mask_fraction": 0.5,
                        "complexity": 1000,
                    }
                )
            ],
            dataset="unit",
            adapters=["pictologics"],
            workloads=[WORKLOAD_BY_NAME["texture"]],
            repeats=1,
            timing_observations=2,
        )[0]
        self.assertEqual(task.shape, (10, 10, 10))
        self.assertEqual(task.spacing, (1.0, 1.0, 1.0))
        self.assertEqual(task.to_dict()["shape"], [10, 10, 10])
        self.assertEqual(task.to_dict()["spacing"], [1.0, 1.0, 1.0])
        record = run._record_template(task, "unit-run")
        self.assertEqual(record["shape"], [10, 10, 10])
        self.assertEqual(record["spacing"], [1.0, 1.0, 1.0])
        self.assertEqual(record["subject_id"], task.subject_id)
        payload = adapter_payload(
            "pictologics",
            list(task.scheduled_families),
            benchmark_workload=task.workload,
        )
        payload["timing"] = contractual_timing(1.0, 2)
        payload["metadata"] = {
            "input": {
                "image_sha256": task.image_sha256,
                "source_image_sha256": task.source_image_sha256,
                "mask_sha256": task.mask_sha256,
                "modality": task.modality,
                "input_contract": task.input_contract,
                "representation_id": task.representation_id,
                "representation_derivation_sha256": (
                    task.representation_derivation_sha256
                ),
                "configured_levels": task.configured_levels,
                "occupied_levels": task.occupied_levels,
            },
            "preprocessing": {
                "discretization": task.discretization,
                "bins": task.bins,
                "bin_width": task.bin_width,
                "intensity_range": None,
            },
            "aggregation": {
                "requested": "3d_average",
                "effective_directional": "3d_average",
            },
            "timing_contract": timing_contract_metadata(),
        }
        observed_metrics = metrics_with_contract(1.0, 2)
        with self.assertRaisesRegex(
            run.AdapterProcessError,
            "3D-merged aggregation",
        ):
            run._validate_measured_result(
                payload,
                observed_metrics,
                expected_task=task,
            )

        payload["metadata"]["aggregation"] = {
            "requested": "3d_merge",
            "effective_directional": "3d_merge",
        }
        payload["metadata"]["timing_contract"]["version"] = 999
        with self.assertRaisesRegex(
            run.AdapterProcessError,
            "timing contract",
        ):
            run._validate_measured_result(
                payload,
                observed_metrics,
                expected_task=task,
            )

        payload["metadata"]["timing_contract"]["version"] = run.TIMING_CONTRACT_VERSION
        payload["metadata"]["package_initialization"] = {
            "jit_warmup_performed": True,
            "outside_measured_region": True,
        }
        payload["values"] = {"all": {name: 1.0 for name in payload["features"]["all"]}}
        payload["timing"] = single_window_contractual_timing(0.2, 2)
        observed_metrics = metrics_with_contract(0.2, 2)
        for key in CONTRACTUAL_METRIC_KEYS:
            observed_metrics[key] = payload["timing"][key]
        self.assertEqual(
            run._validate_measured_result(
                payload,
                observed_metrics,
                expected_task=task,
            ),
            len(payload["features"]["all"]),
        )

        payload["timing"] = single_window_contractual_timing(0.09, 2)
        observed_metrics = metrics_with_contract(0.09, 2)
        for key in CONTRACTUAL_METRIC_KEYS:
            observed_metrics[key] = payload["timing"][key]
        with self.assertRaisesRegex(
            run.AdapterProcessError,
            "adaptive batching counts",
        ):
            run._validate_measured_result(
                payload,
                observed_metrics,
                expected_task=task,
            )

        payload["timing"] = single_window_contractual_timing(0.2, 2)
        observed_metrics = metrics_with_contract(0.2, 2)
        for key in CONTRACTUAL_METRIC_KEYS:
            observed_metrics[key] = payload["timing"][key]
        for invalid in ("1.0", True):
            payload["values"] = {
                "all": {name: invalid for name in payload["features"]["all"]}
            }
            with (
                self.subTest(value=invalid),
                self.assertRaisesRegex(
                    run.AdapterProcessError,
                    "non-finite/non-scalar",
                ),
            ):
                run._validate_measured_result(
                    payload,
                    observed_metrics,
                    expected_task=task,
                )

    def stable_patchers(self, fake):
        def contractual_fake(adapter, **kwargs):
            payload, observed_metrics = fake(adapter, **kwargs)
            iterations = int(kwargs["iterations"])
            measured = iterations
            duration = float(observed_metrics["duration_sec"])
            payload["timing"] = contractual_timing(duration, measured)
            payload["metadata"] = {
                "input": {
                    "image_sha256": kwargs["image_sha256"],
                    "source_image_sha256": kwargs["source_image_sha256"],
                    "mask_sha256": kwargs["mask_sha256"],
                    "modality": kwargs.get("modality"),
                    "input_contract": kwargs["input_contract"],
                    "representation_id": kwargs["input_representation_id"],
                    "representation_derivation_sha256": kwargs.get(
                        "representation_derivation_sha256"
                    ),
                    "configured_levels": kwargs.get("configured_levels"),
                    "occupied_levels": kwargs.get("occupied_levels"),
                },
                "preprocessing": {
                    "discretization": kwargs["discretization"],
                    "bins": kwargs["bins"],
                    "bin_width": kwargs["bin_width"],
                    "intensity_range": (
                        [kwargs["intensity_min"], kwargs["intensity_max"]]
                        if kwargs.get("intensity_min") is not None
                        else None
                    ),
                },
                "aggregation": {
                    "requested": kwargs["aggregation"],
                    "effective_directional": kwargs["aggregation"],
                },
                "timing_contract": timing_contract_metadata(),
            }
            if adapter == "pictologics":
                payload["metadata"]["package_initialization"] = {
                    "jit_warmup_performed": True,
                    "outside_measured_region": True,
                }
            if adapter == "medimage":
                payload["metadata"]["preprocessing"]["intensity_type"] = (
                    run.MEDIMAGE_BENCHMARK_INTENSITY_TYPE
                )
            if "values" not in payload:
                payload["values"] = {
                    "all": {
                        feature_name: 1.0 for feature_name in payload["features"]["all"]
                    }
                }
            contractual_metrics = metrics_with_contract(duration, measured)
            for key in CONTRACTUAL_METRIC_KEYS:
                observed_metrics[key] = contractual_metrics[key]
            observed_metrics["adapter_event_count"] = 3 + 2 * measured
            return payload, observed_metrics

        @contextmanager
        def process_with_stable_bindings():
            with (
                mock.patch.object(
                    run,
                    "run_adapter_process",
                    side_effect=contractual_fake,
                ) as process,
                mock.patch.object(run, "_verify_execution_bindings"),
                mock.patch.object(
                    run, "_benchmark_sources_sha256", return_value="test-sources"
                ),
                mock.patch.object(
                    run.psutil,
                    "virtual_memory",
                    return_value=mock.Mock(
                        available=16 * 1024**3,
                        total=16 * 1024**3,
                    ),
                ),
                mock.patch.object(
                    run,
                    "observe_task_power_state",
                    return_value={
                        "observed_at_utc": "2026-09-02T12:00:00Z",
                        "platform": "Darwin",
                        "power_mode_tag": "macos-high-power-pmset-2",
                        "energy_mode": "high_power",
                        "energy_mode_observation_status": "observed",
                        "power_source": "AC Power",
                        "pmset_lowpowermode": 2,
                        "probe_errors": [],
                    },
                ),
            ):
                yield process

        process = process_with_stable_bindings()
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            return_value={"pictologics": {"version": "1"}, "slow": {"version": "1"}},
        )
        machine = mock.patch.object(
            run,
            "_machine_info",
            return_value={
                "cpu_model": "fake",
                "cpu_count_physical": 1,
                "memory_total_bytes": 1024**3,
            },
        )
        commit = mock.patch.object(run, "_git_commit", return_value="test")
        return process, environments, machine, commit

    def test_guardrail_skips_only_strictly_larger_matching_workload(
        self,
    ) -> None:
        self.make_dataset([1000, 1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            first_family = kwargs["families"][0]
            case_id = Path(kwargs["image"]).name.split("-fbn32")[0].split("-image")[0]
            calls.append((adapter, first_family, case_id))
            duration = (
                1.0
                if adapter == "pictologics"
                else (20.0 if first_family == "morphology" else 2.0)
            )
            return (
                adapter_payload(
                    adapter,
                    kwargs["families"],
                    benchmark_workload=kwargs.get("benchmark_workload"),
                ),
                metrics(duration),
            )

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            result = run.main(
                self.args(
                    "--adapters",
                    "pictologics,slow",
                    "--workloads",
                    "morphology,texture",
                    "--guardrail-skip-ratio",
                    "10",
                    "--enable-speed-truncation",
                    "--keep-going",
                )
            )
        self.assertEqual(result, 0)
        # Texture's first comparison block deliberately runs the candidate before
        # its baseline. Deferred reconciliation must still establish the cutoff.
        self.assertNotIn(("slow", "morphology", "case-3"), calls)
        # case-1 and case-2 intentionally contain identical image bytes, so the
        # content-addressed input stage reuses case-1's snapshot filename.
        self.assertEqual(
            sum(
                adapter == "slow" and family == "morphology"
                for adapter, family, _ in calls
            ),
            2,
        )
        self.assertIn(("slow", "histogram", "case-3"), calls)
        self.assertEqual(process.call_count, 11)

        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            counts = ledger.status_counts()
            self.assertEqual(counts[STATUS_MEASURED], 11)
            self.assertEqual(counts[STATUS_SKIPPED], 1)
            decision = ledger.guardrail_decision(
                "slow\x1fmorphology\x1fsubject:shared-series|mask:M1|"
                "representation:original_continuous_image"
            )
            self.assertEqual(int(decision["cutoff_complexity"]), 1000)

        with (self.report / "summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        skipped = [row for row in rows if row["task_status"] == STATUS_SKIPPED]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["case_id"], "case-3")
        self.assertEqual(skipped[0]["success"], "False")

    def test_guardrail_does_not_cross_real_world_cases(self) -> None:
        self.make_dataset([1000, 8000], modalities=["ct", "ct"])
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-fbn32")[0].split("-image")[0]
            calls.append((adapter, case_id))
            duration = 1.0 if adapter == "pictologics" else 20.0
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(duration)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics,slow",
                        "--workloads",
                        "texture",
                        "--guardrail-skip-ratio",
                        "10",
                        "--enable-speed-truncation",
                        "--keep-going",
                    )
                ),
                0,
            )

        self.assertIn(("slow", "case-2"), calls)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertIsNotNone(
                ledger.guardrail_decision("slow\x1ftexture\x1fcase:case-1|modality:ct")
            )
            self.assertIsNotNone(
                ledger.guardrail_decision("slow\x1ftexture\x1fcase:case-2|modality:ct")
            )

    def test_resume_does_not_rerun_measured_tasks_and_rejects_parameter_drift(
        self,
    ) -> None:
        self.make_dataset([1000])
        modalities = []

        def fake(adapter, **kwargs):
            modalities.append(kwargs["modality"])
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        common = self.args("--adapters", "pictologics", "--workloads", "texture")
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)
        self.assertEqual(modalities, ["ct"])

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            process.assert_not_called()

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunSpecMismatch):
                run.main([*common, "--resume", "--timeout", "64"])

    def test_resume_rejects_machine_drift(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--workloads", "texture")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)

        process = mock.patch.object(run, "run_adapter_process", side_effect=fake)
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            return_value={"pictologics": {"version": "1"}},
        )
        machine = mock.patch.object(
            run,
            "_machine_info",
            return_value={
                "machine_id": "different-machine",
                "cpu_model": "fake",
                "cpu_count_physical": 1,
                "memory_total_bytes": 1024**3,
            },
        )
        commit = mock.patch.object(run, "_git_commit", return_value="test")
        with process as process_mock, environments, machine, commit:
            with self.assertRaises(RunSpecMismatch):
                run.main([*common, "--resume"])
            process_mock.assert_not_called()

    def test_resume_rejects_corrupted_committed_payload(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--workloads", "texture")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)
        payload = next((self.report / "records").rglob("*.json"))
        payload.write_text("{}\n", encoding="utf-8")

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunIntegrityError):
                run.main([*common, "--resume"])
            process.assert_not_called()

    def test_non_resume_run_rejects_orphaned_benchmark_artifacts(self) -> None:
        self.make_dataset([1000])
        orphaned = self.report / "records" / "case-1" / "stale.json"
        orphaned.parent.mkdir(parents=True)
        orphaned.write_text("{}\n", encoding="utf-8")

        def fake(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            with self.assertRaises(RunAlreadyExists):
                run.main(
                    self.args("--adapters", "pictologics", "--workloads", "texture")
                )
            process.assert_not_called()

    def test_adapter_command_receives_intensity_range(self) -> None:
        environment_dir = self.root / "adapter-env"
        python = run._env_python(environment_dir)
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        payload = json.dumps(
            {
                "features": {"all": ["feature"]},
                "timing": {"duration_sec": 1.0},
            }
        )
        host = {
            "host_wall_time_sec": 1.0,
            "host_peak_rss_bytes": 1024,
            "host_cpu_time_sec": 1.0,
        }
        with (
            mock.patch.object(run, "_adapter_env_dir", return_value=environment_dir),
            mock.patch.object(
                run,
                "_run_process_command",
                return_value=(payload, "", host),
            ) as process,
        ):
            run.run_adapter_process(
                "external",
                image="image.nii.gz",
                mask="mask.nii.gz",
                discretization="fbs",
                intensity_min=-1024.0,
                intensity_max=3096.0,
                image_sha256="a" * 64,
                mask_sha256="b" * 64,
            )
        command = process.call_args.args[0]
        self.assertEqual(command[command.index("--intensity-min") + 1], "-1024.0")
        self.assertEqual(command[command.index("--intensity-max") + 1], "3096.0")
        self.assertEqual(command[command.index("--image-sha256") + 1], "a" * 64)
        self.assertEqual(command[command.index("--mask-sha256") + 1], "b" * 64)
        environment = process.call_args.kwargs["environment"]
        expected_thread_environment = run._benchmark_thread_environment()
        self.assertEqual(
            {key: environment[key] for key in run.BENCHMARK_THREAD_VARIABLES},
            expected_thread_environment,
        )
        self.assertEqual(
            {key: environment[key] for key in run.BENCHMARK_INITIALIZATION_ENV},
            run.BENCHMARK_INITIALIZATION_ENV,
        )

    def test_thread_policy_uses_all_physical_cores_for_one_process(self) -> None:
        policy = run._benchmark_thread_policy(14)
        self.assertEqual(policy["mode"], "all_physical_cores_per_isolated_task")
        self.assertEqual(policy["requested_threads"], 14)
        self.assertEqual(policy["concurrent_adapter_processes"], 1)
        self.assertEqual(
            policy["environment"],
            {key: "14" for key in run.BENCHMARK_THREAD_VARIABLES},
        )

    def test_medimage_command_receives_explicit_intensity_type(self) -> None:
        environment_dir = self.root / "adapter-env"
        python = run._env_python(environment_dir)
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        payload = json.dumps(
            {
                "features": {"all": ["feature"]},
                "timing": {"duration_sec": 1.0},
            }
        )
        host = {
            "host_wall_time_sec": 1.0,
            "host_peak_rss_bytes": 1024,
            "host_cpu_time_sec": 1.0,
        }
        with (
            mock.patch.object(run, "_adapter_env_dir", return_value=environment_dir),
            mock.patch.object(
                run,
                "_run_process_command",
                return_value=(payload, "", host),
            ) as process,
        ):
            run.run_adapter_process(
                "medimage",
                image="image.nii.gz",
                mask="mask.nii.gz",
            )
        command = process.call_args.args[0]
        self.assertEqual(
            command[command.index("--intensity-type") + 1],
            run.MEDIMAGE_BENCHMARK_INTENSITY_TYPE,
        )

    def test_enabled_guardrail_requires_selected_baseline(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(ValueError, "baseline adapter"):
            run.main(
                self.args(
                    "--adapters",
                    "slow",
                    "--workloads",
                    "texture",
                    "--enable-speed-truncation",
                )
            )

    def test_timeout_policy_allows_a_single_nonbaseline_adapter(self) -> None:
        self.make_dataset([1000])
        with (
            mock.patch.object(
                run,
                "_adapter_environment_snapshots",
                return_value={"pyradiomics": {"version": "3.1.0"}},
            ),
            mock.patch.object(
                run,
                "_machine_info",
                return_value={
                    "cpu_model": "fake",
                    "cpu_count_physical": 1,
                    "memory_total_bytes": 1024**3,
                },
            ),
            mock.patch.object(run, "_git_commit", return_value="test"),
        ):
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pyradiomics",
                        "--workloads",
                        "texture",
                        "--dry-run",
                    )
                ),
                0,
            )

    def test_sampled_cpu_time_includes_process_descendants(self) -> None:
        parent = mock.Mock(pid=100)
        child = mock.Mock(pid=101)
        parent.children.return_value = [child]
        parent.cpu_times.return_value = mock.Mock(user=1.0, system=0.5)
        child.cpu_times.return_value = mock.Mock(user=2.0, system=0.25)

        self.assertEqual(run._safe_cpu_time(parent), 3.75)

    def test_selected_baseline_must_support_every_scheduled_workload(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(
            ValueError,
            "baseline adapter.*unsupported families",
        ):
            run.main(
                self.args(
                    "--adapters",
                    "pyradiomics",
                    "--guardrail-baseline",
                    "pyradiomics",
                    "--workloads",
                    "all",
                    "--dry-run",
                    "--enable-speed-truncation",
                )
            )

    def test_unknown_workload_is_rejected(self) -> None:
        self.make_dataset([1000])
        with self.assertRaisesRegex(ValueError, "unknown benchmark workloads"):
            run.main(
                self.args(
                    "--adapters",
                    "pictologics",
                    "--workloads",
                    "not-a-workload",
                )
            )

    def test_timeout_skips_strictly_larger_tasks_in_the_same_series(self) -> None:
        self.make_dataset([1000, 1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-fbn32")[0].split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "slow":
                raise run.AdapterTimeout(
                    "slow",
                    5.0,
                    5.1,
                    999999,
                    partial_duration_samples_sec=(1.25,),
                    partial_cpu_time_samples_sec=(1.0,),
                )
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            result = run.main(
                self.args(
                    "--adapters",
                    "pictologics,slow",
                    "--workloads",
                    "texture",
                    "--timeout",
                    "5",
                    "--keep-going",
                )
            )
        self.assertEqual(result, 0)
        self.assertEqual(sum(adapter == "slow" for adapter, _ in calls), 2)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            records = ledger.records()
        timed = [
            record for record in records if record["task_status"] == STATUS_TIMED_OUT
        ]
        skipped = [
            record
            for record in records
            if record["task_status"] == STATUS_SKIPPED_TIMEOUT
        ]
        self.assertEqual(len(timed), 2)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["case_id"], "case-3")
        self.assertIsNone(timed[0]["duration_sec"])
        self.assertEqual(timed[0]["censor_lower_bound_sec"], 5.1)
        self.assertEqual(timed[0]["partial_duration_samples_sec"], [1.25])
        self.assertEqual(timed[0]["partial_cpu_time_samples_sec"], [1.0])
        self.assertEqual(timed[0]["partial_completed_iterations"], 1)
        self.assertFalse(timed[0]["success"])

    def test_timeout_cutoff_applies_to_the_baseline_adapter(self) -> None:
        self.make_dataset([1000, 8000], synthetic_series=True)
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-fbn32")[0].split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "pictologics":
                raise run.AdapterTimeout(adapter, 5.0, 5.1, 999999)
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics,slow",
                        "--workloads",
                        "texture",
                        "--timeout",
                        "5",
                        "--keep-going",
                    )
                ),
                0,
            )
        self.assertEqual(
            calls,
            [
                ("pictologics", "case-1"),
                ("slow", "case-1"),
                ("slow", "case-2"),
            ],
        )
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_TIMED_OUT], 1)
            self.assertEqual(ledger.status_counts()[STATUS_MEASURED], 2)
            self.assertEqual(ledger.status_counts()[STATUS_SKIPPED_TIMEOUT], 1)

    def test_timeout_cutoff_does_not_cross_masks_or_real_world_cases(self) -> None:
        self.make_dataset([1000, 8000], modalities=["ct", "ct"])
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-fbn32")[0].split("-image")[0]
            calls.append((adapter, case_id))
            if adapter == "slow":
                raise run.AdapterTimeout(adapter, 5.0, 5.1, 999999)
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics,slow",
                        "--workloads",
                        "texture",
                        "--timeout",
                        "5",
                        "--keep-going",
                    )
                ),
                0,
            )

        self.assertIn(("slow", "case-1"), calls)
        self.assertIn(("slow", "case-2"), calls)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_TIMED_OUT], 2)
            self.assertNotIn(STATUS_SKIPPED_TIMEOUT, ledger.status_counts())

    def test_zero_process_tree_rss_is_rejected(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            observed = metrics(1.0)
            observed["peak_rss_bytes"] = 0
            observed["host_peak_rss_bytes"] = 0
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), observed

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--workloads",
                        "texture",
                    )
                ),
                1,
            )
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["task_status"], STATUS_FAILED)
        self.assertIn("peak RSS", record["error"])

    def test_memory_estimate_never_prevents_a_task_launch(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--workloads",
                        "texture",
                        "--memory-cap-gib",
                        "0.01",
                    )
                ),
                0,
            )
            process.assert_called_once()
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["task_status"], STATUS_MEASURED)
        self.assertEqual(record["memory_preflight_decision"], "launch")
        self.assertTrue(record["memory_estimate_exceeds_budget"])
        self.assertTrue(record["success"])

    def test_generic_failure_never_activates_speed_truncation(self) -> None:
        self.make_dataset([1000, 8000])
        calls = []

        def fake(adapter, **kwargs):
            case_id = Path(kwargs["image"]).name.split("-fbn32")[0].split("-image")[0]
            calls.append(case_id)
            if case_id == "case-1":
                raise RuntimeError("ordinary adapter failure")
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--workloads",
                        "texture",
                        "--keep-going",
                    )
                ),
                1,
            )
        self.assertEqual(calls, ["case-1", "case-2"])
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            counts = ledger.status_counts()
            self.assertEqual(counts[STATUS_FAILED], 1)
            self.assertEqual(counts[STATUS_MEASURED], 1)
            self.assertIsNone(
                ledger.guardrail_decision(
                    "pictologics\x1ftexture\x1fcase:case-1|modality:ct"
                )
            )

    def test_interrupted_task_is_retried_on_resume(self) -> None:
        self.make_dataset([1000])

        def interrupted(adapter, **kwargs):
            raise run.AdapterInterrupted(adapter, 0.2, 999999)

        patchers = self.stable_patchers(interrupted)
        common = self.args(
            "--adapters",
            "pictologics",
            "--workloads",
            "texture",
        )
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 130)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(next(iter(ledger.status_counts())), STATUS_INTERRUPTED)
        run_meta = json.loads(
            (self.report / "run_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_meta["run_status"], "interrupted")
        self.assertEqual(run_meta["status_counts"][STATUS_INTERRUPTED], 1)

        def measured(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(measured)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            self.assertEqual(process.call_count, 1)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_MEASURED], 1)

    def test_atomic_payload_is_adopted_after_ledger_interruption(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--workloads", "texture")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)

        connection = sqlite3.connect(self.report / "benchmark.sqlite3")
        try:
            connection.execute(
                "UPDATE tasks SET status = ?, record_json = NULL WHERE status = ?",
                (STATUS_INTERRUPTED, STATUS_MEASURED),
            )
            connection.execute(
                "UPDATE task_attempts SET status = ?", (STATUS_INTERRUPTED,)
            )
            connection.commit()
        finally:
            connection.close()

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            process.assert_not_called()
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            self.assertEqual(ledger.status_counts()[STATUS_MEASURED], 1)

    def test_atomic_payload_with_conflicting_embedded_record_is_not_adopted(
        self,
    ) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        common = self.args("--adapters", "pictologics", "--workloads", "texture")
        patchers = self.stable_patchers(fake)
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main(common), 0)

        payload_path = next((self.report / "records").rglob("*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["benchmark"]["record"]["case_id"] = "WRONG-CASE"
        payload["benchmark"]["record"]["duration_sec"] = 999.0
        payload_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        connection = sqlite3.connect(self.report / "benchmark.sqlite3")
        try:
            connection.execute(
                "UPDATE tasks SET status = ?, record_json = NULL WHERE status = ?",
                (STATUS_INTERRUPTED, STATUS_MEASURED),
            )
            connection.execute(
                "UPDATE task_attempts SET status = ?",
                (STATUS_INTERRUPTED,),
            )
            connection.commit()
        finally:
            connection.close()

        patchers = self.stable_patchers(fake)
        with patchers[0] as process, patchers[1], patchers[2], patchers[3]:
            self.assertEqual(run.main([*common, "--resume"]), 0)
            self.assertEqual(process.call_count, 1)
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["case_id"], "case-1")
        self.assertEqual(record["duration_sec"], 1.0)
        self.assertEqual(record["attempt"], 2)

    def test_run_uses_snapshot_if_source_changes_after_validation(self) -> None:
        self.make_dataset([1000])
        source = self.dataset / "case-1-fbn32.nii.gz"
        expected_hash = digest(source)
        seen_paths = []

        def fake(adapter, **kwargs):
            staged = Path(kwargs["image"])
            seen_paths.append(staged)
            self.assertEqual(digest(staged), expected_hash)
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        def mutate_source(_adapters):
            nib.save(
                nib.Nifti1Image(
                    np.full((10, 10, 10), 7.0, dtype=np.float32),
                    np.eye(4),
                ),
                source,
            )
            return {"pictologics": {"version": "1"}}

        patchers = self.stable_patchers(fake)
        environments = mock.patch.object(
            run,
            "_adapter_environment_snapshots",
            side_effect=mutate_source,
        )
        with patchers[0], environments, patchers[2], patchers[3]:
            self.assertEqual(
                run.main(
                    self.args(
                        "--adapters",
                        "pictologics",
                        "--workloads",
                        "texture",
                    )
                ),
                0,
            )
        self.assertNotEqual(digest(source), expected_hash)
        self.assertEqual(len(seen_paths), 1)
        self.assertTrue(
            seen_paths[0].resolve().is_relative_to((self.report / "inputs").resolve())
        )

    def test_staged_input_mutation_during_adapter_execution_is_rejected(self) -> None:
        self.make_dataset([1000])

        def fake(adapter, **kwargs):
            staged = Path(kwargs["image"])
            staged.chmod(0o644)
            staged.write_bytes(staged.read_bytes() + b"tampered")
            return adapter_payload(
                adapter,
                kwargs["families"],
                benchmark_workload=kwargs.get("benchmark_workload"),
            ), metrics(1.0)

        patchers = self.stable_patchers(fake)
        with (
            patchers[0],
            patchers[1],
            patchers[2],
            patchers[3],
            self.assertRaisesRegex(RunIntegrityError, "staged task image changed"),
        ):
            run.main(
                self.args(
                    "--adapters",
                    "pictologics",
                    "--workloads",
                    "texture",
                )
            )
        with BenchmarkLedger(self.report / "benchmark.sqlite3") as ledger:
            record = ledger.records()[0]
        self.assertEqual(record["task_status"], STATUS_INTERRUPTED)
        self.assertIn("changed during the run", record["error"])

    def test_process_timeout_reaps_fake_process(self) -> None:
        with self.assertRaises(run.AdapterTimeout) as caught:
            run._run_process_command(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                adapter_name="fake",
                timeout=0.05,
                sample_interval=0.01,
                termination_grace=0.1,
            )
        self.assertFalse(psutil.pid_exists(caught.exception.pid))
        self.assertEqual(caught.exception.phase, "startup_or_warmup")

    def test_process_timeout_reaps_descendant_process(self) -> None:
        child_pid_path = self.root / "child.pid"
        child_code = "import time; time.sleep(30)"
        parent_code = "\n".join(
            [
                "from pathlib import Path",
                "import subprocess, sys, time",
                f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])",
                f"Path({str(child_pid_path)!r}).write_text(str(child.pid))",
                "time.sleep(30)",
            ]
        )
        with self.assertRaises(run.AdapterTimeout):
            run._run_process_command(
                [sys.executable, "-c", parent_code],
                adapter_name="process-tree-fixture",
                timeout=0.5,
                sample_interval=0.01,
                termination_grace=0.2,
            )

        self.assertTrue(child_pid_path.is_file())
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2.0
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(psutil.pid_exists(child_pid))

    def test_process_timeout_retains_completed_call_samples(self) -> None:
        code = (
            "import json,time; p='BENCH_EVENT '; "
            "print(p+json.dumps({'event':'worker_ready'}),flush=True); "
            "print(p+json.dumps({'event':'calculation_start','iteration':1}),flush=True); "
            "print(p+json.dumps({'event':'calculation_complete','iteration':1,"
            "'calculation_sec':0.0125,'cpu_time_sec':0.01}),flush=True); "
            "print(p+json.dumps({'event':'calculation_start','iteration':2}),flush=True); "
            "time.sleep(30)"
        )
        with self.assertRaises(run.AdapterTimeout) as caught:
            run._run_process_command(
                [sys.executable, "-c", code],
                adapter_name="fake",
                timeout=0.1,
                sample_interval=0.01,
                termination_grace=0.1,
            )
        self.assertEqual(caught.exception.phase, "prepared_calculation_region")
        self.assertEqual(caught.exception.partial_duration_samples_sec, (0.0125,))
        self.assertEqual(caught.exception.partial_cpu_time_samples_sec, (0.01,))


if __name__ == "__main__":
    unittest.main()

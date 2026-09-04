from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import patch

from bench.benchmark_contract import load_benchmark_contract
from bench.benchmark_ledger import sha256_file
from bench.benchmark_models import RUN_SPEC_SCHEMA_VERSION
from bench.benchmark_workspace import _task_inventory
from scripts.launch_benchmark import (
    PMSET_MODE_ATTEMPTS,
    _darwin_power_state,
    _effective_host_settings,
    _existing_result_resume_errors,
    _git_source_state,
    _host_profile_preflight,
    _known_sync_root,
    _load_host_profile,
    _portable_command,
    _profile_value,
    _run_controller_command,
    _validate_machine_id,
    build_commands,
)


class BenchmarkWorkspaceTests(unittest.TestCase):
    def test_git_source_state_distinguishes_clean_and_dirty_commits(self) -> None:
        clean_results = [
            SimpleNamespace(returncode=0, stdout=""),
            SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
        ]
        with patch(
            "scripts.launch_benchmark.subprocess.run", side_effect=clean_results
        ):
            clean = _git_source_state(Path("/repository"))

        self.assertEqual(clean["status"], "clean")
        self.assertEqual(clean["commit"], "a" * 40)
        self.assertEqual(clean["dirty_entries"], [])

        dirty_results = [
            SimpleNamespace(returncode=0, stdout=" M bench/run.py\n"),
            SimpleNamespace(returncode=0, stdout="b" * 40 + "\n"),
        ]
        with patch(
            "scripts.launch_benchmark.subprocess.run", side_effect=dirty_results
        ):
            dirty = _git_source_state(Path("/repository"))

        self.assertEqual(dirty["status"], "dirty")
        self.assertEqual(dirty["commit"], "b" * 40)
        self.assertEqual(dirty["dirty_entries"], [" M bench/run.py"])

    def test_publication_mac_profile_records_observed_energy_mode(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        profile = _load_host_profile(
            repository / "configs/benchmark/hosts/mac-m4pro-01.json"
        )["payload"]

        self.assertNotIn("low_power_mode", profile["required_runtime_state"])
        self.assertNotIn(
            "energy_mode_observation_required",
            profile["required_runtime_state"],
        )
        self.assertEqual(
            profile["benchmark_settings"]["energy_mode_policy"],
            "observed_at_task_boundaries_never_gated",
        )
        self.assertNotIn("energy_mode", profile["benchmark_settings"])
        self.assertNotIn("pmset_lowpowermode", profile["benchmark_settings"])

    def test_darwin_power_probe_records_high_power(self) -> None:
        completed = [
            SimpleNamespace(
                returncode=0,
                stdout="Now drawing from 'AC Power'\n",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "Battery Power:\n lowpowermode 1\n"
                    "AC Power:\n lowpowermode 2\n sleep 1\n"
                ),
            ),
            SimpleNamespace(
                returncode=0,
                stdout="PreventSystemSleep 1\nPreventUserIdleSystemSleep 0\n",
            ),
        ]
        with (
            patch("scripts.launch_benchmark.subprocess.run", side_effect=completed),
            patch("scripts.launch_benchmark.time.sleep"),
        ):
            state = _darwin_power_state()

        self.assertEqual(state["power_source"], "AC Power")
        self.assertEqual(state["low_power_mode"], 2)
        self.assertEqual(state["energy_mode"], "high_power")
        self.assertTrue(state["sleep_assertion"])
        self.assertEqual(state["probe_errors"], [])

    def test_observed_energy_mode_is_bound_into_run_settings(self) -> None:
        profile = {
            "payload": {
                "benchmark_settings": {
                    "energy_mode_policy": (
                        "observed_tagged_per_session_never_gated"
                    )
                }
            }
        }
        preflight = {
            "host_profile": {
                "observed_runtime_state": {
                    "power_source": "AC Power",
                    "low_power_mode": 2,
                    "energy_mode": "high_power",
                    "probe_errors": [],
                }
            }
        }

        settings = _effective_host_settings(profile, preflight)

        self.assertEqual(settings["power_source"], "AC Power")
        self.assertEqual(settings["energy_mode"], "high_power")
        self.assertEqual(settings["pmset_lowpowermode"], 2)
        self.assertEqual(settings["energy_mode_observation_status"], "observed")
        self.assertEqual(settings["power_mode_tag"], "macos-high-power-pmset-2")
        self.assertEqual(settings["power_state_probe_errors"], [])

    def test_unavailable_energy_mode_is_explicitly_tagged(self) -> None:
        settings = _effective_host_settings(
            {"payload": {"benchmark_settings": {}}},
            {
                "host_profile": {
                    "observed_runtime_state": {
                        "power_source": "AC Power",
                        "low_power_mode": None,
                        "energy_mode": None,
                        "probe_errors": ["low_power_mode_unavailable"],
                    }
                }
            },
        )

        self.assertEqual(settings["energy_mode"], "unavailable")
        self.assertIsNone(settings["pmset_lowpowermode"])
        self.assertEqual(settings["energy_mode_observation_status"], "unavailable")
        self.assertEqual(settings["power_mode_tag"], "macos-energy-mode-unavailable")
        self.assertEqual(
            settings["power_state_probe_errors"],
            ["low_power_mode_unavailable"],
        )

    def test_darwin_power_probe_retries_transient_missing_mode(self) -> None:
        completed = [
            SimpleNamespace(
                returncode=0,
                stdout="Now drawing from 'AC Power'\n",
            ),
            SimpleNamespace(returncode=0, stdout="AC Power:\n sleep 1\n"),
            SimpleNamespace(returncode=0, stdout="AC Power:\n sleep 1\n"),
            SimpleNamespace(
                returncode=0,
                stdout="AC Power:\n lowpowermode 2\n sleep 1\n",
            ),
            SimpleNamespace(
                returncode=0,
                stdout="PreventSystemSleep 1\n",
            ),
        ]
        with (
            patch("scripts.launch_benchmark.subprocess.run", side_effect=completed),
            patch("scripts.launch_benchmark.time.sleep"),
        ):
            state = _darwin_power_state()

        self.assertEqual(state["energy_mode"], "high_power")
        self.assertEqual(state["probe_errors"], [])

    def test_darwin_power_probe_outlasts_previous_retry_window(self) -> None:
        completed = [
            SimpleNamespace(returncode=0, stdout="Now drawing from 'AC Power'\n"),
            *[
                SimpleNamespace(returncode=0, stdout="AC Power:\n sleep 1\n")
                for _ in range(5)
            ],
            SimpleNamespace(
                returncode=0,
                stdout="AC Power:\n lowpowermode 2\n sleep 1\n",
            ),
            SimpleNamespace(returncode=0, stdout="PreventSystemSleep 1\n"),
        ]
        with (
            patch("scripts.launch_benchmark.subprocess.run", side_effect=completed),
            patch("scripts.launch_benchmark.time.sleep") as sleep,
        ):
            state = _darwin_power_state()

        self.assertEqual(state["energy_mode"], "high_power")
        self.assertEqual(state["probe_errors"], [])
        self.assertEqual(sleep.call_count, 5)

    def test_darwin_power_probe_falls_back_to_active_profile(self) -> None:
        completed = [
            SimpleNamespace(returncode=0, stdout="Now drawing from 'AC Power'\n"),
            *[
                SimpleNamespace(returncode=0, stdout="AC Power:\n sleep 1\n")
                for _ in range(PMSET_MODE_ATTEMPTS)
            ],
            SimpleNamespace(
                returncode=0,
                stdout="System-wide power settings:\n lowpowermode 2\n",
            ),
            SimpleNamespace(returncode=0, stdout="PreventSystemSleep 1\n"),
        ]
        with (
            patch("scripts.launch_benchmark.subprocess.run", side_effect=completed),
            patch("scripts.launch_benchmark.time.sleep"),
        ):
            state = _darwin_power_state()

        self.assertEqual(state["low_power_mode"], 2)
        self.assertEqual(state["energy_mode"], "high_power")
        self.assertEqual(state["probe_errors"], [])

    def test_energy_mode_never_gates_host_preflight(self) -> None:
        profile = {
            "path": "/profile.json",
            "sha256": "a" * 64,
            "payload": {
                "profile_id": "test-mac",
                "expected_hardware": {"platform": "Darwin"},
                "required_runtime_state": {
                    "energy_mode_observation_required": True,
                    "low_power_mode": 2,
                },
                "benchmark_settings": {},
            },
        }
        unavailable = {
            "power_source": "AC Power",
            "low_power_mode": None,
            "energy_mode": None,
            "sleep_assertion": True,
            "probe_errors": ["low_power_mode_unavailable"],
        }
        with (
            patch("scripts.launch_benchmark.platform.system", return_value="Darwin"),
            patch(
                "scripts.launch_benchmark._darwin_power_state",
                return_value=unavailable,
            ),
        ):
            preflight = _host_profile_preflight(
                profile,
                require_sleep_assertion=False,
            )

        self.assertEqual(preflight["status"], "pass")
        self.assertEqual(preflight["errors"], [])

    def test_controller_launcher_forwards_interrupt_and_waits(self) -> None:
        class FakeProcess:
            pid = 4321

            def __init__(self):
                self.forwarded_signals = []

            def poll(self):
                return None

            def send_signal(self, signum):
                self.forwarded_signals.append(signum)

            def wait(self):
                signal.raise_signal(signal.SIGINT)
                return 130

        process = FakeProcess()
        with mock.patch(
            "scripts.launch_benchmark.subprocess.Popen", return_value=process
        ) as popen:
            if os.name == "nt":
                returncode = _run_controller_command(["controller"])
            else:
                with mock.patch("scripts.launch_benchmark.os.killpg") as killpg:
                    returncode = _run_controller_command(["controller"])

        self.assertEqual(returncode, 130)
        if os.name == "nt":
            popen.assert_called_once_with(
                ["controller"],
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            self.assertEqual(process.forwarded_signals, [signal.SIGINT])
        else:
            popen.assert_called_once_with(["controller"], start_new_session=True)
            killpg.assert_called_once_with(4321, signal.SIGINT)

    def test_launcher_scales_project_progress_for_appended_repeats(self) -> None:
        workspace = {
            "launch_policy": {
                "speed_truncation_enabled": False,
                "runtime_limit_policy": (
                    "per-task timeout censoring, then skip strictly larger images "
                    "for the same adapter, workload, mask, and input configuration"
                ),
                "timeout_seconds": 1800.0,
                "checkpoint_interval_tasks": 25,
                "progress_interval_seconds": 15.0,
            },
            "endpoint_contract": {"path": "contract.json"},
            "adapter_order": ["pictologics", "pyradiomics"],
            "task_inventory": {
                "fresh_process_repeats": 3,
                "measured_observations_per_process": 3,
                "adapter_count": 2,
                "workload_count": 2,
            },
            "datasets": {
                "pillar1_morphology": {"case_count": 2},
                "pillar2_whole_anatomy": {"case_count": 3},
                "pillar3_ibsi2_phase3": {"case_count": 5},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data" / "benchmark"
            root.mkdir(parents=True)
            commands = build_commands(
                root,
                workspace,
                result_root=Path(temporary) / "results",
                dry_run=True,
                machine_id="test-host",
                host_profile_id="test-host",
                host_profile_sha256="a" * 64,
                host_settings={"power_source": "AC Power"},
                repeats=5,
            )
        self.assertEqual(len(commands), 3)
        for command in commands:
            self.assertIn("--extend-repeats", command)
            self.assertEqual(command[command.index("--repeats") + 1], "5")
            self.assertEqual(command[command.index("--project-total-tasks") + 1], "200")
            self.assertNotIn("--enable-speed-truncation", command)
            self.assertNotIn("--guardrail-skip-ratio", command)
            self.assertEqual(
                command[command.index("--machine-id") + 1], "test-host"
            )
            self.assertEqual(
                command[command.index("--host-profile-id") + 1], "test-host"
            )
            self.assertEqual(
                command[command.index("--host-profile-sha256") + 1], "a" * 64
            )
            self.assertEqual(
                json.loads(command[command.index("--host-settings-json") + 1]),
                {"power_source": "AC Power"},
            )
            report_dir = Path(command[command.index("--report-dir") + 1])
            self.assertEqual(report_dir.parent.name, "test-host")
        offsets = [
            int(command[command.index("--project-task-offset") + 1])
            for command in commands
        ]
        self.assertEqual(offsets, [0, 40, 100])

    def test_machine_id_rejects_path_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "machine ID"):
            _validate_machine_id("../windows")

    def test_host_profile_is_checksum_bound_and_rejects_identity_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "host.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile_id": "test-host",
                        "machine_id": "test-host",
                        "machine_label": "Test host",
                        "cpu_model": "Test CPU",
                        "expected_hardware": {},
                        "required_runtime_state": {},
                        "benchmark_settings": {},
                    }
                ),
                encoding="utf-8",
            )
            profile = _load_host_profile(path)
            expected_sha256 = sha256_file(path)

        self.assertEqual(profile["sha256"], expected_sha256)
        self.assertEqual(
            _profile_value(None, profile, "machine_id", flag="--machine-id"),
            "test-host",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _profile_value(
                "different-host", profile, "machine_id", flag="--machine-id"
            )

    def test_cloud_storage_path_is_detected(self) -> None:
        path = Path("/Users/test/Library/CloudStorage/OneDrive-Personal/results")
        self.assertEqual(_known_sync_root(path), "path_marker")

    def test_portable_command_keeps_repository_symlink_path_lexical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            repository = temporary_root / "workspace" / "project"
            result_root = temporary_root / "results"
            interpreter = repository / ".venv" / "bin" / "python"
            report_dir = result_root / "host" / "pillar"
            command = [str(interpreter), "--report-dir", str(report_dir)]
            self.assertEqual(
                _portable_command(command, repository, result_root),
                [
                    ".venv/bin/python",
                    "--report-dir",
                    "{RESULT_ROOT}/host/pillar",
                ],
            )

    def test_task_inventory_distinguishes_scheduled_and_eligible(self) -> None:
        contract = load_benchmark_contract()
        datasets = {
            "pillar1_morphology": {"case_count": 1},
            "pillar2_whole_anatomy": {"case_count": 2},
            "pillar3_ibsi2_phase3": {"case_count": 3},
        }
        result = _task_inventory(datasets, contract)

        self.assertEqual(result["totals"]["case_count"], 6)
        self.assertEqual(result["totals"]["scheduled_task_records"], 540)
        self.assertEqual(result["totals"]["eligible_calculation_tasks"], 486)
        self.assertEqual(result["totals"]["preempted_unsupported_tasks"], 54)
        self.assertEqual(
            result["totals"][
                "minimum_post_warmup_verification_calls_if_all_eligible_complete"
            ],
            486,
        )
        self.assertEqual(
            result["totals"]["measured_observations_if_all_eligible_complete"],
            1458,
        )
        self.assertEqual(
            len(result["supported_workloads_by_adapter"]["pyradiomics"]), 3
        )
        self.assertEqual(
            len(result["supported_workloads_by_adapter"]["pictologics"]), 6
        )

    def test_existing_result_contract_mismatch_is_rejected_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            machine_root = Path(temporary) / "test-host"
            report_dir = machine_root / "pillar1_morphology"
            report_dir.mkdir(parents=True)
            (report_dir / "benchmark.sqlite3").touch()
            (report_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "schema_version": RUN_SPEC_SCHEMA_VERSION - 1,
                        "endpoint_contract_sha256": "old",
                        "workloads": ["morphology", "intensity", "texture", "ivh"],
                        "adapters": ["pictologics"],
                        "manifest_sha256": "dataset",
                    }
                ),
                encoding="utf-8",
            )
            workspace = {
                "endpoint_contract": {"sha256": "contract"},
                "launch_policy": {
                    "reported_workloads": [
                        "morphology",
                        "spatial_autocorrelation",
                        "local_intensity",
                        "intensity",
                        "texture",
                        "ivh",
                    ]
                },
                "adapter_order": ["pictologics", "pyradiomics"],
                "datasets": {
                    "pillar1_morphology": {"manifest_sha256": "dataset"},
                    "pillar2_whole_anatomy": {"manifest_sha256": "dataset"},
                    "pillar3_ibsi2_phase3": {"manifest_sha256": "dataset"},
                },
            }

            errors = _existing_result_resume_errors(
                machine_root,
                workspace,
                source_commit=None,
            )

        self.assertEqual(len(errors), 1)
        self.assertIn("run-spec schema", errors[0])
        self.assertIn("endpoint contract differs", errors[0])
        self.assertIn("workload set differs", errors[0])
        self.assertIn("use a new empty --result-root", errors[0])

    def test_launch_loader_rejects_changed_dataset_manifest(self) -> None:
        from scripts.launch_benchmark import _load_workspace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = {}
            for pillar, directory in (
                ("pillar1_morphology", "pillar1"),
                ("pillar2_whole_anatomy", "pillar2_a1"),
                ("pillar3_ibsi2_phase3", "ibsi2_phase3"),
            ):
                path = root / directory
                path.mkdir()
                manifest = path / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                datasets[pillar] = {
                    "path": directory,
                    "manifest_sha256": sha256_file(manifest),
                }
            workspace = {
                "schema_version": 4,
                "benchmark_timing_executed": False,
                "datasets": datasets,
            }
            (root / "workspace_manifest.json").write_text(
                json.dumps(workspace), encoding="utf-8"
            )
            _load_workspace(root)
            (root / "pillar1" / "manifest.json").write_text(
                '{"changed":true}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "changed after validation"):
                _load_workspace(root)


if __name__ == "__main__":
    unittest.main()

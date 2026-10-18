"""Protocol planning tests only: no adapter or controller process is launched."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.benchmark_ledger import sha256_file
from scripts import launch_benchmark as launcher


def _arguments(**overrides):
    values = dict(
        host_profile=Path("host.json"), result_root="final-results",
        screen_only=False, repeat_timeouts=False, timeout=None,
        memory_cap_gib=None, execution_order=None, repeats=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_final_protocol_and_legacy_defaults_remain_distinct():
    preset = launcher.load_final_run_preset()
    assert preset["sha256"] == sha256_file(launcher.FINAL_RUN_PRESET_PATH)
    final = _arguments()
    launcher._resolve_launch_options(final, preset)
    assert (final.execution_order, final.repeats, final.timeout, final.memory_cap_gib) == (
        "rotated", 3, 900.0, 12.0,
    )
    legacy = _arguments(host_profile=None, result_root=None)
    launcher._resolve_launch_options(legacy, None)
    assert (legacy.execution_order, legacy.repeats, legacy.result_root) == (
        "adapter-first", None, "results/benchmark",
    )
    custom_legacy = _arguments(repeats=5, timeout=1800, execution_order="rotated")
    launcher._resolve_launch_options(custom_legacy, None)
    assert (custom_legacy.repeats, custom_legacy.timeout) == (5, 1800)


@pytest.mark.parametrize("overrides", [
    {"host_profile": None}, {"result_root": None}, {"timeout": 1800},
    {"timeout": float("nan")}, {"memory_cap_gib": 16}, {"repeats": 5},
    {"execution_order": "adapter-first"}, {"screen_only": True},
    {"repeat_timeouts": True},
])
def test_final_preset_rejects_protocol_overrides(overrides):
    with pytest.raises(ValueError, match="requires|conflict"):
        launcher._resolve_launch_options(_arguments(**overrides), launcher.load_final_run_preset())


def _workspace():
    payload = launcher.load_final_run_preset()["payload"]
    return {
        "adapter_order": payload["adapter_order"],
        "task_inventory": {
            "fresh_process_repeats": 3, "measured_observations_per_process": 3,
            "adapter_count": 5, "workload_count": 6,
        },
        "launch_policy": {"checkpoint_interval_tasks": 25},
        "endpoint_contract": {"path": "configs/benchmark/calculation_only_workload.json"},
        "datasets": {pillar: {"case_count": 1} for pillar, _ in launcher.PILLARS},
    }


def _profile():
    return {
        "sha256": "a" * 64,
        "payload": {
            "profile_id": "test-host", "machine_id": "test-host",
            "machine_label": "Test host", "cpu_model": "Test CPU",
            "benchmark_settings": {"thread_policy": "all_physical_cores_per_isolated_task"},
        },
    }


@pytest.mark.parametrize("field,value", [
    ("fresh_process_repeats", 4), ("measured_observations_per_process", 1),
    ("adapter_order", ["pictologics"]), ("thread_policy", "single_thread"),
])
def test_preset_rejects_different_workspace_or_host_protocol(field, value):
    workspace, profile = _workspace(), _profile()
    if field == "adapter_order":
        workspace[field] = value
    elif field == "thread_policy":
        profile["payload"]["benchmark_settings"][field] = value
    else:
        workspace["task_inventory"][field] = value
    with pytest.raises(ValueError, match="conflict"):
        launcher._validate_preset_workspace(launcher.load_final_run_preset(), workspace, profile)


@pytest.mark.parametrize("validate_plans", [False, True])
def test_final_plan_binds_preset_and_never_starts_calculations(
    tmp_path, monkeypatch, capsys, validate_plans,
):
    workspace_root = tmp_path / "data" / "benchmark"
    workspace_root.mkdir(parents=True)
    (workspace_root / "workspace_manifest.json").write_text("{}", encoding="utf-8")
    result_root = tmp_path / "final-results"
    command = [
        "launch_benchmark.py", "--preset", "final", "--host-profile", "host.json",
        "--workspace-root", str(workspace_root), "--result-root", str(result_root),
    ]
    if validate_plans:
        command.append("--validate-plans")
    monkeypatch.setattr("sys.argv", command)
    monkeypatch.setattr(launcher, "_load_workspace", lambda root: _workspace())
    monkeypatch.setattr(launcher, "_load_host_profile", lambda path: _profile())
    monkeypatch.setattr(launcher, "_result_preflight", lambda *args, **kwargs: {"status": "pass"})

    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not start benchmark calculations")

    monkeypatch.setattr(launcher, "_run_controller_command", forbidden)
    monkeypatch.setattr(launcher, "_enable_windows_sleep_prevention", forbidden)
    dispatched = []

    def fake_controller(command, **kwargs):
        assert "--dry-run" in command
        dispatched.append(command)
        return SimpleNamespace(stdout="Validated 90 tasks\nImmutable run fingerprint: test\n")

    monkeypatch.setattr(launcher.subprocess, "run", fake_controller if validate_plans else forbidden)
    assert launcher.main() == 0
    output = capsys.readouterr().out
    plan, _ = json.JSONDecoder().raw_decode(output)
    assert plan["radiomic_calculation_started"] is False
    assert plan["mode"] == ("controller_dry_run" if validate_plans else "print_only")
    assert plan["run_preset"]["preset_id"] == "final-calculation-only-v1"
    assert plan["run_preset"]["sha256"] == sha256_file(launcher.FINAL_RUN_PRESET_PATH)
    assert len(plan["commands"]) == 3
    for command in plan["commands"]:
        assert command[command.index("--execution-order") + 1] == "rotated"
        assert command[command.index("--repeats") + 1] == "3"
        assert command[command.index("--timeout") + 1] == "900.0"
        assert command[command.index("--memory-cap-gib") + 1] == "12.0"
        assert "--resume" in command
        assert "--execute-adapters" not in command
        assert "--extend-repeats" not in command
        settings = json.loads(command[command.index("--host-settings-json") + 1])
        assert settings["benchmark_preset"]["sha256"] == plan["run_preset"]["sha256"]
    assert not result_root.exists()
    if validate_plans:
        assert len(dispatched) == 3
        attestation = workspace_root / "host_attestations" / "test-host" / "controller_dry_run.json"
        record = json.loads(attestation.read_text(encoding="utf-8"))
        assert record["run_preset"] == plan["run_preset"]
        assert record["radiomic_calculation_started"] is False

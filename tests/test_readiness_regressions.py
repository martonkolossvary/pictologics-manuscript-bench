"""Regressions for the pre-production timing and host-control review."""

from functools import lru_cache
import json
import subprocess
import sys
import time
import types
from unittest import mock

import pandas as pd
import psutil
import pytest

from bench import env, report, run
from bench.adapters import pictologics_adapter
from bench.adapters.base import run_timed_computation
from bench.power_provenance import observe_task_power_state


def wait_dead(pid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.01)
    pytest.fail(f"Descendant {pid} survived task cleanup")


def descendant_script(pid_file, *, exit_parent):
    return (
        "import subprocess, sys, time; from pathlib import Path; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"Path({str(pid_file)!r}).write_text(str(p.pid)); "
        + ("print('done', flush=True)" if exit_parent else "time.sleep(60)")
    )


@pytest.mark.parametrize("exit_code", [0, 7])
def test_parent_exit_cleans_descendant_and_releases_pipes(tmp_path, exit_code):
    pid_file = tmp_path / "child.pid"
    command = [
        sys.executable,
        "-c",
        descendant_script(pid_file, exit_parent=True) + f"; sys.exit({exit_code})",
    ]
    if exit_code:
        with pytest.raises(run.AdapterProcessError) as caught:
            run._run_process_command(command, adapter_name="orphan", timeout=10)
        assert caught.value.returncode == exit_code
        stdout = caught.value.stdout
    else:
        stdout, _, _ = run._run_process_command(
            command, adapter_name="orphan", timeout=10
        )
    assert "done" in stdout
    wait_dead(int(pid_file.read_text()))


def test_interruption_retains_diagnostics_after_cleanup():
    started = time.monotonic()
    script = "import sys,time; print('started', flush=True); print('detail', file=sys.stderr, flush=True); time.sleep(30)"
    with pytest.raises(run.AdapterInterrupted) as caught:
        run._run_process_command(
            [sys.executable, "-c", script],
            adapter_name="interrupted",
            stop_requested=lambda: time.monotonic() - started > 1,
        )
    assert "started" in caught.value.stdout
    assert "detail" in caught.value.stderr
    assert caught.value.host_metrics["host_peak_rss_bytes"] > 0
    wait_dead(caught.value.pid)


def test_forced_controller_exit_cleans_worker_tree(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = descendant_script(pid_file, exit_parent=False)
    controller = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from bench.run import _run_process_command; import sys; "
            f'_run_process_command([sys.executable, "-c", {script!r}], adapter_name="crash")',
        ]
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists(), "fixture did not start"
        child_pid = int(pid_file.read_text())
        worker_pid = psutil.Process(child_pid).ppid()
        controller.kill()
        controller.wait(timeout=5)
        wait_dead(child_pid)
        wait_dead(worker_pid)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
        if child_pid is not None and psutil.pid_exists(child_pid):
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize("prepared", [False, True])
def test_timeout_retains_memory_cpu_events_and_bounded_output(prepared):
    script = (
        "import json, sys, time; x=bytearray(32*1024*1024); "
        'print("diagnostic", flush=True); '
        'print("x"*70000 + " stderr-end", file=sys.stderr, flush=True); '
        + (
            'print(\'BENCH_EVENT {"event":"worker_ready"}\', flush=True); '
            if prepared
            else ""
        )
        + "time.sleep(30)"
    )
    with pytest.raises(run.AdapterTimeout) as caught:
        run._run_process_command(
            [sys.executable, "-c", script],
            adapter_name="timeout",
            timeout=1,
            sample_interval=0.005,
        )
    exc = caught.value
    assert exc.host_metrics["host_peak_rss_bytes"] > 32 * 1024 * 1024
    assert exc.host_metrics["host_cpu_time_sec"] is not None
    assert exc.host_metrics["adapter_event_count"] == int(prepared)
    assert "diagnostic" in exc.stdout
    assert exc.stderr.endswith(" stderr-end\n")
    assert len(exc.stderr) <= 65536
    assert exc.censor_bound_scope == exc.phase + "_elapsed_wall_time"


def test_lifetime_peak_is_not_a_phase_measurement():
    events = [
        dict(event=name, worker_self_peak_rss_bytes=500 * 1024 * 1024, iteration=1)
        for name in ("worker_ready", "calculation_start", "calculation_complete")
    ]
    script = "\n".join(
        f"print({('BENCH_EVENT ' + json.dumps(event))!r}, flush=True)"
        for event in events
    )
    with mock.patch.object(run, "_rss_bytes", return_value=20 * 1024 * 1024):
        _, _, metrics = run._run_process_command(
            [sys.executable, "-c", script], adapter_name="memory", timeout=5
        )
    assert metrics["host_peak_rss_bytes"] == 500 * 1024 * 1024
    assert metrics["worker_lifetime_peak_rss_bytes"] == 500 * 1024 * 1024
    assert metrics["worker_ready_rss_bytes"] == 20 * 1024 * 1024
    assert metrics["calculation_peak_rss_bytes"] == 20 * 1024 * 1024
    assert metrics["memory_phase_observation_status"] == "partial"
    assert metrics["incremental_calculation_peak_rss_bytes"] is None
    with mock.patch.object(run, "_rss_bytes", return_value=0):
        _, _, missing = run._run_process_command(
            [sys.executable, "-c", script], adapter_name="memory", timeout=5
        )
    assert missing["worker_ready_rss_bytes"] is None
    assert missing["calculation_peak_rss_bytes"] is None
    assert missing["memory_phase_observation_status"] == "unavailable"


def test_local_intensity_geometry_is_rebuilt_inside_every_native_call():
    constructed = []

    @lru_cache(maxsize=32)
    def offsets():
        constructed.append(True)
        return 1

    module = types.ModuleType("pictologics.features.intensity")
    module._sphere_offsets_for_radius = offsets
    with mock.patch.dict(sys.modules, {"pictologics.features.intensity": module}):
        prepare = pictologics_adapter._calculation_preparer(["local_intensity"])
    offsets()  # package warmup has populated geometry
    calls = []

    def compute(_state):
        before = len(constructed)
        value = offsets()
        assert len(constructed) == before + 1
        calls.append(value)
        return {"value": float(value)}

    _, timing = run_timed_computation(
        compute,
        iterations=3,
        prepare_fn=prepare,
        target_observation_window_sec=1e-9,
        calibration_cv_threshold=0.99,
        calibration_span_ratio=1000,
    )
    assert len(calls) >= 5  # warmup, calibration, all measured calls
    assert len(constructed) == len(calls) + 1
    assert timing["measured_iterations"] == 3


@pytest.mark.parametrize("version", [None, "uv 0.9.0", "uv 0.7.22"])
def test_medimage_installer_rejected_before_environment_mutation(tmp_path, version):
    profile = env.load_runtime_profiles()["medimage"]
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing environment")
    with (
        mock.patch.object(env, "env_dir_for_profile", return_value=tmp_path),
        mock.patch.object(env, "_find_uv", return_value="uv" if version else None),
        mock.patch.object(
            env.subprocess, "run", return_value=mock.Mock(returncode=0, stdout=version)
        ),
        mock.patch.object(env, "_create_venv") as create,
    ):
        with pytest.raises(env.EnvironmentError, match="uv >=0.8,<0.9"):
            env.create_profile(profile, force=True)
    assert sentinel.read_text() == "existing environment"
    create.assert_not_called()


@pytest.mark.parametrize("active_returncode, expected", [(0, 0), (1, None)])
@pytest.mark.parametrize("active", ["AC Power", "Battery Power"])
def test_macos_never_uses_inactive_profile(active_returncode, expected, active):
    values = [mock.Mock(returncode=0, stdout=f"Now drawing from '{active}'\n")]
    content = (
        "Battery Power:\n lowpowermode 1\nAC Power:\n sleep 1\n"
        if active == "AC Power"
        else "Battery Power:\n sleep 1\nAC Power:\n lowpowermode 1\n"
    )
    values += [mock.Mock(returncode=0, stdout=content)] * 3
    values += [mock.Mock(returncode=active_returncode, stdout=" lowpowermode 0\n")]
    with (
        mock.patch("bench.power_provenance._completed", side_effect=values),
        mock.patch("bench.power_provenance.time.sleep"),
    ):
        state = observe_task_power_state("Darwin")
    assert state["pmset_lowpowermode"] == expected
    assert state["probe_attempts"] == 3


def test_public_timeout_export_retains_phase_and_bound_scope():
    rows = [
        dict(
            case_id="case",
            adapter="pictologics",
            workload="local_intensity",
            status="TIMED_OUT",
            censor_lower_bound_sec=1800,
            timeout_phase=phase,
            censor_bound_scope=phase + "_elapsed_wall_time",
        )
        for phase in ("startup_or_warmup", "prepared_calculation_region")
    ]
    for index, row in enumerate(rows):
        row["task_id"] = str(index)
    public = report.publication_task_observations(pd.DataFrame(rows), [])
    assert list(public["timeout_phase"]) == [row["timeout_phase"] for row in rows]
    assert list(public["censor_bound_scope"]) == [
        row["censor_bound_scope"] for row in rows
    ]
    frame = pd.DataFrame(rows)
    frame["_report_workload"] = frame["workload"]
    frame["_report_status"] = frame["status"]
    diagnostic = report._diagnostic_table(frame)
    assert set(diagnostic["timeout_phase"]) == set(public["timeout_phase"])

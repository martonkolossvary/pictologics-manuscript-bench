from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError, atomic_write_json
from bench.benchmark_models import task_plan_fingerprint
from scripts.resume_windows_benchmark import repair_retry_records
from tests.test_benchmark_ledger import make_run_spec, make_task


def fixture_ledger(tmp_path, legacy):
    failed = make_task()
    measured = replace(failed, ordinal=2, case_id="measured-case")
    spec = replace(
        make_run_spec(failed),
        task_plan_sha256=task_plan_fingerprint([failed, measured]),
    )
    with BenchmarkLedger(tmp_path / "benchmark.sqlite3") as ledger:
        ledger.initialize(spec, [failed, measured], resume=False)
        for task, status in ((failed, "failed"), (measured, "measured")):
            ledger.mark_running(task.task_id)
            record = {"task_id": task.task_id, "task_status": status, "attempt": 1}
            if status == "measured":
                digest = atomic_write_json(
                    tmp_path / "payload.json",
                    {
                        "benchmark": {
                            "task_id": task.task_id,
                            "status": status,
                            "run_fingerprint": spec.run_fingerprint,
                            "record": record,
                        }
                    },
                )
                ledger.mark_terminal(
                    task.task_id,
                    status,
                    record,
                    payload_path="payload.json",
                    payload_sha256=digest,
                    duration_sec=1,
                )
            else:
                ledger.mark_terminal(
                    task.task_id, status, record, error="worker failed"
                )
        if legacy:
            # Exact old recovery transition: failed record left on interrupted head.
            with ledger.connection:
                ledger.connection.execute(
                    "UPDATE tasks SET status='interrupted', error='prior failed attempt queued for explicit resume' WHERE task_id=?",
                    (failed.task_id,),
                )
                ledger.connection.execute(
                    "INSERT INTO events(created_at,event_type,task_id,details_json) VALUES(1,'failed_task_requeued',?,?)",
                    (
                        failed.task_id,
                        json.dumps(
                            {"failed_attempt": 1, "prior_error": "worker failed"}
                        ),
                    ),
                )
        measured_before = dict(ledger.task_rows()[1])
    return failed, measured_before


@pytest.mark.parametrize("legacy", [False, True])
def test_retry_repair_preserves_measured_data_and_attempt_history(tmp_path, legacy):
    failed, measured_before = fixture_ledger(tmp_path, legacy)
    result = repair_retry_records(tmp_path, tmp_path / "backups")
    assert result["repaired"] == 1
    assert result["measured"] == 1
    assert Path(result["backup"]).is_file()
    with BenchmarkLedger(tmp_path / "benchmark.sqlite3") as ledger:
        assert dict(ledger.task_rows()[1]) == measured_before
        records, payloads = ledger.verified_records_and_payloads()
        assert len(records) == len(payloads) == 1
        assert ledger.status(failed.task_id) == "interrupted"
        attempt = ledger.connection.execute(
            "SELECT status,error FROM task_attempts WHERE task_id=?", (failed.task_id,)
        ).fetchone()
        assert tuple(attempt) == ("failed", "worker failed")
        audit = json.loads(
            ledger.connection.execute(
                "SELECT details_json FROM events WHERE event_type='failed_record_recovery'"
            ).fetchone()[0]
        )
        assert (
            json.loads(audit["previous_task_row"]["record_json"])["task_status"]
            == "failed"
        )
        assert ledger.mark_running(failed.task_id) == 2
    assert repair_retry_records(tmp_path, tmp_path / "backups")["repaired"] == 0


def test_retry_repair_rejects_unexplained_status_mismatch(tmp_path):
    fixture_ledger(tmp_path, legacy=True)
    with sqlite3.connect(tmp_path / "benchmark.sqlite3") as connection:
        connection.execute("DELETE FROM events WHERE event_type='failed_task_requeued'")
    with pytest.raises(RunIntegrityError, match="lacks recovery evidence"):
        repair_retry_records(tmp_path, tmp_path / "backups")


def test_retry_repair_rolls_back_if_a_measured_payload_is_corrupt(tmp_path):
    failed, _ = fixture_ledger(tmp_path, legacy=False)
    (tmp_path / "payload.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="checksum changed"):
        repair_retry_records(tmp_path, tmp_path / "backups")
    with sqlite3.connect(tmp_path / "benchmark.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT status FROM tasks WHERE task_id=?", (failed.task_id,)
            ).fetchone()[0]
            == "failed"
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE event_type='failed_record_recovery'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows console regression")
@pytest.mark.parametrize("wrapper", [False, True])
def test_real_windows_ctrl_c_reaches_controller(tmp_path, wrapper):
    fixture = tmp_path / "bench"
    fixture.mkdir()
    (fixture / "__init__.py").write_text("", encoding="utf-8")
    (fixture / "cli.py").write_text(
        "import signal,time\nfrom pathlib import Path\n"
        "stopped=False\n"
        "def stop(*args):\n global stopped\n stopped=True\n"
        "signal.signal(signal.SIGBREAK,stop)\n"
        "Path('ready').write_text('ready')\n"
        "while not stopped: time.sleep(.01)\n"
        "Path('checkpoint').write_text('safe stop')\n",
        encoding="utf-8",
    )
    repository = str(Path(__file__).resolve().parents[1])
    # Isolated console: CTRL_C_EVENT cannot reach the user's benchmark/terminal.
    trigger = (
        "def trigger():\n"
        " deadline=time.monotonic()+10\n"
        " while not Path('ready').exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        " ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0,0)\n"
        "threading.Thread(target=trigger,daemon=True).start()\n"
    )
    probe = (
        f"import sys,ctypes,time,threading\nfrom pathlib import Path\nsys.path.insert(0,{repository!r})\n"
        + trigger
    )
    if wrapper:
        legacy = "import subprocess,sys; sys.exit(subprocess.Popen([sys.executable,'-m','bench.cli','run'],creationflags=subprocess.CREATE_NEW_PROCESS_GROUP).wait())"
        probe += "from scripts.resume_windows_benchmark import run_with_stop\n"
        probe += f"raise SystemExit(run_with_stop([sys.executable,'-c',{legacy!r}],cwd=Path.cwd()))\n"
    else:
        probe += "from scripts.launch_benchmark import _run_controller_command\nraise SystemExit(_run_controller_command([sys.executable,'-m','bench.cli','run']))\n"
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    process = subprocess.Popen(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startup,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
        assert (tmp_path / "checkpoint").read_text() == "safe stop"
    finally:
        if process.poll() is None:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
            )
            process.wait(timeout=5)

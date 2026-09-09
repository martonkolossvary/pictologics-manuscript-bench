#!/usr/bin/env python3
"""Resume an existing Windows run without changing its bound benchmark source.

Maintenance is performed under the normal run lock and backed up separately.
Only failed retry records are changed; measured payloads and fingerprints are
verified and preserved. Ctrl+C is forwarded directly to controller groups to
avoid the older launcher's infinite Windows wait.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid

import psutil


def repair_retry_records(report_dir: Path, backup_dir: Path) -> dict:
    # Import from the run's original repository (selected in main), never
    # substitute the newer controller/adapter implementations into a live run.
    from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError, RunLock

    path = report_dir / "benchmark.sqlite3"
    if not path.is_file():
        return {"repaired": 0, "measured": 0}
    with RunLock(report_dir / ".benchmark.lock"), BenchmarkLedger(path) as ledger:
        connection = ledger.connection
        candidates = []
        for row in connection.execute(
            "SELECT * FROM tasks WHERE status IN ('failed', 'interrupted')"
        ):
            raw = row["record_json"]
            if raw is None and row["status"] == "interrupted":
                continue
            record = json.loads(raw) if raw is not None else None
            if (
                row["status"] == "interrupted"
                and isinstance(record, dict)
                and record.get("task_status") == "interrupted"
            ):
                continue
            if (
                not isinstance(record, dict)
                or record.get("task_id") != row["task_id"]
                or record.get("task_status") != "failed"
                or record.get("attempt", row["attempt"]) != row["attempt"]
                or row["payload_path"] is not None
                or row["payload_sha256"] is not None
            ):
                raise RunIntegrityError(
                    "Retry repair rejected an unexpected task record"
                )
            attempt = connection.execute(
                "SELECT * FROM task_attempts WHERE task_id = ? AND attempt = ?",
                (row["task_id"], row["attempt"]),
            ).fetchone()
            if attempt is None or attempt["status"] != "failed":
                raise RunIntegrityError(
                    "Retry repair requires an intact failed attempt"
                )
            if row["status"] == "interrupted":
                evidence = connection.execute(
                    "SELECT details_json FROM events WHERE task_id = ? AND event_type = 'failed_task_requeued'",
                    (row["task_id"],),
                ).fetchall()
                if row[
                    "error"
                ] != "prior failed attempt queued for explicit resume" or not any(
                    json.loads(event[0]).get("failed_attempt") == row["attempt"]
                    and json.loads(event[0]).get("prior_error") == attempt["error"]
                    for event in evidence
                ):
                    raise RunIntegrityError(
                        "Interrupted/failed mismatch lacks recovery evidence"
                    )
            candidates.append(dict(row))

        if not candidates:
            _, payloads = ledger.verified_records_and_payloads()
            return {"repaired": 0, "measured": len(payloads)}

        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{report_dir.name}-{uuid.uuid4().hex}.sqlite3"
        with sqlite3.connect(backup) as destination:
            connection.backup(destination)

        measured_before = [
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM tasks WHERE status = 'measured' ORDER BY ordinal"
            )
        ]
        attempts_before = [
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM task_attempts ORDER BY task_id, attempt"
            )
        ]
        metadata_before = [
            tuple(row)
            for row in connection.execute("SELECT * FROM metadata ORDER BY key")
        ]
        with connection:
            for row in candidates:
                now = time.time()
                connection.execute(
                    "UPDATE tasks SET status = 'interrupted', record_json = NULL, "
                    "finished_at = ?, error = 'prior failed attempt queued for explicit resume' "
                    "WHERE task_id = ? AND attempt = ? AND record_json = ?",
                    (now, row["task_id"], row["attempt"], row["record_json"]),
                )
                connection.execute(
                    "INSERT INTO events(created_at,event_type,task_id,details_json) VALUES(?,?,?,?)",
                    (
                        now,
                        "failed_record_recovery",
                        row["task_id"],
                        json.dumps(
                            {
                                "failed_attempt": row["attempt"],
                                "previous_task_row": row,
                                "backup_path": str(backup),
                                "reason": "clear stale failed record before retry; preserve failed attempt",
                            },
                            sort_keys=True,
                        ),
                    ),
                )
            _, payloads = ledger.verified_records_and_payloads()
            if measured_before != [
                tuple(row)
                for row in connection.execute(
                    "SELECT * FROM tasks WHERE status = 'measured' ORDER BY ordinal"
                )
            ]:
                raise RunIntegrityError("Measured tasks changed during retry repair")
            if attempts_before != [
                tuple(row)
                for row in connection.execute(
                    "SELECT * FROM task_attempts ORDER BY task_id, attempt"
                )
            ]:
                raise RunIntegrityError("Attempt history changed during retry repair")
            if metadata_before != [
                tuple(row)
                for row in connection.execute("SELECT * FROM metadata ORDER BY key")
            ]:
                raise RunIntegrityError(
                    "Immutable metadata changed during retry repair"
                )
        return {
            "repaired": len(candidates),
            "measured": len(payloads),
            "backup": str(backup),
        }


def _controller_groups(launcher_pid: int) -> list[int]:
    try:
        children = psutil.Process(launcher_pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    controllers = {}
    for child in children:
        try:
            arguments = child.cmdline()
            if any(
                arguments[index : index + 3] == ["-m", "bench.cli", "run"]
                for index in range(len(arguments) - 2)
            ):
                controllers[child.pid] = child.ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Windows venv python.exe is a redirector. Its child is the real Python
    # process; the redirector PID is the group ID created by the launcher.
    return [pid for pid, parent in controllers.items() if parent not in controllers]


def run_with_stop(command: list[str], *, cwd: Path) -> int:
    if os.name != "nt":
        raise RuntimeError("This compatibility wrapper is for Windows runs")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Console signal handling requires the main thread")
    requests = 0

    def stop(_signum, _frame):
        nonlocal requests
        requests += 1

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGBREAK)
    }
    process = None
    forwarded = set()
    try:
        process = subprocess.Popen(
            command, cwd=cwd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        while True:
            if requests:
                for group in _controller_groups(process.pid):
                    if group not in forwarded:
                        try:
                            os.kill(group, signal.CTRL_BREAK_EVENT)
                            forwarded.add(group)
                            print(
                                "Stop requested; waiting for the controller checkpoint.",
                                flush=True,
                            )
                        except ProcessLookupError:
                            pass
                if requests >= 2:
                    # Explicit second Ctrl+C requests forced termination of
                    # this launcher tree only. Worker Job handles close too.
                    subprocess.run(
                        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                    )
            try:
                return int(process.wait(timeout=0.1))
            except subprocess.TimeoutExpired:
                continue
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--host-profile", default="configs/benchmark/hosts/windows-9800x3d-01.json"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Resume calculations after repair; default repairs and verifies only",
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    result_root = args.result_root.resolve()
    sys.path.insert(0, str(repository))
    profile = json.loads((repository / args.host_profile).read_text(encoding="utf-8"))
    machine = profile["machine_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", machine):
        raise ValueError("Invalid machine ID")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True
    ).strip():
        raise RuntimeError("Restore the clean original checkout before resuming")
    for pillar in (
        "pillar1_morphology",
        "pillar2_whole_anatomy",
        "pillar3_ibsi2_phase3",
    ):
        report_dir = result_root / machine / pillar
        metadata = report_dir / "run_meta.json"
        if (
            metadata.is_file()
            and json.loads(metadata.read_text(encoding="utf-8")).get("git_commit")
            != commit
        ):
            raise RuntimeError("Source commit differs from the existing run")
        result = repair_retry_records(
            report_dir, result_root.with_name(result_root.name + "-recovery")
        )
        print(json.dumps({"pillar": pillar, **result}), flush=True)
    if not args.execute:
        return 0
    return run_with_stop(
        [
            sys.executable,
            str(repository / "scripts/launch_benchmark.py"),
            "--workspace-root",
            str(repository / "data/benchmark"),
            "--result-root",
            str(result_root),
            "--host-profile",
            str(repository / args.host_profile),
            "--execute",
            "--confirm",
            "CALCULATE",
        ],
        cwd=repository,
    )


if __name__ == "__main__":
    raise SystemExit(main())

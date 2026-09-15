import json
import sqlite3

import pytest

from bench.benchmark_ledger import BenchmarkLedger
from bench.benchmark_models import STATUS_FAILED, STATUS_INTERRUPTED, STATUS_SKIPPED_TIMEOUT
from scripts.launch_benchmark import _with_observed_project_offset, PILLARS
from tests.test_benchmark_ledger import make_task, make_run_spec


@pytest.mark.parametrize('previous_status', [STATUS_FAILED, STATUS_INTERRUPTED])
def test_policy_skip_after_requeued_worker_preserves_attempt(previous_status, tmp_path):
    task = make_task()
    with BenchmarkLedger(tmp_path/'benchmark.sqlite3') as ledger:
        ledger.initialize(make_run_spec(task), [task], resume=False)
        assert ledger.mark_running(task.task_id) == 1
        ledger.mark_terminal(task.task_id, previous_status,
                             {'task_id': task.task_id, 'task_status': previous_status}, error='original error')
        old = tuple(ledger.connection.execute('SELECT * FROM task_attempts').fetchone())
        if previous_status == STATUS_FAILED:
            ledger.recover_failed()
        ledger.mark_terminal(task.task_id, STATUS_SKIPPED_TIMEOUT,
                             {'task_id': task.task_id, 'task_status': STATUS_SKIPPED_TIMEOUT, 'attempt': 0})
        attempts = ledger.connection.execute('SELECT * FROM task_attempts ORDER BY attempt').fetchall()
        assert tuple(attempts[0]) == old
        assert attempts[1]['attempt'] == 2
        row = ledger.task_row(task.task_id)
        assert row['attempt'] == json.loads(row['record_json'])['attempt'] == 2
        assert row['started_at'] is None
        ledger.connection.execute("UPDATE tasks SET status='pending', record_json=NULL WHERE task_id=?", (task.task_id,))
        ledger.connection.commit()
        assert ledger.mark_running(task.task_id) == 3


def test_project_progress_uses_other_ledgers_actual_terminal_counts(tmp_path):
    for pillar, _ in PILLARS:
        root = tmp_path/pillar
        root.mkdir()
        with sqlite3.connect(root/'benchmark.sqlite3') as connection:
            connection.execute('CREATE TABLE tasks(status TEXT)')
            connection.executemany('INSERT INTO tasks VALUES(?)', [('measured',), ('pending',), ('interrupted',), ('skipped_timeout_cutoff',)])
    command = ['python', '--report-dir', str(tmp_path/PILLARS[0][0]), '--project-task-offset', '999']
    assert _with_observed_project_offset(command)[-1] == '4'
    assert command[-1] == '999'

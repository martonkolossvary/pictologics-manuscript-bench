import json
from pathlib import Path
import sqlite3

from scripts.summarize_screening import summarize


def test_background_stop_marker_is_latched(tmp_path, monkeypatch):
    from bench.run import _SignalStop

    marker = tmp_path / "STOP"
    monkeypatch.setenv("PICTOBENCH_STOP_FILE", str(marker))
    stop = _SignalStop()
    assert not stop.requested
    marker.touch()
    assert stop.requested
    assert stop.signal_number is not None
    marker.unlink()
    assert stop.requested


def test_existing_stop_marker_prevents_controller_launch(tmp_path, monkeypatch):
    from scripts.launch_benchmark import _run_controller_command

    marker = tmp_path / "STOP"
    marker.touch()
    def unexpected(*args, **kwargs):
        raise AssertionError("must not start a controller")
    monkeypatch.setattr("scripts.launch_benchmark.subprocess.Popen", unexpected)
    assert _run_controller_command(["unused"], marker) == 130


def test_live_wal_and_pending_repeats_are_reported_without_modifying_ledger(tmp_path: Path):
    root = tmp_path / "host"
    path = root / "pillar1_morphology" / "benchmark.sqlite3"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE metadata(key TEXT, value TEXT)")
        writer.execute("CREATE TABLE tasks(ordinal INTEGER, spec_json TEXT, case_id TEXT, adapter TEXT, workload_key TEXT, repeat INTEGER, status TEXT, attempt INTEGER, error TEXT, duration_sec REAL, record_json TEXT)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        for ordinal, case_id, status, repeat in (
            (1, "success", "measured", 1), (2, "success", "pending", 2),
            (3, "slow", "timed_out_censored", 1), (4, "unknown", "pending", 1),
            (5, "mixed", "measured", 1), (6, "mixed", "timed_out_censored", 2),
        ):
            spec = {"shape": [32, 64, 96], "requested_families": ["glcm"], "mask_voxels": 20}
            record = {"host_wall_time_sec": 901, "censor_lower_bound_sec": 900} if status == "timed_out_censored" else {"host_wall_time_sec": 20}
            writer.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                ordinal, json.dumps(spec), case_id, "test", "texture", repeat,
                status, int(status != "pending"), None, .1 if status == "measured" else None, json.dumps(record),
            ))
        writer.commit()
        summary, rows = summarize(root)
        assert len(rows) == 6  # Includes records still only in the WAL.
        assert rows[2]["whole_task_elapsed_seconds"] == 901
        assert rows[2]["censor_lower_bound_seconds"] == 900
        assert rows[2]["calculation_median_seconds"] is None
        assert rows[0]["calculation_median_seconds"] == .1
        assert rows[2]["shape"] == "32x64x96"
        pillar = summary["pillars"]["pillar1_morphology"]
        assert pillar["configurations"] == 4
        assert pillar["successful_configurations"] == 1
        assert pillar["unattempted_configurations"] == 1
        assert pillar["other_configurations"] == 1
        assert pillar["mixed_success_and_issue_configurations"] == 1
        assert summary["pillars"]["pillar2_whole_anatomy"]["status"] == "missing"
        assert writer.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 6

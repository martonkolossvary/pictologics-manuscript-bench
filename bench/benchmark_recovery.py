"""Offline, explicitly requested amendment of the defective v1 memory policy.

Measurement bytes and their original fingerprints remain unchanged. The ledger
records which exact historical payloads predate the amendment. Normal resume
still rejects all unapproved specification/source changes.
"""
from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any, Mapping

from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError, atomic_write_json
from bench.benchmark_memory import LEGACY_MEMORY_POLICY_ID, MEMORY_POLICY_ID, MemoryPreflightPolicy
from bench.benchmark_models import RunSpec, TaskSpec, canonical_json, fingerprint, run_spec_identity

LEGACY_SOURCE_COMMIT = "7282e02c3dac0180f00c391eecc487b07c8d83cf"
REPAIRABLE_EXECUTION_FILES = frozenset({
    "bench/benchmark_memory.py", "bench/benchmark_ledger.py", "bench/run.py",
    "bench/cli.py", "bench/benchmark_eta.py", "bench/benchmark_guardrails.py",
})


def _git_bytes(repository: Path, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{LEGACY_SOURCE_COMMIT}:{relative}"],
                            cwd=repository, capture_output=True, check=False)
    if result.returncode:
        raise RunIntegrityError(f"cannot read the audited legacy source: {relative}")
    return result.stdout


def source_evidence(repository: Path, old: Mapping[str, Any]) -> dict[str, Any]:
    from bench.run import _benchmark_execution_sources, _benchmark_sources_sha256

    tree = ast.parse(_git_bytes(repository, "bench/run.py"))
    source_paths = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EXECUTION_SOURCE_FILES"
            for target in node.targets
        ):
            source_paths = ast.literal_eval(node.value)
    if not isinstance(source_paths, tuple):
        raise RunIntegrityError("legacy execution dependency list is unavailable")
    paths = set(source_paths) | {f"bench/adapters/{a}_adapter.py" for a in old["adapters"]}
    # Git blobs can have LF while the audited Windows checkout has CRLF. Match
    # the exact old aggregate, rather than assuming either checkout convention.
    blobs = {p: _git_bytes(repository, p) for p in paths}
    candidates = [
        {p: hashlib.sha256(blob).hexdigest() for p, blob in blobs.items()},
        {p: hashlib.sha256(blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
         for p, blob in blobs.items()},
    ]
    legacy = next((mapping for mapping in candidates if
        _benchmark_sources_sha256(old["adapters"], execution_files=mapping)
        == old["benchmark_sources_sha256"]), None)
    if legacy is None:
        raise RunIntegrityError(
            "legacy execution source/controller environment differs from the audited version; "
            "use the original controller Python environment"
        )
    current = _benchmark_execution_sources(old["adapters"])
    if current.keys() != legacy.keys():
        raise RunIntegrityError("execution dependency set changed beyond the memory repair")
    changed = {p for p in current if current[p] != legacy[p]}
    line_endings_only = {p for p in changed if
        (repository/p).read_bytes().replace(b"\r\n", b"\n") == blobs[p].replace(b"\r\n", b"\n")}
    unexpected = changed-REPAIRABLE_EXECUTION_FILES-line_endings_only
    if unexpected:
        raise RunIntegrityError("non-repair execution files changed: " + ", ".join(sorted(unexpected)))
    return {"legacy_commit": LEGACY_SOURCE_COMMIT, "changed_execution_files": sorted(changed),
            "line_ending_only_changes": sorted(line_endings_only),
            "old_execution_files": legacy, "new_execution_files": current,
            "new_sources_sha256": _benchmark_sources_sha256(old["adapters"]),
            "repair_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def corrected_spec(old: Mapping[str, Any], new_sources_sha256: str, *, repeat_timeouts: bool = False) -> RunSpec:
    policy = dict(old.get("memory_preflight") or {})
    if old.get("schema_version") != 15 or policy.get("policy_id") != LEGACY_MEMORY_POLICY_ID:
        raise RunIntegrityError("only schema-15 runs with the defective v1 memory policy can be repaired")
    kwargs = {key: policy[key] for key in (
        "budget_fraction", "reserve_bytes", "user_cap_bytes", "safety_factor",
        "empirical_growth_exponent", "fixed_overhead_bytes")}
    kwargs["adapter_input_multipliers"] = tuple(sorted(policy["adapter_input_multipliers"].items()))
    kwargs["quadratic_mask_terms"] = tuple((x["adapter"], x["workload"], x["bytes_per_mask_voxel_pair"])
                                            for x in policy["quadratic_mask_terms"])
    updated = MemoryPreflightPolicy(**kwargs).to_dict()
    expected_legacy = dict(updated)
    expected_legacy.update(policy_id=LEGACY_MEMORY_POLICY_ID,
        empirical_scope={"primary": ["adapter", "workload", "guardrail_group"],
                         "cross_stratum_upper": ["adapter", "workload"]})
    expected_legacy.pop("empirical_anchor")
    expected_legacy.pop("empirical_baseline")
    if canonical_json(expected_legacy) != canonical_json(policy):
        raise RunIntegrityError("legacy memory policy contains unsupported changes")
    values = {field.name: old[field.name] for field in fields(RunSpec) if field.name in old}
    guardrail = json.loads(canonical_json(old.get("guardrail") or {}))
    guardrail.setdefault("timeout_cutoff", {})["skip_identical_repeats"] = not repeat_timeouts
    values.update(memory_preflight=updated, benchmark_sources_sha256=new_sources_sha256)
    values["guardrail"] = guardrail
    spec = RunSpec.create(**values)
    for key, value in old.items():
        if key not in {"memory_preflight", "benchmark_sources_sha256", "guardrail"} and spec.to_dict().get(key) != value:
            raise RunIntegrityError(f"memory repair would change unrelated protocol field: {key}")
    return spec


def plan_repair(ledger: BenchmarkLedger, repository: Path, *, repeat_timeouts: bool = False) -> dict[str, Any]:
    old = json.loads(ledger.metadata_value("run_spec_json") or "null")
    if not isinstance(old, dict) or fingerprint(run_spec_identity(old)) != ledger.metadata_value("run_fingerprint"):
        raise RunIntegrityError("stored specification does not match the run fingerprint")
    records, payloads = ledger.verified_records_and_payloads()
    if old.get("memory_preflight", {}).get("policy_id") == MEMORY_POLICY_ID:
        if old.get("guardrail", {}).get("timeout_cutoff", {}).get("skip_identical_repeats") != (not repeat_timeouts):
            raise RunIntegrityError("requested timeout-repeat policy differs from the recorded recovery")
        amendments = ledger.protocol_amendments()
        if not amendments:
            raise RunIntegrityError("run is already v2 but has no recovery amendment")
        from bench.run import _benchmark_sources_sha256
        if old["benchmark_sources_sha256"] != _benchmark_sources_sha256(old["adapters"]):
            raise RunIntegrityError("source changed after the recorded recovery")
        return {"already_repaired": True, "measured": len(payloads), "new_spec": old,
                "new_fingerprint": ledger.metadata_value("run_fingerprint"), "requeue": []}
    if any(row["status"] == "running" for row in ledger.task_rows()):
        raise RunIntegrityError("run has an unfinished running task; stop/resume it cleanly with the original source first")
    evidence = source_evidence(repository, old)
    new = corrected_spec(old, evidence["new_sources_sha256"], repeat_timeouts=repeat_timeouts)
    preflights = {r["task_id"] for r in records if r["task_status"] == "skipped_memory_preflight"
                  and r.get("memory_preflight_policy_id") == LEGACY_MEMORY_POLICY_ID}
    candidates = []
    for row in ledger.task_rows():
        record = json.loads(row["record_json"] or "{}")
        if (row["task_id"] in preflights or
                (row["status"] == "skipped_memory_cutoff"
                 and record.get("memory_cutoff_evidence_task_id") in preflights)):
            if row["payload_path"] is not None or row["payload_sha256"] is not None:
                raise RunIntegrityError("memory-skip recovery found unexpected payload metadata")
            candidates.append(dict(row))
    return {"already_repaired": False, "old_spec": old, "new_spec": new.to_dict(),
            "old_fingerprint": ledger.metadata_value("run_fingerprint"),
            "new_fingerprint": new.run_fingerprint, "measured": len(payloads),
            "source_evidence": evidence, "preflight_ids": sorted(preflights), "requeue": candidates}


def apply_repair(ledger: BenchmarkLedger, plan: Mapping[str, Any], backup_dir: Path) -> dict[str, Any]:
    """Caller must hold the normal run lock; one atomic ledger transaction."""
    if plan["already_repaired"]:
        return {"already_repaired": True, "requeued": 0, "measured": plan["measured"]}
    if ledger.metadata_value("run_fingerprint") != plan["old_fingerprint"]:
        raise RunIntegrityError("run changed after recovery planning")
    if any(row["status"] == "running" for row in ledger.task_rows()):
        raise RunIntegrityError("cannot amend a running task")
    backup_dir.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(backup_dir/"benchmark.sqlite3") as target:
        ledger.connection.backup(target)
    for name in ("run_spec.json", "run_meta.json", "host_observations.json"):
        path = ledger.path.parent/name
        if path.is_file():
            (backup_dir/name).write_bytes(path.read_bytes())
    atomic_write_json(backup_dir/"recovery_plan.json", dict(plan))
    now = time.time()
    amendment_id = fingerprint([plan["old_fingerprint"], plan["new_fingerprint"], now])
    connection = ledger.connection
    measured_before = [tuple(row) for row in connection.execute("SELECT * FROM tasks WHERE status='measured' ORDER BY ordinal")]
    attempts_before = [tuple(row) for row in connection.execute("SELECT * FROM task_attempts ORDER BY task_id,attempt")]
    with connection:
        connection.execute("INSERT INTO protocol_amendments VALUES(?,?,?,?,?,?,?)", (
            amendment_id, now, plan["old_fingerprint"], plan["new_fingerprint"],
            canonical_json(plan["old_spec"]), canonical_json(plan["new_spec"]),
            canonical_json({"reason":"repair false memory estimates; retain historical measurement provenance",
                            "backup":str(backup_dir), "source_evidence":plan["source_evidence"]})))
        connection.execute("INSERT OR IGNORE INTO measurement_origins SELECT task_id, ?, ?, payload_sha256 FROM tasks WHERE status='measured'",
                           (amendment_id, plan["old_fingerprint"]))
        for row in plan["requeue"]:
            current = dict(ledger.task_row(row["task_id"]))
            if current != row:
                raise RunIntegrityError("task changed after recovery planning")
            connection.execute("INSERT INTO events(created_at,event_type,task_id,details_json) VALUES(?,?,?,?)",
                (now,"memory_policy_requeued",row["task_id"],canonical_json({"amendment_id":amendment_id,"previous_task_row":row})))
            connection.execute("UPDATE tasks SET status='pending', started_at=NULL, finished_at=NULL, duration_sec=NULL, record_json=NULL, error=NULL WHERE task_id=?",(row["task_id"],))
        for task_id in plan["preflight_ids"]:
            connection.execute("DELETE FROM memory_cutoffs WHERE evidence_task_id=?",(task_id,))
        for key,value in {"run_fingerprint":plan["new_fingerprint"],"run_spec_json":canonical_json(plan["new_spec"]),
                          "run_status":"interrupted","updated_at":str(now)}.items():
            connection.execute("INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(key,value))
        if measured_before != [tuple(row) for row in connection.execute("SELECT * FROM tasks WHERE status='measured' ORDER BY ordinal")]:
            raise RunIntegrityError("measured task rows changed during recovery")
        if attempts_before != [tuple(row) for row in connection.execute("SELECT * FROM task_attempts ORDER BY task_id,attempt")]:
            raise RunIntegrityError("attempt history changed during recovery")
        ledger.verified_records_and_payloads()
    return {"already_repaired":False,"requeued":len(plan["requeue"]),"measured":plan["measured"],
            "backup":str(backup_dir),"amendment_id":amendment_id}


def sync_recovery_exports(ledger: BenchmarkLedger, repository: Path) -> None:
    """Idempotently finish filesystem writes if a previous repair was interrupted."""
    amendments = ledger.protocol_amendments()
    if not amendments:
        raise RunIntegrityError("no recorded recovery to export")
    spec = json.loads(ledger.metadata_value("run_spec_json"))
    path = ledger.path.parent/"run_spec.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        allowed = [spec] + [json.loads(item["old_spec_json"]) for item in amendments]
        if existing not in allowed:
            raise RunIntegrityError("run_spec.json changed outside the recorded recovery")
    atomic_write_json(path,spec)
    meta_path=ledger.path.parent/"run_meta.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=repository,capture_output=True,text=True,check=True).stdout.strip()
    meta.update(run_fingerprint=ledger.metadata_value("run_fingerprint"),git_commit=commit,
                run_status=ledger.metadata_value("run_status"),status_counts=ledger.status_counts(),
                protocol_amendments=amendments,derived_exports_stale=True,
                memory_cutoffs=[dict(r) for r in ledger.connection.execute("SELECT * FROM memory_cutoffs")])
    atomic_write_json(meta_path,meta)

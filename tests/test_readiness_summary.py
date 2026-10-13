"""Readiness must preserve coverage limits and fail closed on stale evidence."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from bench.benchmark_ledger import sha256_file
from bench.benchmark_models import fingerprint
from bench.compliance.models import ComparisonRecord
from bench.compliance.report import comparison_to_csv
from scripts import summarize_readiness as readiness


def _row(tag, **overrides):
    row = ComparisonRecord(
        specification="IBSI 1", phase="phase1", adapter="mirp", software_version="2.7.0",
        configuration="digital_phantom", profile="fixture", aggregation="not_applicable",
        family="intensity", feature_name=tag, feature_tag=tag, semantic_key=tag,
        ibsi_code=tag, standardized=True, observed_supported=True, mapped=True,
        attempted=True, finite=True, referencable=True, evaluated=True, passed=True,
        status="pass", value=1.0, reference_value=1.0, tolerance=0.01,
    )
    return replace(row, **overrides)


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    rows = [
        _row("passed"),
        _row("unsupported", observed_supported=False, mapped=False, attempted=False,
             finite=False, value=None, evaluated=False, passed=None, status="unsupported"),
        _row("coverage_only", standardized=False, reference_value=None, tolerance=None,
             referencable=False, evaluated=False, passed=None, status="not_standardized"),
    ]
    # The real official-table validator is independently covered by compliance
    # tests. This small reference grid isolates evidence/hash/coverage behavior.
    references = [SimpleNamespace(**row.to_dict()) for row in rows]
    monkeypatch.setattr(readiness, "_reference_rows", lambda run: references)
    profile = {"version": "2.7.0", "distribution": "mirp", "distribution_metadata_version": "2.7.0"}
    freeze = ["mirp==2.7.0"]
    digest = hashlib.sha256("\n".join(freeze).encode()).hexdigest()
    run = {
        "schema_version": 1, "kind": "ibsi1_digital_phantom", "adapters": ["mirp"],
        "controller_source_sha256": "recorded-source", "tasks": ["mirp__intensity"],
        "configured_adapter_profiles": {"mirp": profile},
        "verified_adapter_environments": {"mirp": {
            **profile, "profile": "mirp", "platform": "Windows", "machine": "AMD64",
            "freeze": freeze, "freeze_sha256": digest,
            "environment_lock": {"freeze_sha256": digest},
        }},
    }
    run["fingerprint"] = fingerprint(run)
    _write(root / "run_manifest.json", run)
    _write(root / "payload.json", {"values": "test payload"})
    _write(root / "run_state.json", {"schema_version": 1, "tasks": {"mirp__intensity": {
        "status": "completed", "payload_path": "payload.json",
        "payload_sha256": sha256_file(root / "payload.json"),
    }}})
    _write_rows(root, rows)
    return root, rows


def _write_rows(root, rows, complete=True):
    (root / "comparisons.csv").write_text(comparison_to_csv(rows), encoding="utf-8")
    _write(root / "result_manifest.json", {
        "schema_version": 1, "kind": "ibsi1_digital_phantom", "record_count": len(rows),
        "comparison_table_sha256": sha256_file(root / "comparisons.csv"), "complete": complete,
    })


def test_summary_preserves_accuracy_and_coverage_denominators(bundle):
    root, _ = bundle
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = readiness.summarize_run(root)
    assert result["integrity"] == "recorded_evidence_verified"
    assert result["status"] == "qualified_with_coverage_limits"
    counts = result["adapters"][0]["counts"]
    assert counts["defined_instances"] == 3
    assert counts["standardized_instances"] == 2
    assert counts["coverage_only_instances"] == 1
    assert counts["unsupported_standardized"] == 1
    assert counts["conditional_accuracy"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert counts["overall_standardized_success"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize("outcome,expected", [("fail", "numerical_failures"),
                                             ("error", "execution_or_mapping_errors"),
                                             ("missing", "incomplete")])
def test_complete_manifest_is_not_a_pass(bundle, outcome, expected):
    root, rows = bundle
    if outcome == "fail":
        rows[0] = replace(rows[0], value=2.0, passed=False, status="fail")
    else:
        rows[0] = replace(rows[0], finite=False, value=None, evaluated=False, passed=None, status=outcome)
    _write_rows(root, rows)
    result = readiness.summarize_run(root)
    assert result["status"] == "needs_review"
    assert result["adapters"][0]["status"] == expected


@pytest.mark.parametrize("mutation", ["csv", "fingerprint", "version", "duplicate", "payload", "false_pass"])
def test_stale_or_inconsistent_evidence_never_qualifies(bundle, mutation):
    root, rows = bundle
    if mutation == "csv":
        with (root / "comparisons.csv").open("a") as stream:
            stream.write("changed")
    elif mutation == "fingerprint":
        run = json.loads((root / "run_manifest.json").read_text())
        run["controller_source_sha256"] = "changed"
        _write(root / "run_manifest.json", run)
    elif mutation == "payload":
        _write(root / "payload.json", {"changed": True})
    else:
        if mutation == "version":
            rows[0] = replace(rows[0], software_version="different")
        elif mutation == "duplicate":
            rows[1] = rows[0]
        else:
            rows[0] = replace(rows[0], value=2.0)
        _write_rows(root, rows)
    result = readiness.summarize_run(root)
    assert result["status"] == "invalid_evidence"
    assert result["integrity"] == "unverified"
    assert not result["adapters"]


def test_running_or_missing_evidence_is_incomplete(bundle, tmp_path):
    root, rows = bundle
    _write_rows(root, rows, complete=False)
    assert readiness.summarize_run(root)["adapters"][0]["status"] == "incomplete"
    (root / "result_manifest.json").unlink()
    assert readiness.summarize_run(root)["status"] == "incomplete"
    assert readiness.summarize_run(tmp_path / "absent")["status"] == "missing"


def test_cli_writes_separate_snapshot_and_defers_timing(bundle, tmp_path, capsys):
    root, _ = bundle
    output = tmp_path / "snapshot.json"
    assert readiness.main(["--run-dir", str(root), "--output", str(output)]) == 0
    snapshot = json.loads(output.read_text())
    assert snapshot["final_timing_status"] == "deferred_until_pictologics_finalized"
    assert "timing" not in snapshot["runs"][0]["adapters"][0]["counts"]
    with pytest.raises(SystemExit):
        readiness.main(["--run-dir", str(root), "--output", str(root / "snapshot.json")])
    capsys.readouterr()

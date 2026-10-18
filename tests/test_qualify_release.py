from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import qualify_release as release


def environments():
    return {name: {"profile": name, "version": "1.0", "distribution_metadata_version": "1.0",
                   "python": "3.12.1", "profile_fingerprint": name + "-profile",
                   "freeze_sha256": name + "-freeze", "environment_lock": {"sha256": name},
                   "platform": "Windows", "machine": "AMD64"}
            for name in release.ADAPTERS}


def snapshot():
    envs = environments()
    source = {f"bench/adapters/{name}_adapter.py": name + "-source" for name in release.ADAPTERS}
    source["bench/run.py"] = "protocol-source"
    return {"schema_version": 1, "kind": "untimed_release_qualification", "complete": True,
            "status": "core_checks_only", "platform": "windows-amd64", "scope": "core",
            "requested_adapters": list(release.ADAPTERS), "environments": envs,
            "source_sha256": source, "stable_identity": release.stable_identity(envs, source),
            "feature_inventory": {a: {"intensity": {"mean": 1.0}} for a in release.ADAPTERS},
            "diagnostic_fixture_identity": {"fixture": "hash"}}


def save_baseline(root):
    root.mkdir()
    (root / "evidence.bin").write_bytes(b"saved evidence")
    report = snapshot()
    report["artifacts_sha256"] = release.artifact_identity(root)
    report["report_fingerprint"] = release.report_fingerprint(report)
    path = root / "release-qualification.json"
    release.write_json(path, report)
    return path, report


def test_plan_is_untimed_and_has_dependency_order(tmp_path):
    plan = release.build_plan(tmp_path, ["pictologics"], workspace=tmp_path / "workspace")
    seen = set()
    for step in plan:
        assert set(step["requires"]) <= seen
        seen.add(step["name"])
        assert step["command"][:2] == [release.sys.executable, "-m"]
        assert "benchmark" not in step["command"]
        assert "env" not in step["command"]
        assert "--resume" not in step["command"]
    assert {"smoke", "calculations", "spatial", "robustness", "spatial_policy", "ibsi1", "phase1", "phase2"} <= seen
    assert "--verify-only" in next(s["command"] for s in plan if s["name"] == "inputs")
    assert "--workspace-root" in next(s["command"] for s in plan if s["name"] == "robustness")
    for stage, table, manifest in (("ibsi1", "ibsi1_references.csv", "ibsi1_reference_manifest.json"),
                                   ("phase2", "ibsi2_phase2_references.csv", "ibsi2_phase2_reference_manifest.json")):
        command = next(s["command"] for s in plan if s["name"] == stage)
        assert command[command.index("--references") + 1].endswith(table)
        assert command[command.index("--reference-manifest") + 1].endswith(manifest)
    assert len(release.build_plan(tmp_path, ["pictologics"], scope="core")) == 5
    assert "--verify-only" not in next(s["command"] for s in release.build_plan(tmp_path, ["pictologics"], bootstrap=True) if s["name"] == "inputs")


def test_plan_does_not_create_outputs_or_verify_environments(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(release, "environment_snapshot", lambda: pytest.fail("plan launched verification"))
    output = tmp_path / "uncreated"
    assert release.main(["--plan", "--output-dir", str(output)]) == 0
    assert not output.exists()
    assert len(json.loads(capsys.readouterr().out)) == 13


@pytest.mark.parametrize("argument", ["nan", "inf", "0", "-1"])
def test_invalid_watchdogs_are_rejected(argument, tmp_path):
    with pytest.raises(SystemExit):
        release.main(["--output-dir", str(tmp_path), "--step-timeout", argument])


def test_verified_baseline_rejects_changed_binary_or_summary(tmp_path):
    path, report = save_baseline(tmp_path / "baseline")
    assert release.load_baseline(path) == report
    (path.parent / "evidence.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence files"):
        release.load_baseline(path)
    (path.parent / "evidence.bin").write_bytes(b"saved evidence")
    report["feature_inventory"]["pictologics"]["intensity"]["mean"] = 5
    release.write_json(path, report)
    with pytest.raises(ValueError, match="checksummed"):
        release.load_baseline(path)


def test_baseline_rejects_extra_files_and_missing_inventory(tmp_path):
    path, report = save_baseline(tmp_path / "baseline")
    (path.parent / "unrecorded.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="evidence files"):
        release.load_baseline(path)
    report["artifacts_sha256"] = release.artifact_identity(path.parent)
    del report["feature_inventory"]["mirp"]
    report["report_fingerprint"] = release.report_fingerprint(report)
    release.write_json(path, report)
    with pytest.raises(ValueError, match="inventory adapter grid"):
        release.load_baseline(path)


@pytest.mark.parametrize("field,value", [("platform", "darwin-arm64"), ("scope", "full"), ("requested_adapters", ["pictologics"])])
def test_baseline_comparison_requires_matching_selection(field, value):
    before, after = snapshot(), snapshot()
    after[field] = value
    with pytest.raises(ValueError):
        release.compare_baseline(before, after)


def test_baseline_exposes_candidate_feature_and_stable_protocol_changes():
    before, after = snapshot(), snapshot()
    after["environments"]["pictologics"]["version"] = "2.0"
    after["source_sha256"]["bench/adapters/pictologics_adapter.py"] = "candidate-source"
    after["feature_inventory"]["pictologics"]["intensity"] = {"mean": 1.5, "extra": 2}
    comparison = release.compare_baseline(before, after)
    assert comparison["comparison_is_matched"]
    assert comparison["stable_adapters_frozen"]
    assert comparison["current_pictologics_version"] == "2.0"
    assert comparison["feature_changes"][0]["added"] == ["extra"]
    assert comparison["feature_changes"][0]["value_changes"] == [{"feature": "mean", "before": 1.0, "after": 1.5}]
    after["stable_identity"]["mirp"]["freeze_sha256"] = "changed"
    after["source_sha256"]["bench/run.py"] = "changed"
    after["diagnostic_fixture_identity"] = {"fixture": "changed"}
    comparison = release.compare_baseline(before, after)
    assert not comparison["comparison_is_matched"]
    assert comparison["stable_adapter_changes"] == ["mirp"]
    assert comparison["shared_protocol_source_changes"] == ["bench/run.py"]
    assert comparison["diagnostic_fixture_changes"]


def test_full_baseline_exposes_changed_common_preprocessing_and_compliance_coverage():
    before = snapshot()
    before["scope"] = "full"
    before["compliance"] = {
        "ibsi1": {"adapters": [{"adapter": "pictologics", "status": "qualified_with_coverage_limits",
                                "counts": {"evaluated_standardized": 168}}], "numerical_findings": []},
        "phase1": {"adapters": {"pictologics": {"evaluated": 33}}, "failures": [], "status_counts": {"pass": 33}},
        "phase2": {"adapters": [], "numerical_findings": []},
    }
    after = deepcopy(before)
    after["environments"]["pictologics"]["freeze_sha256"] = "new-candidate-environment"
    after["compliance"]["ibsi1"]["adapters"][0]["counts"]["evaluated_standardized"] = 169
    comparison = release.compare_baseline(before, after)
    assert comparison["stable_adapters_frozen"]
    assert comparison["ibsi2_phase2_preprocessing_changed"]
    assert not comparison["comparison_is_matched"]
    assert [item["phase"] for item in comparison["compliance_changes"]] == ["ibsi1"]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", None])
def test_inventory_rejects_nonfinite_and_non_numeric(value):
    with pytest.raises(ValueError):
        release.validate_inventory({"pictologics": {"intensity": {"mean": value}}}, ["pictologics"])


def test_inventory_and_comparison_detect_removed_workloads_and_tolerance():
    before, after = snapshot(), snapshot()
    before["feature_inventory"]["pictologics"]["texture"] = {"contrast": 2.0}
    after["feature_inventory"]["pictologics"]["intensity"]["mean"] += 1e-10
    result = release.compare_baseline(before, after)
    assert len(result["feature_changes"]) == 1
    assert result["feature_changes"][0]["removed"] == ["contrast"]


def calculations_report(tmp_path):
    from bench.adapters.registry import get_adapter
    from bench.benchmark_workloads import WORKLOADS

    directory = tmp_path / "calculations"
    directory.mkdir()
    fixture = directory / "fixture.bin"
    fixture.write_bytes(b"fixture")
    checks = []
    for workload in WORKLOADS:
        if get_adapter("pyradiomics").supports_workload(workload.name):
            checks.append({"workload": workload.name, "status": "passed", "feature_names": ["a"],
                           "values": {"a": 1}, "public_equivalence": {"features": 1}, "repeat_equivalence": {"features": 1}})
        else:
            checks.append({"workload": workload.name, "status": "unsupported"})
    report = {"kind": "untimed_prepared_calculation_qualification", "complete": True, "status": "passed",
              "timing_observations": False, "requested_adapters": ["pyradiomics"],
              "fixture": {"raw": {"path": str(fixture), "sha256": release.sha256(fixture)}},
              "adapters": [{"adapter": "pyradiomics", "status": "passed", "qualification": {
                  "adapter": "pyradiomics", "timing_observations": False, "checks": checks}}]}
    path = directory / "qualification.json"
    release.write_json(path, report)
    return path, report


def test_calculations_requires_exact_grid_and_complete_equivalence(tmp_path):
    path, report = calculations_report(tmp_path)
    assert release.inspect_calculations(tmp_path, ["pyradiomics"])
    report["adapters"][0]["qualification"]["checks"].append(deepcopy(report["adapters"][0]["qualification"]["checks"][0]))
    release.write_json(path, report)
    with pytest.raises(ValueError, match="workload grid"):
        release.inspect_calculations(tmp_path, ["pyradiomics"])
    report["adapters"][0]["qualification"]["checks"].pop()
    report["adapters"][0]["qualification"]["checks"][0]["repeat_equivalence"]["features"] = 0
    release.write_json(path, report)
    with pytest.raises(ValueError, match="repeat_equivalence"):
        release.inspect_calculations(tmp_path, ["pyradiomics"])


def phase1_report(tmp_path):
    from bench.compliance.references import IBSI2_PHASE1_TEST_IDS, IBSI2_PHASE1_NONSTANDARDIZED_IDS

    for name in ("phase1", "references", "candidates"):
        (tmp_path / name).mkdir()
    release.write_json(tmp_path / "references/phase1.json", {})
    release.write_json(tmp_path / "candidates/ibsi2_phase1_candidates.json", {})
    rows = [{"adapter": "pictologics", "test_id": test, "native_supported": True,
             "standardized": test not in IBSI2_PHASE1_NONSTANDARDIZED_IDS,
             "evaluated": test not in IBSI2_PHASE1_NONSTANDARDIZED_IDS,
             "passed": True if test not in IBSI2_PHASE1_NONSTANDARDIZED_IDS else None,
             "status": "pass" if test not in IBSI2_PHASE1_NONSTANDARDIZED_IDS else "not_standardized",
             "candidate_supplied": True} for test in IBSI2_PHASE1_TEST_IDS]
    manifest = {"schema_version": 1, "kind": "ibsi2_phase1_response_map_compliance", "adapters": ["pictologics"],
                "defined_rows": len(rows), "evaluated_rows": sum(r["evaluated"] for r in rows),
                "passed_rows": sum(r["passed"] is True for r in rows), "publication_complete": True,
                "reference_manifest_sha256": release.sha256(tmp_path / "references/phase1.json"),
                "candidate_manifest_sha256": release.sha256(tmp_path / "candidates/ibsi2_phase1_candidates.json")}
    return rows, manifest


def write_phase1(tmp_path, rows, manifest):
    release.write_json(tmp_path / "phase1/phase1_results.json", rows)
    manifest["results_sha256"] = release.sha256(tmp_path / "phase1/phase1_results.json")
    release.write_json(tmp_path / "phase1/result_manifest.json", manifest)


def test_phase1_retains_numerical_failure_and_coverage_but_rejects_missing_results(tmp_path):
    rows, manifest = phase1_report(tmp_path)
    rows[0].update(passed=False, status="fail")
    manifest["passed_rows"] -= 1
    write_phase1(tmp_path, rows, manifest)
    result = release.inspect_phase1(tmp_path, ["pictologics"])
    assert result["status"] == "numerical_findings"
    assert len(result["failures"]) == 1
    assert result["adapters"]["pictologics"]["standardized"] == 33
    assert result["adapters"]["pictologics"]["overall_standardized_success"]["denominator"] == 33
    rows[0].update(evaluated=False, passed=None, status="candidate_not_supplied", candidate_supplied=False)
    write_phase1(tmp_path, rows, manifest)
    with pytest.raises(ValueError, match="inconsistent or missing"):
        release.inspect_phase1(tmp_path, ["pictologics"])


def test_phase1_rejects_duplicate_grid_even_when_manifest_claims_complete(tmp_path):
    rows, manifest = phase1_report(tmp_path)
    rows[-1] = deepcopy(rows[0])
    write_phase1(tmp_path, rows, manifest)
    with pytest.raises(ValueError, match="missing or duplicated"):
        release.inspect_phase1(tmp_path, ["pictologics"])


def test_phase1_rejects_another_configured_version(tmp_path):
    rows, manifest = phase1_report(tmp_path)
    manifest["configured_adapter_profiles"] = {"pictologics": {
        "version": "old", "distribution_metadata_version": "old", "distribution": "pictologics"}}
    write_phase1(tmp_path, rows, manifest)
    expected = environments()
    expected["pictologics"]["distribution"] = "pictologics"
    with pytest.raises(ValueError, match="configured version"):
        release.inspect_phase1(tmp_path, ["pictologics"], expected)


def test_policy_retains_observed_legacy_failure_as_reviewable_evidence(tmp_path):
    from scripts.check_pictologics_spatial_policy import PROBES, assess_acceptance

    report = {"kind": "untimed_pictologics_spatial_policy", "observations_complete": True,
              "physical_memory": {"bytes": 32 << 30},
              "probes": [{"probe": name, "status": "observed", "native_cap_bytes": 4 << 30,
                          "version": "1.0"} for name in PROBES]}
    report["acceptance"] = assess_acceptance(report["probes"], report["physical_memory"])
    release.write_json(tmp_path / "spatial-policy.json", report)
    assert release.inspect_policy(tmp_path, environments())["acceptance"]["status"] == "failed"
    report["acceptance"]["status"] = "passed"
    release.write_json(tmp_path / "spatial-policy.json", report)
    with pytest.raises(ValueError, match="inconsistent"):
        release.inspect_policy(tmp_path, environments())


def mock_assessment(tmp_path, monkeypatch, *, policy="passed", fail_step=False, changed_environment=False):
    initial = snapshot()
    after = deepcopy(initial["environments"])
    if changed_environment:
        after["pictologics"]["freeze_sha256"] = "changed"
    snapshots = iter([initial["environments"], after])
    monkeypatch.setattr(release, "environment_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(release, "source_identity", lambda: initial["source_sha256"])
    monkeypatch.setattr(release, "build_plan", lambda *a, **kw: [
        {"name": "first", "command": [], "requires": []},
        {"name": "dependent", "command": [], "requires": ["first"]}])
    calls = []

    def run(step, output, env, timeout):
        calls.append(step["name"])
        (output / "logs" / (step["name"] + ".log")).write_text("retained evidence")
        return dict(step, status="execution_failed" if fail_step else "completed")

    monkeypatch.setattr(release, "run_step", run)
    monkeypatch.setattr(release, "inspect_smoke", lambda *a: [])
    monkeypatch.setattr(release, "inspect_calculations", lambda *a: initial["feature_inventory"])
    monkeypatch.setattr(release, "inspect_spatial", lambda *a: {})
    monkeypatch.setattr(release, "inspect_robustness", lambda *a: {"status": "qualified_with_degenerate_limits"})
    monkeypatch.setattr(release, "inspect_policy", lambda *a: {"acceptance": {"status": policy}})
    monkeypatch.setattr(release, "fixture_identity", lambda *a: initial["diagnostic_fixture_identity"])
    args = SimpleNamespace(output_dir=tmp_path / "assessment", baseline=None, scope="core",
                           profiles=list(release.ADAPTERS), bootstrap_inputs=False,
                           workspace_root=None, step_timeout=10)
    return args, calls


def test_assessment_accepts_robustness_success_and_seals_logs(tmp_path, monkeypatch):
    args, calls = mock_assessment(tmp_path, monkeypatch)
    report = release.assess(args)
    assert report["complete"] and report["status"] == "core_checks_only"
    assert calls == ["first", "dependent"]
    assert report["report_fingerprint"] == release.report_fingerprint(report)
    assert "logs/first.log" in report["artifacts_sha256"]
    assert release.load_baseline(args.output_dir / "release-qualification.json") == report


def test_assessment_marks_legacy_policy_as_needs_review(tmp_path, monkeypatch):
    args, _ = mock_assessment(tmp_path, monkeypatch, policy="failed")
    report = release.assess(args)
    assert report["complete"] and report["status"] == "needs_review"


def test_assessment_retains_robustness_findings_as_reviewable_evidence(tmp_path, monkeypatch):
    args, _ = mock_assessment(tmp_path, monkeypatch)
    monkeypatch.setattr(release, "inspect_robustness", lambda *a: {"status": "needs_review"})
    report = release.assess(args)
    assert report["complete"] and report["status"] == "needs_review"


def test_assessment_blocks_dependencies_after_failure(tmp_path, monkeypatch):
    args, calls = mock_assessment(tmp_path, monkeypatch, fail_step=True)
    report = release.assess(args)
    assert not report["complete"]
    assert calls == ["first"]
    assert report["steps"][1]["status"] == "blocked_by_failed_dependency"


def test_assessment_rejects_candidate_environment_change_during_run(tmp_path, monkeypatch):
    args, _ = mock_assessment(tmp_path, monkeypatch, changed_environment=True)
    report = release.assess(args)
    assert not report["complete"]
    assert "Adapter environments changed" in report["errors"][0]


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    args, _ = mock_assessment(tmp_path, monkeypatch)
    args.output_dir.mkdir()
    preserved = args.output_dir / "old.json"
    preserved.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="fresh output"):
        release.assess(args)
    assert preserved.read_bytes() == b"unchanged"


def test_run_step_retains_logs_for_a_nonzero_exit(tmp_path):
    (tmp_path / "logs").mkdir()
    step = {"name": "example", "requires": [], "command": [release.sys.executable, "-c",
            "import sys; print('retained stdout'); print('retained stderr', file=sys.stderr); sys.exit(7)"]}
    report = release.run_step(step, tmp_path, dict(release.os.environ), 10)
    assert report["status"] == "execution_failed" and report["returncode"] == 7
    assert (tmp_path / "logs/example.stdout.log").read_text().strip() == "retained stdout"
    assert report["stderr_sha256"] == release.sha256(tmp_path / "logs/example.stderr.log")


def test_run_step_timeout_stops_owned_process_tree_and_records_failure(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    calls = []

    class Process:
        pid = 123456789

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                raise release.subprocess.TimeoutExpired("example", timeout)
            return 0

    monkeypatch.setattr(release.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(release.subprocess, "run", lambda command, **kw: (calls.append(("killtree", command)) or SimpleNamespace(returncode=0)))
    monkeypatch.setattr(release.os, "killpg", lambda pid, signal: calls.append(("killgroup", pid)), raising=False)
    report = release.run_step({"name": "example", "command": ["example"], "requires": []}, tmp_path, {}, 1)
    assert report["status"] == "execution_failed" and "timed out" in report["error"]
    if release.os.name == "nt":
        assert calls[1] == ("killtree", ["taskkill", "/PID", "123456789", "/T", "/F"])
    else:
        assert calls[1] == ("killgroup", 123456789)
    assert calls[-1] == ("wait", 30)


def test_json_reader_rejects_nonfinite_and_duplicate_keys(tmp_path):
    path = tmp_path / "bad.json"
    for value in ('{"value": NaN}', '{"value": 1e999}', '{"value": 1, "value": 2}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            release.read_json(path)

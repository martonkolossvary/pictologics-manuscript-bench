"""Run a fresh, untimed release assessment without installing or timing adapters.

Exit 0: requested checks completed without findings (core scope is partial).
Exit 1: missing, invalid, incomplete or failed execution evidence.
Exit 2: completed evidence has numerical findings or baseline changes to review.
No exit code authorizes performance timing or constitutes IBSI certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ("medimage", "mirp", "pyradiomics", "zrad", "pictologics")
STABLE = ADAPTERS[:-1]
SUCCESS = {"core_checks_only", "qualified_with_coverage_limits", "needs_review"}
LIMITATIONS = [
    "Untimed diagnostics do not authorize final timing or constitute IBSI certification.",
    "Core scope excludes IBSI checks; full scope covers IBSI 1 digital phantom and IBSI 2 Phases 1/2 only.",
    "Feature inventory is observed on bounded diagnostic fixtures, not every possible clinical input.",
    "Prepared/public/repeat agreement does not independently establish numerical accuracy.",
    "Real-input crops, when requested, do not validate full clinical volumes or IBSI 2 Phase 3.",
    "The actual publication host, large-volume feasibility and peak memory remain unqualified.",
    "Checksums detect changed evidence; they do not authenticate who produced it.",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def number(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else invalid(value)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def report_fingerprint(report):
    content = {key: value for key, value in report.items() if key != "report_fingerprint"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def artifact_identity(output):
    result = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == output / "release-qualification.json":
            continue
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError(f"Evidence file escapes output directory: {path}")
        result[path.relative_to(output).as_posix()] = sha256(path)
    return result


def load_baseline(path):
    path = path.expanduser().resolve()
    report = read_json(path)
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("kind") != "untimed_release_qualification"
            or report.get("complete") is not True or report.get("status") not in SUCCESS
            or report.get("report_fingerprint") != report_fingerprint(report)):
        raise ValueError("Baseline must be a completed, checksummed release qualification snapshot")
    hashes = report.get("artifacts_sha256")
    if not hashes or hashes != artifact_identity(path.parent):
        raise ValueError("Baseline evidence files are missing, changed or unexpected")
    if report.get("stable_identity") != stable_identity(report["environments"], report["source_sha256"]):
        raise ValueError("Baseline stable adapter identity is inconsistent")
    validate_inventory(report["feature_inventory"], report["requested_adapters"])
    return report


def source_identity():
    paths = [*ROOT.joinpath("bench").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
             *ROOT.joinpath("configs").rglob("*.json"), *ROOT.joinpath("configs/adapters").glob("*.yaml"),
             *ROOT.joinpath("reproducibility/environment-locks").rglob("*.txt"),
             ROOT / "reproducibility/inputs/manifest.json",
             *ROOT.joinpath("reproducibility/contracts").glob("*"),
             ROOT / "pyproject.toml", ROOT / "poetry.lock"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def environment_snapshot():
    """Verify all frozen environments, even when only Pictologics is assessed."""
    from bench import env

    profiles = env.load_runtime_profiles()
    return {name: env.verify_profile(profiles[name], smoke=False) for name in ADAPTERS}


def stable_identity(environments, source):
    return {name: {
        "version": environments[name]["version"],
        "distribution_metadata_version": environments[name]["distribution_metadata_version"],
        "python": environments[name]["python"],
        "profile_fingerprint": environments[name]["profile_fingerprint"],
        "freeze_sha256": environments[name]["freeze_sha256"],
        "environment_lock": environments[name]["environment_lock"],
        "adapter_source_sha256": source[f"bench/adapters/{name}_adapter.py"],
    } for name in STABLE}


def validate_inventory(inventory, profiles):
    if not profiles or len(profiles) != len(set(profiles)) or set(inventory) != set(profiles):
        raise ValueError("Feature inventory adapter grid differs from requested adapters")
    for adapter, workloads in inventory.items():
        if not isinstance(workloads, dict) or not workloads:
            raise ValueError(f"Empty feature inventory: {adapter}")
        for workload, values in workloads.items():
            if not isinstance(values, dict) or not values or any(
                not isinstance(name, str) or not name or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                for name, value in values.items()
            ):
                raise ValueError(f"Invalid feature values: {adapter}/{workload}")


def compare_baseline(previous, current):
    """Compare a verified saved snapshot; platform-specific freezes stay separate."""
    if previous.get("kind") != "untimed_release_qualification" or previous.get("complete") is not True:
        raise ValueError("Baseline must be a completed release qualification snapshot")
    if previous.get("platform") != current["platform"]:
        raise ValueError("Release baselines must use the same native platform")
    if (previous.get("scope") != current["scope"]
            or set(previous.get("requested_adapters", [])) != set(current["requested_adapters"])):
        raise ValueError("Baseline scope and requested adapter set must match the current assessment")
    stable_changes = [name for name in STABLE
                      if previous.get("stable_identity", {}).get(name) != current["stable_identity"][name]]
    before = previous.get("feature_inventory", {})
    after = current.get("feature_inventory", {})
    validate_inventory(before, previous["requested_adapters"])
    validate_inventory(after, current["requested_adapters"])
    changes = []
    for adapter in sorted(before):
        for workload in sorted(set(before[adapter]) | set(after[adapter])):
            old, new = before[adapter].get(workload, {}), after[adapter].get(workload, {})
            added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
            values = [{"feature": name, "before": old[name], "after": new[name]}
                      for name in sorted(set(old) & set(new))
                      if not math.isclose(old[name], new[name], rel_tol=1e-9, abs_tol=1e-12)]
            if added or removed or values:
                changes.append({"adapter": adapter, "workload": workload,
                                "added": added, "removed": removed, "value_changes": values})
    old_source, new_source = previous["source_sha256"], current["source_sha256"]
    source_changes = [path for path in sorted(set(old_source) | set(new_source))
                      if old_source.get(path) != new_source.get(path)]
    candidate_paths = {"bench/adapters/pictologics_adapter.py", "configs/adapters/pictologics.yaml"}
    protocol_changes = [path for path in source_changes if path not in candidate_paths
                        and not (path.startswith("reproducibility/environment-locks/")
                                 and path.endswith("/pictologics.txt"))]
    fixture_changed = previous.get("diagnostic_fixture_identity") != current.get("diagnostic_fixture_identity")
    preprocessing_changed = (current["scope"] == "full" and any(
        previous["environments"]["pictologics"].get(key) != current["environments"]["pictologics"].get(key)
        for key in ("profile_fingerprint", "freeze_sha256")))
    compliance_changes = []
    if current["scope"] == "full":
        for phase in ("ibsi1", "phase1", "phase2"):
            old, new = previous["compliance"][phase], current["compliance"][phase]
            def outcome(value):
                if phase == "phase1":
                    return {key: value[key] for key in ("adapters", "failures", "status_counts")}
                return {"adapters": {item["adapter"]: {"status": item["status"], "counts": item["counts"]}
                                     for item in value["adapters"]}, "numerical_findings": value["numerical_findings"]}
            if outcome(old) != outcome(new):
                compliance_changes.append({"phase": phase, "before": outcome(old), "after": outcome(new)})
    return {"stable_adapters_frozen": not stable_changes,
            "stable_adapter_changes": stable_changes, "feature_changes": changes,
            "source_changes": source_changes, "shared_protocol_source_changes": protocol_changes,
            "diagnostic_fixture_changes": fixture_changed,
            "ibsi2_phase2_preprocessing_changed": preprocessing_changed,
            "ibsi2_phase2_attribution": "Common preprocessing uses Pictologics, including for the four frozen extraction adapters.",
            "compliance_changes": compliance_changes,
            "comparison_is_matched": not (stable_changes or protocol_changes or fixture_changed or preprocessing_changed),
            "compared_adapters": sorted(before), "rtol": 1e-9, "atol": 1e-12,
            "previous_pictologics_version": previous.get("environments", {}).get("pictologics", {}).get("version"),
            "current_pictologics_version": current["environments"]["pictologics"]["version"]}


def build_plan(output, profiles, *, scope="full", bootstrap=False, workspace=None):
    """All commands are allow-listed untimed entry points; no generic shell input."""
    python = sys.executable
    plan = []

    def add(name, module, args, *, requires=()):
        plan.append({"name": name, "command": [python, "-m", module, *map(str, args)],
                     "requires": list(requires)})

    add("smoke", "scripts.smoke_adapter_environments", ["--profiles", *profiles, "--output-dir", output / "smoke"])
    add("calculations", "scripts.qualify_adapter_calculations", ["--profiles", *profiles, "--output-dir", output / "calculations"])
    # Spatial comparison deliberately includes the four native implementations.
    add("spatial", "scripts.qualify_spatial_statistics", ["--output-dir", output / "spatial"])
    robustness = ["--profiles", *profiles, "--output-dir", output / "robustness"]
    if workspace is not None:
        robustness.extend(["--workspace-root", workspace])
    add("robustness", "scripts.qualify_adapter_robustness", robustness)
    add("spatial_policy", "scripts.check_pictologics_spatial_policy", ["--output", output / "spatial-policy.json"])
    if scope == "core":
        return plan
    inputs = ["--component", "ibsi1", "--component", "ibsi2-phase1",
              "--component", "ibsi2-phase2", "--component", "workbook"]
    if not bootstrap:
        inputs.append("--verify-only")
    add("inputs", "scripts.bootstrap_reproducibility_inputs", inputs)
    references = output / "references"
    add("ibsi1_reference", "bench.cli", ["compliance", "import-ibsi1", "--workbook",
        ROOT / "data/downloads/IBSI-1-submission-table.xlsx", "--output-dir", references / "ibsi1"], requires=["inputs"])
    add("phase1_reference", "bench.cli", ["compliance", "validate-ibsi2-phase1", "--reference-dir",
        ROOT / "data/ibsi2/references/phase1/maps", "--manifest-out", references / "phase1.json"], requires=["inputs"])
    add("phase2_reference", "bench.cli", ["compliance", "import-ibsi2-phase2", "--csv",
        ROOT / "data/ibsi2/references/phase2/source/reference_values.csv", "--output-dir", references / "phase2"], requires=["inputs"])
    selected = ",".join(profiles)
    add("ibsi1", "bench.cli", ["compliance", "run-ibsi1", "--adapters", selected,
        "--image", ROOT / "data/ibsi1/digital_phantom/image/phantom.nii.gz",
        "--mask", ROOT / "data/ibsi1/digital_phantom/mask/mask.nii.gz",
        "--references", references / "ibsi1/ibsi1_references.csv",
        "--reference-manifest", references / "ibsi1/ibsi1_reference_manifest.json",
        "--output-dir", output / "ibsi1", "--timeout", "600"], requires=["ibsi1_reference"])
    add("candidates", "bench.cli", ["compliance", "generate-ibsi2-candidates", "--adapters", selected,
        "--phases", "phase1,phase2", "--output-dir", output / "candidates"], requires=["inputs"])
    add("phase1", "bench.cli", ["compliance", "evaluate-ibsi2-phase1", "--adapters", selected,
        "--reference-manifest", references / "phase1.json", "--reference-dir", ROOT / "data/ibsi2/references/phase1/maps",
        "--candidate-manifest", output / "candidates/ibsi2_phase1_candidates.json",
        "--output-dir", output / "phase1"], requires=["candidates", "phase1_reference"])
    add("phase2", "bench.cli", ["compliance", "run-ibsi2-phase2", "--adapters", selected,
        "--candidate-manifest", output / "candidates/ibsi2_phase2_candidates.json",
        "--references", references / "phase2/ibsi2_phase2_references.csv",
        "--reference-manifest", references / "phase2/ibsi2_phase2_reference_manifest.json",
        "--output-dir", output / "phase2", "--timeout", "600"], requires=["candidates", "phase2_reference"])
    return plan


def run_step(step, output, environment, timeout):
    """Keep logs on disk and stop the entire owned process tree on timeout."""
    record = dict(step)
    paths = {name: output / "logs" / f"{step['name']}.{name}.log"
             for name in ("stdout", "stderr")}
    try:
        with paths["stdout"].open("w", encoding="utf-8") as stdout, paths["stderr"].open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(step["command"], cwd=ROOT, env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=os.name != "nt")
            try:
                returncode = process.wait(timeout=timeout)
            except BaseException:
                # subprocess.run's timeout kills only the immediate controller;
                # adapter workers must not continue after an interrupted stage.
                if os.name == "nt":
                    stopped = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                             capture_output=True, timeout=30, check=False)
                    if stopped.returncode and process.poll() is None:
                        raise RuntimeError("Could not stop qualification process tree")
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=30)
                raise
        record.update(returncode=returncode,
                      status="completed" if returncode == 0 else "execution_failed")
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        record.update(status="execution_failed", error=str(exc))
    finally:
        for name, path in paths.items():
            if path.is_file():
                record[f"{name}_sha256"] = sha256(path)
    return record


def require_environment(actual, expected, adapter):
    keys = ("profile", "version", "distribution_metadata_version", "profile_fingerprint",
            "freeze_sha256", "environment_lock", "python", "platform", "machine")
    if any(actual.get(key) != expected[adapter].get(key) for key in keys):
        raise ValueError(f"Stage environment differs from verified snapshot: {adapter}")


def inspect_smoke(output, profiles):
    from bench.adapters.registry import get_adapter
    from bench.benchmark_workloads import WORKLOADS

    rows = read_json(output / "smoke/smoke-results.json")
    expected = Counter((adapter, workload.name) for adapter in profiles for workload in WORKLOADS
                       if get_adapter(adapter).supports_workload(workload.name))
    if (Counter((r["adapter"], r["workload"]) for r in rows) != expected
            or any(r.get("status") != "passed" or type(r.get("finite_features")) is not int
                   or r["finite_features"] <= 0 for r in rows)):
        raise ValueError("Smoke result grid is missing, duplicated or unsuccessful")
    return rows


def inspect_phase1(output, profiles, environments=None):
    """Verify the fresh response-map result grid and retain numerical failures."""
    from bench.compliance.references import IBSI2_PHASE1_TEST_IDS, IBSI2_PHASE1_NONSTANDARDIZED_IDS

    manifest = read_json(output / "phase1/result_manifest.json")
    rows = read_json(output / "phase1/phase1_results.json")
    if (manifest.get("schema_version") != 1
            or manifest.get("kind") != "ibsi2_phase1_response_map_compliance"
            or Counter(manifest.get("adapters", [])) != Counter(profiles)):
        raise ValueError("Phase 1 manifest identity differs")
    if environments is not None:
        recorded = manifest.get("configured_adapter_profiles", {})
        for adapter in profiles:
            if any(recorded.get(adapter, {}).get(key) != environments[adapter][key]
                   for key in ("version", "distribution_metadata_version", "distribution")):
                raise ValueError(f"Phase 1 configured version differs: {adapter}")
    if sha256(output / "phase1/phase1_results.json") != manifest["results_sha256"]:
        raise ValueError("Phase 1 result checksum mismatch")
    if sha256(output / "references/phase1.json") != manifest["reference_manifest_sha256"]:
        raise ValueError("Phase 1 reference checksum mismatch")
    if sha256(output / "candidates/ibsi2_phase1_candidates.json") != manifest["candidate_manifest_sha256"]:
        raise ValueError("Phase 1 candidate checksum mismatch")
    expected = Counter((adapter, test) for adapter in profiles for test in IBSI2_PHASE1_TEST_IDS)
    if Counter((row["adapter"], row["test_id"]) for row in rows) != expected:
        raise ValueError("Phase 1 requested grid is missing or duplicated")
    for row in rows:
        standardized = row["test_id"] not in IBSI2_PHASE1_NONSTANDARDIZED_IDS
        if row.get("standardized") is not standardized or type(row.get("native_supported")) is not bool:
            raise ValueError("Phase 1 support or standardization identity differs")
        if not row["native_supported"]:
            valid = (row.get("status") == "native_unsupported" and row.get("evaluated") is False
                     and row.get("passed") is None and row.get("candidate_supplied") is False)
        elif not standardized:
            valid = (row.get("status") == "not_standardized" and row.get("evaluated") is False
                     and row.get("passed") is None and row.get("candidate_supplied") is True)
        else:
            valid = (row.get("evaluated") is True and type(row.get("passed")) is bool
                     and row.get("candidate_supplied") is True
                     and row.get("status") in {"pass", "fail", "geometry_mismatch", "nonfinite_candidate"}
                     and (row["status"] == "pass") == row["passed"])
        if not valid:
            raise ValueError(f"Phase 1 inconsistent or missing result: {row['adapter']}/{row['test_id']}")
    if (manifest.get("defined_rows") != len(rows)
            or manifest.get("evaluated_rows") != sum(r["evaluated"] for r in rows)
            or manifest.get("passed_rows") != sum(r["passed"] is True for r in rows)
            or manifest.get("publication_complete") is not True or manifest.get("processing_failures")
            or manifest.get("missing_supported_candidate_maps")):
        raise ValueError("Phase 1 execution evidence is incomplete")
    failed = [row for row in rows if row.get("passed") is False]
    counts = {}
    for adapter in profiles:
        selected = [row for row in rows if row["adapter"] == adapter]
        evaluated = sum(row["evaluated"] for row in selected)
        passed = sum(row["passed"] is True for row in selected)
        standardized = sum(row["standardized"] for row in selected)
        counts[adapter] = {
            "defined": len(selected), "standardized": standardized, "evaluated": evaluated,
            "passed": passed, "native_supported": sum(row["native_supported"] for row in selected),
            "native_unsupported": sum(not row["native_supported"] for row in selected),
            "conditional_accuracy": {"numerator": passed, "denominator": evaluated,
                                     "value": passed / evaluated if evaluated else None},
            "overall_standardized_success": {"numerator": passed, "denominator": standardized,
                                             "value": passed / standardized if standardized else None},
        }
    return {"status": "numerical_findings" if failed else "qualified_with_coverage_limits",
            "defined": len(rows), "evaluated": sum(bool(r.get("evaluated")) for r in rows),
            "passed": sum(r.get("passed") is True for r in rows), "failures": failed,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "adapters": counts}


def inspect_calculations(output, profiles, environments=None):
    from bench.adapters.registry import get_adapter
    from bench.benchmark_workloads import WORKLOADS

    report = read_json(output / "calculations/qualification.json")
    if (report.get("complete") is not True or report.get("status") != "passed"
            or report.get("kind") != "untimed_prepared_calculation_qualification"
            or report.get("timing_observations") is not False
            or Counter(report.get("requested_adapters", [])) != Counter(profiles)):
        raise ValueError("Calculation qualification did not pass")
    records = report["adapters"]
    if Counter(r["adapter"] for r in records) != Counter(profiles):
        raise ValueError("Calculation qualification adapter grid differs")
    inventory = {}
    for record in records:
        adapter = record["adapter"]
        worker = record["qualification"]
        if record.get("status") != "passed" or worker.get("adapter") != adapter or worker.get("timing_observations") is not False:
            raise ValueError(f"Invalid calculation worker identity: {adapter}")
        if environments is not None:
            require_environment(record["environment"], environments, adapter)
            if worker.get("distribution_version") != environments[adapter]["distribution_metadata_version"]:
                raise ValueError(f"Calculation worker version differs: {adapter}")
        expected = {w.name for w in WORKLOADS if get_adapter(adapter).supports_workload(w.name)}
        if Counter(c["workload"] for c in worker["checks"]) != Counter(w.name for w in WORKLOADS):
            raise ValueError(f"Incomplete calculation workload grid: {adapter}")
        checks = [c for c in worker["checks"] if c["status"] == "passed"]
        if Counter(c["workload"] for c in checks) != Counter(expected):
            raise ValueError(f"Incomplete calculation feature inventory: {adapter}")
        for check in worker["checks"]:
            if check["workload"] not in expected:
                if check["status"] != "unsupported":
                    raise ValueError("Unsupported workload has an invalid outcome")
                continue
            names = check.get("feature_names", [])
            if Counter(names) != Counter(check["values"].keys()):
                raise ValueError("Feature name/value inventory differs")
            for name in ("public_equivalence", "repeat_equivalence"):
                if check.get(name, {}).get("features") != len(names):
                    raise ValueError(f"Missing complete {name} evidence")
        inventory[adapter] = {c["workload"]: c["values"] for c in checks}
    validate_inventory(inventory, profiles)
    for value in report["fixture"].values():
        if sha256(value["path"]) != value["sha256"]:
            raise ValueError("Calculation fixture checksum mismatch")
    return inventory


def inspect_spatial(output, environments):
    report = read_json(output / "spatial/spatial-qualification.json")
    if (report.get("kind") != "untimed_spatial_qualification"
            or report.get("complete") is not True or report.get("errors")):
        raise ValueError("Spatial qualification is incomplete or failed")
    adapters = ("pictologics", "medimage", "mirp", "zrad")
    if set(report["environments"]) != set(adapters):
        raise ValueError("Spatial environment grid differs")
    for adapter in adapters:
        require_environment(report["environments"][adapter], environments, adapter)
    expected = Counter()
    for fixture in ("isotropic", "anisotropic"):
        expected.update((adapter, fixture, None, None, "passed") for adapter in adapters)
        expected.update(("pictologics", fixture, method, None, "passed")
                        for method in ("pairwise", "fft", "memory_guard_fallback"))
        expected.update(("pictologics", fixture, None, case, "expected_undefined")
                        for case in ("empty_roi", "single_voxel", "constant_intensity"))
    expected.update([("pyradiomics", None, None, None, "unsupported")])
    observed = Counter((row.get("adapter"), row.get("fixture"), row.get("method"),
                        row.get("case"), row.get("status")) for row in report["checks"])
    if observed != expected:
        raise ValueError("Spatial result grid is missing, duplicated or unsuccessful")
    if Counter(f["name"] for f in report["fixtures"]) != Counter(["isotropic", "anisotropic"]):
        raise ValueError("Spatial fixture grid differs")
    for fixture in report["fixtures"]:
        for kind in ("image", "mask"):
            if sha256(output / "spatial" / f"{fixture['name']}-{kind}.nii.gz") != fixture[f"{kind}_sha256"]:
                raise ValueError("Spatial fixture checksum mismatch")
    return report


def inspect_policy(output, environments):
    from scripts.check_pictologics_spatial_policy import assess_acceptance, PROBES

    report = read_json(output / "spatial-policy.json")
    if (report.get("kind") != "untimed_pictologics_spatial_policy"
            or report.get("observations_complete") is not True
            or Counter(p.get("probe") for p in report["probes"]) != Counter(PROBES)
            or any(p.get("status") != "observed" for p in report["probes"])):
        raise ValueError("Spatial policy observations are incomplete or unsupported")
    if any(p.get("version") != environments["pictologics"]["distribution_metadata_version"] for p in report["probes"]):
        raise ValueError("Spatial policy worker version differs")
    if report["acceptance"] != assess_acceptance(report["probes"], report["physical_memory"]):
        raise ValueError("Spatial policy acceptance outcome is inconsistent")
    return report


def inspect_compliance(output, profiles, environments=None):
    from scripts.summarize_readiness import summarize_run
    from bench.compliance.report import load_comparison_csv

    results = {}
    for name in ("ibsi1", "phase2"):
        summary = summarize_run(output / name)
        if (summary["integrity"] != "recorded_evidence_verified"
                or not summary.get("processing_complete")
                or set(a["adapter"] for a in summary["adapters"]) != set(profiles)
                or any(a["status"] not in {"qualified", "qualified_with_coverage_limits", "numerical_failures"}
                       for a in summary["adapters"])):
            raise ValueError(f"{name} missing, invalid or unsuccessful execution evidence: {summary}")
        if environments is not None:
            recorded = read_json(output / name / "run_manifest.json")["verified_adapter_environments"]
            for adapter in profiles:
                require_environment(recorded[adapter], environments, adapter)
        summary["numerical_findings"] = [
            {"adapter": r.adapter, "version": r.software_version, "ibsi_code": r.ibsi_code,
             "feature": r.feature_name, "value": r.value, "reference": r.reference_value, "tolerance": r.tolerance}
            for r in load_comparison_csv(output / name / "comparisons.csv") if r.passed is False
        ]
        results[name] = summary
    results["phase1"] = inspect_phase1(output, profiles, environments)
    return results


def inspect_robustness(output, profiles, environments):
    from scripts.qualify_adapter_robustness import validate_report

    report = read_json(output / "robustness/robustness.json")
    validate_report(report, profiles)
    for record in report["adapters"]:
        require_environment(record["environment"], environments, record["adapter"])
    return report


def fixture_identity(output, robustness, spatial):
    calculations = read_json(output / "calculations/qualification.json")
    return {
        "calculations": {name: item["sha256"] for name, item in calculations["fixture"].items()},
        "robustness": {f["name"]: {"kind": f["kind"], "sha256": f["sha256"],
                                  "source_sha256": f.get("source", {}).get("source_sha256")}
                       for f in robustness["fixtures"]},
        "spatial": {f["name"]: {name: f[name] for name in ("image_sha256", "mask_sha256")}
                    for f in spatial["fixtures"]},
    }


def assess(args):
    from bench import env

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; existing evidence is never overwritten")
    previous = load_baseline(args.baseline) if args.baseline else None
    if previous is not None and (
        previous["platform"] != env.native_platform_key() or previous["scope"] != args.scope
        or set(previous["requested_adapters"]) != set(args.profiles)
    ):
        raise ValueError("Baseline platform, scope and requested adapter set must match")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    report = {
        "schema_version": 1, "kind": "untimed_release_qualification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope, "requested_adapters": args.profiles,
        "platform": env.native_platform_key(), "timing_observations": False,
        "final_timing": "paused", "complete": False, "status": "running",
        "actual_publication_host_qualified": False,
        "limitations": LIMITATIONS,
        "controller_executable": sys.executable, "watchdog_seconds": args.step_timeout,
        "real_input_crops_requested": args.workspace_root is not None,
        "source_sha256": source_identity(), "steps": [], "errors": [],
        "baseline": {"path": str(args.baseline.resolve()), "sha256": sha256(args.baseline)} if args.baseline else None,
    }
    target = output / "release-qualification.json"
    write_json(target, report)
    try:
        report["environments"] = environment_snapshot()
        report["stable_identity"] = stable_identity(report["environments"], report["source_sha256"])
        plan = build_plan(output, args.profiles, scope=args.scope, bootstrap=args.bootstrap_inputs,
                          workspace=args.workspace_root)
        states = {}
        child_env = dict(os.environ, PYTHONPATH=str(ROOT), PICTOLOGICS_DISABLE_WARMUP="1")
        for step in plan:
            record = dict(step)
            if any(states.get(name) != "completed" for name in step["requires"]):
                record["status"] = "blocked_by_failed_dependency"
            else:
                print(json.dumps({"step": step["name"], "status": "running"}), flush=True)
                record = run_step(step, output, child_env, args.step_timeout)
            states[step["name"]] = record["status"]
            report["steps"].append(record)
            write_json(target, report)
            print(json.dumps({"step": step["name"], "status": record["status"]}), flush=True)
        if any(value != "completed" for value in states.values()):
            raise ValueError("One or more untimed stages did not complete; see per-stage logs")
        report["smoke"] = inspect_smoke(output, args.profiles)
        report["feature_inventory"] = inspect_calculations(output, args.profiles, report["environments"])
        report["spatial"] = inspect_spatial(output, report["environments"])
        report["robustness"] = inspect_robustness(output, args.profiles, report["environments"])
        report["spatial_policy"] = inspect_policy(output, report["environments"])
        report["diagnostic_fixture_identity"] = fixture_identity(output, report["robustness"], report["spatial"])
        if args.scope == "full":
            report["compliance"] = inspect_compliance(output, args.profiles, report["environments"])
        if previous is not None:
            report["baseline_comparison"] = compare_baseline(previous, report)
        if source_identity() != report["source_sha256"]:
            raise ValueError("Source changed during assessment; rerun in a fresh directory")
        report["environments_after"] = environment_snapshot()
        if report["environments_after"] != report["environments"]:
            raise ValueError("Adapter environments changed during assessment")
        if args.baseline and sha256(args.baseline) != report["baseline"]["sha256"]:
            raise ValueError("Baseline snapshot changed during assessment")
        report["complete"] = True
        comparison = report.get("baseline_comparison", {})
        findings = bool(comparison.get("feature_changes")
                        or comparison.get("compliance_changes")
                        or comparison.get("comparison_is_matched") is False
                        or report["robustness"].get("status") == "needs_review"
                        or report["spatial_policy"]["acceptance"]["status"] != "passed")
        if args.scope == "full":
            compliance = report["compliance"]
            findings |= bool(compliance["ibsi1"]["numerical_findings"]
                             or compliance["phase2"]["numerical_findings"] or compliance["phase1"]["failures"])
        report["status"] = ("needs_review" if findings else
                            "qualified_with_coverage_limits" if args.scope == "full" else "core_checks_only")
    except (Exception, KeyboardInterrupt):
        report["complete"] = False
        report["status"] = "incomplete_or_failed"
        report["errors"].append(traceback.format_exc())
    try:
        report["artifacts_sha256"] = artifact_identity(output)
    except (OSError, ValueError):
        report.update(complete=False, status="incomplete_or_failed")
        report["errors"].append(traceback.format_exc())
    report["report_fingerprint"] = report_fingerprint(report)
    write_json(target, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=ADAPTERS, default=list(ADAPTERS))
    parser.add_argument("--scope", choices=("core", "full"), default="full")
    parser.add_argument("--bootstrap-inputs", action="store_true", help="Fetch only pinned IBSI 1/2 inputs; otherwise require existing inputs")
    parser.add_argument("--workspace-root", type=Path, help="Add bounded real CT/MRI/PET crops to robustness checks")
    parser.add_argument("--baseline", type=Path, help="Prior same-platform release-qualification.json for stable freeze and feature comparison")
    parser.add_argument("--step-timeout", type=float, default=3600.0, help="Safety watchdog only; no timing observations")
    parser.add_argument("--plan", action="store_true", help="Print commands without creating outputs or launching workers")
    args = parser.parse_args(argv)
    if len(args.profiles) != len(set(args.profiles)):
        parser.error("Duplicate profiles are not allowed")
    if not math.isfinite(args.step_timeout) or args.step_timeout <= 0:
        parser.error("--step-timeout must be finite and positive")
    if args.plan:
        print(json.dumps(build_plan(args.output_dir.resolve(), args.profiles, scope=args.scope,
                                    bootstrap=args.bootstrap_inputs, workspace=args.workspace_root), indent=2))
        return 0
    try:
        report = assess(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "incomplete_or_failed", "complete": False, "error": str(error)}))
        return 1
    print(json.dumps({"status": report["status"], "complete": report["complete"],
                      "report": str(args.output_dir.resolve() / "release-qualification.json")}), flush=True)
    return 1 if not report["complete"] else 2 if report["status"] == "needs_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import pytest

from scripts import qualify_adapter_robustness as robustness


@pytest.fixture
def fixtures(tmp_path):
    return robustness.create_fixtures(tmp_path / "fixtures")


def stub_worker(monkeypatch, *, broken=None, failure=None):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.7.1")

    def prepare(paths, _families, workload, _discrete):
        fixture = Path(paths["raw"]).parent.name
        if fixture == broken:
            raise failure
        if fixture == "empty_roi":
            values = {}
        elif fixture in {"constant_roi", "single_voxel_roi"} and workload == "spatial_autocorrelation":
            values = {"a": float("nan")}
        else:
            values = {"a": 1.0}
        return values, lambda: values.copy()

    monkeypatch.setattr(robustness.calculations, "_pictologics", prepare)


def worker_result(tmp_path, monkeypatch, fixtures):
    stub_worker(monkeypatch)
    output = tmp_path / "worker.json"
    assert robustness.worker("pictologics", fixtures, output, "test-run") == 0
    return json.loads(output.read_text())


def validate_worker(result, fixtures):
    robustness.validate_worker_report(
        result, "pictologics", fixtures, expected_version="0.7.1",
        worker_executable=robustness.sys.executable,
        report_id="test-run", source_sha256=robustness._source_hashes(),
    )


def test_fixture_inventory_exercises_geometry_intensity_and_degenerate_rois(fixtures):
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label

    assert robustness.validate_fixtures(fixtures) == robustness._digest(fixtures)
    by_name = {fixture["name"]: fixture for fixture in fixtures}
    assert set(by_name) == set(robustness.SYNTHETIC_FIXTURES)
    assert all(fixture["source"] == {"kind": "synthetic", "clinical_image": False} for fixture in fixtures)
    for name in ("synthetic_ct", "synthetic_mri", "synthetic_pet"):
        fixture = by_name[name]
        assert fixture["kind"] == "normal" and fixture["roi_voxels"] > 32
        assert len(set(nib.load(fixture["paths"]["raw"]).header.get_zooms())) == 3
    assert by_name["synthetic_ct"]["roi_intensity_range"][0] < 0
    assert 0 < by_name["synthetic_pet"]["roi_intensity_range"][0] < 15
    oblique = np.asarray(by_name["anisotropic_oblique"]["affine"])
    assert abs(oblique[0, 1]) > 0
    assert label(nib.load(by_name["disconnected_roi"]["paths"]["mask"]).get_fdata())[1] == 2
    assert nib.load(by_name["image_edge_roi"]["paths"]["mask"]).get_fdata()[0].any()
    assert by_name["small_roi"]["roi_voxels"] == 8
    assert by_name["single_voxel_roi"]["roi_voxels"] == 1
    assert by_name["empty_roi"]["roi_voxels"] == 0
    assert by_name["constant_roi"]["roi_intensity_range"] == [5.0, 5.0]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "kind", "roi_voxels", "hash", "levels"])
def test_fixture_validation_rejects_inventory_metadata_and_content_drift(fixtures, mutation):
    changed = deepcopy(fixtures)
    if mutation == "missing":
        changed.pop()
    elif mutation == "duplicate":
        changed.append(changed[0])
    elif mutation == "kind":
        changed[0]["kind"] = "degenerate"
    elif mutation == "roi_voxels":
        changed[0]["roi_voxels"] += 1
    elif mutation == "hash":
        with Path(changed[0]["paths"]["raw"]).open("ab") as file:
            file.write(b"tampered")
    else:
        changed[0]["paths"]["levels"] = changed[0]["paths"]["raw"]
        changed[0]["sha256"]["levels"] = changed[0]["sha256"]["raw"]
    with pytest.raises(ValueError):
        robustness.validate_fixtures(changed)


def test_compare_outputs_preserves_undefined_classification_without_nonstandard_json():
    values = {"finite": 2.0, "nan": float("nan"), "pos": float("inf"), "neg": -float("inf")}
    result = robustness.compare_outputs(values, values)
    assert result["finite_values"] == {"finite": 2.0}
    assert result["undefined_values"] == {"nan": "nan", "pos": "positive_infinity", "neg": "negative_infinity"}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("left,right", [
    ({"a": 1}, {}), ({"a": 1}, {"a": 2}),
    ({"a": float("nan")}, {"a": float("inf")}),
    ({"a": 1}, {"a": True}), ({"a": 1}, {"a": "1"}),
    ({"a": 1}, {"a": [1]}),
])
def test_compare_outputs_fails_mismatches_even_for_degenerate_inputs(left, right):
    with pytest.raises(robustness.EquivalenceError):
        robustness.compare_outputs(left, right)


def test_worker_records_exact_inventory_and_separates_finite_empty_and_undefined(tmp_path, monkeypatch, fixtures):
    result = worker_result(tmp_path, monkeypatch, fixtures)
    validate_worker(result, fixtures)
    assert result["timing_observations"] is False
    assert len(result["checks"]) == 60
    assert result["outcome_counts"] == {
        "passed": 36, "characterized_empty": 6, "characterized_finite": 16, "characterized_undefined": 2,
    }


@pytest.mark.parametrize("error,expected", [
    (ValueError("input has no usable ROI"), "failed"),
    (ValueError("operands could not be broadcast together"), "failed"),
    (FloatingPointError("unexpected numerical operation"), "failed"),
    (ZeroDivisionError("unexpected divisor"), "failed"),
    (RuntimeError("feature is not one finite scalar"), "failed"),
    (RuntimeError("mask cache has inconsistent dimensions"), "failed"),
    (ImportError("missing dependency"), "failed"),
    (AttributeError("API changed"), "failed"),
    (RuntimeError("unexpected implementation failure"), "failed"),
    (robustness.EquivalenceError("value mismatch"), "failed"),
])
def test_worker_does_not_disguise_unexpected_errors_as_undefined(tmp_path, monkeypatch, fixtures, error, expected):
    stub_worker(monkeypatch, broken="constant_roi", failure=error)
    output = tmp_path / "worker.json"
    assert robustness.worker("pictologics", fixtures, output) == int(expected == "failed")
    result = json.loads(output.read_text())
    records = [check for check in result["checks"] if check["fixture"] == "constant_roi"]
    assert {check["status"] for check in records} == {expected}
    assert all(check["error_stage"] == "public_reference_and_preparation" for check in records)
    assert len(result["checks"]) == 60


@pytest.mark.parametrize("mutation", [None, "message", "module", "type", "missing_module", "workload", "stage", "finding"])
def test_observed_rejection_is_verified_again_when_loading_saved_evidence(tmp_path, monkeypatch, fixtures, mutation):
    stub_worker(monkeypatch)
    original = robustness.calculations._pictologics

    def prepare(paths, families, workload, discrete):
        if Path(paths["raw"]).parent.name == "constant_roi" and workload == "spatial_autocorrelation":
            raise RuntimeError("Pictologics feature morans_i_index_N365 is not one finite scalar")
        return original(paths, families, workload, discrete)

    monkeypatch.setattr(robustness.calculations, "_pictologics", prepare)
    output = tmp_path / "worker.json"
    assert robustness.worker("pictologics", fixtures, output, "test-run") == 0
    result = json.loads(output.read_text())
    check = next(check for check in result["checks"] if check["status"] == "characterized_rejection")
    assert check["equivalence_verified"] is False
    if mutation is None:
        validate_worker(result, fixtures)
        return
    if mutation == "message":
        check["error_message"] = "unexpected ROI implementation failure"
    elif mutation == "module":
        check["error_module"] = "other.module"
    elif mutation == "type":
        check["error_type"] = "ValueError"
    elif mutation == "missing_module":
        del check["error_module"]
    elif mutation == "stage":
        check["error_stage"] = "repeated_calculation"
    elif mutation == "finding":
        check["finding"] = "invented_explanation"
    else:
        # Swap the workload labels so the complete inventory still matches.
        other = next(row for row in result["checks"] if row["fixture"] == "constant_roi" and row["workload"] == "morphology")
        other["workload"], check["workload"] = check["workload"], other["workload"]
    with pytest.raises(ValueError):
        validate_worker(result, fixtures)


def test_worker_fails_repeat_error_after_successful_degenerate_calculation(tmp_path, monkeypatch, fixtures):
    stub_worker(monkeypatch)
    original = robustness.calculations._pictologics

    def prepare(paths, families, workload, discrete):
        if Path(paths["raw"]).parent.name != "small_roi":
            return original(paths, families, workload, discrete)
        values = iter(({"a": 1.0}, {"a": 2.0}))
        return {"a": 1.0}, lambda: next(values)

    monkeypatch.setattr(robustness.calculations, "_pictologics", prepare)
    output = tmp_path / "worker.json"
    assert robustness.worker("pictologics", fixtures, output) == 1
    records = json.loads(output.read_text())["checks"]
    assert all(check["status"] == "failed" for check in records if check["fixture"] == "small_roi")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign_run", "executable", "source", "empty_baseline", "false_pass", "false_undefined", "counts"])
def test_worker_validation_rejects_incomplete_foreign_or_contradictory_evidence(tmp_path, monkeypatch, fixtures, mutation):
    result = worker_result(tmp_path, monkeypatch, fixtures)
    if mutation == "missing":
        result["checks"].pop()
    elif mutation == "duplicate":
        result["checks"].append(result["checks"][0])
    elif mutation == "foreign_run":
        result["report_id"] = "old-run"
    elif mutation == "executable":
        result["executable"] = str(tmp_path / "other-python")
    elif mutation == "source":
        result["source_sha256"] = {}
    elif mutation == "empty_baseline":
        result["baseline_feature_names"]["morphology"] = []
    elif mutation == "false_pass":
        result["checks"][0]["status"] = "characterized_finite"
    elif mutation == "false_undefined":
        result["checks"][0]["undefined_values"] = {"a": "nan"}
    else:
        result["outcome_counts"]["passed"] += 1
    with pytest.raises(ValueError):
        validate_worker(result, fixtures)


def test_qualification_retains_venv_launcher_and_validates_saved_report(tmp_path, monkeypatch):
    from bench import env

    stub_worker(monkeypatch)
    runtime_repo = tmp_path / "runtime"
    profile_dir = runtime_repo / "configs/adapters"
    profile_dir.mkdir(parents=True)
    profile_dir.joinpath("pictologics.yaml").write_bytes(
        (robustness.ROOT / "configs/adapters/pictologics.yaml").read_bytes())
    output = tmp_path / "report"
    launcher = str(runtime_repo / ".venvs/adapters/pictologics/bin/python")
    calls = []
    profile = env.load_runtime_profiles()["pictologics"]
    lock, lock_path, locked_freeze = env._resolve_environment_lock(profile)
    resolved_base = "/Frameworks/Python.framework/python3.12"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if str(path) == launcher:
            return original_resolve(Path(resolved_base), *args, **kwargs)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            assert kwargs["cwd"] == runtime_repo
            return SimpleNamespace(stdout=json.dumps({
                "environment": {"profile": "pictologics", "distribution": profile.distribution,
                                "version": profile.version, "profile_fingerprint": env._profile_fingerprint(profile),
                                "distribution_metadata_version": "0.7.1", "python": platform.python_version(),
                                "python_executable": resolved_base, "freeze": locked_freeze,
                                "freeze_sha256": lock.freeze_sha256,
                                "environment_lock": {"platform_key": lock.platform_key,
                                                     "path": lock_path.relative_to(robustness.ROOT).as_posix(),
                                                     "sha256": lock.sha256, "freeze_sha256": lock.freeze_sha256}},
                "worker_executable": launcher,
            }), returncode=0)
        assert command[0] == launcher
        fixtures_path = Path(command[command.index("--fixtures") + 1])
        run_id = command[command.index("--report-id") + 1]
        # Mock only executable identity to model the native worker's environment.
        with monkeypatch.context() as patch:
            patch.setattr(robustness.sys, "executable", launcher)
            code = robustness.worker("pictologics", json.loads(fixtures_path.read_text()),
                                     output / "pictologics.json", run_id)
        return SimpleNamespace(stdout="", stderr="", returncode=code)

    monkeypatch.setattr(robustness.subprocess, "run", run)
    report = robustness.qualify(["pictologics"], output, runtime_repo, 10)
    assert report["status"] == "qualified_with_degenerate_limits"
    assert robustness.validate_report(report, ["pictologics"])
    assert len(calls) == 2
    changed = deepcopy(report)
    changed["adapters"] = []
    with pytest.raises(ValueError, match="adapter inventory"):
        robustness.validate_report(changed, ["pictologics"])
    with pytest.raises(ValueError, match="fresh"):
        robustness.qualify(["pictologics"], output, runtime_repo, 10)


@pytest.mark.parametrize("profiles,timeout", [([], 1), (["pictologics"] * 2, 1), (["unknown"], 1), (["pictologics"], float("nan"))])
def test_qualification_rejects_invalid_scope_before_creating_output(tmp_path, profiles, timeout):
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        robustness.qualify(profiles, output, timeout=timeout)
    assert not output.exists()


def test_finite_constant_spatial_is_review_finding_not_accuracy_pass(tmp_path, monkeypatch, fixtures):
    stub_worker(monkeypatch)
    original = robustness.calculations._pictologics

    def prepare(paths, families, workload, discrete):
        if Path(paths["raw"]).parent.name == "constant_roi" and workload == "spatial_autocorrelation":
            return {"a": 1.0}, lambda: {"a": 1.0}
        return original(paths, families, workload, discrete)

    monkeypatch.setattr(robustness.calculations, "_pictologics", prepare)
    output = tmp_path / "worker.json"
    assert robustness.worker("pictologics", fixtures, output, "test-run") == 0
    result = json.loads(output.read_text())
    validate_worker(result, fixtures)
    assert result["status"] == "needs_review"
    assert result["review_findings"] == [{"fixture": "constant_roi", "workload": "spatial_autocorrelation",
                                          "finding": "finite_undefined_spatial"}]
    record = next(check for check in result["checks"] if check["status"] == "needs_review")
    assert record["equivalence_verified"] and record["finite_values"] == {"a": 1.0}
    changed = deepcopy(result)
    changed["status"] = "qualified_with_degenerate_limits"
    with pytest.raises(ValueError, match="review findings"):
        validate_worker(changed, fixtures)


@pytest.mark.parametrize("adapter,fixture,workload,error_type,module,message,expected", [
    ("zrad", "small_roi", "texture", "DataStructureError", "zrad.exceptions",
     "3D bounding box dimension < 3.", ("characterized_rejection", None)),
    ("zrad", "constant_roi", "spatial_autocorrelation", "DataStructureError", "zrad.exceptions",
     "There determinator is zero in Moran I.", ("characterized_rejection", None)),
    ("zrad", "constant_roi", "texture", "DataStructureError", "other.module",
     "Not enough bins to calculate gradient.", None),
    ("zrad", "synthetic_ct", "texture", "DataStructureError", "zrad.exceptions",
     "3D bounding box dimension < 3.", None),
    ("mirp", "empty_roi", "texture", "RuntimeError", "builtins",
     "MIRP native loading expected exactly one image", ("characterized_rejection", None)),
    ("mirp", "synthetic_ct", "texture", "RuntimeError", "builtins",
     "MIRP native loading expected exactly one image", None),
    ("mirp", "empty_roi", "texture", "RuntimeError", "other.module",
     "MIRP native loading expected exactly one image", None),
    ("pyradiomics", "single_voxel_roi", "ivh", "ValueError", "builtins",
     "mask only contains 1 segmented voxel! Cannot extract features for a single voxel.", None),
    ("pictologics", "constant_roi", "morphology", "RuntimeError", "builtins",
     "Pictologics feature morans_i_index_N365 is not one finite scalar", None),
    ("pictologics", "synthetic_ct", "spatial_autocorrelation", "RuntimeError", "builtins",
     "Pictologics feature morans_i_index_N365 is not one finite scalar", None),
    ("medimage", "constant_roi", "texture", "RuntimeError", "builtins",
     "MEDimage failed supported feature family histogram: index 1 is out of bounds for axis 0 with size 1",
     ("needs_review", "native_degenerate_error")),
    ("medimage", "constant_roi", "texture", "RuntimeError", "builtins",
     "MEDimage failed supported feature family histogram: unexpected internal error", None),
    ("medimage", "constant_roi", "texture", "RuntimeError", "other.module",
     "MEDimage failed supported feature family histogram: index 1 is out of bounds for axis 0 with size 1", None),
])
def test_native_input_limits_are_scoped(adapter, fixture, workload, error_type, module, message, expected):
    assert robustness._native_error_outcome(adapter, fixture, workload, error_type, module, message) == expected


def test_medimage_native_degenerate_bug_is_review_findings_not_rejection(tmp_path, monkeypatch, fixtures):
    stub_worker(monkeypatch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.9.8")
    original = robustness.calculations._pictologics

    def prepare(paths, families, workload, discrete):
        if Path(paths["raw"]).parent.name == "constant_roi" and workload == "texture":
            raise RuntimeError("MEDimage failed supported feature family histogram: index 1 is out of bounds for axis 0 with size 1")
        return original(paths, families, workload, discrete)

    monkeypatch.setattr(robustness.calculations, "_medimage", prepare)
    output = tmp_path / "worker.json"
    assert robustness.worker("medimage", fixtures, output, "test-run") == 0
    result = json.loads(output.read_text())
    robustness.validate_worker_report(result, "medimage", fixtures, expected_version="0.9.8",
                                      worker_executable=robustness.sys.executable, report_id="test-run",
                                      source_sha256=robustness._source_hashes())
    assert result["status"] == "needs_review"
    review = next(check for check in result["checks"] if check["status"] == "needs_review")
    assert review["equivalence_verified"] is False
    assert review["finding"] == "native_degenerate_error"
    review["error_message"] = "unrelated implementation error"
    with pytest.raises(ValueError, match="native error finding"):
        robustness.validate_worker_report(result, "medimage", fixtures, expected_version="0.9.8",
                                          worker_executable=robustness.sys.executable, report_id="test-run",
                                          source_sha256=robustness._source_hashes())


def test_real_crops_preserve_world_geometry_and_attest_source_hashes(tmp_path):
    import nibabel as nib
    import numpy as np

    data = tmp_path / "workspace/ibsi2_phase3"
    data.mkdir(parents=True)
    cases = []
    raw = np.arange(15 ** 3, dtype=np.float32).reshape((15,) * 3)
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[3:12, 3:12, 3:12] = 1
    affine = np.array([[0, -2, 0, 19], [1, 0, 0, -11], [0, 0, 3, 7], [0, 0, 0, 1]])
    for modality in ("ct", "mri", "pet"):
        image_path, mask_path = data / f"{modality}.nii.gz", data / f"{modality}-mask.nii.gz"
        nib.save(nib.Nifti1Image(raw, affine), image_path)
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        cases.append({"case_id": modality, "modality": modality, "image_path": image_path.name,
                      "mask_path": mask_path.name, "image_sha256": robustness._sha256(image_path),
                      "mask_sha256": robustness._sha256(mask_path)})
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}))
    fixtures = robustness.create_fixtures(tmp_path / "fixtures", data.parent)
    robustness.validate_fixtures(fixtures, include_real=True)
    real = [fixture for fixture in fixtures if fixture["name"] in robustness.REAL_FIXTURES]
    assert len(real) == 3
    for fixture in real:
        source = fixture["source"]
        assert source["clinical_provenance_verified"] is False and source["whole_case_validated"] is False
        assert source["crop_selection"] == "roi_boundary_voxel_nearest_centroid"
        starts = np.array(source["crop_start_xyz"])
        assert np.allclose(np.array(fixture["affine"])[:3, 3], (affine @ np.r_[starts, 1])[:3])
        assert fixture["shape"] == [9, 9, 9]
        assert 32 <= fixture["roi_voxels"] < 9 ** 3
        crop = nib.load(fixture["paths"]["raw"]).get_fdata()
        assert np.array_equal(crop, raw[tuple(slice(start, start + 9) for start in starts)])
    cases[0]["image_sha256"] = "incorrect"
    manifest.write_text(json.dumps({"cases": cases}))
    with pytest.raises(ValueError, match="source hashes differ"):
        robustness.create_real_crops(data.parent, tmp_path / "tampered-crops")

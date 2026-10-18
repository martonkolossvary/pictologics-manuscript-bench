"""Exercise bounded geometry and intensity edge cases without adapter timings.

Normal cases require complete public/prepared/repeated feature equivalence.
Degenerate cases characterize rejections or undefined values separately; they
are not counted as successful finite-feature calculations. Known native defects
and finite results where the formula is undefined remain needs_review findings.
Exit zero means the diagnostic completed, not that every input is supported.
No benchmark timing,
calibration or ledger API is invoked. --timeout is a worker safety watchdog.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

from scripts import qualify_adapter_calculations as calculations

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = calculations.ADAPTERS
SYNTHETIC_FIXTURES = (
    "synthetic_ct", "anisotropic_oblique", "disconnected_roi", "image_edge_roi",
    "synthetic_mri", "synthetic_pet", "small_roi", "constant_roi", "single_voxel_roi", "empty_roi",
)
REAL_FIXTURES = ("real_ct_crop", "real_mri_crop", "real_pet_crop")
DEGENERATE_FIXTURES = frozenset({"small_roi", "constant_roi", "single_voxel_roi", "empty_roi"})


class EquivalenceError(ValueError):
    """A broken comparison is never an acceptable degenerate-input rejection."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_hashes():
    paths = [Path(__file__), Path(calculations.__file__), ROOT / "bench/benchmark_workloads.py",
             ROOT / "bench/env.py", *sorted((ROOT / "bench/adapters").glob("*.py"))]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _levels(raw, mask):
    """Shared FBN32 diagnostic input, separate from manuscript IVH preparation."""
    import numpy as np

    output = np.ones(raw.shape, dtype=np.int16)
    values = np.asarray(raw, dtype=np.float64)[mask > 0]
    if values.size and values.max() > values.min():
        output[mask > 0] = np.minimum(
            32, 1 + np.floor(32 * (values - values.min()) / (values.max() - values.min()))
        ).astype(np.int16)
    return output


def _save_fixture(output_dir, name, raw, mask, affine, *, kind="normal", source=None):
    import nibabel as nib
    import numpy as np

    raw = np.asarray(raw, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)
    target = output_dir / name
    target.mkdir(parents=True)
    paths = {}
    for key, array in (("raw", raw.astype(np.float32)), ("mask", mask.astype(np.uint8)),
                       ("levels", _levels(raw, mask))):
        path = target / f"{key}.nii.gz"
        nib.save(nib.Nifti1Image(array, affine), path)
        paths[key] = str(path.resolve())
    roi_values = raw[mask > 0]
    return {"name": name, "kind": kind, "paths": paths,
            "sha256": {key: _sha256(path) for key, path in paths.items()},
            "shape": list(raw.shape), "roi_voxels": int(np.count_nonzero(mask)),
            "affine": nib.load(paths["raw"]).affine.tolist(),
            "roi_intensity_range": [float(roi_values.min()), float(roi_values.max())] if roi_values.size else None,
            "source": source or {"kind": "synthetic", "clinical_image": False}}


def create_fixtures(output_dir, workspace_root=None):
    import numpy as np

    output_dir = Path(output_dir)
    shape = (12, 11, 10)
    rng = np.random.default_rng(20261013)
    x, y, z = np.indices(shape)
    mask = (((x - 5.2) / 3.8) ** 2 + ((y - 5.0) / 3.5) ** 2 +
            ((z - 4.4) / 3.1) ** 2 < 1).astype(np.uint8)
    raw = rng.uniform(-180, 220, shape).astype(np.float32)
    affine = np.diag([1.2, 1.3, 1.5, 1.0])
    affine[:3, 3] = [17.5, -25.0, 8.0]
    fixtures = [_save_fixture(output_dir, "synthetic_ct", raw, mask, affine)]
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    oblique = affine.copy()
    oblique[:3, :3] = rotation @ affine[:3, :3]
    fixtures.append(_save_fixture(output_dir, "anisotropic_oblique", raw, mask, oblique))
    disconnected = np.zeros(shape, dtype=np.uint8)
    disconnected[1:5, 1:5, 1:5] = 1
    disconnected[7:11, 6:10, 5:9] = 1
    fixtures.append(_save_fixture(output_dir, "disconnected_roi", raw, disconnected, affine))
    edge = np.zeros(shape, dtype=np.uint8)
    edge[:5, 2:8, 2:8] = 1
    fixtures.append(_save_fixture(output_dir, "image_edge_roi", raw, edge, affine))
    fixtures.append(_save_fixture(output_dir, "synthetic_mri", rng.uniform(10, 1500, shape), mask, affine))
    fixtures.append(_save_fixture(output_dir, "synthetic_pet", rng.uniform(0.01, 15, shape), mask, affine))
    small = np.zeros(shape, dtype=np.uint8)
    small[4:6, 4:6, 4:6] = 1
    fixtures.append(_save_fixture(output_dir, "small_roi", raw, small, affine, kind="degenerate"))
    fixtures.append(_save_fixture(output_dir, "constant_roi", np.full(shape, 5.0), mask,
                                  affine, kind="degenerate"))
    single = np.zeros(shape, dtype=np.uint8)
    single[5, 5, 5] = 1
    fixtures.append(_save_fixture(output_dir, "single_voxel_roi", raw, single, affine, kind="degenerate"))
    fixtures.append(_save_fixture(output_dir, "empty_roi", raw, np.zeros_like(mask),
                                  affine, kind="degenerate"))
    if workspace_root is not None:
        fixtures.extend(create_real_crops(Path(workspace_root), output_dir))
    return fixtures


def create_real_crops(workspace_root, output_dir):
    """One deterministic 9^3 CT/MRI/PET crop; never assert whole-case validation."""
    import nibabel as nib
    import numpy as np

    manifest_path = workspace_root / "ibsi2_phase3" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = []
    for modality in ("ct", "mri", "pet"):
        cases = sorted((case for case in manifest["cases"] if case["modality"] == modality),
                       key=lambda case: case["case_id"])
        if not cases:
            raise ValueError(f"workspace has no {modality} cases")
        case = cases[0]
        image_path = (manifest_path.parent / case["image_path"]).resolve()
        mask_path = (manifest_path.parent / case["mask_path"]).resolve()
        if not all(path.is_relative_to(manifest_path.parent.resolve()) for path in (image_path, mask_path)):
            raise ValueError("manifest input paths must stay inside the dataset directory")
        hashes = {"image": _sha256(image_path), "mask": _sha256(mask_path)}
        if hashes != {"image": case["image_sha256"], "mask": case["mask_sha256"]}:
            raise ValueError(f"source hashes differ for {case['case_id']}")
        image, mask_image = nib.load(image_path), nib.load(mask_path)
        if image.shape != mask_image.shape or not np.allclose(image.affine, mask_image.affine):
            raise ValueError(f"source geometry differs for {case['case_id']}")
        raw, mask = image.get_fdata(dtype=np.float32), mask_image.get_fdata() > 0
        coordinates = np.argwhere(mask)
        if not len(coordinates):
            raise ValueError(f"empty source ROI for {case['case_id']}")
        # Choose the actual ROI voxel nearest the ROI centroid, not a possibly
        # empty centre of the bounding box of a disconnected mask.
        centroid = coordinates.mean(axis=0)
        centre = coordinates[np.argmin(np.sum((coordinates - centroid) ** 2, axis=1))]

        def crop_at(point):
            starts = np.maximum(0, np.minimum(point - 4, np.asarray(raw.shape) - 9))
            stops = np.minimum(starts + 9, raw.shape)
            slices = tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))
            return starts, stops, raw[slices], mask[slices]

        def suitable(crop_raw, crop_mask):
            # PyRadiomics requires a background label. Keep actual source
            # pixels and mask values; never manufacture padding or a new ROI.
            return 32 <= np.count_nonzero(crop_mask) < crop_mask.size and np.isfinite(crop_raw).all()

        starts, stops, crop_raw, crop_mask = crop_at(centre)
        selection = "roi_voxel_nearest_centroid"
        if not suitable(crop_raw, crop_mask):
            from scipy.ndimage import binary_erosion

            boundary = np.argwhere(mask & ~binary_erosion(mask, border_value=1))
            order = np.argsort(np.sum((boundary - centroid) ** 2, axis=1), kind="stable")
            for index in order:
                starts, stops, crop_raw, crop_mask = crop_at(boundary[index])
                if suitable(crop_raw, crop_mask):
                    selection = "roi_boundary_voxel_nearest_centroid"
                    break
            else:
                raise ValueError(f"no suitable source crop with ROI and background for {case['case_id']}")
        affine = image.affine.copy()
        affine[:3, 3] = (image.affine @ np.r_[starts, 1])[:3]
        fixtures.append(_save_fixture(
            output_dir, f"real_{modality}_crop", crop_raw, crop_mask, affine,
            source={"kind": "manifest_input_crop", "clinical_provenance_verified": False,
                    "case_id": case["case_id"], "modality": modality,
                    "manifest": str(manifest_path.resolve()), "manifest_sha256": _sha256(manifest_path),
                    "image": str(image_path.resolve()), "mask": str(mask_path.resolve()),
                    "source_sha256": hashes, "crop_start_xyz": starts.tolist(),
                    "crop_stop_xyz": stops.tolist(), "crop_selection": selection,
                    "whole_case_validated": False}))
    return fixtures


def compare_outputs(reference, observed):
    """Compare non-finite classifications explicitly; JSON never contains NaN."""
    if set(reference) != set(observed):
        raise EquivalenceError("public/prepared/repeat feature names differ")
    finite, undefined = {}, {}
    for name in sorted(reference):
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in (reference[name], observed[name])):
            raise EquivalenceError(f"non-numeric feature: {name}")
        try:
            a, b = float(reference[name]), float(observed[name])
        except (TypeError, ValueError, OverflowError) as error:
            raise EquivalenceError(f"non-scalar feature: {name}") from error
        if math.isfinite(a) and math.isfinite(b):
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                raise EquivalenceError(f"value mismatch: {name}: {a} versus {b}")
            finite[name] = b
        elif math.isnan(a) and math.isnan(b):
            undefined[name] = "nan"
        elif a == b:
            undefined[name] = "positive_infinity" if a > 0 else "negative_infinity"
        else:
            raise EquivalenceError(f"non-finite classification differs: {name}")
    return {"finite_values": finite, "undefined_values": undefined}


def validate_fixtures(fixtures, *, include_real=False, inspect_images=True):
    """Require the declared inventory and re-read each saved input's identity."""
    expected = set(SYNTHETIC_FIXTURES) | (set(REAL_FIXTURES) if include_real else set())
    names = [fixture["name"] for fixture in fixtures]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("fixture inventory differs from required robustness suite")
    for fixture in fixtures:
        expected_kind = "degenerate" if fixture["name"] in DEGENERATE_FIXTURES else "normal"
        if fixture["kind"] != expected_kind or set(fixture["paths"]) != {"raw", "mask", "levels"}:
            raise ValueError(f"fixture metadata differs: {fixture['name']}")
        if set(fixture["sha256"]) != set(fixture["paths"]):
            raise ValueError("fixture hash inventory differs")
        for key, path in fixture["paths"].items():
            if _sha256(path) != fixture["sha256"][key]:
                raise ValueError(f"fixture hash differs: {fixture['name']}/{key}")
        if not inspect_images:
            continue
        # The controller has nibabel. Some locked adapter environments do not;
        # workers check bytes against this controller-validated inventory.
        import nibabel as nib
        import numpy as np

        images = {}
        for key, path in fixture["paths"].items():
            image = nib.load(path)
            if list(image.shape) != fixture["shape"] or len(image.shape) != 3 or not np.allclose(
                image.affine, fixture["affine"], rtol=0, atol=1e-6
            ):
                raise ValueError("fixture geometry differs")
            images[key] = image.get_fdata()
            if not np.isfinite(images[key]).all():
                raise ValueError("fixture input must be finite")
        mask = images["mask"] > 0
        if not np.isin(images["mask"], [0, 1]).all() or int(mask.sum()) != fixture["roi_voxels"]:
            raise ValueError("fixture ROI metadata differs")
        values = images["raw"][mask]
        intensity_range = [float(values.min()), float(values.max())] if values.size else None
        if intensity_range != fixture["roi_intensity_range"]:
            raise ValueError("fixture intensity metadata differs")
        if not np.array_equal(images["levels"], _levels(images["raw"], images["mask"])):
            raise ValueError("fixture diagnostic levels differ")
    return _digest(fixtures)


def _is_input_rejection(error):
    """Observe domain errors, but fail programming/import/IO errors explicitly."""
    if isinstance(error, EquivalenceError):
        return False
    if isinstance(error, (ValueError, FloatingPointError, ZeroDivisionError)):
        return True
    return isinstance(error, RuntimeError) and any(token in str(error).lower() for token in (
        "finite scalar", "non-finite", "zero features", "empty", "roi", "mask", "singular", "qhull",
    ))


def _invalid_empty_features(finite):
    # Zero voxel-counting volume is well-defined even when no ROI exists.
    # Do not generalize this exception to other finite empty-ROI features.
    return any(name != "volume_voxel_counting_YEKZ" or value != 0.0 for name, value in finite.items())


def _native_error_outcome(adapter, fixture, workload, error_type, error_module, message):
    """Recognize only observed, scoped native input limits, never arbitrary bugs."""
    if adapter == "zrad" and error_type == "DataStructureError" and error_module == "zrad.exceptions":
        expected = set()
        if fixture in {"small_roi", "single_voxel_roi"}:
            expected.add("3D bounding box dimension < 3.")
        if fixture == "empty_roi":
            expected.add("No valid voxels in 3D array.")
        if fixture == "constant_roi":
            expected.update({"spatial_autocorrelation": {"There determinator is zero in Moran I."},
                             "texture": {"Not enough bins to calculate gradient."},
                             "ivh": {"Intensity range is zero."}}.get(workload, set()))
        if message in expected:
            return "characterized_rejection", None
    if (adapter == "mirp" and fixture == "empty_roi" and error_type == "RuntimeError" and
            message == "MIRP native loading expected exactly one image"):
        return "characterized_rejection", None
    if adapter == "medimage" and error_type == "RuntimeError":
        histogram_error = (fixture in {"constant_roi", "single_voxel_roi"} and workload == "texture" and
                           message == "MEDimage failed supported feature family histogram: index 1 is out of bounds for axis 0 with size 1")
        covariance_error = (fixture == "single_voxel_roi" and workload == "morphology" and
                            message == "MEDimage failed supported feature family morphology: 0-dimensional array given. Array must be at least two-dimensional")
        if histogram_error or covariance_error:
            return "needs_review", "native_degenerate_error"
    return None


def _finite_finding(fixture, workload, finite):
    if fixture == "empty_roi" and _invalid_empty_features(finite):
        return "finite_empty_roi"
    if (fixture in {"constant_roi", "single_voxel_roi", "empty_roi"} and
            workload == "spatial_autocorrelation" and finite):
        return "finite_undefined_spatial"
    return None


def _status(checks):
    if any(check["status"] == "failed" for check in checks):
        return "failed"
    if any(check["status"] == "needs_review" for check in checks):
        return "needs_review"
    return "qualified_with_degenerate_limits"


def _review_findings(checks):
    return [{key: check[key] for key in ("fixture", "workload", "finding")}
            for check in checks if check["status"] == "needs_review"]


def worker(adapter_name, fixtures, output, report_id=None):
    from importlib.metadata import version
    import platform
    from bench.adapters.registry import get_adapter
    from bench.benchmark_workloads import WORKLOADS

    distribution = {"medimage": "medimage-pkg", "mirp": "mirp", "pyradiomics": "pyradiomics",
                    "zrad": "z-rad", "pictologics": "pictologics"}[adapter_name]
    fixtures_sha256 = validate_fixtures(fixtures, include_real=any(f['name'] in REAL_FIXTURES for f in fixtures),
                                      inspect_images=False)
    result = {"adapter": adapter_name, "distribution": distribution, "version": version(distribution),
              "python": platform.python_version(), "executable": sys.executable,
              "report_id": report_id, "fixtures_sha256": fixtures_sha256, "source_sha256": _source_hashes(),
              "timing_observations": False, "complete": False, "checks": []}
    capabilities = get_adapter(adapter_name)
    surfaces = {}
    for fixture in fixtures:
        for workload in WORKLOADS:
            record = {"fixture": fixture["name"], "kind": fixture["kind"], "workload": workload.name}
            if not capabilities.supports_workload(workload.name):
                record["status"] = "unsupported"
            else:
                stage = "public_reference_and_preparation"
                try:
                    families = [name for name in workload.families if capabilities.supports(name)]
                    reference, calculate = getattr(calculations, "_" + adapter_name)(
                        fixture["paths"], families, workload.name, workload.name in {"texture", "ivh"})
                    stage = "prepared_calculation"
                    observed = calculate()
                    stage = "public_equivalence"
                    values = compare_outputs(reference, observed)
                    stage = "repeated_calculation"
                    compare_outputs(observed, calculate())
                    record["equivalence_verified"] = True
                    stage = "outcome_classification"
                    record.update(values)
                    record["feature_names"] = sorted(observed)
                    record["omitted_from_baseline"] = sorted(surfaces.get(workload.name, set()) - set(observed))
                    record["added_to_baseline"] = sorted(set(observed) - surfaces.get(workload.name, set(observed)))
                    if fixture["kind"] == "normal":
                        if not observed or values["undefined_values"]:
                            raise ValueError("normal fixture must return non-empty, finite features")
                        if workload.name in surfaces and surfaces[workload.name] != set(observed):
                            raise ValueError("normal fixture changed complete baseline feature surface")
                        surfaces.setdefault(workload.name, set(observed))
                        record["status"] = "passed"
                    else:
                        finding = _finite_finding(fixture["name"], workload.name, values["finite_values"])
                        if finding:
                            record.update(status="needs_review", finding=finding,
                                          interpretation="Native public/prepared agreement is verified, but this finite result is not an accuracy pass for an undefined ROI.")
                        else:
                            record["status"] = ("characterized_empty" if not observed else
                                                "characterized_undefined" if values["undefined_values"] else
                                                "characterized_finite")
                except Exception as error:
                    # Adapter contracts may reject native NaNs before exporting
                    # them. Preserve that exception; never label it a finite pass.
                    native_outcome = _native_error_outcome(adapter_name, fixture["name"], workload.name,
                                                           type(error).__name__, type(error).__module__, str(error))
                    if fixture["kind"] == "degenerate" and stage in {
                        "public_reference_and_preparation", "prepared_calculation"
                    } and (native_outcome is not None or _is_input_rejection(error)):
                        record["status"], finding = native_outcome or ("characterized_rejection", None)
                        if finding:
                            record["finding"] = finding
                        record["equivalence_verified"] = False
                        record["interpretation"] = "Observed native error; mathematical undefinedness and public/prepared equivalence are not established. This is not a finite-feature pass."
                    else:
                        record["status"] = "failed"
                    record["error"] = traceback.format_exc()
                    record["error_type"] = type(error).__name__
                    record["error_module"] = type(error).__module__
                    record["error_message"] = str(error)
                    record["error_stage"] = stage
            result["checks"].append(record)
            _write_json(output, result)
    result["complete"] = True
    result["baseline_feature_names"] = {name: sorted(surface) for name, surface in surfaces.items()}
    result["outcome_counts"] = {status: sum(check["status"] == status for check in result["checks"])
                                for status in sorted({check["status"] for check in result["checks"]})}
    result["status"] = _status(result["checks"])
    result["review_findings"] = _review_findings(result["checks"])
    _write_json(output, result)
    return int(any(check["status"] == "failed" for check in result["checks"]))


def validate_worker_report(result, adapter_name, fixtures, *, expected_version, worker_executable,
                           report_id, source_sha256):
    """Fail closed on incomplete, stale, foreign, duplicate or contradictory checks."""
    from bench.adapters.registry import get_adapter
    from bench.benchmark_workloads import WORKLOADS

    capabilities = get_adapter(adapter_name)
    if (result.get("adapter") != adapter_name or result.get("distribution") != capabilities.distribution or
            result.get("version") != expected_version or result.get("report_id") != report_id or
            result.get("complete") is not True or result.get("timing_observations") is not False or
            result.get("fixtures_sha256") != _digest(fixtures) or result.get("source_sha256") != source_sha256 or
            Path(result.get("executable", "")).resolve() != Path(worker_executable).resolve()):
        raise ValueError("worker identity, completion or provenance differs")
    fixture_map = {fixture["name"]: fixture for fixture in fixtures}
    expected = {(fixture, workload.name) for fixture in fixture_map for workload in WORKLOADS}
    checks = result.get("checks", [])
    observed = [(check.get("fixture"), check.get("workload")) for check in checks]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("worker check inventory differs")
    baselines = result.get("baseline_feature_names", {})
    supported = {workload.name for workload in WORKLOADS if capabilities.supports_workload(workload.name)}
    if set(baselines) != supported or any(not names or len(names) != len(set(names)) for names in baselines.values()):
        raise ValueError("worker baseline feature inventory differs")
    counts = {}
    for check in checks:
        fixture = fixture_map[check["fixture"]]
        status = check["status"]
        counts[status] = counts.get(status, 0) + 1
        if check["kind"] != fixture["kind"]:
            raise ValueError("worker fixture kind differs")
        if not capabilities.supports_workload(check["workload"]):
            if status != "unsupported":
                raise ValueError("unsupported workload has a calculation outcome")
            continue
        if (status == "characterized_rejection" or status == "needs_review" and
                check.get("finding") == "native_degenerate_error") and fixture["kind"] == "degenerate":
            if (check.get("equivalence_verified") is not False or not check.get("error") or
                    not check.get("error_type") or check.get("error_stage") not in {
                        "public_reference_and_preparation", "prepared_calculation"}):
                raise ValueError("rejection evidence is incomplete")
            if status == "needs_review" and _native_error_outcome(
                adapter_name, check["fixture"], check["workload"], check.get("error_type"),
                check.get("error_module"), check.get("error_message")
            ) != ("needs_review", "native_degenerate_error"):
                raise ValueError("native error finding does not match observed input limit")
            continue
        if check.get("equivalence_verified") is not True:
            raise ValueError("public/prepared/repeat equivalence is missing")
        finite = check.get("finite_values", {})
        undefined = check.get("undefined_values", {})
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in finite.values()) or any(value not in {"nan", "positive_infinity", "negative_infinity"}
                                                    for value in undefined.values()):
            raise ValueError("invalid finite/undefined value classification")
        names = set(finite) | set(undefined)
        if (set(finite) & set(undefined) or check.get("feature_names") != sorted(names) or
                check.get("omitted_from_baseline") != sorted(set(baselines[check["workload"]]) - names) or
                check.get("added_to_baseline") != sorted(names - set(baselines[check["workload"]]))):
            raise ValueError("worker feature inventory differs")
        if fixture["kind"] == "normal":
            if status != "passed" or undefined or not finite or names != set(baselines[check["workload"]]):
                raise ValueError("normal fixture is not a complete finite pass")
        else:
            expected_status = ("characterized_empty" if not names else "characterized_undefined" if undefined
                               else "characterized_finite")
            finding = _finite_finding(fixture["name"], check["workload"], finite)
            if finding:
                expected_status = "needs_review"
                if check.get("finding") != finding:
                    raise ValueError("finite undefined-ROI finding is missing")
            if status != expected_status:
                raise ValueError("degenerate outcome classification differs")
    if result.get("outcome_counts") != counts:
        raise ValueError("worker outcome summary differs")
    if result.get("status") != _status(checks) or result.get("review_findings") != _review_findings(checks):
        raise ValueError("worker review findings or status differ")


def validate_report(report, profiles=None):
    """Revalidate saved successful evidence, including all input and source hashes.

    This does not reverify installed environments; callers requiring continued
    environment identity must compare fresh env.verify_profile snapshots.
    """
    from bench import env

    profiles = list(profiles if profiles is not None else report["requested_adapters"])
    if (not profiles or len(profiles) != len(set(profiles)) or set(profiles) - set(ADAPTERS) or
            report.get("requested_adapters") != profiles or report.get("schema_version") != 1 or
            report.get("kind") != "untimed_adapter_robustness" or not report.get("report_id") or
            report.get("complete") is not True or report.get("timing_observations") is not False or
            report.get("status") not in {"qualified_with_degenerate_limits", "needs_review"} or
            report.get("platform") != env.native_platform_key() or
            report.get("source_sha256") != _source_hashes()):
        raise ValueError("robustness report identity, completeness or source provenance differs")
    fixtures = report["fixtures"]
    digest = validate_fixtures(fixtures, include_real=report["real_input_crops_requested"])
    if report.get("fixtures_sha256") != digest:
        raise ValueError("report fixture manifest checksum differs")
    records = report.get("adapters", [])
    if [record.get("adapter") for record in records] != profiles:
        raise ValueError("robustness adapter inventory differs")
    profile_map = env.load_runtime_profiles()
    for record in records:
        name = record["adapter"]
        profile = profile_map[name]
        provenance = record["environment"]
        if (record["status"] not in {"qualified_with_degenerate_limits", "needs_review"} or
                record["status"] != record["qualification"].get("status") or provenance.get("profile") != name or
                provenance.get("distribution") != profile.distribution or
                provenance.get("version") != profile.version or
                provenance.get("profile_fingerprint") != env._profile_fingerprint(profile) or
                provenance.get("distribution_metadata_version") != (profile.metadata_version or profile.version) or
                record.get("profile_sha256") != _sha256(ROOT / "configs/adapters" / f"{name}.yaml") or
                record.get("worker_returncode") != 0 or
                provenance.get("python") != record["qualification"].get("python")):
            raise ValueError("robustness adapter runtime identity differs")
        lock_info = env._resolve_environment_lock(profile)
        if lock_info is None:
            raise ValueError("robustness qualification requires a checked-in native environment lock")
        lock, lock_path, locked_freeze = lock_info
        if (provenance.get("freeze") != locked_freeze or provenance.get("freeze_sha256") != lock.freeze_sha256 or
                provenance.get("environment_lock") != {
                    "platform_key": lock.platform_key, "path": lock_path.relative_to(ROOT).as_posix(),
                    "sha256": lock.sha256, "freeze_sha256": lock.freeze_sha256,
                } or Path(provenance.get("python_executable", "")).resolve() != Path(record["worker_executable"]).resolve()):
            raise ValueError("robustness environment freeze, lock or executable differs")
        validate_worker_report(record["qualification"], name, fixtures,
                               expected_version=profile.metadata_version or profile.version,
                               worker_executable=record["worker_executable"], report_id=report["report_id"],
                               source_sha256=report["source_sha256"])
    if report["status"] != _status(records):
        raise ValueError("robustness report status differs from adapter findings")
    return True


def qualify(profiles, output_dir, runtime_repo=ROOT, timeout=600.0, workspace_root=None):
    from bench import env

    output_dir, runtime_repo = Path(output_dir).resolve(), Path(runtime_repo).resolve()
    if output_dir.exists():
        raise ValueError(f"output directory must be fresh: {output_dir}")
    if not math.isfinite(timeout) or timeout <= 0 or not profiles or len(set(profiles)) != len(profiles) or set(profiles) - set(ADAPTERS):
        raise ValueError("positive watchdog and unique, non-empty profiles are required")
    output_dir.mkdir(parents=True)
    fixtures = create_fixtures(output_dir / "fixtures", workspace_root)
    fixtures_path = output_dir / "fixtures.json"
    _write_json(fixtures_path, fixtures)
    report = {"schema_version": 1, "kind": "untimed_adapter_robustness",
              "report_id": str(uuid.uuid4()), "fixtures_sha256": validate_fixtures(fixtures, include_real=workspace_root is not None),
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": env.native_platform_key(), "complete": False,
              "timing_observations": False, "watchdog_seconds": timeout,
              "requested_adapters": list(profiles), "fixtures": fixtures,
              "real_input_crops_requested": workspace_root is not None,
              "limitations": ["Bounded crops do not validate full clinical volumes.",
                              "Degenerate outcomes are characterized, not finite-feature passes.",
                              "Diagnostic IVH uses common FBN32 levels, not manuscript modality-specific IVH inputs.",
                              "Public/prepared agreement is not independent IBSI accuracy validation."],
              "source_sha256": _source_hashes(),
              "adapters": []}
    _write_json(output_dir / "robustness.json", report)
    for name in profiles:
        record = {"adapter": name}
        try:
            profile_path = Path("configs/adapters") / f"{name}.yaml"
            if _sha256(ROOT / profile_path) != _sha256(runtime_repo / profile_path):
                raise ValueError(f"runtime profile differs: {profile_path}")
            record["profile_sha256"] = _sha256(ROOT / profile_path)
            command = ("import json; from bench import env; "
                       f"profile = env.load_runtime_profiles()[{name!r}]; "
                       "print(json.dumps(dict(environment=env.verify_profile(profile, smoke=False), "
                       "worker_executable=str(env.env_python(env.env_dir_for_profile(profile))))))")
            verified = subprocess.run([sys.executable, "-c", command], cwd=runtime_repo,
                                      env=dict(os.environ, PYTHONPATH=str(runtime_repo)),
                                      text=True, capture_output=True, check=True, timeout=timeout)
            verified_runtime = json.loads(verified.stdout)
            record.update(verified_runtime)
            profile = env.load_runtime_profiles()[name]
            if record["environment"]["distribution_metadata_version"] != (profile.metadata_version or profile.version):
                raise ValueError("verified runtime version differs from current profile")
            child_env = dict(os.environ, PYTHONPATH=str(ROOT), PICTOLOGICS_DISABLE_WARMUP="1")
            for key in ("PICTOLOGICS_NUM_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
                child_env[key] = "1"
            # Launch the venv entry point, never provenance's symlink-resolved
            # base Python (which drops the virtual environment on macOS).
            completed = subprocess.run(
                [record["worker_executable"], "-m", "scripts.qualify_adapter_robustness", "--worker", name,
                 "--fixtures", str(fixtures_path), "--output-dir", str(output_dir), "--report-id", report["report_id"]],
                cwd=ROOT, env=child_env, text=True, capture_output=True, timeout=timeout)
            (output_dir / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output_dir / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            record["worker_returncode"] = completed.returncode
            if not (output_dir / f"{name}.json").is_file():
                raise RuntimeError(f"worker exited {completed.returncode} without a report: {completed.stderr[-4000:]}")
            record["qualification"] = json.loads((output_dir / f"{name}.json").read_text(encoding="utf-8"))
            validate_fixtures(fixtures, include_real=workspace_root is not None)
            validate_worker_report(record["qualification"], name, fixtures,
                                   expected_version=profile.metadata_version or profile.version,
                                   worker_executable=record["worker_executable"], report_id=report["report_id"],
                                   source_sha256=report["source_sha256"])
            record["status"] = record["qualification"]["status"] if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as error:
            for stream in ("stdout", "stderr"):
                value = getattr(error, stream) or ""
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                (output_dir / f"{name}.{stream}.log").write_text(value, encoding="utf-8")
            partial = output_dir / f"{name}.json"
            if partial.is_file():
                record["qualification"] = json.loads(partial.read_text(encoding="utf-8"))
            record.update(status="failed", error=traceback.format_exc(), watchdog_exceeded=True)
        except Exception:
            record.update(status="failed", error=traceback.format_exc())
        report["adapters"].append(record)
        report["complete"] = len(report["adapters"]) == len(profiles)
        report["status"] = _status(report["adapters"])
        if report["complete"] and report["status"] != "failed":
            try:
                validate_report(report, list(profiles))
            except Exception:
                report.update(status="failed", validation_error=traceback.format_exc())
        _write_json(output_dir / "robustness.json", report)
        print(json.dumps({"adapter": name, "status": record["status"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=ADAPTERS, default=list(ADAPTERS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, default=ROOT)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--worker", choices=ADAPTERS, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report-id", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, json.loads(args.fixtures.read_text(encoding="utf-8")),
                      args.output_dir / f"{args.worker}.json", args.report_id)
    report = qualify(args.profiles, args.output_dir, args.runtime_repo, args.timeout, args.workspace_root)
    return int(not report["complete"] or report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())

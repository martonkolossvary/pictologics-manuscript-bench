"""Untimed spatial-statistic checks against independent inverse-distance sums.

Small fixtures qualify definitions, geometry and the two Pictologics algorithms;
they do not establish large-image feasibility or measure performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SPATIAL_CODES = {"N365", "NPT7"}


def reference_statistics(data, mask, spacing):
    """Independent dense ordered-pair IBSI definition, for small fixtures only."""
    roi = np.asarray(mask) != 0
    values = np.asarray(data, dtype=np.float64)[roi]
    n = values.size
    if n < 2:
        return dict.fromkeys(SPATIAL_CODES, float("nan"))
    centered = values - values.mean()
    denominator = float(centered @ centered)
    if denominator == 0:
        return dict.fromkeys(SPATIAL_CODES, float("nan"))
    points = np.argwhere(roi) * np.asarray(spacing)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    weights = 1.0 / distances
    total_weight = weights.sum()
    return {
        "N365": float(n * np.sum(weights * np.outer(centered, centered))
                      / (total_weight * denominator)),
        "NPT7": float((n - 1) * np.sum(weights * (values[:, None] - values[None, :]) ** 2)
                      / (2 * total_weight * denominator)),
    }


def compare_spatial(actual, expected, *, atol=1e-10, rtol=1e-9):
    if set(actual) != SPATIAL_CODES or set(expected) != SPATIAL_CODES:
        raise ValueError("Expected exactly Moran's I (N365) and Geary's C (NPT7)")
    errors = {}
    for code in sorted(SPATIAL_CODES):
        observed, reference = float(actual[code]), float(expected[code])
        if not (math.isfinite(observed) and math.isfinite(reference)):
            raise ValueError(f"Nonfinite spatial result: {code}")
        if not math.isclose(observed, reference, abs_tol=atol, rel_tol=rtol):
            raise ValueError(f"{code}: {observed} differs from reference {reference}")
        errors[code] = abs(observed - reference)
    return errors


def pictologics_branch_checks(fixture_path):
    from pictologics.features import intensity
    from pictologics.loader import Image

    with np.load(fixture_path) as fixture:
        data, mask, spacing = fixture["data"], fixture["mask"], tuple(fixture["spacing"])
    expected = reference_statistics(data, mask, spacing)
    image = Image(array=data, spacing=spacing, origin=(0.0, 0.0, 0.0))
    roi = Image(array=mask, spacing=spacing, origin=(0.0, 0.0, 0.0))
    original_threshold = intensity._FFT_MIN_PAIRS_PER_POINT
    original_pair = intensity._calculate_spatial_features_numba
    original_fft = intensity._spatial_sums_fft
    original_cap = intensity._fft_memory_cap
    calls = []

    def pair(*args, **kwargs):
        calls.append("pairwise")
        return original_pair(*args, **kwargs)

    def fft(*args, **kwargs):
        calls.append("fft")
        return original_fft(*args, **kwargs)

    results = []
    try:
        intensity._calculate_spatial_features_numba = pair
        intensity._spatial_sums_fft = fft
        # Force each branch only inside this diagnostic worker; never in a benchmark.
        for method, threshold in (("pairwise", 10**30), ("fft", 0)):
            intensity._FFT_MIN_PAIRS_PER_POINT = threshold
            calls.clear()
            values = intensity.calculate_spatial_intensity_features(image, roi)
            actual = {name.rsplit("_", 1)[-1]: value for name, value in values.items()}
            errors = compare_spatial(actual, expected)
            if calls != [method]:
                raise AssertionError(f"Requested {method}, observed {calls}")
            results.append({"method": method, "values": actual, "absolute_errors": errors,
                            "status": "passed"})
        # The memory guard must select exact pairs even if FFT would otherwise win.
        import warnings
        intensity._fft_memory_cap = lambda: 0
        calls.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = intensity.calculate_spatial_intensity_features(image, roi)
        compare_spatial({k.rsplit("_", 1)[-1]: v for k, v in values.items()}, expected)
        if calls != ["pairwise"] or not any("pair loop" in str(w.message) for w in caught):
            raise AssertionError("FFT memory guard did not warn and fall back to pairs")
        results.append({"method": "memory_guard_fallback", "status": "passed"})
    finally:
        intensity._FFT_MIN_PAIRS_PER_POINT = original_threshold
        intensity._calculate_spatial_features_numba = original_pair
        intensity._spatial_sums_fft = original_fft
        intensity._fft_memory_cap = original_cap

    for name, values, selected in (
        ("empty_roi", data, np.zeros_like(mask)),
        ("single_voxel", data, np.eye(mask.size, 1, dtype=np.uint8).reshape(mask.shape)),
        ("constant_intensity", np.ones_like(data), mask),
    ):
        result = intensity.calculate_spatial_intensity_features(
            Image(array=values, spacing=spacing, origin=(0.0, 0.0, 0.0)),
            Image(array=selected, spacing=spacing, origin=(0.0, 0.0, 0.0)),
        )
        if set(k.rsplit("_", 1)[-1] for k in result) != SPATIAL_CODES or not all(
            math.isnan(float(v)) for v in result.values()
        ):
            raise AssertionError(f"Expected defined keys with NaN for {name}")
        results.append({"case": name, "status": "expected_undefined"})
    if intensity.calculate_spatial_intensity_features(image, roi, enabled=False) != {}:
        raise AssertionError("Disabled spatial extraction returned features")
    return results


def qualify(runtime_repo: Path, output_dir: Path, timeout: float) -> dict:
    # This also allows a new qualification script to audit an older frozen checkout.
    sys.path.insert(0, str(runtime_repo))
    import nibabel as nib
    from bench import env
    from bench.ibsi_mapping import classify_feature
    from bench.run import run_adapter_process

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "kind": "untimed_spatial_qualification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": env.native_platform_key(), "final_timing": "deferred",
        "runtime_repo": str(runtime_repo), "environments": {}, "fixtures": [],
        "checks": [], "errors": [], "complete": False,
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": {
            str(path.relative_to(runtime_repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((runtime_repo / "bench").rglob("*.py"))
        },
    }
    profiles = env.load_runtime_profiles()
    for adapter in ("pictologics", "medimage", "mirp", "zrad"):
        report["environments"][adapter] = env.verify_profile(profiles[adapter], smoke=False)
    rng = np.random.default_rng(20261013)
    # MEDimage and MIRP load float32, so give every implementation identical,
    # exactly representable input rather than testing loader precision here.
    data = rng.normal(40, 12, size=(9, 8, 7)).astype(np.float32)
    x, y, z = np.indices(data.shape)
    mask = (((x - 4.1) / 3.4) ** 2 + ((y - 3.2) / 2.8) ** 2
            + ((z - 3.0) / 2.5) ** 2 < 1).astype(np.uint8)
    for fixture_name, spacing in (("isotropic", (1., 1., 1.)),
                                  ("anisotropic", (0.8, 1.7, 2.6))):
        image_path, mask_path = [output_dir / f"{fixture_name}-{name}.nii.gz"
                                 for name in ("image", "mask")]
        affine = np.diag([*spacing, 1.])
        nib.save(nib.Nifti1Image(data, affine), image_path)
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        # NIfTI stores spacing in float32: reference the actual on-disk geometry.
        stored_spacing = tuple(float(s) for s in nib.load(image_path).header.get_zooms())
        expected = reference_statistics(data, mask, stored_spacing)
        fixture_path = output_dir / f"{fixture_name}.npz"
        np.savez(fixture_path, data=data, mask=mask, spacing=stored_spacing)
        report["fixtures"].append({"name": fixture_name, "shape": list(data.shape),
            "roi_voxels": int(mask.sum()), "spacing": stored_spacing, "reference": expected,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        for adapter in ("pictologics", "medimage", "mirp", "zrad"):
            check = {"adapter": adapter, "fixture": fixture_name}
            try:
                payload, _ = run_adapter_process(adapter, image=str(image_path),
                    mask=str(mask_path), discretization="raw", families=["morphology"],
                    benchmark_workload="spatial_autocorrelation", timed=False,
                    iterations=1, include_values=True, timeout=timeout)
                values = payload["values"]["all"]
                actual = {}
                for name, value in values.items():
                    code, status = classify_feature(adapter, name)
                    if status != "mapped" or code not in SPATIAL_CODES or code in actual:
                        raise ValueError(f"Unexpected spatial output: {name}")
                    actual[code] = value
                # MEDimage/MIRP retain float32 moments; the formula check must
                # allow that native precision. Pictologics/ ZRad and the forced
                # Pictologics branches retain the stricter float64 check.
                tolerance = ({"atol": 1e-7, "rtol": 1e-6}
                             if adapter in {"medimage", "mirp"}
                             else {"atol": 1e-10, "rtol": 1e-9})
                check.update(values=actual, native_names=sorted(values), tolerance=tolerance)
                check.update(absolute_errors=compare_spatial(actual, expected, **tolerance),
                             status="passed")
            except Exception as exc:
                check.update(status="failed", error=str(exc))
                report["errors"].append(check)
            report["checks"].append(check)
        environment = os.environ.copy()
        environment["PICTOLOGICS_DISABLE_WARMUP"] = "1"
        environment["NUMBA_NUM_THREADS"] = "2"
        environment["PICTOLOGICS_NUM_THREADS"] = "2"
        process = subprocess.run([
            str(env.env_python(env.env_dir_for_profile(profiles["pictologics"]))),
            str(Path(__file__).resolve()), "--pictologics-worker", str(fixture_path),
        ], capture_output=True, text=True, env=environment, timeout=timeout, check=False)
        try:
            if process.returncode:
                raise RuntimeError(process.stderr)
            checks = json.loads(process.stdout)
            report["checks"].extend(dict(adapter="pictologics", fixture=fixture_name, **c)
                                    for c in checks)
        except Exception as exc:
            report["errors"].append({"adapter": "pictologics", "fixture": fixture_name,
                                     "status": "failed", "error": str(exc)})
    report["checks"].append({"adapter": "pyradiomics", "status": "unsupported",
                              "reason": "No native Moran's I or Geary's C workload"})
    report["complete"] = not report["errors"]
    (output_dir / "spatial-qualification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/spatial-qualification"))
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--pictologics-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pictologics_worker:
        print(json.dumps(pictologics_branch_checks(args.pictologics_worker), allow_nan=False))
        return 0
    report = qualify(args.runtime_repo.resolve(), args.output_dir.resolve(), args.timeout)
    print(json.dumps({"complete": report["complete"], "checks": len(report["checks"]),
                      "errors": report["errors"]}, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

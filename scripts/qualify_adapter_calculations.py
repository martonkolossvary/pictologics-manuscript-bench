"""Qualify adapters' calculation boundaries without collecting timings.

Compares the prepared calculation endpoint with the package's public extraction
path (MEDimage exposes family extractors), then repeats the prepared calculation
to detect retained state. This is an API/value equivalence check, not IBSI
compliance or a performance observation. Each adapter runs in its locked native
environment. No benchmark ledger, calibration or timing routine is used.
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
import traceback


STABLE_ADAPTERS = ("medimage", "mirp", "pyradiomics", "zrad")
ADAPTERS = (*STABLE_ADAPTERS, "pictologics")
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_values(reference, observed, *, rtol=1e-9, atol=1e-12):
    """Require the complete finite scalar feature surface and numerical parity."""
    if not reference or set(reference) != set(observed):
        raise ValueError(
            f"feature names differ: missing={sorted(set(reference) - set(observed))}; "
            f"extra={sorted(set(observed) - set(reference))}"
        )
    mismatches = []
    maximum = 0.0
    for name in sorted(reference):
        a, b = reference[name], observed[name]
        if isinstance(a, bool) or isinstance(b, bool):
            raise ValueError(f"non-numeric feature: {name}")
        a, b = float(a), float(b)
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"non-finite feature: {name}")
        maximum = max(maximum, abs(a - b))
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            mismatches.append({"feature": name, "reference": a, "observed": b})
    if mismatches:
        raise ValueError(f"value mismatch: {json.dumps(mismatches)}")
    return {"features": len(reference), "max_absolute_difference": maximum}


def create_fixture(output_dir: Path):
    import nibabel as nib
    import numpy as np

    shape = (12, 11, 10)
    raw = np.random.default_rng(20261013).uniform(1.0, 100.0, shape).astype(np.float32)
    x, y, z = np.indices(shape)
    mask = (
        ((x - 5.2) / 3.8) ** 2
        + ((y - 5.0) / 3.5) ** 2
        + ((z - 4.4) / 3.1) ** 2 < 1
    ).astype(np.uint8)
    # Ensure all 32 bins occur in the ROI: identity IVH needs full-span levels.
    levels = np.floor((raw - 1.0) / 99.0 * 32).astype(np.int16) + 1
    levels[tuple(np.argwhere(mask)[0])] = 1
    levels[tuple(np.argwhere(mask)[-1])] = 32
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, array in (("raw", raw), ("levels", levels), ("mask", mask)):
        path = (output_dir / f"{name}.nii.gz").resolve()
        nib.save(nib.Nifti1Image(array, np.diag([1.2, 1.3, 1.5, 1.0])), path)
        paths[name] = str(path)
    return paths


def _pictologics(fixture, families, workload, discrete):
    from pictologics.loader import load_image
    from bench.adapters import pictologics_adapter as adapter

    image = load_image(fixture["levels" if discrete else "raw"])
    mask = adapter._to_mask_image(image, load_image(fixture["mask"]))
    arguments = dict(image=image, mask=mask, families=families,
                     discretization="identity" if discrete else "raw", bins=32,
                     bin_width=1.0, intensity_range=(1.0, 32.0) if discrete else None,
                     benchmark_workload=workload)
    reference = adapter._compute_features(**arguments)
    calculate, finalize = adapter._prepare_calculation_only(**arguments)
    prepare = adapter._calculation_preparer(families)

    def prepared():
        if prepare is not None:
            prepare()
        return finalize(calculate())

    return reference, prepared


def _mirp(fixture, families, workload, discrete):
    import mirp
    from bench.adapters import mirp_adapter as adapter
    from bench.ibsi_mapping import IBSI_FAMILY_TO_MIRP

    native = [IBSI_FAMILY_TO_MIRP[name] for name in families]
    kwargs = dict(
        image_path=fixture["levels" if discrete else "raw"],
        mask_path=fixture["mask"], modality="ct",
    )
    image, mask = adapter._load_native_pair(mirp, **kwargs)
    settings = adapter._build_settings(
        families=native, discretization="identity" if discrete else "raw",
        bins=32, bin_width=1.0, intensity_range=(1.0, 32.0) if discrete else None,
        aggregation="3d_merge", ibsi_compliant=False,
        identity_ivh_bins=adapter._identity_ivh_bin_count(image, mask)
        if workload == "ivh" else None,
    )
    # Use independent native image/mask objects for the public reference.
    reference_image, reference_mask = adapter._load_native_pair(mirp, **kwargs)
    reference = adapter._extract_public(
        mirp, image=reference_image, mask=reference_mask,
        settings=settings, families=native,
    )
    selected_names = adapter._benchmark_feature_names(workload)
    if workload == "morphology":
        # MIRP expands morph_max_2d_diam into three axis-specific outputs.
        # Partition the public output by excluding only the two spatial names.
        reference = {name: value for name, value in reference.items()
                     if name not in adapter._SPATIAL_AUTOCORRELATION_FEATURES}
    elif selected_names is not None:
        reference = {name: value for name, value in reference.items() if name in selected_names}
    calculate, finalize = adapter._prepare_calculation_only(
        image=image, mask=mask, settings=settings, families=native,
        identity_input=discrete, selected_feature_names=selected_names,
    )
    return reference, lambda: finalize(calculate())


def _pyradiomics(fixture, families, workload, discrete):
    import radiomics
    import SimpleITK as sitk
    from radiomics import featureextractor
    from bench.adapters import pyradiomics_adapter as adapter

    image = sitk.ReadImage(fixture["levels" if discrete else "raw"])
    mask = adapter._binary_mask(sitk.ReadImage(fixture["mask"]), sitk)
    registry = {name: cls.getFeatureNames() for name, cls in radiomics.getFeatureClasses().items()}
    selection = adapter._native_family_selection(families, registry)
    settings = adapter._default_settings()
    settings.update(adapter._aggregation_settings("3d_merge", families))
    if discrete:
        adapter._validate_identity_image(image, mask, sitk)
        settings.update(binWidth=1.0, resegmentRange=[1.0, 32.0], resegmentMode="absolute")
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**selection)
    reference = adapter._finite_calculated_values(extractor.execute(image, mask))
    calculate, finalize = adapter._prepare_calculation_only(
        image=image, mask=mask, selection=selection,
        settings=settings, radiomics_module=radiomics,
    )
    return reference, lambda: finalize(calculate())


def _medimage(fixture, families, workload, discrete):
    import numpy as np
    from bench.adapters import medimage_adapter as adapter

    image, spacing = adapter._load_nifti(Path(fixture["levels" if discrete else "raw"]))
    mask, _ = adapter._load_nifti(Path(fixture["mask"]))
    mask = (mask > 0).astype(np.uint8)
    modules = adapter._medimage_modules()
    discretization = "identity" if discrete else "raw"
    intensity_range = (1.0, 32.0) if discrete else None
    reference = adapter._prepare_and_compute_features(
        families=families, modules=modules, image_full=image, binary_mask=mask,
        spacing=spacing, intensity_type="definite", discretization=discretization,
        bins=32, bin_width=1.0, intensity_range=intensity_range,
        benchmark_workload=workload,
    )
    raw, quant, ivh, width, intensity_mask = adapter._prepare_roi_volumes(
        image, mask, discretization=discretization, bins=32, bin_width=1.0,
        intensity_range=intensity_range, apply_discretization=discrete,
    )
    valid = raw[np.isfinite(raw)]

    def calculate():
        payloads = adapter._compute_feature_payloads(
            families=families, modules=modules, image_full=image,
            vol_raw=raw, vol_quant=quant, vol_ivh=ivh,
            morphology_mask_bool=mask, intensity_mask_bool=intensity_mask.astype(np.uint8),
            spacing=np.asarray(spacing), intensity_type="definite", wd=width,
            discretization=discretization, intensity_range=intensity_range,
            prepared_ivh_range=(float(valid.min()), float(valid.max())),
            benchmark_workload=workload,
        )
        return adapter._finalize_feature_payloads(payloads)

    return reference, calculate


def _zrad(fixture, families, workload, discrete):
    from zrad.radiomics import Radiomics
    from bench.adapters import zrad_adapter as adapter

    image, mask = adapter._load_nifti_pair(
        Path(fixture["levels" if discrete else "raw"]), Path(fixture["mask"])
    )
    preparation = dict(
        image=image, mask=mask, families=families,
        discretization="identity" if discrete else "raw", bins=32, bin_width=1.0,
        intensity_range=(1.0, 32.0) if discrete else None,
        preprocessing_classes=adapter._zrad_preprocessing_classes(),
    )
    reference_roi = adapter._prepare_roi_data(**preparation)
    adapter._clear_zrad_local_intensity_cache(families)
    try:
        reference = Radiomics(aggr_dim="3D", aggr_method="MERG").extract_features(
            roi_data=reference_roi, families=adapter._native_families(families, workload),
            include_metadata=False,
        )
        _, reference = adapter._finalize_zrad_features(reference, families=families)
    finally:
        adapter._clear_zrad_local_intensity_cache(families)
    roi = adapter._prepare_roi_data(**preparation)
    engine = Radiomics(aggr_dim="3D", aggr_method="MERG")

    def calculate():
        adapter._clear_zrad_local_intensity_cache(families)
        try:
            raw = adapter._compute_zrad_raw_features(
                roi_data=roi, families=families, aggr_dim="3D", aggr_method="MERG",
                radiomics_engine=engine, benchmark_workload=workload,
            )
            return adapter._finalize_zrad_features(raw, families=families)[1]
        finally:
            adapter._clear_zrad_local_intensity_cache(families)

    return reference, calculate


def worker(adapter_name: str, fixture, output: Path):
    from importlib.metadata import version
    import platform
    from bench.adapters.registry import get_adapter
    from bench.benchmark_workloads import WORKLOADS

    distribution = {"medimage": "medimage-pkg", "mirp": "mirp", "pyradiomics": "pyradiomics", "zrad": "z-rad", "pictologics": "pictologics"}[adapter_name]
    result = {
        "adapter": adapter_name, "distribution": distribution,
        "distribution_version": version(distribution), "python": platform.python_version(),
        "executable": sys.executable, "timing_observations": False,
        "checks": [],
    }
    capabilities = get_adapter(adapter_name)
    for workload in WORKLOADS:
        record = {"workload": workload.name}
        if not capabilities.supports_workload(workload.name):
            record["status"] = "unsupported"
        else:
            try:
                families = [name for name in workload.families if capabilities.supports(name)]
                reference, calculate = globals()["_" + adapter_name](
                    fixture, families, workload.name, workload.name in {"texture", "ivh"}
                )
                observed = calculate()
                record["public_equivalence"] = compare_values(reference, observed)
                record["repeat_equivalence"] = compare_values(observed, calculate())
                record["feature_names"] = sorted(observed)
                record["values"] = {name: float(value) for name, value in observed.items()}
                record["status"] = "passed"
            except Exception:
                record.update(status="failed", error=traceback.format_exc())
        result["checks"].append(record)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return int(any(check["status"] == "failed" for check in result["checks"]))


def qualify(profiles, output_dir, runtime_repo, timeout):
    """Verify locked runtimes, then launch calculation-only qualification workers."""
    from bench import env

    output_dir = output_dir.resolve()
    runtime_repo = runtime_repo.resolve()
    fixture = create_fixture(output_dir / "fixture")
    fixture_path = output_dir / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    report = {
        "schema_version": 1, "kind": "untimed_prepared_calculation_qualification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_adapters": list(profiles), "complete": False,
        "platform": env.native_platform_key(),
        "timing_observations": False, "rtol": 1e-9, "atol": 1e-12,
        "fixture": {key: {"path": value, "sha256": _sha256(Path(value))} for key, value in fixture.items()},
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in [Path(__file__), *sorted((ROOT / "bench" / "adapters").glob("*.py"))]
        },
        "adapters": [],
    }
    profiles_by_name = env.load_runtime_profiles()
    for name in profiles:
        record = {"adapter": name}
        try:
            profile = profiles_by_name[name]
            # Verify the actual environment using its own checkout's verifier,
            # while requiring exactly the same version/lock profile as ours.
            profile_path = Path("configs/adapters") / f"{name}.yaml"
            if _sha256(ROOT / profile_path) != _sha256(runtime_repo / profile_path):
                raise ValueError(f"runtime profile differs: {profile_path}")
            verify_command = (
                "import json; from bench import env; "
                f"profile = env.load_runtime_profiles()[{name!r}]; "
                "print(json.dumps(dict("
                "environment=env.verify_profile(profile, smoke=False), "
                "worker_executable=str(env.env_python(env.env_dir_for_profile(profile))))))"
            )
            verify_env = dict(os.environ, PYTHONPATH=str(runtime_repo))
            verified = subprocess.run(
                [sys.executable, "-c", verify_command], cwd=runtime_repo,
                env=verify_env, text=True, capture_output=True, check=True, timeout=timeout,
            )
            verified_runtime = json.loads(verified.stdout)
            provenance = verified_runtime["environment"]
            expected_version = profile.metadata_version or profile.version
            if provenance["distribution_metadata_version"] != expected_version:
                raise ValueError("verified runtime version differs from current profile")
            record["environment"] = provenance
            # Provenance resolves the executable for identity. On POSIX that
            # follows bin/python out of the venv, so it is not a launch path.
            record["worker_executable"] = verified_runtime["worker_executable"]
            worker_path = output_dir / f"{name}.json"
            child_env = dict(os.environ, PYTHONPATH=str(ROOT))
            child_env["PICTOLOGICS_DISABLE_WARMUP"] = "1"
            for key in ("PICTOLOGICS_NUM_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
                child_env[key] = "1"
            completed = subprocess.run(
                [record["worker_executable"], "-m", "scripts.qualify_adapter_calculations",
                 "--worker", name, "--fixture", str(fixture_path), "--output-dir", str(output_dir)],
                cwd=ROOT, env=child_env, text=True, capture_output=True, timeout=timeout,
            )
            (output_dir / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output_dir / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if not worker_path.is_file():
                raise RuntimeError(f"qualification worker exited {completed.returncode}: {completed.stderr[-4000:]}")
            record["qualification"] = json.loads(worker_path.read_text(encoding="utf-8"))
            record["status"] = "passed" if completed.returncode == 0 else "failed"
        except Exception:
            record.update(status="failed", error=traceback.format_exc())
        report["adapters"].append(record)
        report["complete"] = len(report["adapters"]) == len(profiles)
        report["status"] = "passed" if all(item["status"] == "passed" for item in report["adapters"]) else "failed"
        (output_dir / "qualification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"adapter": name, "status": record["status"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=ADAPTERS, default=list(STABLE_ADAPTERS))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calculation-qualification"))
    parser.add_argument("--runtime-repo", type=Path, default=ROOT)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--worker", choices=ADAPTERS, help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker, json.loads(args.fixture.read_text(encoding="utf-8")), args.output_dir / f"{args.worker}.json")
    report = qualify(args.profiles, args.output_dir, args.runtime_repo, args.timeout)
    return int(not report["complete"] or report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())

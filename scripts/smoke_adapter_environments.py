"""Exercise native adapter workloads on a small synthetic volume, without timing.

This is an installation/API portability check, not scientific validation or a
performance result. Publication hosts must also run the frozen input audit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from bench import env
from bench.adapters.registry import get_adapter
from bench.benchmark_workloads import WORKLOADS
from bench.run import run_adapter_process


def smoke(profiles: list[str], output_dir: Path, timeout: float) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (12, 11, 10)
    rng = np.random.default_rng(20261012)
    raw = rng.uniform(1.0, 100.0, size=shape)
    # FBN32 also gives the native IVH implementation enough upper-tail levels
    # to define I10 on this small fixture (eight uniform levels do not).
    levels = np.floor((raw - 1.0) / 99.0 * 32).astype(np.int16) + 1
    x, y, z = np.indices(shape)
    # Avoid equal principal axes: legacy native ellipsoid formulas can be
    # undefined on an exactly symmetric tiny ROI.
    mask = (((x - 5.2) / 3.8) ** 2 + ((y - 5.0) / 3.5) ** 2 + ((z - 4.4) / 3.1) ** 2 < 1).astype(
        np.uint8
    )
    affine = np.diag([1.2, 1.3, 1.5, 1.0])
    paths = {}
    for name, array in (("raw", raw), ("levels", levels), ("mask", mask)):
        path = (output_dir / f"{name}.nii.gz").resolve()
        nib.save(nib.Nifti1Image(array, affine), path)
        paths[name] = str(path)

    results = []
    runtime_profiles = env.load_runtime_profiles()
    for name in profiles:
        profile = runtime_profiles[name]
        env.verify_profile(profile, smoke=False)
        adapter = get_adapter(name)
        for workload in WORKLOADS:
            if not adapter.supports_workload(workload.name):
                continue
            discrete = workload.name in {"texture", "ivh"}
            payload, _ = run_adapter_process(
                name,
                image=paths["levels" if discrete else "raw"],
                mask=paths["mask"],
                discretization="identity" if discrete else "raw",
                bins=32,
                intensity_min=1.0 if discrete else None,
                intensity_max=32.0 if discrete else None,
                modality="CT",
                families=[
                    family for family in workload.families if adapter.supports(family)
                ],
                benchmark_workload=workload.name,
                iterations=1,
                include_values=True,
                timed=False,
                timeout=timeout,
            )
            values = payload.get("values", {}).get("all", {})
            finite = sum(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in values.values()
            )
            if not values or finite != len(values):
                raise RuntimeError(
                    f"{name}/{workload.name}: finite outputs {finite}/{len(values)}"
                )
            record = {
                "adapter": name,
                "workload": workload.name,
                "version": profile.version,
                "finite_features": finite,
                "status": "passed",
            }
            results.append(record)
            print(json.dumps(record), flush=True)
            (output_dir / "smoke-results.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", nargs="+", choices=sorted(env.load_runtime_profiles())
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/native-smoke")
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    smoke(
        args.profiles or sorted(env.load_runtime_profiles()),
        args.output_dir,
        args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

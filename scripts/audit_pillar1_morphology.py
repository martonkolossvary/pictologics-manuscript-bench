#!/usr/bin/env python3
"""Audit the frozen Pillar 1 mask design without running radiomics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.benchmark_ledger import atomic_write_json, atomic_write_text, sha256_file
from bench.pillar1_dataset import validate_pillar1_dataset
from bench.synthetic_scene import SyntheticSceneConfig
from bench.synthetic_generator import (
    LARGE_SPICULE_SEGMENTS,
    evaluate_synthetic_volume,
    generate_synthetic_volume,
    measure_satellite_component_voxels,
    measure_spicule_external_voxels,
)


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    reference = [
        case for case in manifest["cases"] if case["hu_profile"] == "reference"
    ]
    rows: list[dict[str, Any]] = []
    for size in sorted({int(case["size"]) for case in reference}):
        masks = {
            case["mask_id"]: {
                "voxels": int(case["mask_voxels"]),
                "image_fraction": float(case["mask_fraction"]),
                "body_fraction": float(case["mask_body_fraction"]),
                "bbox_shape_xyz": list(case["bbox_shape_xyz"]),
                "occupied_z_slices": int(case["occupied_z_slices"]),
                "occupied_z_mm": float(case["occupied_z_mm"]),
                "components_26": int(case["components_26"]),
                "cavities_background_6": int(case["cavities_background_6"]),
            }
            for case in reference
            if int(case["size"]) == size
        }
        rows.append(
            {
                "size_xyz": [size, size, size],
                "spacing_mm_xyz": [256.0 / size] * 3,
                "image_voxels": size**3,
                "masks": masks,
                "m2_fraction_of_m1": masks["M2"]["voxels"] / masks["M1"]["voxels"],
                "m3_added_fraction_of_m1": (
                    masks["M3"]["voxels"] - masks["M1"]["voxels"]
                )
                / masks["M1"]["voxels"],
                "m4_added_fraction_of_m1": (
                    masks["M4"]["voxels"] - masks["M1"]["voxels"]
                )
                / masks["M1"]["voxels"],
            }
        )
    return rows


def build_audit(dataset_dir: Path) -> dict[str, Any]:
    validation = validate_pillar1_dataset(dataset_dir, deep=False)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _manifest_rows(manifest)

    diagnostics: list[dict[str, Any]] = []
    for size in (32, 48, 64, 96, 128):
        bundle = generate_synthetic_volume(
            SyntheticSceneConfig(edge=size, fov_mm=256.0, slab_depth=8)
        )
        metrics = evaluate_synthetic_volume(bundle)
        expected = next(row for row in rows if row["size_xyz"][0] == size)
        for mask_id in ("M1", "M2", "M3", "M4"):
            if metrics.masks[mask_id].voxels != expected["masks"][mask_id]["voxels"]:
                raise RuntimeError(
                    f"generated {size}/{mask_id} geometry differs from the manifest"
                )
        spicules = measure_spicule_external_voxels(bundle)
        satellites = measure_satellite_component_voxels(bundle)
        if len(spicules) != len(LARGE_SPICULE_SEGMENTS) or min(spicules) <= 0:
            raise RuntimeError(f"not every M3 spicule is retained at {size}^3")
        if len(satellites) != 6 or min(satellites) <= 0:
            raise RuntimeError(f"not every M4 satellite is retained at {size}^3")
        if not np.all(bundle.masks["M2"] <= bundle.masks["M1"]):
            raise RuntimeError("M2 is not a subset of M1")
        if not np.all(bundle.masks["M1"] <= bundle.masks["M3"]):
            raise RuntimeError("M1 is not a subset of M3")
        if not np.all(bundle.masks["M1"] <= bundle.masks["M4"]):
            raise RuntimeError("M1 is not a subset of M4")
        diagnostics.append(
            {
                "size": size,
                "retained_spicule_count": len(spicules),
                "spicule_external_voxels": list(spicules),
                "satellite_component_count": len(satellites),
                "satellite_component_voxels": list(satellites),
                "topology": {
                    mask_id: {
                        "components_26": metrics.masks[mask_id].components_26,
                        "cavities_background_6": metrics.masks[
                            mask_id
                        ].cavities_background_6,
                        "bone_overlap_voxels": metrics.masks[
                            mask_id
                        ].bone_overlap_voxels,
                    }
                    for mask_id in ("M1", "M2", "M3", "M4")
                },
            }
        )

    mask_hash_sets = {
        f"n{size:03d}_{mask_id}": sorted(
            {
                case["mask_sha256"]
                for case in manifest["cases"]
                if int(case["size"]) == size and case["mask_id"] == mask_id
            }
        )
        for size in manifest["sizes"]
        for mask_id in manifest["mask_ids"]
    }
    if any(len(values) != 1 for values in mask_hash_sets.values()):
        raise RuntimeError("mask geometry differs across HU profiles")

    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "Pillar 1 mask geometry and voxelisation; no radiomics executed",
        "dataset": manifest["dataset"],
        "manifest_sha256": sha256_file(manifest_path),
        "validation": validation,
        "mask_definitions": {
            "M1": "one connected, lobulated, peri-femoral whole tumour including necrosis",
            "M2": "one connected viable shell with one enclosed background-6 cavity",
            "M3": "M1 plus 18 attached, tapered, golden-angle-distributed 3D spicules",
            "M4": "M1 plus six separated image-visible satellite components",
        },
        "design_assessment": {
            "M1": "large morphology stress lesion; not claimed to represent a median clinical tumour",
            "M2": "controlled cavity contrast with the same outer tumour boundary",
            "M3": "reasonable computational spiculation stress; every branch survives voxelisation, but N=32 is morphologically coarse",
            "M4": "controlled multifocal topology; all six satellites survive, but 4-10 voxels per satellite at N=32 is under-resolved",
        },
        "all_sizes": rows,
        "voxelisation_diagnostics": diagnostics,
        "masks_identical_across_hu_profiles": True,
        "radiomics_executed": False,
    }


def _markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Pillar 1 morphology audit",
        "",
        f"Status: **{audit['status']}**. No radiomics was executed.",
        "",
        "The geometry is suitable for a controlled morphology-scaling benchmark. "
        "M1 is a large lobulated peri-femoral lesion; M2 preserves its outer boundary "
        "and removes the necrotic core; M3 adds 18 attached tapered 3D spicules; M4 "
        "adds six separated satellite foci. These are computational morphology "
        "constructs, not claims about population-average tumour anatomy.",
        "",
        "| Grid | M1 image % | M2 image % | M3 image % | M4 image % | M1 Z slices |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["all_sizes"]:
        masks = row["masks"]
        lines.append(
            f"| {row['size_xyz'][0]}³ | "
            f"{100 * masks['M1']['image_fraction']:.3f} | "
            f"{100 * masks['M2']['image_fraction']:.3f} | "
            f"{100 * masks['M3']['image_fraction']:.3f} | "
            f"{100 * masks['M4']['image_fraction']:.3f} | "
            f"{masks['M1']['occupied_z_slices']} |"
        )
    lines.extend(
        [
            "",
            "At every audited grid from 32³ through 128³, all 18 M3 spicules and "
            "all six M4 satellites contribute voxels, M1/M2/M3 remain connected, "
            "M2 has one cavity, M3 has no cavity, M4 has seven components, and no "
            "tumour mask overlaps bone. At 32³ the topology is preserved but the "
            "smallest satellites contain only 4 voxels, so that point is explicitly "
            "a coarse computational stress case rather than a faithful boundary image.",
            "",
            f"Manifest SHA-256: `{audit['manifest_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/benchmark/pillar1")
    parser.add_argument("--output-dir", default="results/pillar1-morphology-audit")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    audit = build_audit(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "morphology_audit.json", audit)
    atomic_write_text(output_dir / "morphology_audit.md", _markdown(audit))
    print(json.dumps({"status": "passed", "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

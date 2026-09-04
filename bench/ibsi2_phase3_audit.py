from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
from scipy import ndimage

from bench.dataset_manifest import (
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)


MODALITIES = ("ct", "mri", "pet")
CASE_PATTERN = re.compile(
    r"^(?P<subject>STS_\d{3})_(?P<modality>ct|mri|pet)_image\.nii\.gz$"
)
AUDIT_SCHEMA_VERSION = 1


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {}
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cavity_count(mask: np.ndarray) -> int:
    coordinates = np.argwhere(mask)
    lower = np.maximum(coordinates.min(axis=0) - 1, 0)
    upper = np.minimum(coordinates.max(axis=0) + 2, np.asarray(mask.shape))
    region = mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    padded = np.pad(region, 1, constant_values=False)
    background_labels, count = ndimage.label(
        ~padded,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    exterior = int(background_labels[0, 0, 0])
    cavity_labels = set(range(1, int(count) + 1)) - {exterior}
    return len(cavity_labels)


def discover_pairs(dataset_root: Path) -> list[dict[str, Any]]:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("dataset must contain images/ and masks/ directories")

    pairs: list[dict[str, Any]] = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        match = CASE_PATTERN.fullmatch(image_path.name)
        if match is None:
            continue
        subject = match.group("subject")
        modality = match.group("modality")
        mask_path = masks_dir / f"{subject}_{modality}_mask.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"missing mask for {image_path.name}: {mask_path.name}"
            )
        pairs.append(
            {
                "case_id": f"{subject}_{modality}",
                "subject": subject,
                "modality": modality,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    if not pairs:
        raise FileNotFoundError("no STS image-mask pairs were discovered")
    return pairs


def _manifest_index(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries: dict[str, Mapping[str, Any]] = {}
    for raw in manifest.get("files", []):
        if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
            entries[str(raw["path"])] = raw
    return entries


def inspect_pair(
    dataset_root: Path,
    pair: Mapping[str, Any],
    manifest_files: Mapping[str, Mapping[str, Any]],
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    image_path = Path(pair["image_path"])
    mask_path = Path(pair["mask_path"])
    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))

    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 3 or tuple(mask.shape) != shape:
        raise ValueError(f"{pair['case_id']}: image/mask shape mismatch")
    if not np.allclose(image.affine, mask.affine, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{pair['case_id']}: image/mask affine mismatch")

    image_values = np.asanyarray(image.dataobj)
    mask_values = np.asanyarray(mask.dataobj)
    if not np.all(np.isfinite(image_values)):
        raise ValueError(f"{pair['case_id']}: non-finite image values")
    if not np.all(np.isfinite(mask_values)):
        raise ValueError(f"{pair['case_id']}: non-finite mask values")
    observed_mask_values = np.unique(mask_values)
    if not np.all(np.isin(observed_mask_values, (0, 1))):
        raise ValueError(f"{pair['case_id']}: mask is not canonical binary")
    roi = mask_values == 1
    coordinates = np.argwhere(roi)
    if coordinates.size == 0:
        raise ValueError(f"{pair['case_id']}: empty mask")

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"{pair['case_id']}: invalid spacing {spacing}")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    bbox_shape = upper - lower + 1
    image_voxels = int(np.prod(shape, dtype=np.int64))
    roi_voxels = int(coordinates.shape[0])
    bbox_voxels = int(np.prod(bbox_shape, dtype=np.int64))
    voxel_volume = float(np.prod(spacing))
    components_26 = int(
        ndimage.label(roi, structure=np.ones((3, 3, 3), dtype=np.uint8))[1]
    )
    roi_values = np.asarray(image_values[roi], dtype=np.float64)

    image_relative = image_path.relative_to(dataset_root).as_posix()
    mask_relative = mask_path.relative_to(dataset_root).as_posix()
    image_hash = sha256_file(image_path) if verify_hashes else None
    mask_hash = sha256_file(mask_path) if verify_hashes else None
    manifest_image = manifest_files.get(image_relative, {})
    manifest_mask = manifest_files.get(mask_relative, {})
    image_hash_matches = (
        image_hash == manifest_image.get("sha256") if verify_hashes else None
    )
    mask_hash_matches = (
        mask_hash == manifest_mask.get("sha256") if verify_hashes else None
    )

    return {
        "case_id": str(pair["case_id"]),
        "subject": str(pair["subject"]),
        "modality": str(pair["modality"]),
        "image_path": image_relative,
        "mask_path": mask_relative,
        "image_sha256": image_hash,
        "mask_sha256": mask_hash,
        "manifest_image_hash_matches": image_hash_matches,
        "manifest_mask_hash_matches": mask_hash_matches,
        "image_dtype": str(image.get_data_dtype()),
        "mask_dtype": str(mask.get_data_dtype()),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "image_voxels": image_voxels,
        "roi_voxels": roi_voxels,
        "roi_fraction": roi_voxels / image_voxels,
        "roi_volume_ml": roi_voxels * voxel_volume / 1000.0,
        "roi_z_slices": int(np.unique(coordinates[:, 2]).size),
        "roi_z_fraction": float(np.unique(coordinates[:, 2]).size / shape[2]),
        "bbox_x_vox": int(bbox_shape[0]),
        "bbox_y_vox": int(bbox_shape[1]),
        "bbox_z_vox": int(bbox_shape[2]),
        "bbox_x_mm": float(bbox_shape[0] * spacing[0]),
        "bbox_y_mm": float(bbox_shape[1] * spacing[1]),
        "bbox_z_mm": float(bbox_shape[2] * spacing[2]),
        "bbox_voxels": bbox_voxels,
        "bbox_occupancy": roi_voxels / bbox_voxels,
        "components_26": components_26,
        "cavities_background_6": _cavity_count(roi),
        "roi_raw_unique_values": int(np.unique(roi_values).size),
        "roi_raw_min": float(np.min(roi_values)),
        "roi_raw_max": float(np.max(roi_values)),
        "roi_raw_mean": float(np.mean(roi_values)),
        "roi_raw_sd": float(np.std(roi_values, ddof=0)),
    }


def _modality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "subjects": len({row["subject"] for row in rows}),
        "unique_shapes": len(
            {(row["shape_x"], row["shape_y"], row["shape_z"]) for row in rows}
        ),
        "unique_spacings": len(
            {
                (
                    round(row["spacing_x_mm"], 6),
                    round(row["spacing_y_mm"], 6),
                    round(row["spacing_z_mm"], 6),
                )
                for row in rows
            }
        ),
        "orientations": dict(
            sorted(Counter(row["orientation"] for row in rows).items())
        ),
        "shape_x": _quantiles(row["shape_x"] for row in rows),
        "shape_y": _quantiles(row["shape_y"] for row in rows),
        "shape_z": _quantiles(row["shape_z"] for row in rows),
        "spacing_x_mm": _quantiles(row["spacing_x_mm"] for row in rows),
        "spacing_y_mm": _quantiles(row["spacing_y_mm"] for row in rows),
        "spacing_z_mm": _quantiles(row["spacing_z_mm"] for row in rows),
        "image_voxels": _quantiles(row["image_voxels"] for row in rows),
        "roi_voxels": _quantiles(row["roi_voxels"] for row in rows),
        "roi_fraction_percent": _quantiles(100.0 * row["roi_fraction"] for row in rows),
        "roi_volume_ml": _quantiles(row["roi_volume_ml"] for row in rows),
        "roi_z_slices": _quantiles(row["roi_z_slices"] for row in rows),
        "roi_z_fraction_percent": _quantiles(
            100.0 * row["roi_z_fraction"] for row in rows
        ),
        "bbox_occupancy_percent": _quantiles(
            100.0 * row["bbox_occupancy"] for row in rows
        ),
        "components_26": dict(
            sorted(Counter(str(row["components_26"]) for row in rows).items())
        ),
        "cavities_background_6": dict(
            sorted(Counter(str(row["cavities_background_6"]) for row in rows).items())
        ),
        "roi_raw_unique_values": _quantiles(
            row["roi_raw_unique_values"] for row in rows
        ),
    }


def summarise_audit(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    subjects = sorted({row["subject"] for row in rows})
    grouped_subjects: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped_subjects[row["subject"]].add(row["modality"])
    incomplete = {
        subject: sorted(set(MODALITIES) - modalities)
        for subject, modalities in grouped_subjects.items()
        if modalities != set(MODALITIES)
    }
    by_modality = {
        modality: _modality_summary(
            [row for row in rows if row["modality"] == modality]
        )
        for modality in MODALITIES
    }

    per_subject_volume_ratio: list[float] = []
    for subject in subjects:
        volumes = [row["roi_volume_ml"] for row in rows if row["subject"] == subject]
        if len(volumes) == len(MODALITIES) and min(volumes) > 0:
            per_subject_volume_ratio.append(max(volumes) / min(volumes))

    actual_dtypes = sorted({row["image_dtype"] for row in rows})
    manifest_image_dtype = manifest.get("image_dtype")
    manifest_sizes = manifest.get("sizes")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": str(manifest.get("dataset", "unknown")),
        "source_manifest_sha256": manifest_sha256,
        "subjects": len(subjects),
        "modalities": list(MODALITIES),
        "paired_cases": len(rows),
        "image_files": len(rows),
        "mask_files": len(rows),
        "complete_subject_blocks": len(subjects) - len(incomplete),
        "incomplete_subject_blocks": incomplete,
        "all_masks_binary_nonempty": True,
        "all_image_mask_shapes_and_affines_match": True,
        "all_manifest_hashes_match": all(
            row["manifest_image_hash_matches"] is True
            and row["manifest_mask_hash_matches"] is True
            for row in rows
        ),
        "actual_image_dtypes": actual_dtypes,
        "source_manifest_image_dtype": manifest_image_dtype,
        "source_manifest_dtype_matches": manifest_image_dtype in actual_dtypes,
        "source_manifest_sizes": manifest_sizes,
        "source_manifest_size_assessment": (
            "modality surrogate labels; not observed matrix dimensions and prohibited "
            "as scaling variables"
        ),
        "modality_summary": by_modality,
        "multi_component_cases": sorted(
            row["case_id"] for row in rows if row["components_26"] > 1
        ),
        "all_cases": {
            "image_voxels": _quantiles(row["image_voxels"] for row in rows),
            "roi_voxels": _quantiles(row["roi_voxels"] for row in rows),
            "roi_fraction_percent": _quantiles(
                100.0 * row["roi_fraction"] for row in rows
            ),
            "roi_volume_ml": _quantiles(row["roi_volume_ml"] for row in rows),
        },
        "within_subject_max_to_min_roi_volume_ratio": _quantiles(
            per_subject_volume_ratio
        ),
        "case_inventory_sha256": _sha256_json(rows),
    }


def _format_quantile_line(
    label: str, values: Mapping[str, float], unit: str = ""
) -> str:
    display_label = f"{label} ({unit})" if unit else label
    return (
        f"| {display_label} | {values['min']:.3g} | {values['median']:.3g} | "
        f"{values['max']:.3g} |"
    )


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# IBSI 2 Phase 3 fixed-cohort inventory",
        "",
        "This is a read-only geometry and pairing audit. It does not contain radiomic "
        "feature calculations or benchmark timings.",
        "",
        "## Inventory",
        "",
        f"- Subjects: **{summary['subjects']}**",
        f"- Modalities: **{', '.join(summary['modalities'])}**",
        f"- Complete subject–modality blocks: **{summary['complete_subject_blocks']}**",
        f"- Paired 3D image–mask cases: **{summary['paired_cases']}**",
        f"- NIfTI files: **{summary['image_files']} images + {summary['mask_files']} masks**",
        "- All masks are nonempty canonical binary masks; each mask matches its image "
        "in shape and affine.",
        "",
        "## Observed geometry by modality",
        "",
        "| Measure | Minimum | Median | Maximum |",
        "|---|---:|---:|---:|",
    ]
    for modality in MODALITIES:
        data = summary["modality_summary"][modality]
        lines.append(
            _format_quantile_line(
                f"{modality.upper()} · image voxels", data["image_voxels"]
            )
        )
        lines.append(
            _format_quantile_line(
                f"{modality.upper()} · ROI voxels", data["roi_voxels"]
            )
        )
        lines.append(
            _format_quantile_line(
                f"{modality.upper()} · ROI/image", data["roi_fraction_percent"], "%"
            )
        )
        lines.append(
            _format_quantile_line(
                f"{modality.upper()} · physical ROI", data["roi_volume_ml"], "mL"
            )
        )
    lines.extend(
        [
            "",
            "## Manifest finding",
            "",
            f"The source manifest declares image dtype `{summary['source_manifest_image_dtype']}` "
            f"and sizes `{summary['source_manifest_sizes']}`, whereas the observed image dtype "
            f"set is `{summary['actual_image_dtypes']}` and matrix dimensions vary by subject. "
            "The size values are modality surrogates and must not be used in scaling models. "
            "The source file hashes do match the manifest.",
            "",
            "## Benchmark use",
            "",
            "Use all 153 fixed image–mask pairs as a real-world variation pillar. Do not "
            "rescale them or assign equivalent cube sizes. Preserve subject identity so CT, "
            "MRI, and PET observations from the same patient remain a paired block. Model "
            "runtime against observed image voxels, ROI voxels, bounding-box voxels, topology, "
            "spacing, and modality; interpret this as association across heterogeneous clinical "
            "cases, not a controlled scaling experiment.",
            "",
            "## Provenance limitation",
            "",
            "The local manifest binds the copied files by SHA-256 but does not preserve an "
            "upstream release identifier, licence text, or source-tree checksum. These must be "
            "added before the cohort is redistributed or described as a frozen publication "
            "release.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = _manifest_index(manifest)
    pairs = discover_pairs(dataset_root)
    rows = [
        inspect_pair(
            dataset_root,
            pair,
            manifest_files,
            verify_hashes=verify_hashes,
        )
        for pair in pairs
    ]
    summary = summarise_audit(
        rows,
        manifest,
        manifest_sha256=sha256_file(manifest_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "cases.csv", rows)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_bytes(
        output_dir / "summary.md",
        render_markdown(summary).encode("utf-8"),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the IBSI 2 Phase 3 STS image-mask cohort without modifying it."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-hash-verification", action="store_true")
    args = parser.parse_args()
    summary = audit_dataset(
        args.dataset_root,
        args.output_dir,
        verify_hashes=not args.skip_hash_verification,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

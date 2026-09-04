from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from bench.dataset_manifest import sha256_file
from bench.ibsi2_phase3_audit import audit_dataset


def _write_fixture(root: Path) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    entries = []
    for index, modality in enumerate(("ct", "mri", "pet"), start=1):
        image_values = np.arange(120, dtype=np.float32).reshape(4, 5, 6) + index
        mask_values = np.zeros((4, 5, 6), dtype=np.uint8)
        mask_values[1:3, 1:4, 2:5] = 1
        image_path = images / f"STS_001_{modality}_image.nii.gz"
        mask_path = masks / f"STS_001_{modality}_mask.nii.gz"
        nib.save(nib.Nifti1Image(image_values, affine), image_path)
        nib.save(nib.Nifti1Image(mask_values, affine), mask_path)
        for kind, path in (("image", image_path), ("mask", mask_path)):
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "kind": kind,
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "image_dtype": "int16",
                "sizes": [50, 115, 130],
                "files": entries,
            }
        ),
        encoding="utf-8",
    )


def test_audit_dataset_builds_complete_subject_block(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_fixture(source)

    summary = audit_dataset(source, output)

    assert summary["subjects"] == 1
    assert summary["paired_cases"] == 3
    assert summary["complete_subject_blocks"] == 1
    assert summary["incomplete_subject_blocks"] == {}
    assert summary["all_manifest_hashes_match"] is True
    assert summary["actual_image_dtypes"] == ["float32"]
    assert summary["source_manifest_dtype_matches"] is False
    assert summary["modality_summary"]["ct"]["roi_voxels"]["median"] == 18
    assert (output / "cases.csv").is_file()
    assert (output / "summary.json").is_file()
    assert "3D image–mask cases: **3**" in (output / "summary.md").read_text(
        encoding="utf-8"
    )

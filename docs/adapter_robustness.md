# Untimed adapter robustness checks

The robustness diagnostic exercises all supported benchmark workloads on small,
deterministic inputs. It compares each native public extraction path with the
prepared calculation endpoint, then repeats the prepared calculation to detect
retained state. It does not collect adapter timings, invoke calibration, write a
benchmark ledger, or establish independent IBSI accuracy.

Run it directly from the checkout containing the locked native environments:

```text
python -m scripts.qualify_adapter_robustness --output-dir artifacts/robustness-new
```

The output directory must be new. `--profiles medimage mirp pyradiomics zrad`
restricts the run to the four frozen competitors. The unified
[release qualification command](unified_release_qualification.md) includes this
diagnostic automatically. `--timeout` is a safety watchdog for each worker, not
a runtime measurement or a benchmark timeout classification.

## Inputs and scope

Ten synthetic fixtures include CT-like values crossing zero, positive MRI-like
and PET-like intensities, anisotropic spacing, an oblique orientation and nonzero
origin, disconnected components, a mask touching the image boundary, an
eight-voxel ROI, a constant ROI, a single voxel, and an empty ROI. These are
intensity and geometry challenges under the shared extraction configuration;
they do not validate modality-specific clinical preprocessing, PET SUV
conversion, DICOM ingestion, or entire patient images.

To add prepared CT, MRI and PET source inputs:

```text
python -m scripts.qualify_adapter_robustness --output-dir artifacts/robustness-with-crops --workspace-root PATH_TO_VALIDATED_WORKSPACE
```

The workspace must contain `ibsi2_phase3/manifest.json` and its referenced image
and mask files. The diagnostic verifies their declared SHA-256 hashes and
matching geometry. It chooses the first case by case ID in each modality and
crops at most 9 × 9 × 9 voxels around an actual ROI voxel nearest its centroid.
If that crop lacks background or sufficient ROI voxels, it selects a suitable
ROI boundary voxel in increasing distance from the centroid. A crop must contain
at least 32 ROI voxels and a background label, which PyRadiomics requires. It
never invents background padding or changes source mask values.
The crop affine preserves the source's world coordinates. Reports retain the
source paths and hashes, manifest hash, crop bounds, saved affine, and derived
input hashes. The source manifest's clinical provenance is not independently
verified, and a crop never counts as whole-case validation.

Each image has shared FBN32 diagnostic levels derived from its ROI. These levels
exercise the texture and IVH calculation paths; they do not replace the
manuscript's modality-specific IVH preparation.

## Outcome interpretation

Normal inputs must produce the complete baseline feature inventory, all finite
values, agreement with the public path, and agreement on repeated calculation.
Any missing feature, mismatch, unexpected exception or watchdog interruption
fails the diagnostic.

Degenerate inputs have separate outcomes:

| Outcome | Meaning |
| --- | --- |
| `characterized_finite` | Public and prepared outputs agree; this is a reported native convention, not independent mathematical validation. |
| `characterized_undefined` | The paths agree on explicitly recorded NaN or infinity classifications. |
| `characterized_empty` | Both paths return no features. |
| `characterized_rejection` | An observed input rejection is recorded. Public/prepared equivalence was not established. |
| `needs_review` | A scoped native defect or a finite result for an undefined spatial ROI is preserved as a scientific finding. |
| `failed` | A calculation mismatch, unrecognized exception, incomplete evidence, or another diagnostic failure. |

The top-level status remains `needs_review` when any native finding exists;
otherwise completed checks use `qualified_with_degenerate_limits`. Neither
status asserts universal input support. `complete` means every requested check
was attempted. Exit zero means the diagnostic completed without an unexpected
failure; inspect status and `review_findings` before claiming readiness. The
unified command carries review findings into its release assessment.

Saved successful or reviewable reports are revalidated against the exact
fixture inventory, input hashes, source hashes, native environment locks,
feature sets, outcomes, interpreter identity and requested adapters. Revalidating
a saved report does not independently reverify a currently installed runtime;
the unified command performs fresh environment verification.

## Observed native limits

The initial Windows synthetic run with the pinned versions established these
specific behaviors; fresh reports remain authoritative for each release and
platform:

- MEDimage 0.9.8 raises an indexing error in the histogram gradient for constant
  and single-voxel texture inputs, and a covariance/eigendecomposition error
  for single-voxel morphology. These are retained as `needs_review` native
  errors, not converted to successful extractions.
- MIRP 2.7.0 returns finite Moran's I and Geary's C for the constant ROI, although
  their zero-variance formulas are undefined. Public/prepared agreement does
  not resolve this convention; it remains a `needs_review` finding. Its empty
  ROI loader produces no usable image and is recorded as a rejection.
- Z-Rad 26.8.0 explicitly rejects small bounding boxes, empty 3D masks, and
  several constant-input calculations through its native `DataStructureError`.
  These domain rejections are characterized separately from feature passes.
- Pictologics 0.7.1 can return a zero voxel-counting volume for an empty ROI.
  That single well-defined value is allowed without treating other empty-ROI
  finite values as valid.

The completed Windows run on 2026-10-18 included all 13 fixtures: six normal
synthetic fixtures, three real-source crops, and four degenerate fixtures. All
243 supported normal fixture/workload combinations passed public/prepared/repeat
agreement. The real CT and MRI crops each contained 438 ROI voxels, and PET
contained 452, within 9 × 9 × 9 source crops. No mask padding was added.

| Adapter | Normal checks passed | Degenerate outcomes | Unsupported combinations |
| --- | ---: | --- | ---: |
| MEDimage 0.9.8 | 54 | 8 finite conventions, 13 rejections, 3 review findings | 0 |
| MIRP 2.7.0 | 54 | 11 finite conventions, 12 rejections, 1 review finding | 0 |
| PyRadiomics 3.1.0 | 27 | 6 finite conventions, 6 rejections | 39 |
| Z-Rad 26.8.0 | 54 | 3 finite conventions, 21 rejections | 0 |
| Pictologics 0.7.1 | 54 | 15 finite conventions, 9 rejections | 0 |

There were 390 fixture/workload combinations in total. These counts describe
execution and within-adapter agreement, not cross-adapter numerical equality or
independent accuracy. The native review findings are the four listed above.

Native adapter algorithms and published values are not changed by this
diagnostic. macOS results must come from their own completed runs; Windows
evidence cannot substitute for them.

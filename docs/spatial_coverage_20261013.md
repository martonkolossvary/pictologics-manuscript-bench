# Untimed calculation and spatial review, 2026-10-13

Final timing is deferred pending the final Pictologics version. The Windows
0.7.1 screen stopped cooperatively with 1,291 measured processes preserved and
one interrupted task. These remain exploratory observations of their recorded
version. No stable-adapter timing endpoint or package version changed in this review.

## Calculation endpoints and feature inventory

On Windows AMD64, all 21 supported workloads of the four stable adapters returned
the same feature names and values as their public native extraction paths
(MEDimage exposes family extractors). Repeated prepared calculations also matched.
These direct calls use no clock, calibration or benchmark ledger. The small
anisotropic fixture qualifies the calculation boundary, not every possible input
or large-case resource feasibility. The optional Pictologics qualification also
passed all six prepared/unprepared workload and repeat checks.

| Adapter | Morphology | Spatial | Local intensity | Intensity | Texture | IVH |
|---|---:|---:|---:|---:|---:|---:|
| Pictologics 0.7.1 | 27 | 2 | 2 | 18 | 118 | 7 |
| MEDimage 0.9.8 | 27 | 2 | 2 | 18 | 117 | 7 |
| MIRP 2.7.0 | 30 | 2 | 2 | 22 | 119 | 15 |
| PyRadiomics 3.1.0 | 17 | unsupported | unsupported | 16 | 77 | unsupported |
| Z-Rad 26.8.0 | 23 | 2 | 2 | 18 | 118 | 6 |

These are finite native outputs, not counts of distinct standardized definitions.
The new prepared/unprepared checks reproduced Pictologics' existing smoke inventory
as well as qualifying the four stable adapters. Grouped texture retains legitimate
native sharing. Report output/IBSI coverage with timing; these different native
workloads do not establish an equal-feature speed comparison.

## Spatial features

Pictologics exposes two global spatial autocorrelation features:

- **Moran's I, N365:** `morans_i_index_N365`.
- **Geary's C, NPT7:** `gearys_c_measure_NPT7`.

Both use all ROI pairs with inverse Euclidean distance in physical units. The
FFT path evaluates the same definitions using padded convolutions. It does not
introduce voxel sampling, neighbourhood truncation, local Moran statistics or
additional features. MEDimage, MIRP and Z-Rad expose both; MIRP explicitly uses
`no_approximation=True`. PyRadiomics remains unsupported.

Local intensity peak (VJGA) and global intensity peak (0F91) are a separate
two-feature workload. Empty/single-voxel ROIs or constant intensity make the two
Pictologics spatial statistics undefined: both keys contain NaN, not valid finite
measurements. Disabling spatial extraction returns an empty mapping.

The independent isotropic/anisotropic fixture audit exercised both Pictologics
branches, its memory fallback and degenerate inputs. Pairwise/FFT errors were at
most **3.33e-16**; Z-Rad agreed to float64 precision. MEDimage/MIRP retain float32
moments, with observed errors at most **2.52e-9**. Their explicit check tolerances
are absolute 1e-7/relative 1e-6; Pictologics/Z-Rad use 1e-10/1e-9. These are
mathematical checks, not performance observations.

[Reviewed upstream implementation](https://github.com/martonkolossvary/pictologics/blob/f2fdb963b17b483f339ff547c23fa519bf4850b9/pictologics/features/intensity.py).
Upstream main and release 0.7.1 pointed to that commit at review time.

## Predicted spatial resource limits

Pictologics 0.7.1 chooses FFT when `N*N >= 100*grid_points`. Each padded dimension
is `next_fast_len(2*bbox_dimension-1)` and its estimated allocation is
`32*grid_points` bytes. Above the smaller of 16 GiB and half physical RAM, it
falls back to exact quadratic pairs. On Windows, unavailable `os.sysconf` causes
an 8 GiB RAM assumption: **the FFT cap is 4 GiB even on this 32 GiB host**.

The prepared Pillar 1 masks give these source-policy predictions without spatial
extraction:

| Image size | M1/M2 FFT estimate | M3/M4 FFT estimate | Windows selection |
|---|---:|---:|---|
| 256 cubed | 0.93 GiB | 2.06 / 2.01 GiB | FFT all |
| 320 cubed | 1.85 GiB | 4.02 / 4.00 GiB (both above 4) | FFT M1/M2; pairs M3/M4 |
| 384 cubed | 3.13 GiB | 6.95 GiB | FFT M1/M2; pairs M3/M4 |
| 512 cubed | 7.42 GiB | 16.48 / 16.09 GiB | Pairs all |

These are predictions, not observed memory peaks or timeouts. Even a 16 GiB FFT
cap selects pairs for 512-cubed M3/M4. The separate 12 GiB benchmark worker cap
and actual available RAM also apply. Do not override only Pictologics' native
algorithm to improve a comparison. Recheck the final release: feature support
does not imply completion inside the resource budget.

## IBSI 1 findings

All family processes completed on the official 3D-merged digital phantom, with
174 instances and 169 standardized checks per adapter.

| Adapter | Passed / evaluated standardized | Unsupported standardized |
|---|---:|---:|
| Pictologics 0.7.1 | 169 / 169 | 0 |
| MEDimage 0.9.8 | 168 / 168 | 1 |
| MIRP 2.7.0 | 169 / 169 | 0 |
| PyRadiomics 3.1.0 | 107 / 108 | 61 |
| Z-Rad 26.8.0 | 165 / 169 | 0 |

MEDimage lacks dependence count percentage (6XV8). The five numerical failures
were independently reproduced from the native formulas:

- PyRadiomics least axis length (7J51): 8.47810949 versus 8.54, tolerance 0.05.
  Population covariance divides by 74; sample covariance divides by 73 and
  gives 8.53598122. Retain its native formula in the benchmark.
- Z-Rad global intensity peak (0F91): 1.39024390 versus 3.1. Its global sphere
  convolution zero-pads beyond image edges and retains the full-kernel divisor.
  Normalizing by the in-image neighbourhood gives 3.10344828.
- Z-Rad histogram gradients omit empty interior levels. Counts `[50,1,16,7]`
  at levels `[1,3,4,6]` yield maximum gradient 3 at level 4 and minimum -49.
  Including empty levels gives the references 8 at level 3 and -50
  (12CE, 8E6O, VQB3).

Keep these failures in accuracy tables. They permit a disclosed native-workload
timing comparison, but prevent a blanket numerical-equivalence claim.

## IBSI 2 preparation

Phase 1 passed every evaluable supported standardized map: MEDimage 27/27,
MIRP 32/32, PyRadiomics 1/1 and Z-Rad 26/26. Each adapter has 33 standardized
definitions and 36 defined tests; MIRP also produced one nonstandardized map.
Unsupported tests remain visible in the full denominator.

Two preparation defects were fixed: stable-only Phase 2 generation now records
Pictologics as the separate common-preprocessing dependency; Pictologics LPS
geometry is converted back to NIfTI RAS when saving images. The voxel arrays are
unchanged by that world-coordinate conversion. An early unresampled-grid check
and an oblique/nonzero-origin regression reject mirrored geometry. Earlier
diagnostic bundles remain separate from regenerated evidence.

The regenerated Phase 2 evidence completed without execution or numerical errors:

| Adapter | Passed / evaluated standardized | Unsupported standardized |
|---|---:|---:|
| MEDimage | 252 / 252 | 71 |
| MIRP | 323 / 323 | 0 |
| PyRadiomics | 30 / 30 | 293 |
| Z-Rad | 252 / 252 | 71 |

Each adapter has 396 designed Phase 2 instances and 323 standardized checks.
MIRP also has one finite coverage-only output with no consensus reference.
Phase 2 tests filter response-map statistics, not the timed texture workload;
PyRadiomics' limited exact native filtering coverage must not be mistaken for
its broader radiomic-feature coverage. Future Pictologics releases need their own
versioned IBSI 2 evidence; this stable-four bundle freezes 0.7.1 only as the shared
preprocessing dependency.

Follow [staged readiness](final_version_readiness.md) to freeze stable evidence,
requalify future Pictologics and collect final timings in a fresh common run.
Native Windows/macOS CI now includes untimed endpoint and spatial checks; the
intended publication Mac still needs accuracy and host qualification. Raw local
qualification artifacts remain outside Git.

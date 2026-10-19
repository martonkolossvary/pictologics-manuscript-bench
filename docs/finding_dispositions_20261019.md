# Decisions on outstanding findings, 2026-10-19

The reporting decisions below resolve how the known findings enter the study.
They do not repair upstream packages or turn failed numerical checks into
passes. All timing remains paused. Keep the four competitors at their pinned
versions and qualify the final Pictologics release in a separate environment.

## Accuracy: retain five failed standardized checks

The completed Windows full qualification at source `80408fe` reproduced these
IBSI 1 digital-phantom failures. Values and tolerances come from its original
comparison table; the existing IBSI workbook rounding rule remains in force.

| ID | Adapter/version | Feature | Observed | Reference | Tolerance | Disposition |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A1 | PyRadiomics 3.1.0 | Least axis length, `7J51` | 8.4781094932 | 8.54 | 0.05 | Native covariance convention differs; retain failure. |
| A2 | Z-Rad 26.8.0 | Global intensity peak, `0F91` | 1.3902439024 | 3.1 | 0 | Native image-boundary normalization differs; retain failure. |
| A3 | Z-Rad 26.8.0 | Maximum histogram gradient, `12CE` | 3 | 8 | 0 | Native histogram omits unoccupied interior levels; retain failure. |
| A4 | Z-Rad 26.8.0 | Intensity at maximum histogram gradient, `8E6O` | 4 | 3 | 0 | Same histogram defect as A3; retain failure. |
| A5 | Z-Rad 26.8.0 | Minimum histogram gradient, `VQB3` | -49 | -50 | 0 | Same histogram defect as A3; retain failure. |

PyRadiomics divides the coordinate covariance by the 74 ROI voxels. Dividing
by 73 instead reproduces 8.53598122. Z-Rad's global convolution keeps the full
sphere divisor when zero-padding beyond the image; using only in-image sphere
voxels reproduces 3.10344828. Its histogram uses counts `[50, 1, 16, 7]` at
observed levels `[1, 3, 4, 6]`; including the empty interior bins reproduces the
three reference gradients. These are explanations of the observed differences,
not corrections applied to benchmark outputs. The
[IBSI feature definitions](https://ibsi.readthedocs.io/en/master/03_Image_features.html)
define the reference calculation and the complete histogram level range.

The installed Windows sources reviewed were `radiomics/shape.py` (SHA-256
`aba403f94462ab9aab9996bd6b2589b466384c3137539230b81959e17c67a8d8`)
and `zrad/radiomics/intensity.py` (SHA-256
`b1a41d8bf6ab7ad67df895113036946ca7377dcc5e034638e748e84f62c02928`).
These findings are version-specific. macOS CI checks execution and bounded
correctness; a full IBSI assessment on the intended Mac is still required.
PyRadiomics' macOS source build is the pinned 3.1.0 tag commit `6a761c4` but
reports distribution metadata `3.0.1a1`, as recorded in its profile.

Report all five rows as evaluated numerical failures, retaining their original
values, tolerance policy and denominators. PyRadiomics remains 107/108 evaluated
and 107/169 standardized checks; Z-Rad remains 165/169 for both. Keep unsupported
features separate from failures. Do not tune tolerances, drop these rows, or
modify the stable native packages to improve their scores.

The future native-workload comparison may retain these implementations with
explicit accuracy annotations: A1 affects morphology, A2 local intensity, and
A3–A5 texture. Such timings describe each declared native output surface.
They cannot support a claim of identical outputs or equivalent-feature speedup.
Do not divide grouped runtimes by feature counts, subtract a failed feature's
estimated cost, or silently remove it from a group. Any separate matched-feature
experiment requires its own prespecified endpoint and fresh qualification.

## Degenerate inputs: retain four scientific findings

The Windows full baseline and both native CI platforms at `80408fe` showed the
same findings. Public/prepared/repeated agreement on normal fixtures does not
erase these limits.

| ID | Adapter/version | Input and workload | Observation | Reporting decision |
| --- | --- | --- | --- | --- |
| D1 | MEDimage 0.9.8 | Constant ROI, texture | Histogram gradient raises an indexing error. | Preserve `needs_review` / `native_degenerate_error`. |
| D2 | MEDimage 0.9.8 | Single voxel, texture | Same indexing error. | Preserve `needs_review` / `native_degenerate_error`. |
| D3 | MEDimage 0.9.8 | Single voxel, morphology | Covariance/eigendecomposition receives a zero-dimensional array. | Preserve `needs_review` / `native_degenerate_error`. |
| D4 | MIRP 2.7.0 | Constant ROI, spatial autocorrelation | Moran's I and Geary's C both return 1, consistently across the tested paths. | Preserve `needs_review` / `finite_undefined_spatial`; zero-variance formulas do not establish these as accurate values. |

Retain raw native values and exceptions. An error is neither an extraction
success nor a zero-duration observation. A finite convention on an undefined
ROI is not a numerical pass. Do not impute missing outputs or exclude cases
after seeing an adapter's behavior. If a future corpus contains these inputs,
report the affected case/workload coverage explicitly and apply any common
input-eligibility rule before starting the experiment.

The MEDimage fixes, if pursued upstream, should handle one-bin histogram and
single-voxel covariance inputs explicitly, with regression tests distinguishing
undefined features from valid values. MIRP should document or revise its
constant-ROI spatial convention and test the zero-variance boundary. These are
prepared remediation recommendations; no upstream issue was submitted.
Requalify any changed package in a new environment. The frozen competitors'
known limits remain reportable without changing their algorithms.

### Exposure in the prepared benchmark inputs

An input-only scan checked all 283 frozen cases and matched all 962 unique input
file hashes to the manifests. It recomputed counts/bounding boxes for 203 unique
masks and raw/texture ROI statistics for every case. The five inspected triggers
(empty ROI, single voxel, bounding-box dimension below three, constant raw ROI,
constant texture ROI) were absent. The smallest ROI contains 239 voxels; the
minimum bounding-box dimension is six and minimum occupied texture level count
is 16. Thus the tested constant/single-voxel triggers for D1–D4 are absent from
this specific prepared cohort. No case was removed.

However, 98 texture ROIs contain unoccupied interior levels: 42 Pillar 1 cases,
all 10 Pillar 2 cases and 46 real-world cases. This valid input condition is
relevant to A3–A5. It is not a degenerate-input exclusion and does not establish
that every one of those 98 cases produces a numerical mismatch. Retain the
Z-Rad histogram caveat for the cohort as well as the digital phantom.

This scan ran no adapters. It does not establish feature accuracy, finite
feature outputs, covariance rank, every possible native degeneracy, or runtime
feasibility. IVH files were hash-checked but their voxels were not rescanned.
Changed inputs require another audit. The local evidence contains per-case
statistics, manifest identities, all input hashes and the read-only audit code.

## Pictologics: keep RAM detection open for release acceptance

| ID | Adapter/version | Observation | Disposition |
| --- | --- | --- | --- |
| P1 | Pictologics 0.7.1 | On the Windows host with 33,396,543,488 bytes physical RAM, the native FFT cap is 4,294,967,296 bytes instead of the policy's 16,698,271,744 bytes. | Pending upstream resolution and fresh release qualification. |

The Unix-only `os.sysconf` path falls back to an assumed 8 GiB machine. This
affects algorithm selection and possible feasibility, not the definitions of
Moran's I (`N365`) and Geary's C (`NPT7`). Both remain supported exact spatial
statistics. Do not raise the cap only for Pictologics in the benchmark.

The [existing feedback package](pictologics_release_feedback.md) specifies the
cross-platform RAM query, conservative fallback, half-RAM/16-GiB boundaries,
forced pairwise/FFT agreement and final-release acceptance checks. Keep P1 open
until those checks succeed under the reviewed release policy on Windows and
the intended Mac. A documented disposition alone is not acceptance.

## Diagnostic correction

The robustness qualifier previously recognized any `ValueError`, some other
numerical exception classes, and runtime errors containing generic words such
as `ROI` or `mask` as input rejections. That could hide an unrelated bug on a
degenerate fixture. Recognition now requires the observed adapter, supported
workload, fixture, exception class, exception module and exact message. It is
limited to preparation or initial calculation; comparison and repeated-call
errors fail. Saved reports must satisfy the same rule.

Replaying the recorded evidence through this stricter classifier preserves all
61 characterized rejections and the three MEDimage error findings. MIRP's
finite spatial finding is unchanged. Newly encountered errors require review;
they are not automatically added to the recognized observations. A changed
package's message or feature order can therefore fail qualification pending
inspection. This is intentional evidence validation, not an upstream API rule.

A fresh Windows run of the revised checker completed all 390 combinations:
243 supported normal checks passed, 39 combinations were unsupported, and the
degenerate results retained 43 finite conventions, 61 rejections and four review
findings. There were no unexpected failures. The repository suite passed
610 tests and 140 subtests; regression tests cover broad exception masking,
wrong fixture/workload/module and tampered saved rejection evidence.

Historical qualification bundles remain immutable and retain their original
source identity. The fresh robustness evidence belongs to the revised checker;
it does not relabel the earlier full baseline as having run under new code.
The [readiness report](untimed_readiness_20261018.md) continues to state the full
baseline's accuracy and coverage, including all outstanding findings.

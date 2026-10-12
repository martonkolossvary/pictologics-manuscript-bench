# Adapter review, 2026-10-12

The harness is suitable for comparing **native workload implementations**, with
the following conditions and limits. It does not establish equal feature coverage
or numerical equivalence across packages merely because a timing succeeds.

## Corrected issues

- Pictologics 0.7.1 is pinned to its verified wheel hash, with complete native
  environment locks. The adapter now calls `calculate_all_texture_features`
  with explicit family selection and the correct GLDZM morphology mask. The
  old full-matrix path bypassed new compact/shared/sparse optimizations and
  would not represent the current native implementation.
- Pictologics' cached neighbour and zone offsets are reset outside each timed
  call, so constructing them remains inside calculation time. The existing
  local-intensity sphere-cache policy is retained. JIT machine-code caches stay
  warm under the common warmup protocol.
- `PICTOLOGICS_NUM_THREADS` is explicitly set alongside Numba, BLAS, OpenMP and
  ITK controls. An inherited Pictologics override cannot silently reduce only
  that adapter's requested thread allowance.
- PyRadiomics retains separate morphology and resegmented intensity bounds.
  The previous shared bounding box could crop away morphology-mask voxels when
  intensity resegmentation removed boundary voxels. A mask-volume regression
  covers both `resegmentShape` settings.
- Windows interpreter discovery accepts the standard `py` launcher as well
  as uv-managed interpreters and versioned executable names. Native adapter
  smoke jobs now accompany controller CI on Windows and macOS arm64.

## Existing controls retained

All adapters consume the same checksum-bound raw, FBN32 texture and appropriate
IVH representations. Preprocessing and loading stay outside the calculation
clock; native matrix, mesh and neighbourhood construction stay inside. Warmup,
calibration, finite-value checks and repeat-equivalence checks apply to every
adapter. A task timeout is whole-worker censoring, never a 900-second per-call
timing. Memory estimates, actual limits and unsupported features are distinct
outcomes. Resource cutoffs apply only within the recorded case/scaling series.

Use rotated adapter order to reduce position/thermal bias. Do not compare
timings obtained while multiple benchmark processes or unrelated heavy jobs
compete for the host. A common requested thread allowance does not mean every
native implementation actually uses all threads.

## Interpretation and remaining qualification

The six partitions are morphology, spatial autocorrelation, local intensity,
intensity, texture and IVH. Texture remains grouped to preserve legitimate
native sharing, including Pictologics' shared GLSZM/GLDZM traversal. Native
feature inventories differ: PyRadiomics has no GLDZM, IVH, local-intensity
or spatial-autocorrelation workload here. Include output counts/IBSI coverage
with runtimes, and compare only matched cases/workloads. An equal-feature
speedup claim needs an independently specified common-feature experiment.

Pictologics 0.7.1 spatial autocorrelation selects exact pairwise or exact FFT
algorithms using native size/memory rules, including a lower FFT allocation cap
on Windows. This is native platform behavior, not a sampling approximation;
report each operating system separately. Memory exclusions can also differ with
available RAM and must remain visible in the denominator.

Native Windows 0.7.1 checks returned finite, repeat-equivalent results for all
six workloads (27/2/2/18/118/7 features, 174 total). Prepared and untimed paths
matched. The 95 grouped texture outputs matched the full-matrix reference exactly
on a small nondegenerate fixture. Against an independent anisotropic inverse-
distance NumPy formula, the spatial pair branch differed by at most 5.2e-18 and
the forced FFT branch by at most 2.22e-16. These are focused API/math regressions,
not a replacement for the complete IBSI compliance suite.

All five locked Windows environments subsequently passed every supported
untimed workload on a small nondegenerate FBN32 fixture: 27 adapter/workload
checks, with feature counts matching the frozen workload inventories. The full
controller suite passed 421 tests and 108 subtests (18 optional tests skipped),
and Ruff and the publication-source audit passed. The copied 283-case workspace
also passed deep input/provenance validation. Long-path test failures in this
deep checkout were resolved by using a short temporary root; deployment should
use short local paths as documented in the handoff guide.

macOS arm64 dependency resolution was checked, but native execution of this
upgrade must be established by CI and by qualification on the intended Mac.
The repository does not currently provide qualified Intel-macOS or Windows-ARM64
locks. No cross-platform speed or compliance conclusion should be inferred from
Windows-only smoke evidence.

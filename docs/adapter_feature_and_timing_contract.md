# Adapter feature and timing contract

The frozen endpoint is
`configs/benchmark/calculation_only_workload.json`. Its checksum is included
in every task and RunSpec.

## Feature surface

Five adapters are isolated and version-pinned through `configs/adapters/`.
The scheduler uses six native calculation workloads:

| Workload | Requested families | Representation |
|---|---|---|
| morphology | morphology excluding Moran's I and Geary's C | raw |
| spatial_autocorrelation | Moran's I and Geary's C | raw |
| local_intensity | local and global intensity peaks | raw |
| intensity | first-order intensity | raw |
| texture | histogram, GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM | FBN32 identity |
| ivh | IVH | frozen IVH identity |

Scheduling every texture family in a separate process would repeatedly charge
packages for work they deliberately share. The texture group preserves native
multi-family paths while keeping the timed task on one stored image. Spatial
autocorrelation is separated because it is a quadratic pairwise calculation;
local intensity is separated because it uses a physical spherical
neighbourhood rather than first-order statistics. IVH is independent because
its frozen image differs from the raw intensity image. Reports do not merge
workload runtimes after timing.

The expected current native feature names are checksum-bound in
`reproducibility/contracts/adapter_feature_surface.csv`. The untimed audit
checks input routing, adapter version identity, finite output, and retention of
that surface:

```bash
poetry run python scripts/audit_adapter_feature_surface.py \
  --dataset-dir data/benchmark/pillar1 \
  --case-id CASE_ID \
  --output-dir artifacts/feature-surface-audit \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json
```

This audit calculates features and must be run only when the operator intends
to perform the prebenchmark feature-surface check. It does not produce timing
observations.

## Calculation endpoint

The timed endpoint begins after file loading, hash verification, mask
preparation, resegmentation, stored-representation construction, and
adapter-native preparation. It includes matrix/mesh/neighbourhood construction
and radiomic arithmetic. Result normalisation and serialisation occur after the
timer.

Directional texture aggregation is fixed to IBSI 3D merged. A package may
report unsupported members inside a grouped request; a task is preempted only
when the package supports none of its members. PyRadiomics therefore has no
spatial-autocorrelation, local-intensity, IVH, or GLDZM outputs; the first three
are independent unsupported workloads, while missing GLDZM remains visible in
the grouped texture output count and IBSI coverage. The Pictologics baseline
must be included in every comparative run.

## Native grouping audit

Pictologics 0.5.1 is the clearest reason for grouped timing. Its pinned
`calculate_all_texture_matrices` implementation explicitly builds the selected
GLCM, GLRLM, NGTDM, and NGLDM structures in a shared local pass and performs
GLSZM and GLDZM through a shared zone traversal. The adapter
therefore calls that API once for the complete texture workload and then
derives every selected texture-family output from those matrices.

The other adapters retain their own native boundaries:

- MIRP generates the complete ordered feature-object sequence once and keeps
  adjacent matrix-cache sharing inside the clock.
- Z-Rad receives the complete family list in one `Radiomics.extract_features`
  call after shared ROI preparation.
- PyRadiomics executes all selected native feature classes within one timed
  workload. Its shape constructor performs mesh and diameter calculations, so
  that constructor is deliberately inside every morphology call.
- MEDimage exposes family extractors rather than a stable cross-family public
  call; the adapter times their complete class sequence in one workload after
  shared array preparation.

## Observation policy

Each fresh process performs one untimed warmup followed by three measured
observation windows. Only the first independent post-warmup window may use the
single-window shortcut when its per-call duration is at least 100 ms. Otherwise
3–12 untimed calibration windows must converge; a later slow outlier cannot
bypass this requirement. The latest three normalized estimates must reach CV
<= 5% and max/min <= 1.10. The fastest estimate selects 1–4096 calls with 100%
headroom; every measured window must reach 50 ms. Failure to converge or reach
the minimum is terminal and produces no timing. Preparation and finalization
remain outside every individual calculation clock. Samples are
stored both as raw window totals and normalized per native workload call. The
within-process primary statistic is the median. Three fresh processes are
scheduled per case/workload/adapter, with their median used for the principal
summary.

The finalized warmup output is the per-process reference. Every calibration
and measured call must return exactly the same feature names and shapes, finite
values, and numerically equivalent values at `rtol=1e-9`, `atol=1e-12`.
Equivalence validation is outside the timer; any mismatch fails the task.
The controller also requires the same feature order and equivalent values
across the three fresh-process repeats, records the reference task/repeat in
each payload, and replays these checks whenever a run is resumed.

## Payload validation

The controller verifies adapter identity, distribution/version provenance,
selected workload and requested families, aggregation, representation ID and hashes, discretization
mode, timing-contract metadata, warmup/measured counts, event-stream
completeness, calibration convergence and raw windows, repeated-result
equivalence, finite scalar values, expected feature count, task-boundary power
observations, and controller-side memory metrics. Any mismatch is a failed
task, not a timing observation.

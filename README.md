# Pictologics radiomics benchmark

This is the source-only reproduction package for a three-pillar,
cross-platform radiomics benchmark. It contains code, pinned configuration,
input checksums, environment definitions, tests, and operating instructions.
It intentionally contains no downloaded scientific inputs, generated benchmark
workspace, adapter environments, run ledger, timing output, compliance output,
or manuscript figure.

The benchmark compares Pictologics, PyRadiomics, MIRP, MEDimage, and Z-Rad in
isolated environments. Every timed workload is one canonical IBSI feature
group of one or more families using a manifest-bound stored representation.
The controller is transactional, resumable, checksum-attested, and fail-closed.

## Repository contract

This repository is the first and only public release of this benchmark. Its
own names carry no release-line suffix; there are no previous benchmark paths,
identifiers, migration branches, or compatibility instructions. Numbers remain
only where they are intrinsic to an external standard or dependency, such as
IBSI 2 and PyRadiomics 3.1.0.

The canonical names and paths are:

- workspace: `data/benchmark`
- Pillar 1 profile: `configs/benchmark/pillar1.json`
- Pillar 2 profile: `configs/benchmark/pillar2_a1.json`
- endpoint contract: `configs/benchmark/calculation_only_workload.json`
- input routing: `manifest_harmonized`
- feature-surface contract:
  `reproducibility/contracts/adapter_feature_surface.csv`
- launcher: `scripts/launch_benchmark.py`

Working roots are ignored and forbidden in publication commits: `.venv/`,
`.venvs/`, `data/`, `results/`, `artifacts/`, caches, ledgers, NIfTI inputs,
logs, and adapter payloads. The sole generated exception is a bundle created
and validated below `benchmark-results/`; it contains public reports, not live
SQLite state or scientific input data.

## Benchmark design

| Pillar | Dataset | Cases | Question |
|---|---|---:|---|
| 1 | Controlled CT-inspired 3D phantom, four masks | 120 | Geometry and resolution scaling |
| 2 | Dense whole-anatomy mask | 10 | Large-ROI scaling |
| 3 | IBSI 2 Phase 3 CT/MRI/PET cohort | 153 | Fixed real-world variation |

Each case binds an original image, a binary mask, a mask-specific FBN32
texture image, and a mask-specific IVH image. Representation routing is fixed:

| Native timed workload | Included families | Stored input | Adapter mode |
|---|---|---|---|
| morphology | morphology except Moran's I and Geary's C | original image | raw |
| spatial_autocorrelation | Moran's I and Geary's C | original image | raw |
| local_intensity | local and global intensity peaks | original image | raw |
| intensity | first-order intensity statistics | original image | raw |
| texture | histogram, GLCM, GLRLM, GLSZM, GLDZM, NGTDM, NGLDM | mask-specific FBN32 indices | identity |
| ivh | IVH | mask-specific FBS1 indices for CT/synthetic; FBN1000 indices for MRI/PET | identity |

These are the largest scientifically valid groups that share both a frozen
input representation and a comparable scaling class. Reports retain all six
workloads independently. Spatial autocorrelation is isolated because its
pairwise algorithm is quadratic in ROI voxels and PyRadiomics does not expose
those features. Local-intensity peaks are isolated because their spherical
neighbourhood/convolution is distinct from first-order statistics and is also
not exposed by PyRadiomics. IVH has its own stopwatch because it uses a
different stored image.

The texture workload intentionally remains grouped. In particular,
Pictologics constructs GLSZM and GLDZM in a shared zone traversal; splitting
GLDZM would erase that legitimate native batching advantage. PyRadiomics does
not expose GLDZM, so native output count and IBSI coverage must accompany every
texture runtime comparison.

The controller exposes no per-run discretization controls. Re-binning or
silently substituting another input image is not permitted.

## Prerequisites

- Git
- CPython 3.12 for the controller and the Pictologics, MIRP, and Z-Rad adapters
- CPython 3.10 for MEDimage and CPython 3.11 for the PyRadiomics source build;
  Windows instead uses CPython 3.9 with the official, checksum-pinned 3.1.0
  wheel. These runtimes may be installed system-wide or managed by `uv`.
- Poetry 2.2 or newer, below 3.0
- `uv` 0.8 on Windows; MEDimage's legacy Python metadata cannot be resolved
  correctly by current pip releases
- network access for the first input and environment bootstrap
- enough local SSD space for inputs, environments, staged inputs, and results
- a local, non-synchronised result directory

For benchmark-quality timing, use an otherwise idle machine, AC power, a fixed
power mode, no pending operating-system updates, and no competing workloads.
The power mode is provenance, not an eligibility gate: every observed mode is
accepted and reported so results can be compared or stratified accordingly.

## Complete macOS setup

The checked-in macOS host profile is for the publication host only: Apple M4
Pro, 14 physical/logical CPU cores, 48 GiB RAM, arm64, and AC power. Its actual
Energy Mode is observed at launch and immediately before and after every task.
It does not gate execution or alter the protocol fingerprint, so a stopped run
remains resumable after a mode change; affected tasks and the complete result
are explicitly classified as mixed-mode. A
different Mac requires its own reviewed public-safe host profile; do not edit
the checked-in profile to make a mismatched host pass.

1. Clone and enter the source repository.

   ```bash
   git clone https://github.com/martonkolossvary/pictologics-manuscript-bench.git
   cd pictologics-manuscript-bench
   ```

2. Confirm the clone contains publication source only.

   ```bash
   python3 scripts/audit_repository_contents.py
   ```

3. Install the controller and all five isolated adapter environments.

   ```bash
   ./scripts/setup.sh
   ```

   This runs `poetry sync`, creates or reuses the environments declared in
   `configs/adapters/`, and verifies distribution versions and fingerprints.
   Rebuilding existing environments is explicit:

   ```bash
   poetry run python -m bench.cli env create --force
   poetry run python -m bench.cli env verify
   ```

4. Fetch every external input at its pinned revision and verify every byte.

   ```bash
   poetry run python scripts/bootstrap_reproducibility_inputs.py
   poetry run python scripts/bootstrap_reproducibility_inputs.py --verify-only
   ```

   The source manifest is `reproducibility/inputs/manifest.json`. Downloads and
   upstream checkouts are written below ignored `data/` paths.

5. Generate and deeply validate the three-pillar workspace. This prepares
   images and masks but does not run radiomic calculations.

   ```bash
   poetry run python scripts/prepare_benchmark_workspace.py \
     --ibsi2-source data/ibsi2_validation \
     --output-root data/benchmark \
     --resume

   poetry run python scripts/prepare_benchmark_workspace.py \
     --output-root data/benchmark \
     --validate-only
   ```

   Repeat the `--validate-only` command after pulling or changing benchmark
   source. The generated manifest is schema- and checksum-bound, and the
   launcher rejects stale workspace metadata before any calculation starts.

6. Choose a local result root outside iCloud, Dropbox, OneDrive, Google Drive,
   or another synchronised folder. Qualify the Mac without calculating:

   ```bash
   ./scripts/qualify_benchmark_host.sh \
     /absolute/local/path/pictobench-results \
     configs/benchmark/hosts/mac-m4pro-01.json
   ```

   Qualification runs project metadata checks, Ruff, the full test suite,
   adapter environment verification, result-volume preflight, host checks, and
   controller dry-runs for all pillars. The attestation below
   `data/benchmark/host_attestations/mac-m4pro-01/` must say
   `radiomic_calculation_started: false`.

7. Before a real run, confirm AC power, select the Energy Mode you intend to
   keep for the complete machine run, quiesce background load, defer updates,
   and confirm sufficient result disk space. The launcher records the observed
   Energy Mode label, raw `pmset lowpowermode` value, observation status,
   portable power-mode tag, and any probe errors in the session history and
   in every task record. Each task has live start/end observations with three
   short retries for transiently missing macOS profile output; the probe source
   and attempt count are retained. A change during calculation is tagged
   `mixed-within-task` and reported by QC.
   Energy Mode never blocks execution; if macOS does not report it, the run is
   explicitly tagged `macos-energy-mode-unavailable`. Keep one mode for a
   publication-quality run when practical; if it changes, the report and
   contribution path disclose `mixed-power-modes` rather than hiding the mix.

At this point the macOS environment is ready. No benchmark calculation has
been started.

## Windows setup

Use PowerShell from a local NTFS path:

```powershell
git clone https://github.com/martonkolossvary/pictologics-manuscript-bench.git
Set-Location pictologics-manuscript-bench
python scripts/audit_repository_contents.py
./scripts/setup.ps1
poetry run python scripts/bootstrap_reproducibility_inputs.py
poetry run python scripts/bootstrap_reproducibility_inputs.py --verify-only
poetry run python scripts/prepare_benchmark_workspace.py `
  --ibsi2-source data/ibsi2_validation `
  --output-root data/benchmark --resume
poetry run python scripts/prepare_benchmark_workspace.py `
  --output-root data/benchmark --validate-only
./scripts/qualify_benchmark_host.ps1 `
  -ResultRoot C:\pictobench-results-SOURCE_COMMIT `
  -HostProfile configs/benchmark/hosts/windows-9800x3d-01.json
```

The checked-in Windows profile is specific to the AMD Ryzen 7 9800X3D
qualification host. A different Windows machine needs its own reviewed,
public-safe profile. Use one stable, non-identifying machine ID per host. Keep
workspace and results out of synchronised folders and within the Windows
path-length budget. The setup script checks Python 3.9, 3.10, and 3.12 and
finds a per-user `uv.exe` even when its Scripts directory is not on `PATH`.

## Inspect the plan without calculating

The launcher is print-only unless told otherwise:

```bash
poetry run python scripts/launch_benchmark.py \
  --workspace-root data/benchmark \
  --result-root /absolute/local/path/pictobench-results \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json
```

To invoke every controller in dry-run mode and persist a qualification record:

```bash
poetry run python scripts/launch_benchmark.py \
  --workspace-root data/benchmark \
  --result-root /absolute/local/path/pictobench-results \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json \
  --validate-plans
```

Neither command starts adapter calculations.

## Explicit calculation boundary

This is the first command that starts the benchmark on macOS:

```bash
./scripts/run_benchmark.sh \
  --workspace-root data/benchmark \
  --result-root /absolute/local/path/pictobench-results \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json \
  --execute --confirm CALCULATE
```

The selected machine namespace must be empty for its first run from a source
commit. Trial or interrupted ledgers created with different protocol source
cannot be resumed or mixed with the corrected run; keep them outside the active
result root or remove them before starting. Controller dry-runs and real
execution also require a clean Git worktree so every result names the complete,
committed source that produced it.

The wrapper uses `caffeinate -dimsu`; host preflight requires its live
sleep-prevention assertion. On Windows use:

```powershell
./scripts/run_benchmark.ps1 `
  --workspace-root data/benchmark `
  --result-root C:\pictobench-results-SOURCE_COMMIT `
  --host-profile configs/benchmark/hosts/windows-9800x3d-01.json `
  --execute --confirm CALCULATE
```

The launcher holds `SetThreadExecutionState(ES_CONTINUOUS |
ES_SYSTEM_REQUIRED)` only while calculations are running and releases it on
exit. It records the active power-scheme GUID using a
localization-independent tag plus AC/battery and battery-saver state at launch
and immediately before and after every task. Power-scheme changes remain
visible provenance and do not invalidate a safe resume.

The launcher always supplies `--resume`. Re-running the exact command resumes
the transactional ledgers without repeating committed measured tasks. A
changed immutable protocol or host identity is rejected. To append more fresh
process repeats, supply an absolute repeat horizon above three.

To stop safely, press `Ctrl-C` once and wait for `Benchmark interrupted safely`
and the shell prompt. The launcher forwards the signal to the active controller
and waits for its final checkpoint. Then re-run the identical command to
continue. A terminal closure, power loss, or forced kill can prevent the final
human-readable summary from being refreshed, but the transactional ledger
recovers an in-flight task as interrupted on the identical resume command;
committed measured tasks are not recalculated. A task that ended in `failed`
is preserved in the attempt history and automatically requeued once on the
next explicit resume, so a transient adapter failure cannot permanently poison
the run. Repeated failures remain visible in the ledger and QC report.

## Timing and timeout policy

- One untimed warmup call occurs in every fresh adapter process, followed by
  at least one independent untimed post-warmup verification call. The first
  verification window is sufficient by itself only when its per-call duration
  already exceeds the 100 ms headroom target. A later slow outlier cannot take
  this shortcut. This prevents JIT compilation in the warmup from hiding a
  much faster steady-state call.
- If the verified native workload call is shorter than the 100 ms headroom target,
  3–12 untimed calibration windows must converge: the latest three normalized
  call times must have CV at most 5% and max/min at most 1.10. The batch is
  sized from the fastest observed call with 100% headroom. Three measured
  windows then use 1–4096 calls and each must last at least 50 ms; a task that
  cannot stabilize or meet the minimum fails visibly instead of publishing a
  noisy timing. Raw calibration windows and call counts are retained.
- Every warmup, calibration, and measured native call must return identical
  feature names and shapes, finite values, and numerically equivalent values
  (`rtol=1e-9`, `atol=1e-12`). These checks run outside the calculation clock;
  any mismatch fails the task. The controller applies the same requirement
  across all three fresh-process repeats and revalidates it on resume.
- Three fresh processes run per eligible case/workload/adapter; reports use
  their median.
- Adapter tasks run one at a time. Every task receives the host's full physical
  core count through the OpenMP, BLAS, Numba, NumExpr, ITK, and BLIS controls;
  the exact requested count and environment are immutable run provenance.
- Loading, checksum verification, mask preparation, resegmentation,
  representation construction, and result serialisation are outside the timer.
- Matrix, mesh, neighbourhood construction, and feature arithmetic are inside.
- Startup/warmup and calculation receive separate timeout clocks. A timeout is
  a censored terminal outcome with its lower bound and completed samples; it is
  never converted into a duration.
- The controller terminates the adapter process tree, waits the configured
  grace period, and escalates if needed.
- No adapter is skipped from a predicted runtime or a relative-speed estimate.
  The first task that reaches the fixed timeout is retained as a censored
  observation. Subsequent tasks for the same adapter, workload, mask/subject
  scaling series, and stored representation are not launched when the image is
  strictly larger; they receive the explicit terminal status
  `skipped_timeout_cutoff`. Equal-sized repeats, different masks or
  configurations, and unrelated real-world cases still run.
- Memory estimation is advisory in the publication launcher. It separates fixed
  process overhead from input-dependent growth and retains the quadratic Z-Rad
  spatial-autocorrelation diagnostic, but an estimate above the live budget is
  only tagged and never prevents a calculation. An estimate is not measured
  memory.
- Power state is task provenance, not eligibility. The controller probes at
  both task boundaries and QC identifies unavailable observations or a mode
  transition; results can therefore be stratified without blocking execution.
- Progress ETA uses complete ledger task turnaround and learns voxel-scaling
  separately by adapter, workload, and synthetic series, while separating a
  non-negative fixed startup term once three sizes are observed. It prefers an
  earlier repeat of the identical task, reports an extrapolation range, honors
  timeout bounds and persisted timeout cutoffs, and does not publish a numeric
  all-pillar ETA from unseen future tasks. Total voxel count already represents
  the three-dimensional volume and is never cubed again.
- QC reports failures as errors and censored timeouts, timeout-cutoff skips,
  interruptions, timing instability, adapter stderr, incomplete phase RSS,
  unavailable task power, and empty feature results with counts by adapter and
  workload. Expected censoring remains a warning rather than a measured result.

Runtime ratios use exact case/workload/repeat pairs. Native output counts and
IBSI coverage accompany runtime; runtime is not divided by or weighted by
feature count because shared matrix construction does not have a linear
per-feature cost.

## Package completed results for GitHub

Keep the three live SQLite ledgers and payload directories on local storage.
After all pillars finish, create the Git-ready bundle with one command:

```bash
poetry run python scripts/package_benchmark_submission.py \
  --result-root /absolute/local/path/pictobench-results \
  --machine-id MACHINE_ID
```

The packager regenerates all three reports from the authoritative ledgers,
verifies every measured payload and report checksum, requires publication
attestation and one source commit/machine identity, and emits only public-safe
artifacts. `task_observations.csv` retains task identities, terminal statuses,
memory/timeout fields, and every raw wall/CPU timing sample, but excludes local
paths, stderr, and raw error messages.

The destination has this form:

```text
benchmark-results/<os>-<architecture>/
  <machine-id>--<cpu-model>--<power-mode-tag>--<date>--<digest>/
```

The power-mode tag makes this important source of timing variation visible in
GitHub paths; the manifest retains the label, raw value, observation status,
and probe diagnostics.

Validate it before committing:

```bash
poetry run python scripts/package_benchmark_submission.py --validate
python3 scripts/audit_repository_contents.py
git status --short
```

Commit the new bundle and open a pull request. Do not commit `results/`, copy
or merge ledgers between hosts, or edit generated bundle files. We welcome
repeatable submissions from other operating systems, architectures, processor
families, and memory configurations; see `benchmark-results/README.md` for the
naming contract and pull-request checklist.

A report is publication-attested only when the run is terminal, its immutable
fingerprint is present, dataset hashes were verified, and every measured
payload passes ledger verification.

IBSI compliance is an independent accuracy workflow. See
`docs/ibsi_compliance.md`; do not pool compliance runtime with performance.

## Development validation

These checks do not launch adapter calculations:

```bash
poetry check
poetry run ruff check .
poetry run pytest -q
poetry run python -m bench.cli env verify
python3 scripts/audit_repository_contents.py
bash -n scripts/*.sh
```

GitHub Actions checks the controller on macOS and Windows and shell syntax on
Linux. It does not fetch scientific inputs or execute the benchmark.

## Method details

- `docs/public_benchmark_data.md` — input bootstrap and workspace construction
- `docs/three_pillar_benchmark_design.md` — datasets and interpretation
- `docs/adapter_feature_and_timing_contract.md` — calculation endpoint
- `docs/timing_metrics_methodology.md` — timing, censoring, resources, reporting
- `docs/cross_platform_benchmark_execution.md` — qualification and execution
- `docs/ibsi_compliance.md` — independent compliance workflows

Licensed under Apache-2.0. See `LICENSE` and `NOTICE`.

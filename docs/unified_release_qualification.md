# One untimed release-qualification workflow

`python -m scripts.qualify_release` creates one fresh evidence bundle for a
frozen checkout and its existing native adapter environments. It never installs
or rebuilds an environment, starts a timing benchmark, resumes a benchmark
ledger, or changes the `.venvs` junction. Keep the current performance run paused.

Use the existing controller environment from the repository root. All five
configured native environments must already pass the repository verifier,
including their recorded metadata and platform locks. The workflow verifies all
five before and after execution, even when `--profiles` selects fewer adapters.
Spatial checks always exercise Pictologics, MEDimage, MIRP and Z-Rad; PyRadiomics'
missing spatial workload remains explicitly unsupported.

## Baseline and candidate commands

Review the exact commands without creating files or launching workers:

```text
python -m scripts.qualify_release --output-dir qualification/baseline/windows-amd64 --plan
```

Run the complete baseline using already-materialized, hash-pinned IBSI inputs:

```text
python -m scripts.qualify_release --output-dir qualification/baseline/windows-amd64
```

The default scope is `full`: environment verification, untimed smoke, complete
observed feature inventory with public/prepared/repeat equivalence, bounded
robustness fixtures, independent spatial formulas and Pictologics branch checks,
physical-RAM policy probes, IBSI 1 digital phantom, and IBSI 2 Phases 1 and 2.
References and candidate response maps are generated inside this new bundle;
input data remain in their normal repository locations. Missing inputs fail the
assessment. Explicit `--bootstrap-inputs` permits fetching only the pinned IBSI
input components and workbook; it does not bootstrap Python environments.

After the next Pictologics environment and profile have been separately prepared
and reviewed, use a separate checkout and a fresh output root. Keep the other
four adapter profiles, locks, installed environments and adapter source frozen:

```text
python -m scripts.qualify_release --output-dir qualification/candidate/windows-amd64 --baseline PATH_TO_BASELINE/release-qualification.json
```

Retain the entire original baseline directory, including logs and binary
fixtures/maps. The baseline report must retain its original filename. Before
launching workers, the workflow validates its report fingerprint and every
recorded artifact, and rejects missing, changed or unexpected files. This detects
changes to saved evidence; it does not authenticate the original execution.

Baselines must have the same native platform, scope and requested adapter set.
Use a separate baseline for Windows AMD64 and macOS ARM64. Selecting only
`--profiles pictologics` is supported, but requires a baseline made with that
same selection and still verifies the four frozen environments and runs the
four-implementation spatial checks.

For bounded checks without IBSI, add `--scope core` to both baseline and candidate
commands. A core result is explicitly partial. Add
`--workspace-root PATH_TO_VALIDATED_WORKSPACE` to include deterministic small
CT/MRI/PET crops from `ibsi2_phase3/manifest.json`; source hashes are checked.
Without this option robustness uses synthetic fixtures only. Crops are never
reported as full-volume or IBSI 2 Phase 3 accuracy validation.

## Reading the result

The main artifact is `release-qualification.json`. It records commands and
dependency outcomes, per-stage log hashes, source identities, before/after
environment snapshots, feature names and values, detailed stage results,
limitations, all output-file hashes and a report fingerprint. Partial results
are saved during execution. Each run needs a fresh output directory; there is
no resume or overwrite mode. A stage watchdog limits execution and stops its
owned process tree on timeout; no elapsed-time observation is collected by the
workflow. `--step-timeout` is a safety limit, not a benchmark metric.

| Exit | Meaning |
| --- | --- |
| `0` | Requested checks completed with no recorded findings; `core_checks_only` remains partial. Full scope is `qualified_with_coverage_limits`. |
| `1` | Missing, invalid, incomplete or failed execution evidence. Read `errors`, `steps` and `logs/`. |
| `2` | Completed evidence needs review: numerical findings, RAM-policy failure/unavailability, feature changes, compliance changes or a comparison that is not matched. |

Invalid command-line arguments also use argparse's exit `2`; those produce no
completed assessment. Inspect the report's `complete` and `status` together.
Neither a zero exit nor `complete: true` permits final timing or means IBSI
certification. `complete` means the requested evidence was produced and checked;
it can accompany `needs_review`.

The known Pictologics 0.7.1 RAM-policy discrepancy is a reviewable observation,
so an otherwise complete baseline can legitimately exit `2`. A changed private
spatial API or missing probe makes execution evidence incomplete instead. Do not
change the installed package or relax a scientific reference just to obtain a
green result. See [the release feedback and acceptance checks](pictologics_release_feedback.md).

`baseline_comparison` reports stable-adapter changes, added/removed feature names,
changed values using `rtol=1e-9` and `atol=1e-12`, shared source changes, diagnostic
fixture changes, and IBSI outcome/coverage changes. Pictologics' adapter source,
profile and platform-lock updates are identified separately from shared protocol
source changes. A changed stable adapter, fixture or shared protocol prevents a
matched-comparison claim.

IBSI 2 Phase 2's common preprocessing uses the Pictologics environment even for
the four frozen extraction adapters. A Pictologics profile or environment change
therefore sets `ibsi2_phase2_preprocessing_changed` and prevents a fully matched
comparison claim. Differences in their Phase 2 results may reflect this shared
preprocessing change, rather than changes to those four packages. Existing
baseline response maps and their provenance remain intact.

## Coverage boundaries

Public/prepared/repeat equivalence checks the complete finite feature set on the
diagnostic fixture; it is not independent proof of accuracy or exhaustive clinical
coverage. Normal robustness fixtures require complete finite agreement.
Degenerate fixtures separately characterize undefined values or rejections and
never become finite-feature successes. Reproducible native defects and finite
values for undefined spatial formulas remain `needs_review` findings; unexpected
execution or equivalence errors fail the assessment. The spatial formula check uses small
isotropic/anisotropic fixtures and disposable branch overrides; the installed
implementation is not modified.

IBSI reports retain unsupported and nonstandardized calculations separately.
IBSI 1 and Phase 2 summaries preserve their full reference grids and report both
passed/evaluated and passed/all-standardized denominators. Phase 1 preserves the
full defined test grid, unsupported counts and numerical failures. Numerical
failures require review; unsupported coverage is not relabeled as a pass.

This workflow does not certify full clinical volumes, IBSI 2 Phase 3, large-image
feasibility, peak memory, cross-platform equivalence or the actual publication
host. It preserves current benchmark timing and ledger state. Follow
[final-version readiness](final_version_readiness.md) and the
[host execution workflow](cross_platform_benchmark_execution.md) before any
separately authorized final timing run.

## Continuous integration and the publication Mac

The native Windows AMD64 and macOS ARM64 CI jobs execute this command with
`--scope core`. They require complete evidence and publish the entire core
bundle as an artifact. A completed `needs_review` result is visibly annotated
and retained; it is not a numerical pass or permission to time adapters. Missing
evidence and execution failures fail CI. These jobs do not download the full
clinical workspace or run the complete IBSI assessment.

On the intended Mac, after preparing its independent locked environments and
validated input workspace, run:

```bash
poetry run python -m scripts.qualify_release --output-dir artifacts/release-full-macos-arm64 --workspace-root data/benchmark --bootstrap-inputs
```

This runs full untimed IBSI qualification plus bounded real CT/MRI/PET crops.
It does not run host timing preflight or certify the hardware's performance.
Preserve the complete output directory for a later same-platform baseline
comparison. The Windows qualification bundle cannot replace this Mac run.

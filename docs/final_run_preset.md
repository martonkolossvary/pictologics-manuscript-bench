# Final-run preset

The opt-in `--preset final` launcher option applies
[`configs/benchmark/final_run.json`](../configs/benchmark/final_run.json).
Preparing or printing this plan does not start timing. Keep the existing
exploratory ledger paused until the final Pictologics release is qualified.

| Setting | Final protocol |
| --- | --- |
| Execution order | Rotated adapter order within matched configurations |
| Adapters | Pictologics, PyRadiomics, MIRP, MEDimage, Z-Rad |
| Fresh processes | 3 per eligible configuration |
| Measured observations | 3 per process, following the untimed warmup |
| Worker ceiling | 900 seconds for the whole task, including initialization |
| Memory | Combined worker cap of 12 GiB, reduced when available RAM is lower |
| Threads | All physical cores per isolated task; one adapter process at a time |
| Inputs | Manifest-harmonized, calculation-only workload contract |
| Censoring | Existing timeout and memory preflight/cutoff policies; unsupported and censored rows are not zero |

The thread allowance remains the engine's existing policy and is recorded in
each immutable run specification. The preset does not force a library to use
threads it cannot use. Initial adapter order is fixed; the engine rotates that
order across configurations rather than completing one adapter's full run first.
Repeated timeouts and screen-only execution are disabled in this final preset.
The existing warmup, within-process median, and between-process median policies
remain in the checksum-bound endpoint contract.

## Prepare a reviewable plan now

Use a new result root for the final release, separate from every exploratory
or earlier-release ledger. Give both the result root and the matching public
host profile explicitly. Paths below are examples for the two reviewed hosts.

Windows, print only:

```powershell
poetry run python -m scripts.launch_benchmark --preset final --host-profile configs/benchmark/hosts/windows-9800x3d-01.json --result-root C:/BenchResults/final-pictologics
```

macOS, print only:

```bash
poetry run python -m scripts.launch_benchmark --preset final --host-profile configs/benchmark/hosts/mac-m4pro-01.json --result-root "$HOME/BenchResults/final-pictologics"
```

After preparing the full input workspace for the final source commit, append
`--validate-plans` to run controller validation only. No adapter is invoked.
The checked-out source must be clean. If the launcher or input-preparation
sources changed, first refresh the input-only workspace attestation with
`python -m scripts.prepare_benchmark_workspace --output-root data/benchmark --validate-only`.
The dry-run record is saved under
`data/benchmark/host_attestations/<machine-id>/controller_dry_run.json`.
Full untimed scientific qualification on the intended machine is still required;
a valid launch plan alone does not certify numerical accuracy.

The printed plan and dry-run record include the complete preset, its ID and
SHA-256. The ID and hash also enter the frozen host settings passed to the
controller, so existing source, environment and run-fingerprint guards apply to
the preset. Conflicting CLI overrides, a different workspace observation count
or adapter order, and a different host thread policy fail before dispatch.
An explicit override equal to the preset is accepted.

## Resume and eventual execution

Create the final result root only after the final release and all five adapter
environments have been frozen and qualified. When timing is authorized later,
use the platform wrapper with the same preset, host profile and result root,
adding the launcher's existing `--execute --confirm CALCULATE` acknowledgement.
The Windows wrapper holds its sleep assertion; the macOS wrapper uses
`caffeinate`. No execution command is run as part of preset preparation.

Interrupted final runs may resume with the identical preset, source, inputs,
environment locks and host settings. Do not reuse an exploratory result root
or remove ledger files to bypass a mismatch. Existing fingerprint checks remain
unchanged. Calls without `--preset final` preserve historical defaults, including
adapter-first ordering and the existing repeat-extension interface, so archived
launch commands keep their original meaning.

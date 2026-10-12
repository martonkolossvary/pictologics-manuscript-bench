# Screen before completing the benchmark

The current Pictologics pin is 0.7.1. Use a **new result root** for this release;
the completed 0.5.1 run is historical evidence and must not be resumed with the
new adapter or environment. Its observed boundaries are in
[historical_screening_20261012.md](historical_screening_20261012.md).

## What screening does

`--screen-only` uses the normal three-observation timing protocol and full
three-repeat plan, but runs only the first unresolved process for each
case/adapter/workload. A successful process remains a measured repeat. The
controller records actual timeouts, predicted memory rejections, actual memory
failures, unsupported workloads, errors and propagated cutoffs separately.
It stops before completing successful configurations' remaining repeats.

Use `--execution-order rotated` to alternate adapters within matched
case/workload blocks. Tasks are serial; background execution means detached
from the terminal, not concurrent adapter calculations. The default deadline
is 900 seconds for the entire process, including startup, warmup and calibration;
the memory cap is 12 GiB, reduced when the host needs more free memory.
Changing either policy requires a new result namespace. A single successful
screen is evidence of feasibility, not a guarantee that all repeats will finish.

## Prepare and verify

Follow the platform setup in the README. Rebuild the Pictologics environment
after updating source, then verify all environments:

```text
poetry run python -m bench.cli env create --profiles pictologics --force
poetry run python -m bench.cli env verify
poetry run python -m scripts.smoke_adapter_environments
poetry run python -m scripts.prepare_benchmark_workspace --output-root data/benchmark --validate-only
```

The smoke is an untimed installation/API check. It cannot establish a timeout
boundary or IBSI compliance. Workspace validation checks all 283 prepared cases
and refreshes its manifest; copying an existing prepared workspace is allowed
when its checksums and generation provenance pass. Keep the source committed
and unchanged during the run. Do not rebuild live adapter environments.

On Windows, use a short repository, result and Numba cache path. Even with a
short result root, Numba's generated kernel cache filename can exceed the legacy
Windows path limit inside a deeply nested checkout. Set `NUMBA_CACHE_DIR` to a
short local directory before both smoke and screening. JIT code remains warm;
per-call geometric caches are reset according to the timing contract.

## Windows background screening

From the repository root in PowerShell, replace the example local paths:

```powershell
$env:NUMBA_CACHE_DIR = 'C:\bench-cache'
$controllerPython = (poetry env info --executable).Trim()
$screenRoot = 'C:\bench-screen-071'
New-Item -ItemType Directory -Force $screenRoot | Out-Null
$screenArgs = @(
  '-u', '-m', 'scripts.launch_benchmark',
  '--workspace-root', 'data/benchmark', '--result-root', $screenRoot,
  '--host-profile', 'configs/benchmark/hosts/windows-9800x3d-01.json',
  '--execution-order', 'rotated', '--screen-only',
  '--stop-file', "$screenRoot\STOP", '--execute', '--confirm', 'CALCULATE'
)
$screenProcess = Start-Process -FilePath $controllerPython -ArgumentList $screenArgs `
  -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput "$screenRoot\stdout.log" `
  -RedirectStandardError "$screenRoot\stderr.log"
$screenProcess.Id | Set-Content "$screenRoot\launcher.pid"
```

Use paths without spaces in this PowerShell `-ArgumentList` example. The checked-in
host profile applies only to the named hardware; create an accurate public-safe
profile for a different host. Keep the machine on AC power and otherwise idle.

To stop cooperatively, create `C:\bench-screen-071\STOP`. The controller
terminates its owned worker tree, records interruption and checkpoints. Check
the log/process exit before restarting. Remove the marker, then repeat the
same launch to resume. Do not force-kill the process for a routine stop.

## macOS background screening

Run from a native arm64 checkout after setup and host qualification:

```bash
mkdir -p /absolute/local/bench-screen-071
nohup ./scripts/run_benchmark.sh \
  --workspace-root data/benchmark \
  --result-root /absolute/local/bench-screen-071 \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json \
  --execution-order rotated --screen-only \
  --stop-file /absolute/local/bench-screen-071/STOP \
  --execute --confirm CALCULATE \
  > /absolute/local/bench-screen-071/stdout.log 2>&1 &
```

The wrapper holds `caffeinate`; the same stop-marker procedure applies.
Native locks currently cover Windows AMD64 and macOS arm64. Intel macOS and
Windows ARM64 are not qualified by these locks.

## Inspect and continue

Export the live ledger from either platform (substitute the machine ID):

```text
poetry run python -m scripts.summarize_screening --machine-root RESULT_ROOT/MACHINE_ID --output-dir artifacts/screening-status
```

`resource_issues.csv` identifies the exact case, image shape, ROI voxels, mask,
requested families, representation, binning settings, repeat and outcome.
`tasks.csv` also includes pending and measured tasks; `summary.json` reports
coverage and mixed success/failure configurations. Reads include committed WAL
rows. Each pillar is a coherent snapshot, not a simultaneous snapshot of all
pillars. Whole-process elapsed time, per-call median and censor bounds are
separate columns. These files can contain diagnostic error paths; review before
publishing them.

When ready, omit `--screen-only` and keep all other protocol arguments and the
result root unchanged. Successful screening repeats are reused; ordinary
failures may be retried on explicit resume. Known resource cutoffs are retained.
Keep `--execution-order rotated` for the final comparison. Do not run the
completion command while screening is still active.

To continue on another computer, check out the same Git commit, rebuild that
platform's native environments, copy/bootstrap and validate the input workspace,
and use the destination host's own result root and profile. Host-specific timings
cannot be resumed as if the second machine were the first. The historical and
current source, locked settings and portable reports travel through GitHub;
large data, virtual environments and live SQLite ledgers remain outside Git.

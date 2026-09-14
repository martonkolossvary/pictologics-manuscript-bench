# Windows run with enforced resource limits

This protocol uses one 900-second worker wall-time deadline, a maximum combined
worker memory budget of 12 GiB, and adapter stages across all three datasets.
Pictologics finishes first; the remaining adapters follow the workspace order.
Within each adapter, repeat 1 visits image sizes ascending before repeats 2 and 3.
Successful tasks retain three fresh processes and three observations per process.

The deadline covers import, preparation, warmup, calibration, observations and
serialization. Controller validation, input staging and verified process cleanup
add overhead. Windows sets a Job Object memory limit before releasing the
worker, covering its supervisor and descendants. The budget is reduced if live
available RAM cannot provide the configured cap plus the host reserve.

Preflight rejects estimated requirements above that budget. Rejections and
explicit allocation failures establish persistent cutoffs for the same adapter,
workload, mask/subject and representation: equal-sized repeats and larger images
are skipped. Different masks/configurations and unrelated real-world patients
remain eligible. Estimated skips, actual allocation failures and cutoff skips
are distinguishable in the ledger and reports; none is a measured runtime.

The corrected predictor uses measurements from the same configuration and the
nearest measured image size. It does not extrapolate tiny differences in
interpreter memory across distant sizes or unrelated patients. Existing
schema-15 runs created with the old predictor require the explicit recovery in
[memory_policy_recovery.md](memory_policy_recovery.md) before resuming under
the corrected source.

For a new experiment, use a fresh result directory. The old schema-14 run at
`C:\BenchResults\full-2dbd2dd-20260908` has a different timeout and resource
protocol and must not be resumed under this source. Its measurements remain
available as separate historical evidence.

## Start or resume

From PowerShell, run exactly this command for both the first launch and every
resume. The launcher automatically resumes all three shared ledgers and passes
over completed adapter stages.

```powershell
Set-Location C:\Projects\pictologics-manuscript-bench
.\scripts\run_benchmark.ps1 `
  --workspace-root data/benchmark `
  --result-root C:\BenchResults\full-mem12-15m-20260909 `
  --host-profile configs/benchmark/hosts/windows-9800x3d-01.json `
  --timeout 900 `
  --memory-cap-gib 12 `
  --execution-order adapter-first `
  --execute --confirm CALCULATE
```

To inspect the commands without starting workers, omit
`--execute --confirm CALCULATE`. To validate the three immutable plans without
calculating features, replace those two options with `--validate-plans`.

## Stop

Press **Ctrl+C once** and wait for the safe interruption message and shell
prompt. The launcher forwards the stop request to the controller, which closes
the worker tree and checkpoints the ledger. Run the identical command above
again. Committed measurements and resource cutoffs are retained; only unfinished
work is retried. Do not use the old compatibility recovery helper for this run.

## What the shorter limit changes

Adapter order improves time to useful baseline results; it does not reduce
the number of requested calculations. The shorter deadline and persistent
resource cutoffs reduce expensive attempts. In the old run, median MEDimage
64-cubed local-intensity and spatial-autocorrelation calls took about 249 and
227 seconds respectively. Five native calls (warmup, calibration and three
measurements) imply about 20.7 and 18.9 minutes before overhead. Many such tasks
will therefore time out under this 15-minute policy and suppress larger images
in their exact series. After a timeout, remaining identical case/workload repeats
are now skipped as well. Different same-sized cases remain eligible. Successful
tasks retain all three fresh-process repeats. To retain repeated timeout attempts
when creating a new run, pass `--repeat-timeouts`; this is fingerprint-bound.

Fifteen minutes is a per-worker ceiling, not a project deadline. A reliable
completion date needs observations under the new policy, particularly from the
real-world dataset. The ETA omits persisted memory and timeout skips and does not
borrow Pictologics timings to predict an unobserved adapter. Its extrapolation
range is not a statistical confidence interval.

Adapter stages also separate adapter measurements in time. Keep host load and
power settings consistent, retain their provenance, and consider a separate
rotated confirmation run if a publication comparison is sensitive to time or
thermal drift. Do not pool runs with different resource policies.

# Cross-platform benchmark execution

## Shared sequence

Every host follows the same order:

1. audit the clean source clone;
2. install and verify the controller and isolated adapters;
3. fetch and checksum-verify external inputs;
4. prepare and deeply validate `data/benchmark`;
5. choose a local non-synchronised result root;
6. run tests, environment verification, host/result preflight, and all
   controller dry-runs;
7. inspect the qualification attestation;
8. start calculations only with `--execute --confirm CALCULATE`.

The Python launcher is the single command generator on every platform. It is
print-only by default, and `--validate-plans` invokes only controller dry-runs.
Dry-runs and calculations require a clean Git worktree; ignored scientific
inputs, adapter environments, and external result roots do not make it dirty.

## macOS publication host

Use `configs/benchmark/hosts/mac-m4pro-01.json` only on the exact matching
Apple M4 Pro host. Connect AC power, select one Energy Mode and keep it fixed
for the complete machine run, defer updates, close background workloads, and
keep results on a local unsynchronised SSD. The launcher records the observed
mode, raw `pmset lowpowermode` value, observation status, portable power-mode
tag, and probe errors in the per-session provenance history. Energy Mode never
gates a launch: any observed mode is accepted, and an unavailable observation
is explicitly tagged `macos-energy-mode-unavailable`. During execution the
controller probes again immediately before and after every task. A task whose
boundaries differ is tagged `mixed-within-task` and QC flags it. A resume under
a different mode remains safe because power state is task/session provenance
rather than protocol identity. Reports distinguish single-mode, mixed-mode,
and unavailable-mode runs, and a mixed contribution is named
`mixed-power-modes`. Keep one mode throughout when practical.

```bash
./scripts/setup.sh
./scripts/qualify_benchmark_host.sh \
  /absolute/local/path/pictobench-results \
  configs/benchmark/hosts/mac-m4pro-01.json
```

The real-run wrapper starts `caffeinate -dimsu` before host preflight:

```bash
./scripts/run_benchmark.sh \
  --workspace-root data/benchmark \
  --result-root /absolute/local/path/pictobench-results \
  --host-profile configs/benchmark/hosts/mac-m4pro-01.json \
  --execute --confirm CALCULATE
```

A different Mac needs a separate reviewed host JSON with a stable public-safe
machine ID, observed hardware expectations, required runtime state, and fixed
benchmark settings. It does not need to expose the same Energy Mode options:
available modes are observed per host, every mode is allowed, and an
unavailable mode is recorded as such rather than inferred or blocked.

## Windows

Use a local NTFS path and a stable, non-identifying machine ID:

```powershell
./scripts/setup.ps1
./scripts/qualify_benchmark_host.ps1 `
  -ResultRoot C:\pictobench-results-SOURCE_COMMIT `
  -HostProfile configs/benchmark/hosts/windows-9800x3d-01.json
./scripts/run_benchmark.ps1 `
  --workspace-root data/benchmark `
  --result-root C:\pictobench-results-SOURCE_COMMIT `
  --host-profile configs/benchmark/hosts/windows-9800x3d-01.json `
  --execute --confirm CALCULATE
```

Keep repository, workspace, staged inputs, and results within the launcher's
path-length budget. The checked-in profile is valid only for its exact AMD
Ryzen 7 9800X3D host; create a separate public-safe profile for any other
Windows machine.

Windows setup requires Python 3.9, 3.10, and 3.12 plus `uv` 0.8. PyRadiomics
uses its official SHA-256-pinned CPython 3.9 Windows wheel because the upstream
3.1.0 release provides no newer Windows wheels and its source build requires a
native compiler. The launcher holds a `SetThreadExecutionState` system-sleep
assertion during execution and
releases it on exit. Session and task-boundary provenance includes AC/battery
state, battery saver, the active power-scheme GUID, the localized display name,
and a GUID-derived portable tag. The GUID keeps results comparable across
Windows display languages; the display name is retained only as diagnostics.

## Resume and machine separation

Results are namespaced by machine ID and pillar. Resume requires an identical
RunSpec fingerprint, host identity, environment fingerprints, endpoint,
dataset hashes, task plan, resource policy, thread policy, and timeout. Committed measured
tasks are not repeated. Do not merge ledgers between hosts; compare completed,
attested reports after execution.

For an orderly stop, press `Ctrl-C` once and wait for the controller's safe
interruption message and the shell prompt. The launcher forwards the signal and
waits for the active ledger checkpoint. Re-run the identical command to
continue. After an ungraceful terminal closure, forced kill, or power loss, the
same resume command transactionally converts any in-flight task to interrupted
before retrying it; already committed measured tasks remain untouched.

Qualification records are generated locally below the ignored workspace and
must never be committed as claimed benchmark results.

## Publish a completed multi-machine result

Never commit a live ledger or working result directory. Once all three pillars
are terminal, package the machine namespace into the architecture-aware,
checksum-bound contribution format:

```bash
poetry run python scripts/package_benchmark_submission.py \
  --result-root /absolute/local/path/pictobench-results \
  --machine-id MACHINE_ID
poetry run python scripts/package_benchmark_submission.py --validate
python3 scripts/audit_repository_contents.py
```

The generated directory under `benchmark-results/<os>-<architecture>/` may be
committed and submitted by pull request. Its identity includes machine ID, CPU
model, packaging date, and a digest of the source commit, machine profile, and
three run fingerprints. This allows many machines and repeat submissions to
coexist without overwriting one another.

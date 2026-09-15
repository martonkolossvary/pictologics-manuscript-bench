# Recovering the schema-15 memory predictor

The original `fixed_overhead_incremental_empirical_upper` predictor amplified
small differences in startup RSS and allowed unrelated masks/patients to affect
each other. It falsely rejected feasible Pictologics calculations at 256 cubed.
The corrected `nearest_size_scoped_incremental_v2` predictor uses the largest
measured input not above the target, within the same adapter/workload/stratum.
Repeated measurements at one size contribute a conservative peak; they do not
define a growth slope. The static bounds, Z-Rad pairwise-memory bound, available
memory reserve and enforced Windows worker limit remain active.

The corrected timeout policy skips remaining identical case/workload repeats
after an observed timeout. It still runs unrelated cases at the same size and
preserves three fresh processes for successful configurations. It does not
change the 900-second total worker deadline or native calculation/timing code.

## Stop, update, preview, apply

Stop the original launcher cleanly with Ctrl+C and wait for its checkpoint and
shell prompt. Keep its checkout unchanged until it stops. Update the repository
to the clean, reviewed repair commit, using the same controller Python
environment. The helper checks the original source aggregate and controller
environment against commit `7282e02c3dac0180f00c391eecc487b07c8d83cf`, rejects
unrelated execution changes, and verifies every committed measurement.

From the updated production repository:

```powershell
poetry run python scripts/prepare_benchmark_workspace.py `
  --output-root data/benchmark --validate-only

poetry run python scripts/repair_memory_policy.py `
  --machine-root C:\BenchResults\full-mem12-15m-20260909\windows-9800x3d-01
```

The first command revalidates the existing inputs and refreshes the workspace's
launcher checksum. It does not regenerate dataset files or calculate features.
Keep a copy of the original `data/benchmark/workspace_manifest.json` with the
recovery records before refreshing it. Without this step, the launcher rejects
the old workspace manifest after the source update.

The preview reports verified measurements, memory skips to reopen and the new
fingerprints. It acquires all three run locks before opening any ledger. It
does not change task states or measurement payloads. To apply that repair:

```powershell
poetry run python scripts/repair_memory_policy.py `
  --machine-root C:\BenchResults\full-mem12-15m-20260909\windows-9800x3d-01 `
  --apply
```

Resume with the normal command in `revised_windows_run.md`. The helper never
launches calculations. If retaining all timeout attempts is desired, use
`--repeat-timeouts` for both the repair and the subsequent launcher commands.
Do not change this option on an ordinary resume: it is part of the protocol.

## What recovery preserves

Each pillar receives a SQLite backup, original metadata files and a recovery
plan beneath `MACHINE_ROOT/recovery/memory-policy-v2-TIMESTAMP/`. One transaction
per ledger records its old/new specifications and source evidence, archives the
original skipped rows in events, clears only old estimated-memory cutoffs and
requeues their affected tasks. Actual memory failures, timeouts, attempt history,
measured task rows and payload bytes are preserved. Requeued tasks are reevaluated
when reached; genuinely oversized tasks can still be rejected under v2.

Historical payloads retain their original run fingerprint. The new
`measurement_origins` table binds each such task and exact payload checksum to
the recorded amendment. New measurements use the revised fingerprint. Report
loading and resume validate this chain; they do not broadly accept arbitrary
old fingerprints. Host observations retain their original session identity.
The controller includes amendments in subsequent metadata checkpoints.

The helper validates all pillars before changing their task states. If a repair
is interrupted between pillars, running it again recognizes completed amendments
and finishes remaining work. It also finishes metadata exports after an
interruption following the SQLite commit. CSV/QC exports are refreshed by the
next normal controller checkpoint; the ledger remains authoritative throughout.

This correction changes task eligibility and timeout-repeat policy, not the
native feature implementation or calculation timer. Retain the amendment and
original provenance when interpreting or publishing the combined measurements.

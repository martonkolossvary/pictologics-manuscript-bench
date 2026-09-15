# Ledger repair and feasibility ordering

Reopened policy skips can already have attempt 0 in their history. A subsequent
skip previously inserted attempt 0 again and crashed with a SQLite uniqueness
error. Terminal decisions now append an unused attempt when no worker is active.
Failed/interrupted history is retained; an active worker or validated atomic
payload adoption still finalizes its own attempt.

The default launcher runs these stages for Pictologics, then each remaining
adapter in workspace order:

1. Screen unknown configurations in pillars 1, 2 and 3, smallest images first.
2. Complete remaining repeats for successful configurations in pillars 1, 2 and 3.

A screen executes one full normal task, including warmup, calibration and three
observations within the same 900-second total deadline and memory budget. It
counts as a measured repeat. Any existing successful repeat establishes
eligibility. Persisted timeout/memory exclusions are applied to their exact
scope; all skipped tasks remain in the experimental grid. A failed screen
does not establish success, and completion refuses to run unknown cases.
`--repeat-timeouts` retains its explicit opt-in behavior during completion.

Stop/resume derives eligibility from the ledger; no separate pilot results or
mutable eligibility file are required. Stage selection is recorded as an event
and does not alter task identities or the repeat count. The shared Python
launcher/controller implements this on Windows and macOS. macOS memory
enforcement remains sampled process-tree RSS; this change does not add a kernel
memory cap.

For existing stopped schema-15 runs already using the corrected v2 memory policy,
use the original controller Python environment with the clean reviewed checkout:

```powershell
python scripts/repair_execution_order.py --machine-root C:\BenchResults\full-mem12-15m-20260909\windows-9800x3d-01
python scripts/repair_execution_order.py --machine-root C:\BenchResults\full-mem12-15m-20260909\windows-9800x3d-01 --apply
python scripts/prepare_benchmark_workspace.py --output-root data/benchmark --validate-only
```

The repair acquires every pillar's normal run lock, verifies payloads and the
known prior controller source, backs up each ledger and metadata file, and adds
an explicit source amendment. It preserves task rows, attempt history, payload
bytes, original measurement fingerprints and all calculation settings. Normal
resume still rejects unrelated source or protocol changes. Back up the existing
workspace manifest before revalidation when retaining the complete input audit.
No repair command starts calculations. Resume using the usual launcher command;
it automatically selects screening and completion stages from saved evidence.

For direct controller use, `--execution-phase screen` and `--execution-phase
complete` are execution selectors over the same full plan. The default direct
controller selector `all` retains the existing single-pillar behavior.

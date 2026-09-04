# Community benchmark results

This directory accepts publication-attested result bundles from the same
benchmark source and protocol on different hardware. Raw ledgers, staged NIfTI
inputs, adapter payloads, logs, and working reports remain under ignored local
`results/` paths and must not be committed.

Create a bundle only after all three pillars have completed:

```bash
poetry run python scripts/package_benchmark_submission.py \
  --result-root /absolute/local/path/pictobench-results \
  --machine-id YOUR_STABLE_PUBLIC_MACHINE_ID
```

The packager verifies the ledgers and payloads, regenerates each report, checks
publication attestation, removes local paths and diagnostic stderr from the
public task table, and copies only checksummed report artifacts. It refuses a
dirty source tree, a source-commit mismatch, incomplete pillars, mixed machine
identities, altered artifacts, or an existing destination.

Bundles use this collision-resistant layout:

```text
benchmark-results/
  <operating-system>-<architecture>/
    <machine-id>--<cpu-model>--<power-mode-tag>--<YYYY-MM-DD>--<identity-digest>/
      submission_manifest.json
      pillar1_morphology/
      pillar2_whole_anatomy/
      pillar3_ibsi2_phase3/
```

For example:

```text
benchmark-results/macos-arm64/mac-m4pro-01--apple-m4-pro--macos-high-power-pmset-2--2026-09-01--1a2b3c4d5e6f/
```

The operating system and architecture form the first namespace. The machine
ID, CPU model, power-mode tag, date, and digest distinguish repeated
submissions, power policies, and different machines. When the host cannot
report a mode, the tag states that it is unavailable rather than guessing.
The manifest records the full public-safe hardware profile, raw power-mode
observation and status, memory, operating-system release, Python version,
source commit, three immutable run fingerprints, and every artifact checksum.

Before committing, validate every bundle in the checkout:

```bash
poetry run python scripts/package_benchmark_submission.py --validate
python3 scripts/audit_repository_contents.py
```

Commit only the newly generated bundle and open a pull request. In the pull
request, state whether the machine was dedicated/idle, describe its power and
thermal settings, and disclose any deviations. Do not edit generated CSV,
JSON, workbook, or figure files. We especially welcome results from other CPU
architectures, operating systems, memory capacities, and processor families.

# Qualification while the final Pictologics release is pending

Final performance timing is deferred until Pictologics is finalized. Preserve
the exploratory 0.7.1 screening evidence separately. A screening measurement
belongs to its recorded release, source, machine and resource policy; it is
not automatically a final-release observation.

Use the [release qualification command](unified_release_qualification.md) to collect a
fresh untimed assessment and compare it with a saved same-platform baseline.
The [robustness suite](adapter_robustness.md) adds synthetic modality, geometry
and degenerate-ROI checks. The [final-run preset](final_run_preset.md) prepares
the later common timing protocol, and the
[Pictologics feedback package](pictologics_release_feedback.md) checks its
Windows spatial-memory policy without running large calculations.

## Freeze the stable adapters

Keep PyRadiomics, MIRP, MEDimage and Z-Rad at their reviewed profile and native
environment-lock versions. Preserve the committed source checkout and the
environment metadata used for qualification. Run untimed checks in separate
output directories, for example `qualification/stable-four/SOURCE/PLATFORM/`:

```text
python -m bench.cli env verify --profiles pyradiomics mirp medimage zrad
python -m scripts.smoke_adapter_environments --profiles pyradiomics mirp medimage zrad --output-dir qualification/stable-four/SOURCE/PLATFORM/smoke
python -m scripts.qualify_adapter_calculations --output-dir qualification/stable-four/SOURCE/PLATFORM/calculations
python -m scripts.qualify_spatial_statistics --output-dir qualification/spatial/SOURCE/PLATFORM
```

Follow [the compliance workflow](ibsi_compliance.md) with
`--adapters pyradiomics,mirp,medimage,zrad` on `run-ibsi1`,
`generate-ibsi2-candidates`, `evaluate-ibsi2-phase1`, and `run-ibsi2-phase2`.
Give each stage its own output directory. Candidate-manifest adapters must
match the requested adapters in the same order. Import the pinned official
references; do not relax reference hashes to obtain a pass.

IBSI 2 Phase 2 uses common preprocessing implemented by the pinned Pictologics
environment, even when Pictologics is absent from the extraction adapter list.
The generator records and fingerprints that extra environment as a preprocessing
dependency. Keep it frozen with the completed stable-four evidence. Upgrading
Pictologics later does not retroactively change those prepared images, but it
invalidates resuming unfinished candidate generation with changed preprocessing.

The calculation qualifier calls each prepared endpoint directly, without a
clock, and compares its complete output with the public native extraction path
and a repeated calculation. The spatial qualifier independently verifies Moran's
I and Geary's C on isotropic/anisotropic fixtures and exercises Pictologics'
pairwise, FFT and memory-fallback branches. Undefined degenerate ROIs are recorded
explicitly. The diagnostic branch overrides exist only in disposable test workers.

An unsuccessful standardized numerical check is a scientific finding to
investigate and report. A feature absent from a native package remains
unsupported. Untimed installation smoke does not certify accuracy, and
coverage-only calculations without consensus references do not become passes.

Keep IBSI 1's 174 calculation instances, 172 feature definitions and 169
standardized checks distinct. IBSI 2 Phase 2 has 396 designed calculations,
324 published reference rows and 323 standardized checks per adapter. Report
conditional accuracy as passed/evaluated and overall standardized success as
passed/all standardized checks; both are necessary to expose coverage limits.

## Read the existing evidence

The following command reads existing IBSI 1 or IBSI 2 Phase 2 bundles without
running native adapters, refreshing workspace manifests or modifying evidence:

```text
python -m scripts.summarize_readiness --run-dir qualification/stable-four/SOURCE/PLATFORM/ibsi1 --run-dir qualification/stable-four/SOURCE/PLATFORM/ibsi2-phase2 --output qualification/stable-four-readiness.json
```

Use one `--run-dir` per available run. Missing directories are reported as
missing evidence. The snapshot verifies the recorded run fingerprint, table
checksum, official reference identity and complete row grid, input hashes,
recorded environment freeze/lock identity, and completed payload checksums.
It recomputes evaluated tolerance outcomes and preserves each adapter/version
separately. Missing, incomplete, invalid, execution-error and numerical-failure
evidence cannot qualify. Unsupported rows and nonstandardized coverage-only
rows remain visible, including runs with no evaluated accuracy evidence.

Exit code zero means only that every supplied bundle has verified recorded
evidence with no identified execution or numerical failures. It is not a
full-project readiness decision or permission to begin timing. A compliance
runner's own zero exit code merely means it wrote its results and is not a
substitute for this inspection. Snapshots always label final timing deferred.

The snapshot checks historical recorded identities; it does not reverify
current installations, authenticate the original execution, or prove that
source and environment files are still unchanged. Keep the original source
commit and rerun environment verification when resuming work. Inputs and
references must remain accessible at their recorded paths. IBSI 2 Phase 1
response-map reports, native smoke, workspace validation and publication-host
qualification remain separate evidence; the snapshot does not certify them.
Run it after the corresponding compliance stage finishes. Do not publish
diagnostic paths without reviewing the snapshot.

## Requalify each Pictologics change

Use a separate development checkout and a fresh directory such as
`qualification/pictologics/VERSION-SOURCE/PLATFORM/`. Update the exact release
or commit, wheel hash and platform locks. Rebuild only its environment, verify
it, run its untimed smoke and `scripts.qualify_adapter_calculations --profiles
pictologics` (with `python -m`), and repeat the relevant compliance stages with
`--adapters pictologics`. Use a separate IBSI 2 candidate bundle. Retain prior
version evidence rather than overwriting it.

Do not resume an incomplete qualification run after modifying its source or
environment. Compliance identity presently includes every Python source file
under `bench/`; even a Pictologics-only adapter edit changes that identity.
Complete the four stable adapters' runs in their frozen checkout. Completed
accuracy evidence remains useful under its recorded version and protocol.

## Final timing and another machine

After the intended Pictologics release is fixed, commit the five-adapter
configuration, verify all native environments, smoke every supported workload,
validate the prepared inputs, and qualify the actual host. Start a **new final
timing result root** under that frozen source and use rotated adapter order.
Reuse validated input data, not old timing cells or ledgers. Do not combine
earlier exploratory timings with the new release as one matched timing run.
Collect all five adapters under the final common protocol on the same host.

The benchmark binds source, all adapter profiles, environment fingerprints,
inputs and resource settings. Do not weaken its resume guards to mix releases.
Likewise, do not rebuild environments or change source in a running checkout.
Only an unchanged same-host run may reuse its own completed screening repeats.

Another machine needs its own accurate host profile, native environments,
qualification and result namespace. Current locks cover Windows AMD64 and
macOS ARM64. Transfer the clean source and validated input workspace; do not
copy virtual environments between platforms or merge their live ledgers.
Review platform-specific results separately, including memory exclusions.
See [screening and handoff](screening_and_handoff.md) for cooperative stopping
and [host qualification](cross_platform_benchmark_execution.md) for dry-runs.

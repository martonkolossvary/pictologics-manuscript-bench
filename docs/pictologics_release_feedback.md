# Pictologics spatial RAM detection: upstream feedback and release checks

This is a prepared upstream feedback draft, not a submitted issue. All checks
below are untimed and preserve the paused benchmark ledger.

## Reproducer and evidence

Pictologics 0.7.1 computes Moran's I (`N365`) and Geary's C (`NPT7`) using exact
inverse-physical-distance weights. Its FFT and pairwise implementations evaluate
the same definitions, subject to floating-point differences. In
[`_fft_memory_cap`](https://github.com/martonkolossvary/pictologics/blob/f2fdb963b17b483f339ff547c23fa519bf4850b9/pictologics/features/intensity.py),
the intended FFT cap is the smaller of 16 GiB and half of physical RAM.
The helper reads RAM using `os.sysconf`; when unavailable or failing it assumes
8 GiB RAM and returns a 4 GiB cap. Python documents
[`os.sysconf` as Unix-only](https://docs.python.org/3.12/library/os.html#os.sysconf),
so this fallback also applies on Windows hosts with substantially more RAM. This can select the
quadratic pairwise algorithm even when FFT fits the documented policy.

From the repository root, using the existing controller environment:

```sh
python -m scripts.check_pictologics_spatial_policy --output artifacts/spatial-policy.json
```

The report records independently detected total physical RAM, the native cap,
OS capabilities, installed version, interpreter and intensity-module SHA-256.
Each probe uses a fresh Pictologics interpreter. Three probes simulate missing,
`OSError`-raising and `ValueError`-raising `sysconf` inside disposable processes;
no package source or benchmark environment is modified. The diagnostic disables
import-time warmup and never calls a feature extractor or timing routine.

Predictions cover dense synthetic ROI boxes from 8 to 512 voxels per dimension.
Only integer FFT geometry is computed; images, pair matrices and FFT arrays are
never allocated. These predictions explicitly apply the **0.7.1 heuristic**
(`N² >= 100 * FFT grid points`, 32 estimated bytes per grid point), even when
run with a later release. They are not measurements of runtime, actual peak RAM,
or the later release's algorithm choice. Bounding boxes and occupancy both
matter; a full 320³ synthetic box is not equivalent to every 320³ benchmark mask.

The separate benchmark process-memory limit still applies. Increasing the FFT
cap does not prove a case will fit that limit or finish before its timeout.

## Native Windows observation, 2026-10-17

The existing Pictologics 0.7.1 environment produced a complete local
RAM-policy report (`artifacts/release-core-20261017-r1/spatial-policy.json`)
on Windows AMD64 with Python 3.12.10 and SciPy 1.18.1. This is the policy
diagnostic's result, not a claim that the containing release assessment passed.

| Observation | Value |
| --- | --- |
| Total physical RAM, independently read with `psutil.virtual_memory().total` | 33,396,543,488 bytes (about 31.10 GiB) |
| Expected cap from `min(16 GiB, physical_RAM // 2)` | 16,698,271,744 bytes (about 15.55 GiB) |
| Native `os.sysconf` available | No |
| Native cap and each of the three simulated-failure caps | 4,294,967,296 bytes (4 GiB) |
| Diagnostic completion / policy acceptance | Complete / failed on all four probes |

The installed intensity-module SHA-256 was
`77a37549543bc68932f605cc421e20503150bf1f9dbab226d7866b8cae205458`;
the report also records the checker hash and actual source/interpreter paths.
The current checkout's `.venvs` junction points to the preserved benchmark
environment, which explains the different resolved source path in the report.
Neither that environment nor the timing ledger was changed by this check.

For illustration, the 0.7.1 geometry model gives a dense 320³ ROI a 640³ FFT
grid and an 8,388,608,000-byte estimate (7.8125 GiB). Its crossover favors FFT,
but the observed 4 GiB cap selects pairs. That estimate is below this host's
15.55 GiB half-RAM cap. The diagnostic did not allocate this ROI or measure
its runtime, peak memory, or actual branch execution.

## Suggested upstream resolution

Use a cross-platform physical-RAM query, such as an existing supported system
dependency or Windows `GlobalMemoryStatusEx`, before choosing the documented
half-RAM limit. Keep a conservative fallback for genuinely unavailable or
invalid memory information. A diagnostic indicating the detection source and
effective cap would make platform behavior easier to audit. Do not disable the
memory guard or replace exact spatial statistics with approximations.

The benchmark must retain the unmodified upstream implementation. A one-off
benchmark monkeypatch that raises only Pictologics' cap would alter the tested
software and cannot serve as a fix.

## Untimed acceptance checks for the next release

1. On Windows with more than 8 GiB RAM and on the intended macOS host, confirm
   physical-RAM detection and the documented cap. Include a lower-memory case
   and a case at/above 32 GiB to cover half-RAM and the 16 GiB ceiling. Mocked
   boundary tests belong upstream; do not allocate those amounts of RAM.
2. Missing or failing `sysconf` must still use another available RAM source.
   Test the genuinely unavailable/invalid-source case upstream and document its
   conservative fallback. A result on an 8 GiB host alone cannot distinguish
   correct detection from the old assumption. These diagnostic probes alter
   `os.sysconf` after package import and exercise the 0.7.1 helper's lookup path;
   cached or aliased detection in a new release requires adapted fault injection.
3. Preserve both IBSI feature names, exact physical-distance weighting, disabled
   extraction behavior, and undefined empty/single-voxel/constant-ROI behavior.
   Reuse `python -m scripts.qualify_spatial_statistics` for the independent
   formula, forced pairwise/FFT, anisotropic-spacing and memory-guard checks;
   do not duplicate those numerical tests in the RAM-policy reproducer.
4. Confirm the guard still warns and uses exact pairs when the FFT estimate
   exceeds the cap. Recompute the existing workload-specific geometry
   predictions only after reviewing any changed heuristic or memory estimate.
5. Pin and qualify the final release in a **new environment** before starting
   fresh matched timing runs. Keep the old version's environment and ledger.

For a candidate release in a separate environment:

```sh
python -m scripts.check_pictologics_spatial_policy --python PATH_TO_CANDIDATE_PYTHON --output artifacts/candidate-spatial-policy.json --require-physical-ram-cap
```

The default diagnostic exits successfully when all observations were collected,
including a recorded failure of the legacy policy. The acceptance flag instead
requires every probe to match `min(16 GiB, physical_RAM // 2)` and fails when RAM
cannot be independently determined. A missing/changed private helper yields an
explicit unsupported result and exit code 2, requiring inspection and adaptation
of this diagnostic. If upstream intentionally changes its policy, review and
update the acceptance rule rather than assuming the new behavior is defective.
Private API compatibility is not an upstream public-API requirement.

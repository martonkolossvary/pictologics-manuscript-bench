# Pictologics RAM investigation on the Windows host, 2026-10-19

The 4 GiB FFT cap comes from Pictologics 0.7.1's explicit Windows fallback.
Windows correctly exposes this host's usable RAM, and the native interpreter is
64-bit. The operating system is not restricting this interpreter to 4 GiB.
No feature extraction, performance timing or large-array allocation was needed
for this investigation. The frozen installation and paused ledger are unchanged.

## Machine evidence and root cause

| Observation | Result |
| --- | --- |
| Physically installed RAM, Windows `GetPhysicallyInstalledSystemMemory` | 34,359,738,368 bytes / 32 GiB |
| OS-visible physical RAM, `GlobalMemoryStatusEx` and controller `psutil` | 33,396,543,488 bytes / about 31.10 GiB |
| Installed minus OS-visible RAM | 963,194,880 bytes; this probe does not attribute it to a device |
| Pictologics interpreter | Python 3.12.10, 64-bit, Windows AMD64 |
| Native `os.sysconf` | Absent |
| Native `_fft_memory_cap()` | 4,294,967,296 bytes / 4 GiB |
| Intended `min(16 GiB, physical RAM // 2)` | 16,698,271,744 bytes / about 15.55 GiB |
| Available physical RAM at 14:30:29 UTC | 16,756,015,104 bytes; a changing snapshot |
| Available commit reported by Windows at that snapshot | 13,073,080,320 bytes; a changing snapshot |

The native helper catches the missing Unix query and assumes 8 GiB, then halves
that assumed capacity. Its own docstring explicitly describes this Windows
fallback. It never attempts the available Windows API. `psutil` is not installed
in this frozen Pictologics environment and is not a declared Pictologics runtime
dependency; relying on it would require a dependency change.

Microsoft distinguishes installed RAM from OS-visible RAM and currently
available memory. `GlobalMemoryStatusEx` provides the usable physical total
needed here. Its commit fields are not the disk size of the page file, and an
availability snapshot is not a reservation. See
[installed RAM](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getphysicallyinstalledsystemmemory),
[memory status fields](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex)
and [query semantics](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex).

The installed intensity-module SHA-256 remains
`77a37549543bc68932f605cc421e20503150bf1f9dbab226d7866b8cae205458`.
The latest public release checked was still
[v0.7.1](https://github.com/martonkolossvary/pictologics/releases/tag/v0.7.1).
The same helper remains in upstream
[main at `2f5d7da`](https://github.com/martonkolossvary/pictologics/blob/2f5d7da412dbd9ebc550b8db190da51726b075c7/pictologics/features/intensity.py).

## Effect on this benchmark's actual geometry

All 962 input-file hashes were reverified. The 283 cases' previously audited
ROI counts and bounding boxes were used with native SciPy 1.18.1 FFT lengths.
The installed function's original scalar decision block was then executed for
each case under both caps, without calling either calculation kernel. Its
decisions agreed with the geometry model in all 566 comparisons.

| Dataset | Cases | FFT with current 4 GiB cap | FFT with corrected 15.55 GiB cap | Cases switching from pairs to FFT |
| --- | ---: | ---: | ---: | ---: |
| Pillar 1 | 120 | 90 | 108 | 18 |
| Pillar 2 whole anatomy | 10 | 7 | 9 | 2 |
| Real-world cohort | 153 | 138 | 138 | 0 |
| Total | 283 | 235 | 255 | 20 |

The changed cases are:

| Case geometry | Cases | Native FFT workspace estimate, GiB |
| --- | ---: | ---: |
| Pillar 1, M1/M2, 512 cubed, all three HU profiles | 6 | 7.416 |
| Pillar 1, M3, 320 cubed, all three HU profiles | 3 | 4.023 |
| Pillar 1, M4, 320 cubed, all three HU profiles | 3 | 4.005 |
| Pillar 1, M3/M4, 384 cubed, all three HU profiles | 6 | 6.952 |
| Pillar 2, 320 cubed | 1 | 6.706 |
| Pillar 2, 384 cubed | 1 | 11.263 |

Seven cases still exceed the corrected cap: Pillar 1 M3/M4 at 512 cubed across
all three HU profiles, and Pillar 2 at 512 cubed. The other 21 pairwise cases
remain below the voxel-pair/FFT-grid crossover. Thus correcting detection does
not make every spatial workload use FFT. These counts predict branch selection,
not completed extractions, measured peak memory, speedup or timeout outcomes.
The two supported spatial features remain Moran's I and Geary's C with the same
mathematical definitions.

The warning also loses useful precision. For M4 at 320 cubed, the FFT estimate
is 4,299,816,960 bytes, only 4,849,664 bytes (4.625 MiB) above the current cap.
The native warning prints both as `4.0 GB`, making the reason for fallback
unclear. Its displayed units are binary GiB despite the `GB` label. An upstream
warning should expose sufficient precision, the memory-query source and the
effective cap. Any runtime estimate embedded in that warning is an upstream
heuristic, not a timing observation from this investigation.

## A separate constraint: the benchmark's process budget

Correcting RAM detection alone does not establish memory feasibility. The final
preset limits the complete process tree to at most 12 GiB, and its dynamic
budget may be lower. The native FFT cap only guards the estimated FFT work;
input arrays, interpreter/library memory and other live allocations also matter.

At the recorded availability snapshot, the benchmark's reserve rule yields
10,076,706,407 bytes (about 9.38 GiB). Pillar 2 at 384 cubed would select FFT after
the proposed correction, although its 11.26 GiB FFT estimate alone exceeds that
snapshot budget. With no empirical history, the existing input-size preflight
estimates only 2,235,564,032 bytes (about 2.08 GiB) and would allow launch.
This is an identified limitation of the static estimate; the hard process limit
remains separate. Neither estimate is a measured peak or proof of OOM.

A disposable job created by the benchmark's process-tree implementation was
queried directly and confirmed the 12,884,901,888-byte job limit. Inside it, the
native helper still returned 4 GiB and the isolated candidate returned 15.55 GiB.
Therefore physical-RAM detection does not account for the job budget. The child
also reported an immediate nested job with no memory limit, while the explicit
controller job handle had the 12 GiB limit and accounted for four processes.
Simply querying the child's immediate job would miss that enclosing constraint;
Microsoft documents that a null job handle selects the immediate job in a
[nested hierarchy](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject).

When the final spatial implementation is fixed, review a version-bound FFT-aware
preflight estimate and the complete process budget before enabling large cases.
Retain identical external resource rules across adapters. Do not turn the
15.55 GiB library workspace cap into permission to exceed the benchmark limit.

## Concrete Windows proposal and validation

The [review-only patch](upstream/pictologics-windows-ram.patch) adds a standard-
library `ctypes` call to Windows `GlobalMemoryStatusEx`, validates its result,
and retains the conservative 8 GiB assumption if the real query fails. It keeps
the existing valid POSIX query and 16 GiB/half-RAM rule. It adds no dependency.

The isolated candidate returned 16,698,271,744 bytes on this host. Five test
groups passed, covering the half-RAM and ceiling boundaries, invalid/failed
queries, the Windows API declaration, the POSIX arithmetic, and Windows behavior
with missing or failing `sysconf`. A copied intensity module with the patch
imported successfully; every other existing function remained source-identical.
The original scalar dispatch selected the predicted methods on all 283 cases.

The patch was not installed, released, submitted upstream or used for benchmark
timing. It does not implement available-memory/commit/job-budget awareness or an
alternate macOS query when `sysconf` fails. Native macOS validation and complete
final-release spatial/accuracy qualification remain required. The broader
[release acceptance checks](pictologics_release_feedback.md) still apply.

Local evidence is retained in `readiness-20261019/ram-investigation/`: Windows
API snapshots, per-case geometry and decisions, original-dispatch verification,
the isolated candidate and tests, and the disposable-job probe. Scientific
inputs and generated reports remain outside the source-only repository.

# Timing metrics methodology

## Unit of observation

A measured task is one adapter, case, grouped native workload, stored
representation, and fresh-process repeat. The adapter performs one warmup and
three measured observation windows. Only the first independent post-warmup
window may take the long-call shortcut when its per-call duration is at least
100 ms. Otherwise calibration uses 3–12 untimed windows; a later slow outlier
cannot bypass convergence. Calibration converges only when the latest three
per-call estimates have CV at most 5% and max/min at most 1.10.
The measured batch is sized from the fastest calibration estimate with 100%
headroom, is capped at 4096 calls, and every measured window must last at least
50 ms. Failure to converge or satisfy that hard minimum fails the task rather
than emitting a noisy observation. The calculation clocks are accumulated and
normalized per call; preparation and finalization are excluded for every call.
All calibration evidence, normalized samples, raw window totals, and call
counts are stored, and the normalized process median is the task duration.

The finalized warmup result is also the reference output. Every calibration
and measured native call must preserve the exact feature-name/shape structure,
contain only finite values, and be numerically equivalent at `rtol=1e-9` and
`atol=1e-12`. Comparison happens after the clock stops. A mismatch is a failed
task and never a timing observation.

The controller then applies the same name/order and value comparison across
the three fresh-process repeats for a case/adapter/workload. The first
committed repeat is the reference; later payloads retain its task ID and repeat
number. Resume reloads every committed payload, verifies its checksum, and
replays the cross-process comparisons before any new calculation begins.

Tasks execute serially. The controller sets OpenMP, BLAS, Numba, NumExpr, ITK,
and BLIS thread controls to the host's physical-core count before each adapter
process imports its libraries. This preserves native parallel implementations,
including Pictologics' Numba-parallel paths, without allowing concurrent tasks
to contend for the host. The exact thread policy is fingerprinted in RunSpec.

## Progress and ETA

The terminal ETA predicts wall-clock completion; it is not a sum of the
calculation-only samples. Each completed observation uses its ledger turnaround
(`finished_at - started_at`), which includes process startup, warmup,
calibration, calculation, validation, and task commit. Pending tasks are
predicted separately. An earlier fresh-process repeat of the same case is the
preferred reference. Otherwise, the controller fits a robust power law within
the same adapter, native workload, and synthetic scaling series, using ROI
voxel count (or total image voxels when ROI count is unavailable). A broader
adapter/workload fit and finally observed turnaround medians are fallbacks.
With three or more observed size levels, the fit includes a non-negative fixed
turnaround term so process startup/import overhead is not incorrectly scaled
with voxel count.

Voxel count already equals the product of the three image dimensions; it is
never cubed again. The learned exponent is constrained to the reviewed range
0–2: linear voxel traversal through the quadratic dense pairwise ROI work in
spatial autocorrelation. An uncertainty range is retained, and extrapolated
task times are capped at the combined startup/warmup and calculation timeout
bound. Pending tasks already covered by a persisted timeout cutoff contribute
zero calculation time to the ETA. An equal-sized fresh-process repeat is not
skipped; its ETA uses the controller turnaround observed for the prior timeout.
Until at least one measured task exists, ETA remains unavailable rather
than learning from fast unsupported or skipped records. The current
pillar's pending task specifications are fully known, so its ETA can be summed.
A numeric project ETA is deliberately withheld while later-pillar task shapes
and timings are not represented in the active ledger. Models never borrow
measurements from another source fingerprint, machine, or result directory.

Reports form candidate/baseline pairs only when `case_id`, workload, and repeat
match exactly. Unmatched and non-measured outcomes are excluded. The reported
ratio is candidate duration divided by Pictologics duration; values above one
mean Pictologics was faster.

## Timing boundary

Outside the timer:

- Python process startup and package import
- NIfTI loading and checksum verification
- binary-mask validation and resegmentation
- stored FBN/FBS representation construction and validation
- adapter-native preparation that is not feature calculation
- result normalisation and JSON serialisation

Inside the timer:

- matrix, mesh, and neighbourhood construction
- radiomic feature arithmetic

The controller records wall-clock samples from the adapter and process-tree CPU
and RSS observations from the host.

The controller also probes live power state immediately before and after every
task, with three short retries for transiently missing macOS profile output.
Both observations, timestamps, raw mode, portable tag, power source, probe
source, attempt count, and diagnostics are retained. A boundary change produces
the task tag `mixed-within-task`. Power state never gates execution or changes
the resume fingerprint.

## Timeout procedure

`--timeout` is a safety ceiling, not an imputed runtime. Startup/warmup has one
clock. Receipt of the `worker_ready` event starts a fresh calculation clock.
On expiry, the controller terminates the adapter process tree, waits
`--termination-grace`, escalates if required, and records:

- terminal status `timed_out_censored`
- timeout phase and lower-bound elapsed time
- any completed wall/CPU call samples
- process and memory observations available before termination

No timeout, failure, unsupported task, policy skip, interruption,
or pending task enters an estimate of measured duration.

The first timeout establishes a persistent cutoff only for the same adapter,
workload, mask/subject scaling series, and stored representation. Later planned
tasks in that exact scope with a strictly larger image voxel count are recorded
as `skipped_timeout_cutoff` without starting an adapter process. Equal-sized
repeats, different masks/configurations, unrelated real-world cases, and all
smaller images still run. The originating timeout, phase, lower bound, and
evidence task ID are retained.

## Resource policy

Memory estimation describes task requirements from static workload properties
and earlier measured process-tree peaks. Fixed interpreter/library RSS is not
scaled with image volume: the input-ratio exponent of 1.5 applies only to
observed growth above the lowest same-adapter/workload process baseline. The
maximum projected increment across strata is retained. Z-Rad spatial
autocorrelation also has a conservative quadratic diagnostic because it
materializes pairwise ROI-coordinate distances. In the publication protocol this estimate is
observe-only: `memory_estimate_exceeds_budget` is retained, but every task is
launched. Measured host/process-tree RSS, not this estimate, is the reported
memory result. Memory estimates never gate execution.

There is no predictive or relative-speed truncation in the publication
protocol. Only an actually observed timeout suppresses strictly larger images
in its exact adapter/workload/mask/configuration scaling series. This avoids
known larger-image timeouts without extrapolating across distinct inputs.

## Reporting

The SQLite ledger is authoritative. Report generation verifies the immutable
RunSpec fingerprint, task completeness, and every measured payload checksum.
Primary outputs retain raw samples, terminal-status counts, exact matched
pairs, host/environment provenance, feature-surface denominators, and report
artifact checksums.

QC treats failures, censored timeouts, timeout-cutoff skips, interruptions,
explicit development-only relative-speed policy skips,
unstable within-process samples, adapter stderr, incomplete phase memory,
missing feature outputs, unavailable task power, and in-task power changes as
explicit issues. Each issue carries adapter, workload, case, repeat, session,
and task-level power provenance; summaries count issues by type, severity,
adapter, and workload.

The six execution workloads are also the six reporting units: morphology,
spatial autocorrelation, local intensity, first-order intensity, texture, and
IVH. The first two partitions prevent pairwise spatial statistics from
dominating ordinary shape timing; the next two prevent spherical-neighbourhood
work from dominating basic intensity statistics. Texture remains grouped to
preserve native matrix sharing. No post-hoc runtime aggregation or
feature-count normalization is performed.

Do not pool pillars, convert censored bounds into durations, divide runtime by
feature count, or present a report as publication-attested when execution or
dataset verification is incomplete.

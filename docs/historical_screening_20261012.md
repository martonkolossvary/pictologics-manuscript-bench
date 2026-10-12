# Historical Windows screening: reviewed 12 October 2026

The prior full run finished on **20 September 2026 at 17:58:50 CEST**. All **25,470 planned task records** are terminal: **19,772 measured**, **417 timed out**, **1,582 skipped after a timeout**, **242 rejected by memory preflight**, **910 skipped after a memory cutoff**, and **2,547 unsupported**. Each pillar records `completed_with_failures`; no pending, running, interrupted, ordinary failed, or actual memory-limit-failure task remains.

**These are historical Pictologics 0.5.1 results, not measurements of the updated 0.7.1 release.** Use them to choose new screening cases. Do not transfer their cutoffs or timings into a new software version, host, or resource policy without fresh screening.

## Evidence and protocol

The source was the completed `full-mem12-15m-20260909` result namespace, with `pillar1_morphology/benchmark.sqlite3`, `pillar2_whole_anatomy/benchmark.sqlite3`, and `pillar3_ibsi2_phase3/benchmark.sqlite3`. On 12 October, the database, WAL and SHM files were copied while no benchmark Python process was visible, and the copies were opened read-only. Including the WAL is essential: reading only the immutable main database omitted the last completed tasks and falsely showed pending/running work. Original results and input files were not modified.

Counts and boundaries below come from `tasks.spec_json`, `tasks.record_json`, `timeout_cutoffs`, `memory_cutoffs` and `metadata`. Source identity comes from the final `run_meta.json` and `run_spec.json`. This inventory is not a new payload-checksum audit or numerical-equivalence audit.

- Final benchmark source commit: `5f659f9e44a4e9ea1868e5dcc5e07b15e23429bb`.
- Final execution-source SHA-256: `f84312f08eee2372593d5cc167642099350c2344ae7c153f4eb87c58381f3031`.
- Protocol: schema 15, `native_partitioned_workloads_3d_merge`, `manifest_harmonized` inputs, 8 requested threads and one adapter process tree at a time.
- Each successful task contains three observations in one fresh process; three fresh-process repeats were planned. Warmup and calibration are additional calls.
- The 900-second deadline covers the **whole worker task**, including import, preparation, warmup, calibration, observations and serialization. A timeout is not a measured 900-second feature-call runtime; many timed-out tasks had already produced one or two observations.
- Memory policy: `nearest_size_scoped_incremental_v2`, at most 12 GiB per Windows Job Object process tree, reduced when available RAM requires a reserve. Memory preflight estimates can suppress launch; these are not observed out-of-memory failures.
- The run retains measurements through documented memory-policy and execution-order amendments. Some early configurations have three timeout attempts; later policy skips equal-sized repeats. Counts are historical observed task counts, not a uniform one-attempt screening design.

| Adapter | Native version | Python | Measured | Timeouts | Timeout skips | Memory preflight | Memory skips | Unsupported |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pictologics | 0.5.1 | 3.12.10 | 4,927 | 15 | 146 | 2 | 4 | 0 |
| pyradiomics | 3.1.0 | 3.9.13 | 2,517 | 7 | 17 | 1 | 5 | 2,547 |
| mirp | 2.7.0 | 3.12.10 | 4,390 | 108 | 326 | 50 | 220 | 0 |
| medimage | 0.9.8 | 3.10.11 | 3,660 | 287 | 1,093 | 14 | 40 | 0 |
| zrad | 26.8.0 | 3.12.10 | 4,278 | 0 | 0 | 175 | 641 | 0 |

The 417 timeout tasks consumed approximately **104.25 worker-hours**. Memory exclusions consumed no native calculation time and must remain separate from this total. PyRadiomics supports morphology, intensity and texture here; its other three workloads are unsupported.

## Synthetic timeout boundaries

Pillar 1 has 120 cases: three image profiles (`high_contrast`, `low_contrast`, `reference`) × four masks (M1–M4) × sizes 32, 48, 64, 96, 128, 192, 256, 320, 384 and 512 cubed. “All profiles” below means each of those three profiles independently. Pillar 2 has the same ten sizes with profile `reference` and the dense A1 whole-anatomy mask.

Raw morphology, spatial-autocorrelation, local-intensity and intensity workloads use `original_hu_int16`; texture uses the shared mask-specific FBN32 representation. IVH uses the shared FBS1 representation. The displayed size is the observed boundary image size; larger same-series images may have been skipped rather than attempted.

| Dataset | Adapter | Workload | Profiles / masks | Observed timeout size | Timed-out tasks |
| --- | --- | --- | --- | --- | ---: |
| Pillar 1 | medimage | local_intensity | All profiles / M1, M2, M3, M4 | 64³ | 36 |
| Pillar 1 | medimage | spatial_autocorrelation | All profiles / M1, M3, M4 | 64³ | 27 |
| Pillar 1 | medimage | spatial_autocorrelation | All profiles / M2 | 96³ | 9 |
| Pillar 1 | medimage | texture | All profiles / M1, M2, M4 | 192³ | 9 |
| Pillar 1 | medimage | texture | All profiles / M3 | 256³ | 3 |
| Pillar 1 | mirp | local_intensity | All profiles / M1, M2, M3, M4 | 512³ | 12 |
| Pillar 1 | mirp | spatial_autocorrelation | All profiles / M1, M2, M3, M4 | 128³ | 36 |
| Pillar 1 | pictologics | spatial_autocorrelation | All profiles / M1, M2, M3, M4 | 256³ | 12 |
| Pillar 1 | pyradiomics | morphology | All profiles / M2, M3 | 512³ | 6 |
| Pillar 2 | medimage | local_intensity | reference / A1 | 48³ | 3 |
| Pillar 2 | medimage | spatial_autocorrelation | reference / A1 | 64³ | 3 |
| Pillar 2 | mirp | local_intensity | reference / A1 | 512³ | 1 |
| Pillar 2 | mirp | spatial_autocorrelation | reference / A1 | 64³ | 3 |
| Pillar 2 | pictologics | spatial_autocorrelation | reference / A1 | 128³ | 2 |
| Pillar 2 | pictologics | spatial_autocorrelation | reference / A1 | 192³ | 1 |
| Pillar 2 | pyradiomics | morphology | reference / A1 | 384³ | 1 |

Pictologics A1 spatial autocorrelation is a mixed-repeat boundary: 128³ repeat 1 succeeded, repeats 2 and 3 timed out, and 192³ repeat 1 also timed out. The final persistent cutoff is 128³ (1,199,081 ROI voxels). This is direct evidence that a single successful screening repeat does not guarantee all later repeats will fit the deadline.

Z-Rad had no observed timeout, but many spatial cases were never launched because of memory preflight. Neither no timeout nor a policy skip establishes that larger cases would finish.

## Synthetic memory-preflight boundaries

Every row below describes a **prediction that prevented launch**, not measured memory exhaustion. The final run has no task classified as an actual worker memory-limit failure. The same adapter/workload/profile/mask/representation cutoff then excludes equal-sized repeats and larger images. Estimate and available-budget differences explain why a prediction below 12 GiB can still reject a task.

| Dataset | Adapter | Workload | Profiles / masks | First rejected size | Rejected configurations | Estimate GiB |
| --- | --- | --- | --- | --- | ---: | ---: |
| Pillar 1 | medimage | morphology | All profiles / M1, M2, M3, M4 | 512³ | 12 | 22.36–22.91 |
| Pillar 1 | mirp | intensity | All profiles / M1, M2, M3, M4 | 384³ | 12 | 13.29–14.39 |
| Pillar 1 | mirp | ivh | All profiles / M1, M2, M3, M4 | 384³ | 12 | 13.65–14.40 |
| Pillar 1 | mirp | morphology | All profiles / M1, M2, M3, M4 | 384³ | 12 | 14.02–14.50 |
| Pillar 1 | mirp | texture | All profiles / M1, M3, M4; high_contrast / M2 | 512³ | 10 | 11.46–26.73 |
| Pillar 1 | zrad | intensity | All profiles / M1, M2, M3, M4 | 512³ | 12 | 18.23–18.32 |
| Pillar 1 | zrad | ivh | All profiles / M1, M2, M3, M4 | 512³ | 12 | 16.35–17.67 |
| Pillar 1 | zrad | local_intensity | All profiles / M1, M2, M3, M4 | 512³ | 12 | 14.41–14.42 |
| Pillar 1 | zrad | morphology | All profiles / M1, M2, M3, M4 | 512³ | 12 | 13.74–14.33 |
| Pillar 1 | zrad | spatial_autocorrelation | All profiles / M1, M2, M3, M4 | 64³ | 12 | 27.66–40.72 |
| Pillar 1 | zrad | texture | All profiles / M1, M2, M3, M4 | 512³ | 12 | 22.61–22.66 |
| Pillar 2 | medimage | morphology | reference / A1 | 384³ | 1 | 11.37 |
| Pillar 2 | medimage | texture | reference / A1 | 256³ | 1 | 16.77 |
| Pillar 2 | mirp | intensity | reference / A1 | 384³ | 1 | 13.86 |
| Pillar 2 | mirp | ivh | reference / A1 | 384³ | 1 | 14.34 |
| Pillar 2 | mirp | morphology | reference / A1 | 384³ | 1 | 14.30 |
| Pillar 2 | mirp | texture | reference / A1 | 384³ | 1 | 16.85 |
| Pillar 2 | pictologics | local_intensity | reference / A1 | 512³ | 1 | 12.76 |
| Pillar 2 | pictologics | texture | reference / A1 | 512³ | 1 | 18.31 |
| Pillar 2 | pyradiomics | texture | reference / A1 | 384³ | 1 | 18.03 |
| Pillar 2 | zrad | intensity | reference / A1 | 512³ | 1 | 18.03 |
| Pillar 2 | zrad | ivh | reference / A1 | 512³ | 1 | 17.27 |
| Pillar 2 | zrad | local_intensity | reference / A1 | 512³ | 1 | 14.58 |
| Pillar 2 | zrad | morphology | reference / A1 | 384³ | 1 | 11.60 |
| Pillar 2 | zrad | spatial_autocorrelation | reference / A1 | 32³ | 1 | 21.33 |
| Pillar 2 | zrad | texture | reference / A1 | 256³ | 1 | 18.19 |

MIRP texture at 512³ completed all three repeats for `low_contrast/M2` and `reference/M2`; the other ten profile/mask configurations were rejected. Thus a single adapter-level image-size cutoff would lose valid cases.

Z-Rad spatial preflight uses a quadratic ROI term (64 bytes per ROI-voxel pair), so ROI size matters more than the image edge length. The Pillar 1 64³ boundaries contain 21,346–25,977 ROI voxels; A1 is rejected already at 32³ with 18,694 ROI voxels. The resulting predicted footprints are not observed allocations.

## Real-world IBSI2 cases

The real-world cohort contains 51 subjects × CT, MRI and PET = 153 case/modality combinations. Each case has its own cutoff scope; a timeout on one patient does not justify suppressing another patient with the same image size. Texture uses shared FBN32 images. IVH uses FBS1 for CT and FBN1000 for MRI/PET.

| Adapter | Workload | Timed-out tasks | Distinct timed-out cases | Example case and shape | ROI voxels in example |
| --- | --- | ---: | ---: | --- | ---: |
| medimage | local_intensity | 99 | 99 | `STS_020_pet` (44 × 56 × 97) | 23,914 |
| medimage | spatial_autocorrelation | 96 | 96 | `STS_024_mri` (78 × 83 × 43) | 24,892 |
| medimage | texture | 2 | 2 | `STS_022_ct` (219 × 232 × 54) | 262,362 |
| mirp | spatial_autocorrelation | 56 | 52 | `STS_026_mri` (120 × 135 × 43) | 118,353 |

MEDimage texture timed out on exactly `STS_022_ct` (219 × 232 × 54; 262,362 ROI voxels) and `STS_021_ct` (219 × 216 × 83; 284,696 ROI voxels). MIRP spatial autocorrelation had 56 timeout tasks across 52 cases; some early repeats were retried. MEDimage local intensity had 99 distinct timed-out cases, and spatial autocorrelation had 96.

Pictologics completed all 2,754 real-world tasks. PyRadiomics completed all 1,377 supported real-world tasks. Neither had an observed real-world timeout or memory-preflight rejection in this historical run.

Z-Rad spatial autocorrelation was rejected by memory preflight on **97 distinct real-world cases**, followed by **194 repeat skips**; the other 56 cases completed all three spatial repeats. No other real-world workload had a memory exclusion. Rejected ROI sizes ranged from 14,910 to 748,026 voxels, with estimated footprints from 13.75 to 33,351.86 GiB. These are model estimates; they are not measured peak RSS or proof of native OOM.

## Dataset identity and portable reuse

| Dataset | Cases | Manifest SHA-256 |
| --- | ---: | --- |
| pillar1_morphology | 120 | `b394d744daf9ac950fa7ed8c73b7d08f1cb72e62dec8bcbbf7456bac38e15c1d` |
| pillar2_whole_anatomy | 10 | `5c18ee808f36d7e54e682f2c082eefaaff8a1eb861e33db70f85f83cd9f17dd5` |
| pillar3_ibsi2_phase3 | 153 | `ed733b22df33a5ebadb0b269025009e2f0ad24507bf5f9f307571d9df623c2d9` |

The complete inspected input workspace occupied 1,688,215,492 bytes (about 1.57 GiB). Its three dataset directories, including metadata, occupied 1,688,194,460 bytes: Pillar 1 1,006,179,330 bytes; Pillar 2 355,156,181 bytes; IBSI2 326,858,949 bytes. The balance is the workspace manifest and host attestation. This is input storage, not the larger result namespace with staged inputs and payloads.

For another host or a new Pictologics version, copy the prepared input workspace, create native adapter environments on that host, validate the copied inputs and start a new result namespace. Do not copy Windows virtual environments to macOS. Preserve the completed historical namespace unchanged.

`scripts/prepare_benchmark_workspace.py --validate-only` refreshes `workspace_manifest.json`; it is not a read-only command. Deep validation also binds generation/preparation source files and `poetry.lock`. If those files changed, identical input bytes alone do not satisfy current provenance: review the source change and use the repository’s explicit supported preparation/revalidation path rather than editing hashes to force acceptance.

On 12 October, a read-only comparison found that every recorded generator/preparer source-file hash, including `poetry.lock`, matched the reviewed checkout. This supports reusing the prepared inputs while those source files remain unchanged. It does not replace validation of file contents, geometry and voxel values in the copied workspace.

## Interpretation limits for a new comparison

This report establishes where this historical host/protocol stopped or declined calculation. It does not establish a universal performance boundary, validate macOS execution, certify numerical equivalence, or establish new-release performance. Adapter-native feature counts differ, and grouped texture workloads contain different native feature sets. Compare matched case/workload coverage and compatible feature semantics; do not infer speedup from total completed-task counts.

Run the current adapter versions with the intended timeout, thread and memory policy before the full benchmark. Keep actual timeout, predicted memory rejection, actual allocation failure, unsupported workload and ordinary error separate. A short screening protocol with fewer observations has a different whole-task deadline burden, so its boundary must not be presented as equivalent to this three-observation benchmark.

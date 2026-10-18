# Untimed readiness evidence, 2026-10-18

The final-run preset, unified release qualifier, robustness diagnostic and
Pictologics RAM-policy feedback package are implemented. Performance timing
remains paused. Existing exploratory ledgers and native environments were
preserved; these checks do not replace final matched timing on the publication
machine.

## Windows numerical evidence

These completed native evaluations use Pictologics 0.7.1, MEDimage 0.9.8,
MIRP 2.7.0, PyRadiomics 3.1.0 and Z-Rad 26.8.0. The Phase 1/2 evaluations
were verified with the capability-schema correction in commit `0bc2a9a`.
Each table cell is **passed / evaluated / all standardized checks**. Missing
native support is counted in the final denominator, not as a numerical pass.

| Adapter | IBSI 1 digital phantom | IBSI 2 Phase 1 response maps | IBSI 2 Phase 2 features |
| --- | --- | --- | --- |
| Pictologics | 169 / 169 / 169 | 28 / 28 / 33 | 197 / 197 / 323 |
| MEDimage | 168 / 168 / 169 | 27 / 27 / 33 | 252 / 252 / 323 |
| MIRP | 169 / 169 / 169 | 32 / 32 / 33 | 323 / 323 / 323 |
| PyRadiomics | 107 / 108 / 169 | 1 / 1 / 33 | 30 / 30 / 323 |
| Z-Rad | 165 / 169 / 169 | 26 / 26 / 33 | 252 / 252 / 323 |

The five IBSI 1 numerical differences remain PyRadiomics least-axis length,
Z-Rad global intensity peak and three Z-Rad histogram-gradient features. Their
interpretation is documented in the [earlier accuracy review](spatial_coverage_20261013.md).
No numerical failures were found among the evaluated standardized IBSI 2 rows.
Nonstandardized calculations remain separate: Phase 1 defines 36 tests per
adapter, and Phase 2 defines 396 calculations per adapter. The table deliberately
does not equate complete execution with full native coverage or certification.

The unified full run exposed a stale validator that accepted only Pictologics
capability schema `1.0.0`. The installed package emits `1.1.0`, whose reviewed
change adds the Gaussian filter entry. The validator now accepts these two
specific schemas and retains every execution-field and provenance check.
Unreviewed versions still fail. The original failed bundle was retained, and
the saved native candidate maps passed validation and both evaluations after
the correction. No package algorithms or recorded capability metadata were
modified.

## Robustness and spatial statistics

All 243 supported normal fixture/workload checks passed public/prepared/repeat
agreement, including all 81 supported checks on real CT/MRI/PET source crops.
The [robustness report](adapter_robustness.md) records 13 fixtures, their scope,
the 39 unsupported combinations and the distinct degenerate outcomes.
Three MEDimage degenerate-input errors and MIRP's finite spatial results for a
constant ROI remain review findings.

Pictologics, MEDimage, MIRP and Z-Rad calculate Moran's I (`N365`) and Geary's C
(`NPT7`); PyRadiomics does not implement the spatial workload. Small independent
formula checks include physical spacing and Pictologics' exact pairwise/FFT
branches. They do not establish large-volume feasibility.

The [RAM-policy reproducer](pictologics_release_feedback.md) confirms Pictologics
0.7.1's 4 GiB fallback cap on this approximately 31.10 GiB Windows host, compared
with the documented half-RAM cap of approximately 15.55 GiB. This remains an
upstream release acceptance item; the benchmark does not patch its environment.

## Reuse and remaining work

Keep all five qualification environments frozen. Preserve entire qualification
bundles, including binary fixtures/maps, because later baseline comparison
verifies every recorded artifact. A `needs_review` assessment can be complete;
it must retain the findings above rather than presenting them as passes.

Run the [full untimed qualification command](unified_release_qualification.md)
on the intended Mac with its own locked environments and validated workspace.
Native CI uses the bounded core scope; Windows scientific evidence cannot
substitute for a complete assessment on that machine. Requalify the final
Pictologics release in a separate environment, review feature and numerical
changes, then obtain authorization to start the [frozen final timing protocol](final_run_preset.md)
in a fresh result root. Earlier exploratory measurements remain separate.

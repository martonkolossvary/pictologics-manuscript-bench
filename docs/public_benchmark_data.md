# Public benchmark data

The Git repository contains no scientific payload. All inputs are materialised
under ignored `data/` paths from the checksum inventory in
`reproducibility/inputs/manifest.json`.

## Bootstrap

From a clean clone:

```bash
poetry run python scripts/bootstrap_reproducibility_inputs.py
poetry run python scripts/bootstrap_reproducibility_inputs.py --verify-only
```

The bootstrapper checks out the declared upstream repositories at exact
commits, copies only listed files, downloads the declared IBSI workbook, and
verifies every SHA-256 digest. Valid files are reused. A mismatched destination
fails closed; replacement requires the operator to inspect it and explicitly
use `--force`.

For an air-gapped or pre-populated source checkout, use repeatable
`--repository-source NAME=/absolute/path` options. Each checkout must be at the
commit declared by the input manifest.

## Workspace construction

Build all benchmark pillars serially:

```bash
poetry run python scripts/prepare_benchmark_workspace.py \
  --ibsi2-source data/ibsi2_validation \
  --output-root data/benchmark \
  --resume
```

Then perform deep validation without feature calculation:

```bash
poetry run python scripts/prepare_benchmark_workspace.py \
  --output-root data/benchmark \
  --validate-only
```

The workspace manifest binds each dataset manifest, the calculation endpoint,
the adapter order, task counts, timing policy, and preparation provenance.
Pillar 1 is recomputed byte-for-byte during deep validation; `--shallow` is
development-only and is not acceptable for qualification.

## Frozen inventory

| Pillar | Cases | Construction |
|---|---:|---|
| `pillar1` | 120 | 3 profiles × 4 masks × 10 cubic grids |
| `pillar2_a1` | 10 | 1 dense mask × 10 cubic grids |
| `ibsi2_phase3` | 153 | 51 subjects × CT/MRI/PET |

Every logical case contains an original image, a binary mask, a stored
mask-specific FBN32 texture representation, and a stored IVH representation.
The source and stored files, geometry, spacing, dtype, affine, ROI statistics,
configured levels, occupied levels, and derivation hashes are recorded in the
dataset manifests.

Pillar 3 files are copied byte-for-byte from `data/ibsi2_validation`; they are
not cropped, resampled, scaled, or assigned synthetic grid sizes.

## Publication boundary

Do not commit `data/` or a packaged copy of the workspace. A publication clone
must reproduce it from the pinned source manifest. Attribution and license
files fetched by the bootstrapper must accompany any separately distributed
input bundle.

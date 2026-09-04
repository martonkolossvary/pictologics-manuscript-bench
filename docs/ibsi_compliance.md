# IBSI compliance workflows

Compliance establishes accuracy and feature coverage independently of the
performance benchmark. This source repository contains no imported reference
tables, candidate maps, comparison rows, or compliance reports.

## Prepare references

Bootstrap the pinned inputs first:

```bash
poetry run python scripts/bootstrap_reproducibility_inputs.py
```

Import the official IBSI 1 workbook:

```bash
poetry run python -m bench.cli compliance import-ibsi1 \
  --workbook data/downloads/IBSI-1-submission-table.xlsx \
  --output-dir data/compliance/ibsi1-reference
```

Validate the IBSI 2 Phase 1 reference bundle:

```bash
poetry run python -m bench.cli compliance validate-ibsi2-phase1 \
  --reference-dir data/ibsi2/references/phase1/maps \
  --manifest-out data/compliance/ibsi2-phase1-reference-manifest.json
```

Import the IBSI 2 Phase 2 reference table:

```bash
poetry run python -m bench.cli compliance import-ibsi2-phase2 \
  --csv data/ibsi2/references/phase2/source/reference_values.csv \
  --output-dir data/compliance/ibsi2-phase2-reference
```

The importers verify known source hashes and fail closed unless the operator
explicitly authorises an unknown hash for investigation. Such an override is
not publication-attested.

## IBSI 1 digital phantom

```bash
poetry run python -m bench.cli compliance run-ibsi1 \
  --image data/ibsi1/digital_phantom/image/phantom.nii.gz \
  --mask data/ibsi1/digital_phantom/mask/mask.nii.gz \
  --references data/compliance/ibsi1-reference/reference_values.csv \
  --reference-manifest data/compliance/ibsi1-reference/reference_manifest.json \
  --output-dir results/compliance/ibsi1 \
  --resume
```

This evaluates the declared 3D-merged digital-phantom profile using official
reference values and tolerances. Native support, finite output, referencable
rows, evaluated checks, and passing checks remain separate denominators.

## IBSI 2 response maps

Generate package-native candidate maps:

```bash
poetry run python -m bench.cli compliance generate-ibsi2-candidates \
  --output-dir results/compliance/ibsi2-candidates \
  --phases phase1,phase2 \
  --resume
```

Evaluate Phase 1:

```bash
poetry run python -m bench.cli compliance evaluate-ibsi2-phase1 \
  --reference-manifest data/compliance/ibsi2-phase1-reference-manifest.json \
  --reference-dir data/ibsi2/references/phase1/maps \
  --candidate-manifest results/compliance/ibsi2-candidates/candidate_manifest.json \
  --output-dir results/compliance/ibsi2-phase1
```

Evaluate Phase 2 statistics:

```bash
poetry run python -m bench.cli compliance run-ibsi2-phase2 \
  --candidate-manifest results/compliance/ibsi2-candidates/candidate_manifest.json \
  --references data/compliance/ibsi2-phase2-reference/reference_values.csv \
  --reference-manifest data/compliance/ibsi2-phase2-reference/reference_manifest.json \
  --output-dir results/compliance/ibsi2-phase2 \
  --resume
```

Candidate manifests bind package versions, configurations, paths, and hashes.
Evaluation rejects unattested or changed candidate files. Compliance outputs
stay under ignored `results/` and are generated anew by each reproduction.

# Three-pillar benchmark design

The pillars answer different scaling questions and are reported separately.
They are not a pooled factorial dataset.

## Pillar 1: controlled 3D morphology

`configs/benchmark/pillar1.json` defines ten cubic grids from 32³ to 512³
over a fixed 256 mm field of view, three CT-inspired intensity/texture
profiles, and four alternative masks. X, Y, Z, and isotropic spacing change
together; no slice is repeated or tiled.

- M1: connected whole tumour including the visible necrotic region.
- M2: viable shell excluding the enclosed necrotic cavity.
- M3: M1 with attached tapered 3D spicules.
- M4: M1 with six image-visible satellite foci.

The masks share the same raw scene within each profile/grid combination so the
effect of mask geometry is paired. M3 is a segmentation-boundary stress, not a
separately rendered biological phenotype. The synthetic scene is an analytic
CT-inspired phantom, not a diagnostic patient image.

## Pillar 2: dense whole anatomy

`configs/benchmark/pillar2_a1.json` uses the reference scene on the same ten
grids. A1 includes generated non-air anatomy while retaining a background
guard at every image face. It tests dense-ROI image and memory scaling and is
not a fifth tumour phenotype or a bounding-box crop.

## Pillar 3: fixed real-world variation

The IBSI 2 Phase 3 cohort contributes all complete CT, MRI, and PET cases from
51 subject blocks. Original images and masks are copied without spatial or
value transformation. Geometry and voxel counts are measured from the files
and bound in the manifest.

## Common representation contract

Raw families use the original image. Histogram and texture families use the
stored mask-specific FBN32 index image with identity adapter discretization.
IVH uses a stored mask-specific index image: FBS1 for CT/synthetic data and
FBN1000 for MRI/PET. This makes the representation identical across adapters
and keeps representation construction outside the timed region.

## Interpretation rules

- The execution comparison unit is the same case, grouped workload, stored
  representation, and fresh-process repeat.
- Each package runs its declared current native workload surface. Equal group
  labels do not imply equal individual output counts.
- Morphology, spatial autocorrelation, local intensity, first-order intensity,
  texture, and IVH remain separate execution and reporting workloads. Moran's
  I and Geary's C are isolated from morphology because their pairwise scaling
  is exceptional; local peaks are isolated from first-order statistics because
  they require a physical spherical neighbourhood. IVH uses a different stored
  representation.
- Texture remains one native workload so packages retain shared matrix-building
  advantages, including Pictologics' shared GLSZM/GLDZM zone traversal. Package
  output counts and IBSI coverage disclose unsupported families such as
  PyRadiomics GLDZM.
- Report native output count and IBSI coverage beside runtime.
- Do not normalise runtime by feature count.
- Treat a timeout as a censored observation and explicitly skip only strictly
  larger images in the same adapter/workload/mask/configuration scaling series.
- Report synthetic and real-world pillars separately.

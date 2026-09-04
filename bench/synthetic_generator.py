"""Deterministic volumetric generator for the public benchmark design.

This module creates the analytic source arrays for the frozen Pillar 1
morphology dataset and the Pillar 2 dense whole-anatomy workload.  Geometry is
fixed in physical coordinates across the resolution ladder and keeps a complete
background-voxel guard on all six faces at the coarsest ``32^3`` grid.  The
generator never imports or invokes a radiomics adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from scipy import ndimage

from bench.synthetic_scene import (
    HU_PROFILES,
    SyntheticSceneConfig,
    _affine,
    _anatomy_slab,
    _axis_centres,
    _ellipsoid_score,
    _retain_component_overlapping_primary,
    _slab_indices,
    _tapered_spicule,
    _texture,
)


SYNTHETIC_VOLUMETRIC_GENERATOR = "synthetic_volumetric_phantom"
SYNTHETIC_VOLUMETRIC_GENERATOR_VERSION = 2
SYNTHETIC_VOLUMETRIC_GENERATOR_STATUS = "production_three_pillar_geometry"


SYNTHETIC_MASK_IDS = ("M1", "M2", "M3", "M4", "A1")
REFERENCE_FOV_MM = 256.0
COARSEST_EDGE = 32
COARSE_GUARD_VOXELS = 1
# Maximal radii on a 0.25-mm deterministic search grid for the frozen curved,
# lobulated body model below.  Increasing either radius by 0.25 mm activates a
# face voxel at 32^3; this pair leaves exactly one background voxel on all six
# faces while maximising the observed coarse-grid A1 voxel count.
DENSE_BODY_RADII_MM = (110.75, 117.5)
DENSE_BODY_TAPER_FRACTION = 0.04
DENSE_BODY_Z_HALF_EXTENT_MM = 120.0


@dataclass(frozen=True)
class SyntheticVolume:
    """One generated volume and its registered mask catalogue."""

    image: np.ndarray
    labels: np.ndarray
    masks: Mapping[str, np.ndarray]
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MaskMetric:
    mask_id: str
    voxels: int
    volume_ml: float
    image_fraction: float
    body_fraction: float
    bbox_shape_xyz: tuple[int, int, int]
    face_margins_voxels: tuple[int, int, int, int, int, int]
    minimum_face_margin_voxels: int
    occupied_z_slices: int
    occupied_z_mm: float
    components_26: int
    cavities_background_6: int
    bone_overlap_voxels: int
    within_1p5mm_of_bone_voxels: int
    within_3p5mm_of_bone_voxels: int


@dataclass(frozen=True)
class SyntheticMetrics:
    shape_xyz: tuple[int, int, int]
    spacing_mm_xyz: tuple[float, float, float]
    image_voxels: int
    body_voxels: int
    body_fraction: float
    masks: Mapping[str, MaskMetric]
    adjacent_slice_r_z: float
    adjacent_slice_pairs: int
    identical_adjacent_image_slices: int
    distinct_image_slices: int
    distinct_label_slices: int


def _warped_xy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world coordinates to a visibly curved and gently twisting limb."""

    angle = 0.11 * np.sin(z / 61.0) + 0.075 * z / 128.0
    cosine = np.cos(angle)
    sine = np.sin(angle)
    shift_x = 5.5 * np.sin(z / 52.0) + 1.8 * z / 128.0
    shift_y = 3.8 * np.sin(z / 43.0 + 0.55)
    centered_x = x - shift_x
    centered_y = y - shift_y
    return (
        cosine * centered_x + sine * centered_y,
        -sine * centered_x + cosine * centered_y,
    )


def _body_and_soft_support(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    body_radii_mm: tuple[float, float] = DENSE_BODY_RADII_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body, non-bone soft tissue, and the warped femoral cylinder."""

    xw, yw = _warped_xy(x, y, z)
    z_fraction = z / np.float32(128.0)
    center_x = 2.5 + 2.0 * np.sin(z / 67.0)
    center_y = 1.0 - 1.5 * np.cos(z / 74.0)
    taper = 1.0 - DENSE_BODY_TAPER_FRACTION * z_fraction**2
    rx = float(body_radii_mm[0]) * taper
    ry = float(body_radii_mm[1]) * taper
    dx = xw - center_x
    dy = yw - center_y
    theta = np.arctan2(dy / ry, dx / rx)
    outer_radius = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    outer_limit = (
        1.0
        + 0.035 * np.sin(theta + 0.45)
        + 0.024 * np.sin(3.0 * theta + z / 45.0)
        + 0.010 * np.cos(5.0 * theta - z / 61.0)
    )
    body = (outer_radius <= outer_limit) & (np.abs(z) < DENSE_BODY_Z_HALF_EXTENT_MM)
    femur_x = -19.0 + 1.4 * np.sin(z / 58.0)
    femur_y = 8.0 + 1.0 * np.cos(z / 61.0)
    femur_r2 = (xw - femur_x) ** 2 + (yw - femur_y) ** 2
    bone = body & (femur_r2 <= 14.5**2)
    return np.asarray(body), np.asarray(body & ~bone), np.asarray(bone)


def _large_primary_score(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Curved lobulated peri-femoral lesion in physical coordinates."""

    center_x = 42.0 + 7.0 * np.sin(z / 52.0)
    center_y = -7.0 + 5.0 * np.sin(z / 43.0 + 0.70)
    dx = x - center_x
    dy = y - center_y
    theta = np.arctan2(dy / 58.0, dx / 65.0)
    radial = np.sqrt((dx / 65.0) ** 2 + (dy / 58.0) ** 2 + (z / 104.0) ** 2)
    boundary = (
        1.0
        + 0.045 * np.sin(3.0 * theta + z / 39.0)
        + 0.025 * np.cos(5.0 * theta - z / 47.0)
        + 0.015 * np.sin(2.0 * theta - z / 31.0)
    )
    return np.asarray(boundary - radial, dtype=np.float32)


def _large_core(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    center_x = 50.0 + 4.0 * np.sin((z - 5.0) / 55.0)
    center_y = -11.0 + 3.0 * np.sin(z / 47.0 + 0.4)
    score = (
        1.0
        - ((x - center_x) / 36.0) ** 2
        - ((y - center_y) / 29.0) ** 2
        - ((z - 4.0) / 57.0) ** 2
    )
    return np.asarray(score >= 0)


def _spicule_segments() -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float], float, float],
    ...,
]:
    """Return eighteen deterministic, spatially distributed 3D projections."""

    segments = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    center = np.asarray((42.0, -7.0, 0.0), dtype=float)
    radii = np.asarray((65.0, 58.0, 104.0), dtype=float)
    for index in range(18):
        unit_z = 1.0 - 2.0 * (index + 0.5) / 18.0
        unit_xy = np.sqrt(max(0.0, 1.0 - unit_z**2))
        phi = index * golden_angle + 0.37
        direction = np.asarray(
            (unit_xy * np.cos(phi), unit_xy * np.sin(phi), unit_z),
            dtype=float,
        )
        start = center + 0.74 * radii * direction
        end = center + radii * direction + (30.0 + 4.0 * (index % 4)) * direction
        segments.append(
            (
                tuple(float(value) for value in start),
                tuple(float(value) for value in end),
                9.0,
                2.8,
            )
        )
    return tuple(segments)


LARGE_SPICULE_SEGMENTS = _spicule_segments()


def _large_lesion_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    soft_tissue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score = _large_primary_score(x, y, z)
    primary = (score >= 0) & soft_tissue
    core = _large_core(x, y, z) & primary
    spicules = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=bool)
    for start, end, base_radius, tip_radius in LARGE_SPICULE_SEGMENTS:
        spicules |= _tapered_spicule(
            x,
            y,
            z,
            start,
            end,
            base_radius,
            tip_radius,
        )
    spiculated = (primary | spicules) & soft_tissue

    satellite_specs = (
        ((-62.0, -57.0, 70.0), (10.0, 9.0, 12.0)),
        ((-58.0, 62.0, 68.0), (10.0, 9.0, 12.0)),
        ((-62.0, -60.0, -67.0), (10.0, 9.0, 12.0)),
        ((-58.0, 66.0, -65.0), (10.0, 9.0, 12.0)),
        ((0.0, 86.0, 0.0), (10.0, 8.0, 11.0)),
        ((0.0, -92.0, 0.0), (10.0, 8.0, 11.0)),
    )
    satellites = np.zeros_like(primary)
    for center, radii in satellite_specs:
        satellites |= _ellipsoid_score(x, y, z, center, radii) >= 0
    satellites &= soft_tissue & ~primary
    return primary, core, spiculated, satellites


def generate_synthetic_volume(
    config: SyntheticSceneConfig = SyntheticSceneConfig(),
    *,
    slab_order: Literal["forward", "reverse"] = "forward",
) -> SyntheticVolume:
    """Generate actual review arrays without invoking benchmark adapters."""

    if config.edge < COARSEST_EDGE:
        raise ValueError(
            f"volumetric geometry requires edge >= {COARSEST_EDGE}; "
            "smaller grids cannot preserve its one-voxel face guard"
        )
    if not np.isclose(config.fov_mm, REFERENCE_FOV_MM):
        raise ValueError(
            f"volumetric geometry is frozen to a {REFERENCE_FOV_MM:g}-mm field"
        )

    image = np.empty(config.shape, dtype=np.int16)
    labels = np.empty(config.shape, dtype=np.uint8)
    masks = {
        name: np.zeros(config.shape, dtype=np.uint8) for name in SYNTHETIC_MASK_IDS
    }
    profile = HU_PROFILES[config.hu_profile]
    axis = _axis_centres(config)
    x = axis[:, None, None]
    y = axis[None, :, None]

    for z_start, z_stop in _slab_indices(config, slab_order):
        z = axis[None, None, z_start:z_stop]
        xw, yw = _warped_xy(x, y, z)
        slab_image, slab_labels = _anatomy_slab(
            xw,
            yw,
            z,
            seed=config.seed,
            hu_profile=profile,
            body_radii_mm=DENSE_BODY_RADII_MM,
            body_taper_fraction=DENSE_BODY_TAPER_FRACTION,
            body_z_half_extent_mm=DENSE_BODY_Z_HALF_EXTENT_MM,
        )
        body, soft_tissue, _bone = _body_and_soft_support(x, y, z)
        primary, core, spiculated, satellites = _large_lesion_fields(
            x, y, z, soft_tissue
        )
        viable = primary & ~core
        tumour_texture = _texture(
            x, y, z, seed=config.seed, name="large_tumour", correlation_mm=12.0
        )
        slab_image[viable] = (
            float(profile["viable_hu"])
            + float(profile["viable_texture_hu"]) * tumour_texture
        )[viable]
        slab_image[core] = (
            float(profile["necrotic_hu"])
            + float(profile["necrotic_texture_hu"]) * tumour_texture
        )[core]
        slab_image[satellites] = (
            float(profile["satellite_hu"])
            + float(profile["satellite_texture_hu"]) * tumour_texture
        )[satellites]
        slab_labels[viable] = 8
        slab_labels[core] = 9
        slab_labels[satellites] = 10
        scanner = _texture(
            x, y, z, seed=config.seed, name="large_scanner", correlation_mm=3.3
        )
        slab_image += float(profile["scanner_noise_hu"]) * scanner

        image[:, :, z_start:z_stop] = np.rint(
            np.clip(slab_image, -1024.0, 2000.0)
        ).astype(np.int16)
        labels[:, :, z_start:z_stop] = slab_labels
        masks["M1"][:, :, z_start:z_stop] = primary
        masks["M2"][:, :, z_start:z_stop] = viable
        masks["M3"][:, :, z_start:z_stop] = spiculated
        masks["M4"][:, :, z_start:z_stop] = primary | satellites
        masks["A1"][:, :, z_start:z_stop] = body

    masks["M3"] = _retain_component_overlapping_primary(masks["M3"], masks["M1"])
    a1_margins = _face_margins_voxels(masks["A1"] != 0)
    if min(a1_margins) < COARSE_GUARD_VOXELS:
        raise RuntimeError(
            "whole-anatomy support violates the one-voxel image-face guard: "
            f"observed margins={a1_margins}"
        )
    metadata = {
        "generator": SYNTHETIC_VOLUMETRIC_GENERATOR,
        "generator_version": SYNTHETIC_VOLUMETRIC_GENERATOR_VERSION,
        "status": SYNTHETIC_VOLUMETRIC_GENERATOR_STATUS,
        "shape_xyz": list(config.shape),
        "spacing_mm_xyz": list(config.spacing_mm),
        "fov_mm_xyz": [config.fov_mm] * 3,
        "coordinate_system": "RAS+",
        "array_axis_order": "xyz",
        "hu_profile_id": config.hu_profile,
        "seed": config.seed,
        "geometry_contract": {
            "reference_fov_mm": REFERENCE_FOV_MM,
            "coarsest_edge": COARSEST_EDGE,
            "coarse_guard_voxels_all_faces": COARSE_GUARD_VOXELS,
            "body_radii_mm_xy": list(DENSE_BODY_RADII_MM),
            "maximization_search_step_mm": 0.25,
            "maximality_witness": (
                "increasing either transverse radius by 0.25 mm activates "
                "an N=32 face voxel"
            ),
            "body_taper_fraction": DENSE_BODY_TAPER_FRACTION,
            "body_z_half_extent_mm": DENSE_BODY_Z_HALF_EXTENT_MM,
            "observed_a1_face_margins_voxels": list(a1_margins),
        },
        "mask_definitions": {
            "M1": "dense curved peri-femoral whole tumour including necrosis",
            "M2": "viable shell excluding the image-visible necrotic cavity",
            "M3": "M1 with eighteen attached tapered three-dimensional spicules",
            "M4": "M1 plus six image-visible satellite foci",
            "A1": "dense whole anatomy including bone and pathology",
        },
        "spicule_count": len(LARGE_SPICULE_SEGMENTS),
        "satellite_count": 6,
    }
    return SyntheticVolume(
        image=image,
        labels=labels,
        masks=masks,
        affine=_affine(config),
        spacing_mm=config.spacing_mm,
        metadata=metadata,
    )


def _slice_digest_count(volume: np.ndarray) -> int:
    return len(
        {
            np.ascontiguousarray(volume[:, :, index]).tobytes()
            for index in range(volume.shape[2])
        }
    )


def _face_margins_voxels(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """Return low/high background margins for x, y, and z image faces."""

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("cannot measure image-face margins of an empty mask")
    low = np.min(coordinates, axis=0)
    high = np.max(coordinates, axis=0)
    shape = np.asarray(mask.shape, dtype=int)
    margins = (
        int(low[0]),
        int(shape[0] - 1 - high[0]),
        int(low[1]),
        int(shape[1] - 1 - high[1]),
        int(low[2]),
        int(shape[2] - 1 - high[2]),
    )
    return margins


def _cavity_count(mask: np.ndarray) -> int:
    background = ~mask
    labels, count = ndimage.label(
        background,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    exterior = np.unique(
        np.concatenate(
            (
                labels[0, :, :].ravel(),
                labels[-1, :, :].ravel(),
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            )
        )
    )
    return int(sum(identifier not in exterior for identifier in range(1, count + 1)))


def evaluate_synthetic_volume(
    bundle: SyntheticVolume,
) -> SyntheticMetrics:
    """Measure topology, Z support, burden, and longitudinal correlation."""

    image = np.asarray(bundle.image)
    labels = np.asarray(bundle.labels)
    spacing = tuple(float(value) for value in bundle.spacing_mm)
    image_voxels = int(image.size)
    body = np.asarray(bundle.masks["A1"]) != 0
    body_voxels = int(np.count_nonzero(body))
    bone = np.isin(labels, (6, 7))
    distance_to_bone = ndimage.distance_transform_edt(~bone, sampling=spacing)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    records: dict[str, MaskMetric] = {}
    voxel_volume_mm3 = float(np.prod(spacing))
    for mask_id, source in bundle.masks.items():
        mask = np.asarray(source) != 0
        coordinates = np.argwhere(mask)
        if coordinates.size == 0:
            raise ValueError(f"synthetic mask {mask_id} is empty")
        low = np.min(coordinates, axis=0)
        high = np.max(coordinates, axis=0)
        z_slices = int(high[2] - low[2] + 1)
        _, component_count = ndimage.label(mask, structure=structure)
        is_tumour = mask_id != "A1"
        face_margins = _face_margins_voxels(mask)
        records[mask_id] = MaskMetric(
            mask_id=mask_id,
            voxels=int(coordinates.shape[0]),
            volume_ml=float(coordinates.shape[0] * voxel_volume_mm3 / 1000.0),
            image_fraction=float(coordinates.shape[0] / image_voxels),
            body_fraction=float(coordinates.shape[0] / body_voxels),
            bbox_shape_xyz=tuple(int(value) for value in high - low + 1),
            face_margins_voxels=face_margins,
            minimum_face_margin_voxels=min(face_margins),
            occupied_z_slices=z_slices,
            occupied_z_mm=float(z_slices * spacing[2]),
            components_26=int(component_count),
            cavities_background_6=_cavity_count(mask),
            bone_overlap_voxels=(
                int(np.count_nonzero(mask & bone))
                if is_tumour
                else int(np.count_nonzero(mask & bone))
            ),
            within_1p5mm_of_bone_voxels=(
                int(np.count_nonzero(mask & (distance_to_bone <= 1.5)))
                if is_tumour
                else 0
            ),
            within_3p5mm_of_bone_voxels=(
                int(np.count_nonzero(mask & (distance_to_bone <= 3.5)))
                if is_tumour
                else 0
            ),
        )

    m1 = np.asarray(bundle.masks["M1"]) != 0
    pair_mask = m1[:, :, :-1] & m1[:, :, 1:]
    first = image[:, :, :-1][pair_mask].astype(np.float64)
    second = image[:, :, 1:][pair_mask].astype(np.float64)
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        raise ValueError("adjacent-slice correlation is undefined")
    identical = sum(
        np.array_equal(image[:, :, index], image[:, :, index + 1])
        for index in range(image.shape[2] - 1)
    )
    return SyntheticMetrics(
        shape_xyz=tuple(int(value) for value in image.shape),
        spacing_mm_xyz=spacing,
        image_voxels=image_voxels,
        body_voxels=body_voxels,
        body_fraction=float(body_voxels / image_voxels),
        masks=records,
        adjacent_slice_r_z=float(np.corrcoef(first, second)[0, 1]),
        adjacent_slice_pairs=int(first.size),
        identical_adjacent_image_slices=int(identical),
        distinct_image_slices=_slice_digest_count(image),
        distinct_label_slices=_slice_digest_count(labels),
    )


def measure_spicule_external_voxels(
    bundle: SyntheticVolume,
) -> tuple[int, ...]:
    """Count actual retained exterior voxels contributed by every M3 branch.

    Counts are deliberately measured after voxelisation, body clipping, bone
    exclusion, and final connected-component retention. A positive value means
    the declared analytic branch is represented outside M1 on that grid.
    """

    shape = tuple(int(value) for value in bundle.image.shape)
    if len(set(shape)) != 1:
        raise ValueError("spicule audit requires a cubic synthetic grid")
    axis = np.arange(shape[0], dtype=np.float32) * float(bundle.affine[0, 0]) + float(
        bundle.affine[0, 3]
    )
    x = axis[:, None, None]
    y = axis[None, :, None]
    z = axis[None, None, :]
    _body, soft_tissue, _bone = _body_and_soft_support(x, y, z)
    retained_exterior = (np.asarray(bundle.masks["M3"]) != 0) & ~(
        np.asarray(bundle.masks["M1"]) != 0
    )
    return tuple(
        int(
            np.count_nonzero(
                _tapered_spicule(
                    x,
                    y,
                    z,
                    start,
                    end,
                    base_radius,
                    tip_radius,
                )
                & soft_tissue
                & retained_exterior
            )
        )
        for start, end, base_radius, tip_radius in LARGE_SPICULE_SEGMENTS
    )


def measure_satellite_component_voxels(
    bundle: SyntheticVolume,
) -> tuple[int, ...]:
    """Return sorted actual voxel counts for the six M4-only components."""

    satellites = (np.asarray(bundle.masks["M4"]) != 0) & ~(
        np.asarray(bundle.masks["M1"]) != 0
    )
    labels, count = ndimage.label(
        satellites, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    return tuple(
        sorted(
            int(np.count_nonzero(labels == identifier))
            for identifier in range(1, count + 1)
        )
    )


def measure_synthetic_scale(
    edge: int,
    *,
    fov_mm: float = 256.0,
    slab_depth: int = 8,
) -> Mapping[str, int | float]:
    """Measure M1/A1 scale without materialising a full image volume."""

    if edge < COARSEST_EDGE:
        raise ValueError(f"edge must be >= {COARSEST_EDGE}")
    if not np.isclose(fov_mm, REFERENCE_FOV_MM):
        raise ValueError(f"fov_mm must equal {REFERENCE_FOV_MM:g}")

    config = SyntheticSceneConfig(edge=edge, fov_mm=fov_mm, slab_depth=slab_depth)
    axis = _axis_centres(config)
    x = axis[:, None, None]
    y = axis[None, :, None]
    m1_count = 0
    a1_count = 0
    m1_z = 0
    a1_z = 0
    m1_low = np.full(3, edge, dtype=int)
    m1_high = np.full(3, -1, dtype=int)
    a1_low = np.full(3, edge, dtype=int)
    a1_high = np.full(3, -1, dtype=int)
    for z_start, z_stop in _slab_indices(config, "forward"):
        z = axis[None, None, z_start:z_stop]
        body, soft, _bone = _body_and_soft_support(x, y, z)
        primary = (_large_primary_score(x, y, z) >= 0) & soft
        m1_count += int(np.count_nonzero(primary))
        a1_count += int(np.count_nonzero(body))
        m1_z += int(np.count_nonzero(np.any(primary, axis=(0, 1))))
        a1_z += int(np.count_nonzero(np.any(body, axis=(0, 1))))
        for mask, low, high in (
            (primary, m1_low, m1_high),
            (body, a1_low, a1_high),
        ):
            coordinates = np.argwhere(mask)
            if coordinates.size:
                coordinates[:, 2] += z_start
                low[:] = np.minimum(low, np.min(coordinates, axis=0))
                high[:] = np.maximum(high, np.max(coordinates, axis=0))

    def margins(low: np.ndarray, high: np.ndarray) -> tuple[int, ...]:
        return (
            int(low[0]),
            int(edge - 1 - high[0]),
            int(low[1]),
            int(edge - 1 - high[1]),
            int(low[2]),
            int(edge - 1 - high[2]),
        )

    a1_margins = margins(a1_low, a1_high)
    if min(a1_margins) < COARSE_GUARD_VOXELS:
        raise RuntimeError(f"A1 violates the face guard at edge={edge}: {a1_margins}")
    return {
        "edge": int(edge),
        "spacing_mm": float(fov_mm / edge),
        "image_voxels": int(edge**3),
        "m1_voxels": m1_count,
        "m1_image_fraction": float(m1_count / edge**3),
        "m1_z_slices": m1_z,
        "m1_z_mm": float(m1_z * fov_mm / edge),
        "m1_face_margins_voxels": margins(m1_low, m1_high),
        "a1_voxels": a1_count,
        "a1_image_fraction": float(a1_count / edge**3),
        "a1_z_slices": a1_z,
        "a1_z_mm": float(a1_z * fov_mm / edge),
        "a1_face_margins_voxels": a1_margins,
    }


__all__ = [
    "COARSEST_EDGE",
    "COARSE_GUARD_VOXELS",
    "DENSE_BODY_RADII_MM",
    "DENSE_BODY_TAPER_FRACTION",
    "DENSE_BODY_Z_HALF_EXTENT_MM",
    "LARGE_SPICULE_SEGMENTS",
    "MaskMetric",
    "SYNTHETIC_MASK_IDS",
    "SyntheticMetrics",
    "SyntheticVolume",
    "evaluate_synthetic_volume",
    "generate_synthetic_volume",
    "measure_synthetic_scale",
    "measure_satellite_component_voxels",
    "measure_spicule_external_voxels",
]

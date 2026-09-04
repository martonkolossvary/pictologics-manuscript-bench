"""Current analytic CT-inspired scene primitives for Benchmark.

Array axes and affine columns are ``x, y, z`` in RAS+ physical space. Only
output arrays are volume-sized; rendering temporaries are bounded by
``slab_depth``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

# The profiles share geometry and the same normalised texture field.  Only the
# HU calibration and texture/noise amplitudes differ.  Values are engineering
# choices for a CT-inspired benchmark, not claims about scanner protocols.
HU_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "low_contrast": {
        "display_label": "low contrast",
        "fat_hu": -82.0,
        "muscle_hu": (45.0, 49.0, 42.0, 47.0),
        "muscle_texture_hu": 8.0,
        "fascia_hu": -18.0,
        "fascia_texture_hu": 5.0,
        "fat_texture_hu": 6.0,
        "cortex_hu": 860.0,
        "cortex_texture_hu": 30.0,
        "marrow_hu": 120.0,
        "marrow_texture_hu": 16.0,
        "viable_hu": 39.0,
        "viable_texture_hu": 15.0,
        "necrotic_hu": 16.0,
        "necrotic_texture_hu": 5.0,
        "satellite_hu": 41.0,
        "satellite_texture_hu": 13.0,
        "scanner_noise_hu": 2.0,
    },
    "reference": {
        "display_label": "reference",
        "fat_hu": -92.0,
        "muscle_hu": (47.0, 53.0, 43.0, 50.0),
        "muscle_texture_hu": 10.0,
        "fascia_hu": -24.0,
        "fascia_texture_hu": 6.0,
        "fat_texture_hu": 7.0,
        "cortex_hu": 920.0,
        "cortex_texture_hu": 38.0,
        "marrow_hu": 135.0,
        "marrow_texture_hu": 20.0,
        "viable_hu": 44.0,
        "viable_texture_hu": 22.0,
        "necrotic_hu": 8.0,
        "necrotic_texture_hu": 6.0,
        "satellite_hu": 46.0,
        "satellite_texture_hu": 18.0,
        "scanner_noise_hu": 3.0,
    },
    "high_contrast": {
        "display_label": "higher contrast",
        "fat_hu": -105.0,
        "muscle_hu": (49.0, 56.0, 44.0, 53.0),
        "muscle_texture_hu": 12.0,
        "fascia_hu": -31.0,
        "fascia_texture_hu": 7.0,
        "fat_texture_hu": 8.0,
        "cortex_hu": 980.0,
        "cortex_texture_hu": 45.0,
        "marrow_hu": 145.0,
        "marrow_texture_hu": 24.0,
        "viable_hu": 68.0,
        "viable_texture_hu": 26.0,
        "necrotic_hu": 12.0,
        "necrotic_texture_hu": 8.0,
        "satellite_hu": 70.0,
        "satellite_texture_hu": 22.0,
        "scanner_noise_hu": 4.5,
    },
}

# Exposed for the production-profile compiler and discrete branch-resolution QC.
SPICULE_SEGMENTS = (
    ((46.0, -7.0, 7.0), (69.0, 4.0, 9.0), 4.4, 1.8),
    ((31.0, -20.0, 3.0), (16.0, -45.0, 1.0), 4.2, 1.8),
    ((28.0, -3.0, -14.0), (14.0, 5.0, -38.0), 4.0, 1.8),
    ((39.0, -5.0, 23.0), (50.0, 4.0, 47.0), 4.0, 1.8),
    ((45.0, -17.0, 7.0), (64.0, -34.0, 15.0), 4.0, 1.8),
    ((25.0, -13.0, 7.0), (5.0, -20.0, 15.0), 4.0, 1.8),
    ((39.0, 0.0, 5.0), (48.0, 23.0, 9.0), 4.0, 1.8),
    ((34.0, -10.0, -20.0), (34.0, -17.0, -47.0), 4.0, 1.8),
    ((34.0, -10.0, 27.0), (30.0, -22.0, 53.0), 4.0, 1.8),
    ((24.0, -5.0, 0.0), (3.0, 8.0, -3.0), 4.0, 1.8),
    ((47.0, -1.0, -4.0), (68.0, 12.0, -15.0), 4.0, 1.8),
    ((31.0, -19.0, -8.0), (20.0, -38.0, -24.0), 4.0, 1.8),
    ((24.7, -0.5, 15.5), (8.9, 15.6, 33.3), 4.0, 1.7),
    ((43.6, -0.9, 16.0), (59.7, 14.4, 34.4), 4.0, 1.7),
    ((43.7, -16.8, -10.3), (58.3, -27.0, -33.5), 4.0, 1.7),
    ((23.7, -0.5, -4.0), (5.9, 15.8, -19.4), 4.0, 1.7),
    ((44.8, -19.6, -2.5), (63.7, -36.5, -15.7), 4.0, 1.7),
    ((23.1, -19.0, 14.3), (4.7, -34.3, 30.2), 4.0, 1.7),
)


@dataclass(frozen=True)
class SyntheticSceneConfig:
    """Physical and numerical controls for one synthetic volume."""

    edge: int = 128
    fov_mm: float = 256.0
    seed: int = 20260809
    slab_depth: int = 8
    hu_profile: str = "reference"

    def __post_init__(self) -> None:
        if self.edge < 24:
            raise ValueError("edge must be at least 24 for inspectable anatomy")
        if not np.isfinite(self.fov_mm) or self.fov_mm <= 0:
            raise ValueError("fov_mm must be finite and positive")
        if self.slab_depth < 1:
            raise ValueError("slab_depth must be positive")
        if self.hu_profile not in HU_PROFILES:
            raise ValueError(
                f"hu_profile must be one of {tuple(HU_PROFILES)}, "
                f"received {self.hu_profile!r}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.edge, self.edge, self.edge)

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        spacing = float(self.fov_mm) / float(self.edge)
        return (spacing, spacing, spacing)

def _affine(config: SyntheticSceneConfig) -> np.ndarray:
    spacing = config.spacing_mm[0]
    first_center = -0.5 * config.fov_mm + 0.5 * spacing
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing
    affine[1, 1] = spacing
    affine[2, 2] = spacing
    affine[:3, 3] = first_center
    return affine


def _axis_centres(config: SyntheticSceneConfig) -> np.ndarray:
    spacing = config.spacing_mm[0]
    return (
        -0.5 * config.fov_mm
        + (np.arange(config.edge, dtype=np.float32) + 0.5) * spacing
    )


def _phase(seed: int, name: str, term: int) -> float:
    digest = hashlib.sha256(f"{seed}:{name}:{term}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return float(2.0 * np.pi * fraction)


def _unit_interval(seed: int, name: str, term: int, field: str) -> float:
    digest = hashlib.sha256(f"{seed}:{name}:{term}:{field}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _texture(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    seed: int,
    name: str,
    correlation_mm: float,
) -> np.ndarray:
    """Deterministic isotropic random-wave field, invariant to slab traversal.

    Directions, wavelengths, phases, and weights are all derived independently
    from SHA-256.  Sixteen modes are enough to suppress the conspicuous diagonal
    banding produced by the original four hand-selected directions while keeping
    a 192-cube deterministic render practical.
    """

    mode_count = 16
    out = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=np.float32)
    weights: list[float] = []
    modes: list[tuple[float, float, float, float, float]] = []
    for term in range(mode_count):
        # Uniform on S2: azimuth is uniform and cos(polar angle) is uniform.
        azimuth = 2.0 * np.pi * _unit_interval(seed, name, term, "azimuth")
        dz = 2.0 * _unit_interval(seed, name, term, "polar") - 1.0
        radial = np.sqrt(max(0.0, 1.0 - dz * dz))
        dx = radial * np.cos(azimuth)
        dy = radial * np.sin(azimuth)
        # Log-uniform wavelengths prevent a single visible carrier frequency.
        scale = float(
            np.exp(
                np.log(0.62)
                + _unit_interval(seed, name, term, "scale")
                * (np.log(1.75) - np.log(0.62))
            )
        )
        weight = 0.65 + 0.70 * _unit_interval(seed, name, term, "weight")
        weights.append(weight)
        modes.append((dx, dy, dz, scale, _phase(seed, name, term)))
    normalizer = float(np.sqrt(0.5 * sum(weight * weight for weight in weights)))
    for weight, (dx, dy, dz, scale, phase) in zip(weights, modes):
        k = np.float32(2.0 * np.pi * scale / correlation_mm)
        argument = k * (dx * x + dy * y + dz * z) + phase
        out += np.float32(weight) * np.sin(argument, dtype=np.float32)
    out /= np.float32(normalizer)
    return out


def _ellipsoid_score(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    q = (
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
    )
    return np.asarray(1.0 - q, dtype=np.float32)


def _capsule(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    vx, vy, vz = (end[i] - start[i] for i in range(3))
    denominator = vx * vx + vy * vy + vz * vz
    t = (x - start[0]) * vx + (y - start[1]) * vy + (z - start[2]) * vz
    t = np.clip(t / denominator, 0.0, 1.0)
    distance2 = (
        (x - (start[0] + t * vx)) ** 2
        + (y - (start[1] + t * vy)) ** 2
        + (z - (start[2] + t * vz)) ** 2
    )
    return np.asarray(distance2 <= radius_mm**2)


def _tapered_spicule(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    base_radius_mm: float,
    tip_radius_mm: float,
) -> np.ndarray:
    """Return a bounded 3D cone-like projection joined to the lesion surface."""

    vx, vy, vz = (end[index] - start[index] for index in range(3))
    denominator = vx * vx + vy * vy + vz * vz
    raw_t = (
        (x - start[0]) * vx + (y - start[1]) * vy + (z - start[2]) * vz
    ) / denominator
    t = np.clip(raw_t, 0.0, 1.0)
    distance2 = (
        (x - (start[0] + t * vx)) ** 2
        + (y - (start[1] + t * vy)) ** 2
        + (z - (start[2] + t * vz)) ** 2
    )
    radius = base_radius_mm + t * (tip_radius_mm - base_radius_mm)
    return np.asarray((raw_t >= 0.0) & (raw_t <= 1.0) & (distance2 <= radius**2))


def _tumour_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return primary score, necrotic core, satellites, and spicules."""

    components = (
        ((34.0, -10.0, 5.0), (22.0, 16.0, 28.0)),
        ((43.0, -7.0, 11.0), (14.0, 11.0, 18.0)),
        ((27.0, -4.0, -8.0), (13.0, 10.0, 17.0)),
        ((36.0, -18.0, -3.0), (15.0, 9.0, 15.0)),
    )
    primary_score = _ellipsoid_score(x, y, z, *components[0])
    for center, radii in components[1:]:
        primary_score = np.maximum(
            primary_score,
            _ellipsoid_score(x, y, z, center, radii),
        )

    core_a = _ellipsoid_score(x, y, z, (36.0, -9.0, 7.0), (9.5, 7.0, 12.0)) >= 0
    core_b = _ellipsoid_score(x, y, z, (31.0, -8.0, 0.0), (6.5, 5.0, 8.0)) >= 0
    necrotic = (core_a | core_b) & (primary_score >= 0)

    satellite_components = (
        ((69.0, 2.0, 20.0), (8.0, 6.5, 9.0)),
        ((57.0, -38.0, -20.0), (7.0, 6.0, 8.0)),
        ((13.0, -39.0, 23.0), (7.0, 6.0, 8.0)),
        ((65.0, -25.0, -5.0), (6.5, 5.5, 7.0)),
        ((36.0, 24.0, -28.0), (7.0, 5.5, 7.5)),
        ((4.0, -35.0, -25.0), (6.5, 5.5, 7.0)),
    )
    satellites = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=bool)
    for center, radii in satellite_components:
        satellites |= _ellipsoid_score(x, y, z, center, radii) >= 0
    # Preserve strict mask and label semantics even if future lobe parameters
    # move: a satellite is, by definition, outside the primary component.
    satellites &= primary_score < 0
    spicules = np.zeros_like(satellites)
    for start, end, base_radius, tip_radius in SPICULE_SEGMENTS:
        spicules |= _tapered_spicule(
            x,
            y,
            z,
            start,
            end,
            base_radius,
            tip_radius,
        )
    return primary_score, necrotic, satellites, spicules


def _soft_tissue_support(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Return the analytic body support excluding the femoral cortex."""

    z_fraction = z / np.float32(128.0)
    center_x = 2.5 + 2.0 * np.sin(z / 67.0)
    center_y = 1.0 - 1.5 * np.cos(z / 74.0)
    taper = 1.0 - 0.11 * z_fraction**2
    rx = 82.0 * taper
    ry = 70.0 * taper
    dx = x - center_x
    dy = y - center_y
    theta = np.arctan2(dy / ry, dx / rx)
    outer_radius = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    outer_limit = (
        1.0
        + 0.035 * np.sin(theta + 0.45)
        + 0.024 * np.sin(3.0 * theta + z / 45.0)
        + 0.010 * np.cos(5.0 * theta - z / 61.0)
    )
    body = outer_radius <= outer_limit
    femur_x = -19.0 + 1.4 * np.sin(z / 58.0)
    femur_y = 8.0 + 1.0 * np.cos(z / 61.0)
    femur_r2 = (x - femur_x) ** 2 + (y - femur_y) ** 2
    return np.asarray(body & (femur_r2 > 14.5**2))



def _anatomy_slab(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    seed: int,
    hu_profile: Mapping[str, Any],
    body_radii_mm: tuple[float, float] = (82.0, 70.0),
    body_taper_fraction: float = 0.11,
    body_z_half_extent_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render asymmetric thigh anatomy before lesion overlays.

    The optional geometry arguments let the volumetric benchmark use the same
    tissue renderer with a denser, hashable anatomical support.
    ``body_z_half_extent_mm`` is a physical support guard, not a display crop.
    """

    body_rx_mm, body_ry_mm = (float(value) for value in body_radii_mm)
    if body_rx_mm <= 0 or body_ry_mm <= 0:
        raise ValueError("body radii must be positive")
    if not 0 <= body_taper_fraction < 1:
        raise ValueError("body_taper_fraction must be in [0, 1)")
    if body_z_half_extent_mm is not None and body_z_half_extent_mm <= 0:
        raise ValueError("body_z_half_extent_mm must be positive when supplied")

    z_fraction = z / np.float32(128.0)
    center_x = 2.5 + 2.0 * np.sin(z / 67.0)
    center_y = 1.0 - 1.5 * np.cos(z / 74.0)
    taper = 1.0 - float(body_taper_fraction) * z_fraction**2
    rx = body_rx_mm * taper
    ry = body_ry_mm * taper
    dx = x - center_x
    dy = y - center_y
    theta = np.arctan2(dy / ry, dx / rx)
    outer_radius = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    outer_limit = (
        1.0
        + 0.035 * np.sin(theta + 0.45)
        + 0.024 * np.sin(3.0 * theta + z / 45.0)
        + 0.010 * np.cos(5.0 * theta - z / 61.0)
    )
    body = outer_radius <= outer_limit
    if body_z_half_extent_mm is not None:
        body &= np.abs(z) < float(body_z_half_extent_mm)

    muscle_dx = x - (center_x + 1.5)
    muscle_dy = y - (center_y - 1.5)
    muscle_rx = np.maximum(rx - 13.0, 20.0)
    muscle_ry = np.maximum(ry - 12.0, 20.0)
    muscle_radius = np.sqrt((muscle_dx / muscle_rx) ** 2 + (muscle_dy / muscle_ry) ** 2)
    muscle = body & (muscle_radius <= 1.0 + 0.018 * np.cos(4.0 * theta - z / 53.0))

    image = np.full(np.broadcast_shapes(x.shape, y.shape, z.shape), -1000.0, np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    image[body] = float(hu_profile["fat_hu"])
    labels[body] = 1

    # Four unequal grouped-muscle regions.  The anterior group is the containing
    # muscle bed; three bounded, off-centre elliptical groups are embedded in it.
    # This avoids the visually artificial four-way Voronoi/pinwheel junction.
    scale_x = body_rx_mm / 82.0
    scale_y = body_ry_mm / 70.0
    anterior = (
        (muscle_dx - 5.0 * scale_x - scale_x * np.sin(z / 57.0)) / (39.0 * scale_x)
    ) ** 2 + ((muscle_dy + 31.0 * scale_y) / (24.0 * scale_y)) ** 2
    medial = (
        (muscle_dx - 35.0 * scale_x - 3.0 * scale_x * np.sin(z / 29.0))
        / (23.0 * scale_x)
    ) ** 2 + (
        (muscle_dy - 5.0 * scale_y - 1.2 * scale_y * np.cos(z / 51.0))
        / (37.0 * scale_y)
    ) ** 2
    posterior = (
        (muscle_dx + 1.0 * scale_x - 3.0 * scale_x * np.sin(z / 33.0))
        / (42.0 * scale_x)
    ) ** 2 + ((muscle_dy - 32.0 * scale_y) / (25.0 * scale_y)) ** 2
    lateral = (
        (muscle_dx + 35.0 * scale_x - 3.0 * scale_x * np.cos(z / 31.0))
        / (26.0 * scale_x)
    ) ** 2 + (
        (muscle_dy + 2.0 * scale_y - scale_y * np.cos(z / 69.0)) / (36.0 * scale_y)
    ) ** 2
    # An anterior oval establishes a non-radial upper group.  Outside it, label
    # 2 remains a modest connective/background muscle bed between groups.
    compartment = np.full(np.broadcast_shapes(x.shape, y.shape, z.shape), 2, np.uint8)
    anterior_group = anterior <= 1.20
    medial_group = medial <= 1.05
    posterior_group = posterior <= 1.08
    lateral_group = lateral <= 1.05
    # Resolve the small peripheral overlaps locally by the lowest elliptical
    # score; no global power diagram or shared four-way junction is created.
    bounded_scores = np.stack(
        np.broadcast_arrays(medial, posterior, lateral), axis=0
    ).astype(np.float32, copy=False)
    bounded_active = np.stack(
        np.broadcast_arrays(medial_group, posterior_group, lateral_group), axis=0
    )
    bounded_scores = np.where(bounded_active, bounded_scores, np.inf)
    bounded_choice = np.argmin(bounded_scores, axis=0)
    any_bounded = np.any(bounded_active, axis=0)
    bounded_codes = np.asarray([3, 4, 5], dtype=np.uint8)
    compartment[any_bounded] = bounded_codes[bounded_choice[any_bounded]]
    compartment[anterior_group & ~any_bounded] = 2

    # Short, curved fascial arcs follow selected outer portions of the grouped
    # ellipses.  They are spatially windowed in x/y/z and therefore cannot form
    # full-length dark planes in either longitudinal view.
    medial_arc = (
        (np.abs(np.sqrt(medial) - 1.0) < 0.055)
        & (muscle_dx > 12.0 * scale_x)
        & (muscle_dy > -24.0 * scale_y)
        & (muscle_dy < 31.0 * scale_y)
        & (np.abs(z + 12.0) < 32.0)
    )
    posterior_arc = (
        (np.abs(np.sqrt(posterior) - 1.0) < 0.055)
        & (muscle_dy > 13.0 * scale_y)
        & (muscle_dx > -24.0 * scale_x)
        & (muscle_dx < 28.0 * scale_x)
        & (np.abs(z - 18.0) < 30.0)
    )
    lateral_arc = (
        (np.abs(np.sqrt(lateral) - 1.0) < 0.055)
        & (muscle_dx < -13.0 * scale_x)
        & (muscle_dy > -27.0 * scale_y)
        & (muscle_dy < 27.0 * scale_y)
        & (np.abs(z + 20.0) < 27.0)
    )
    fascia = muscle & (medial_arc | posterior_arc | lateral_arc)
    muscle_texture = _texture(x, y, z, seed=seed, name="muscle", correlation_mm=13.0)
    muscle_hu = tuple(float(value) for value in hu_profile["muscle_hu"])
    compartment_hu = np.choose(
        compartment,
        [-1000.0, float(hu_profile["fat_hu"]), *muscle_hu],
    ).astype(np.float32)
    image[muscle] = (
        compartment_hu + float(hu_profile["muscle_texture_hu"]) * muscle_texture
    )[muscle]
    labels[muscle] = compartment[muscle]

    fascia_texture = _texture(x, y, z, seed=seed, name="fascia", correlation_mm=11.0)
    image[fascia] = (
        float(hu_profile["fascia_hu"])
        + float(hu_profile["fascia_texture_hu"]) * fascia_texture
    )[fascia]
    labels[fascia] = 11

    fat_texture = _texture(x, y, z, seed=seed, name="fat", correlation_mm=18.0)
    fat = body & ~muscle
    image[fat] += (float(hu_profile["fat_texture_hu"]) * fat_texture)[fat]

    # Eccentric femur with marrow and a dense cortical annulus.
    femur_x = -19.0 + 1.4 * np.sin(z / 58.0)
    femur_y = 8.0 + 1.0 * np.cos(z / 61.0)
    femur_r2 = (x - femur_x) ** 2 + (y - femur_y) ** 2
    cortex = body & (femur_r2 <= 14.5**2)
    marrow = body & (femur_r2 <= 8.5**2)
    bone_texture = _texture(x, y, z, seed=seed, name="bone", correlation_mm=8.0)
    image[cortex] = (
        float(hu_profile["cortex_hu"])
        + float(hu_profile["cortex_texture_hu"]) * bone_texture
    )[cortex]
    labels[cortex] = 6
    image[marrow] = (
        float(hu_profile["marrow_hu"])
        + float(hu_profile["marrow_texture_hu"]) * bone_texture
    )[marrow]
    labels[marrow] = 7

    return image, labels


def _slab_indices(
    config: SyntheticSceneConfig, order: Literal["forward", "reverse"]
) -> list[tuple[int, int]]:
    slabs = [
        (start, min(config.edge, start + config.slab_depth))
        for start in range(0, config.edge, config.slab_depth)
    ]
    if order == "reverse":
        slabs.reverse()
    elif order != "forward":
        raise ValueError("slab_order must be 'forward' or 'reverse'")
    return slabs


def _retain_component_overlapping_primary(
    candidate: np.ndarray,
    primary: np.ndarray,
) -> np.ndarray:
    """Discard discretely detached M3 fragments after voxelisation.

    The analytic capsules begin inside the expanded primary, but a thin oblique
    capsule can alias into detached voxels on a coarse grid.  Keeping the single
    26-connected component that overlaps M1 makes the topology contractual, not
    an accidental property of a particular spacing.  A uint8 work array bounds
    the additional memory to one byte per output voxel.
    """

    from scipy import ndimage

    component_labels = np.empty(candidate.shape, dtype=np.uint8)
    count = ndimage.label(
        candidate != 0,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
        output=component_labels,
    )
    if count <= 1:
        return np.asarray(candidate, dtype=np.uint8)
    overlapping = component_labels[primary != 0]
    overlapping = overlapping[overlapping != 0]
    if overlapping.size == 0:
        raise RuntimeError("M3 has no component overlapping the primary tumour")
    identifiers, counts = np.unique(overlapping, return_counts=True)
    keep = int(identifiers[int(np.argmax(counts))])
    return np.asarray(component_labels == keep, dtype=np.uint8)

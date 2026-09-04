"""Frozen calculation workloads for the performance benchmark.

Radiomic packages expose different native calculation boundaries.  Scheduling
one process per small IBSI family repeatedly charges packages that deliberately
share matrix construction across families.  The performance benchmark therefore
schedules the largest groups that consume one harmonised stored representation.

Two algorithms are deliberately kept out of broader groups even though they use
the same continuous image: local-intensity peaks require a physical spherical
neighbourhood, while Moran's I and Geary's C require pairwise spatial
autocorrelation.  Their scaling and package support differ materially from
ordinary first-order and morphology calculations.  IVH remains separate because
its frozen input is not the continuous intensity image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkWorkload:
    name: str
    families: tuple[str, ...]
    representation_family: str
    feature_partition: str = "complete"

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError(f"workload {self.name!r} has no feature families")
        if self.representation_family not in self.families:
            raise ValueError(
                f"workload {self.name!r} representation family is not selected"
            )


WORKLOADS: tuple[BenchmarkWorkload, ...] = (
    BenchmarkWorkload(
        name="morphology",
        families=("morphology",),
        representation_family="morphology",
        feature_partition="exclude_spatial_autocorrelation",
    ),
    BenchmarkWorkload(
        name="spatial_autocorrelation",
        families=("morphology",),
        representation_family="morphology",
        feature_partition="spatial_autocorrelation_only",
    ),
    BenchmarkWorkload(
        name="local_intensity",
        families=("local_intensity",),
        representation_family="local_intensity",
    ),
    BenchmarkWorkload(
        name="intensity",
        families=("intensity",),
        representation_family="intensity",
    ),
    BenchmarkWorkload(
        name="texture",
        families=(
            "histogram",
            "glcm",
            "glrlm",
            "glszm",
            "gldzm",
            "ngtdm",
            "ngldm",
        ),
        representation_family="glcm",
    ),
    BenchmarkWorkload(
        name="ivh",
        families=("ivh",),
        representation_family="ivh",
    ),
)

WORKLOAD_ORDER = tuple(workload.name for workload in WORKLOADS)
WORKLOAD_BY_NAME = {workload.name: workload for workload in WORKLOADS}


def parse_workloads(value: str | None) -> list[BenchmarkWorkload]:
    token = str(value or "").strip().lower()
    if not token:
        raise ValueError("at least one benchmark workload is required")
    if token == "all":
        return list(WORKLOADS)
    names = list(
        dict.fromkeys(part.strip().lower() for part in token.split(",") if part.strip())
    )
    unknown = sorted(set(names).difference(WORKLOAD_BY_NAME))
    if unknown:
        raise ValueError(f"unknown benchmark workloads: {', '.join(unknown)}")
    return [WORKLOAD_BY_NAME[name] for name in WORKLOAD_ORDER if name in names]


def families_for_workloads(workloads: Iterable[BenchmarkWorkload]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(family for workload in workloads for family in workload.families)
    )


__all__ = [
    "BenchmarkWorkload",
    "WORKLOADS",
    "WORKLOAD_BY_NAME",
    "WORKLOAD_ORDER",
    "families_for_workloads",
    "parse_workloads",
]

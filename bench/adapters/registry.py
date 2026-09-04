"""Static adapter capabilities used by planning, validation, and reporting.

The registry describes what the harness requests from each upstream package.  It
does not claim that an implementation is IBSI-conformant; conformance is a
separate, reference-data result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bench.ibsi_families import FAMILY_ORDER


SelectionMode = Literal["native", "feature_selection", "post_filter"]


@dataclass(frozen=True)
class AdapterCapabilities:
    name: str
    distribution: str
    families: tuple[str, ...]
    selection_mode: SelectionMode
    notes: str = ""
    unsupported_workloads: frozenset[str] = frozenset()

    def supports(self, family: str) -> bool:
        return family in self.families

    def supports_workload(self, workload: str) -> bool:
        return workload not in self.unsupported_workloads


_ALL_IBSI_FAMILIES = tuple(FAMILY_ORDER)

ADAPTERS: dict[str, AdapterCapabilities] = {
    "pictologics": AdapterCapabilities(
        name="pictologics",
        distribution="pictologics",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
    ),
    "pyradiomics": AdapterCapabilities(
        name="pyradiomics",
        distribution="pyradiomics",
        families=(
            "morphology",
            "intensity",
            "histogram",
            "glcm",
            "glrlm",
            "glszm",
            "ngtdm",
            "ngldm",
        ),
        selection_mode="feature_selection",
        unsupported_workloads=frozenset(
            {"spatial_autocorrelation", "local_intensity", "ivh"}
        ),
        notes="IBSI families are selected through explicit PyRadiomics feature names.",
    ),
    "mirp": AdapterCapabilities(
        name="mirp",
        distribution="mirp",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
    ),
    "medimage": AdapterCapabilities(
        name="medimage",
        distribution="medimage-pkg",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
        notes="The adapter uses MEDimage's family modules because no stable unified API is exposed.",
    ),
    "zrad": AdapterCapabilities(
        name="zrad",
        distribution="z-rad",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
        notes=(
            "Z-Rad 26.8 selects native feature families; Moran's I and Geary's C "
            "use its separate opt-in morphology_correlation group."
        ),
    ),
}


def get_adapter(name: str) -> AdapterCapabilities:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown adapter {name!r}; choose one of: {choices}") from exc


def validate_families(name: str, families: list[str]) -> tuple[list[str], list[str]]:
    """Return (supported, unsupported) families in canonical order."""

    capabilities = get_adapter(name)
    requested = set(families)
    supported = [
        family
        for family in FAMILY_ORDER
        if family in requested and capabilities.supports(family)
    ]
    unsupported = [
        family
        for family in FAMILY_ORDER
        if family in requested and not capabilities.supports(family)
    ]
    unknown = sorted(requested.difference(FAMILY_ORDER))
    if unknown:
        raise ValueError(f"Unknown IBSI feature families: {', '.join(unknown)}")
    return supported, unsupported

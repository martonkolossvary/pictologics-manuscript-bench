"""Versioned identifiers for IBSI features with known naming conflicts.

``ibsi_codes`` provides the adapter crosswalk, while this registry records the
cases where a bare identifier is not sufficient to identify a feature
reproducibly. Consumers should use the semantic key as their durable join key
and retain source/version metadata for every accepted identifier alias.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


IBSI_IDENTIFIER_SCHEMA_VERSION = "1.0.0"
IBSI_IDENTIFIER_REGISTRY_VERSION = "2026-07-20"

DICOM_HISTOGRAM_SOURCE = "DICOM PS3.16 CID 7478"
DICOM_GLSZM_SOURCE = "DICOM PS3.16 CID 7476"
DICOM_GLCM_SOURCE = "DICOM PS3.16 CID 7467"
IBSI_FEATURES_SOURCE = "IBSI image feature specification"
MIRP_SOURCE = "MIRP feature catalogue"
BENCHMARK_CROSSWALK_SOURCE = "Pictologics benchmark source crosswalk"


@dataclass(frozen=True)
class IdentifierReference:
    """An identifier asserted by a named, versioned source."""

    value: str
    source: str
    version: str
    url: str
    note: str = ""


@dataclass(frozen=True)
class IdentifierAlias:
    """A source-scoped alias, optionally known to conflict with another key."""

    value: str
    source: str
    version: str
    note: str
    conflicts_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureIdentifier:
    """Canonical semantic identity plus source-specific external identifiers."""

    semantic_key: str
    family: str
    display_name: str
    general_id: IdentifierReference
    specific_id: Optional[IdentifierReference] = None
    parameters: tuple[tuple[str, str], ...] = ()
    aliases: tuple[IdentifierAlias, ...] = ()

    @property
    def preferred_id(self) -> str:
        """Return the concrete identifier when available, otherwise the general ID."""

        return self.specific_id.value if self.specific_id else self.general_id.value

    @property
    def parameter_map(self) -> Mapping[str, str]:
        """Expose parameters as an immutable mapping."""

        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True)
class IdentifierMatch:
    semantic_key: str
    role: str
    source: str
    version: str
    conflicts_with: tuple[str, ...] = ()


class IdentifierConflictError(ValueError):
    """Raised when an unscoped identifier names more than one semantic feature."""

    def __init__(self, identifier: str, matches: tuple[IdentifierMatch, ...]):
        keys = sorted({match.semantic_key for match in matches})
        super().__init__(
            f"Identifier {identifier!r} is ambiguous across semantic keys: "
            + ", ".join(keys)
        )
        self.identifier = identifier
        self.matches = matches


class IdentifierParameterError(ValueError):
    """Raised when a parameterised general ID lacks feature parameters."""

    def __init__(self, identifier: str, matches: tuple[IdentifierMatch, ...]):
        super().__init__(
            f"General identifier {identifier!r} requires parameters or a "
            "parameter-specific identifier"
        )
        self.identifier = identifier
        self.matches = matches


_DICOM_HISTOGRAM_URL = (
    "https://dicom.nema.org/medical/Dicom/2024c/output/chtml/part16/sect_CID_7478.html"
)
_DICOM_GLSZM_URL = (
    "https://dicom.nema.org/medical/dicom/current/output/chtml/part16/"
    "sect_cid_7476.html"
)
_DICOM_GLCM_URL = (
    "https://dicom.nema.org/medical/dicom/current/output/chtml/part16/"
    "sect_cid_7467.html"
)
_IBSI_FEATURES_URL = "https://ibsi.readthedocs.io/en/latest/03_Image_features.html"
_MIRP_FEATURES_URL = "https://oncoray.github.io/mirp/features_names.html"


FEATURE_IDENTIFIERS = (
    FeatureIdentifier(
        semantic_key="intensity_histogram.percentile_10",
        family="histogram",
        display_name="10th discretised intensity percentile",
        general_id=IdentifierReference(
            "GPMT",
            DICOM_HISTOGRAM_SOURCE,
            "CID version 20190121",
            _DICOM_HISTOGRAM_URL,
        ),
        aliases=(
            IdentifierAlias(
                "1PR",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "Source-crosswalk alias; current DICOM and MIRP use GPMT.",
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="intensity_histogram.percentile_90",
        family="histogram",
        display_name="90th discretised intensity percentile",
        general_id=IdentifierReference(
            "OZ0C",
            DICOM_HISTOGRAM_SOURCE,
            "CID version 20190121",
            _DICOM_HISTOGRAM_URL,
        ),
        aliases=(
            IdentifierAlias(
                "GPMT",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "The source crosswalk assigned GPMT to the 90th percentile, but GPMT "
                "is the current identifier for the 10th percentile.",
                conflicts_with=("intensity_histogram.percentile_10",),
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="glszm.small_zone_emphasis",
        family="glszm",
        display_name="Small zone emphasis",
        general_id=IdentifierReference(
            "5QRC",
            DICOM_GLSZM_SOURCE,
            "CID version 20190121",
            _DICOM_GLSZM_URL,
        ),
        aliases=(
            IdentifierAlias(
                "P001",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "Placeholder found in the source crosswalk; current DICOM "
                "and MIRP use 5QRC.",
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="ivh.volume_at_intensity_fraction_10",
        family="ivh",
        display_name="Volume at intensity fraction 0.10",
        general_id=IdentifierReference(
            "BC2M",
            IBSI_FEATURES_SOURCE,
            "0.0.1dev; accessed 2026-07-20",
            _IBSI_FEATURES_URL,
            "General parameterised feature identifier.",
        ),
        specific_id=IdentifierReference(
            "NK6P", MIRP_SOURCE, "2.5.0", _MIRP_FEATURES_URL
        ),
        parameters=(("intensity_fraction", "0.10"),),
        aliases=(
            IdentifierAlias(
                "BC2M_10",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "Non-standard composite token retained as a source alias.",
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="ivh.volume_at_intensity_fraction_90",
        family="ivh",
        display_name="Volume at intensity fraction 0.90",
        general_id=IdentifierReference(
            "BC2M",
            IBSI_FEATURES_SOURCE,
            "0.0.1dev; accessed 2026-07-20",
            _IBSI_FEATURES_URL,
            "General parameterised feature identifier.",
        ),
        specific_id=IdentifierReference(
            "4279", MIRP_SOURCE, "2.5.0", _MIRP_FEATURES_URL
        ),
        parameters=(("intensity_fraction", "0.90"),),
        aliases=(
            IdentifierAlias(
                "BC2M_90",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "Non-standard composite token retained as a source alias.",
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="ivh.intensity_at_volume_fraction_10",
        family="ivh",
        display_name="Intensity at volume fraction 0.10",
        general_id=IdentifierReference(
            "GBPN",
            IBSI_FEATURES_SOURCE,
            "0.0.1dev; accessed 2026-07-20",
            _IBSI_FEATURES_URL,
            "General parameterised feature identifier.",
        ),
        specific_id=IdentifierReference(
            "PWN1", MIRP_SOURCE, "2.5.0", _MIRP_FEATURES_URL
        ),
        parameters=(("volume_fraction", "0.10"),),
        aliases=(
            IdentifierAlias(
                "GBPN_10",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "Non-standard composite token retained as a source alias.",
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="ivh.intensity_at_volume_fraction_90",
        family="ivh",
        display_name="Intensity at volume fraction 0.90",
        general_id=IdentifierReference(
            "GBPN",
            IBSI_FEATURES_SOURCE,
            "0.0.1dev; accessed 2026-07-20",
            _IBSI_FEATURES_URL,
            "General parameterised feature identifier.",
        ),
        specific_id=IdentifierReference(
            "BOHI", MIRP_SOURCE, "2.5.0", _MIRP_FEATURES_URL
        ),
        parameters=(("volume_fraction", "0.90"),),
        aliases=(
            IdentifierAlias(
                "GBPN_90",
                BENCHMARK_CROSSWALK_SOURCE,
                "2026-05-26",
                "Non-standard composite token retained as a source alias.",
            ),
        ),
    ),
    FeatureIdentifier(
        semantic_key="ivh.volume_fraction_difference_10_90",
        family="ivh",
        display_name="Volume fraction difference between intensity fractions 0.10 and 0.90",
        general_id=IdentifierReference(
            "DDTU",
            IBSI_FEATURES_SOURCE,
            "0.0.1dev; accessed 2026-07-20",
            _IBSI_FEATURES_URL,
            "The source crosswalk used this general ID as a concrete 10/90 ID.",
        ),
        specific_id=IdentifierReference(
            "WITY", MIRP_SOURCE, "2.5.0", _MIRP_FEATURES_URL
        ),
        parameters=(
            ("intensity_fraction_low", "0.10"),
            ("intensity_fraction_high", "0.90"),
        ),
    ),
    FeatureIdentifier(
        semantic_key="ivh.intensity_fraction_difference_10_90",
        family="ivh",
        display_name="Intensity fraction difference between volume fractions 0.10 and 0.90",
        general_id=IdentifierReference(
            "CNV2",
            IBSI_FEATURES_SOURCE,
            "0.0.1dev; accessed 2026-07-20",
            _IBSI_FEATURES_URL,
            "The source crosswalk used this general ID as a concrete 10/90 ID.",
        ),
        specific_id=IdentifierReference(
            "JXJA", MIRP_SOURCE, "2.5.0", _MIRP_FEATURES_URL
        ),
        parameters=(
            ("volume_fraction_low", "0.10"),
            ("volume_fraction_high", "0.90"),
        ),
    ),
    FeatureIdentifier(
        semantic_key="glcm.sum_entropy",
        family="glcm",
        display_name="Sum entropy",
        general_id=IdentifierReference(
            "P6QZ",
            DICOM_GLCM_SOURCE,
            "CID version 20190121",
            _DICOM_GLCM_URL,
        ),
        aliases=(
            IdentifierAlias(
                "P6QZ1",
                MIRP_SOURCE,
                "2.5.0",
                "MIRP 2.5.0 native metadata identifier; canonical IBSI/DICOM ID is P6QZ.",
            ),
        ),
    ),
)


FEATURES_BY_SEMANTIC_KEY = MappingProxyType(
    {feature.semantic_key: feature for feature in FEATURE_IDENTIFIERS}
)


def get_feature(semantic_key: str) -> FeatureIdentifier:
    """Return a registry record by its durable semantic key."""

    return FEATURES_BY_SEMANTIC_KEY[semantic_key]


def find_identifier(
    identifier: str, *, source: Optional[str] = None
) -> tuple[IdentifierMatch, ...]:
    """Find every semantic feature asserted for an identifier.

    A source filter is required when a reused identifier such as ``GPMT`` must
    be interpreted according to a particular cited crosswalk.
    """

    value = identifier.strip().casefold()
    source_filter = source.strip().casefold() if source else None
    matches: list[IdentifierMatch] = []

    for feature in FEATURE_IDENTIFIERS:
        references = (("general", feature.general_id),)
        if feature.specific_id:
            references += (("specific", feature.specific_id),)
        for role, reference in references:
            if reference.value.casefold() != value:
                continue
            if source_filter and reference.source.casefold() != source_filter:
                continue
            matches.append(
                IdentifierMatch(
                    feature.semantic_key, role, reference.source, reference.version
                )
            )

        for alias in feature.aliases:
            if alias.value.casefold() != value:
                continue
            if source_filter and alias.source.casefold() != source_filter:
                continue
            matches.append(
                IdentifierMatch(
                    feature.semantic_key,
                    "alias",
                    alias.source,
                    alias.version,
                    alias.conflicts_with,
                )
            )

    return tuple(matches)


def resolve_identifier(
    identifier: str,
    *,
    source: Optional[str] = None,
    parameters: Optional[Mapping[str, str]] = None,
) -> FeatureIdentifier:
    """Resolve an identifier only when it selects one concrete semantic feature.

    Parameterised general IDs such as ``BC2M`` and ``DDTU`` require an exact
    parameter subset, even when this compact registry currently contains only
    one concrete instance of the general feature.
    """

    matches = find_identifier(identifier, source=source)
    if not matches:
        raise KeyError(identifier)

    if parameters:
        requested = {str(name): str(value) for name, value in parameters.items()}
        matches = tuple(
            match
            for match in matches
            if all(
                get_feature(match.semantic_key).parameter_map.get(name) == value
                for name, value in requested.items()
            )
        )
        if not matches:
            raise KeyError((identifier, tuple(sorted(requested.items()))))

    semantic_keys = {match.semantic_key for match in matches}
    if len(semantic_keys) != 1:
        if not parameters and all(match.role == "general" for match in matches):
            raise IdentifierParameterError(identifier, matches)
        raise IdentifierConflictError(identifier, matches)

    feature = get_feature(semantic_keys.pop())
    matched_roles = {match.role for match in matches}
    if feature.parameters and matched_roles == {"general"} and not parameters:
        raise IdentifierParameterError(identifier, matches)
    return feature


def validate_registry() -> None:
    """Validate structural invariants without external packages or I/O."""

    if len(FEATURES_BY_SEMANTIC_KEY) != len(FEATURE_IDENTIFIERS):
        raise ValueError("Duplicate canonical semantic key in IBSI identifier registry")

    known_keys = set(FEATURES_BY_SEMANTIC_KEY)
    specific_ids: dict[str, str] = {}
    for feature in FEATURE_IDENTIFIERS:
        if not feature.semantic_key or not feature.family:
            raise ValueError("Every identifier record needs a semantic key and family")
        parameter_names = [name for name, _ in feature.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError(f"Duplicate parameter in {feature.semantic_key}")

        for reference in (feature.general_id, feature.specific_id):
            if reference is None:
                continue
            if not all(
                (reference.value, reference.source, reference.version, reference.url)
            ):
                raise ValueError(
                    f"Incomplete source metadata in {feature.semantic_key}"
                )

        if feature.specific_id:
            token = feature.specific_id.value.casefold()
            previous = specific_ids.setdefault(token, feature.semantic_key)
            if previous != feature.semantic_key:
                raise ValueError(
                    f"Specific identifier {feature.specific_id.value} is reused by "
                    f"{previous} and {feature.semantic_key}"
                )

        for alias in feature.aliases:
            if not all((alias.value, alias.source, alias.version, alias.note)):
                raise ValueError(f"Incomplete alias metadata in {feature.semantic_key}")
            unknown = set(alias.conflicts_with) - known_keys
            if unknown:
                raise ValueError(
                    f"Unknown conflict targets in {feature.semantic_key}: {sorted(unknown)}"
                )


validate_registry()


__all__ = [
    "DICOM_GLCM_SOURCE",
    "DICOM_GLSZM_SOURCE",
    "DICOM_HISTOGRAM_SOURCE",
    "FEATURE_IDENTIFIERS",
    "FEATURES_BY_SEMANTIC_KEY",
    "IBSI_FEATURES_SOURCE",
    "IBSI_IDENTIFIER_REGISTRY_VERSION",
    "IBSI_IDENTIFIER_SCHEMA_VERSION",
    "IdentifierAlias",
    "IdentifierConflictError",
    "IdentifierMatch",
    "IdentifierParameterError",
    "IdentifierReference",
    "FeatureIdentifier",
    "BENCHMARK_CROSSWALK_SOURCE",
    "MIRP_SOURCE",
    "find_identifier",
    "get_feature",
    "resolve_identifier",
    "validate_registry",
]

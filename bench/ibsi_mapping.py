from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from bench.ibsi_codes import CODE_TO_NAME
from bench.ibsi_families import CODE_TO_FAMILY

IBSI_MAPPING_SCHEMA_VERSION = "2026-07-21"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


NAME_TO_CODE = {_normalize(name): code for code, name in CODE_TO_NAME.items()}
NAME_TO_CODES: Dict[str, list[str]] = {}
for _code, _name in CODE_TO_NAME.items():
    NAME_TO_CODES.setdefault(_normalize(_name), []).append(_code)


def _code_from_name(name: str) -> Optional[str]:
    return NAME_TO_CODE.get(_normalize(name))


def _code_from_name_in_family(name: str, family: Optional[str]) -> Optional[str]:
    norm = _normalize(name)
    codes = NAME_TO_CODES.get(norm, [])
    if not codes:
        return None
    if family:
        fam_codes = [c for c in codes if CODE_TO_FAMILY.get(c) == family]
        return sorted(fam_codes)[0] if fam_codes else None
    return sorted(codes)[0]


def map_pictologics(feature_name: str) -> Optional[str]:
    # Pictologics encodes IBSI codes as suffixes, sometimes with extra token (e.g., BC2M_10)
    parts = feature_name.split("_")
    # try longest suffix first (3 -> 1)
    for n in range(3, 0, -1):
        if len(parts) >= n:
            code = "_".join(parts[-n:])
            if code in CODE_TO_NAME:
                return code
    return None


PYRADIOMICS_MAP = {
    ("shape", "MeshVolume"): "Volume",
    ("shape", "VoxelVolume"): "Volume voxel counting",
    ("shape", "SurfaceArea"): "Surface area",
    ("shape", "SurfaceVolumeRatio"): "Surface to volume ratio",
    # These three upstream methods are marked deprecated because they are
    # algebraically redundant with Sphericity, but PyRadiomics still
    # calculates and returns them when they are explicitly enabled by name.
    ("shape", "Compactness1"): "Compactness 1",
    ("shape", "Compactness2"): "Compactness 2",
    ("shape", "SphericalDisproportion"): "Spherical disproportion",
    ("shape", "Sphericity"): "Sphericity",
    ("shape", "Maximum3DDiameter"): "Maximum 3D diameter",
    ("shape", "MajorAxisLength"): "Major axis length",
    ("shape", "MinorAxisLength"): "Minor axis length",
    ("shape", "LeastAxisLength"): "Least axis length",
    ("shape", "Elongation"): "Elongation",
    ("shape", "Flatness"): "Flatness",
    ("firstorder", "Mean"): "Mean intensity",
    ("firstorder", "Variance"): "Intensity variance",
    ("firstorder", "Skewness"): "Intensity skewness",
    ("firstorder", "Kurtosis"): "Intensity kurtosis",
    ("firstorder", "Median"): "Median intensity",
    ("firstorder", "Minimum"): "Minimum intensity",
    ("firstorder", "Maximum"): "Maximum intensity",
    ("firstorder", "Range"): "Intensity range",
    ("firstorder", "InterquartileRange"): "Intensity interquartile range",
    ("firstorder", "MeanAbsoluteDeviation"): "Intensity Mean absolute deviation",
    (
        "firstorder",
        "RobustMeanAbsoluteDeviation",
    ): "Intensity Robust mean absolute deviation",
    ("firstorder", "Energy"): "Intensity energy",
    ("firstorder", "RootMeanSquared"): "Root mean square intensity",
    ("firstorder", "Entropy"): "Discretised intensity entropy",
    ("firstorder", "Uniformity"): "Discretised intensity uniformity",
    ("firstorder", "10Percentile"): "10th intensity percentile",
    ("firstorder", "90Percentile"): "90th intensity percentile",
    ("glcm", "JointAverage"): "Joint average",
    ("glcm", "JointEntropy"): "Joint entropy",
    ("glcm", "JointEnergy"): "Angular second moment",
    ("glcm", "MaximumProbability"): "Joint maximum",
    # PyRadiomics names the IBSI Joint Variance implementation ``SumSquares``.
    # Its upstream docstring explicitly identifies this feature as Joint
    # Variance; ``JointVariance`` is not a PyRadiomics feature name.
    ("glcm", "SumSquares"): "Joint variance",
    ("glcm", "SumAverage"): "Sum average",
    ("glcm", "SumEntropy"): "Sum entropy",
    ("glcm", "DifferenceAverage"): "Difference average",
    ("glcm", "DifferenceVariance"): "Difference variance",
    ("glcm", "DifferenceEntropy"): "Difference entropy",
    ("glcm", "Contrast"): "Contrast",
    ("glcm", "Id"): "Inverse difference",
    ("glcm", "Idn"): "Normalised inverse difference",
    ("glcm", "Idm"): "Inverse difference moment",
    ("glcm", "Idmn"): "Normalised inverse difference moment",
    ("glcm", "InverseVariance"): "Inverse variance",
    ("glcm", "Correlation"): "Correlation",
    ("glcm", "Autocorrelation"): "Autocorrelation",
    ("glcm", "ClusterTendency"): "Cluster tendency",
    ("glcm", "ClusterShade"): "Cluster shade",
    ("glcm", "ClusterProminence"): "Cluster prominence",
    ("glcm", "Imc1"): "Information correlation 1",
    ("glcm", "Imc2"): "Information correlation 2",
    ("glrlm", "ShortRunEmphasis"): "Short runs emphasis",
    ("glrlm", "LongRunEmphasis"): "Long runs emphasis",
    ("glrlm", "LowGrayLevelRunEmphasis"): "Low grey level run emphasis",
    ("glrlm", "HighGrayLevelRunEmphasis"): "High grey level run emphasis",
    ("glrlm", "ShortRunLowGrayLevelEmphasis"): "Short run low grey level emphasis",
    ("glrlm", "ShortRunHighGrayLevelEmphasis"): "Short run high grey level emphasis",
    ("glrlm", "LongRunLowGrayLevelEmphasis"): "Long run low grey level emphasis",
    ("glrlm", "LongRunHighGrayLevelEmphasis"): "Long run high grey level emphasis",
    ("glrlm", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "glrlm",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("glrlm", "RunLengthNonUniformity"): "Run length non-uniformity",
    (
        "glrlm",
        "RunLengthNonUniformityNormalized",
    ): "Normalised run length non-uniformity",
    ("glrlm", "RunPercentage"): "Run percentage",
    ("glrlm", "GrayLevelVariance"): "Grey level variance",
    ("glrlm", "RunVariance"): "Run length variance",
    ("glrlm", "RunEntropy"): "Run entropy",
    ("glszm", "SmallAreaEmphasis"): "Small zone emphasis",
    ("glszm", "LargeAreaEmphasis"): "Large zone emphasis",
    ("glszm", "LowGrayLevelZoneEmphasis"): "Low grey level zone emphasis",
    ("glszm", "HighGrayLevelZoneEmphasis"): "High grey level zone emphasis",
    ("glszm", "SmallAreaLowGrayLevelEmphasis"): "Small zone low grey level emphasis",
    ("glszm", "SmallAreaHighGrayLevelEmphasis"): "Small zone high grey level emphasis",
    ("glszm", "LargeAreaLowGrayLevelEmphasis"): "Large zone low grey level emphasis",
    ("glszm", "LargeAreaHighGrayLevelEmphasis"): "Large zone high grey level emphasis",
    ("glszm", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "glszm",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("glszm", "SizeZoneNonUniformity"): "Zone size non-uniformity",
    ("glszm", "SizeZoneNonUniformityNormalized"): "Normalised zone size non-uniformity",
    ("glszm", "ZonePercentage"): "Zone percentage",
    ("glszm", "GrayLevelVariance"): "Grey level variance",
    ("glszm", "ZoneVariance"): "Zone size variance",
    ("glszm", "ZoneEntropy"): "Zone size entropy",
    ("gldm", "SmallDependenceEmphasis"): "Low dependence emphasis",
    ("gldm", "LargeDependenceEmphasis"): "High dependence emphasis",
    ("gldm", "LowGrayLevelEmphasis"): "Low grey level count emphasis",
    ("gldm", "HighGrayLevelEmphasis"): "High grey level count emphasis",
    (
        "gldm",
        "SmallDependenceLowGrayLevelEmphasis",
    ): "Low dependence low grey level emphasis",
    (
        "gldm",
        "SmallDependenceHighGrayLevelEmphasis",
    ): "Low dependence high grey level emphasis",
    (
        "gldm",
        "LargeDependenceLowGrayLevelEmphasis",
    ): "High dependence low grey level emphasis",
    (
        "gldm",
        "LargeDependenceHighGrayLevelEmphasis",
    ): "High dependence high grey level emphasis",
    ("gldm", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    ("gldm", "DependenceNonUniformity"): "Dependence count non-uniformity",
    (
        "gldm",
        "DependenceNonUniformityNormalized",
    ): "Normalised dependence count non-uniformity",
    ("gldm", "DependenceVariance"): "Dependence count variance",
    ("gldm", "DependenceEntropy"): "Dependence count entropy",
    ("gldm", "GrayLevelVariance"): "Grey level variance",
    ("ngtdm", "Coarseness"): "Coarseness",
    ("ngtdm", "Contrast"): "Contrast",
    ("ngtdm", "Busyness"): "Busyness",
    ("ngtdm", "Complexity"): "Complexity",
    ("ngtdm", "Strength"): "Strength",
}


# These are not wrapper-derived approximations.  PyRadiomics 3.1.0 explicitly
# documents each source output as mathematically identical to the second IBSI
# definition.  The deprecated names themselves intentionally return no value,
# so retain the active source token and disclose its shared semantic provenance.
# DependencePercentage is deliberately absent: PyRadiomics documents it as an
# invariant but emits no active scalar output for it.
PYRADIOMICS_DOCUMENTED_EXACT_ALIASES: Dict[str, Dict[str, str]] = {
    "original_glcm_DifferenceAverage": {
        "8S9J": "PyRadiomics documents Dissimilarity as mathematically equal to Difference Average",
    },
    "original_glcm_ClusterTendency": {
        "OEEB": "PyRadiomics documents Sum Variance as mathematically equal to Cluster Tendency",
    },
    "original_firstorder_Uniformity": {
        "5SPA": "PyRadiomics documents NGLDM normalised GLNU as mathematically equal to first-order Uniformity",
    },
}

PYRADIOMICS_EXACT_ALIAS_EVIDENCE = (
    "https://pyradiomics.readthedocs.io/en/v3.1.0/removedfeatures.html"
)

PYRADIOMICS_EXPLICIT_DEPRECATED_FEATURES = {
    ("shape", "Compactness1"),
    ("shape", "Compactness2"),
    ("shape", "SphericalDisproportion"),
}


PYRADIOMICS_FAMILY_TO_IBSI = {
    "shape": "morphology",
    "firstorder": None,
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldm": "ngldm",
    "ngtdm": "ngtdm",
}


PYRADIOMICS_CODE_TO_FEATURES: Dict[str, List[tuple[str, str]]] = {}
for (family, feat), ibsi_name in PYRADIOMICS_MAP.items():
    ibsi_family = PYRADIOMICS_FAMILY_TO_IBSI.get(family)
    code = _code_from_name_in_family(ibsi_name, ibsi_family)
    if code:
        PYRADIOMICS_CODE_TO_FEATURES.setdefault(code, []).append((family, feat))

for _source_name, _aliases in PYRADIOMICS_DOCUMENTED_EXACT_ALIASES.items():
    _source_parts = _source_name.split("_", 2)
    if len(_source_parts) != 3 or _source_parts[0] != "original":
        raise RuntimeError(f"Invalid PyRadiomics exact-alias source: {_source_name}")
    _source_feature = (_source_parts[1], _source_parts[2])
    for _alias_code in _aliases:
        PYRADIOMICS_CODE_TO_FEATURES.setdefault(_alias_code, []).append(_source_feature)


def map_pyradiomics(feature_name: str) -> Optional[str]:
    parts = feature_name.split("_")
    if len(parts) < 3:
        return None
    if parts[0] != "original":
        return None
    family = parts[1]
    feat = "_".join(parts[2:])
    ibsi_name = PYRADIOMICS_MAP.get((family, feat))
    if not ibsi_name:
        return None
    ibsi_family = PYRADIOMICS_FAMILY_TO_IBSI.get(family)
    return _code_from_name_in_family(ibsi_name, ibsi_family)


def pyradiomics_semantic_aliases(feature_name: str) -> Dict[str, str]:
    """Return documented exact IBSI identities for one emitted native token."""

    return dict(PYRADIOMICS_DOCUMENTED_EXACT_ALIASES.get(feature_name, {}))


def documented_semantic_aliases(adapter: str, feature_name: str) -> Dict[str, str]:
    """Return reviewed one-to-many semantic identities for an adapter output."""

    if str(adapter).strip().lower() == "pyradiomics":
        return pyradiomics_semantic_aliases(feature_name)
    return {}


def pyradiomics_feature_selection(codes: Iterable[str]) -> Dict[str, List[str]]:
    features: Dict[str, List[str]] = {}
    for code in codes:
        for family, feat in PYRADIOMICS_CODE_TO_FEATURES.get(code, []):
            features.setdefault(family, [])
            if feat not in features[family]:
                features[family].append(feat)
    for family in features:
        features[family].sort()
    return features


MIRP_PREFIX_MAP = {
    # morphology
    "morph_volume": "Volume",
    "morph_vol_approx": "Volume voxel counting",
    "morph_area_mesh": "Surface area",
    "morph_av": "Surface to volume ratio",
    "morph_comp_1": "Compactness 1",
    "morph_comp_2": "Compactness 2",
    "morph_sph_dispr": "Spherical disproportion",
    "morph_sphericity": "Sphericity",
    "morph_asphericity": "Asphericity",
    "morph_com": "Center of mass shift",
    "morph_integ_int": "Integrated intensity",
    "morph_diam": "Maximum 3D diameter",
    "morph_vol_dens_conv_hull": "Volume density (convex hull)",
    "morph_area_dens_conv_hull": "Area density (convex hull)",
    "morph_vol_dens_aabb": "Volume density (AABB)",
    "morph_area_dens_aabb": "Area density (AABB)",
    "morph_pca_maj_axis": "Major axis length",
    "morph_pca_min_axis": "Minor axis length",
    "morph_pca_least_axis": "Least axis length",
    "morph_pca_elongation": "Elongation",
    "morph_pca_flatness": "Flatness",
    "morph_vol_dens_aee": "Volume density (AEE)",
    "morph_area_dens_aee": "Area density (AEE)",
    "morph_vol_dens_ombb": "Volume density (OMBB)",
    "morph_area_dens_ombb": "Area density (OMBB)",
    "morph_vol_dens_mvee": "Volume density (MVEE)",
    "morph_area_dens_mvee": "Area density (MVEE)",
    "morph_moran_i": "Moran's I index",
    "morph_geary_c": "Geary's C measure",
    # local intensity
    "loc_peak_loc": "Local intensity peak ",
    "loc_peak_glob": "Global intensity peak",
    # intensity stats
    "stat_mean": "Mean intensity",
    "stat_var": "Intensity variance",
    "stat_skew": "Intensity skewness",
    "stat_kurt": "Intensity kurtosis",
    "stat_median": "Median intensity",
    "stat_min": "Minimum intensity",
    "stat_p10": "10th intensity percentile",
    "stat_p90": "90th intensity percentile",
    "stat_max": "Maximum intensity",
    "stat_iqr": "Intensity interquartile range",
    "stat_range": "Intensity range",
    "stat_mad": "Intensity Mean absolute deviation",
    "stat_rmad": "Intensity Robust mean absolute deviation",
    "stat_medad": "Intensity Median absolute deviation",
    "stat_cov": "Intensity Coefficient of variation",
    "stat_qcod": "Intensity Quartile coefficient of dispersion",
    "stat_energy": "Intensity energy",
    "stat_rms": "Root mean square intensity",
    # intensity histogram
    "ih_mean": "Mean discretised intensity",
    "ih_var": "Discretised intensity variance",
    "ih_skew": "Discretised intensity skewness",
    "ih_kurt": "Discretised intensity kurtosis",
    "ih_median": "Median discretised intensity",
    "ih_min": "Minimum discretised intensity",
    "ih_p10": "10th discretised intensity percentile",
    "ih_p90": "90th discretised intensity percentile",
    "ih_max": "Maximum discretised intensity",
    "ih_mode": "Intensity histogram mode",
    "ih_iqr": "Discretised intensity interquartile range",
    "ih_range": "Discretised intensity range",
    "ih_mad": "Intensity histogram mean absolute deviation",
    "ih_rmad": "Intensity histogram robust mean absolute deviation",
    "ih_medad": "Intensity histogram median absolute deviation",
    "ih_cov": "Intensity histogram coefficient of variation",
    "ih_qcod": "Intensity histogram quartile coefficient of dispersion",
    "ih_entropy": "Discretised intensity entropy",
    "ih_uniformity": "Discretised intensity uniformity",
    "ih_max_grad": "Maximum histogram gradient",
    "ih_max_grad_g": "Maximum histogram gradient intensity",
    "ih_min_grad": "Minimum histogram gradient",
    "ih_min_grad_g": "Minimum histogram gradient intensity",
    # IVH (only IBSI-defined percentiles map cleanly)
    "ivh_v10": "Volume at intensity fraction 0.10",
    "ivh_v90": "Volume at intensity fraction 0.90",
    "ivh_i10": "Intensity at volume fraction 0.10",
    "ivh_i90": "Intensity at volume fraction 0.90",
    "ivh_diff_v10_v90": "Volume fraction difference between intensity 0.10 and 0.90 fractions",
    "ivh_diff_i10_i90": "Intensity fraction difference between volume 0.10 and 0.90 fractions",
    "ivh_auc": "Area under the IVH curve",
    # GLCM (cm_*)
    "cm_joint_max": "Joint maximum",
    "cm_joint_avg": "Joint average",
    "cm_joint_var": "Joint variance",
    "cm_joint_entr": "Joint entropy",
    "cm_diff_avg": "Difference average",
    "cm_diff_var": "Difference variance",
    "cm_diff_entr": "Difference entropy",
    "cm_sum_avg": "Sum average",
    "cm_sum_var": "Sum variance",
    "cm_sum_entr": "Sum entropy",
    "cm_energy": "Angular second moment",
    "cm_contrast": "Contrast",
    "cm_dissimilarity": "Dissimilarity",
    "cm_inv_diff": "Inverse difference",
    "cm_inv_diff_norm": "Normalised inverse difference",
    "cm_inv_diff_mom": "Inverse difference moment",
    "cm_inv_diff_mom_norm": "Normalised inverse difference moment",
    "cm_inv_var": "Inverse variance",
    "cm_corr": "Correlation",
    "cm_auto_corr": "Autocorrelation",
    "cm_clust_tend": "Cluster tendency",
    "cm_clust_shade": "Cluster shade",
    "cm_clust_prom": "Cluster prominence",
    "cm_info_corr1": "Information correlation 1",
    "cm_info_corr2": "Information correlation 2",
    # GLRLM (rlm_*)
    "rlm_sre": "Short runs emphasis",
    "rlm_lre": "Long runs emphasis",
    "rlm_lgre": "Low grey level run emphasis",
    "rlm_hgre": "High grey level run emphasis",
    "rlm_srlge": "Short run low grey level emphasis",
    "rlm_srhge": "Short run high grey level emphasis",
    "rlm_lrlge": "Long run low grey level emphasis",
    "rlm_lrhge": "Long run high grey level emphasis",
    "rlm_glnu": "Grey level non-uniformity",
    "rlm_glnu_norm": "Normalised grey level non-uniformity",
    "rlm_rlnu": "Run length non-uniformity",
    "rlm_rlnu_norm": "Normalised run length non-uniformity",
    "rlm_r_perc": "Run percentage",
    "rlm_gl_var": "Grey level variance",
    "rlm_rl_var": "Run length variance",
    "rlm_rl_entr": "Run entropy",
    # GLSZM (szm_*)
    "szm_sze": "Small zone emphasis",
    "szm_lze": "Large zone emphasis",
    "szm_lgze": "Low grey level zone emphasis",
    "szm_hgze": "High grey level zone emphasis",
    "szm_szlge": "Small zone low grey level emphasis",
    "szm_szhge": "Small zone high grey level emphasis",
    "szm_lzlge": "Large zone low grey level emphasis",
    "szm_lzhge": "Large zone high grey level emphasis",
    "szm_glnu": "Grey level non-uniformity",
    "szm_glnu_norm": "Normalised grey level non-uniformity",
    "szm_zsnu": "Zone size non-uniformity",
    "szm_zsnu_norm": "Normalised zone size non-uniformity",
    "szm_z_perc": "Zone percentage",
    "szm_gl_var": "Grey level variance",
    "szm_zs_var": "Zone size variance",
    "szm_zs_entr": "Zone size entropy",
    # GLDZM (dzm_*)
    "dzm_sde": "Small distance emphasis",
    "dzm_lde": "Large distance emphasis",
    "dzm_lgze": "Low grey level zone emphasis",
    "dzm_hgze": "High grey level zone emphasis",
    "dzm_sdlge": "Small distance low grey level emphasis",
    "dzm_sdhge": "Small distance high grey level emphasis",
    "dzm_ldlge": "Large distance low grey level emphasis",
    "dzm_ldhge": "Large distance high grey level emphasis",
    "dzm_glnu": "Grey level non-uniformity",
    "dzm_glnu_norm": "Normalised grey level non-uniformity",
    "dzm_zdnu": "Zone distance non-uniformity",
    "dzm_zdnu_norm": "Normalised zone distance non-uniformity",
    "dzm_z_perc": "Zone percentage",
    "dzm_gl_var": "Grey level variance",
    "dzm_zd_var": "Zone distance variance",
    "dzm_zd_entr": "Zone distance entropy",
    # NGTDM (ngt_*)
    "ngt_coarseness": "Coarseness",
    "ngt_contrast": "Contrast",
    "ngt_busyness": "Busyness",
    "ngt_complexity": "Complexity",
    "ngt_strength": "Strength",
    # NGLDM (ngl_*)
    "ngl_lde": "Low dependence emphasis",
    "ngl_hde": "High dependence emphasis",
    "ngl_lgce": "Low grey level count emphasis",
    "ngl_hgce": "High grey level count emphasis",
    "ngl_ldlge": "Low dependence low grey level emphasis",
    "ngl_ldhge": "Low dependence high grey level emphasis",
    "ngl_hdlge": "High dependence low grey level emphasis",
    "ngl_hdhge": "High dependence high grey level emphasis",
    "ngl_glnu": "Grey level non-uniformity",
    "ngl_glnu_norm": "Normalised grey level non-uniformity",
    "ngl_dcnu": "Dependence count non-uniformity",
    "ngl_dcnu_norm": "Normalised dependence count non-uniformity",
    "ngl_dc_perc": "Dependence count percentage",
    "ngl_gl_var": "Grey level variance",
    "ngl_dc_var": "Dependence count variance",
    "ngl_dc_entr": "Dependence count entropy",
    "ngl_dc_energy": "Dependence count energy",
}


MIRP_PREFIX_TO_FAMILY = {
    "morph_": "morphology",
    "loc_": "local_intensity",
    "stat_": "intensity",
    "ih_": "histogram",
    "ivh_": "ivh",
    "cm_": "glcm",
    "rlm_": "glrlm",
    "szm_": "glszm",
    "dzm_": "gldzm",
    "ngt_": "ngtdm",
    "ngl_": "ngldm",
}


def _mirp_family_for_prefix(prefix: str) -> Optional[str]:
    for key, family in MIRP_PREFIX_TO_FAMILY.items():
        if prefix.startswith(key):
            return family
    return None


def map_mirp(feature_name: str) -> Optional[str]:
    # Remove discretization suffixes like _fbn_n32
    base = feature_name
    base = re.sub(r"_fbn_n\d+", "", base)
    base = re.sub(r"_fbs_w[0-9.]+", "", base)

    # MIRP feature names can include extra tokens; find longest prefix match
    for prefix in sorted(MIRP_PREFIX_MAP, key=len, reverse=True):
        ibsi_name = MIRP_PREFIX_MAP[prefix]
        if base.startswith(prefix):
            family = _mirp_family_for_prefix(prefix)
            return _code_from_name_in_family(ibsi_name, family)

    return None


RADIOMICSJ_PREFIX_TO_FAMILY = {
    "Morphology": "morphology",
    "LocalIntensity": "local_intensity",
    "IntensityBasedStatistical": "intensity",
    "IntensityHistogram": "histogram",
    "IntensityVolumeHistogram": "ivh",
    "GLCM": "glcm",
    "GLRLM": "glrlm",
    "GLSZM": "glszm",
    "GLDZM": "gldzm",
    "NGTDM": "ngtdm",
    "NGLDM": "ngldm",
}

RADIOMICSJ_MAP: Dict[tuple[str, str], str] = {
    ("Morphology", "VolumeMesh"): "Volume",
    ("Morphology", "VolumeVoxelCounting"): "Volume voxel counting",
    ("Morphology", "SurfaceAreaMesh"): "Surface area",
    ("Morphology", "SurfaceToVolumeRatio"): "Surface to volume ratio",
    ("Morphology", "Compactness1"): "Compactness 1",
    ("Morphology", "Compactness2"): "Compactness 2",
    ("Morphology", "SphericalDisproportion"): "Spherical disproportion",
    ("Morphology", "Sphericity"): "Sphericity",
    ("Morphology", "Asphericity"): "Asphericity",
    ("Morphology", "CentreOfMassShift"): "Center of mass shift",
    ("Morphology", "Maximum3DDiameter"): "Maximum 3D diameter",
    ("Morphology", "MajorAxisLength"): "Major axis length",
    ("Morphology", "MinorAxisLength"): "Minor axis length",
    ("Morphology", "LeastAxisLength"): "Least axis length",
    ("Morphology", "Elongation"): "Elongation",
    ("Morphology", "Flatness"): "Flatness",
    ("Morphology", "VolumeDensity_AxisAlignedBoundingBox"): "Volume density (AABB)",
    ("Morphology", "AreaDensity_AxisAlignedBoundingBox"): "Area density (AABB)",
    (
        "Morphology",
        "VolumeDensity_ApproximateEnclosingEllipsoid",
    ): "Volume density (AEE)",
    ("Morphology", "AreaDensity_ApproximateEnclosingEllipsoid"): "Area density (AEE)",
    ("Morphology", "VolumeDensity_ConvexHull"): "Volume density (convex hull)",
    ("Morphology", "AreaDensity_ConvexHull"): "Area density (convex hull)",
    ("Morphology", "IntegratedIntensity"): "Integrated intensity",
    ("LocalIntensity", "LocalIntensityPeak"): "Local intensity peak ",
    ("LocalIntensity", "GlobalIntensityPeak"): "Global intensity peak",
    ("IntensityBasedStatistical", "Mean"): "Mean intensity",
    ("IntensityBasedStatistical", "Variance"): "Intensity variance",
    ("IntensityBasedStatistical", "Skewness"): "Intensity skewness",
    ("IntensityBasedStatistical", "Kurtosis"): "Intensity kurtosis",
    ("IntensityBasedStatistical", "Median"): "Median intensity",
    ("IntensityBasedStatistical", "Minimum"): "Minimum intensity",
    ("IntensityBasedStatistical", "Percentile10"): "10th intensity percentile",
    ("IntensityBasedStatistical", "Percentile90"): "90th intensity percentile",
    ("IntensityBasedStatistical", "Maximum"): "Maximum intensity",
    ("IntensityBasedStatistical", "Interquartile"): "Intensity interquartile range",
    ("IntensityBasedStatistical", "Range"): "Intensity range",
    (
        "IntensityBasedStatistical",
        "MeanAbsoluteDeviation",
    ): "Intensity Mean absolute deviation",
    (
        "IntensityBasedStatistical",
        "RobustMeanAbsoluteDeviation",
    ): "Intensity Robust mean absolute deviation",
    (
        "IntensityBasedStatistical",
        "MedianAbsoluteDeviation",
    ): "Intensity Median absolute deviation",
    (
        "IntensityBasedStatistical",
        "CoefficientOfVariation",
    ): "Intensity Coefficient of variation",
    (
        "IntensityBasedStatistical",
        "QuartileCoefficientOfDispersion",
    ): "Intensity Quartile coefficient of dispersion",
    ("IntensityBasedStatistical", "Energy"): "Intensity energy",
    ("IntensityBasedStatistical", "RootMeanSquared"): "Root mean square intensity",
    ("IntensityHistogram", "MeanDiscretisedIntensity"): "Mean discretised intensity",
    ("IntensityHistogram", "Variance"): "Discretised intensity variance",
    ("IntensityHistogram", "Skewness"): "Discretised intensity skewness",
    ("IntensityHistogram", "Kurtosis"): "Discretised intensity kurtosis",
    ("IntensityHistogram", "Median"): "Median discretised intensity",
    ("IntensityHistogram", "Minimum"): "Minimum discretised intensity",
    ("IntensityHistogram", "Percentile10"): "10th discretised intensity percentile",
    ("IntensityHistogram", "Percentile90"): "90th discretised intensity percentile",
    ("IntensityHistogram", "Maximum"): "Maximum discretised intensity",
    ("IntensityHistogram", "Mode"): "Intensity histogram mode",
    (
        "IntensityHistogram",
        "Interquartile",
    ): "Discretised intensity interquartile range",
    ("IntensityHistogram", "Range"): "Discretised intensity range",
    (
        "IntensityHistogram",
        "MeanAbsoluteDeviation",
    ): "Intensity histogram mean absolute deviation",
    (
        "IntensityHistogram",
        "RobustMeanAbsoluteDeviation",
    ): "Intensity histogram robust mean absolute deviation",
    (
        "IntensityHistogram",
        "MedianAbsoluteDeviation",
    ): "Intensity histogram median absolute deviation",
    (
        "IntensityHistogram",
        "CoefficientOfVariation",
    ): "Intensity histogram coefficient of variation",
    (
        "IntensityHistogram",
        "QuartileCoefficientOfDispersion",
    ): "Intensity histogram quartile coefficient of dispersion",
    ("IntensityHistogram", "Entropy"): "Discretised intensity entropy",
    ("IntensityHistogram", "Uniformity"): "Discretised intensity uniformity",
    ("IntensityHistogram", "MaximumHistogramGradient"): "Maximum histogram gradient",
    (
        "IntensityHistogram",
        "MaximumHistogramGradientIntensity",
    ): "Maximum histogram gradient intensity",
    ("IntensityHistogram", "MinimumHistogramGradient"): "Minimum histogram gradient",
    (
        "IntensityHistogram",
        "MinimumHistogramGradientIntensity",
    ): "Minimum histogram gradient intensity",
    (
        "IntensityVolumeHistogram",
        "VolumeAtIntensityFraction10",
    ): "Volume at intensity fraction 0.10",
    (
        "IntensityVolumeHistogram",
        "VolumeAtIntensityFraction90",
    ): "Volume at intensity fraction 0.90",
    (
        "IntensityVolumeHistogram",
        "IntensityAtVolumeFraction10",
    ): "Intensity at volume fraction 0.10",
    (
        "IntensityVolumeHistogram",
        "IntensityAtVolumeFraction90",
    ): "Intensity at volume fraction 0.90",
    (
        "IntensityVolumeHistogram",
        "VolumeFractionDifferenceBetweenIntensityFractions",
    ): "Volume fraction difference between intensity 0.10 and 0.90 fractions",
    (
        "IntensityVolumeHistogram",
        "IntensityFractionDifferenceBetweenVolumeFractions",
    ): "Intensity fraction difference between volume 0.10 and 0.90 fractions",
    ("GLCM", "JointMaximum"): "Joint maximum",
    ("GLCM", "JointAverage"): "Joint average",
    ("GLCM", "JointVariance"): "Joint variance",
    ("GLCM", "JointEntropy"): "Joint entropy",
    ("GLCM", "DifferenceAverage"): "Difference average",
    ("GLCM", "DifferenceVariance"): "Difference variance",
    ("GLCM", "DifferenceEntropy"): "Difference entropy",
    ("GLCM", "SumAverage"): "Sum average",
    ("GLCM", "SumVariance"): "Sum variance",
    ("GLCM", "SumEntropy"): "Sum entropy",
    ("GLCM", "AngularSecondMoment"): "Angular second moment",
    ("GLCM", "Contrast"): "Contrast",
    ("GLCM", "Dissimilarity"): "Dissimilarity",
    ("GLCM", "InverseDifference"): "Inverse difference",
    ("GLCM", "NormalizedInverseDifference"): "Normalised inverse difference",
    ("GLCM", "InverseDifferenceMoment"): "Inverse difference moment",
    (
        "GLCM",
        "NormalizedInverseDifferenceMoment",
    ): "Normalised inverse difference moment",
    ("GLCM", "InverseVariance"): "Inverse variance",
    ("GLCM", "Correlation"): "Correlation",
    ("GLCM", "Autocorrelation"): "Autocorrelation",
    ("GLCM", "Autocorrection"): "Autocorrelation",
    ("GLCM", "ClusterTendency"): "Cluster tendency",
    ("GLCM", "ClusterShade"): "Cluster shade",
    ("GLCM", "ClusterProminence"): "Cluster prominence",
    ("GLCM", "InformationalMeasureOfCorrelation1"): "Information correlation 1",
    ("GLCM", "InformationalMeasureOfCorrelation2"): "Information correlation 2",
    ("GLRLM", "ShortRunEmphasis"): "Short runs emphasis",
    ("GLRLM", "LongRunEmphasis"): "Long runs emphasis",
    ("GLRLM", "LowGrayLevelRunEmphasis"): "Low grey level run emphasis",
    ("GLRLM", "HighGrayLevelRunEmphasis"): "High grey level run emphasis",
    ("GLRLM", "ShortRunLowGrayLevelEmphasis"): "Short run low grey level emphasis",
    ("GLRLM", "ShortRunHighGrayLevelEmphasis"): "Short run high grey level emphasis",
    ("GLRLM", "LongRunLowGrayLevelEmphasis"): "Long run low grey level emphasis",
    ("GLRLM", "LongRunHighGrayLevelEmphasis"): "Long run high grey level emphasis",
    ("GLRLM", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "GLRLM",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("GLRLM", "RunLengthNonUniformity"): "Run length non-uniformity",
    (
        "GLRLM",
        "RunLengthNonUniformityNormalized",
    ): "Normalised run length non-uniformity",
    ("GLRLM", "RunPercentage"): "Run percentage",
    ("GLRLM", "GrayLevelVariance"): "Grey level variance",
    ("GLRLM", "RunVariance"): "Run length variance",
    ("GLRLM", "RunLengthVariance"): "Run length variance",
    ("GLRLM", "RunEntropy"): "Run entropy",
    ("GLSZM", "SmallZoneEmphasis"): "Small zone emphasis",
    ("GLSZM", "LargeZoneEmphasis"): "Large zone emphasis",
    ("GLSZM", "LowGrayLevelZoneEmphasis"): "Low grey level zone emphasis",
    ("GLSZM", "HighGrayLevelZoneEmphasis"): "High grey level zone emphasis",
    ("GLSZM", "SmallZoneLowGrayLevelEmphasis"): "Small zone low grey level emphasis",
    ("GLSZM", "SmallZoneHighGrayLevelEmphasis"): "Small zone high grey level emphasis",
    ("GLSZM", "LargeZoneLowGrayLevelEmphasis"): "Large zone low grey level emphasis",
    ("GLSZM", "LargeZoneHighGrayLevelEmphasis"): "Large zone high grey level emphasis",
    ("GLSZM", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "GLSZM",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("GLSZM", "ZoneSizeNonUniformity"): "Zone size non-uniformity",
    ("GLSZM", "ZoneSizeNonUniformityNormalized"): "Normalised zone size non-uniformity",
    ("GLSZM", "SizeZoneNonUniformity"): "Zone size non-uniformity",
    ("GLSZM", "SizeZoneNonUniformityNormalized"): "Normalised zone size non-uniformity",
    ("GLSZM", "ZonePercentage"): "Zone percentage",
    ("GLSZM", "GrayLevelVariance"): "Grey level variance",
    ("GLSZM", "ZoneSizeVariance"): "Zone size variance",
    ("GLSZM", "ZoneSizeEntropy"): "Zone size entropy",
    ("GLDZM", "SmallDistanceEmphasis"): "Small distance emphasis",
    ("GLDZM", "LargeDistanceEmphasis"): "Large distance emphasis",
    ("GLDZM", "LowGrayLevelZoneEmphasis"): "Low grey level zone emphasis",
    ("GLDZM", "HighGrayLevelZoneEmphasis"): "High grey level zone emphasis",
    (
        "GLDZM",
        "SmallDistanceLowGrayLevelEmphasis",
    ): "Small distance low grey level emphasis",
    (
        "GLDZM",
        "SmallDistanceHighGrayLevelEmphasis",
    ): "Small distance high grey level emphasis",
    (
        "GLDZM",
        "LargeDistanceLowGrayLevelEmphasis",
    ): "Large distance low grey level emphasis",
    (
        "GLDZM",
        "LargeDistanceHighGrayLevelEmphasis",
    ): "Large distance high grey level emphasis",
    ("GLDZM", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "GLDZM",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("GLDZM", "ZoneDistanceNonUniformity"): "Zone distance non-uniformity",
    (
        "GLDZM",
        "ZoneDistanceNonUniformityNormalized",
    ): "Normalised zone distance non-uniformity",
    ("GLDZM", "ZonePercentage"): "Zone percentage",
    ("GLDZM", "GrayLevelVariance"): "Grey level variance",
    ("GLDZM", "ZoneDistanceVariance"): "Zone distance variance",
    ("GLDZM", "ZoneDistanceEntropy"): "Zone distance entropy",
    ("NGTDM", "Coarseness"): "Coarseness",
    ("NGTDM", "Contrast"): "Contrast",
    ("NGTDM", "Busyness"): "Busyness",
    ("NGTDM", "Complexity"): "Complexity",
    ("NGTDM", "Strength"): "Strength",
    ("NGLDM", "LowDependenceEmphasis"): "Low dependence emphasis",
    ("NGLDM", "HighDependenceEmphasis"): "High dependence emphasis",
    ("NGLDM", "LowGrayLevelCountEmphasis"): "Low grey level count emphasis",
    ("NGLDM", "HighGrayLevelCountEmphasis"): "High grey level count emphasis",
    (
        "NGLDM",
        "LowDependenceLowGrayLevelEmphasis",
    ): "Low dependence low grey level emphasis",
    (
        "NGLDM",
        "LowDependenceHighGrayLevelEmphasis",
    ): "Low dependence high grey level emphasis",
    (
        "NGLDM",
        "HighDependenceLowGrayLevelEmphasis",
    ): "High dependence low grey level emphasis",
    (
        "NGLDM",
        "HighDependenceHighGrayLevelEmphasis",
    ): "High dependence high grey level emphasis",
    ("NGLDM", "GrayLevelNonUniformity"): "Grey level non-uniformity",
    (
        "NGLDM",
        "GrayLevelNonUniformityNormalized",
    ): "Normalised grey level non-uniformity",
    ("NGLDM", "DependenceCountNonUniformity"): "Dependence count non-uniformity",
    (
        "NGLDM",
        "DependenceCountNonUniformityNormalized",
    ): "Normalised dependence count non-uniformity",
    ("NGLDM", "DependenceCountPercentage"): "Dependence count percentage",
    ("NGLDM", "GrayLevelVariance"): "Grey level variance",
    ("NGLDM", "DependenceCountVariance"): "Dependence count variance",
    ("NGLDM", "DependenceCountEntropy"): "Dependence count entropy",
    ("NGLDM", "DependenceCountEnergy"): "Dependence count energy",
}


def map_radiomicsj(feature_name: str) -> Optional[str]:
    parts = feature_name.split("_", 1)
    if len(parts) != 2:
        return None
    family_prefix, suffix = parts
    family = RADIOMICSJ_PREFIX_TO_FAMILY.get(family_prefix)
    if not family:
        return None
    ibsi_name = RADIOMICSJ_MAP.get((family_prefix, suffix))
    if not ibsi_name:
        return None
    return _code_from_name_in_family(ibsi_name, family)


CAPTK_OUTPUT_FAMILY_TO_IBSI = {
    "Intensity": "intensity",
    "Histogram": "histogram",
    "Volumetric": "morphology",
    "Morphologic": "morphology",
    "GLCM": "glcm",
    "GLRLM": "glrlm",
    "GLSZM": "glszm",
    "NGTDM": "ngtdm",
    "NGLDM": "ngldm",
}

CAPTK_INTENSITY_MAP: Dict[str, str] = {
    "minimum": "Minimum intensity",
    "maximum": "Maximum intensity",
    "mean": "Mean intensity",
    "median": "Median intensity",
    "variance": "Intensity variance",
    "skewness": "Intensity skewness",
    "kurtosis": "Intensity kurtosis",
    "range": "Intensity range",
    "interquartilerange": "Intensity interquartile range",
    "meanabsolutedeviation": "Intensity Mean absolute deviation",
    "robustmeanabsolutedeviation1090": "Intensity Robust mean absolute deviation",
    "medianabsolutedeviation": "Intensity Median absolute deviation",
    "coefficientofvariation": "Intensity Coefficient of variation",
    "quartilecoefficientofvariation": "Intensity Quartile coefficient of dispersion",
    "energy": "Intensity energy",
    "rootmeansquare": "Root mean square intensity",
    "tenthpercentile": "10th intensity percentile",
    "ninetiethpercentile": "90th intensity percentile",
}

CAPTK_HISTOGRAM_MAP: Dict[str, str] = {
    "mean": "Mean discretised intensity",
    "variance": "Discretised intensity variance",
    "skewness": "Discretised intensity skewness",
    "kurtosis": "Discretised intensity kurtosis",
    "median": "Median discretised intensity",
    "minimum": "Minimum discretised intensity",
    "maximum": "Maximum discretised intensity",
    "mode": "Intensity histogram mode",
    "interquartilerange": "Discretised intensity interquartile range",
    "range": "Discretised intensity range",
    "meanabsolutedeviation": "Intensity histogram mean absolute deviation",
    "robustmeanabsolutedeviation1090": "Intensity histogram robust mean absolute deviation",
    "medianabsolutedeviation": "Intensity histogram median absolute deviation",
    "coefficientofvariation": "Intensity histogram coefficient of variation",
    "quartilecoefficientofvariation": "Intensity histogram quartile coefficient of dispersion",
    "entropy": "Discretised intensity entropy",
    "uniformity": "Discretised intensity uniformity",
    "tenthpercentile": "10th discretised intensity percentile",
    "ninetiethpercentile": "90th discretised intensity percentile",
}

CAPTK_MORPHOLOGY_MAP: Dict[str, str] = {
    # CaPTk commonly exposes voxel-integrated physical size rather than mesh volume.
    "volume": "Volume voxel counting",
    "physicalsize": "Volume voxel counting",
    "numberofpixels": "Volume voxel counting",
    "pixels": "Volume voxel counting",
    "flatness": "Flatness",
    "elongation": "Elongation",
    "feretdiameter": "Maximum 3D diameter",
}

CAPTK_GLCM_MAP: Dict[str, str] = {
    "energy": "Angular second moment",
    "angularsecondmoment": "Angular second moment",
    "entropy": "Joint entropy",
    "jointentropy": "Joint entropy",
    "homogeneity": "Inverse difference moment",
    "inversedifferencemoment": "Inverse difference moment",
    "contrast": "Contrast",
    "correlation": "Correlation",
    "sumaverage": "Sum average",
    "variance": "Joint variance",
    "clustershade": "Cluster shade",
    "clusterprominence": "Cluster prominence",
    "autocorrelation": "Autocorrelation",
}

CAPTK_GLRLM_MAP: Dict[str, str] = {
    "shortrunemphasis": "Short runs emphasis",
    "longrunemphasis": "Long runs emphasis",
    "lowgreylevelrunemphasis": "Low grey level run emphasis",
    "lowgraylevelrunemphasis": "Low grey level run emphasis",
    "highgreylevelrunemphasis": "High grey level run emphasis",
    "highgraylevelrunemphasis": "High grey level run emphasis",
    "shortrunlowgreylevelemphasis": "Short run low grey level emphasis",
    "shortrunlowgraylevelemphasis": "Short run low grey level emphasis",
    "shortrunhighgreylevelemphasis": "Short run high grey level emphasis",
    "shortrunhighgraylevelemphasis": "Short run high grey level emphasis",
    "longrunlowgreylevelemphasis": "Long run low grey level emphasis",
    "longrunlowgraylevelemphasis": "Long run low grey level emphasis",
    "longrunhighgreylevelemphasis": "Long run high grey level emphasis",
    "longrunhighgraylevelemphasis": "Long run high grey level emphasis",
    "greylevelnonuniformity": "Grey level non-uniformity",
    "graylevelnonuniformity": "Grey level non-uniformity",
    "runlengthnonuniformity": "Run length non-uniformity",
    "runpercentage": "Run percentage",
}

CAPTK_GLSZM_MAP: Dict[str, str] = {
    "smallzoneemphasis": "Small zone emphasis",
    "largezoneemphasis": "Large zone emphasis",
    "lowgreylevelemphasis": "Low grey level zone emphasis",
    "highgreylevelemphasis": "High grey level zone emphasis",
    "smallzonelowgreylevelemphasis": "Small zone low grey level emphasis",
    "smallzonehighgreylevelemphasis": "Small zone high grey level emphasis",
    "largezonelowgreylevelemphasis": "Large zone low grey level emphasis",
    "largezonehighgreylevelemphasis": "Large zone high grey level emphasis",
    "greylevelnonuniformity": "Grey level non-uniformity",
    "greylevelnonuniformitynormalized": "Normalised grey level non-uniformity",
    "zonesizenonuniformity": "Zone size non-uniformity",
    "zonesizenoneuniformitynormalized": "Normalised zone size non-uniformity",
    "zonepercentage": "Zone percentage",
    "greylevelvariance": "Grey level variance",
    "zonesizevariance": "Zone size variance",
    "zonesizeentropy": "Zone size entropy",
}

CAPTK_NGTDM_MAP: Dict[str, str] = {
    "coarsness": "Coarseness",
    "coarseness": "Coarseness",
    "contrast": "Contrast",
    "busyness": "Busyness",
    "complexity": "Complexity",
    "strength": "Strength",
}

CAPTK_NGLDM_MAP: Dict[str, str] = {
    "lowdependenceemphasis": "Low dependence emphasis",
    "highdependenceemphasis": "High dependence emphasis",
    "lowgraylevelcountemphasis": "Low grey level count emphasis",
    "highgraylevelcountemphasis": "High grey level count emphasis",
    "lowdependencelowgraylevelemphasis": "Low dependence low grey level emphasis",
    "lowdependencehighgraylevelemphasis": "Low dependence high grey level emphasis",
    "highdependencelowgraylevelemphasis": "High dependence low grey level emphasis",
    "highdependencehighgraylevelemphasis": "High dependence high grey level emphasis",
    "graylevelnonuniformity": "Grey level non-uniformity",
    "graylevelnonuniformitynormalized": "Normalised grey level non-uniformity",
    "dependencecountnonuniformity": "Dependence count non-uniformity",
    "dependencecountnonuniformitynormalized": "Normalised dependence count non-uniformity",
    "dependencecountpercentage": "Dependence count percentage",
    "graylevelvariance": "Grey level variance",
    "dependencecountvariance": "Dependence count variance",
    "entropy": "Dependence count entropy",
    "energy": "Dependence count energy",
}

CAPTK_FAMILY_MAPS: Dict[str, Dict[str, str]] = {
    "Intensity": CAPTK_INTENSITY_MAP,
    "Histogram": CAPTK_HISTOGRAM_MAP,
    "Volumetric": CAPTK_MORPHOLOGY_MAP,
    "Morphologic": CAPTK_MORPHOLOGY_MAP,
    "GLCM": CAPTK_GLCM_MAP,
    "GLRLM": CAPTK_GLRLM_MAP,
    "GLSZM": CAPTK_GLSZM_MAP,
    "NGTDM": CAPTK_NGTDM_MAP,
    "NGLDM": CAPTK_NGLDM_MAP,
}

CAPTK_FEATURE_PARAM_PREFIXES = (
    "Bins-",
    "Radius-",
    "Directions-",
    "Axis-",
    "Dimension-",
    "Range-",
    "OffsetType-",
    "Neighborhood-",
)


def _captk_split_feature(feature_name: str) -> tuple[Optional[str], Optional[str]]:
    parts = feature_name.split("_", 2)
    if len(parts) < 3:
        return None, None
    tail = parts[2]

    family_token = None
    rest = None
    for family in CAPTK_OUTPUT_FAMILY_TO_IBSI:
        if tail == family:
            family_token = family
            rest = ""
            break
        prefix = f"{family}_"
        if tail.startswith(prefix):
            family_token = family
            rest = tail[len(prefix) :]
            break

    if not family_token:
        return None, None

    token_parts = [p for p in (rest or "").split("_") if p]
    while token_parts:
        first = token_parts[0]
        if first in {"X", "Y", "Z", "2D", "3D"}:
            token_parts = token_parts[1:]
            continue
        if any(first.startswith(prefix) for prefix in CAPTK_FEATURE_PARAM_PREFIXES):
            token_parts = token_parts[1:]
            continue
        break

    if not token_parts:
        return family_token, None

    feature_token = "_".join(token_parts)
    feature_token = re.sub(r"_Offset_[0-9]+$", "", feature_token, flags=re.IGNORECASE)
    feature_token = re.sub(r"_Label-?[0-9]+$", "", feature_token, flags=re.IGNORECASE)
    return family_token, feature_token


def map_captk(feature_name: str) -> Optional[str]:
    family_token, feature_token = _captk_split_feature(feature_name)
    if not family_token or not feature_token:
        return None

    family_map = CAPTK_FAMILY_MAPS.get(family_token)
    if not family_map:
        return None
    ibsi_name = family_map.get(_normalize(feature_token))
    if not ibsi_name:
        return None

    ibsi_family = CAPTK_OUTPUT_FAMILY_TO_IBSI.get(family_token)
    if not ibsi_family:
        return None
    return _code_from_name_in_family(ibsi_name, ibsi_family)


RADIOMICS_DEVELOP_TOKEN_ALIASES: Dict[str, str] = {
    # Morphology aliases relative to MIRP naming.
    "morph_vol": "morph_volume",
    "morph_approx_vol": "morph_vol_approx",
    "morph_area": "morph_area_mesh",
    "morph_pca_major": "morph_pca_maj_axis",
    "morph_pca_minor": "morph_pca_min_axis",
    "morph_pca_least": "morph_pca_least_axis",
    "morph_v_dens_aabb": "morph_vol_dens_aabb",
    "morph_v_dens_aee": "morph_vol_dens_aee",
    "morph_v_dens_conv_hull": "morph_vol_dens_conv_hull",
    "morph_a_dens_aabb": "morph_area_dens_aabb",
    "morph_a_dens_aee": "morph_area_dens_aee",
    "morph_a_dens_conv_hull": "morph_area_dens_conv_hull",
    # Local intensity aliases.
    "loc_peak_local": "loc_peak_loc",
    "loc_peak_global": "loc_peak_glob",
    # Median absolute deviation naming differs across snapshots.
    "stat_medmad": "stat_medad",
    "ih_medmad": "ih_medad",
    # Histogram gradient suffix aliases.
    "ih_max_grad_gl": "ih_max_grad_g",
    "ih_min_grad_gl": "ih_min_grad_g",
    # IVH difference aliases.
    "ivh_v10minusv90": "ivh_diff_v10_v90",
    "ivh_i10minusi90": "ivh_diff_i10_i90",
    # GLCM information correlation aliases.
    "cm_info_corr_1": "cm_info_corr1",
    "cm_info_corr_2": "cm_info_corr2",
    "cm_info_corr1": "cm_info_corr1",
    "cm_info_corr2": "cm_info_corr2",
}

RADIOMICS_DEVELOP_DIRECT_MAP: Dict[str, tuple[str, str]] = {
    # Present in radiomics-develop outputs but not MIRP prefix map.
    "morph_v_dens_ombb": ("Volume density (OMBB)", "morphology"),
    "morph_a_dens_ombb": ("Area density (OMBB)", "morphology"),
    "morph_v_dens_mvee": ("Volume density (MVEE)", "morphology"),
    "morph_a_dens_mvee": ("Area density (MVEE)", "morphology"),
}


def map_radiomics_develop(feature_name: str) -> Optional[str]:
    token = feature_name.strip()
    if token.startswith("F"):
        token = token[1:]
    token = token.lower()

    direct = RADIOMICS_DEVELOP_DIRECT_MAP.get(token)
    if direct:
        ibsi_name, family = direct
        return _code_from_name_in_family(ibsi_name, family)

    alias = RADIOMICS_DEVELOP_TOKEN_ALIASES.get(token, token)
    code = map_mirp(alias)
    if code:
        return code

    # Fallback for sparse naming variants.
    alias = alias.replace("__", "_")
    return map_mirp(alias)


SERA_TOKEN_ALIASES: Dict[str, str] = {
    # SERA uses _3D suffixes and aggregation suffixes for texture families.
    # We currently map 3D merged ("_comb") variants for GLCM/GLRLM.
    "cm_info_corr_1": "cm_info_corr1",
    "cm_info_corr_2": "cm_info_corr2",
    "cm_info_corr1": "cm_info_corr1",
    "cm_info_corr2": "cm_info_corr2",
    "morph_volume": "morph_volume",
    "morph_area_mesh": "morph_area_mesh",
}

SERA_DIRECT_MAP: Dict[str, tuple[str, str]] = {
    "morph_vol_dens_ombb": ("Volume density (OMBB)", "morphology"),
    "morph_area_dens_ombb": ("Area density (OMBB)", "morphology"),
    "morph_vol_dens_mvee": ("Volume density (MVEE)", "morphology"),
    "morph_area_dens_mvee": ("Area density (MVEE)", "morphology"),
}


def map_sera(feature_name: str) -> Optional[str]:
    token = feature_name.strip()
    if token.startswith("F"):
        token = token[1:]
    token = token.lower()

    # SERA has duplicated 3D texture outputs for averaged and merged variants
    # in some families. For apples-to-apples IBSI comparison we map merged only.
    if token.endswith("_3d_avg"):
        return None

    # 2D/2.5D and moment invariant outputs are intentionally excluded from
    # current parity benchmarking.
    if (
        token.startswith("mi_")
        or token.endswith("_2d")
        or token.endswith("_2_5d")
        or token.endswith("_2d_avg")
        or token.endswith("_2d_comb")
        or token.endswith("_2_5d_avg")
        or token.endswith("_2_5d_comb")
    ):
        return None

    alias = token
    if alias.endswith("_3d_comb"):
        alias = alias[: -len("_3d_comb")]
    elif alias.endswith("_3d"):
        alias = alias[: -len("_3d")]
    alias = SERA_TOKEN_ALIASES.get(alias, alias)
    direct = SERA_DIRECT_MAP.get(alias)
    if direct:
        ibsi_name, family = direct
        return _code_from_name_in_family(ibsi_name, family)
    return map_mirp(alias)


QIFE_MORPH_MAP: Dict[str, tuple[str, str]] = {
    # QIFE morphology volume is voxel-integrated physical size.
    "morph_volume": ("Volume voxel counting", "morphology"),
    "morph_surface_area": ("Surface area", "morphology"),
    "morph_surface_to_volume_ratio": ("Surface to volume ratio", "morphology"),
    "morph_max3d_diameter": ("Maximum 3D diameter", "morphology"),
    "morph_sphericity": ("Sphericity", "morphology"),
    "morph_spherical_disproportion": ("Spherical disproportion", "morphology"),
    "morph_asphericity": ("Asphericity", "morphology"),
}

QIFE_GLCM_MAP: Dict[str, tuple[str, str]] = {
    "energy": ("Angular second moment", "glcm"),
    "entropy": ("Joint entropy", "glcm"),
    "correlation": ("Correlation", "glcm"),
    "contrast": ("Contrast", "glcm"),
    "homogeneity": ("Inverse difference moment", "glcm"),
    "variance": ("Joint variance", "glcm"),
    "inertia": ("Contrast", "glcm"),
    "inversevariance": ("Inverse variance", "glcm"),
    "maxprobability": ("Joint maximum", "glcm"),
    "clustershade": ("Cluster shade", "glcm"),
    "clustertendency": ("Cluster tendency", "glcm"),
    "summean": ("Sum average", "glcm"),
}


QIFE_EXCLUDE = {
    "fqife_morph_max2d_diameter_slice",
    "morph_max2d_diameter_slice",
}


def map_qife(feature_name: str) -> Optional[str]:
    token = feature_name.strip().lower()
    if token.startswith("fqife_"):
        token = token[len("fqife_") :]

    if token.startswith("morph_"):
        match = QIFE_MORPH_MAP.get(token)
        if not match:
            return None
        ibsi_name, family = match
        return _code_from_name_in_family(ibsi_name, family)

    if token.startswith("glcm_"):
        glcm_token = token[len("glcm_") :]
        glcm_token = re.sub(r"_d[0-9]+.*$", "", glcm_token)
        glcm_token = _normalize(glcm_token)
        match = QIFE_GLCM_MAP.get(glcm_token)
        if not match:
            return None
        ibsi_name, family = match
        return _code_from_name_in_family(ibsi_name, family)

    return None


def map_medimage(feature_name: str) -> Optional[str]:
    # MEDimage output naming is MIRP/radiomics-develop aligned.
    return map_radiomics_develop(feature_name)


def map_zrad(feature_name: str) -> Optional[str]:
    token = feature_name.strip().lower()
    if token.startswith("f") and "_" in token:
        token = token[1:]

    # Aggregation suffixes used by Z-Rad texture outputs.
    for suffix in (
        "_3d_comb",
        "_3d_avg",
        "_3d",
        "_2_5d_comb",
        "_2_5d_avg",
        "_2_5d",
        "_2d_comb",
        "_2d_avg",
        "_2d",
    ):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break

    # Z-Rad feature tokens are MIRP/radiomics-develop aligned.
    return map_mirp(token)


MITK_TOKEN_MAP: Dict[str, tuple[str, str]] = {
    # intensity / histogram
    "mean": ("Mean intensity", "intensity"),
    "unbiasedvariance": ("Intensity variance", "intensity"),
    "biasedvariance": ("Intensity variance", "intensity"),
    "skewness": ("Intensity skewness", "intensity"),
    "excesskurtosis": ("Intensity kurtosis", "intensity"),
    "kurtosis": ("Intensity kurtosis", "intensity"),
    "median": ("Median intensity", "intensity"),
    "minimum": ("Minimum intensity", "intensity"),
    "maximum": ("Maximum intensity", "intensity"),
    "range": ("Intensity range", "intensity"),
    "meanabsolutedeviation": ("Intensity Mean absolute deviation", "intensity"),
    "robustmeanabsolutedeviation": (
        "Intensity Robust mean absolute deviation",
        "intensity",
    ),
    "medianabsolutedeviation": ("Intensity Median absolute deviation", "intensity"),
    "coefficientofvariation": ("Intensity Coefficient of variation", "intensity"),
    "quantilecoefficientofdispersion": (
        "Intensity Quartile coefficient of dispersion",
        "intensity",
    ),
    "energy": ("Intensity energy", "intensity"),
    "rootmeansquare": ("Root mean square intensity", "intensity"),
    "uniformity": ("Discretised intensity uniformity", "histogram"),
    "entropy": ("Discretised intensity entropy", "histogram"),
    "meanvalue": ("Mean discretised intensity", "histogram"),
    "variancevalue": ("Discretised intensity variance", "histogram"),
    "skewnessvalue": ("Discretised intensity skewness", "histogram"),
    "excesskurtosisvalue": ("Discretised intensity kurtosis", "histogram"),
    "excesskurtosisindex": ("Discretised intensity kurtosis", "histogram"),
    "medianvalue": ("Median discretised intensity", "histogram"),
    "minimumvalue": ("Minimum discretised intensity", "histogram"),
    "percentile10value": ("10th discretised intensity percentile", "histogram"),
    "percentile90value": ("90th discretised intensity percentile", "histogram"),
    "maximumvalue": ("Maximum discretised intensity", "histogram"),
    "modevalue": ("Intensity histogram mode", "histogram"),
    "interquantilerangevalue": (
        "Discretised intensity interquartile range",
        "histogram",
    ),
    "rangevalue": ("Discretised intensity range", "histogram"),
    "meanabsolutedeviationvalue": (
        "Intensity histogram mean absolute deviation",
        "histogram",
    ),
    "robustmeanabsolutedeviationvalue": (
        "Intensity histogram robust mean absolute deviation",
        "histogram",
    ),
    "medianabsolutedeviationvalue": (
        "Intensity histogram median absolute deviation",
        "histogram",
    ),
    "coefficientofvariationvalue": (
        "Intensity histogram coefficient of variation",
        "histogram",
    ),
    "quantilecoefficientofdispersionvalue": (
        "Intensity histogram quartile coefficient of dispersion",
        "histogram",
    ),
    "entropyvalue": ("Discretised intensity entropy", "histogram"),
    "uniformityvalue": ("Discretised intensity uniformity", "histogram"),
    "maximumgradient": ("Maximum histogram gradient", "histogram"),
    "minimumgradient": ("Minimum histogram gradient", "histogram"),
    "maximumgradientindex": ("Maximum histogram gradient intensity", "histogram"),
    "minimumgradientindex": ("Minimum histogram gradient intensity", "histogram"),
    "10thpercentile": ("10th intensity percentile", "intensity"),
    "90thpercentile": ("90th intensity percentile", "intensity"),
    "interquartilerange": ("Intensity interquartile range", "intensity"),
    "localintensitypeak": ("Local intensity peak ", "local_intensity"),
    "globalintensitypeak": ("Global intensity peak", "local_intensity"),
    # morphology
    "voxelvolume": ("Volume voxel counting", "morphology"),
    "volumemeshbased": ("Volume", "morphology"),
    "volumevoxelbased": ("Volume voxel counting", "morphology"),
    "surfacemeshbased": ("Surface area", "morphology"),
    "surfacevoxelbased": ("Surface area", "morphology"),
    "surfacetovolumeratiomeshbased": ("Surface to volume ratio", "morphology"),
    "surfacetovolumeratiovoxelbased": ("Surface to volume ratio", "morphology"),
    "sphericitymeshbased": ("Sphericity", "morphology"),
    "sphericityvoxelbased": ("Sphericity", "morphology"),
    "asphericitymeshbased": ("Asphericity", "morphology"),
    "asphericityvoxelbased": ("Asphericity", "morphology"),
    "compactness1meshbased": ("Compactness 1", "morphology"),
    "compactness1voxelbased": ("Compactness 1", "morphology"),
    "compactness2meshbased": ("Compactness 2", "morphology"),
    "compactness2voxelbased": ("Compactness 2", "morphology"),
    "sphericaldisproportionmeshbased": ("Spherical disproportion", "morphology"),
    "sphericaldisproportionvoxelbased": ("Spherical disproportion", "morphology"),
    "centremassshift": ("Center of mass shift", "morphology"),
    "centreofmassshift": ("Center of mass shift", "morphology"),
    "maximum3ddiameter": ("Maximum 3D diameter", "morphology"),
    "pcamajoraxislength": ("Major axis length", "morphology"),
    "pcaminoraxislength": ("Minor axis length", "morphology"),
    "pcaleastaxislength": ("Least axis length", "morphology"),
    "pcaelongation": ("Elongation", "morphology"),
    "pcaflatness": ("Flatness", "morphology"),
    "volumeintegratedintensity": ("Integrated intensity", "morphology"),
    "volumemoransiindex": ("Moran's I index", "morphology"),
    "volumegearyscmeasure": ("Geary's C measure", "morphology"),
    "volumedensityaxisalignedboundingbox": ("Volume density (AABB)", "morphology"),
    "surfacedensityaxisalignedboundingbox": ("Area density (AABB)", "morphology"),
    "volumedensityorientedminimumboundingbox": ("Volume density (OMBB)", "morphology"),
    "surfacedensityorientedminimumboundingbox": ("Area density (OMBB)", "morphology"),
    "volumedensityapproxenclosingellipsoid": ("Volume density (AEE)", "morphology"),
    "surfacedensityapproxenclosingellipsoid": ("Area density (AEE)", "morphology"),
    "volumedensityapproxminimumvolumeenclosingellipsoid": (
        "Volume density (MVEE)",
        "morphology",
    ),
    "surfacedensityapproxminimumvolumeenclosingellipsoid": (
        "Area density (MVEE)",
        "morphology",
    ),
    "volumedensityconvexhull": ("Volume density (convex hull)", "morphology"),
    "surfacedensityconvexhull": ("Area density (convex hull)", "morphology"),
    # glcm
    "jointaverage": ("Joint average", "glcm"),
    "jointentropy": ("Joint entropy", "glcm"),
    "jointmax": ("Joint maximum", "glcm"),
    "jointmaximum": ("Joint maximum", "glcm"),
    "maximumprobability": ("Joint maximum", "glcm"),
    "jointvariance": ("Joint variance", "glcm"),
    "differenceaverage": ("Difference average", "glcm"),
    "differencevariance": ("Difference variance", "glcm"),
    "differenceentropy": ("Difference entropy", "glcm"),
    "sumaverage": ("Sum average", "glcm"),
    "sumvariance": ("Sum variance", "glcm"),
    "sumentropy": ("Sum entropy", "glcm"),
    "angularsecondmoment": ("Angular second moment", "glcm"),
    "contrast": ("Contrast", "glcm"),
    "dissimilarity": ("Dissimilarity", "glcm"),
    "inversedifference": ("Inverse difference", "glcm"),
    "inversedifferencenormalized": ("Normalised inverse difference", "glcm"),
    "inversedifferencemoment": ("Inverse difference moment", "glcm"),
    "inversedifferencemomentnormalized": (
        "Normalised inverse difference moment",
        "glcm",
    ),
    "inversevariance": ("Inverse variance", "glcm"),
    "correlation": ("Correlation", "glcm"),
    "autocorrelation": ("Autocorrelation", "glcm"),
    "clustertendency": ("Cluster tendency", "glcm"),
    "clustershade": ("Cluster shade", "glcm"),
    "clusterprominence": ("Cluster prominence", "glcm"),
    "firstmeasureofinformationcorrelation": ("Information correlation 1", "glcm"),
    "secondmeasureofinformationcorrelation": ("Information correlation 2", "glcm"),
    "haralickcorrelation": ("Correlation", "glcm"),
    "inertia": ("Contrast", "glcm"),
    "homogeneity1": ("Inverse difference", "glcm"),
    # glrlm
    "shortrunemphasis": ("Short runs emphasis", "glrlm"),
    "longrunemphasis": ("Long runs emphasis", "glrlm"),
    "greylevelnonuniformity": ("Grey level non-uniformity", "glrlm"),
    "greylevelnonuniformitynormalized": (
        "Normalised grey level non-uniformity",
        "glrlm",
    ),
    "runlengthnonuniformity": ("Run length non-uniformity", "glrlm"),
    "runlengthnonuniformitynormalized": (
        "Normalised run length non-uniformity",
        "glrlm",
    ),
    "lowgreylevelrunemphasis": ("Low grey level run emphasis", "glrlm"),
    "highgreylevelrunemphasis": ("High grey level run emphasis", "glrlm"),
    "shortrunlowgreylevelemphasis": ("Short run low grey level emphasis", "glrlm"),
    "shortrunhighgreylevelemphasis": ("Short run high grey level emphasis", "glrlm"),
    "longrunlowgreylevelemphasis": ("Long run low grey level emphasis", "glrlm"),
    "longrunhighgreylevelemphasis": ("Long run high grey level emphasis", "glrlm"),
    "runpercentage": ("Run percentage", "glrlm"),
    "greylevelvariance": ("Grey level variance", "glrlm"),
    "runlengthvariance": ("Run length variance", "glrlm"),
    "runlengthentropy": ("Run entropy", "glrlm"),
    # glszm
    "smallzoneemphasis": ("Small zone emphasis", "glszm"),
    "largezoneemphasis": ("Large zone emphasis", "glszm"),
    "lowgreylevelemphasis": ("Low grey level zone emphasis", "glszm"),
    "highgreylevelemphasis": ("High grey level zone emphasis", "glszm"),
    "smallzonelowgreylevelemphasis": ("Small zone low grey level emphasis", "glszm"),
    "smallzonehighgreylevelemphasis": ("Small zone high grey level emphasis", "glszm"),
    "largezonelowgreylevelemphasis": ("Large zone low grey level emphasis", "glszm"),
    "largezonehighgreylevelemphasis": ("Large zone high grey level emphasis", "glszm"),
    "greylevelnonuniformitynormalizedszm": (
        "Normalised grey level non-uniformity",
        "glszm",
    ),
    "greylevelnonuniformityszm": ("Grey level non-uniformity", "glszm"),
    "zonesizenonuniformity": ("Zone size non-uniformity", "glszm"),
    "zonesizenonuniformitynormalized": ("Normalised zone size non-uniformity", "glszm"),
    "zonepercentage": ("Zone percentage", "glszm"),
    "zonesizevariance": ("Zone size variance", "glszm"),
    "zonesizeentropy": ("Zone size entropy", "glszm"),
    # gldzm
    "smalldistanceemphasis": ("Small distance emphasis", "gldzm"),
    "largedistanceemphasis": ("Large distance emphasis", "gldzm"),
    "smalldistancelowgreylevelemphasis": (
        "Small distance low grey level emphasis",
        "gldzm",
    ),
    "smalldistancehighgreylevelemphasis": (
        "Small distance high grey level emphasis",
        "gldzm",
    ),
    "largedistancelowgreylevelemphasis": (
        "Large distance low grey level emphasis",
        "gldzm",
    ),
    "largedistancehighgreylevelemphasis": (
        "Large distance high grey level emphasis",
        "gldzm",
    ),
    "distancesizenonuniformity": ("Zone distance non-uniformity", "gldzm"),
    "distancesizenonuniformitynormalized": (
        "Normalised zone distance non-uniformity",
        "gldzm",
    ),
    "zonedistancevariance": ("Zone distance variance", "gldzm"),
    "zonedistanceentropy": ("Zone distance entropy", "gldzm"),
    # ngtdm
    "coarsness": ("Coarseness", "ngtdm"),
    "coarseness": ("Coarseness", "ngtdm"),
    "busyness": ("Busyness", "ngtdm"),
    "complexity": ("Complexity", "ngtdm"),
    "strength": ("Strength", "ngtdm"),
    # ngldm
    "lowdependenceemphasis": ("Low dependence emphasis", "ngldm"),
    "highdependenceemphasis": ("High dependence emphasis", "ngldm"),
    "lowgreylevelcountemphasis": ("Low grey level count emphasis", "ngldm"),
    "highgreylevelcountemphasis": ("High grey level count emphasis", "ngldm"),
    "lowdependencelowgreylevelemphasis": (
        "Low dependence low grey level emphasis",
        "ngldm",
    ),
    "lowdependencehighgreylevelemphasis": (
        "Low dependence high grey level emphasis",
        "ngldm",
    ),
    "highdependencelowgreylevelemphasis": (
        "High dependence low grey level emphasis",
        "ngldm",
    ),
    "highdependencehighgreylevelemphasis": (
        "High dependence high grey level emphasis",
        "ngldm",
    ),
    "greylevelnonuniformitynormalised": (
        "Normalised grey level non-uniformity",
        "ngldm",
    ),
    "dependencecountnonuniformity": ("Dependence count non-uniformity", "ngldm"),
    "dependencecountnonuniformitynormalised": (
        "Normalised dependence count non-uniformity",
        "ngldm",
    ),
    "dependencecountpercentage": ("Dependence count percentage", "ngldm"),
    "dependencecountvariance": ("Dependence count variance", "ngldm"),
    "dependencecountentropy": ("Dependence count entropy", "ngldm"),
    "dependencecountenergy": ("Dependence count energy", "ngldm"),
    # ivh
    "volumefractionat010intensity": ("Volume at intensity fraction 0.10", "ivh"),
    "volumefractionat090intensity": ("Volume at intensity fraction 0.90", "ivh"),
    "intensityat010volume": ("Intensity at volume fraction 0.10", "ivh"),
    "intensityat090volume": ("Intensity at volume fraction 0.90", "ivh"),
    "differencevolumefractionat010and090intensity": (
        "Volume fraction difference between intensity 0.10 and 0.90 fractions",
        "ivh",
    ),
    "differenceintensityat010and090volume": (
        "Intensity fraction difference between volume 0.10 and 0.90 fractions",
        "ivh",
    ),
    "areaunderivhcurve": ("Area under the IVH curve", "ivh"),
}


def map_mitk(feature_name: str) -> Optional[str]:
    lower_full = feature_name.strip().lower()
    # Disambiguate duplicated "Voxel Volume" labels by context.
    if lower_full.startswith("volumetric features::voxel volume"):
        # This is per-voxel physical size, not ROI volume.
        return None
    if lower_full.startswith("first order::voxel volume"):
        return None

    token = feature_name.strip()
    context_family = _mitk_context_family(feature_name)
    # Most MITK labels are encoded at the end of feature IDs.
    for sep in ("::", ":", "|", ";"):
        if sep in token:
            token = token.split(sep)[-1]
    token = token.replace("co-occ.", "")
    token = token.replace("co-occ", "")
    token = token.replace("(", " ").replace(")", " ")
    token = re.sub(r"^\d+\s*[\.\)]\s*", "", token).strip()
    token = re.sub(r"\s+", " ", token).strip()
    # Texture outputs often carry aggregation suffixes (e.g., "Contrast Means").
    # Strip only trailing qualifiers so single-term first-order names remain intact.
    if " " in token:
        token = re.sub(
            r"\b(Means|Mean|Std\.?|Comb\.?)$", "", token, flags=re.IGNORECASE
        ).strip()
    norm = _normalize(token)

    if norm in MITK_TOKEN_MAP:
        ibsi_name, family = MITK_TOKEN_MAP[norm]
        if context_family:
            code = _code_from_name_in_family(ibsi_name, context_family)
            if code:
                return code
        return _code_from_name_in_family(ibsi_name, family)

    if norm.startswith("overall"):
        overall_norm = norm[len("overall") :]
        if overall_norm in MITK_TOKEN_MAP:
            ibsi_name, family = MITK_TOKEN_MAP[overall_norm]
            if context_family:
                code = _code_from_name_in_family(ibsi_name, context_family)
                if code:
                    return code
            return _code_from_name_in_family(ibsi_name, family)

    # Handle ambiguous "Grey level non-uniformity" feature labels by family hints.
    if norm == "greylevelnonuniformity":
        lower = feature_name.lower()
        if "distance zone" in lower:
            return _code_from_name_in_family("Grey level non-uniformity", "gldzm")
        if "run" in lower:
            return _code_from_name_in_family("Grey level non-uniformity", "glrlm")
        if "zone" in lower:
            return _code_from_name_in_family("Grey level non-uniformity", "glszm")
        if "dependence" in lower:
            return _code_from_name_in_family("Grey level non-uniformity", "ngldm")

    if context_family:
        code = _code_from_name_in_family(token, context_family)
        if code:
            return code
    return _code_from_name(token)


MODDICOM_ALIASES = {
    "stat_10thpercentile": "stat_p10",
    "stat_90thpercentile": "stat_p90",
    "ih_10thpercentile": "ih_p10",
    "ih_90thpercentile": "ih_p90",
    "stat_nic_entropy": "ih_entropy",
    "stat_nic_kurt": "ih_kurt",
    "stat_nic_skew": "ih_skew",
    "stat_nic_uniformity": "ih_uniformity",
    "ih_nic_entropy": "ih_entropy",
    "ih_nic_kurt": "ih_kurt",
    "ih_nic_skew": "ih_skew",
    "ih_nic_uniformity": "ih_uniformity",
    "cm_auto_corr": "cm_auto_corr",
    "cm_clust_tend": "cm_clust_tend",
    "cm_clust_shade": "cm_clust_shade",
    "cm_clust_prom": "cm_clust_prom",
    "cm_info_corr_1": "cm_info_corr1",
    "cm_info_corr_2": "cm_info_corr2",
    "rlm_r_perc": "rlm_r_perc",
    "szm_z_entr": "szm_zs_entr",
    "zsm_z_perc": "szm_z_perc",
    "morph_surface": "morph_area_mesh",
    "l_major": "morph_pca_maj_axis",
    "l_minor": "morph_pca_min_axis",
    "l_least": "morph_pca_least_axis",
}


def map_moddicom(feature_name: str) -> Optional[str]:
    token = feature_name.strip()
    if token.startswith("F_"):
        token = token[2:]
    token = token.replace(".", "_")
    token = token.replace("__", "_")
    token = token.lower().strip("_")
    token = re.sub(r"^(cm|rlm|szm|zsm)_2_5dmerged_", r"\1_", token)
    token = re.sub(r"^(cm|rlm|szm|zsm)_merged_", r"\1_", token)
    token = re.sub(r"^(cm|rlm|szm|zsm)_2_5d_", r"\1_", token)
    token = re.sub(r"^(cm|rlm|szm|zsm)_25d_", r"\1_", token)
    token = MODDICOM_ALIASES.get(token, token)
    return map_mirp(token)


CERR_SHAPE_MAP: Dict[str, tuple[str, str]] = {
    "majoraxis": ("Major axis length", "morphology"),
    "minoraxis": ("Minor axis length", "morphology"),
    "leastaxis": ("Least axis length", "morphology"),
    "flatness": ("Flatness", "morphology"),
    "elongation": ("Elongation", "morphology"),
    "max3ddiameter": ("Maximum 3D diameter", "morphology"),
    "surfarea": ("Surface area", "morphology"),
    "volume": ("Volume", "morphology"),
    "compactness1": ("Compactness 1", "morphology"),
    "compactness2": ("Compactness 2", "morphology"),
    "spherdisprop": ("Spherical disproportion", "morphology"),
    "sphericity": ("Sphericity", "morphology"),
    "surftovolratio": ("Surface to volume ratio", "morphology"),
}

CERR_FIRSTORDER_MAP: Dict[str, tuple[str, str]] = {
    "min": ("Minimum intensity", "intensity"),
    "max": ("Maximum intensity", "intensity"),
    "mean": ("Mean intensity", "intensity"),
    "range": ("Intensity range", "intensity"),
    "var": ("Intensity variance", "intensity"),
    "median": ("Median intensity", "intensity"),
    "skewness": ("Intensity skewness", "intensity"),
    "kurtosis": ("Intensity kurtosis", "intensity"),
    "rms": ("Root mean square intensity", "intensity"),
    "energy": ("Intensity energy", "intensity"),
    "meanabsdev": ("Intensity Mean absolute deviation", "intensity"),
    "medianabsdev": ("Intensity Median absolute deviation", "intensity"),
    "p10": ("10th intensity percentile", "intensity"),
    "p90": ("90th intensity percentile", "intensity"),
    "robustmeanabsdev": ("Intensity Robust mean absolute deviation", "intensity"),
    "interquartilerange": ("Intensity interquartile range", "intensity"),
    "coeffdispersion": ("Intensity Quartile coefficient of dispersion", "intensity"),
    "coeffvariation": ("Intensity Coefficient of variation", "intensity"),
}

CERR_GLCM_MAP: Dict[str, tuple[str, str]] = {
    "energy": ("Angular second moment", "glcm"),
    "jointentropy": ("Joint entropy", "glcm"),
    "jointmax": ("Joint maximum", "glcm"),
    "jointavg": ("Joint average", "glcm"),
    "jointvar": ("Joint variance", "glcm"),
    "contrast": ("Contrast", "glcm"),
    "invdiffmom": ("Inverse difference moment", "glcm"),
    "invdiffmomnorm": ("Normalised inverse difference moment", "glcm"),
    "invdiff": ("Inverse difference", "glcm"),
    "invdiffnorm": ("Normalised inverse difference", "glcm"),
    "invvar": ("Inverse variance", "glcm"),
    "dissimilarity": ("Dissimilarity", "glcm"),
    "diffentropy": ("Difference entropy", "glcm"),
    "diffvar": ("Difference variance", "glcm"),
    "diffavg": ("Difference average", "glcm"),
    "sumavg": ("Sum average", "glcm"),
    "sumvar": ("Sum variance", "glcm"),
    "sumentropy": ("Sum entropy", "glcm"),
    "corr": ("Correlation", "glcm"),
    "haralickcorr": ("Correlation", "glcm"),
    "clusttendency": ("Cluster tendency", "glcm"),
    "clustshade": ("Cluster shade", "glcm"),
    "clustpromin": ("Cluster prominence", "glcm"),
    "autocorr": ("Autocorrelation", "glcm"),
    "firstinfcorr": ("Information correlation 1", "glcm"),
    "secondinfcorr": ("Information correlation 2", "glcm"),
}

CERR_GLRLM_MAP: Dict[str, tuple[str, str]] = {
    "graylevelnonuniformity": ("Grey level non-uniformity", "glrlm"),
    "graylevelnonuniformitynorm": ("Normalised grey level non-uniformity", "glrlm"),
    "graylevelvariance": ("Grey level variance", "glrlm"),
    "highgraylevelrunemphasis": ("High grey level run emphasis", "glrlm"),
    "longrunemphasis": ("Long runs emphasis", "glrlm"),
    "longrunhighgraylevelemphasis": ("Long run high grey level emphasis", "glrlm"),
    "longrunlowgraylevelemphasis": ("Long run low grey level emphasis", "glrlm"),
    "lowgraylevelrunemphasis": ("Low grey level run emphasis", "glrlm"),
    "runentropy": ("Run entropy", "glrlm"),
    "runlengthnonuniformity": ("Run length non-uniformity", "glrlm"),
    "runlengthnonuniformitynorm": ("Normalised run length non-uniformity", "glrlm"),
    "runlengthvariance": ("Run length variance", "glrlm"),
    "runpercentage": ("Run percentage", "glrlm"),
    "shortrunemphasis": ("Short runs emphasis", "glrlm"),
    "shortrunhighgraylevelemphasis": ("Short run high grey level emphasis", "glrlm"),
    "shortrunlowgraylevelemphasis": ("Short run low grey level emphasis", "glrlm"),
}

CERR_GLSZM_MAP: Dict[str, tuple[str, str]] = {
    "smallareaemphasis": ("Small zone emphasis", "glszm"),
    "largeareaemphasis": ("Large zone emphasis", "glszm"),
    "graylevelnonuniformity": ("Grey level non-uniformity", "glszm"),
    "graylevelnonuniformitynorm": ("Normalised grey level non-uniformity", "glszm"),
    "sizezonenonuniformity": ("Zone size non-uniformity", "glszm"),
    "sizezonenonuniformitynorm": ("Normalised zone size non-uniformity", "glszm"),
    "zonepercentage": ("Zone percentage", "glszm"),
    "lowgraylevelzoneemphasis": ("Low grey level zone emphasis", "glszm"),
    "highgraylevelzoneemphasis": ("High grey level zone emphasis", "glszm"),
    "smallarealowgraylevelemphasis": ("Small zone low grey level emphasis", "glszm"),
    "smallareahighgraylevelemphasis": ("Small zone high grey level emphasis", "glszm"),
    "largearealowgraylevelemphasis": ("Large zone low grey level emphasis", "glszm"),
    "largeareahighgraylevelemphasis": ("Large zone high grey level emphasis", "glszm"),
    "graylevelvariance": ("Grey level variance", "glszm"),
    "sizezonevariance": ("Zone size variance", "glszm"),
    "zoneentropy": ("Zone size entropy", "glszm"),
}

CERR_NGTDM_MAP: Dict[str, tuple[str, str]] = {
    "coarseness": ("Coarseness", "ngtdm"),
    "contrast": ("Contrast", "ngtdm"),
    "busyness": ("Busyness", "ngtdm"),
    "complexity": ("Complexity", "ngtdm"),
    "strength": ("Strength", "ngtdm"),
}

CERR_NGLDM_MAP: Dict[str, tuple[str, str]] = {
    "lowdependenceemphasis": ("Low dependence emphasis", "ngldm"),
    "highdependenceemphasis": ("High dependence emphasis", "ngldm"),
    "lowgraylevelcountemphasis": ("Low grey level count emphasis", "ngldm"),
    "highgraylevelcountemphasis": ("High grey level count emphasis", "ngldm"),
    "lowdependencelowgraylevelemphasis": (
        "Low dependence low grey level emphasis",
        "ngldm",
    ),
    "lowdependencehighgraylevelemphasis": (
        "Low dependence high grey level emphasis",
        "ngldm",
    ),
    "highdependencelowgraylevelemphasis": (
        "High dependence low grey level emphasis",
        "ngldm",
    ),
    "highdependencehighgraylevelemphasis": (
        "High dependence high grey level emphasis",
        "ngldm",
    ),
    "graylevelnonuniformity": ("Grey level non-uniformity", "ngldm"),
    "graylevelnonuniformitynorm": ("Normalised grey level non-uniformity", "ngldm"),
    "dependencecountnonuniformity": ("Dependence count non-uniformity", "ngldm"),
    "dependencecountnonuniformitynorm": (
        "Normalised dependence count non-uniformity",
        "ngldm",
    ),
    "dependencecountpercentage": ("Dependence count percentage", "ngldm"),
    "graylevelvariance": ("Grey level variance", "ngldm"),
    "dependencecountvariance": ("Dependence count variance", "ngldm"),
    "entropy": ("Dependence count entropy", "ngldm"),
    "energy": ("Dependence count energy", "ngldm"),
}

CERR_IVH_MAP: Dict[str, tuple[str, str]] = {
    "ix10": ("Intensity at volume fraction 0.10", "ivh"),
    "ix90": ("Intensity at volume fraction 0.90", "ivh"),
    "vx10": ("Volume at intensity fraction 0.10", "ivh"),
    "vx90": ("Volume at intensity fraction 0.90", "ivh"),
}

CERR_LOCAL_INTENSITY_MAP: Dict[str, tuple[str, str]] = {
    "peak": ("Local intensity peak ", "local_intensity"),
}

CERR_GLCM_FEATURELIST_TOKENS: Dict[str, str] = {
    "energy": "energy",
    "jointentropy": "jointEntropy",
    "jointmax": "jointMax",
    "jointavg": "jointAvg",
    "jointvar": "jointVar",
    "contrast": "contrast",
    "invdiffmom": "invDiffMoment",
    "invdiffmomnorm": "invDiffMomNorm",
    "invdiff": "invDiff",
    "invdiffnorm": "invDiffNorm",
    "invvar": "invVar",
    "dissimilarity": "dissimilarity",
    "diffentropy": "diffEntropy",
    "diffvar": "diffVar",
    "diffavg": "diffAvg",
    "sumavg": "sumAvg",
    "sumvar": "sumVar",
    "sumentropy": "sumEntropy",
    "corr": "corr",
    "clusttendency": "clustTendency",
    "clustshade": "clustShade",
    "clustpromin": "clustProm",
    "autocorr": "autoCorr",
    "firstinfcorr": "firstInfCorr",
    "secondinfcorr": "secondInfCorr",
}

CERR_GLSZM_FEATURELIST_TOKENS: Dict[str, str] = {
    "graylevelnonuniformity": "grayLevelNonUniformity",
    "graylevelnonuniformitynorm": "grayLevelNonUniformityNorm",
    "graylevelvariance": "grayLevelVariance",
    "highgraylevelzoneemphasis": "highGrayLevelZoneEmphasis",
    "lowgraylevelzoneemphasis": "lowGrayLevelZoneEmphasis",
    "largeareaemphasis": "largeAreaEmphasis",
    "largeareahighgraylevelemphasis": "largeAreaHighGrayLevelEmphasis",
    "largearealowgraylevelemphasis": "largeAreaLowGrayLevelEmphasis",
    "sizezonenonuniformity": "sizeZoneNonUniformity",
    "sizezonenonuniformitynorm": "sizeZoneNonUniformityNorm",
    "sizezonevariance": "sizeZoneVariance",
    "zonepercentage": "zonePercentage",
    "smallareaemphasis": "smallAreaEmphasis",
    "smallarealowgraylevelemphasis": "smallAreaLowGrayLevelEmphasis",
    "smallareahighgraylevelemphasis": "smallAreaHighGrayLevelEmphasis",
    "zoneentropy": "zoneEntropy",
}


def _build_cerr_code_to_feature_tokens(
    *,
    mapping: Dict[str, tuple[str, str]],
    token_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    code_to_tokens: Dict[str, List[str]] = {}
    for key, (ibsi_name, family) in mapping.items():
        code = _code_from_name_in_family(ibsi_name, family)
        if not code:
            continue
        token = key
        if token_overrides and key in token_overrides:
            token = token_overrides[key]
        code_to_tokens.setdefault(code, [])
        if token not in code_to_tokens[code]:
            code_to_tokens[code].append(token)
    for code in code_to_tokens:
        code_to_tokens[code].sort()
    return code_to_tokens


CERR_GLCM_CODE_TO_TOKENS = _build_cerr_code_to_feature_tokens(
    mapping=CERR_GLCM_MAP,
    token_overrides=CERR_GLCM_FEATURELIST_TOKENS,
)
CERR_GLRLM_CODE_TO_TOKENS = _build_cerr_code_to_feature_tokens(mapping=CERR_GLRLM_MAP)
CERR_GLSZM_CODE_TO_TOKENS = _build_cerr_code_to_feature_tokens(
    mapping=CERR_GLSZM_MAP,
    token_overrides=CERR_GLSZM_FEATURELIST_TOKENS,
)

# CERR supports reliable per-feature restriction for GLCM/GLRLM.
# GLSZM featureList handling in CERR currently lowercases user tokens but not flag names,
# which disables all GLSZM flags when a subset is passed.
CERR_FEATURE_LEVEL_CLASSES = {"glcm", "glrlm"}
IBSI_FAMILY_TO_CERR_CLASS = {
    "morphology": "shape",
    "local_intensity": "peakValley",
    "intensity": "firstOrder",
    "histogram": "firstOrder",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    # In current CERR code, NGTDM extraction is gated on the GLDM flag.
    "ngtdm": "gldm",
    "ngldm": "gldm",
}


def cerr_feature_selection(codes: Iterable[str]) -> Dict[str, List[str]]:
    """
    Build CERR featureClass selections from IBSI codes.

    Returns: {class_name: [featureList tokens]}.
    Classes without stable feature-level controls are set to ["all"].
    """
    classes: Dict[str, set[str]] = {}
    normalized_codes = {str(c).strip() for c in codes if str(c).strip()}

    for code in normalized_codes:
        family = CODE_TO_FAMILY.get(code)
        if not family:
            continue
        class_name = IBSI_FAMILY_TO_CERR_CLASS.get(family)
        if not class_name:
            continue
        classes.setdefault(class_name, set())

        if class_name == "glcm":
            for token in CERR_GLCM_CODE_TO_TOKENS.get(code, []):
                classes[class_name].add(token)
        elif class_name == "glrlm":
            for token in CERR_GLRLM_CODE_TO_TOKENS.get(code, []):
                classes[class_name].add(token)
        elif class_name == "glszm":
            for token in CERR_GLSZM_CODE_TO_TOKENS.get(code, []):
                classes[class_name].add(token)

    selection: Dict[str, List[str]] = {}
    for class_name, tokens in classes.items():
        if class_name in CERR_FEATURE_LEVEL_CLASSES and tokens:
            selection[class_name] = sorted(tokens)
        else:
            selection[class_name] = ["all"]

    return selection


CERR_FIRSTORDER_EXCLUDE = {
    "totalenergy",
    "robustmedianabsdev",
    "std",
    "entropy",
}


def _map_cerr_token(token: str, mapping: Dict[str, tuple[str, str]]) -> Optional[str]:
    key = _normalize(token)
    match = mapping.get(key)
    if not match:
        return None
    ibsi_name, family = match
    return _code_from_name_in_family(ibsi_name, family)


def map_cerr(feature_name: str) -> Optional[str]:
    if feature_name.startswith("shapeS_"):
        token = feature_name[len("shapeS_") :]
        return _map_cerr_token(token, CERR_SHAPE_MAP)

    if feature_name.startswith("Original_firstOrderS_"):
        token = feature_name[len("Original_firstOrderS_") :]
        if _normalize(token) in CERR_FIRSTORDER_EXCLUDE:
            return None
        return _map_cerr_token(token, CERR_FIRSTORDER_MAP)

    if feature_name.startswith("Original_glcmFeatS_AvgS_"):
        token = feature_name[len("Original_glcmFeatS_AvgS_") :]
        return _map_cerr_token(token, CERR_GLCM_MAP)
    if feature_name.startswith("Original_glcmFeatS_CombS_"):
        token = feature_name[len("Original_glcmFeatS_CombS_") :]
        return _map_cerr_token(token, CERR_GLCM_MAP)

    if feature_name.startswith("Original_rlmFeatS_AvgS_"):
        token = feature_name[len("Original_rlmFeatS_AvgS_") :]
        return _map_cerr_token(token, CERR_GLRLM_MAP)
    if feature_name.startswith("Original_rlmFeatS_CombS_"):
        token = feature_name[len("Original_rlmFeatS_CombS_") :]
        return _map_cerr_token(token, CERR_GLRLM_MAP)

    if feature_name.startswith("Original_szmFeatS_"):
        token = feature_name[len("Original_szmFeatS_") :]
        return _map_cerr_token(token, CERR_GLSZM_MAP)

    if feature_name.startswith("Original_ngtdmFeatS_"):
        token = feature_name[len("Original_ngtdmFeatS_") :]
        return _map_cerr_token(token, CERR_NGTDM_MAP)

    if feature_name.startswith("Original_ngldmFeatS_"):
        token = feature_name[len("Original_ngldmFeatS_") :]
        return _map_cerr_token(token, CERR_NGLDM_MAP)

    if feature_name.startswith("Original_ivhFeaturesS_"):
        token = feature_name[len("Original_ivhFeaturesS_") :]
        return _map_cerr_token(token, CERR_IVH_MAP)

    if feature_name.startswith("Original_peakValleyFeatureS_"):
        token = feature_name[len("Original_peakValleyFeatureS_") :]
        return _map_cerr_token(token, CERR_LOCAL_INTENSITY_MAP)

    return None


PYRADIOMICS_EXCLUDE = {
    "original_firstorder_TotalEnergy",  # not in IBSI glossary
    "original_glcm_MCC",  # maximal correlation coefficient not in IBSI
    "original_shape_Maximum2DDiameterColumn",
    "original_shape_Maximum2DDiameterRow",
    "original_shape_Maximum2DDiameterSlice",
}

MIRP_IVH_EXCLUDE = {
    "ivh_v25",
    "ivh_v50",
    "ivh_v75",
    "ivh_i25",
    "ivh_i50",
    "ivh_i75",
    "ivh_diff_v25_v75",
    "ivh_diff_i25_i75",
}

CERR_SHAPE_EXCLUDE = {
    "shapeS_filledVolume",
    "shapeS_max2dDiameterAxialPlane",
    "shapeS_max2dDiameterSagittalPlane",
    "shapeS_max2dDiameterCoronalPlane",
}

CERR_GLCM_EXCLUDE_PREFIXES = (
    "Original_glcmFeatS_MaxS_",
    "Original_glcmFeatS_MinS_",
    "Original_glcmFeatS_StdS_",
    "Original_glcmFeatS_MadS_",
)
CERR_GLCM_EXCLUDE = {
    "Original_glcmFeatS_AvgS_haralickCorr",
    "Original_glcmFeatS_CombS_diffAvg",
}

CERR_GLRLM_EXCLUDE_PREFIXES = (
    "Original_rlmFeatS_MaxS_",
    "Original_rlmFeatS_MinS_",
    "Original_rlmFeatS_StdS_",
    "Original_rlmFeatS_MadS_",
)

CERR_OTHER_EXCLUDE_PREFIXES = (
    "Original_peakValleyFeatureS_peak",
    "Original_peakValleyFeatureS_valley",
)

CERR_IVH_EXCLUDE = {
    "Original_ivhFeaturesS_meanHist",
    "Original_ivhFeaturesS_maxHist",
    "Original_ivhFeaturesS_minHist",
    "Original_ivhFeaturesS_I50",
    "Original_ivhFeaturesS_rangeHist",
    "Original_ivhFeaturesS_MOHx10",
    "Original_ivhFeaturesS_MOCx10",
    "Original_ivhFeaturesS_MOHx90",
    "Original_ivhFeaturesS_MOCx90",
}

RADIOMICSJ_EXCLUDE_PREFIXES = (
    "OperationalInfo_",
    "Diagnostics_",
    "Fractal_",
)

RADIOMICSJ_EXCLUDE = {
    "IntensityBasedStatistical_TotalEnergy",
    "IntensityBasedStatistical_StandardDeviation",
    "IntensityBasedStatistical_StandardError",
}

CAPTK_GLCM_EXCLUDE = set()
CAPTK_GLRLM_EXCLUDE = {
    "totalruns",
}
CAPTK_HISTOGRAM_EXCLUDE = {
    "energy",
    "rootmeansquare",
    "twentyfifthpercentile",
    "seventyfifthpercentile",
}
CAPTK_EXCLUDE_TOKENS = {
    "sum",
    "standarddeviation",
    "fifthpercentile",
    "ninetyfifthpercentile",
    "fifthpercentilemean",
    "ninetyfifthpercentilemean",
    "area",
    "equivalentsphericalradius",
    "equivalentsphericalperimeter",
    "perimeter",
    "perimeteronborder",
    "perimeteronborderratio",
    "pixelsonborder",
    "roundness",
    "largestcomponentsize",
    "eccentricity",
    "ellipsediameter",
    "orientedboundingboxsize",
    "greylevelmean",
    "zonesizemean",
}


def _mirp_excluded(feature_name: str) -> bool:
    base = re.sub(r"_fbn_n\\d+", "", feature_name)
    base = re.sub(r"_fbs_w[0-9.]+", "", base)
    if base.startswith("image_") or base == "sample_name":
        return True
    if base in MIRP_IVH_EXCLUDE:
        return True
    return False


def _cerr_excluded(feature_name: str) -> bool:
    if feature_name in CERR_SHAPE_EXCLUDE:
        return True
    if feature_name in CERR_IVH_EXCLUDE:
        return True
    if feature_name in CERR_GLCM_EXCLUDE:
        return True
    if feature_name in CERR_OTHER_EXCLUDE_PREFIXES:
        return True
    if feature_name.startswith(CERR_GLCM_EXCLUDE_PREFIXES):
        return True
    if feature_name.startswith(CERR_GLRLM_EXCLUDE_PREFIXES):
        return True
    if feature_name.startswith("Original_firstOrderS_"):
        token = feature_name[len("Original_firstOrderS_") :]
        if _normalize(token) in CERR_FIRSTORDER_EXCLUDE:
            return True
    return False


def _radiomicsj_excluded(feature_name: str) -> bool:
    if feature_name in RADIOMICSJ_EXCLUDE:
        return True
    if feature_name.startswith(RADIOMICSJ_EXCLUDE_PREFIXES):
        return True
    return False


def _captk_excluded(feature_name: str) -> bool:
    family_token, feature_token = _captk_split_feature(feature_name)
    if not family_token or not feature_token:
        return False
    key = _normalize(feature_token)
    if re.match(r"^bin[0-9]+(probability|frequency)$", key):
        return True
    if re.match(r"^ellipsediameteraxis[0-9]+$", key):
        return True
    if re.match(r"^orientedboundingboxsizeaxis[0-9]+$", key):
        return True
    if family_token == "Intensity" and key == "mode":
        return True
    if key in CAPTK_EXCLUDE_TOKENS:
        return True
    if family_token == "Histogram" and key in CAPTK_HISTOGRAM_EXCLUDE:
        return True
    if family_token == "GLCM" and key in CAPTK_GLCM_EXCLUDE:
        return True
    if family_token == "GLRLM" and key in CAPTK_GLRLM_EXCLUDE:
        return True
    return False


def _sera_excluded(feature_name: str) -> bool:
    token = feature_name.strip()
    if token.startswith("F"):
        token = token[1:]
    token = token.lower()
    if token.startswith("mi_"):
        return True
    if token.endswith("_3d_avg"):
        return True
    if (
        token.endswith("_2d")
        or token.endswith("_2_5d")
        or token.endswith("_2d_avg")
        or token.endswith("_2d_comb")
        or token.endswith("_2_5d_avg")
        or token.endswith("_2_5d_comb")
    ):
        return True
    return False


def _qife_excluded(feature_name: str) -> bool:
    token = feature_name.strip().lower()
    return token in QIFE_EXCLUDE


ZRAD_EXCLUDE = {
    "bounding_box_min",
    "no_voxels",
    "no_bins",
}


def _zrad_excluded(feature_name: str) -> bool:
    token = feature_name.strip().lower()
    if token.startswith("f") and "_" in token:
        token = token[1:]
    if token in ZRAD_EXCLUDE:
        return True
    return False


MITK_EXCLUDE_TOKENS = {
    "voxelspace",
    "imagedimension",
    "numberofvoxels",
    "sum",
    "mode",
    "modeprobability",
    "robustmean",
    "coveredimageintensityrange",
    "centremassshiftuncorrected",
    "centreofmassshiftuncorrected",
    "boundingboxvolume",
    "surfacemeshmeshbased",
    "sphericitymeshmeshbased",
    "asphericitymeshmeshbased",
    "compactness1oldmeshbased",
    "compactness1oldvoxelbased",
    "compactness1meshmeshbased",
    "compactness2meshmeshbased",
    "sphericaldisproportionmeshmeshbased",
    "pcamajoraxislengthuncorrected",
    "pcaminoraxislengthuncorrected",
    "pcaleastaxislengthuncorrected",
    "pcaelongationuncorrected",
    "pcaflatnessuncorrected",
    "firstrowcolumnentropy",
    "secondrowcolumnentropy",
    "greylevelmean",
    "zonesizemean",
    "dependencecountmean",
    "expectedneighbourhoodsize",
    "averageneighbourhoodsize",
    "averageincompleteneighbourhoodsize",
    "percentageofcompleteneighbourhoods",
    "percentageofdependenceneighbourvoxels",
    "numberofruns",
    "zonedistancemean",
    "greylevelentropy",
}


def _mitk_excluded(feature_name: str) -> bool:
    raw_lower = feature_name.strip().lower()
    if raw_lower in {
        "softwareversion",
        "patient",
        "image",
        "segmentation",
        "endofmeasurement",
    }:
        return True

    token = feature_name.strip()
    for sep in ("::", ":", "|", ";"):
        if sep in token:
            token = token.split(sep)[-1]
    token = token.replace("(", " ").replace(")", " ")
    token_norm = _normalize(token)
    if token_norm in MITK_EXCLUDE_TOKENS:
        return True
    if token_norm.startswith("numberofruns"):
        return True

    token_lower = token.lower()
    if token_norm == "slicenumber":
        return True
    if "number of bins" in token_lower or "bin size" in token_lower:
        return True
    if "standard deviation" in token_lower:
        return True
    if "percentile" in token_lower:
        if all(
            key not in token_lower
            for key in (
                "10th percentile",
                "90th percentile",
                "percentile 10 value",
                "percentile 90 value",
            )
        ):
            return True
    if "run length::" in raw_lower and raw_lower.endswith(" means"):
        return True
    if "run length::" in raw_lower and raw_lower.endswith(" mean"):
        return True
    if raw_lower.startswith("volumetric features::voxel volume"):
        return True
    if raw_lower.startswith("first order::voxel volume"):
        return True
    if "volumetric features::" in raw_lower:
        # Exclude uncorrected aliases; keep canonical mesh-based morphology.
        if " old " in raw_lower or "(uncorrected)" in raw_lower:
            return True
    if (
        " index" in token_lower
        and "moran" not in token_lower
        and "geary" not in token_lower
        and "kurtosis" not in token_lower
    ):
        return True
    if "robust mean value" in token_lower:
        return True
    if "row-column entropy" in token_lower:
        return True
    if token_lower.startswith("overall row "):
        return True
    if token_lower.startswith("mean "):
        if any(
            key in token_lower
            for key in (
                "joint",
                "difference",
                "sum ",
                "angular",
                "contrast",
                "dissimilarity",
                "inverse",
                "correlation",
                "autocorrelation",
                "cluster",
                "measure of information",
                "row ",
            )
        ):
            return True
    # MITK emits statistical spread (Std) over directions for several texture features.
    # We exclude those spread terms from IBSI feature parity.
    if "std" in raw_lower:
        return True
    return False


def _mitk_context_family(feature_name: str) -> Optional[str]:
    lower = feature_name.lower()
    if "volumetric features" in lower or lower.startswith("volfeat"):
        return "morphology"
    if "morphological density" in lower:
        return "morphology"
    if "intensity volume histogram" in lower:
        return "ivh"
    if "local intensity" in lower:
        return "local_intensity"
    if "first order histogram" in lower:
        return "histogram"
    if "first order::" in lower:
        return "intensity"
    if "grey level distance zone" in lower or "gldz" in lower:
        return "gldzm"
    if (
        "co-occurenced based features" in lower
        or "co-occurrence" in lower
        or "cooc" in lower
    ):
        return "glcm"
    if "run length" in lower or "rlm" in lower:
        return "glrlm"
    if "grey level size zone" in lower or "glsz" in lower:
        return "glszm"
    if (
        "neighbourhood grey tone difference" in lower
        or "neighborhood gray tone difference" in lower
    ):
        return "ngtdm"
    if (
        "neighbouring grey level dependence" in lower
        or "neighboring gray level dependence" in lower
    ):
        return "ngldm"
    return None


def _moddicom_excluded(feature_name: str) -> bool:
    token = feature_name.strip().lower()
    if ".nic." in token:
        norm = token.replace(".", "_")
        if norm not in {
            "f_stat_nic_entropy",
            "f_stat_nic_kurt",
            "f_stat_nic_skew",
            "f_stat_nic_uniformity",
            "stat_nic_entropy",
            "stat_nic_kurt",
            "stat_nic_skew",
            "stat_nic_uniformity",
        }:
            return True
    norm = token.replace(".", "_")
    # moddicom exposes entropy/uniformity from non-discretised first-order stats;
    # these are not IBSI histogram definitions and are excluded from parity mapping.
    # Same applies to discretised-stat energy/RMS outputs (non-IBSI histogram metrics).
    if norm in {
        "f_stat_entropy",
        "f_stat_uniformity",
        "stat_entropy",
        "stat_uniformity",
        "f_ih_energy",
        "ih_energy",
        "f_ih_rms",
        "ih_rms",
    }:
        return True
    return False


def classify_feature(adapter: str, feature_name: str) -> tuple[Optional[str], str]:
    """
    Returns (ibsi_code, status) where status is: mapped, excluded, or unmapped.
    """
    if adapter == "pictologics":
        code = map_pictologics(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "pyradiomics":
        if feature_name in PYRADIOMICS_EXCLUDE:
            return (None, "excluded")
        code = map_pyradiomics(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "mirp":
        if _mirp_excluded(feature_name):
            return (None, "excluded")
        code = map_mirp(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "cerr":
        if _cerr_excluded(feature_name):
            return (None, "excluded")
        code = map_cerr(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "radiomicsj":
        if _radiomicsj_excluded(feature_name):
            return (None, "excluded")
        code = map_radiomicsj(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "captk":
        if _captk_excluded(feature_name):
            return (None, "excluded")
        code = map_captk(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "radiomics_develop":
        code = map_radiomics_develop(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "sera":
        if _sera_excluded(feature_name):
            return (None, "excluded")
        code = map_sera(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "qife":
        if _qife_excluded(feature_name):
            return (None, "excluded")
        code = map_qife(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "medimage":
        code = map_medimage(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "zrad":
        if _zrad_excluded(feature_name):
            return (None, "excluded")
        code = map_zrad(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "mitk":
        if _mitk_excluded(feature_name):
            return (None, "excluded")
        code = map_mitk(feature_name)
        return (code, "mapped" if code else "unmapped")

    if adapter == "moddicom":
        if _moddicom_excluded(feature_name):
            return (None, "excluded")
        code = map_moddicom(feature_name)
        return (code, "mapped" if code else "unmapped")

    return (None, "unmapped")


IBSI_FAMILY_TO_PICTOLOGICS = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_MIRP = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "statistics",
    "histogram": "intensity_histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_CERR = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_RADIOMICSJ = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_CAPTK = {
    "morphology": "morphology",
    "intensity": "intensity",
    "histogram": "histogram",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_RADIOMICS_DEVELOP = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_SERA = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_QIFE = {
    "morphology": "morphology",
    "glcm": "glcm",
}

IBSI_FAMILY_TO_MEDIMAGE = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_ZRAD = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_MITK = {
    "morphology": "morphology",
    "local_intensity": "local_intensity",
    "intensity": "intensity",
    "histogram": "histogram",
    "ivh": "ivh",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "gldzm": "gldzm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
}

IBSI_FAMILY_TO_MODDICOM = {
    "morphology": "morphology",
    "intensity": "intensity",
    "histogram": "histogram",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
}


def pictologics_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_PICTOLOGICS.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def mirp_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_MIRP.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def cerr_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_CERR.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def radiomicsj_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_RADIOMICSJ.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def captk_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_CAPTK.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def radiomics_develop_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_RADIOMICS_DEVELOP.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def sera_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_SERA.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def qife_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_QIFE.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def medimage_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_MEDIMAGE.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def zrad_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_ZRAD.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def mitk_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_MITK.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def moddicom_families_for_codes(codes: Iterable[str]) -> List[str]:
    families = set()
    for code in codes:
        fam = CODE_TO_FAMILY.get(code)
        if not fam:
            continue
        mapped = IBSI_FAMILY_TO_MODDICOM.get(fam)
        if mapped:
            families.add(mapped)
    return sorted(families)


def _catalog_add(
    catalog: Dict[str, List[str]], code: Optional[str], token: str
) -> None:
    if not code:
        return
    key = token.strip()
    if not key:
        return
    vals = catalog.setdefault(code, [])
    if key not in vals:
        vals.append(key)


def _catalog_finalize(catalog: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for code, tokens in catalog.items():
        uniq = sorted({t for t in tokens if t})
        if uniq:
            out[code] = uniq
    return out


def _catalog_from_mirp_prefixes(token_fmt: str = "{prefix}") -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = {}
    for prefix in MIRP_PREFIX_MAP:
        code = map_mirp(prefix)
        _catalog_add(catalog, code, token_fmt.format(prefix=prefix))
    return _catalog_finalize(catalog)


def adapter_mapping_catalog(adapter: str) -> Dict[str, List[str]]:
    """
    Return best-effort static mapping catalog:
    {ibsi_code: [adapter feature token(s)]}.
    """
    name = str(adapter).strip().lower()
    catalog: Dict[str, List[str]] = {}

    if name == "pictologics":
        for code in CODE_TO_NAME:
            _catalog_add(catalog, code, f"*_{code}")
        return _catalog_finalize(catalog)

    if name == "pyradiomics":
        for code, feats in PYRADIOMICS_CODE_TO_FEATURES.items():
            for family, feat in feats:
                _catalog_add(catalog, code, f"original_{family}_{feat}")
        return _catalog_finalize(catalog)

    if name == "mirp":
        return _catalog_from_mirp_prefixes("{prefix}")

    if name == "radiomics_develop":
        for code, tokens in _catalog_from_mirp_prefixes("F{prefix}").items():
            for tok in tokens:
                _catalog_add(catalog, code, tok)
        for token in RADIOMICS_DEVELOP_TOKEN_ALIASES:
            _catalog_add(catalog, map_radiomics_develop(f"F{token}"), f"F{token}")
        for token in RADIOMICS_DEVELOP_DIRECT_MAP:
            _catalog_add(catalog, map_radiomics_develop(f"F{token}"), f"F{token}")
        return _catalog_finalize(catalog)

    if name == "sera":
        # SERA 3D merged naming is used for parity timing/mapping.
        for prefix in MIRP_PREFIX_MAP:
            _catalog_add(catalog, map_sera(f"F{prefix}_3D_comb"), f"F{prefix}_3D_comb")
        for token in SERA_DIRECT_MAP:
            _catalog_add(catalog, map_sera(f"F{token}_3D"), f"F{token}_3D")
        return _catalog_finalize(catalog)

    if name == "medimage":
        for code, tokens in adapter_mapping_catalog("radiomics_develop").items():
            for tok in tokens:
                _catalog_add(catalog, code, tok)
        return _catalog_finalize(catalog)

    if name == "zrad":
        for prefix in MIRP_PREFIX_MAP:
            _catalog_add(catalog, map_zrad(f"{prefix}_3D_comb"), f"{prefix}_3D_comb")
        return _catalog_finalize(catalog)

    if name == "qife":
        for token, (ibsi_name, family) in QIFE_MORPH_MAP.items():
            _catalog_add(
                catalog, _code_from_name_in_family(ibsi_name, family), f"Fqife_{token}"
            )
        for token, (ibsi_name, family) in QIFE_GLCM_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Fqife_glcm_{token}",
            )
        return _catalog_finalize(catalog)

    if name == "moddicom":
        # moddicom tokens cover morphology + first-order + GLCM/GLRLM/GLSZM.
        # Runtime availability of morphology depends on optional contour backends.
        supported_families = {
            "morphology",
            "intensity",
            "histogram",
            "glcm",
            "glrlm",
            "glszm",
        }
        # Native moddicom tokens are usually "F_<family>.<token>".
        for prefix in MIRP_PREFIX_MAP:
            code = map_mirp(prefix)
            if not code or CODE_TO_FAMILY.get(code) not in supported_families:
                continue
            dotted = "F_" + prefix.replace("_", ".")
            _catalog_add(catalog, code, dotted)
        for token in MODDICOM_ALIASES:
            code = map_moddicom(token)
            if not code or CODE_TO_FAMILY.get(code) not in supported_families:
                continue
            _catalog_add(catalog, code, token)
        return _catalog_finalize(catalog)

    if name == "cerr":
        for token, (ibsi_name, family) in CERR_SHAPE_MAP.items():
            _catalog_add(
                catalog, _code_from_name_in_family(ibsi_name, family), f"shapeS_{token}"
            )
        for token, (ibsi_name, family) in CERR_FIRSTORDER_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_firstOrderS_{token}",
            )
        for token, (ibsi_name, family) in CERR_GLCM_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_glcmFeatS_AvgS_{token}",
            )
        for token, (ibsi_name, family) in CERR_GLRLM_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_rlmFeatS_AvgS_{token}",
            )
        for token, (ibsi_name, family) in CERR_GLSZM_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_szmFeatS_{token}",
            )
        for token, (ibsi_name, family) in CERR_NGTDM_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_ngtdmFeatS_{token}",
            )
        for token, (ibsi_name, family) in CERR_NGLDM_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_ngldmFeatS_{token}",
            )
        for token, (ibsi_name, family) in CERR_IVH_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_ivhFeaturesS_{token}",
            )
        for token, (ibsi_name, family) in CERR_LOCAL_INTENSITY_MAP.items():
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, family),
                f"Original_peakValleyFeatureS_{token}",
            )
        return _catalog_finalize(catalog)

    if name == "radiomicsj":
        for (family, token), ibsi_name in RADIOMICSJ_MAP.items():
            ibsi_family = RADIOMICSJ_PREFIX_TO_FAMILY.get(family)
            _catalog_add(
                catalog,
                _code_from_name_in_family(ibsi_name, ibsi_family),
                f"{family}_{token}",
            )
        return _catalog_finalize(catalog)

    if name == "captk":
        for family, fmap in CAPTK_FAMILY_MAPS.items():
            ibsi_family = CAPTK_OUTPUT_FAMILY_TO_IBSI.get(family)
            for feature_token, ibsi_name in fmap.items():
                _catalog_add(
                    catalog,
                    _code_from_name_in_family(ibsi_name, ibsi_family),
                    f"*_{family}_{feature_token}",
                )
        return _catalog_finalize(catalog)

    if name == "mitk":
        supported_families = set(IBSI_FAMILY_TO_MITK.keys())
        for token, (ibsi_name, family) in MITK_TOKEN_MAP.items():
            _catalog_add(catalog, _code_from_name_in_family(ibsi_name, family), token)
            # MITK reuses identical feature labels across texture families.
            # Capture all IBSI codes that share the same canonical name in families MITK supports.
            for code in NAME_TO_CODES.get(_normalize(ibsi_name), []):
                if CODE_TO_FAMILY.get(code) in supported_families:
                    _catalog_add(catalog, code, token)
        return _catalog_finalize(catalog)

    return {}


def mapping_catalog_for_adapters(
    adapters: Iterable[str],
) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for adapter in adapters:
        name = str(adapter).strip()
        if not name:
            continue
        out[name] = adapter_mapping_catalog(name)
    return out

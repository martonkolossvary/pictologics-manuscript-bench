from __future__ import annotations

import unittest

from bench.ibsi_families import CODE_TO_FAMILY
from bench.ibsi_identifiers import (
    DICOM_HISTOGRAM_SOURCE,
    IBSI_IDENTIFIER_REGISTRY_VERSION,
    IBSI_IDENTIFIER_SCHEMA_VERSION,
    IdentifierConflictError,
    IdentifierParameterError,
    BENCHMARK_CROSSWALK_SOURCE,
    find_identifier,
    get_feature,
    resolve_identifier,
    validate_registry,
)
from bench.ibsi_mapping import (
    _code_from_name_in_family,
    classify_feature,
    map_mirp,
    map_pyradiomics,
    pyradiomics_feature_selection,
)


class CorrectedFamilyAssignmentTests(unittest.TestCase):
    def test_thirteen_source_texture_swaps_remain_corrected(self) -> None:
        expected = {
            "ACUI": "glcm",
            "65HE": "ngtdm",
            "8CE5": "glrlm",
            "BYLV": "glszm",
            "OVBL": "glrlm",
            "XMSY": "glszm",
            "Y1RO": "glszm",
            "P30P": "glszm",
            "S1RA": "gldzm",
            "7HP3": "gldzm",
            "VIWW": "gldzm",
            "5SPA": "ngldm",
            "1PFV": "ngldm",
        }
        self.assertEqual(
            {code: CODE_TO_FAMILY.get(code) for code in expected}, expected
        )


class StrictSemanticMappingTests(unittest.TestCase):
    def test_duplicate_names_resolve_only_inside_the_requested_family(self) -> None:
        self.assertEqual(_code_from_name_in_family("Contrast", "glcm"), "ACUI")
        self.assertEqual(_code_from_name_in_family("Contrast", "ngtdm"), "65HE")
        self.assertEqual(
            _code_from_name_in_family("Grey level variance", "glrlm"), "8CE5"
        )
        self.assertEqual(
            _code_from_name_in_family("Grey level variance", "glszm"), "BYLV"
        )

    def test_no_cross_family_fallback(self) -> None:
        self.assertIsNone(_code_from_name_in_family("Contrast", "glrlm"))
        self.assertIsNone(
            _code_from_name_in_family("Grey level variance", "morphology")
        )

    def test_mirp_uses_longest_prefix_before_generic_prefix(self) -> None:
        cases = {
            "cm_inv_diff_fbn_n32": "IB1Z",
            "cm_inv_diff_norm_fbn_n32": "NDRX",
            "cm_inv_diff_mom_fbn_n32": "WF0Z",
            "cm_inv_diff_mom_norm_fbn_n32": "1QCO",
            "ih_max_grad_fbn_n32": "12CE",
            "ih_max_grad_g_fbn_n32": "8E6O",
            "ih_min_grad_fbn_n32": "VQB3",
            "ih_min_grad_g_fbn_n32": "RHQZ",
            "rlm_glnu_fbn_n32": "R5YN",
            "rlm_glnu_norm_fbn_n32": "OVBL",
            "szm_zsnu_fbn_n32": "4JP3",
            "szm_zsnu_norm_fbn_n32": "VB3A",
            "dzm_zdnu_fbn_n32": "V294",
            "dzm_zdnu_norm_fbn_n32": "IATH",
            "ngl_dcnu_fbn_n32": "Z87G",
            "ngl_dcnu_norm_fbn_n32": "OKJI",
        }
        self.assertEqual({feature: map_mirp(feature) for feature in cases}, cases)

    def test_pyradiomics_sum_squares_is_ibsi_joint_variance(self) -> None:
        native_name = "original_glcm_SumSquares"
        self.assertEqual(map_pyradiomics(native_name), "UR99")
        self.assertEqual(
            classify_feature("pyradiomics", native_name), ("UR99", "mapped")
        )
        self.assertEqual(
            pyradiomics_feature_selection(["UR99"]),
            {"glcm": ["SumSquares"]},
        )

    def test_pyradiomics_explicit_deprecated_shape_features_are_mapped(self) -> None:
        expected = {
            "original_shape_Compactness1": "SKGS",
            "original_shape_Compactness2": "BQWJ",
            "original_shape_SphericalDisproportion": "KRCK",
        }
        self.assertEqual(
            {native: map_pyradiomics(native) for native in expected},
            expected,
        )
        self.assertEqual(
            pyradiomics_feature_selection(expected.values()),
            {
                "shape": [
                    "Compactness1",
                    "Compactness2",
                    "SphericalDisproportion",
                ]
            },
        )

    def test_zrad_native_ibsi_morphology_outputs_are_not_suppressed(self) -> None:
        expected = {
            "morph_asphericity": "25C7",
            "morph_av": "2PR5",
            "morph_area_mesh": "C0JK",
            "morph_comp_1": "SKGS",
            "morph_comp_2": "BQWJ",
            "morph_diam": "L0JK",
            "morph_integ_int": "99N0",
            "morph_sphericity": "QCFX",
            "morph_sph_dispr": "KRCK",
            "morph_vol_approx": "YEKZ",
            "morph_vol_dens_aabb": "PBX1",
            "morph_area_dens_aabb": "R59B",
            "morph_vol_dens_aee": "6BDE",
            "morph_area_dens_aee": "RDD2",
            "morph_vol_dens_conv_hull": "R3ER",
            "morph_area_dens_conv_hull": "7T7F",
        }
        self.assertEqual(
            {native: classify_feature("zrad", native) for native in expected},
            {native: (code, "mapped") for native, code in expected.items()},
        )


class VersionedIdentifierTests(unittest.TestCase):
    def test_registry_is_versioned_and_structurally_valid(self) -> None:
        self.assertRegex(IBSI_IDENTIFIER_SCHEMA_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertRegex(IBSI_IDENTIFIER_REGISTRY_VERSION, r"^\d{4}-\d{2}-\d{2}$")
        validate_registry()

    def test_histogram_conflict_requires_source_scope(self) -> None:
        matches = find_identifier("GPMT")
        self.assertEqual(
            {match.semantic_key for match in matches},
            {
                "intensity_histogram.percentile_10",
                "intensity_histogram.percentile_90",
            },
        )
        with self.assertRaises(IdentifierConflictError):
            resolve_identifier("GPMT")

        self.assertEqual(
            resolve_identifier("GPMT", source=DICOM_HISTOGRAM_SOURCE).semantic_key,
            "intensity_histogram.percentile_10",
        )
        self.assertEqual(
            resolve_identifier("GPMT", source=BENCHMARK_CROSSWALK_SOURCE).semantic_key,
            "intensity_histogram.percentile_90",
        )
        self.assertEqual(
            resolve_identifier("1PR").semantic_key,
            "intensity_histogram.percentile_10",
        )
        self.assertEqual(
            resolve_identifier("OZ0C").semantic_key,
            "intensity_histogram.percentile_90",
        )

    def test_glszm_and_mirp_aliases_resolve_to_canonical_semantics(self) -> None:
        self.assertEqual(
            resolve_identifier("P001").semantic_key,
            "glszm.small_zone_emphasis",
        )
        self.assertEqual(get_feature("glszm.small_zone_emphasis").preferred_id, "5QRC")
        self.assertEqual(resolve_identifier("P6QZ1").semantic_key, "glcm.sum_entropy")
        self.assertEqual(get_feature("glcm.sum_entropy").preferred_id, "P6QZ")

    def test_ivh_general_specific_and_alias_ids_are_distinct(self) -> None:
        with self.assertRaises(IdentifierParameterError):
            resolve_identifier("BC2M")

        self.assertEqual(
            resolve_identifier(
                "BC2M", parameters={"intensity_fraction": "0.10"}
            ).semantic_key,
            "ivh.volume_at_intensity_fraction_10",
        )

        v10 = resolve_identifier("NK6P")
        self.assertEqual(v10.semantic_key, "ivh.volume_at_intensity_fraction_10")
        self.assertEqual(v10.general_id.value, "BC2M")
        self.assertEqual(v10.specific_id.value, "NK6P")
        self.assertEqual(dict(v10.parameter_map), {"intensity_fraction": "0.10"})
        self.assertEqual(resolve_identifier("BC2M_10"), v10)

        v90 = resolve_identifier("4279")
        self.assertEqual(v90.semantic_key, "ivh.volume_at_intensity_fraction_90")
        self.assertEqual(resolve_identifier("BC2M_90"), v90)

        self.assertEqual(
            resolve_identifier("PWN1").semantic_key,
            "ivh.intensity_at_volume_fraction_10",
        )
        self.assertEqual(
            resolve_identifier("BOHI").semantic_key,
            "ivh.intensity_at_volume_fraction_90",
        )
        self.assertEqual(
            resolve_identifier("WITY").semantic_key,
            "ivh.volume_fraction_difference_10_90",
        )
        with self.assertRaises(IdentifierParameterError):
            resolve_identifier("DDTU")
        self.assertEqual(
            resolve_identifier(
                "DDTU",
                parameters={
                    "intensity_fraction_low": "0.10",
                    "intensity_fraction_high": "0.90",
                },
            ).semantic_key,
            "ivh.volume_fraction_difference_10_90",
        )
        self.assertEqual(
            resolve_identifier("JXJA").semantic_key,
            "ivh.intensity_fraction_difference_10_90",
        )
        with self.assertRaises(IdentifierParameterError):
            resolve_identifier("CNV2")


if __name__ == "__main__":
    unittest.main()

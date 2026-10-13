import math

import numpy as np
import pytest

from scripts.qualify_spatial_statistics import compare_spatial, reference_statistics


def test_two_voxel_definition_and_affine_intensity_invariance():
    data = np.array([0., 2.]).reshape(2, 1, 1)
    mask = np.ones_like(data)
    expected = {"N365": -1., "NPT7": 1.}
    assert reference_statistics(data, mask, (1., 2., 3.)) == expected
    assert reference_statistics(-3 * data + 7, mask, (4., 2., 3.)) == expected


def test_anisotropic_spacing_changes_weights():
    data = np.array([0., 1., 5., 2.]).reshape(2, 2, 1)
    mask = np.ones_like(data)
    isotropic = reference_statistics(data, mask, (1., 1., 1.))
    anisotropic = reference_statistics(data, mask, (1., 4., 1.))
    assert abs(isotropic["N365"] - anisotropic["N365"]) > 0.01


@pytest.mark.parametrize("selected", [0, 1, 4])
def test_degenerate_statistics_are_undefined(selected):
    data = np.ones((2, 2, 1))
    mask = np.zeros(4)
    mask[:selected] = 1
    assert all(math.isnan(v) for v in reference_statistics(data, mask.reshape(data.shape),
                                                         (1., 1., 1.)).values())


@pytest.mark.parametrize("actual", [{"N365": 0}, {"N365": float("nan"), "NPT7": 1},
                                    {"N365": 0.1, "NPT7": 1}])
def test_comparison_rejects_missing_nonfinite_and_wrong_values(actual):
    with pytest.raises(ValueError):
        compare_spatial(actual, {"N365": 0, "NPT7": 1})

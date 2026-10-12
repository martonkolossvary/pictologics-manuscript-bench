from __future__ import annotations

import sys
import types
from functools import lru_cache
from unittest import mock

import numpy as np
import pytest

from bench.adapters import pictologics_adapter
from bench.adapters.base import run_timed_computation
from tests.test_adapter_intensity_integration import fake_pictologics_modules


@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize(
    "families", [["glcm"], ["glszm"], list(pictologics_adapter._TEXTURE_FAMILIES)]
)
def test_texture_preserves_native_grouping_and_resegmented_mask(families, timed):
    modules, _, image, mask = fake_pictologics_modules()
    selected_mask = type(mask)(mask.array.copy())
    selected_mask.array[0, 0, 0] = 0
    expected = {f"{family}_feature": 1.0 for family in families}
    calls = []

    def grouped_features(discrete, intensity_mask, n_bins, **kwargs):
        np.testing.assert_array_equal(intensity_mask, selected_mask.array)
        np.testing.assert_array_equal(kwargs["distance_mask_array"], mask.array)
        assert kwargs["families"] == families
        assert n_bins == 32
        assert discrete.shape == image.array.shape
        calls.append(True)
        return expected

    with modules:
        from pictologics import features, preprocessing

        features.calculate_all_texture_features = grouped_features
        features.calculate_all_texture_matrices = mock.Mock(
            side_effect=AssertionError("must retain the public compact grouped path")
        )
        preprocessing.resegment_mask = lambda *_args, **_kwargs: selected_mask
        arguments = dict(
            image=image,
            mask=mask,
            families=families,
            discretization="fbn",
            bins=32,
            bin_width=3.0,
            intensity_range=(1.0, 8.0),
        )
        if timed:
            calculate, finalize = pictologics_adapter._prepare_calculation_only(
                **arguments
            )
            assert calls == []
            for _ in range(2):
                assert finalize(calculate()) == expected
            assert len(calls) == 2
        else:
            assert pictologics_adapter._compute_features(**arguments) == expected
            assert len(calls) == 1


@pytest.mark.parametrize(
    "families, selected",
    [
        (["glcm"], ["neighbour"]),
        (["glszm"], ["zone"]),
        (["ngtdm", "gldzm"], ["neighbour", "zone"]),
        (["local_intensity", "glcm", "glszm"], ["sphere", "neighbour", "zone"]),
    ],
)
def test_all_selected_geometry_is_rebuilt_inside_every_calculation(families, selected):
    constructed = []

    def geometry(name):
        @lru_cache(maxsize=16)
        def cached():
            constructed.append(name)
            return 1.0

        return cached

    caches = {name: geometry(name) for name in ("sphere", "neighbour", "zone")}
    intensity = types.ModuleType("pictologics.features.intensity")
    texture = types.ModuleType("pictologics.features.texture")
    intensity._sphere_offsets_for_radius = caches["sphere"]
    texture._neighbour_offsets = caches["neighbour"]
    texture._zone_offsets = caches["zone"]
    with mock.patch.dict(
        sys.modules,
        {"pictologics.features.intensity": intensity, "pictologics.features.texture": texture},
    ):
        prepare = pictologics_adapter._calculation_preparer(families)
    for cached in caches.values():
        cached()  # Model a populated package warmup cache.

    def compute(_state):
        before = len(constructed)
        values = {name: caches[name]() for name in selected}
        assert constructed[before:] == selected
        return values

    _, timing = run_timed_computation(
        compute,
        iterations=3,
        prepare_fn=prepare,
        target_observation_window_sec=1e-9,
        calibration_cv_threshold=0.99,
        calibration_span_ratio=1000,
    )
    assert len(constructed) == 3 + len(selected) * timing["total_calculation_calls"]


def test_geometry_hook_is_absent_for_non_geometric_families():
    assert pictologics_adapter._calculation_preparer(["intensity", "histogram", "ivh"]) is None

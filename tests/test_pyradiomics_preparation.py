from __future__ import annotations

import sys
import types
from unittest import mock

import numpy as np
import pytest

from bench.adapters import pyradiomics_adapter


@pytest.mark.parametrize("resegment_shape", [False, True])
def test_intensity_resegmentation_preserves_the_selected_morphology_roi(
    resegment_shape,
):
    """Removing peripheral intensities must not silently shrink native shape."""
    image = np.zeros((7, 7, 7), dtype=float)
    image[2:5, 2:5, 2:5] = 10.0
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1:6, 1:6, 1:6] = 1
    constructed_shapes = []
    radiomics = types.ModuleType("radiomics")

    def check_mask(_image, selected_mask, **_settings):
        coordinates = np.nonzero(selected_mask)
        bounds = tuple(
            slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
        )
        return bounds, None

    def crop_to_mask(source_image, source_mask, bounds, **_settings):
        return source_image[bounds], source_mask[bounds]

    radiomics.imageoperations = types.SimpleNamespace(
        checkMask=check_mask,
        resegmentMask=lambda source, roi, **_settings: (
            (roi > 0) & (source >= 5.0) & (source <= 15.0)
        ).astype(np.uint8),
        cropToTumorMask=crop_to_mask,
    )

    class Shape:
        def __init__(self, _image, selected_mask, **_settings):
            self.volume = float(np.count_nonzero(selected_mask))
            constructed_shapes.append(self.volume)

        def enableFeatureByName(self, _name):
            pass

        def execute(self):
            return {"VoxelVolume": self.volume}

    class FirstOrder:
        def __init__(self, source, selected_mask, **_settings):
            self.values = source[selected_mask > 0]

        def enableFeatureByName(self, _name):
            pass

        def execute(self):
            return {"Mean": float(np.mean(self.values))}

    radiomics.getFeatureClasses = lambda: {"shape": Shape, "firstorder": FirstOrder}
    with mock.patch.dict(sys.modules, {"radiomics": radiomics}):
        calculate, finalize = pyradiomics_adapter._prepare_calculation_only(
            image=image,
            mask=mask,
            selection={"shape": ["VoxelVolume"], "firstorder": ["Mean"]},
            settings={"resegmentRange": [5.0, 15.0], "resegmentShape": resegment_shape},
            radiomics_module=radiomics,
        )

    # Mesh construction must still occur in each native calculation.
    assert constructed_shapes == []
    for _ in range(2):
        values = finalize(calculate())
        assert values["original_shape_VoxelVolume"] == (
            27.0 if resegment_shape else 125.0
        )
        assert values["original_firstorder_Mean"] == 10.0
    assert len(constructed_shapes) == 2

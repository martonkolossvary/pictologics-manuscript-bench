from __future__ import annotations

import json

import pytest

from scripts import qualify_adapter_calculations as qualification


def test_equivalence_rejects_missing_extra_empty_and_nonfinite_features():
    for reference, observed in (
        ({"a": 1.0}, {"b": 1.0}),
        ({"a": 1.0}, {"a": 1.0, "b": 2.0}),
        ({}, {}),
        ({"a": float("nan")}, {"a": 1.0}),
        ({"a": 1.0}, {"a": float("inf")}),
        ({"a": 1.0}, {"a": True}),
        ({"a": 1.0}, {"a": 1.1}),
    ):
        with pytest.raises(ValueError):
            qualification.compare_values(reference, observed)


def test_equivalence_accepts_reordering_and_declared_numerical_tolerance():
    result = qualification.compare_values(
        {"a": 1.0, "b": 0.0}, {"b": 1e-13, "a": 1.0 + 1e-10}
    )
    assert result["features"] == 2
    assert result["max_absolute_difference"] < 1e-9


def test_worker_records_repeat_mismatch_and_continues_other_workloads(tmp_path, monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "3.1.0")
    calls = []

    def prepare(_fixture, _families, workload, _discrete):
        calls.append(workload)
        results = iter(({"feature": 1.0}, {"feature": 2.0 if workload == "morphology" else 1.0}))
        return {"feature": 1.0}, lambda: next(results)

    monkeypatch.setattr(qualification, "_pyradiomics", prepare)
    output = tmp_path / "result.json"
    assert qualification.worker("pyradiomics", {}, output) == 1
    report = json.loads(output.read_text())
    assert report["timing_observations"] is False
    records = {item["workload"]: item for item in report["checks"]}
    assert records["morphology"]["status"] == "failed"
    assert "value mismatch" in records["morphology"]["error"]
    assert records["texture"]["status"] == "passed"
    assert records["intensity"]["status"] == "passed"
    assert records["spatial_autocorrelation"]["status"] == "unsupported"
    assert set(calls) == {"morphology", "intensity", "texture"}


def test_fixture_is_anisotropic_nondegenerate_and_spans_identity_ivh(tmp_path):
    import nibabel as nib
    import numpy as np

    paths = qualification.create_fixture(tmp_path)
    mask = nib.load(paths["mask"]).get_fdata() > 0
    levels = nib.load(paths["levels"])
    assert len(set(levels.header.get_zooms())) == 3
    assert np.count_nonzero(mask) > 32
    assert levels.get_fdata()[mask].min() == 1
    assert levels.get_fdata()[mask].max() == 32
    assert np.isfinite(nib.load(paths["raw"]).get_fdata()).all()

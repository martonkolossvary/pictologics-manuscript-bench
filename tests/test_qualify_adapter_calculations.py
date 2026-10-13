from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_qualification_launches_venv_path_instead_of_resolved_base_python(tmp_path, monkeypatch):
    from bench import env

    runtime_repo = tmp_path / "frozen-runtime"
    profile_dir = runtime_repo / "configs" / "adapters"
    profile_dir.mkdir(parents=True)
    profile_dir.joinpath("pictologics.yaml").write_bytes(
        (qualification.ROOT / "configs/adapters/pictologics.yaml").read_bytes()
    )
    profile = env.load_runtime_profiles()["pictologics"]
    expected_version = profile.metadata_version or profile.version
    launcher = str(runtime_repo / ".venvs/adapters/pictologics/bin/python")
    resolved_base_python = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
    output_dir = tmp_path / "qualification"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            assert kwargs["cwd"] == runtime_repo
            # Execute the generated verifier against a stub to exercise its
            # launch-path serialization, including a POSIX-style venv symlink.
            def verify(actual_profile, *, smoke):
                assert actual_profile.name == "pictologics"
                assert smoke is False
                return {"distribution_metadata_version": expected_version,
                        "python_executable": resolved_base_python}

            stub = SimpleNamespace(
                load_runtime_profiles=lambda: {"pictologics": profile},
                verify_profile=verify,
                env_dir_for_profile=lambda _: Path(launcher).parent.parent,
                env_python=lambda _: Path(launcher),
            )
            import contextlib
            import io
            import bench

            with monkeypatch.context() as patch:
                patch.setattr(bench, "env", stub)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exec(command[2], {})
            return SimpleNamespace(stdout=stdout.getvalue(), returncode=0)
        assert command[0] == launcher
        assert command[0] != resolved_base_python
        assert kwargs["cwd"] == qualification.ROOT
        output_dir.joinpath("pictologics.json").write_text(json.dumps({"checks": []}))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    report = qualification.qualify(["pictologics"], output_dir, runtime_repo, 10)
    assert report["complete"] and report["status"] == "passed"
    assert report["adapters"][0]["worker_executable"] == launcher
    assert report["adapters"][0]["environment"]["python_executable"] == resolved_base_python
    assert len(calls) == 2

from bench.compliance import ibsi2_candidates as candidates


def test_pictologics_export_preserves_ras_world_geometry_and_voxels(tmp_path):
    from types import SimpleNamespace
    import nibabel as nib
    import numpy as np

    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    direction = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    image = SimpleNamespace(array=data, direction=direction,
                            spacing=(0.8, 1.7, 3.), origin=(-123., 42., 7.))
    destination = tmp_path / "export.nii.gz"
    candidates._save_pictologics_image(image, destination)
    exported = nib.load(destination)
    lps = np.eye(4)
    lps[:3, :3] = direction @ np.diag(image.spacing)
    lps[:3, 3] = image.origin
    np.testing.assert_allclose(exported.affine, np.diag([-1., -1., 1., 1.]) @ lps,
                               rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asanyarray(exported.dataobj), data)


def test_stable_only_phase2_records_pictologics_preprocessing_dependency():
    stable = ("medimage", "mirp", "pyradiomics", "zrad")
    assert candidates._environment_dependencies(stable, ("phase1",)) == stable
    assert candidates._environment_dependencies(stable, ("phase2",)) == (*stable, "pictologics")
    assert candidates._environment_dependencies(("pictologics",), ("phase2",)) == ("pictologics",)


def test_preprocessing_environment_changes_invalidate_candidate_resume(tmp_path, monkeypatch):
    source = tmp_path / "bench/compliance/ibsi2_candidates.py"
    monkeypatch.setattr(candidates, "__file__", str(source))
    files = [source, tmp_path / "bench/compliance/ibsi2_protocol.py",
             tmp_path / "bench/compliance/ibsi2_native_backends.py",
             tmp_path / "configs/compliance/ibsi2_phase2_preprocessing_A.json",
             tmp_path / "configs/compliance/ibsi2_phase2_preprocessing_B.json",
             tmp_path / "data/ibsi2/source/manifest.json",
             tmp_path / ".venvs/adapters/mirp/environment.json",
             tmp_path / ".venvs/adapters/pictologics/environment.json"]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    dependencies = candidates._environment_dependencies(("mirp",), ("phase2",))
    before = candidates._source_fingerprint(tmp_path, dependencies)
    files[-1].write_text('{"version": "future"}', encoding="utf-8")
    assert before != candidates._source_fingerprint(tmp_path, dependencies)

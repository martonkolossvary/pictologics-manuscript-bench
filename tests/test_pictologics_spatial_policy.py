from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.check_pictologics_spatial_policy import (
    GIB, PROBES, assess_acceptance, predict_legacy_policy, probe_cap,
)
from scripts import check_pictologics_spatial_policy as policy


def _results(cap):
    return [{"probe": mode, "status": "observed", "native_cap_bytes": cap} for mode in PROBES]


def test_windows_legacy_fallback_rejected_on_32_gib_host():
    result = assess_acceptance(_results(4 * GIB), {"bytes": 32 * GIB})
    assert result["status"] == "failed"
    assert result["mismatching_probes"] == list(PROBES)
    assert result["expected_cap_bytes"] == 16 * GIB


def test_native_match_does_not_hide_unix_missing_sysconf_fallback():
    results = _results(4 * GIB)
    results[0]["native_cap_bytes"] = 16 * GIB
    assert assess_acceptance(results, {"bytes": 64 * GIB})["mismatching_probes"] == list(PROBES[1:])


@pytest.mark.parametrize("physical", [6 * GIB, 16 * GIB, 64 * GIB])
def test_cross_platform_policy_accepts_half_ram_with_hard_ceiling(physical):
    assert assess_acceptance(_results(min(16 * GIB, physical // 2)),
                             {"bytes": physical})["status"] == "passed"


def test_unavailable_and_changed_api_never_report_pass():
    assert assess_acceptance(_results(4 * GIB), {"bytes": None})["status"] == "unavailable"
    result = _results(4 * GIB)
    result[0]["status"] = "unsupported"
    assert assess_acceptance(result, {"bytes": 8 * GIB})["status"] == "unsupported"


def test_missing_duplicate_and_failed_probes_fail_closed():
    good = _results(16 * GIB)
    assert assess_acceptance(good[:-1], {"bytes": 32 * GIB})["status"] == "failed"
    assert assess_acceptance(good + [good[0]], {"bytes": 32 * GIB})["status"] == "failed"
    good[0] = {"probe": "native", "status": "failed", "error": "broken"}
    assert assess_acceptance(good, {"bytes": 32 * GIB})["status"] == "failed"


@pytest.mark.parametrize("mode", PROBES[1:])
@pytest.mark.parametrize("exists", [True, False])
def test_simulations_restore_sysconf_even_if_cap_raises(mode, exists):
    original = lambda _: 123  # noqa: E731
    fake_os = SimpleNamespace(sysconf=original) if exists else SimpleNamespace()

    def cap():
        if mode == "sysconf_missing":
            assert not hasattr(fake_os, "sysconf")
        else:
            with pytest.raises(OSError if mode == "sysconf_oserror" else ValueError):
                fake_os.sysconf("SC_PAGE_SIZE")
        raise RuntimeError("Expected probe failure")

    assert probe_cap(cap, mode, fake_os)["status"] == "failed"
    assert getattr(fake_os, "sysconf", None) is (original if exists else None)


def test_predictions_use_geometry_only_and_memory_boundary_is_inclusive():
    calls = []

    def next_length(n, *, real):
        assert real
        calls.append(n)
        return 640

    estimate = 32 * 640 ** 3
    allowed = predict_legacy_policy((320,) * 3, 320 ** 3, estimate, next_length)
    denied = predict_legacy_policy((320,) * 3, 320 ** 3, estimate - 1, next_length)
    assert calls == [639] * 6
    assert allowed["predicted_method"] == "fft"
    assert denied["memory_guard_fallback"]


def test_sparse_roi_selects_pairs_even_with_large_memory_cap():
    result = predict_legacy_policy((256,) * 3, 2, 16 * GIB, lambda n, **_: n)
    assert result["predicted_method"] == "pairwise"
    assert not result["memory_guard_fallback"]


@pytest.mark.parametrize("shape, count", [((2048, 1, 1), 3), ((2, 2), 1), ((2, 2, 2), 9)])
def test_invalid_or_unbounded_predictions_are_rejected(shape, count):
    with pytest.raises(ValueError):
        predict_legacy_policy(shape, count, 16 * GIB, lambda n, **_: n)


@pytest.mark.parametrize("explicit", [True, False])
def test_cli_preserves_virtualenv_launcher_path(monkeypatch, tmp_path, explicit):
    """Keep the venv path even on a host whose launcher symlinks to base Python."""
    launcher = tmp_path / ".venvs" / "adapters" / "pictologics" / "bin" / "python"
    inspected = []

    def inspect(python, **_kwargs):
        inspected.append(python)
        return {"observations_complete": True, "acceptance": {"status": "failed"}}

    # Model a POSIX symlink even on Windows where creating one needs privileges.
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda self, **kwargs:
                        tmp_path / "base-python" if self == launcher
                        else original_resolve(self, **kwargs))
    monkeypatch.setattr(policy, "inspect_policy", inspect)
    if explicit:
        args = ["--python", str(launcher)]
    else:
        from bench import env

        monkeypatch.setattr(env, "load_runtime_profiles", lambda _: {
            "pictologics": SimpleNamespace(env_dir=".venvs/adapters/pictologics")})
        monkeypatch.setattr(env, "env_python", lambda _: launcher)
        args = ["--runtime-repo", str(tmp_path)]
    assert policy.main(args) == 0
    assert inspected == [launcher]


@pytest.mark.parametrize("complete, acceptance, strict, expected", [
    (True, "failed", False, 0),
    (True, "failed", True, 1),
    (True, "unavailable", True, 1),
    (True, "passed", True, 0),
    (False, "unsupported", False, 2),
    (False, "failed", True, 2),
])
def test_cli_distinguishes_observations_from_release_acceptance(
    monkeypatch, complete, acceptance, strict, expected,
):
    monkeypatch.setattr(policy, "inspect_policy", lambda *_args, **_kwargs: {
        "observations_complete": complete, "acceptance": {"status": acceptance}})
    args = ["--python", "candidate-python"]
    if strict:
        args.append("--require-physical-ram-cap")
    assert policy.main(args) == expected

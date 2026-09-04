from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import call, patch

from scripts.bootstrap_reproducibility_inputs import _atomic_json, _ensure_repository


def test_generated_json_uses_canonical_lf_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"

    _atomic_json(destination, {"value": [1, 2]})

    expected = (json.dumps({"value": [1, 2]}, indent=2, sort_keys=True) + "\n").encode()
    assert destination.read_bytes() == expected
    assert b"\r\n" not in destination.read_bytes()


def test_managed_git_cache_disables_checkout_line_ending_conversion(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cache" / "source"
    target.mkdir(parents=True)
    specification = {"commit": "a" * 40, "url": "https://example.invalid/source.git"}

    with (
        patch(
            "scripts.bootstrap_reproducibility_inputs._verify_repository",
            return_value=target,
        ) as verify,
        patch("scripts.bootstrap_reproducibility_inputs._run") as run,
    ):
        observed = _ensure_repository(
            "source", specification, tmp_path / "cache", overrides={}
        )

    assert observed == target
    assert run.call_args_list == [
        call(["git", "-C", str(target), "config", "core.autocrlf", "false"]),
        call(["git", "-C", str(target), "config", "core.eol", "lf"]),
        call(
            [
                "git",
                "-C",
                str(target),
                "checkout",
                "--detach",
                "--force",
                "a" * 40,
            ]
        ),
    ]
    verify.assert_called_once_with(target, "a" * 40)

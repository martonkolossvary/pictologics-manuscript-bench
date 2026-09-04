from __future__ import annotations

from pathlib import Path

import pytest

import bench.submission as submission
from bench.benchmark_ledger import atomic_write_json, atomic_write_text, sha256_file
from bench.benchmark_models import fingerprint, run_spec_identity
from bench.submission import (
    PILLARS,
    SubmissionError,
    submission_identity,
    validate_submission_bundle,
    validate_submission_tree,
)


def _write_bundle(root: Path) -> Path:
    machine = {
        "machine_id": "mac-m4pro-01",
        "platform": "Darwin",
        "platform_release": "test",
        "machine": "arm64",
        "cpu_model": "Apple M4 Pro",
        "memory_total_bytes": 48 * 1024**3,
    }
    architecture, submission_id = submission_identity(
        machine=machine,
        source_commit="a" * 40,
        submission_date="2026-09-01",
        run_fingerprints={pillar: pillar * 2 for pillar in PILLARS},
    )
    bundle = root / architecture / submission_id
    files = []
    pillars = {}
    for pillar in PILLARS:
        report_dir = bundle / pillar
        report_dir.mkdir(parents=True)
        artifact = report_dir / "summary.csv"
        atomic_write_text(artifact, "metric,value\nruntime,1.0\n")
        report_manifest = {
            "schema_version": 1,
            "publication_attested": True,
            "artifact_count": 1,
            "artifacts": [
                {
                    "path": artifact.name,
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            ],
        }
        report_manifest_path = report_dir / "report_manifest.json"
        atomic_write_json(report_manifest_path, report_manifest)
        for path in (artifact, report_manifest_path):
            files.append(
                {
                    "path": path.relative_to(bundle).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        pillars[pillar] = {
            "run_id": pillar,
            "run_fingerprint": pillar * 2,
            "run_status": "completed",
            "report_path": f"{pillar}/report_manifest.json",
            "report_manifest_sha256": sha256_file(report_manifest_path),
            "artifact_count": 1,
        }
    atomic_write_json(
        bundle / "submission_manifest.json",
        {
            "schema_version": 1,
            "submission_id": submission_id,
            "submission_date": "2026-09-01",
            "source_commit": "a" * 40,
            "architecture": architecture,
            "machine": machine,
            "pillars": pillars,
            "file_count": len(files),
            "files": sorted(files, key=lambda item: item["path"]),
            "manifest_self_excluded": True,
        },
    )
    return bundle


def test_submission_name_includes_platform_architecture_machine_cpu_and_date() -> None:
    architecture, submission_id = submission_identity(
        machine={
            "machine_id": "lab-mac-01",
            "platform": "Darwin",
            "machine": "arm64",
            "cpu_model": "Apple M4 Pro",
            "host_settings": {
                "power_mode_tag": "macos-high-power-pmset-2",
            },
        },
        source_commit="a" * 40,
        submission_date="2026-09-01",
        run_fingerprints={pillar: "b" * 64 for pillar in PILLARS},
    )

    assert architecture == "macos-arm64"
    assert submission_id.startswith(
        "lab-mac-01--apple-m4-pro--macos-high-power-pmset-2--2026-09-01--"
    )


def test_submission_bundle_is_complete_and_checksum_bound(tmp_path: Path) -> None:
    result_root = tmp_path / "benchmark-results"
    result_root.mkdir()
    atomic_write_text(result_root / "README.md", "# Results\n")
    bundle = _write_bundle(result_root)

    manifest = validate_submission_bundle(bundle)

    assert set(manifest["pillars"]) == set(PILLARS)
    assert validate_submission_tree(result_root) == []


def test_submission_bundle_rejects_modified_artifact(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "benchmark-results")
    (bundle / PILLARS[0] / "summary.csv").write_text("changed\n", encoding="utf-8")

    with pytest.raises(SubmissionError, match="size mismatch|checksum mismatch"):
        validate_submission_bundle(bundle)


def test_packager_emits_one_complete_architecture_namespaced_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "a" * 40
    machine_id = "linux-lab-01"
    machine = {
        "machine_id": machine_id,
        "platform": "Linux",
        "platform_release": "test",
        "machine": "x86_64",
        "cpu_model": "Example CPU 9000",
    }
    result_root = tmp_path / "raw-results"
    for pillar in PILLARS:
        run_dir = result_root / machine_id / pillar
        run_dir.mkdir(parents=True)
        run_spec = {
            "schema_version": 9,
            "run_id": pillar,
            "dataset": pillar,
            "benchmark_machine": machine,
        }
        atomic_write_json(run_dir / "run_spec.json", run_spec)
        atomic_write_json(
            run_dir / "run_meta.json",
            {
                "run_fingerprint": fingerprint(run_spec_identity(run_spec)),
                "git_commit": source_commit,
                "run_status": "completed",
            },
        )

    def fake_generate_report(input_dir: Path, output_dir: Path) -> None:
        del input_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "summary.csv"
        atomic_write_text(artifact, "metric,value\nruntime,1.0\n")
        atomic_write_json(
            output_dir / "report_manifest.json",
            {
                "schema_version": 1,
                "publication_attested": True,
                "artifact_count": 1,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                ],
            },
        )

    monkeypatch.setattr(submission, "_git_source_commit", lambda repository: source_commit)
    monkeypatch.setattr("bench.report.generate_report", fake_generate_report)

    bundle = submission.package_submission(
        repository=tmp_path,
        result_root=result_root,
        output_root=tmp_path / "benchmark-results",
        machine_id=machine_id,
        submission_date="2026-09-01",
    )

    assert bundle.parent.name == "linux-x86-64"
    assert bundle.name.startswith(
        "linux-lab-01--example-cpu-9000--power-mode-unavailable--2026-09-01--"
    )
    assert set(validate_submission_bundle(bundle)["pillars"]) == set(PILLARS)

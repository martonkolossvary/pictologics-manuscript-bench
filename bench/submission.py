from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from bench.benchmark_ledger import (
    atomic_write_json,
    native_path,
    read_text,
    sha256_file,
)
from bench.benchmark_models import fingerprint, run_spec_identity
from bench.power_provenance import combine_power_summaries


PILLARS = (
    "pillar1_morphology",
    "pillar2_whole_anatomy",
    "pillar3_ibsi2_phase3",
)
PUBLICATION_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".svg", ".xlsx"}
SUBMISSION_SCHEMA_VERSION = 1
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


class SubmissionError(RuntimeError):
    """Raised when benchmark results are not safe to publish."""


def slug(value: object, *, maximum: int = 80) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    token = token[:maximum].rstrip("-")
    if not token:
        raise SubmissionError(f"cannot derive a public path token from {value!r}")
    return token


def platform_architecture(machine: Mapping[str, Any]) -> str:
    platform_name = slug(machine.get("platform") or "unknown", maximum=32)
    if platform_name == "darwin":
        platform_name = "macos"
    architecture = slug(machine.get("machine") or "unknown", maximum=32)
    return f"{platform_name}-{architecture}"


def submission_identity(
    *,
    machine: Mapping[str, Any],
    source_commit: str,
    submission_date: str,
    run_fingerprints: Mapping[str, str],
    power_mode_summary: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    machine_id = slug(machine.get("machine_id") or "anonymous", maximum=64)
    cpu = slug(machine.get("cpu_model") or "unknown-cpu", maximum=48)
    host_settings = machine.get("host_settings")
    power_mode_tag = "power-mode-unreported"
    if power_mode_summary:
        power_mode_tag = slug(
            power_mode_summary.get("contribution_tag") or power_mode_tag,
            maximum=48,
        )
    elif isinstance(host_settings, Mapping):
        power_mode_tag = slug(
            host_settings.get("power_mode_tag") or power_mode_tag,
            maximum=48,
        )
    identity = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "source_commit": source_commit,
        "submission_date": submission_date,
        "machine": dict(machine),
        "run_fingerprints": dict(sorted(run_fingerprints.items())),
        "power_mode_summary": dict(power_mode_summary or {}),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    submission_id = (
        f"{machine_id}--{cpu}--{power_mode_tag}--{submission_date}--{digest}"
    )
    return platform_architecture(machine), submission_id


def _contained(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SubmissionError(f"artifact path escapes its bundle: {relative}") from exc
    return candidate


def _is_file(path: Path) -> bool:
    return os.path.isfile(native_path(path))


def _is_symlink(path: Path) -> bool:
    return os.path.islink(native_path(path))


def _file_size(path: Path) -> int:
    return os.stat(native_path(path)).st_size


def _relative_file_inventory(root: Path) -> set[str]:
    native_root = native_path(root)
    inventory: set[str] = set()
    for directory, _, filenames in os.walk(native_root):
        for filename in filenames:
            absolute = os.path.join(directory, filename)
            inventory.add(os.path.relpath(absolute, native_root).replace("\\", "/"))
    return inventory


def _child_directories(path: Path) -> list[Path]:
    with os.scandir(native_path(path)) as entries:
        return sorted(
            (
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            ),
            key=lambda item: item.name,
        )


def _relative_display(path: Path, root: Path) -> str:
    return os.path.relpath(native_path(path), native_path(root)).replace("\\", "/")


def validate_report_manifest(report_dir: Path) -> dict[str, Any]:
    manifest_path = report_dir / "report_manifest.json"
    try:
        manifest = json.loads(read_text(manifest_path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"invalid report manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise SubmissionError(f"unsupported report manifest: {manifest_path}")
    if not manifest.get("publication_attested"):
        raise SubmissionError(f"report is not publication-attested: {report_dir}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SubmissionError(f"report has no manifested artifacts: {report_dir}")

    seen: set[str] = set()
    for raw_entry in artifacts:
        if not isinstance(raw_entry, dict):
            raise SubmissionError(f"invalid artifact entry in {manifest_path}")
        relative = str(raw_entry.get("path") or "")
        if (
            relative in seen
            or Path(relative).suffix.casefold() not in PUBLICATION_SUFFIXES
        ):
            raise SubmissionError(f"invalid or duplicate report artifact: {relative!r}")
        seen.add(relative)
        artifact = _contained(report_dir, relative)
        if not _is_file(artifact) or _is_symlink(artifact):
            raise SubmissionError(f"report artifact is missing or linked: {relative}")
        if raw_entry.get("bytes") != _file_size(artifact):
            raise SubmissionError(f"report artifact size mismatch: {relative}")
        if raw_entry.get("sha256") != sha256_file(artifact):
            raise SubmissionError(f"report artifact checksum mismatch: {relative}")
    if manifest.get("artifact_count") != len(seen):
        raise SubmissionError(f"report artifact count mismatch: {report_dir}")
    return manifest


def _git_source_commit(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise SubmissionError(
            "source repository must be clean before packaging results"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise SubmissionError("could not resolve a valid source commit")
    return commit


def _load_run_identity(
    run_dir: Path,
    *,
    machine_id: str,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        run_spec = json.loads(read_text(run_dir / "run_spec.json"))
        run_meta = json.loads(read_text(run_dir / "run_meta.json"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"invalid run metadata: {run_dir}") from exc
    if not isinstance(run_spec, dict) or not isinstance(run_meta, dict):
        raise SubmissionError(f"run metadata must contain JSON objects: {run_dir}")
    run_fingerprint = fingerprint(run_spec_identity(run_spec))
    if run_meta.get("run_fingerprint") != run_fingerprint:
        raise SubmissionError(f"run fingerprint mismatch: {run_dir}")
    if run_meta.get("git_commit") != source_commit:
        raise SubmissionError(
            f"run was not executed from source commit {source_commit}"
        )
    machine = run_spec.get("benchmark_machine")
    if not isinstance(machine, dict) or machine.get("machine_id") != machine_id:
        raise SubmissionError(f"run machine identity mismatch: {run_dir}")
    return run_spec, run_meta, run_fingerprint


def package_submission(
    *,
    repository: Path,
    result_root: Path,
    output_root: Path,
    machine_id: str,
    submission_date: str | None = None,
) -> Path:
    """Generate reports from completed ledgers and create one Git-ready bundle."""

    repository = repository.resolve()
    result_root = result_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source_commit = _git_source_commit(repository)
    resolved_date = submission_date or date.today().isoformat()
    try:
        date.fromisoformat(resolved_date)
    except ValueError as exc:
        raise SubmissionError("submission date must use YYYY-MM-DD") from exc

    machine_root = result_root / machine_id
    run_specs: dict[str, dict[str, Any]] = {}
    run_metas: dict[str, dict[str, Any]] = {}
    run_fingerprints: dict[str, str] = {}
    machine: dict[str, Any] | None = None
    for pillar in PILLARS:
        run_spec, run_meta, run_fingerprint = _load_run_identity(
            machine_root / pillar,
            machine_id=machine_id,
            source_commit=source_commit,
        )
        observed_machine = run_spec["benchmark_machine"]
        if machine is None:
            machine = dict(observed_machine)
        elif machine != observed_machine:
            raise SubmissionError("pillar runs contain different machine identities")
        run_specs[pillar] = run_spec
        run_metas[pillar] = run_meta
        run_fingerprints[pillar] = run_fingerprint
    assert machine is not None
    power_mode_summary = combine_power_summaries(
        [
            dict(run_meta.get("power_mode_summary") or {})
            for run_meta in run_metas.values()
        ]
    )

    architecture, submission_id = submission_identity(
        machine=machine,
        source_commit=source_commit,
        submission_date=resolved_date,
        run_fingerprints=run_fingerprints,
        power_mode_summary=power_mode_summary,
    )
    destination = output_root / architecture / submission_id
    if destination.exists():
        raise SubmissionError(f"submission already exists: {destination}")
    # Do not repeat the potentially long publication ID in the staging name.
    # The final destination is path-budgeted by the launcher, but an expanded
    # temporary name can exceed legacy MAX_PATH on otherwise valid Windows hosts.
    temporary = destination.with_name(
        f".tmp-{os.getpid()}-{hashlib.sha256(submission_id.encode()).hexdigest()[:8]}"
    )
    if temporary.exists():
        raise SubmissionError(f"temporary submission path already exists: {temporary}")

    from bench.report import generate_report

    files: list[dict[str, Any]] = []
    pillars: dict[str, Any] = {}
    try:
        temporary.mkdir(parents=True)
        for pillar in PILLARS:
            run_dir = machine_root / pillar
            report_dir = run_dir / "publication-report"
            generate_report(run_dir, report_dir)
            report_manifest = validate_report_manifest(report_dir)
            pillar_destination = temporary / pillar
            pillar_destination.mkdir()
            report_paths = ["report_manifest.json"] + [
                str(entry["path"]) for entry in report_manifest["artifacts"]
            ]
            for relative in report_paths:
                source = _contained(report_dir, relative)
                target = pillar_destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                bundle_relative = target.relative_to(temporary).as_posix()
                files.append(
                    {
                        "path": bundle_relative,
                        "bytes": _file_size(target),
                        "sha256": sha256_file(target),
                    }
                )
            pillars[pillar] = {
                "run_id": run_specs[pillar].get("run_id"),
                "run_fingerprint": run_fingerprints[pillar],
                "run_status": run_metas[pillar].get("run_status"),
                "report_path": f"{pillar}/report_manifest.json",
                "report_manifest_sha256": sha256_file(
                    pillar_destination / "report_manifest.json"
                ),
                "artifact_count": report_manifest["artifact_count"],
            }

        submission_manifest = {
            "schema_version": SUBMISSION_SCHEMA_VERSION,
            "submission_id": submission_id,
            "submission_date": resolved_date,
            "source_commit": source_commit,
            "architecture": architecture,
            "machine": machine,
            "power_mode_summary": power_mode_summary,
            "pillars": pillars,
            "file_count": len(files),
            "files": sorted(files, key=lambda item: str(item["path"])),
            "manifest_self_excluded": True,
        }
        atomic_write_json(temporary / "submission_manifest.json", submission_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(native_path(temporary), native_path(destination))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_submission_bundle(destination)
    return destination


def validate_submission_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    try:
        manifest = json.loads(read_text(bundle / "submission_manifest.json"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"invalid submission manifest: {bundle}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise SubmissionError(f"unsupported submission manifest: {bundle}")
    if manifest.get("submission_id") != bundle.name:
        raise SubmissionError(f"submission directory/manifest ID mismatch: {bundle}")
    if manifest.get("architecture") != bundle.parent.name:
        raise SubmissionError(f"submission architecture directory mismatch: {bundle}")
    if not SLUG_PATTERN.fullmatch(bundle.parent.name):
        raise SubmissionError(f"invalid architecture directory: {bundle.parent.name}")
    pillars = manifest.get("pillars")
    if not isinstance(pillars, dict) or set(pillars) != set(PILLARS):
        raise SubmissionError(
            f"submission must contain all benchmark pillars: {bundle}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count") != len(files):
        raise SubmissionError(f"submission file inventory is invalid: {bundle}")

    seen: set[str] = set()
    for raw_entry in files:
        if not isinstance(raw_entry, dict):
            raise SubmissionError(f"invalid submission file entry: {bundle}")
        relative = str(raw_entry.get("path") or "")
        if (
            relative in seen
            or Path(relative).suffix.casefold() not in PUBLICATION_SUFFIXES
        ):
            raise SubmissionError(f"invalid or duplicate submission file: {relative!r}")
        seen.add(relative)
        path = _contained(bundle, relative)
        if not _is_file(path) or _is_symlink(path):
            raise SubmissionError(f"submission file is missing or linked: {relative}")
        if raw_entry.get("bytes") != _file_size(path):
            raise SubmissionError(f"submission file size mismatch: {relative}")
        if raw_entry.get("sha256") != sha256_file(path):
            raise SubmissionError(f"submission file checksum mismatch: {relative}")

    actual = _relative_file_inventory(bundle)
    expected = seen | {"submission_manifest.json"}
    if actual != expected:
        raise SubmissionError(
            f"submission has unmanifested or missing files: {sorted(actual ^ expected)}"
        )
    for pillar, raw_pillar in pillars.items():
        if not isinstance(raw_pillar, dict):
            raise SubmissionError(f"invalid pillar metadata: {pillar}")
        report_path = str(raw_pillar.get("report_path") or "")
        report_manifest = _contained(bundle, report_path)
        if raw_pillar.get("report_manifest_sha256") != sha256_file(report_manifest):
            raise SubmissionError(f"pillar report checksum mismatch: {pillar}")
        validate_report_manifest(report_manifest.parent)
    return manifest


def validate_submission_tree(root: Path) -> list[str]:
    if not root.exists():
        return [f"submission root is missing: {root}"]
    failures: list[str] = []
    architectures = _child_directories(root)
    bundles = [
        bundle
        for architecture in architectures
        for bundle in _child_directories(architecture)
    ]
    manifests = [
        bundle / "submission_manifest.json"
        for bundle in bundles
        if _is_file(bundle / "submission_manifest.json")
    ]
    for manifest in manifests:
        try:
            validate_submission_bundle(manifest.parent)
        except SubmissionError as exc:
            failures.append(str(exc))
    with os.scandir(native_path(root)) as entries:
        unexpected = [
            entry.name
            for entry in entries
            if entry.name != "README.md"
            and (
                not entry.is_dir(follow_symlinks=False)
                or not SLUG_PATTERN.fullmatch(entry.name)
            )
        ]
    failures.extend(f"invalid item in submission root: {path}" for path in unexpected)
    manifested_bundles = {
        os.path.normcase(native_path(manifest.parent)) for manifest in manifests
    }
    for bundle in bundles:
        if os.path.normcase(native_path(bundle)) not in manifested_bundles:
            failures.append(
                "submission directory has no manifest: "
                f"{_relative_display(bundle, root)}"
            )
    return failures

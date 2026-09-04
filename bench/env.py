"""Create and verify one native Python environment per radiomics adapter."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROFILE_SCHEMA_VERSION = 1


class EnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvironmentLock:
    """One fully resolved package freeze for a specific native platform."""

    platform_key: str
    path: str
    sha256: str
    freeze_sha256: str


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    distribution: str
    version: str
    python: str
    requirement: str
    env_dir: str
    smoke_imports: tuple[str, ...]
    smoke_entrypoints: tuple[str, ...] = ()
    extra_requirements: tuple[str, ...] = ()
    pip_args: tuple[str, ...] = ()
    upstream: str = ""
    verified_latest_stable: str = ""
    source_commit: str = ""
    metadata_version: str = ""
    notes: str = ""
    environment_locks: tuple[EnvironmentLock, ...] = ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def profiles_dir() -> Path:
    return repo_root() / "configs" / "adapters"


def _profile_from_mapping(path: Path, raw: dict[str, Any]) -> RuntimeProfile:
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise EnvironmentError(f"Unsupported profile schema in {path}")
    name = str(raw.get("name") or path.stem)
    required = ("distribution", "version", "python")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise EnvironmentError(f"Profile {path} is missing: {', '.join(missing)}")
    raw_overrides = raw.get("platform_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise EnvironmentError(f"Profile {path} platform_overrides must be a mapping")
    override = raw_overrides.get(native_platform_key(), {})
    if not isinstance(override, Mapping):
        raise EnvironmentError(
            f"Profile {path} override for {native_platform_key()} must be a mapping"
        )
    allowed_override_fields = {"python", "requirement", "metadata_version"}
    unknown_override_fields = sorted(set(override) - allowed_override_fields)
    if unknown_override_fields:
        raise EnvironmentError(
            f"Profile {path} override for {native_platform_key()} has unsupported "
            f"fields: {', '.join(unknown_override_fields)}"
        )
    distribution = str(raw["distribution"])
    version = str(raw["version"])
    raw_locks = raw.get("environment_locks", {})
    if not isinstance(raw_locks, Mapping):
        raise EnvironmentError(f"Profile {path} environment_locks must be a mapping")
    locks: list[EnvironmentLock] = []
    for platform_key, lock in sorted(raw_locks.items()):
        if not isinstance(lock, Mapping):
            raise EnvironmentError(
                f"Profile {path} lock {platform_key!r} must be a mapping"
            )
        values = {
            key: str(lock.get(key) or "") for key in ("path", "sha256", "freeze_sha256")
        }
        missing_lock = [key for key, value in values.items() if not value]
        if missing_lock:
            raise EnvironmentError(
                f"Profile {path} lock {platform_key!r} is missing: "
                + ", ".join(missing_lock)
            )
        locks.append(
            EnvironmentLock(
                platform_key=str(platform_key),
                path=values["path"],
                sha256=values["sha256"].lower(),
                freeze_sha256=values["freeze_sha256"].lower(),
            )
        )
    return RuntimeProfile(
        name=name,
        distribution=distribution,
        version=version,
        python=str(override.get("python") or raw["python"]),
        requirement=str(
            override.get("requirement")
            or raw.get("requirement")
            or f"{distribution}=={version}"
        ),
        env_dir=str(raw.get("env_dir") or f".venvs/adapters/{name}"),
        smoke_imports=tuple(str(item) for item in raw.get("smoke_imports", ())),
        smoke_entrypoints=tuple(str(item) for item in raw.get("smoke_entrypoints", ())),
        extra_requirements=tuple(
            str(item) for item in raw.get("extra_requirements", ())
        ),
        pip_args=tuple(str(item) for item in raw.get("pip_args", ())),
        upstream=str(raw.get("upstream") or ""),
        verified_latest_stable=str(raw.get("verified_latest_stable") or ""),
        source_commit=str(raw.get("source_commit") or ""),
        metadata_version=str(
            override.get("metadata_version") or raw.get("metadata_version") or ""
        ),
        notes=str(raw.get("notes") or ""),
        environment_locks=tuple(locks),
    )


def load_runtime_profiles(config_dir: Path | None = None) -> dict[str, RuntimeProfile]:
    directory = config_dir or profiles_dir()
    if not directory.is_dir():
        raise EnvironmentError(f"Adapter profile directory not found: {directory}")
    profiles: dict[str, RuntimeProfile] = {}
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        profile = _profile_from_mapping(path, raw)
        if profile.name in profiles:
            raise EnvironmentError(f"Duplicate adapter profile: {profile.name}")
        profiles[profile.name] = profile
    if not profiles:
        raise EnvironmentError(f"No adapter profiles found under {directory}")
    return profiles


def env_dir_for_profile(profile: RuntimeProfile) -> Path:
    base = (repo_root() / ".venvs" / "adapters").resolve()
    target = (repo_root() / profile.env_dir).resolve()
    try:
        relative = target.relative_to(base)
    except ValueError as exc:
        raise EnvironmentError(
            f"Adapter environment must be contained under {base}: {target}"
        ) from exc
    if not relative.parts:
        raise EnvironmentError(
            f"Adapter environment cannot be the shared root: {target}"
        )
    return target


def env_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _python_version(executable: Path) -> str | None:
    try:
        proc = subprocess.run(
            [
                str(executable),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def _major_minor(version: str | None) -> str | None:
    if not version:
        return None
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return ".".join(parts[:2])


def _find_python(requested: str) -> Path:
    if _major_minor(sys.version.split()[0]) == _major_minor(requested):
        return Path(sys.executable).resolve()

    uv = shutil.which("uv")
    if uv:
        proc = subprocess.run(
            [uv, "python", "find", requested],
            capture_output=True,
            text=True,
            check=False,
        )
        candidate = (
            Path(proc.stdout.strip())
            if proc.returncode == 0 and proc.stdout.strip()
            else None
        )
        if candidate and candidate.exists():
            return candidate.resolve()

    candidates = [f"python{requested}", f"python{_major_minor(requested)}"]
    for command in candidates:
        resolved = shutil.which(command)
        if not resolved:
            continue
        executable = Path(resolved).resolve()
        if _major_minor(_python_version(executable)) == _major_minor(requested):
            return executable

    raise EnvironmentError(
        f"Python {requested} is required. Install it (or let uv manage it) before creating this adapter environment."
    )


def _find_uv() -> str | None:
    """Find uv even when a Windows user-script directory is not on PATH."""

    executable = shutil.which("uv")
    if executable:
        return executable
    if os.name != "nt":
        return None
    version_directory = f"Python{sys.version_info.major}{sys.version_info.minor}"
    candidate = Path(site.getuserbase()) / version_directory / "Scripts" / "uv.exe"
    if candidate.is_file():
        return str(candidate.resolve())
    # Python's user base already includes the version on conventional installs.
    candidate = Path(site.getuserbase()) / "Scripts" / "uv.exe"
    return str(candidate.resolve()) if candidate.is_file() else None


def _profile_fingerprint(profile: RuntimeProfile) -> str:
    encoded = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def native_platform_key() -> str:
    """Return the stable key used by checked-in native environment locks."""

    return f"{platform.system()}-{platform.machine()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_environment_lock(
    profile: RuntimeProfile,
) -> tuple[EnvironmentLock, Path, list[str]] | None:
    matches = [
        lock
        for lock in profile.environment_locks
        if lock.platform_key == native_platform_key()
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise EnvironmentError(
            f"Profile {profile.name} defines duplicate locks for {native_platform_key()}"
        )
    lock = matches[0]
    root = repo_root().resolve()
    path = (root / lock.path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EnvironmentError(
            f"Environment lock must remain inside the repository: {path}"
        ) from exc
    if not path.is_file():
        raise EnvironmentError(
            f"Environment lock is missing for {profile.name}: {path}"
        )
    if _file_sha256(path) != lock.sha256:
        raise EnvironmentError(f"Environment lock byte checksum failed: {path}")
    requirements = sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not requirements:
        raise EnvironmentError(f"Environment lock is empty: {path}")
    semantic_hash = hashlib.sha256("\n".join(requirements).encode("utf-8")).hexdigest()
    if semantic_hash != lock.freeze_sha256:
        raise EnvironmentError(f"Environment lock semantic checksum failed: {path}")
    return lock, path, requirements


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".tmp-", suffix=path.suffix, dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=True, capture_output=False)


def _create_venv(profile: RuntimeProfile, target: Path) -> None:
    interpreter = _find_python(profile.python)
    uv = _find_uv()
    if uv:
        # Verification intentionally uses `python -m pip check/freeze`; seed the
        # uv environment so those commands exist in every supported runtime.
        _run(
            [
                uv,
                "venv",
                "--no-project",
                "--seed",
                "--python",
                str(interpreter),
                str(target),
            ]
        )
    else:
        _run([str(interpreter), "-m", "venv", str(target)])


def _install(profile: RuntimeProfile, target: Path) -> None:
    python = env_python(target)
    resolved_lock = _resolve_environment_lock(profile)
    requirements = [profile.requirement, *profile.extra_requirements]
    uv = _find_uv()
    if uv:
        command = [uv, "pip", "install", "--python", str(python), *profile.pip_args]
        if resolved_lock is not None:
            command.extend(["--requirement", str(resolved_lock[1])])
            if "#sha256=" in profile.requirement.casefold():
                command.append(profile.requirement)
        else:
            command.extend(requirements)
        _run(command)
    else:
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ]
        )
        command = [str(python), "-m", "pip", "install", *profile.pip_args]
        if resolved_lock is not None:
            command.extend(["--requirement", str(resolved_lock[1])])
            if "#sha256=" in profile.requirement.casefold():
                command.append(profile.requirement)
        else:
            command.extend(requirements)
        _run(command)


def _distribution_version(python: Path, distribution: str) -> str:
    code = (
        "import importlib.metadata as m, sys; sys.stdout.write(m.version(sys.argv[1]))"
    )
    proc = subprocess.run(
        [str(python), "-c", code, distribution],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _freeze(python: Path) -> list[str]:
    proc = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())


def _runtime_platform(python: Path) -> dict[str, str]:
    code = (
        "import json, platform; print(json.dumps({"
        "'implementation': platform.python_implementation(), "
        "'platform': platform.platform(), 'machine': platform.machine()}))"
    )
    proc = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    return {str(key): str(value) for key, value in payload.items()}


def _run_smoke_checks(profile: RuntimeProfile, python: Path) -> None:
    """Import declared modules and call calculation-free adapter probes."""

    for module in profile.smoke_imports:
        subprocess.run([str(python), "-c", f"import {module}"], check=True)

    if not profile.smoke_entrypoints:
        return
    probe = (
        "import importlib, sys\n"
        "for value in sys.argv[1:]:\n"
        "    module_name, separator, attribute = value.partition(':')\n"
        "    if not separator or not module_name or not attribute:\n"
        "        raise ValueError(f'invalid smoke entrypoint: {value!r}')\n"
        "    target = importlib.import_module(module_name)\n"
        "    callback = getattr(target, attribute)\n"
        "    if not callable(callback):\n"
        "        raise TypeError(f'smoke entrypoint is not callable: {value}')\n"
        "    callback()\n"
    )
    subprocess.run(
        [str(python), "-c", probe, *profile.smoke_entrypoints],
        cwd=str(repo_root()),
        check=True,
    )


def _verify_recorded_environment(
    recorded: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    """Fail when a reusable adapter environment has drifted from its freeze.

    The profile pins the primary distribution, but publication runs also depend
    on every transitive package and on the exact Python interpreter.  Treat the
    first successful ``environment.json`` as immutable build provenance instead
    of silently blessing the environment's current state on each verification.
    """

    if recorded.get("schema_version") != 1:
        raise EnvironmentError("Recorded adapter environment has an unsupported schema")

    stable_fields = (
        "profile",
        "python",
        "python_implementation",
        "distribution",
        "version",
    )
    changed = [
        field for field in stable_fields if recorded.get(field) != observed.get(field)
    ]
    if changed:
        raise EnvironmentError(
            "Adapter environment no longer matches its recorded build metadata "
            f"({', '.join(changed)} changed); rebuild it with --force."
        )

    if recorded.get("profile_fingerprint") != observed.get("profile_fingerprint"):
        recorded_definition = recorded.get("profile_definition")
        observed_definition = observed.get("profile_definition")
        if not isinstance(recorded_definition, Mapping) or not isinstance(
            observed_definition, Mapping
        ):
            raise EnvironmentError(
                "Adapter environment profile fingerprint changed; rebuild it with --force."
            )
        changed_keys = sorted(
            key
            for key in set(recorded_definition) | set(observed_definition)
            if recorded_definition.get(key) != observed_definition.get(key)
        )
        raise EnvironmentError(
            "Adapter environment profile changed "
            f"({', '.join(changed_keys) or 'definition fingerprint'}); "
            "rebuild it with --force."
        )

    recorded_metadata_version = recorded.get("distribution_metadata_version")
    if recorded_metadata_version != observed.get("distribution_metadata_version"):
        raise EnvironmentError(
            "Adapter distribution metadata version changed; rebuild it with --force."
        )

    recorded_freeze = recorded.get("freeze")
    if not isinstance(recorded_freeze, list) or not all(
        isinstance(item, str) and item.strip() for item in recorded_freeze
    ):
        raise EnvironmentError(
            "Recorded adapter environment freeze is missing or invalid"
        )
    canonical_recorded_freeze = sorted(recorded_freeze)
    recorded_freeze_sha256 = hashlib.sha256(
        "\n".join(canonical_recorded_freeze).encode("utf-8")
    ).hexdigest()
    if recorded.get("freeze_sha256") != recorded_freeze_sha256:
        raise EnvironmentError(
            "Recorded adapter environment freeze checksum is invalid"
        )
    if canonical_recorded_freeze != observed.get("freeze"):
        raise EnvironmentError(
            "Adapter environment package freeze has drifted; rebuild it with --force."
        )
    if recorded_freeze_sha256 != observed.get("freeze_sha256"):
        raise EnvironmentError(
            "Adapter environment package freeze checksum has drifted; rebuild it with --force."
        )
    recorded_lock = recorded.get("environment_lock")
    if recorded_lock is not None and recorded_lock != observed.get("environment_lock"):
        raise EnvironmentError(
            "Adapter environment lock provenance changed; rebuild it with --force."
        )


def verify_profile(
    profile: RuntimeProfile,
    *,
    smoke: bool = True,
    verify_recorded: bool = True,
) -> dict[str, Any]:
    target = env_dir_for_profile(profile)
    python = env_python(target)
    if not python.exists():
        raise EnvironmentError(f"Environment is missing for {profile.name}: {target}")
    actual_python = _python_version(python)
    if _major_minor(actual_python) != _major_minor(profile.python):
        raise EnvironmentError(
            f"{profile.name} uses Python {actual_python}; expected Python {profile.python}"
        )
    actual_version = _distribution_version(python, profile.distribution)
    expected_metadata_version = profile.metadata_version or profile.version
    if actual_version != expected_metadata_version:
        raise EnvironmentError(
            f"{profile.name} has {profile.distribution} metadata version {actual_version}; "
            f"expected {expected_metadata_version} for release {profile.version}"
        )
    pip_check = subprocess.run(
        [str(python), "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    )
    if smoke:
        _run_smoke_checks(profile, python)
    freeze = _freeze(python)
    freeze_sha256 = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()
    resolved_lock = _resolve_environment_lock(profile)
    environment_lock = None
    if resolved_lock is not None:
        lock, lock_path, locked_freeze = resolved_lock
        if freeze != locked_freeze or freeze_sha256 != lock.freeze_sha256:
            raise EnvironmentError(
                f"{profile.name} package freeze does not match the checked-in "
                f"{lock.platform_key} lock; rebuild it with --force."
            )
        environment_lock = {
            "platform_key": lock.platform_key,
            "path": lock_path.relative_to(repo_root()).as_posix(),
            "sha256": lock.sha256,
            "freeze_sha256": lock.freeze_sha256,
        }
    runtime_platform = _runtime_platform(python)
    verified = {
        "schema_version": 1,
        "profile": profile.name,
        "profile_definition": asdict(profile),
        "profile_fingerprint": _profile_fingerprint(profile),
        "python_executable": str(python.resolve()),
        "python": actual_python,
        "python_implementation": runtime_platform["implementation"],
        "platform": runtime_platform["platform"],
        "machine": runtime_platform["machine"],
        "distribution": profile.distribution,
        "version": profile.version,
        "distribution_metadata_version": actual_version,
        "pip_check": pip_check.stdout.strip() or "No broken requirements found.",
        "freeze": freeze,
        "freeze_sha256": freeze_sha256,
        "environment_lock": environment_lock,
    }
    metadata_path = target / "environment.json"
    if verify_recorded:
        if not metadata_path.is_file():
            raise EnvironmentError(
                f"Recorded environment provenance is missing for {profile.name}: "
                f"{metadata_path}; rebuild it with --force."
            )
        try:
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentError(
                f"Recorded environment provenance is unreadable for {profile.name}: "
                f"{metadata_path}"
            ) from exc
        if not isinstance(recorded, Mapping):
            raise EnvironmentError(
                f"Recorded environment provenance is invalid for {profile.name}: "
                f"{metadata_path}"
            )
        _verify_recorded_environment(recorded, verified)
    return verified


def create_profile(profile: RuntimeProfile, *, force: bool = False) -> Path:
    target = env_dir_for_profile(profile)
    metadata_path = target / "environment.json"
    if target.exists() and not force:
        if not metadata_path.is_file():
            raise EnvironmentError(
                f"Existing {profile.name} environment has no immutable environment.json; "
                "rebuild it with --force."
            )
        verify_profile(profile)
        return target

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _create_venv(profile, target)
    try:
        _install(profile, target)
        verified = verify_profile(profile, verify_recorded=False)
        _atomic_json(metadata_path, verified)
    except BaseException:
        # A partial environment is not resume-safe.  Keep a marker for diagnosis;
        # the next --force invocation will rebuild it from scratch.
        _atomic_json(target / "INSTALL_FAILED.json", {"profile": profile.name})
        raise
    return target


def create_profiles(names: list[str] | None = None, *, force: bool = False) -> None:
    profiles = load_runtime_profiles()
    selected = names or list(profiles)
    unknown = sorted(set(selected).difference(profiles))
    if unknown:
        raise EnvironmentError(f"Unknown adapter profiles: {', '.join(unknown)}")
    for name in selected:
        print(f"Creating isolated adapter environment: {name}")
        create_profile(profiles[name], force=force)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="bench env")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    create = subparsers.add_parser("create")
    create.add_argument("--profiles", nargs="*")
    create.add_argument("--force", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--profiles", nargs="*")
    args = parser.parse_args(argv)

    profiles = load_runtime_profiles()
    if args.command == "list":
        for profile in profiles.values():
            target = env_dir_for_profile(profile)
            status = "ready" if env_python(target).exists() else "missing"
            print(
                f"{profile.name}: {profile.distribution}=={profile.version}, "
                f"Python {profile.python}, {status}, path={target}"
            )
        return 0
    if args.command == "create":
        create_profiles(args.profiles, force=args.force)
        return 0
    if args.command == "verify":
        selected = args.profiles or list(profiles)
        unknown = sorted(set(selected).difference(profiles))
        if unknown:
            raise EnvironmentError(f"Unknown adapter profiles: {', '.join(unknown)}")
        for name in selected:
            print(json.dumps(verify_profile(profiles[name]), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

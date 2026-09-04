from __future__ import annotations

import copy
import hashlib
import unittest
from unittest.mock import patch

from bench import env


class EnvironmentProfileTests(unittest.TestCase):
    def test_latest_stable_adapter_pins_are_exact(self) -> None:
        profiles = env.load_runtime_profiles()
        self.assertEqual(
            {name: profile.version for name, profile in profiles.items()},
            {
                "medimage": "0.9.8",
                "mirp": "2.7.0",
                "pictologics": "0.5.1",
                "pyradiomics": "3.1.0",
                "zrad": "26.8.0",
            },
        )
        for profile in profiles.values():
            self.assertTrue(profile.verified_latest_stable)
            self.assertTrue(profile.upstream.startswith("https://"))

    def test_every_environment_is_isolated_below_adapter_root(self) -> None:
        base = (env.repo_root() / ".venvs" / "adapters").resolve()
        paths = []
        for profile in env.load_runtime_profiles().values():
            target = env.env_dir_for_profile(profile)
            target.relative_to(base)
            paths.append(target)
        self.assertEqual(len(paths), len(set(paths)))

    def test_escaping_environment_path_is_rejected(self) -> None:
        profile = env.RuntimeProfile(
            name="bad",
            distribution="bad",
            version="1",
            python="3.12",
            requirement="bad==1",
            env_dir="../outside",
            smoke_imports=(),
        )
        with self.assertRaises(env.EnvironmentError):
            env.env_dir_for_profile(profile)

    def test_known_compatibility_constraints_are_explicit(self) -> None:
        profiles = env.load_runtime_profiles()
        self.assertIn("numpy==1.26.4", profiles["medimage"].extra_requirements)
        self.assertIn("pandas==1.5.3", profiles["medimage"].extra_requirements)
        self.assertEqual(
            profiles["medimage"].smoke_entrypoints,
            ("bench.adapters.medimage_adapter:_medimage_modules",),
        )
        if env.native_platform_key() == "Windows-AMD64":
            self.assertEqual(profiles["pyradiomics"].python, "3.9")
            self.assertIn(
                "pyradiomics-3.1.0-cp39-cp39-win_amd64.whl",
                profiles["pyradiomics"].requirement,
            )
            self.assertIn("sha256=1411f192", profiles["pyradiomics"].requirement)
            self.assertEqual(profiles["pyradiomics"].metadata_version, "3.1.0")
        else:
            self.assertIn("6a761c4", profiles["pyradiomics"].requirement)
        self.assertEqual(profiles["pyradiomics"].source_commit, "6a761c4")

    def test_pyradiomics_platform_override_leaves_other_platforms_unchanged(
        self,
    ) -> None:
        with patch("bench.env.native_platform_key", return_value="Darwin-arm64"):
            profile = env.load_runtime_profiles()["pyradiomics"]

        self.assertEqual(profile.python, "3.11")
        self.assertIn(
            "git+https://github.com/AIM-Harvard/pyradiomics.git@6a761c4",
            profile.requirement,
        )
        self.assertEqual(profile.metadata_version, "3.0.1a1")

    def test_smoke_entrypoint_imports_and_calls_declared_probe(self) -> None:
        profile = env.RuntimeProfile(
            name="adapter",
            distribution="adapter",
            version="1",
            python="3.12",
            requirement="adapter==1",
            env_dir=".venvs/adapters/adapter",
            smoke_imports=(),
            smoke_entrypoints=("package.module:probe",),
        )
        python = env.repo_root() / ".venvs/adapters/adapter/bin/python"

        with patch("bench.env.subprocess.run") as run:
            env._run_smoke_checks(profile, python)

        command = run.call_args.args[0]
        self.assertEqual(command[:3], [str(python), "-c", unittest.mock.ANY])
        self.assertEqual(command[3:], ["package.module:probe"])
        self.assertEqual(run.call_args.kwargs["cwd"], str(env.repo_root()))
        self.assertTrue(run.call_args.kwargs["check"])

    def test_native_macos_locks_bind_the_complete_observed_freezes(self) -> None:
        expected_counts = {
            "medimage": 235,
            "mirp": 29,
            "pictologics": 26,
            "pyradiomics": 13,
            "zrad": 21,
        }
        for name, profile in env.load_runtime_profiles().items():
            lock = next(
                item
                for item in profile.environment_locks
                if item.platform_key == "Darwin-arm64"
            )
            path = env.repo_root() / lock.path
            requirements = sorted(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
            self.assertEqual(lock.platform_key, "Darwin-arm64")
            self.assertEqual(path.name, f"{name}.txt")
            self.assertEqual(len(requirements), expected_counts[name])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), lock.sha256)
            self.assertEqual(
                hashlib.sha256("\n".join(requirements).encode()).hexdigest(),
                lock.freeze_sha256,
            )

    @unittest.skipUnless(
        env.native_platform_key() == "Windows-AMD64", "Windows lock test"
    )
    def test_native_windows_locks_bind_the_complete_observed_freezes(self) -> None:
        expected_counts = {
            "medimage": 236,
            "mirp": 30,
            "pictologics": 28,
            "pyradiomics": 13,
            "zrad": 22,
        }
        for name, profile in env.load_runtime_profiles().items():
            with self.subTest(profile=name):
                lock = next(
                    item
                    for item in profile.environment_locks
                    if item.platform_key == "Windows-AMD64"
                )
                path = env.repo_root() / lock.path
                requirements = sorted(
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
                self.assertEqual(path.name, f"{name}.txt")
                self.assertEqual(len(requirements), expected_counts[name])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), lock.sha256
                )
                self.assertEqual(
                    hashlib.sha256("\n".join(requirements).encode()).hexdigest(),
                    lock.freeze_sha256,
                )

    def test_hashed_primary_artifact_is_enforced_when_installing_lock(self) -> None:
        requirement = "adapter @ https://example.invalid/adapter.whl#sha256=abc"
        profile = env.RuntimeProfile(
            name="adapter",
            distribution="adapter",
            version="1",
            python="3.12",
            requirement=requirement,
            env_dir=".venvs/adapters/adapter",
            smoke_imports=(),
        )
        lock = env.EnvironmentLock("Windows-AMD64", "lock.txt", "a", "b")
        lock_path = env.repo_root() / "lock.txt"

        with (
            patch("bench.env._find_uv", return_value="uv"),
            patch(
                "bench.env._resolve_environment_lock",
                return_value=(lock, lock_path, ["adapter==1"]),
            ),
            patch("bench.env._run") as run,
        ):
            env._install(profile, env.repo_root() / ".venvs/adapters/adapter")

        command = run.call_args.args[0]
        self.assertIn(requirement, command)
        self.assertEqual(command[-3:-1], ["--requirement", str(lock_path)])

    def test_recorded_environment_rejects_transitive_package_drift(self) -> None:
        freeze = ["adapter==1.0", "dependency==2.0"]
        freeze_sha256 = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()
        recorded = {
            "schema_version": 1,
            "profile": "adapter",
            "profile_fingerprint": "profile-hash",
            "python": "3.12.1",
            "python_implementation": "CPython",
            "distribution": "adapter",
            "version": "1.0",
            "distribution_metadata_version": "1.0",
            "freeze": freeze,
            "freeze_sha256": freeze_sha256,
        }
        observed = copy.deepcopy(recorded)
        observed["freeze"] = ["adapter==1.0", "dependency==2.1"]
        observed["freeze_sha256"] = hashlib.sha256(
            "\n".join(observed["freeze"]).encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(env.EnvironmentError, "freeze has drifted"):
            env._verify_recorded_environment(recorded, observed)

    def test_recorded_environment_rejects_tampered_freeze_checksum(self) -> None:
        recorded = {
            "schema_version": 1,
            "profile": "adapter",
            "profile_fingerprint": "profile-hash",
            "python": "3.12.1",
            "python_implementation": "CPython",
            "distribution": "adapter",
            "version": "1.0",
            "distribution_metadata_version": "1.0",
            "freeze": ["adapter==1.0"],
            "freeze_sha256": "not-the-freeze-hash",
        }

        with self.assertRaisesRegex(env.EnvironmentError, "checksum is invalid"):
            env._verify_recorded_environment(recorded, recorded)


if __name__ == "__main__":
    unittest.main()

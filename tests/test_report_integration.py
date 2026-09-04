from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


_REPORT_DEPENDENCIES = ("matplotlib", "numpy", "pandas")
_REPORT_DEPS_AVAILABLE = all(
    importlib.util.find_spec(package) is not None for package in _REPORT_DEPENDENCIES
)


@unittest.skipUnless(
    _REPORT_DEPS_AVAILABLE,
    "report integration tests require pandas, numpy, and matplotlib",
)
class ReportIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        import pandas as pd

        from bench import report

        self.pd = pd
        self.report = report

    def test_task_status_is_authoritative_and_censored_durations_are_excluded(
        self,
    ) -> None:
        frame = self.pd.DataFrame(
            [
                {
                    "row_id": "measured",
                    "task_status": "measured",
                    "success": False,
                    "duration_sec": 1.25,
                },
                {
                    "row_id": "timeout",
                    "task_status": "timed_out_censored",
                    "success": True,
                    "duration_sec": 999.0,
                    "censor_lower_bound_sec": 500.0,
                },
                {
                    "row_id": "skipped",
                    "task_status": "skipped_policy",
                    "success": True,
                    "duration_sec": 888.0,
                },
                {
                    "row_id": "failed",
                    "task_status": "failed",
                    "success": True,
                    "duration_sec": 777.0,
                },
                {
                    "row_id": "missing",
                    "task_status": "measured",
                    "success": True,
                    "duration_sec": None,
                },
                {
                    "row_id": "infinite",
                    "task_status": "measured",
                    "success": True,
                    "duration_sec": float("inf"),
                },
            ]
        )

        observations = self.report.timing_observations(frame)

        self.assertEqual(observations["row_id"].tolist(), ["measured"])
        self.assertEqual(observations["duration_sec"].tolist(), [1.25])

    def test_summary_csv_without_task_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "current-summary.csv"
            path.write_text(
                "adapter,family,duration_sec\npictologics,glcm,1.5\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "task_status"):
                self.report.load_timing_csv(path)

    def test_aggregation_never_pools_distinct_workloads(self) -> None:
        frame = self.pd.DataFrame(
            [
                self._timing_row("texture", "glcm", 1.0),
                self._timing_row("texture", "glcm", 3.0),
                self._timing_row(
                    "texture", "glcm", 900.0, status="timed_out_censored"
                ),
                self._timing_row("morphology", "glrlm", 100.0),
            ]
        )

        grouped = self.report.aggregate_timing_observations(frame).set_index("workload")

        self.assertEqual(set(grouped.index), {"texture", "morphology"})
        self.assertEqual(grouped.loc["texture", "duration_steady_sec"], 2.0)
        self.assertEqual(grouped.loc["texture", "timing_observations"], 2)
        self.assertEqual(grouped.loc["morphology", "duration_steady_sec"], 100.0)

    def test_missing_workload_is_not_silently_inferred(self) -> None:
        frame = self.pd.DataFrame(
            [
                self._timing_row(None, "glcm", 2.0),
                self._timing_row(None, "glrlm", 20.0),
            ]
        ).drop(columns=["workload"])

        with self.assertRaisesRegex(ValueError, "explicit workload"):
            self.report.aggregate_timing_observations(frame)

    def test_non_timing_outcomes_remain_diagnostic_only_and_stratified(self) -> None:
        frame = self.pd.DataFrame(
            [
                self._timing_row("texture", "glcm", 1.0),
                self._timing_row(
                    "texture", "glcm", 500.0, status="timed_out_censored"
                ),
                self._timing_row(
                    "morphology", "glrlm", 600.0, status="skipped_policy"
                ),
            ]
        )

        outcomes = self.report.non_timing_outcomes(frame)

        self.assertEqual(
            set(outcomes["_report_workload"]), {"texture", "morphology"}
        )
        self.assertEqual(
            set(outcomes["_report_status"]),
            {"timed_out_censored", "skipped_policy"},
        )

    def test_runtime_ratios_use_exact_measured_pairs_only(self) -> None:
        frame = self.pd.DataFrame(
            [
                self._comparison_row("case-a", "pictologics", "texture", 1, 2.0),
                self._comparison_row("case-a", "pyradiomics", "texture", 1, 6.0),
                self._comparison_row("case-b", "pictologics", "texture", 1, 4.0),
                self._comparison_row(
                    "case-b",
                    "pyradiomics",
                    "texture",
                    1,
                    100.0,
                    status="timed_out_censored",
                ),
                self._comparison_row("case-c", "mirp", "texture", 2, 3.0),
                self._comparison_row(
                    "case-a",
                    "medimage",
                    "texture",
                    1,
                    5.0,
                    status="failed",
                ),
            ]
        )

        matched = self.report.matched_runtime_observations(frame)
        summary = self.report.matched_runtime_summary(matched, dataset_kind="synthetic")

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.loc[0, "case_id"], "case-a")
        self.assertEqual(matched.loc[0, "adapter"], "pyradiomics")
        self.assertEqual(matched.loc[0, "runtime_ratio"], 3.0)
        self.assertEqual(summary.loc[0, "matched_n"], 1)
        self.assertEqual(summary.loc[0, "runtime_ratio_median"], 3.0)

    def test_runtime_ratio_duplicate_identities_are_rejected(self) -> None:
        duplicate = self._comparison_row("case-a", "pictologics", "texture", 1, 2.5)
        frame = self.pd.DataFrame(
            [
                self._comparison_row("case-a", "pictologics", "texture", 1, 2.0),
                duplicate,
                self._comparison_row("case-a", "pyradiomics", "texture", 1, 4.0),
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate measured rows"):
            self.report.matched_runtime_observations(frame)

    def test_ivh_remains_independent_from_intensity_in_matched_results(self) -> None:
        frame = self.pd.DataFrame(
            [
                self._comparison_row("case-a", "pictologics", "intensity", 1, 2.0),
                self._comparison_row("case-a", "pictologics", "ivh", 1, 3.0),
                self._comparison_row("case-a", "pyradiomics", "intensity", 1, 4.0),
                self._comparison_row(
                    "case-a",
                    "pyradiomics",
                    "ivh",
                    1,
                    0.0,
                    status="unsupported",
                ),
            ]
        )

        matched = self.report.matched_runtime_observations(frame)

        self.assertEqual(matched["workload"].tolist(), ["intensity"])
        self.assertEqual(matched["candidate_duration_sec"].tolist(), [4.0])
        self.assertEqual(matched["baseline_duration_sec"].tolist(), [2.0])

    def test_synthetic_summaries_do_not_pool_hu_profiles(self) -> None:
        rows = []
        for subject_id, scale in (("reference", 1.0), ("high_contrast", 2.0)):
            rows.extend(
                [
                    self._comparison_row(
                        f"case-{subject_id}",
                        adapter,
                        "texture",
                        1,
                        duration * scale,
                        subject_id=subject_id,
                    )
                    for adapter, duration in (
                        ("pictologics", 1.0),
                        ("pyradiomics", 3.0),
                    )
                ]
            )
        frame = self.pd.DataFrame(rows)

        absolute = self.report.aggregate_timing_observations(
            frame, dataset_kind="synthetic"
        )
        matched = self.report.matched_runtime_observations(frame)
        relative = self.report.matched_runtime_summary(
            matched, dataset_kind="synthetic"
        )

        self.assertEqual(len(absolute), 4)
        self.assertEqual(len(relative), 2)
        self.assertEqual(
            set(absolute["subject_id"]), {"reference", "high_contrast"}
        )
        self.assertEqual(
            set(relative["subject_id"]), {"reference", "high_contrast"}
        )
        self.assertEqual(set(relative["matched_n"]), {1})

    def test_real_world_summaries_use_modality_and_true_image_voxels(self) -> None:
        rows = [
            self._comparison_row(
                "ct-a",
                adapter,
                "texture",
                repeat,
                duration,
                modality="ct",
                image_voxels=512 * 512 * 73,
                shape=(512, 512, 73),
                spacing=(0.75, 0.75, 3.0),
                dataset_kind="real_world",
                subject_id="STS_001",
            )
            for adapter, repeat, duration in (
                ("pictologics", 1, 2.0),
                ("pyradiomics", 1, 4.0),
                ("pictologics", 2, 3.0),
                ("pyradiomics", 2, 6.0),
            )
        ]
        frame = self.pd.DataFrame(rows)

        absolute = self.report.aggregate_timing_observations(
            frame, dataset_kind="real_world"
        )
        matched = self.report.matched_runtime_observations(frame)
        relative = self.report.matched_runtime_summary(
            matched, dataset_kind="real_world"
        )

        for result in (absolute, relative):
            self.assertIn("modality", result.columns)
            self.assertIn("subject_id", result.columns)
            self.assertIn("image_voxels", result.columns)
            self.assertNotIn("size", result.columns)
            self.assertEqual(result["modality"].unique().tolist(), ["ct"])
            self.assertEqual(result["subject_id"].unique().tolist(), ["STS_001"])
            self.assertEqual(
                result["image_voxels"].unique().tolist(),
                [512 * 512 * 73],
            )
        self.assertEqual(relative.loc[0, "matched_n"], 2)
        self.assertFalse(
            any("throughput" in column.lower() for column in absolute.columns)
        )
        self.assertFalse(
            any("throughput" in column.lower() for column in relative.columns)
        )

    def test_real_world_summaries_do_not_pool_equal_voxel_cases(self) -> None:
        rows = []
        for case_id, shape, spacing in (
            ("ct-wide", (2, 6, 10), (0.7, 0.7, 3.0)),
            ("ct-tall", (3, 5, 8), (1.0, 1.0, 1.5)),
        ):
            rows.extend(
                [
                    self._comparison_row(
                        case_id,
                        adapter,
                        "texture",
                        1,
                        duration,
                        modality="ct",
                        image_voxels=120,
                        shape=shape,
                        spacing=spacing,
                        dataset_kind="real_world",
                    )
                    for adapter, duration in (
                        ("pictologics", 1.0),
                        ("pyradiomics", 3.0),
                    )
                ]
            )
        frame = self.pd.DataFrame(rows)

        absolute = self.report.aggregate_timing_observations(
            frame, dataset_kind="real_world"
        )
        matched = self.report.matched_runtime_observations(frame)
        relative = self.report.matched_runtime_summary(
            matched, dataset_kind="real_world"
        )

        self.assertEqual(len(absolute), 4)
        self.assertEqual(len(relative), 2)
        self.assertEqual(set(absolute["case_id"]), {"ct-wide", "ct-tall"})
        self.assertEqual(set(relative["case_id"]), {"ct-wide", "ct-tall"})
        self.assertEqual(set(relative["matched_n"]), {1})
        self.assertEqual(set(relative["shape"]), {"[2,6,10]", "[3,5,8]"})
        self.assertEqual(set(relative["spacing"]), {"[0.7,0.7,3.0]", "[1.0,1.0,1.5]"})

    def test_real_world_summary_rejects_geometry_voxel_mismatch(self) -> None:
        frame = self.pd.DataFrame(
            [
                self._comparison_row(
                    "ct-a",
                    "pictologics",
                    "texture",
                    1,
                    1.0,
                    modality="ct",
                    image_voxels=121,
                    shape=(2, 6, 10),
                    spacing=(1.0, 1.0, 1.0),
                    dataset_kind="real_world",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "shape does not match"):
            self.report.aggregate_timing_observations(frame, dataset_kind="real_world")

    def test_report_record_loader_is_an_explicit_attestation_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "run_spec.json").write_text(
                json.dumps(
                    {
                        "run_id": "test-run",
                        "dataset": "clinical-validation",
                        "dataset_kind": "real_world",
                    }
                ),
                encoding="utf-8",
            )

            def verified_loader(path: Path):
                self.assertEqual(path, input_dir)
                return (
                    self.pd.DataFrame(
                        [
                            self._comparison_row(
                                "ct-a",
                                "pictologics",
                                "texture",
                                1,
                                1.0,
                                modality="ct",
                                image_voxels=1000,
                                dataset_kind="real_world",
                            )
                        ]
                    ),
                    {
                        "record_source": "verified-ledger",
                        "source_attested": True,
                    },
                )

            records, metadata = self.report.load_report_records(
                input_dir, record_loader=verified_loader
            )

        self.assertEqual(metadata["record_source"], "verified-ledger")
        self.assertTrue(metadata["source_attested"])
        self.assertEqual(metadata["dataset_kind"], "real_world")
        self.assertEqual(records["_dataset_kind"].tolist(), ["real_world"])

    def test_default_loader_requires_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "benchmark.sqlite3").touch()
            authoritative = self.pd.DataFrame(
                [
                    self._comparison_row(
                        "case-a",
                        "pictologics",
                        "texture",
                        1,
                        1.0,
                    )
                ]
            )
            ledger_result = (
                authoritative,
                {
                    "record_source": "benchmark.sqlite3",
                    "source_attested": True,
                    "run_spec": {
                        "run_id": "ledger-run",
                        "dataset": "synthetic-test",
                        "dataset_kind": "synthetic",
                    },
                    "run_status": "completed",
                    "execution_complete": True,
                    "status_counts": {"measured": 1},
                },
            )
            with mock.patch.object(
                self.report,
                "_ledger_record_loader",
                return_value=ledger_result,
            ) as ledger_loader:
                records, metadata = self.report.load_report_records(input_dir)

        ledger_loader.assert_called_once_with(input_dir)
        self.assertEqual(records["duration_sec"].tolist(), [1.0])
        self.assertTrue(metadata["source_attested"])
        self.assertEqual(metadata["run_id"], "ledger-run")

    def test_ledger_run_spec_dataset_kind_is_not_overwritten_by_row_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "benchmark.sqlite3").touch()
            authoritative = self.pd.DataFrame(
                [
                    {
                        "task_status": "measured",
                        "success": True,
                        "duration_sec": 1.0,
                        "adapter": "pictologics",
                    }
                ]
            )
            ledger_result = (
                authoritative,
                {
                    "record_source": "benchmark.sqlite3",
                    "source_attested": True,
                    "run_spec": {
                        "run_id": "ledger-run",
                        "dataset": "clinical-test",
                        "dataset_kind": "real_world",
                    },
                },
            )
            with mock.patch.object(
                self.report,
                "_ledger_record_loader",
                return_value=ledger_result,
            ):
                records, metadata = self.report.load_report_records(input_dir)

        self.assertEqual(metadata["dataset_kind"], "real_world")
        self.assertEqual(records["_dataset_kind"].tolist(), ["real_world"])

    def test_ledger_loader_verifies_payload_and_exposes_completion(self) -> None:
        from bench.benchmark_ledger import (
            BenchmarkLedger,
            sha256_file,
        )
        from bench.benchmark_models import canonical_json

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "original" / "pillar"
            input_dir.mkdir(parents=True)
            ledger_path = input_dir / "benchmark.sqlite3"
            payload_path = input_dir / "records" / "case-a" / "result.json"
            payload_path.parent.mkdir(parents=True)
            run_spec = {
                "run_id": "verified-run",
                "dataset": "synthetic-test",
                "dataset_kind": "synthetic",
            }
            from bench.benchmark_models import fingerprint

            run_fingerprint = fingerprint(run_spec)
            record = self._comparison_row(
                "case-a",
                "pictologics",
                "texture",
                1,
                1.0,
            )
            record["run_id"] = "verified-run"
            payload = {
                "adapter": "pictologics",
                "features": {"all": ["Mean"]},
                "benchmark": {
                    "task_id": record["task_id"],
                    "run_fingerprint": run_fingerprint,
                    "status": "measured",
                    "record": record,
                },
            }
            payload_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with BenchmarkLedger(ledger_path) as ledger:
                ledger.set_metadata("run_fingerprint", run_fingerprint)
                ledger.set_metadata(
                    "run_spec_json",
                    canonical_json(run_spec),
                )
                ledger.set_metadata("run_status", "completed")
                with ledger.connection:
                    ledger.connection.execute(
                        """
                        INSERT INTO tasks(
                            task_id, ordinal, spec_json, case_id, adapter,
                            workload_key, repeat, complexity, status,
                            duration_sec, payload_path, payload_sha256,
                            record_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["task_id"],
                            1,
                            "{}",
                            record["case_id"],
                            record["adapter"],
                            record["workload"],
                            record["repeat"],
                            record["image_voxels"],
                            "measured",
                            record["duration_sec"],
                            "records/case-a/result.json",
                            sha256_file(payload_path),
                            canonical_json(record),
                        ),
                    )

            relocated = Path(temp_dir) / "aggregation" / "pillar"
            relocated.parent.mkdir(parents=True)
            shutil.move(str(input_dir), str(relocated))
            input_dir = relocated
            ledger_path = input_dir / "benchmark.sqlite3"

            # An editable derived CSV must not affect the authoritative load.
            (input_dir / "summary.csv").write_text(
                "task_status,duration_sec\nmeasured,999\n",
                encoding="utf-8",
            )
            records, metadata = self.report.load_report_records(input_dir)
            feature_summary = self.report.summarize_features_from_payloads(
                metadata["verified_payloads"],
                ["pictologics"],
            )

        self.assertEqual(records["duration_sec"].tolist(), [1.0])
        self.assertEqual(metadata["record_source"], "benchmark.sqlite3")
        self.assertTrue(metadata["source_attested"])
        self.assertTrue(metadata["execution_complete"])
        self.assertEqual(metadata["status_counts"], {"measured": 1})
        self.assertEqual(metadata["verified_payload_count"], 1)
        self.assertEqual(
            feature_summary["pictologics"]["denominators"]["attempted"],
            1,
        )

    def test_ledger_loader_rejects_run_spec_fingerprint_mismatch(self) -> None:
        from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError
        from bench.benchmark_models import canonical_json

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            with BenchmarkLedger(input_dir / "benchmark.sqlite3") as ledger:
                ledger.set_metadata("run_fingerprint", "a" * 64)
                ledger.set_metadata(
                    "run_spec_json",
                    canonical_json(
                        {
                            "run_id": "changed-run",
                            "dataset": "changed-dataset",
                            "dataset_kind": "real_world",
                        }
                    ),
                )
                ledger.set_metadata("run_status", "completed")

            with self.assertRaisesRegex(
                RunIntegrityError,
                "differs from its stored fingerprint",
            ):
                self.report.load_report_records(input_dir)

    def test_generated_report_uses_configured_baseline_and_manifests_outputs(
        self,
    ) -> None:
        from bench.benchmark_ledger import sha256_file

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            rows = self.pd.DataFrame(
                [
                    self._comparison_row("case-a", "mirp", "texture", 1, 2.0),
                    self._comparison_row(
                        "case-a", "pictologics", "texture", 1, 4.0
                    ),
                ]
            )
            payloads = []
            for adapter, row in zip(("mirp", "pictologics"), rows.to_dict("records")):
                payloads.append(
                    {
                        "adapter": adapter,
                        "features": {"all": ["Mean"]},
                        "benchmark": {
                            "status": "measured",
                            "task_id": row["task_id"],
                        },
                        "timing": {
                            "duration_samples_sec": [row["duration_sec"]],
                            "cpu_time_samples_sec": [row["duration_sec"]],
                            "preparation_samples_sec": [0.0],
                            "finalization_samples_sec": [0.0],
                        },
                    }
                )

            def verified_loader(path: Path):
                self.assertEqual(path, input_dir)
                return rows, {
                    "record_source": "verified-test-ledger",
                    "source_attested": True,
                    "verified_payloads": payloads,
                    "verified_payload_count": 2,
                    "run_status": "completed",
                    "execution_complete": True,
                    "task_count": 2,
                    "unfinished_task_count": 0,
                    "status_counts": {"measured": 2},
                    "run_fingerprint": "report-test-fingerprint",
                    "run_spec": {
                        "schema_version": 4,
                        "run_id": "report-test",
                        "dataset": "synthetic-test",
                        "dataset_kind": "synthetic",
                        "manifest_sha256": "a" * 64,
                        "dataset_hashes_verified": True,
                        "dataset_values_inspected": True,
                        "adapters": ["mirp", "pictologics"],
                        "workloads": ["texture"],
                        "repeats": 1,
                        "aggregation": "3d_merge",
                        "discretization": "fbn",
                        "bins": 32,
                        "bin_width": 32.0,
                        "timing_observations": 2,
                        "thread_policy": {
                            "mode": "all_physical_cores_per_isolated_task",
                            "environment": {"OMP_NUM_THREADS": "1"},
                        },
                        "benchmark_machine": {
                            "machine_id": "anonymous-test",
                            "cpu_model": "Test CPU",
                        },
                        "adapter_environments": {
                            adapter: {
                                "distribution": adapter,
                                "configured_release_version": "1.2.3",
                                "distribution_version": "1.2.3",
                                "python_version": "3.12.1",
                                "profile_python": "3.12",
                                "packages": [[adapter, "1.2.3"]],
                                "numpy_config": {
                                    "version": "2.0.0",
                                    "show_config": "BLAS: test",
                                },
                                "environment_freeze_sha256": "c" * 64,
                                "environment_metadata_sha256": "b" * 64,
                            }
                            for adapter in ("mirp", "pictologics")
                        },
                        "guardrail": {"baseline_adapter": "mirp"},
                    },
                }

            outputs = self.report.generate_report(
                input_dir,
                output_dir,
                record_loader=verified_loader,
            )
            manifest = json.loads(
                outputs["report_manifest"].read_text(encoding="utf-8")
            )
            pairs = self.pd.read_csv(outputs["matched_pairs"])
            protocol = self.pd.read_csv(outputs["protocol_summary"])
            environments = self.pd.read_csv(outputs["adapter_environments"])
            observations = self.pd.read_csv(outputs["task_observations"])
            markdown = outputs["markdown_summary"].read_text(encoding="utf-8")
            workbook = self.pd.ExcelFile(outputs["workbook"])
            workbook_sheet_names = list(workbook.sheet_names)
            workbook.close()

            manifested = {entry["path"] for entry in manifest["artifacts"]}
            generated = {
                path.name
                for path in output_dir.iterdir()
                if path.name != "report_manifest.json"
            }
            for entry in manifest["artifacts"]:
                artifact = output_dir / entry["path"]
                self.assertEqual(entry["bytes"], artifact.stat().st_size)
                self.assertEqual(entry["sha256"], sha256_file(artifact))

        self.assertTrue(manifest["publication_attested"])
        self.assertEqual(manifest["baseline_adapter"], "mirp")
        self.assertRegex(
            manifest["report_generator"]["provenance_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            manifest["report_generator"]["source_tree_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            set(manifest["report_generator"]["dependencies"]),
            {"matplotlib", "numpy", "openpyxl", "pandas"},
        )
        self.assertEqual(manifested, generated)
        self.assertEqual(pairs["baseline_adapter"].tolist(), ["mirp"])
        self.assertEqual(pairs["runtime_ratio"].tolist(), [2.0])
        self.assertEqual(
            protocol.loc[protocol["field"] == "aggregation", "value"].tolist(),
            ["3d_merge"],
        )
        self.assertEqual(environments["adapter"].tolist(), ["mirp", "pictologics"])
        self.assertEqual(len(observations), 2)
        self.assertEqual(
            json.loads(observations.loc[0, "duration_samples_sec"]),
            [2.0],
        )
        self.assertNotIn("error", observations.columns)
        self.assertNotIn("adapter_stderr", observations.columns)
        self.assertTrue(
            environments["installed_packages_sha256"]
            .str.fullmatch(r"[0-9a-f]{64}")
            .all()
        )
        self.assertEqual(
            environments["environment_freeze_sha256"].tolist(),
            ["c" * 64, "c" * 64],
        )
        self.assertIn("Protocol", workbook_sheet_names)
        self.assertIn("Adapter environments", workbook_sheet_names)
        self.assertIn("Feature workload", workbook_sheet_names)
        self.assertIn("Comparison baseline adapter: `mirp`", markdown)
        self.assertIn("## Executed protocol", markdown)
        self.assertIn("## Adapter environments", markdown)
        self.assertIn("not a common-feature microbenchmark", markdown)
        self.assertIn("mask/resegmentation preparation", markdown)
        self.assertIn("Runtime is not divided by feature count", markdown)
        self.assertIn("local-intensity cache state", markdown)
        self.assertIn("no filter-performance claim is made", markdown)
        self.assertIn(
            "Report generator provenance: "
            f"`{manifest['report_generator']['provenance_sha256']}`",
            markdown,
        )
        self.assertFalse(any("throughput" in path.lower() for path in manifested))

    def test_adapter_environment_summary_preserves_pyradiomics_release_labels(
        self,
    ) -> None:
        summary = self.report._adapter_environment_summary(
            {
                "adapters": ["pyradiomics"],
                "adapter_environments": {
                    "pyradiomics": {
                        "distribution": "pyradiomics",
                        "configured_release_version": "3.1.0",
                        "distribution_version": "3.0.1a1",
                        "python_version": "3.11.15",
                        "profile_python": "3.11",
                        "packages": [["pyradiomics", "3.0.1a1"]],
                        "numpy_config": {
                            "version": "1.26.4",
                            "show_config": "BLAS: Accelerate",
                        },
                        "environment_metadata_sha256": "c" * 64,
                    }
                },
            }
        )

        self.assertEqual(summary["configured_release_version"].tolist(), ["3.1.0"])
        self.assertEqual(
            summary["distribution_metadata_version"].tolist(),
            ["3.0.1a1"],
        )
        self.assertEqual(
            summary["numpy_build_configuration"].tolist(),
            ["BLAS: Accelerate"],
        )

    def test_incomplete_run_is_explicit_in_markdown_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.md"
            self.report._write_markdown_summary(
                path,
                input_dir=Path(temp_dir),
                metadata={
                    "dataset_kind": "synthetic",
                    "run_status": "interrupted",
                    "execution_complete": False,
                    "task_count": 10,
                    "unfinished_task_count": 3,
                    "verified_payload_count": 7,
                    "record_source": "benchmark.sqlite3",
                    "source_attested": True,
                    "status_counts": {"measured": 7, "interrupted": 3},
                },
                measured_n=7,
                matched=self.pd.DataFrame(),
                matched_summary=self.pd.DataFrame(),
                feature_contract=self.pd.DataFrame(),
                diagnostics=self.pd.DataFrame(),
                coverage={},
                figures=[],
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("Run status: `interrupted`", text)
        self.assertIn("Execution complete: no", text)
        self.assertIn("| interrupted | 3 |", text)
        self.assertIn("explicitly partial snapshot", text)

    def test_feature_workload_contract_keeps_counts_without_normalizing(self) -> None:
        frame = self.pd.DataFrame(
            [
                {
                    "adapter": "pictologics",
                    "workload": "texture",
                    "task_status": "measured",
                    "expected_feature_count": 25,
                    "feature_count": 25,
                },
                {
                    "adapter": "pictologics",
                    "workload": "texture",
                    "task_status": "timed_out_censored",
                    "expected_feature_count": 25,
                    "feature_count": None,
                },
            ]
        )
        result = self.report.feature_workload_contract(frame)

        self.assertEqual(result["expected_native_outputs"].tolist(), [25])
        self.assertEqual(result["observed_native_outputs_min"].tolist(), [25])
        self.assertEqual(result["planned_tasks"].tolist(), [2])
        self.assertEqual(result["measured_tasks"].tolist(), [1])
        self.assertEqual(result["measured_fraction_of_planned"].tolist(), [0.5])
        self.assertEqual(result["runtime_normalization"].tolist(), ["none"])

    def test_feature_summary_reports_only_named_denominators(self) -> None:
        measured = {
            "adapter": "pictologics",
            "benchmark": {"status": "measured"},
            "features": {"all": ["Mean", "Variance"]},
            "values": {"all": {"Mean": 1.0, "Variance": None}},
            "feature_denominators": {
                "supported": ["Mean", "Variance", "Skewness"],
                "referenced": ["Mean", "Variance"],
                "passing": ["Mean"],
            },
        }
        excluded = {
            "adapter": "pictologics",
            "benchmark": {"status": "timed_out_censored"},
            "features": {"all": ["MustNotCount"]},
            "values": {"all": {"MustNotCount": 1.0}},
        }
        summary = self.report.summarize_features_from_payloads(
            [measured, excluded], ["pictologics"]
        )["pictologics"]

        self.assertEqual(
            summary["denominators"],
            {
                "supported": 3,
                "attempted": 2,
                "finite": 1,
                "referenced": 2,
                "passing": 1,
            },
        )

    @staticmethod
    def _timing_row(
        workload: str | None,
        family: str,
        duration: float,
        *,
        status: str = "measured",
    ) -> dict[str, object]:
        return {
            "task_status": status,
            "success": status == "measured",
            "adapter": "pictologics",
            "workload": workload,
            "size": 32,
            "duration_sec": duration,
            "feature_count": 5,
            "censor_lower_bound_sec": duration
            if status == "timed_out_censored"
            else None,
        }

    @staticmethod
    def _comparison_row(
        case_id: str,
        adapter: str,
        workload: str,
        repeat: int,
        duration: float,
        *,
        status: str = "measured",
        modality: str = "synthetic",
        image_voxels: int = 32**3,
        shape: tuple[int, int, int] = (32, 32, 32),
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        dataset_kind: str = "synthetic",
        subject_id: str = "reference",
    ) -> dict[str, object]:
        return {
            "task_id": f"{case_id}-{adapter}-{workload}-{repeat}",
            "task_status": status,
            "success": status == "measured",
            "case_id": case_id,
            "subject_id": subject_id,
            "dataset": "test-dataset",
            "modality": modality,
            "adapter": adapter,
            "workload": workload,
            "requested_families": [workload],
            "expected_feature_count": 5,
            "feature_count": 5,
            "repeat": repeat,
            "size": 32,
            "mask_id": "M1",
            "mask_label": "central mask",
            "shape": list(shape),
            "spacing": list(spacing),
            "image_voxels": image_voxels,
            "mask_voxels": 100,
            "duration_sec": duration,
            "peak_rss_bytes": 1024,
            "_dataset_kind": dataset_kind,
            "censor_lower_bound_sec": (
                duration if status == "timed_out_censored" else None
            ),
        }


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bench.benchmark_ledger import (
    BenchmarkLedger,
    RunLock,
    RunLocked,
    RunIntegrityError,
    RunSpecMismatch,
    atomic_write_json,
)
from bench.benchmark_models import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_MEASURED,
    RunSpec,
    TaskSpec,
    task_plan_fingerprint,
)
from bench.run import _benchmark_machine_identity


def make_task(*, bins: int = 32) -> TaskSpec:
    return TaskSpec(
        ordinal=1,
        case_id="case-1",
        dataset="unit-benchmark",
        modality="synthetic",
        size=10,
        variant=1,
        mask_id="M1",
        mask_label="mask",
        image_path="/tmp/image.nii.gz",
        source_image_path="/tmp/image.nii.gz",
        mask_path="/tmp/mask.nii.gz",
        image_sha256="1" * 64,
        source_image_sha256="1" * 64,
        mask_sha256="2" * 64,
        shape=(10, 10, 10),
        spacing=(1.0, 1.0, 1.0),
        image_voxels=1000,
        mask_voxels=500,
        mask_fraction=0.5,
        complexity=1000,
        subject_id="subject-1",
        input_contract="manifest_harmonized",
        representation_id="original_continuous_image",
        representation_derivation_sha256=None,
        configured_levels=None,
        occupied_levels=None,
        adapter="fake",
        workload="texture",
        requested_families=(
            "histogram",
            "glcm",
            "glrlm",
            "glszm",
            "gldzm",
            "ngtdm",
            "ngldm",
        ),
        repeat=1,
        discretization="fbn",
        bins=bins,
        bin_width=32.0,
        intensity_min=None,
        intensity_max=None,
        timing_observations=2,
    )


def make_run_spec(
    task: TaskSpec,
    *,
    benchmark_machine: dict[str, object] | None = None,
) -> RunSpec:
    return RunSpec.create(
        run_id="unit-run",
        dataset="unit-benchmark",
        dataset_kind="synthetic",
        dataset_manifest_schema_version=2,
        dataset_dir="/tmp/unit-benchmark",
        manifest_sha256="manifest",
        dataset_hashes_verified=True,
        dataset_values_inspected=True,
        selected_case_ids=("case-1",),
        adapters=("fake",),
        workloads=("texture",),
        repeats=1,
        aggregation="3d_merge",
        input_contract="manifest_harmonized",
        timing_observations=2,
        capture_values=True,
        timeout_seconds=None,
        keep_going=True,
        task_plan_sha256=task_plan_fingerprint([task]),
        runtime_profiles_sha256="profiles",
        benchmark_sources_sha256="sources",
        benchmark_machine=benchmark_machine or {"machine_id": "unit-machine"},
        adapter_environments={"fake": {"version": "1.0"}},
        thread_policy={
            "mode": "all_physical_cores_per_isolated_task",
            "environment": {"OMP_NUM_THREADS": "1"},
        },
        initialization_policy={"environment": {"PICTOLOGICS_DISABLE_WARMUP": "1"}},
        guardrail={"enabled": True, "skip_ratio": 10.0},
    )


class BenchmarkLedgerTests(unittest.TestCase):
    def test_run_local_payload_paths_are_relocatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "machine" / "pillar"
            payload = run_dir / "records" / "case" / "result.json"
            payload.parent.mkdir(parents=True)
            payload.write_text("{}\n", encoding="utf-8")
            with BenchmarkLedger(run_dir / "benchmark.sqlite3") as ledger:
                encoded = ledger.portable_payload_path(payload)
                self.assertEqual(encoded, "records/case/result.json")
                self.assertEqual(ledger.resolve_payload_path(encoded), payload.resolve())

    def test_repeat_horizon_extends_append_only_without_changing_protocol_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            first_task = make_task()
            first_spec = make_run_spec(first_task)
            second_task = replace(first_task, ordinal=2, repeat=2)
            extended_spec = replace(
                first_spec,
                repeats=2,
                task_plan_sha256=task_plan_fingerprint([first_task, second_task]),
            )
            self.assertEqual(first_spec.run_fingerprint, extended_spec.run_fingerprint)
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(first_spec, [first_task], resume=False)
                ledger.initialize(extended_spec, [first_task, second_task], resume=True)
                self.assertEqual(len(ledger.task_rows()), 2)
                self.assertEqual(
                    int(ledger.metadata_value("run_spec_json") is not None), 1
                )

    def test_resume_rejects_changed_immutable_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            first_task = make_task(bins=32)
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(make_run_spec(first_task), [first_task], resume=False)

            changed_task = make_task(bins=64)
            with BenchmarkLedger(path) as ledger:
                with self.assertRaises(RunSpecMismatch):
                    ledger.initialize(
                        make_run_spec(changed_task), [changed_task], resume=True
                    )

    def test_resume_accepts_changed_observed_energy_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            task = make_task()
            automatic = make_run_spec(
                task,
                benchmark_machine=_benchmark_machine_identity({
                    "machine_id": "unit-machine",
                    "host_settings": {
                        "energy_mode": "automatic",
                        "pmset_lowpowermode": 0,
                    },
                }),
            )
            high_power = make_run_spec(
                task,
                benchmark_machine=_benchmark_machine_identity({
                    "machine_id": "unit-machine",
                    "host_settings": {
                        "energy_mode": "high_power",
                        "pmset_lowpowermode": 2,
                    },
                }),
            )
            self.assertEqual(
                automatic.run_fingerprint,
                high_power.run_fingerprint,
            )
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(automatic, [task], resume=False)
                ledger.initialize(high_power, [task], resume=True)

    def test_running_attempt_is_recovered_as_interrupted_and_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            task = make_task()
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(make_run_spec(task), [task], resume=False)
                ledger.mark_running(task.task_id)

            with BenchmarkLedger(path) as ledger:
                self.assertEqual(ledger.recover_running(), 1)
                self.assertEqual(ledger.status(task.task_id), STATUS_INTERRUPTED)
                self.assertEqual(ledger.mark_running(task.task_id), 2)

    def test_timeout_cutoff_persists_the_earliest_complexity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            task = make_task()
            scope = "adapter\x1ftexture\x1fseries"
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(make_run_spec(task), [task], resume=False)
                ledger.activate_timeout_cutoff(
                    scope_key=scope,
                    adapter="adapter",
                    workload_key="texture",
                    guardrail_group="series",
                    complexity_metric="image_voxels",
                    cutoff_complexity=8_000,
                    reason="first timeout",
                    evidence_task_id=task.task_id,
                )
                ledger.activate_timeout_cutoff(
                    scope_key=scope,
                    adapter="adapter",
                    workload_key="texture",
                    guardrail_group="series",
                    complexity_metric="image_voxels",
                    cutoff_complexity=16_000,
                    reason="later timeout",
                    evidence_task_id=task.task_id,
                )
                self.assertEqual(
                    int(ledger.timeout_cutoff(scope)["cutoff_complexity"]),
                    8_000,
                )
                self.assertEqual(len(ledger.timeout_cutoffs()), 1)

    def test_failed_attempt_is_preserved_and_requeued_on_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.sqlite3"
            task = make_task()
            with BenchmarkLedger(path) as ledger:
                ledger.initialize(make_run_spec(task), [task], resume=False)
                self.assertEqual(ledger.mark_running(task.task_id), 1)
                ledger.mark_terminal(
                    task.task_id,
                    STATUS_FAILED,
                    {
                        "task_id": task.task_id,
                        "task_status": STATUS_FAILED,
                    },
                    error="transient worker failure",
                )
                self.assertEqual(ledger.recover_failed(), 1)
                self.assertEqual(ledger.status(task.task_id), STATUS_INTERRUPTED)
                failed_attempt = ledger.connection.execute(
                    "SELECT status, error FROM task_attempts "
                    "WHERE task_id = ? AND attempt = 1",
                    (task.task_id,),
                ).fetchone()
                self.assertEqual(str(failed_attempt["status"]), STATUS_FAILED)
                self.assertEqual(
                    str(failed_attempt["error"]), "transient worker failure"
                )
                self.assertEqual(ledger.mark_running(task.task_id), 2)

    def test_run_lock_rejects_second_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".benchmark.lock"
            first = RunLock(path).acquire()
            try:
                with self.assertRaises(RunLocked):
                    RunLock(path).acquire()
            finally:
                first.release()

    def test_report_loader_rejects_payload_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger_path = root / "ledger.sqlite3"
            payload_path = root / "payload.json"
            task = make_task()
            run_spec = make_run_spec(task)
            record = {
                "task_id": task.task_id,
                "task_status": STATUS_MEASURED,
            }
            with BenchmarkLedger(ledger_path) as ledger:
                ledger.initialize(run_spec, [task], resume=False)
                ledger.mark_running(task.task_id)
                digest = atomic_write_json(
                    payload_path,
                    {
                        "benchmark": {
                            "task_id": task.task_id,
                            "run_fingerprint": run_spec.run_fingerprint,
                            "status": STATUS_MEASURED,
                            "record": record,
                        }
                    },
                )
                ledger.mark_terminal(
                    task.task_id,
                    STATUS_MEASURED,
                    record,
                    duration_sec=1.0,
                    payload_path=ledger.portable_payload_path(payload_path),
                    payload_sha256=digest,
                )
                records, payloads = ledger.verified_records_and_payloads()
                self.assertEqual(records, [record])
                self.assertEqual(len(payloads), 1)

            payload_path.write_text("{}\n", encoding="utf-8")
            with BenchmarkLedger(ledger_path) as ledger:
                with self.assertRaisesRegex(
                    RunIntegrityError,
                    "checksum changed",
                ):
                    ledger.verified_records_and_payloads()


if __name__ == "__main__":
    unittest.main()

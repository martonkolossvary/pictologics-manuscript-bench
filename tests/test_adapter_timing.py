from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from bench.adapters.base import ResultEquivalenceError, run_timed_computation
from bench.adapters.protocol import (
    add_common_arguments,
    make_payload,
    run_adapter_timing,
    timing_contract_metadata,
)


class AdapterTimingTests(unittest.TestCase):
    def test_primary_duration_is_median_not_optimistic_minimum(self) -> None:
        # One warmup, one verification call, then three measured calls.
        wall = iter(
            [
                0,
                7_000_000_000,
                10_000_000_000,
                11_000_000_000,
                20_000_000_000,
                21_000_000_000,
                30_000_000_000,
                39_000_000_000,
                40_000_000_000,
                45_000_000_000,
            ]
        )
        cpu = iter(
            [
                0,
                700_000_000,
                1_000_000_000,
                1_100_000_000,
                2_000_000_000,
                2_100_000_000,
                3_000_000_000,
                3_900_000_000,
                4_000_000_000,
                4_500_000_000,
            ]
        )
        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            result, stats = run_timed_computation(lambda: "ok", iterations=3)
        self.assertEqual(result, "ok")
        self.assertEqual(stats["duration_min_sec"], 1.0)
        self.assertEqual(stats["duration_sec"], 5.0)
        self.assertEqual(stats["duration_median_sec"], 5.0)
        self.assertEqual(stats["duration_samples_sec"], [1.0, 9.0, 5.0])

    def test_one_warmup_is_discarded(self) -> None:
        calls = []

        def compute():
            calls.append(len(calls))
            return "stable"

        _, stats = run_timed_computation(
            compute,
            iterations=2,
            target_observation_window_sec=1e-12,
        )
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(stats["calibration_calls"], 1)
        self.assertEqual(stats["warmup_iterations"], 1)
        self.assertEqual(stats["measured_iterations"], 2)
        self.assertEqual(stats["total_iterations"], 3)
        self.assertIsNotNone(stats["warmup_duration_sec"])
        self.assertEqual(len(stats["preparation_samples_sec"]), 2)
        self.assertEqual(len(stats["finalization_samples_sec"]), 2)

    def test_steady_state_calibration_sizes_measured_windows(self) -> None:
        wall = iter(index * 100_000_000 for index in range(200))
        cpu = iter(index * 100_000_000 for index in range(200))
        calls = []

        def compute():
            calls.append(None)
            return "ok"

        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            _, stats = run_timed_computation(
                compute,
                iterations=1,
                target_observation_window_sec=1.0,
            )
        self.assertEqual(stats["calibration_calls"], 60)
        self.assertEqual(stats["calibration_rounds"], 3)
        self.assertTrue(stats["calibration_stable"])
        self.assertAlmostEqual(stats["calibration_per_call_sec"], 0.1)
        self.assertEqual(stats["calls_per_observation"], 20)
        self.assertAlmostEqual(stats["observation_window_samples_sec"][0], 2.0)
        self.assertEqual(stats["total_calculation_calls"], 81)
        self.assertEqual(stats["result_equivalence_checks"], 80)
        self.assertEqual(len(calls), 81)

    def test_borderline_warmup_receives_enough_batch_headroom(self) -> None:
        # A 68 ms warmup used to bypass calibration, after which a 49 ms
        # measured call failed the hard 50 ms window requirement. The reviewed
        # 100 ms headroom forces stable three-call measured windows instead.
        durations = [0.068] + [0.049] * 17

        def clock(values):
            current = 0
            samples = []
            for duration in values:
                samples.extend((current, current + int(duration * 1_000_000_000)))
                current = samples[-1]
            return iter(samples)

        wall = clock(durations)
        cpu = clock(durations)
        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            _, stats = run_timed_computation(lambda: "stable", iterations=3)

        self.assertEqual(stats["calibration_rounds"], 3)
        self.assertEqual(stats["calls_per_observation"], 3)
        for sample in stats["observation_window_samples_sec"]:
            self.assertAlmostEqual(sample, 0.147)

    def test_later_slow_calibration_window_cannot_bypass_convergence(self) -> None:
        # The second window is a transient slowdown. It exceeds the 100 ms
        # per-call shortcut, but only the first independent calibration window
        # may take that shortcut; later windows must converge normally.
        durations = (
            [0.05]
            + [0.04] * 2
            + [0.11] * 3
            + [0.04] * 9
            + [0.04] * 9
        )

        def clock(values):
            current = 0
            samples = []
            for duration in values:
                samples.extend((current, current + int(duration * 1_000_000_000)))
                current = samples[-1]
            return iter(samples)

        wall = clock(durations)
        cpu = clock(durations)
        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            _, stats = run_timed_computation(lambda: "stable", iterations=3)

        self.assertEqual(stats["calibration_rounds"], 5)
        self.assertEqual(stats["calibration_calls"], 14)
        self.assertEqual(stats["calls_per_observation"], 3)
        self.assertTrue(stats["calibration_stable"])
        for sample in stats["observation_window_samples_sec"]:
            self.assertAlmostEqual(sample, 0.12)

    def test_every_repeated_result_must_have_the_same_feature_names(self) -> None:
        calls = 0

        def compute():
            nonlocal calls
            calls += 1
            return {"feature-a": 1.0} if calls == 1 else {"feature-b": 1.0}

        with self.assertRaisesRegex(ResultEquivalenceError, "feature names"):
            run_timed_computation(
                compute,
                iterations=1,
                target_observation_window_sec=1e-12,
            )

    def test_every_repeated_result_must_be_numerically_equivalent(self) -> None:
        calls = 0

        def compute():
            nonlocal calls
            calls += 1
            return {"feature": 1.0 if calls == 1 else 1.01}

        with self.assertRaisesRegex(ResultEquivalenceError, "numeric value"):
            run_timed_computation(
                compute,
                iterations=1,
                target_observation_window_sec=1e-12,
            )

    def test_preparation_and_finalization_are_outside_calculation_clock(self) -> None:
        calls = []
        wall = iter(
            [
                0,
                1_000_000_000,
                2_000_000_000,
                3_000_000_000,
                4_000_000_000,
                5_000_000_000,
                10_000_000_000,
                14_000_000_000,
                20_000_000_000,
                22_000_000_000,
                30_000_000_000,
                38_000_000_000,
                40_000_000_000,
                44_000_000_000,
                50_000_000_000,
                52_000_000_000,
                60_000_000_000,
                68_000_000_000,
            ]
        )
        cpu = iter(
            [
                0,
                100_000_000,
                1_000_000_000,
                1_500_000_000,
                2_000_000_000,
                2_500_000_000,
            ]
        )

        def prepare():
            calls.append("prepare")
            return "state"

        def compute(state):
            calls.append(("compute", state))
            return "raw"

        def finalize(raw, state):
            calls.append(("finalize", raw, state))
            return "result"

        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            result, stats = run_timed_computation(
                compute,
                iterations=1,
                prepare_fn=prepare,
                finalize_fn=finalize,
            )
        self.assertEqual(result, "result")
        self.assertEqual(stats["preparation_samples_sec"], [2.0])
        self.assertEqual(stats["duration_samples_sec"], [8.0])
        self.assertEqual(stats["finalization_samples_sec"], [4.0])
        self.assertEqual(
            calls,
            [
                "prepare",
                ("compute", "state"),
                ("finalize", "raw", "state"),
                "prepare",
                ("compute", "state"),
                ("finalize", "raw", "state"),
                "prepare",
                ("compute", "state"),
                ("finalize", "raw", "state"),
            ],
        )

    def test_invalid_iterations_are_rejected(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_timed_computation(lambda: None, iterations=value)

    def test_common_cli_exposes_optional_input_bindings(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        image_digest = "A" * 64
        mask_digest = "b" * 64
        args = parser.parse_args(
            [
                "--image",
                "image.nii.gz",
                "--mask",
                "mask.nii.gz",
                "--image-sha256",
                image_digest,
                "--mask-sha256",
                mask_digest,
            ]
        )
        self.assertEqual(args.image_sha256, image_digest.lower())
        self.assertEqual(args.mask_sha256, mask_digest)

    def test_shared_timing_wrapper_requires_warmup(self) -> None:
        def compute():
            return "ok"

        with patch(
            "bench.adapters.base.run_timed_computation",
            return_value=("ok", {"measured_iterations": 3}),
        ) as timed:
            result = run_adapter_timing(
                compute,
                iterations=3,
            )
        self.assertEqual(result[0], "ok")
        timed.assert_called_once_with(
            compute,
            iterations=3,
            target_observation_window_sec=0.05,
            maximum_calls_per_observation=4096,
            calibration_headroom_factor=2.0,
            calibration_minimum_rounds=3,
            calibration_maximum_rounds=12,
            calibration_cv_threshold=0.05,
            calibration_span_ratio=1.10,
            result_rtol=1e-9,
            result_atol=1e-12,
            event_fn=unittest.mock.ANY,
        )

    def test_timed_payload_declares_scope_and_input_provenance(self) -> None:
        image_digest = "a" * 64
        mask_digest = "b" * 64
        payload = make_payload(
            adapter="pictologics",
            feature_names=["feature"],
            timing={"measured_iterations": 2},
            image_sha256=image_digest,
            mask_sha256=mask_digest,
            modality="CT",
        )
        self.assertEqual(
            payload["metadata"]["timing_contract"],
            timing_contract_metadata(),
        )
        self.assertEqual(
            payload["metadata"]["input"],
            {
                "image_sha256": image_digest,
                "source_image_sha256": image_digest,
                "mask_sha256": mask_digest,
                "modality": "CT",
                "input_contract": "manifest_harmonized",
                "representation_id": "original_continuous_image",
                "representation_derivation_sha256": None,
                "configured_levels": None,
                "occupied_levels": None,
            },
        )


if __name__ == "__main__":
    unittest.main()

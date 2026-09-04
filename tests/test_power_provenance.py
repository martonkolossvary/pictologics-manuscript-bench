from __future__ import annotations

import unittest
from unittest import mock

from bench.power_provenance import (
    combine_power_summaries,
    observe_task_power_state,
    observation_power_tag,
    summarize_power_observations,
    summarize_task_power_records,
)


class PowerProvenanceTests(unittest.TestCase):
    def test_unavailable_power_mode_is_explicit(self) -> None:
        self.assertEqual(
            observation_power_tag({"host_settings": {}}),
            "power-mode-unavailable",
        )
        summary = summarize_power_observations([{"host_settings": {}}])
        self.assertEqual(summary["classification"], "unavailable")
        self.assertEqual(summary["contribution_tag"], "power-mode-unavailable")

    def test_distinct_observed_modes_are_reported_as_mixed(self) -> None:
        summary = summarize_power_observations(
            [
                {"host_settings": {"power_mode_tag": "macos-automatic"}},
                {"host_settings": {"power_mode_tag": "macos-high-power"}},
            ]
        )
        self.assertEqual(summary["classification"], "mixed_mode")
        self.assertEqual(summary["tags"], ["macos-automatic", "macos-high-power"])
        self.assertEqual(summary["contribution_tag"], "mixed-power-modes")

    def test_combined_summary_retains_all_session_provenance(self) -> None:
        combined = combine_power_summaries(
            [
                {
                    "classification": "single_mode",
                    "tags": ["linux-performance"],
                    "session_count": 2,
                    "task_count": 5,
                },
                {
                    "classification": "unavailable",
                    "tags": ["power-mode-unavailable"],
                    "session_count": 1,
                    "task_count": 2,
                },
            ]
        )
        self.assertEqual(combined["classification"], "mixed_mode")
        self.assertEqual(combined["session_count"], 3)
        self.assertEqual(combined["task_count"], 7)
        self.assertEqual(combined["contribution_tag"], "mixed-power-modes")

    def test_macos_task_probe_reads_the_active_ac_profile(self) -> None:
        completed = [
            mock.Mock(returncode=0, stdout="Now drawing from 'AC Power'\n"),
            mock.Mock(
                returncode=0,
                stdout=(
                    "Battery Power:\n lowpowermode 1\n"
                    "AC Power:\n lowpowermode 2\n sleep 1\n"
                ),
            ),
        ]
        with mock.patch("bench.power_provenance._completed", side_effect=completed):
            state = observe_task_power_state("Darwin")

        self.assertEqual(state["power_source"], "AC Power")
        self.assertEqual(state["energy_mode"], "high_power")
        self.assertEqual(state["pmset_lowpowermode"], 2)
        self.assertEqual(state["power_mode_tag"], "macos-high-power-pmset-2")
        self.assertEqual(state["probe_source"], "pmset_custom_active_profile")
        self.assertEqual(state["probe_attempts"], 1)
        self.assertEqual(state["probe_errors"], [])

    def test_macos_task_probe_retries_a_transient_missing_mode(self) -> None:
        completed = [
            mock.Mock(returncode=0, stdout="Now drawing from 'AC Power'\n"),
            mock.Mock(returncode=0, stdout="AC Power:\n sleep 1\n"),
            mock.Mock(
                returncode=0,
                stdout="AC Power:\n lowpowermode 2\n sleep 1\n",
            ),
        ]
        with (
            mock.patch("bench.power_provenance._completed", side_effect=completed),
            mock.patch("bench.power_provenance.time.sleep") as sleep,
        ):
            state = observe_task_power_state("Darwin")

        self.assertEqual(state["energy_mode"], "high_power")
        self.assertEqual(state["probe_attempts"], 2)
        self.assertEqual(sleep.call_count, 1)

    def test_windows_task_probe_uses_guid_not_localized_plan_name(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout=(
                "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e "
                "(Kiegyensúlyozott)"
            ),
        )
        system_power = {
            "power_source": "AC Power",
            "battery_saver": False,
            "battery_life_percent": None,
            "battery_flag": 128,
            "system_power_status_available": True,
        }
        with (
            mock.patch("bench.power_provenance._completed", return_value=completed),
            mock.patch(
                "bench.power_provenance._windows_system_power_status",
                return_value=system_power,
            ),
        ):
            state = observe_task_power_state("Windows")

        self.assertEqual(state["energy_mode"], "balanced")
        self.assertEqual(state["power_mode_tag"], "windows-power-scheme-balanced")
        self.assertEqual(
            state["power_scheme_guid"],
            "381b4222-f694-41f0-9685-ff5bb260df2e",
        )
        self.assertEqual(state["power_scheme_name"], "Kiegyensúlyozott")
        self.assertEqual(state["power_source"], "AC Power")
        self.assertFalse(state["battery_saver"])
        self.assertEqual(state["probe_errors"], [])

    def test_task_summary_uses_task_tags_not_session_settings(self) -> None:
        summary = summarize_task_power_records(
            [
                {"host_power_mode_tag": "macos-high-power-pmset-2"},
                {"host_power_mode_tag": "mixed-within-task"},
            ]
        )
        self.assertEqual(summary["classification"], "mixed_mode")
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["provenance_scope"], "task_start_and_end")


if __name__ == "__main__":
    unittest.main()

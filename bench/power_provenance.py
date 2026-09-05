"""Portable, non-gating power-mode provenance for benchmark sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any


PMSET_ENERGY_MODES = {0: "automatic", 1: "low_power", 2: "high_power"}
WINDOWS_POWER_SCHEME_NAMES = {
    "a1841308-3541-4fab-bc81-f71556f20b4a": "power_saver",
    "381b4222-f694-41f0-9685-ff5bb260df2e": "balanced",
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "high_performance",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "ultimate_performance",
}
TASK_POWER_MODE_ATTEMPTS = 3
TASK_POWER_MODE_RETRY_SECONDS = 0.05


def _slug(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return token[:48].rstrip("-") or "power-mode-unavailable"


def observation_power_tag(observation: Mapping[str, Any]) -> str:
    settings = observation.get("host_settings")
    if not isinstance(settings, Mapping):
        return "power-mode-unavailable"
    explicit = settings.get("power_mode_tag")
    if explicit:
        return _slug(explicit)
    for key in ("power_mode", "power_plan"):
        value = settings.get(key)
        if value:
            return f"{key.replace('_', '-')}-{_slug(value)}"
    return "power-mode-unavailable"


def _completed(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _darwin_task_power_state() -> dict[str, Any]:
    """Observe active macOS power state with bounded transient retries."""

    errors: list[str] = []
    diagnostics: dict[str, Any] = {
        "battery_returncode": None,
        "custom_returncodes": [],
        "custom_active_profile_found": [],
        "custom_lowpowermode_values": [],
        "active_settings_returncode": None,
    }
    power_source = None
    battery = _completed(["/usr/bin/pmset", "-g", "batt"])
    diagnostics["battery_returncode"] = (
        None if battery is None else int(battery.returncode)
    )
    match = re.search(
        r"Now drawing from '([^']+)'",
        battery.stdout if battery is not None else "",
    )
    if battery is not None and battery.returncode == 0 and match:
        power_source = match.group(1)
    else:
        errors.append("power_source_unavailable")

    low_power_mode = None
    probe_source = None
    probe_attempts = 0
    active_heading = "AC Power" if power_source == "AC Power" else "Battery Power"
    for attempt in range(TASK_POWER_MODE_ATTEMPTS):
        probe_attempts = attempt + 1
        completed = _completed(["/usr/bin/pmset", "-g", "custom"])
        diagnostics["custom_returncodes"].append(
            None if completed is None else int(completed.returncode)
        )
        if completed is not None and completed.returncode == 0:
            content = completed.stdout or ""
            block = re.search(
                rf"(?:^|\n){re.escape(active_heading)}:\s*(.*?)(?=\n\S[^\n]*:\s*$|\Z)",
                content,
                flags=re.DOTALL,
            )
            active_content = block.group(1) if block else ""
            diagnostics["custom_active_profile_found"].append(block is not None)
            diagnostics["custom_lowpowermode_values"].append(
                [
                    int(value)
                    for value in re.findall(
                        r"^\s*lowpowermode\s+(\d+)\s*$",
                        content,
                        flags=re.MULTILINE,
                    )
                ]
            )
            match = re.search(
                r"^\s*lowpowermode\s+(\d+)\s*$",
                active_content,
                flags=re.MULTILINE,
            )
            # Some macOS releases emit a single profile without a heading.
            # It is safe to use only when the output contains one unambiguous
            # low-power-mode value.
            if match is None:
                values = re.findall(
                    r"^\s*lowpowermode\s+(\d+)\s*$",
                    content,
                    flags=re.MULTILINE,
                )
                if len(set(values)) == 1:
                    match = re.search(
                        r"^\s*lowpowermode\s+(\d+)\s*$",
                        content,
                        flags=re.MULTILINE,
                    )
            if match:
                low_power_mode = int(match.group(1))
                probe_source = "pmset_custom_active_profile"
                break
        else:
            diagnostics["custom_active_profile_found"].append(False)
            diagnostics["custom_lowpowermode_values"].append([])
        if attempt + 1 < TASK_POWER_MODE_ATTEMPTS:
            time.sleep(TASK_POWER_MODE_RETRY_SECONDS)

    if low_power_mode is None:
        # The active-profile view is an independent final fallback.
        completed = _completed(["/usr/bin/pmset", "-g"])
        diagnostics["active_settings_returncode"] = (
            None if completed is None else int(completed.returncode)
        )
        match = (
            re.search(
                r"^\s*lowpowermode\s+(\d+)\s*$",
                completed.stdout or "",
                flags=re.MULTILINE,
            )
            if completed is not None and completed.returncode == 0
            else None
        )
        if match:
            low_power_mode = int(match.group(1))
            probe_source = "pmset_active_settings"
    if low_power_mode is None:
        errors.append("low_power_mode_unavailable")
    energy_mode = (
        PMSET_ENERGY_MODES.get(low_power_mode, f"unknown_{low_power_mode}")
        if low_power_mode is not None
        else None
    )
    return {
        "power_mode_tag": (
            f"macos-{str(energy_mode).replace('_', '-')}-pmset-{low_power_mode}"
            if low_power_mode is not None
            else "macos-energy-mode-unavailable"
        ),
        "energy_mode": energy_mode,
        "energy_mode_observation_status": (
            "observed" if low_power_mode is not None else "unavailable"
        ),
        "power_source": power_source,
        "pmset_lowpowermode": low_power_mode,
        "probe_source": probe_source,
        "probe_attempts": probe_attempts,
        "probe_errors": errors,
        "probe_diagnostics": diagnostics,
    }


def _windows_system_power_status() -> dict[str, Any]:
    """Read AC/battery state without localized command output."""

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac_line_status", ctypes.c_ubyte),
            ("battery_flag", ctypes.c_ubyte),
            ("battery_life_percent", ctypes.c_ubyte),
            ("battery_saver", ctypes.c_ubyte),
            ("battery_life_time", ctypes.c_uint32),
            ("battery_full_life_time", ctypes.c_uint32),
        ]

    status = SystemPowerStatus()
    try:
        succeeded = bool(
            ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
        )
    except (AttributeError, OSError):
        succeeded = False
    if not succeeded:
        return {
            "power_source": None,
            "battery_saver": None,
            "battery_life_percent": None,
            "battery_flag": None,
            "system_power_status_available": False,
        }
    power_source = {0: "Battery Power", 1: "AC Power"}.get(status.ac_line_status)
    return {
        "power_source": power_source,
        "battery_saver": bool(status.battery_saver),
        "battery_life_percent": (
            None if status.battery_life_percent == 255 else status.battery_life_percent
        ),
        "battery_flag": status.battery_flag,
        "system_power_status_available": True,
    }


def _windows_task_power_state() -> dict[str, Any]:
    errors: list[str] = []
    power_scheme_guid = None
    power_plan = None
    completed = _completed(["powercfg", "/getactivescheme"])
    if completed is not None and completed.returncode == 0:
        guid_match = re.search(
            r"\b([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\b",
            completed.stdout or "",
            flags=re.IGNORECASE,
        )
        if guid_match:
            power_scheme_guid = guid_match.group(1).lower()
        match = re.search(r"\(([^()]*)\)\s*$", completed.stdout or "")
        if match:
            power_plan = match.group(1).strip()
    if not power_scheme_guid:
        errors.append("active_power_plan_unavailable")
    system_power = _windows_system_power_status()
    if system_power["power_source"] is None:
        errors.append("power_source_unavailable")
    canonical_plan = WINDOWS_POWER_SCHEME_NAMES.get(power_scheme_guid or "")
    mode = canonical_plan or power_plan or power_scheme_guid
    tag_component = canonical_plan or (
        f"custom-{power_scheme_guid[:12]}" if power_scheme_guid else None
    )
    return {
        "power_mode_tag": (
            f"windows-power-scheme-{_slug(tag_component)}"
            if tag_component
            else "windows-power-scheme-unavailable"
        ),
        "energy_mode": mode,
        "energy_mode_observation_status": (
            "observed" if power_scheme_guid else "unavailable"
        ),
        "power_source": system_power["power_source"],
        "pmset_lowpowermode": None,
        "power_scheme_guid": power_scheme_guid,
        "power_scheme_name": power_plan,
        "battery_saver": system_power["battery_saver"],
        "battery_life_percent": system_power["battery_life_percent"],
        "probe_source": "powercfg_active_scheme" if power_scheme_guid else None,
        "probe_attempts": 1,
        "probe_errors": errors,
        "probe_diagnostics": {
            "powercfg_returncode": (
                None if completed is None else int(completed.returncode)
            ),
            "system_power_status_available": system_power[
                "system_power_status_available"
            ],
            "battery_flag": system_power["battery_flag"],
        },
    }


def _linux_task_power_state() -> dict[str, Any]:
    errors: list[str] = []
    governor = None
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        governor = governor_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        errors.append("cpu_governor_unavailable")
    return {
        "power_mode_tag": (
            f"linux-governor-{_slug(governor)}"
            if governor
            else "linux-power-mode-unavailable"
        ),
        "energy_mode": governor,
        "energy_mode_observation_status": ("observed" if governor else "unavailable"),
        "power_source": None,
        "pmset_lowpowermode": None,
        "probe_source": "linux_cpu0_scaling_governor" if governor else None,
        "probe_attempts": 1,
        "probe_errors": errors,
        "probe_diagnostics": {
            "governor_path": str(governor_path),
            "governor_read": governor is not None,
        },
    }


def observe_task_power_state(system: str | None = None) -> dict[str, Any]:
    """Capture portable power provenance immediately beside one benchmark task."""

    observed_system = str(system or platform.system())
    if observed_system == "Darwin":
        state = _darwin_task_power_state()
    elif observed_system == "Windows":
        state = _windows_task_power_state()
    elif observed_system == "Linux":
        state = _linux_task_power_state()
    else:
        state = {
            "power_mode_tag": "power-mode-unavailable",
            "energy_mode": None,
            "energy_mode_observation_status": "unavailable",
            "power_source": None,
            "pmset_lowpowermode": None,
            "probe_source": None,
            "probe_attempts": 0,
            "probe_errors": ["unsupported_platform"],
            "probe_diagnostics": {"unsupported_platform": observed_system},
        }
    return {
        "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": observed_system,
        **state,
    }


def summarize_power_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tags = list(dict.fromkeys(observation_power_tag(item) for item in observations))
    return _summary(tags, len(observations))


def summarize_task_power_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the live before/after power tags attached to task records."""

    tags = list(
        dict.fromkeys(
            str(item.get("host_power_mode_tag") or "power-mode-unavailable")
            for item in records
        )
    )
    summary = _summary(tags, 0)
    summary.pop("session_count", None)
    summary["task_count"] = len(records)
    summary["provenance_scope"] = "task_start_and_end"
    return summary


def _summary(tags: list[str], session_count: int) -> dict[str, Any]:
    unavailable = not tags or all("unavailable" in tag for tag in tags)
    mixed = len(tags) > 1 or any("mixed" in tag for tag in tags)
    classification = (
        "unavailable" if unavailable else "single_mode" if not mixed else "mixed_mode"
    )
    contribution_tag = (
        "power-mode-unavailable"
        if unavailable
        else tags[0]
        if not mixed
        else "mixed-power-modes"
    )
    return {
        "classification": classification,
        "tags": tags,
        "session_count": session_count,
        "contribution_tag": contribution_tag,
    }


def combine_power_summaries(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tags: list[str] = []
    session_count = 0
    task_count = 0
    for summary in summaries:
        session_count += int(summary.get("session_count") or 0)
        task_count += int(summary.get("task_count") or 0)
        for raw_tag in summary.get("tags") or []:
            tag = _slug(raw_tag)
            if tag not in tags:
                tags.append(tag)
    combined = _summary(tags, session_count)
    if task_count:
        combined["task_count"] = task_count
        combined["provenance_scope"] = "task_start_and_end"
    return combined


__all__ = [
    "combine_power_summaries",
    "observe_task_power_state",
    "observation_power_tag",
    "summarize_power_observations",
    "summarize_task_power_records",
]

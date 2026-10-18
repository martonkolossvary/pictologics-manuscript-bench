"""Inspect Pictologics' FFT RAM policy without extracting features or timing.

Every native policy probe runs in a disposable interpreter. Synthetic ROI boxes
are integer geometry only: no image, FFT or pairwise calculation is allocated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
PROBES = ("native", "sysconf_missing", "sysconf_oserror", "sysconf_valueerror")


def physical_memory() -> dict:
    """Read total physical RAM independently of the Pictologics implementation."""
    errors = []
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
        if total > 0:
            return {"bytes": total, "method": "psutil.virtual_memory.total"}
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        errors.append(str(exc))
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                            *[(name, ctypes.c_ulonglong) for name in
                              ("total", "available", "total_pagefile", "available_pagefile",
                               "total_virtual", "available_virtual", "extended_virtual")]]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            get_status = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            get_status.argtypes = [ctypes.POINTER(MemoryStatus)]
            get_status.restype = wintypes.BOOL
            if not get_status(ctypes.byref(status)):
                raise ctypes.WinError(ctypes.get_last_error())
            if status.total > 0:
                return {"bytes": int(status.total), "method": "GlobalMemoryStatusEx"}
        except (OSError, ValueError, AttributeError) as exc:
            errors.append(str(exc))
    try:
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        if total > 0:
            return {"bytes": total, "method": "controller_os.sysconf"}
    except (OSError, ValueError, AttributeError) as exc:
        errors.append(str(exc))
    return {"bytes": None, "method": "unavailable", "errors": errors}


def probe_cap(cap_function, mode: str, os_module=os) -> dict:
    """Temporarily simulate sysconf failure; called only in disposable workers."""
    if mode not in PROBES:
        raise ValueError(f"Unknown probe: {mode}")
    sentinel = object()
    original = getattr(os_module, "sysconf", sentinel)

    def failing_sysconf(*_args):
        error = OSError if mode == "sysconf_oserror" else ValueError
        raise error("Diagnostic sysconf failure")

    try:
        if mode == "sysconf_missing":
            if original is not sentinel:
                delattr(os_module, "sysconf")
        elif mode != "native":
            os_module.sysconf = failing_sysconf
        cap = cap_function()
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError(f"Invalid FFT memory cap: {cap!r}")
        return {"probe": mode, "status": "observed", "native_cap_bytes": cap}
    except Exception as exc:
        return {"probe": mode, "status": "failed", "error": str(exc)}
    finally:
        if original is not sentinel:
            os_module.sysconf = original
        elif hasattr(os_module, "sysconf"):
            delattr(os_module, "sysconf")


def predict_legacy_policy(shape, roi_voxels, cap_bytes, next_fast_len) -> dict:
    """An explicitly versioned 0.7.1 model, never a future API assumption."""
    if (len(shape) != 3 or any(not isinstance(n, int) or not 1 <= n <= 1024 for n in shape)
            or not isinstance(roi_voxels, int) or not 1 <= roi_voxels <= math.prod(shape)
            or not isinstance(cap_bytes, int) or cap_bytes < 0):
        raise ValueError("Expected bounded 3D box, valid ROI count and nonnegative cap")
    fft_shape = tuple(next_fast_len(2 * n - 1, real=True) for n in shape)
    grid = math.prod(fft_shape)
    estimate = 32 * grid
    crossover = roi_voxels ** 2 >= 100 * grid
    return {"roi_box_shape": list(shape), "roi_voxels": roi_voxels,
            "fft_shape": list(fft_shape), "fft_estimate_bytes": estimate,
            "cap_bytes": cap_bytes, "crossover_favors_fft": crossover,
            "predicted_method": "fft" if crossover and estimate <= cap_bytes else "pairwise",
            "memory_guard_fallback": crossover and estimate > cap_bytes}


def worker(mode: str) -> dict:
    # Avoid import-time JIT warmup; this diagnostic never calls an extractor.
    os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"
    from pictologics.features import intensity
    from scipy.fft import next_fast_len

    source = Path(intensity.__file__).resolve()
    report = {"version": importlib.metadata.version("pictologics"),
              "source_path": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "python": sys.version, "executable": sys.executable,
              "system": platform.system(), "machine": platform.machine(),
              "native_sysconf_available": callable(getattr(os, "sysconf", None)),
              "scipy_version": importlib.metadata.version("scipy")}
    cap_function = getattr(intensity, "_fft_memory_cap", None)
    if not callable(cap_function):
        report.update(probe=mode, status="unsupported",
                      reason="Native _fft_memory_cap API changed; review the release implementation")
        return report
    report.update(probe_cap(cap_function, mode))
    if mode == "native" and report["status"] == "observed":
        report["legacy_model_predictions"] = [
            predict_legacy_policy((edge,) * 3, edge ** 3, cap, next_fast_len)
            for edge in (8, 256, 320, 384, 512)
            for cap in sorted({4 * GIB, 16 * GIB, report["native_cap_bytes"]})
        ]
    return report


def assess_acceptance(probes: list[dict], physical: dict) -> dict:
    """Require the documented 16 GiB/half-physical-RAM policy on every probe."""
    total = physical.get("bytes")
    if not isinstance(total, int) or total <= 0:
        return {"status": "unavailable", "reason": "Physical RAM could not be independently read"}
    if any(p.get("status") == "unsupported" for p in probes):
        return {"status": "unsupported", "reason": "Private API changed; review and adapt diagnostic"}
    expected = min(16 * GIB, total // 2)
    expected_names = set(PROBES)
    actual_names = [p.get("probe") for p in probes]
    if set(actual_names) != expected_names or len(actual_names) != len(expected_names):
        return {"status": "failed", "reason": "Missing or duplicate probe results"}
    mismatches = [p["probe"] for p in probes if p.get("status") != "observed"
                  or p.get("native_cap_bytes") != expected]
    return {"status": "failed" if mismatches else "passed",
            "expected_cap_bytes": expected, "mismatching_probes": mismatches,
            "policy": "min(16 GiB, total physical RAM // 2)",
            "limitation": "Agreement at 8 GiB RAM alone cannot distinguish correct detection "
                          "from the legacy fallback; include a host above 8 GiB."}


def inspect_policy(python: Path, *, timeout: float = 120.) -> dict:
    physical = physical_memory()
    environment = os.environ.copy()
    environment["PICTOLOGICS_DISABLE_WARMUP"] = "1"
    report = {"schema_version": 1, "kind": "untimed_pictologics_spatial_policy",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "physical_memory": physical, "probes": [], "timing": "not_performed",
              "prediction_model": "Pictologics 0.7.1, 32 bytes/FFT grid point, "
                                  "N squared >= 100 * grid; illustrative for future releases",
              "prediction_limitations": "Integer geometry only; no arrays, measured peak memory, "
                                        "runtime estimate or benchmark feasibility claim"}
    for mode in PROBES:
        try:
            process = subprocess.run([str(python), str(Path(__file__).resolve()), "--worker", mode],
                                     capture_output=True, text=True, env=environment,
                                     timeout=timeout, check=False)
            if process.returncode:
                raise RuntimeError(process.stderr.strip() or process.stdout.strip())
            result = json.loads(process.stdout)
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
            result = {"probe": mode, "status": "failed", "error": str(exc)}
        report["probes"].append(result)
    report["acceptance"] = assess_acceptance(report["probes"], physical)
    report["observations_complete"] = all(p.get("status") == "observed" for p in report["probes"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-repo", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, help="Explicit candidate environment interpreter")
    parser.add_argument("--output", type=Path, help="JSON report path; otherwise print full report")
    parser.add_argument("--timeout", type=float, default=120., help="Worker safety limit; not a timing run")
    parser.add_argument("--require-physical-ram-cap", action="store_true",
                        help="Fail unless all probes follow the documented physical-RAM policy")
    parser.add_argument("--worker", choices=PROBES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args.worker), allow_nan=False))
        return 0
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    python = args.python
    if python is None:
        from bench.env import env_python, load_runtime_profiles
        profile = load_runtime_profiles(args.runtime_repo / "configs" / "adapters")["pictologics"]
        python = env_python(args.runtime_repo / profile.env_dir)
    # POSIX venv launchers are commonly symlinks to the base interpreter. Resolving
    # the symlink bypasses pyvenv.cfg and silently inspects the wrong environment.
    report = inspect_policy(python.absolute(), timeout=args.timeout)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output.resolve()),
                          "observations_complete": report["observations_complete"],
                          "acceptance": report["acceptance"]}, indent=2))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    if not report["observations_complete"]:
        return 2
    return int(args.require_physical_ram_cap and report["acceptance"]["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())

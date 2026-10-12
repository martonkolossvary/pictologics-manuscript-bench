#!/usr/bin/env python3
"""Read live ledgers without modifying them and export feasibility evidence.

Each SQLite read transaction includes committed WAL rows. A missing pillar is
reported as missing, and pending configurations never count as successful.
Exports are diagnostic observations, not publication speed comparisons.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


PILLARS = ("pillar1_morphology", "pillar2_whole_anatomy", "pillar3_ibsi2_phase3")
ISSUE_STATUSES = {
    "timed_out_censored", "skipped_timeout_cutoff", "memory_limit_exceeded",
    "skipped_memory_preflight", "skipped_memory_cutoff", "failed", "interrupted",
}
FIELDS = (
    "pillar", "case_id", "adapter", "workload", "families", "shape", "spacing",
    "image_voxels", "mask_voxels", "mask_id", "subject_id", "representation_id",
    "configured_levels", "occupied_levels", "discretization", "bins", "bin_width",
    "intensity_min", "intensity_max", "repeat", "status", "attempt",
    "whole_task_elapsed_seconds", "calculation_median_seconds", "censor_lower_bound_seconds", "error",
)


def summarize(machine_root: Path) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    pillars: dict = {}
    for pillar in PILLARS:
        path = machine_root / pillar / "benchmark.sqlite3"
        if not path.is_file():
            pillars[pillar] = {"status": "missing", "configurations": 0}
            continue
        # Do not use immutable=1: it ignores live WAL commits.
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            task_rows = db.execute("SELECT * FROM tasks ORDER BY ordinal").fetchall()
            metadata = {r[0]: r[1] for r in db.execute("SELECT key, value FROM metadata")}
        states: dict[tuple, set[str]] = {}
        counts: Counter = Counter()
        for task in task_rows:
            spec = json.loads(task["spec_json"])
            record = json.loads(task["record_json"]) if task["record_json"] else {}
            status = task["status"]
            key = (task["case_id"], task["adapter"], task["workload_key"])
            states.setdefault(key, set()).add(status)
            counts[status] += 1
            row = {field: spec.get(field, "") for field in FIELDS}
            row.update(
                pillar=pillar, adapter=task["adapter"], workload=task["workload_key"],
                case_id=task["case_id"], repeat=task["repeat"], status=status,
                attempt=task["attempt"], error=task["error"] or "",
                families=";".join(spec.get("requested_families", [])),
                shape="x".join(map(str, spec.get("shape", []))),
                spacing="x".join(map(str, spec.get("spacing", []))),
                whole_task_elapsed_seconds=record.get("host_wall_time_sec"),
                calculation_median_seconds=task["duration_sec"] if status == "measured" else None,
                censor_lower_bound_seconds=record.get("censor_lower_bound_sec"),
            )
            rows.append(row)
        # A measured first process establishes feasibility, even though later
        # repeats remain pending. Failures/timeouts are never successful screens.
        successful = sum("measured" in statuses and not statuses & ISSUE_STATUSES for statuses in states.values())
        mixed = sum("measured" in statuses and bool(statuses & ISSUE_STATUSES) for statuses in states.values())
        unattempted = sum(statuses == {"pending"} for statuses in states.values())
        in_progress = sum("running" in statuses for statuses in states.values())
        pillars[pillar] = {
            "status": "present", "task_status_counts": dict(sorted(counts.items())),
            "configurations": len(states), "successful_configurations": successful,
            "unattempted_configurations": unattempted,
            "mixed_success_and_issue_configurations": mixed,
            "in_progress_configurations": in_progress,
            "other_configurations": len(states) - successful - mixed - unattempted,
            "run_status": metadata.get("run_status"),
        }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "feasibility_diagnostic_not_speed_comparison",
        "whole_task_elapsed_is_not_per_call_runtime": True,
        "pillars": pillars,
        "adapter_status_counts": {
            adapter: dict(sorted(Counter(r["status"] for r in rows if r["adapter"] == adapter).items()))
            for adapter in sorted({r["adapter"] for r in rows})
        },
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = summarize(args.machine_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in (
        ("tasks.csv", rows),
        ("resource_issues.csv", [row for row in rows if row["status"] in ISSUE_STATUSES]),
    ):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(selected)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

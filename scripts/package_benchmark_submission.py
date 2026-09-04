#!/usr/bin/env python3
"""Create or validate commit-ready, multi-machine benchmark result bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.submission import package_submission, validate_submission_tree


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="results/benchmark")
    parser.add_argument("--output-root", default="benchmark-results")
    parser.add_argument("--machine-id")
    parser.add_argument("--submission-date", default=None)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repository / output_root
    if args.validate:
        failures = validate_submission_tree(output_root)
        print(json.dumps({"status": "fail" if failures else "pass", "failures": failures}, indent=2))
        return 1 if failures else 0
    if not args.machine_id:
        parser.error("--machine-id is required when packaging")
    destination = package_submission(
        repository=repository,
        result_root=Path(args.result_root),
        output_root=output_root,
        machine_id=args.machine_id,
        submission_date=args.submission_date,
    )
    print(f"Submission created: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

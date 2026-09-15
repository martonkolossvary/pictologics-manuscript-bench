#!/usr/bin/env python3
"""Preview or apply the audited ledger and screening-order fix to stopped runs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.benchmark_ledger import BenchmarkLedger, RunLock
from bench.benchmark_recovery import plan_execution_repair, apply_repair, sync_recovery_exports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Back up and amend controller provenance; never starts workers")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.apply:
        state = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                               cwd=repository, capture_output=True, text=True, check=True)
        if state.stdout.strip():
            parser.error("commit the reviewed fix before applying it; source must be clean")
    roots = [p for p in sorted(args.machine_root.iterdir()) if (p/"benchmark.sqlite3").is_file()]
    if not roots:
        parser.error("no pillar ledgers found")
    with ExitStack() as stack:
        for root in roots:
            stack.enter_context(RunLock(root/".benchmark.lock"))
        ledgers = [stack.enter_context(BenchmarkLedger(root/"benchmark.sqlite3")) for root in roots]
        plans = [plan_execution_repair(ledger, repository) for ledger in ledgers]
        recovery = args.machine_root/"recovery"/time.strftime("execution-order-%Y%m%d-%H%M%S")
        for ledger, plan in zip(ledgers, plans):
            print(json.dumps({"pillar": ledger.path.parent.name, "already_repaired": plan["already_repaired"],
                              "verified_measurements": plan["measured"], "new_fingerprint": plan["new_fingerprint"]}), flush=True)
            if args.apply:
                result = apply_repair(ledger, plan, recovery/ledger.path.parent.name)
                if ledger.protocol_amendments():
                    sync_recovery_exports(ledger, repository)
                print(json.dumps(result), flush=True)
    print("Repair applied. Resume with the normal launcher." if args.apply else "Preview only; task states and payloads are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

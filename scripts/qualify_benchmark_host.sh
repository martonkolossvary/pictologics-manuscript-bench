#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 RESULT_ROOT [HOST_PROFILE]" >&2
    exit 2
fi

RESULT_ROOT="$1"
HOST_PROFILE="${2:-configs/benchmark/hosts/mac-m4pro-01.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

poetry check
poetry run ruff check .
poetry run pytest -q
poetry run python -m bench.cli env verify
poetry run python scripts/launch_benchmark.py \
    --workspace-root data/benchmark \
    --result-root "${RESULT_ROOT}" \
    --host-profile "${HOST_PROFILE}" \
    --validate-plans

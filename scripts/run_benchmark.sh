#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# The Python launcher is the single cross-platform execution contract. It is
# print-only unless the caller explicitly supplies --execute --confirm CALCULATE.
LAUNCHER_ARGS=("$@")
if [[ "$(uname -s)" == "Darwin" ]]; then
    HAS_HOST_PROFILE=false
    for argument in "${LAUNCHER_ARGS[@]}"; do
        if [[ "${argument}" == "--host-profile" ]]; then
            HAS_HOST_PROFILE=true
            break
        fi
    done
    if [[ "${HAS_HOST_PROFILE}" == false ]]; then
        LAUNCHER_ARGS+=(--host-profile configs/benchmark/hosts/mac-m4pro-01.json)
    fi
    # The host profile requires a live sleep-prevention assertion during real
    # calculations. The launcher verifies that this assertion is visible.
    exec /usr/bin/caffeinate -dimsu poetry run python \
        scripts/launch_benchmark.py "${LAUNCHER_ARGS[@]}"
fi

exec poetry run python scripts/launch_benchmark.py "${LAUNCHER_ARGS[@]}"

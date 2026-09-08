#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if ! command -v poetry >/dev/null 2>&1; then
    echo "Poetry is required to install the benchmark controller." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1 || [[ ! "$(uv --version)" =~ ^uv\ 0\.8\.[0-9]+($|\ ) ]]; then
    echo "uv >=0.8,<0.9 is required for the pinned MEDimage wheel on every platform." >&2
    exit 1
fi

poetry sync
# Existing valid environments are verified and reused. Rebuilding requires the
# user's explicit `bench env create --force` command.
poetry run python -m bench.cli env create
poetry run python -m bench.cli env verify

echo "Controller and five isolated adapter environments are verified."

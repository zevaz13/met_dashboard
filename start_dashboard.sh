#!/usr/bin/env bash
# Starts the MET dashboard (Linux/macOS). Requires uv: https://docs.astral.sh/uv/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed or not on PATH." >&2
    echo "Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv sync
exec uv run streamlit run dashboard/Home.py

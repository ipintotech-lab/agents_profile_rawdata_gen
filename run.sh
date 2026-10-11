#!/usr/bin/env bash
# Run the interactive agent trace generator.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Error: $PYTHON not found. Install Python 3.10+ or set PYTHON=/path/to/python." >&2
    exit 1
fi

exec "$PYTHON" run_generator.py "$@"

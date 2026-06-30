#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="$(cd "$SCRIPT_DIR/../../.." && pwd)/.venv"
PYTHON="$ENV_PREFIX/bin/python"

if [[ ! -f "$PYTHON" ]]; then
  echo "ERROR: Failed to find python in env: $PYTHON" >&2
  exit 1
fi

pushd "$SCRIPT_DIR" >/dev/null
"$PYTHON" -m pytest -v unit_tests -o python_files=ut_*.py
popd >/dev/null
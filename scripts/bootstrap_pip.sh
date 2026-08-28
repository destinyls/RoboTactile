#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: bootstrap_pip.sh [VENV_PATH]

Create a pip-only, hash-locked RoboTactile development environment.
The default environment path is <repository>/.venv.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

VENV_PATH="${1:-$PROJECT_ROOT/.venv}"
if [[ -L "$VENV_PATH" ]]; then
  echo "virtual environment path must not be a symlink: $VENV_PATH" >&2
  exit 2
fi

"$PYTHON_BIN" -c \
  'import sys; assert sys.version_info >= (3, 9), "RoboTactile requires Python >= 3.9"'
"$PYTHON_BIN" -m venv "$VENV_PATH"

VENV_PYTHON="$VENV_PATH/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1
"$VENV_PYTHON" -m ensurepip --upgrade
"$VENV_PYTHON" -m pip install \
  --require-hashes \
  -r "$PROJECT_ROOT/requirements/dev.lock.txt"
"$VENV_PYTHON" -m pip install \
  --no-deps \
  --no-build-isolation \
  "$PROJECT_ROOT"
"$VENV_PYTHON" -m pip check
"$VENV_PYTHON" -c \
  'import robotactile_benchmark; print(robotactile_benchmark.__version__)'
"$VENV_PYTHON" -m robotactile_benchmark.cli integrations list >/dev/null

printf 'RoboTactile pip environment ready: %s\n' "$VENV_PYTHON"

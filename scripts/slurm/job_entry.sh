#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: job_entry.sh MODE REPO_ROOT DEPLOY_ROOT PYTHON OUTPUT_DIR" >&2
  exit 2
fi

MODE="$1"
REPO_ROOT="$2"
DEPLOY_ROOT="$3"
PYTHON_BIN="$4"
OUTPUT_DIR="$5"
OUTPUT_DIR="${OUTPUT_DIR//%j/${SLURM_JOB_ID:-nojob}}"

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
  echo "mode must be cpu or gpu" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

export ROBOTACTILE_ROOT="$REPO_ROOT"
export ROBOTACTILE_DEPLOY_ROOT="$DEPLOY_ROOT"
exec "$PYTHON_BIN" "$REPO_ROOT/scripts/slurm/smoke_worker.py" \
  "$MODE" \
  --repo-root "$REPO_ROOT" \
  --deploy-root "$DEPLOY_ROOT" \
  --output "$OUTPUT_DIR"

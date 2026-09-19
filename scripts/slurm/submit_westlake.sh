#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/slurm/submit_westlake.sh {cpu|gpu}

Environment overrides:
  ROBOTACTILE_SLURM_PARTITION  Slurm partition (default: yukaichenglab)
  ROBOTACTILE_SLURM_ACCOUNT    Slurm account (default: yukaicheng)
  ROBOTACTILE_SLURM_QOS        Slurm QoS (default: normal)
  ROBOTACTILE_CLUSTER_PYTHON   Existing Python with NumPy for CPU smoke
  ROBOTACTILE_DEPLOY_ROOT      Writable deployment root
EOF
}

if [[ $# -ne 1 || ( "$1" != "cpu" && "$1" != "gpu" ) ]]; then
  usage >&2
  exit 2
fi

MODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
DEPLOY_ROOT="${ROBOTACTILE_DEPLOY_ROOT:-$REPO_ROOT/deployment}"
PARTITION="${ROBOTACTILE_SLURM_PARTITION:-yukaichenglab}"
ACCOUNT="${ROBOTACTILE_SLURM_ACCOUNT:-yukaicheng}"
QOS="${ROBOTACTILE_SLURM_QOS:-normal}"
PYTHON_BIN="${ROBOTACTILE_CLUSTER_PYTHON:-$HOME/anaconda3/bin/python3}"
SBATCH_BIN="${ROBOTACTILE_SBATCH:-/soft/slurm/bin/sbatch}"

if [[ "$DEPLOY_ROOT" != "$REPO_ROOT/deployment" ]]; then
  echo "cluster smoke requires repo-local deployment: $REPO_ROOT/deployment" >&2
  exit 2
fi
if [[ ! -x "$SBATCH_BIN" ]]; then
  echo "sbatch is unavailable: $SBATCH_BIN (use a login shell)" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p \
  "$DEPLOY_ROOT/runtime/slurm/jobs" \
  "$DEPLOY_ROOT/logs/slurm" \
  "$DEPLOY_ROOT/outputs/slurm-smoke"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$DEPLOY_ROOT/outputs/slurm-smoke/${MODE}-${TIMESTAMP}-%j"
COMMON_ARGS=(
  --parsable
  --job-name="robotactile-${MODE}-smoke"
  --partition="$PARTITION"
  --account="$ACCOUNT"
  --qos="$QOS"
  --nodes=1
  --ntasks=1
  --cpus-per-task=2
  --mem=4G
  --time=00:05:00
  --chdir="$REPO_ROOT"
  --output="$DEPLOY_ROOT/logs/slurm/${MODE}-${TIMESTAMP}-%j.out"
  --error="$DEPLOY_ROOT/logs/slurm/${MODE}-${TIMESTAMP}-%j.err"
)
if [[ "$MODE" == "gpu" ]]; then
  COMMON_ARGS+=(--gres=gpu:1)
else
  "$PYTHON_BIN" -c 'import numpy' >/dev/null
fi

JOB_ID="$("$SBATCH_BIN" "${COMMON_ARGS[@]}" \
  "$REPO_ROOT/scripts/slurm/job_entry.sh" \
  "$MODE" "$REPO_ROOT" "$DEPLOY_ROOT" "$PYTHON_BIN" "$OUTPUT_DIR")"

"$PYTHON_BIN" - \
  "$DEPLOY_ROOT/runtime/slurm/jobs/${JOB_ID}.json" \
  "$JOB_ID" "$MODE" "$PARTITION" "$ACCOUNT" "$QOS" \
  "$REPO_ROOT" "$DEPLOY_ROOT" "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "job_id": sys.argv[2],
    "mode": sys.argv[3],
    "partition": sys.argv[4],
    "account": sys.argv[5],
    "qos": sys.argv[6],
    "repository_root": sys.argv[7],
    "deployment_root": sys.argv[8],
    "output_directory_template": sys.argv[9],
    "evidence_level": "SUBMITTED",
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf '%s\n' "$JOB_ID"

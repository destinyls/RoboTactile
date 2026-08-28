#!/usr/bin/env bash
# Run one unqualified diagnostic N0 Clean episode for every frozen task.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
DEPLOY_ROOT="$REPOSITORY_ROOT/deployment"
QUALIFICATION=""
GPUS="0"
CAMPAIGN_ID="n0-clean-all8-once-v1"
RUN_ID="all8-once-v1"
MASTER_SEED="20260823"
WAIT_TIMEOUT_SECONDS="14400"

usage() {
  cat <<'EOF'
Usage: run_clean_all_tasks_once.sh [OPTIONS]

Options:
  --root PATH                    Deployment root.
  --qualification PATH           Optional source-bound qualification v3 receipt.
  --gpus LIST                    Comma-separated N0 GPU indices (default: 0).
  --campaign-id ID               Diagnostic campaign identity.
  --run-id ID                    No-clobber master run identity.
  --master-seed INTEGER          Frozen campaign seed root.
  --qualification-timeout-s N    Wait limit when --qualification is set (default: 14400).

This executes exactly one Clean episode for each of the eight frozen tasks.
It is an end-to-end diagnostic and cannot authorize a paper success-rate claim.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; DEPLOY_ROOT="$2"; shift 2 ;;
    --qualification) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; QUALIFICATION="$2"; shift 2 ;;
    --gpus) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; GPUS="$2"; shift 2 ;;
    --campaign-id) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; CAMPAIGN_ID="$2"; shift 2 ;;
    --run-id) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; RUN_ID="$2"; shift 2 ;;
    --master-seed) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; MASTER_SEED="$2"; shift 2 ;;
    --qualification-timeout-s) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; WAIT_TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$GPUS" in ''|*[!0-9,]*) echo "invalid GPU list" >&2; exit 2 ;; esac
case "$MASTER_SEED" in ''|*[!0-9]*) echo "invalid master seed" >&2; exit 2 ;; esac
case "$WAIT_TIMEOUT_SECONDS" in ''|*[!0-9]*) echo "invalid timeout" >&2; exit 2 ;; esac
for identifier in "$CAMPAIGN_ID" "$RUN_ID"; do
  case "$identifier" in
    ''|*[!A-Za-z0-9._-]*) echo "invalid identity" >&2; exit 2 ;;
  esac
done
[ "$WAIT_TIMEOUT_SECONDS" -ge 60 ] || { echo "timeout must be at least 60" >&2; exit 2; }

DEPLOY_ROOT="$(cd "$DEPLOY_ROOT" && pwd -P)"
PYTHON_BIN="$DEPLOY_ROOT/runtime/n0-twam/bin/python"
[ -x "$PYTHON_BIN" ] || { echo "N0 runtime Python is unavailable" >&2; exit 2; }
if [ -n "$QUALIFICATION" ]; then
  QUALIFICATION="$(
  "$PYTHON_BIN" - "$DEPLOY_ROOT" "$QUALIFICATION" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
selected = Path(sys.argv[2]).absolute()
try:
    relative = selected.relative_to(root)
except ValueError as error:
    raise SystemExit("qualification must remain below deployment root") from error
if relative.parts[:2] != ("artifacts", "deployment"):
    raise SystemExit("qualification must remain below artifacts/deployment")
print(selected)
PY
  )"

  deadline="$((SECONDS + WAIT_TIMEOUT_SECONDS))"
  until [ -f "$QUALIFICATION" ] && [ ! -L "$QUALIFICATION" ]; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "qualification receipt wait timed out" >&2
      exit 124
    fi
    sleep 10
  done
fi

cd "$REPOSITORY_ROOT"
MANIFEST="$DEPLOY_ROOT/requests/clean-campaigns/$CAMPAIGN_ID/campaign_manifest.json"
if [ ! -f "$MANIFEST" ]; then
  "$PYTHON_BIN" scripts/n0_twam/generate_clean_campaign_requests.py \
    --root "$DEPLOY_ROOT" \
    --campaign-id "$CAMPAIGN_ID" \
    --protocol diagnostic_v1 \
    --master-seed "$MASTER_SEED" \
    --trials-per-task 1
fi

RUN_ARGUMENTS=(
  --root "$DEPLOY_ROOT"
  --manifest "$MANIFEST"
  --execution-profile quick
  --gpus "$GPUS"
  --run-id "$RUN_ID"
  --max-new-trials-per-task 1
)
if [ -n "$QUALIFICATION" ]; then
  RUN_ARGUMENTS+=(--qualification "$QUALIFICATION")
fi

exec "$PYTHON_BIN" scripts/n0_twam/run_clean_campaign_all_tasks.py \
  "${RUN_ARGUMENTS[@]}"

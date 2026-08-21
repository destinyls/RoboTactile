#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

DEPLOY_ROOT="/data1/yanglei/robotactile_univtac_20260821"
GPU_INDEX="1"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

usage() {
  cat <<'EOF'
Usage: smoke_isaac_sim_4_5.sh [--root PATH] [--gpu INDEX] [--run-id ID]

Runs the Isaac Sim 4.5.0 standalone hello-world example in headless mode on
one explicit NVIDIA GPU. A passed receipt means this launch command exited
zero; it is not a UniVTAC task-success or closed-loop qualification receipt.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --gpu)
      [ "$#" -ge 2 ] || die "--gpu requires a value"
      GPU_INDEX="$2"
      shift 2
      ;;
    --run-id)
      [ "$#" -ge 2 ] || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$GPU_INDEX" in
  ''|*[!0-9]*) die "GPU index must be a non-negative integer" ;;
esac
case "$RUN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "run ID may contain only letters, digits, dot, dash, underscore" ;;
esac

initialize_layout "$DEPLOY_ROOT"
require_command nvidia-smi
acquire_lock "isaac-sim-smoke-$RUN_ID"
trap release_lock EXIT

INSTALL_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
INSTALL_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json"
SMOKE_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaac_sim_smoke_${RUN_ID}.json"
EXAMPLE_PATH="$INSTALL_PATH/standalone_examples/api/isaacsim.simulation_app/hello_world.py"

[ -f "$INSTALL_RECEIPT" ] || die "Isaac Sim install receipt is absent"
[ -x "$INSTALL_PATH/python.sh" ] || die "Isaac Sim python.sh is absent or not executable"
[ -f "$EXAMPLE_PATH" ] || die "Isaac Sim hello-world example is absent: $EXAMPLE_PATH"
INSTALL_SHA256="$(receipt_field "$INSTALL_RECEIPT" source_sha256)"
validate_sha256 "$INSTALL_SHA256"
if ! receipt_matches \
  "$INSTALL_RECEIPT" \
  "component=isaac_sim" \
  "version=4.5.0" \
  "status=installed" \
  "install_path=runtime/isaac-sim-4.5.0"; then
  die "Isaac Sim install receipt is incompatible"
fi

if [ -e "$SMOKE_RECEIPT" ]; then
  if receipt_matches \
    "$SMOKE_RECEIPT" \
    "component=isaac_sim_headless_smoke" \
    "version=4.5.0" \
    "status=passed" \
    "gpu_index=$GPU_INDEX" \
    "install_source_sha256=$INSTALL_SHA256"; then
    info "headless smoke run already has a matching passed receipt: $RUN_ID"
    printf '%s\n' "$SMOKE_RECEIPT"
    exit 0
  fi
  die "refusing to overwrite a different smoke receipt: $SMOKE_RECEIPT"
fi

GPU_IDENTITY="$(nvidia-smi \
  --query-gpu=name,driver_version \
  --format=csv,noheader \
  -i "$GPU_INDEX")" || die "nvidia-smi failed for GPU $GPU_INDEX"
[ -n "$GPU_IDENTITY" ] || die "nvidia-smi returned an empty GPU identity"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export ISAAC_SIM_PATH="$INSTALL_PATH"
LOG_PATH="$(new_log_path "isaac-sim-headless-smoke-$RUN_ID")"
if ! run_logged "$LOG_PATH" "$INSTALL_PATH/python.sh" "$EXAMPLE_PATH" --headless; then
  die "Isaac Sim headless smoke failed; see $LOG_PATH"
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$SMOKE_RECEIPT" \
  "component=isaac_sim_headless_smoke" \
  "version=4.5.0" \
  "status=passed" \
  "evidence_boundary=infrastructure_launch_only" \
  "gpu_index=$GPU_INDEX" \
  "gpu_identity=$GPU_IDENTITY" \
  "install_source_sha256=$INSTALL_SHA256" \
  "example_path=runtime/isaac-sim-4.5.0/standalone_examples/api/isaacsim.simulation_app/hello_world.py" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "Isaac Sim 4.5.0 headless infrastructure smoke passed"
printf '%s\n' "$SMOKE_RECEIPT"

#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../live_univtac/common.sh
source "$SCRIPT_DIR/../live_univtac/common.sh"

DEPLOY_ROOT="$(default_deployment_root)"
TASK=""
PORT=29601
MASTER_PORT=29988
GPUS=""
DEBUG_OFFLOAD=false
OBSERVED_TACTILE_ABSENCE=false
SESSION_ID=""
ATTESTATION=""
QUALIFICATION=""
INTEGRATION_CONFIG=""

usage() {
  cat <<'EOF'
Usage: serve_univtac.sh --task TASK --gpus 0[,1,...] [options]

Options:
  --root PATH         repo-contained deployment root
  --port PORT         websocket port (default: 29601)
  --master-port PORT  torch distributed port (default: 29988)
  --debug-offload     keep VAE/T5 on CPU for a slow functional smoke only
  --enable-observed-tactile-absence
                      enable non-paper training-consistent tactile CFG dropout
  --session-id ID     source-bound shard session identity
  --attestation PATH  rank-zero canonical runtime attestation output
  --qualification PATH source-bound qualification v3 artifact
  --integration-config PATH task-specific N0 integration config

The official fast path requires at least 40 GB on every selected GPU because
each rank loads the full model before FSDP2 sharding. Debug offload can start on
24 GB GPUs but is not suitable for latency or benchmark-throughput claims.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --task) [ "$#" -ge 2 ] || die "--task requires a value"; TASK="$2"; shift 2 ;;
    --gpus) [ "$#" -ge 2 ] || die "--gpus requires a value"; GPUS="$2"; shift 2 ;;
    --port) [ "$#" -ge 2 ] || die "--port requires a value"; PORT="$2"; shift 2 ;;
    --master-port) [ "$#" -ge 2 ] || die "--master-port requires a value"; MASTER_PORT="$2"; shift 2 ;;
    --session-id) [ "$#" -ge 2 ] || die "--session-id requires a value"; SESSION_ID="$2"; shift 2 ;;
    --attestation) [ "$#" -ge 2 ] || die "--attestation requires a value"; ATTESTATION="$2"; shift 2 ;;
    --qualification) [ "$#" -ge 2 ] || die "--qualification requires a value"; QUALIFICATION="$2"; shift 2 ;;
    --integration-config) [ "$#" -ge 2 ] || die "--integration-config requires a value"; INTEGRATION_CONFIG="$2"; shift 2 ;;
    --debug-offload) DEBUG_OFFLOAD=true; shift ;;
    --enable-observed-tactile-absence) OBSERVED_TACTILE_ABSENCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

initialize_layout "$DEPLOY_ROOT"
[ -n "$TASK" ] || die "--task is required"
case "$GPUS" in ''|*[!0-9,]*) die "--gpus must be a comma-separated integer list" ;; esac
case "$PORT:$MASTER_PORT" in *[!0-9:]*) die "ports must be integers" ;; esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "--port is out of range"
[ "$MASTER_PORT" -ge 1 ] && [ "$MASTER_PORT" -le 65535 ] || die "--master-port is out of range"

SOURCE_BOUND=false
if [ -n "$SESSION_ID$ATTESTATION$QUALIFICATION$INTEGRATION_CONFIG" ]; then
  [ -n "$SESSION_ID" ] && [ -n "$ATTESTATION" ] && [ -n "$QUALIFICATION" ] && [ -n "$INTEGRATION_CONFIG" ] || \
    die "source-bound server arguments must be supplied together"
  [ "$DEBUG_OFFLOAD" = false ] || die "source-bound paper serving forbids --debug-offload"
  [ "$OBSERVED_TACTILE_ABSENCE" = false ] || \
    die "source-bound paper serving forbids the tactile-drop diagnostic overlay"
  [ -f "$QUALIFICATION" ] && [ ! -L "$QUALIFICATION" ] || die "qualification must be a regular file"
  [ -f "$INTEGRATION_CONFIG" ] && [ ! -L "$INTEGRATION_CONFIG" ] || die "integration config must be a regular file"
  [ ! -e "$ATTESTATION" ] && [ ! -L "$ATTESTATION" ] || die "attestation output already exists"
  SOURCE_BOUND=true
fi

RUNTIME_ROOT="$DEPLOY_ROOT/runtime/n0-twam"
SOURCE_ROOT="$DEPLOY_ROOT/sources/N0-TWAM"
MODEL_ROOT="$DEPLOY_ROOT/artifacts/models/n0_twam"
SERVE_BUNDLE="$MODEL_ROOT/serve-bundle"
SERVE_POOL="$MODEL_ROOT/serve-pools/$TASK"
OUTPUT="$DEPLOY_ROOT/outputs/n0-twam/$TASK"
[ -x "$RUNTIME_ROOT/bin/python" ] || die "official N0 runtime is absent"
[ -f "$SOURCE_ROOT/n0_twam/n0_twam_server.py" ] || die "official N0 source is absent"
[ -d "$SERVE_BUNDLE/transformer" ] || die "prepared N0 serve bundle is absent"
[ -f "$SERVE_POOL/norm_stat_per_robot.json" ] || die "per-task N0 normalizer is absent"
mkdir -p "$OUTPUT"
PACKAGE_PATH="${ROBOTACTILE_PACKAGE_PATH:-$ROBOTACTILE_REPOSITORY_ROOT/src}"
case "$PACKAGE_PATH" in /*) ;; *) die "ROBOTACTILE_PACKAGE_PATH must be absolute" ;; esac
case "$PACKAGE_PATH" in *$'\n'*) die "ROBOTACTILE_PACKAGE_PATH contains a newline" ;; esac
[ ! -L "$PACKAGE_PATH" ] || die "ROBOTACTILE_PACKAGE_PATH cannot be a symlink"
[ -d "$PACKAGE_PATH" ] || [ -f "$PACKAGE_PATH" ] || \
  die "ROBOTACTILE_PACKAGE_PATH is unavailable: $PACKAGE_PATH"
export PYTHONPATH="$PACKAGE_PATH:$SOURCE_ROOT:$SOURCE_ROOT/n0_twam"
SERVE_TASK="$("$RUNTIME_ROOT/bin/python" -c \
  'from robotactile_benchmark.integrations.n0_twam.artifacts import serve_task_id; import sys; print(serve_task_id(sys.argv[1]))' \
  "$TASK")" || die "task is not supported by the official N0 artifact contract"

OLD_IFS="$IFS"
IFS=','
set -- $GPUS
IFS="$OLD_IFS"
NPROC="$#"
[ "$NPROC" -ge 1 ] || die "official N0 serving requires at least one GPU"
for GPU_INDEX in "$@"; do
  GPU_MEMORY="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)" || \
    die "GPU $GPU_INDEX is unavailable"
  case "$GPU_MEMORY" in ''|*[!0-9]*) die "cannot read GPU $GPU_INDEX memory" ;; esac
  if [ "$DEBUG_OFFLOAD" = false ] && [ "$GPU_MEMORY" -lt 40000 ]; then
    die "GPU $GPU_INDEX has ${GPU_MEMORY} MiB; official fast serving requires >=40000 MiB per selected GPU"
  fi
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export TWAM_SERVE_POOL="$SERVE_POOL"
export TWAM_SERVE_TASK="$SERVE_TASK"
export TWAM_SERVE_ACTION_MODE="delta"
export TWAM_SERVE_BUNDLE="$SERVE_BUNDLE"
export TWAM_SERVE_OUT="$OUTPUT"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT

info "starting official N0-TWAM task=$TASK on GPUs=$GPUS websocket_port=$PORT debug_offload=$DEBUG_OFFLOAD tactile_absence_overlay=$OBSERVED_TACTILE_ABSENCE"
OFFLOAD_ARGUMENTS=()
if [ "$DEBUG_OFFLOAD" = true ]; then
  OFFLOAD_ARGUMENTS+=(--debug-offload)
fi
if [ "$OBSERVED_TACTILE_ABSENCE" = true ]; then
  OFFLOAD_ARGUMENTS+=(--enable-observed-tactile-absence)
fi
SOURCE_BOUND_ARGUMENTS=()
if [ "$SOURCE_BOUND" = true ]; then
  SOURCE_BOUND_ARGUMENTS+=(
    --deployment-root "$DEPLOY_ROOT"
    --task "$TASK"
    --session-id "$SESSION_ID"
    --attestation "$ATTESTATION"
    --qualification "$QUALIFICATION"
    --integration-config "$INTEGRATION_CONFIG"
    --n0-source-root "$SOURCE_ROOT"
  )
fi

if [ "$NPROC" -eq 1 ]; then
  export RANK=0
  export LOCAL_RANK=0
  export WORLD_SIZE=1
  export LOCAL_WORLD_SIZE=1
  exec "$RUNTIME_ROOT/bin/python" \
    "$ROBOTACTILE_REPOSITORY_ROOT/scripts/n0_twam/serve_official.py" \
    --port "$PORT" \
    --save-root "$OUTPUT" \
    "${SOURCE_BOUND_ARGUMENTS[@]}" \
    "${OFFLOAD_ARGUMENTS[@]}"
fi

[ -x "$RUNTIME_ROOT/bin/torchrun" ] || die "official N0 torchrun is absent"
exec "$RUNTIME_ROOT/bin/torchrun" \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ROBOTACTILE_REPOSITORY_ROOT/scripts/n0_twam/serve_official.py" \
  --port "$PORT" \
  --save-root "$OUTPUT" \
  "${SOURCE_BOUND_ARGUMENTS[@]}" \
  "${OFFLOAD_ARGUMENTS[@]}"

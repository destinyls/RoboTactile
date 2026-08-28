#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

DEPLOY_ROOT=""
CUDA_RUNFILE=""
CUDA_SHA256=""
ISAAC_ARCHIVE=""
ISAAC_SHA256=""
ROBOTACTILE_WHEEL=""
MICROMAMBA_ARCHIVE=""
TINYGLTF_ARCHIVE=""
GPU_INDEX="0"

usage() {
  cat <<'EOF'
Usage: bootstrap_blackwell_sm120.sh --root PATH
       --cuda-runfile FILE --cuda-sha256 HEX
       --isaac-archive FILE --isaac-sha256 HEX
       --wheel FILE [--micromamba-archive FILE]
       [--tinygltf-archive FILE] [--gpu INDEX]

Builds an isolated CUDA 12.8/sm_120 RoboTactile deployment for a Blackwell
GPU. Installs only N0-TWAM model integration; ACT is never installed. The
root must end in deployment-sm120, and the legacy deployment root is rejected.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --cuda-runfile) [ "$#" -ge 2 ] || die "--cuda-runfile requires a value"; CUDA_RUNFILE="$2"; shift 2 ;;
    --cuda-sha256) [ "$#" -ge 2 ] || die "--cuda-sha256 requires a value"; CUDA_SHA256="$2"; shift 2 ;;
    --isaac-archive) [ "$#" -ge 2 ] || die "--isaac-archive requires a value"; ISAAC_ARCHIVE="$2"; shift 2 ;;
    --isaac-sha256) [ "$#" -ge 2 ] || die "--isaac-sha256 requires a value"; ISAAC_SHA256="$2"; shift 2 ;;
    --wheel) [ "$#" -ge 2 ] || die "--wheel requires a value"; ROBOTACTILE_WHEEL="$2"; shift 2 ;;
    --micromamba-archive) [ "$#" -ge 2 ] || die "--micromamba-archive requires a value"; MICROMAMBA_ARCHIVE="$2"; shift 2 ;;
    --tinygltf-archive) [ "$#" -ge 2 ] || die "--tinygltf-archive requires a value"; TINYGLTF_ARCHIVE="$2"; shift 2 ;;
    --gpu) [ "$#" -ge 2 ] || die "--gpu requires a value"; GPU_INDEX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$DEPLOY_ROOT" ] || die "--root is required"
case "$DEPLOY_ROOT" in */deployment-sm120) ;; *) die "Blackwell root must end in deployment-sm120" ;; esac
[ "$DEPLOY_ROOT" != "$ROBOTACTILE_REPOSITORY_ROOT/deployment" ] || \
  die "refusing to use the legacy deployment root"
for item in "$CUDA_RUNFILE" "$ISAAC_ARCHIVE" "$ROBOTACTILE_WHEEL"; do
  [ -f "$item" ] || die "required input file is absent: $item"
done
validate_sha256 "$CUDA_SHA256"
validate_sha256 "$ISAAC_SHA256"
case "$GPU_INDEX" in ''|*[!0-9]*) die "--gpu must be a non-negative integer" ;; esac

initialize_layout "$DEPLOY_ROOT"
DEPLOY_ROOT="$(cd "$DEPLOY_ROOT" && pwd -P)"
initialize_layout "$DEPLOY_ROOT"
acquire_lock "blackwell-sm120-bootstrap"
trap release_lock EXIT
require_command nvidia-smi

GPU_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU_INDEX" | tr -d '[:space:]')"
[ "$GPU_CAPABILITY" = "12.0" ] || \
  die "Blackwell bootstrap requires compute capability 12.0; detected ${GPU_CAPABILITY:-unknown}"
GPU_IDENTITY="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i "$GPU_INDEX")"

run_step() {
  local label="$1"
  shift
  info "Blackwell bootstrap step: $label"
  "$@"
}

run_step cuda-toolkit \
  bash "$SCRIPT_DIR/install_cuda_toolkit_12_8.sh" \
  --root "$DEPLOY_ROOT" --runfile "$CUDA_RUNFILE" --sha256 "$CUDA_SHA256"
run_step isaac-sim \
  bash "$SCRIPT_DIR/install_isaac_sim_4_5.sh" \
  --root "$DEPLOY_ROOT" --archive "$ISAAC_ARCHIVE" --sha256 "$ISAAC_SHA256"
run_step isaac-sim-smoke \
  bash "$SCRIPT_DIR/smoke_isaac_sim_4_5.sh" \
  --root "$DEPLOY_ROOT" --gpu "$GPU_INDEX" --run-id blackwell-sm120-bootstrap
run_step univtac-source \
  env ROBOTACTILE_DEPLOY_ROOT="$DEPLOY_ROOT" \
  bash "$ROBOTACTILE_REPOSITORY_ROOT/integrations/install_univtac.sh"
run_step isaaclab \
  bash "$SCRIPT_DIR/install_isaaclab_v2_1_1.sh" --root "$DEPLOY_ROOT"
run_step robotactile-isaac \
  bash "$SCRIPT_DIR/install_robotactile_isaac.sh" \
  --root "$DEPLOY_ROOT" --wheel "$ROBOTACTILE_WHEEL"
run_step tacex \
  bash "$SCRIPT_DIR/install_tacex_univtac.sh" \
  --root "$DEPLOY_ROOT" --cuda-architecture 120 --gpu "$GPU_INDEX"

UIPC_COMMAND=(
  bash "$SCRIPT_DIR/install_tacex_uipc_univtac.sh"
  --root "$DEPLOY_ROOT"
  --cuda-architecture 120
  --gpu "$GPU_INDEX"
)
if [ -n "$MICROMAMBA_ARCHIVE" ]; then
  UIPC_COMMAND+=(--micromamba-archive "$MICROMAMBA_ARCHIVE")
fi
if [ -n "$TINYGLTF_ARCHIVE" ]; then
  UIPC_COMMAND+=(--tinygltf-archive "$TINYGLTF_ARCHIVE")
fi
run_step tacex-uipc "${UIPC_COMMAND[@]}"
run_step curobo \
  bash "$SCRIPT_DIR/install_curobo_v0_7_7.sh" \
  --root "$DEPLOY_ROOT" --cuda-architecture 120 --gpu "$GPU_INDEX"

ATTESTATION="$DEPLOY_ROOT/artifacts/deployment/cuda_native_extensions_sm_120.json"
if [ -e "$ATTESTATION" ]; then
  receipt_matches "$ATTESTATION" "status=passed" "cuda_architecture=sm_120" || \
    die "existing CUDA native attestation is incompatible"
else
  run_step cuda-native-attestation \
    bash "$SCRIPT_DIR/attest_cuda_native_extensions.sh" \
    --root "$DEPLOY_ROOT" --cuda-architecture 120 --gpu "$GPU_INDEX"
fi

run_step n0-runtime \
  bash "$ROBOTACTILE_REPOSITORY_ROOT/scripts/n0_twam/install_official_runtime.sh" \
  --root "$DEPLOY_ROOT"
run_step n0-robotactile-client \
  bash "$ROBOTACTILE_REPOSITORY_ROOT/scripts/n0_twam/install_robotactile_client.sh" \
  --root "$DEPLOY_ROOT" --wheel "$ROBOTACTILE_WHEEL"
run_step n0-isaac-client \
  bash "$ROBOTACTILE_REPOSITORY_ROOT/scripts/n0_twam/install_isaac_client.sh" \
  --root "$DEPLOY_ROOT"

SOURCE_MANIFEST_SHA256="$(sha256_file "$ROBOTACTILE_REPOSITORY_ROOT/release/source_manifest.sha256")"
WHEEL_SHA256="$(sha256_file "$ROBOTACTILE_WHEEL")"
ATTESTATION_SHA256="$(sha256_file "$ATTESTATION")"
RECEIPT="$DEPLOY_ROOT/artifacts/deployment/blackwell_sm120_bootstrap.json"
if [ -e "$RECEIPT" ]; then
  receipt_matches \
    "$RECEIPT" \
    "component=blackwell_sm120_bootstrap" \
    "status=installed" \
    "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
    "wheel_sha256=$WHEEL_SHA256" \
    "cuda_native_attestation_sha256=$ATTESTATION_SHA256" || \
    die "existing Blackwell bootstrap receipt is incompatible"
else
  write_receipt \
    "$RECEIPT" \
    "component=blackwell_sm120_bootstrap" \
    "status=installed" \
    "gpu_index=$GPU_INDEX" \
    "gpu_identity=$GPU_IDENTITY" \
    "cuda_compute_capability=$GPU_CAPABILITY" \
    "cuda_toolkit_version=12.8.1" \
    "cuda_architecture=sm_120" \
    "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
    "wheel_sha256=$WHEEL_SHA256" \
    "cuda_native_attestation_sha256=$ATTESTATION_SHA256" \
    "n0_twam_enabled=true" \
    "act_enabled=false" \
    "system_python_modified=false" \
    "system_cuda_modified=false"
fi
info "Blackwell sm_120 deployment bootstrap completed"
printf '%s\n' "$RECEIPT"

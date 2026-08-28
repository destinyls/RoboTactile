#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

DEPLOY_ROOT="$(default_deployment_root)"
CUDA_ROOT=""
CUDA_ARCHITECTURE="auto"
GPU_INDEX="0"
OUTPUT=""

usage() {
  cat <<'EOF'
Usage: attest_cuda_native_extensions.sh [--root PATH]
                                         [--cuda-root PATH]
                                         [--cuda-architecture auto|NN]
                                         [--gpu INDEX]
                                         [--output PATH]

Verifies that the three torch-scatter kernel extensions, modified UIPC, and all
five cuRobo native CUDA extensions contain a cubin for the selected GPU. The
torch-scatter CUDA-version metadata stub is hashed but needs no device code.
Writes a no-clobber receipt.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) DEPLOY_ROOT="$2"; shift 2 ;;
    --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
    --cuda-architecture) CUDA_ARCHITECTURE="$2"; shift 2 ;;
    --gpu) GPU_INDEX="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

validate_cuda_build_options "$CUDA_ROOT" "$CUDA_ARCHITECTURE" "$GPU_INDEX"
initialize_layout "$DEPLOY_ROOT"
acquire_lock "cuda-native-extension-attestation"
trap release_lock EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
UNIVTAC_SOURCE_DIRECTORY="$(resolve_external_pin univtac source_directory)"
CUROBO_SOURCE_DIRECTORY="$(resolve_external_pin curobo source_directory)"
TACEX_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_install.json"
UIPC_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_uipc_install.json"
CUROBO_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/curobo_install.json"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
for receipt in "$TACEX_RECEIPT" "$UIPC_RECEIPT" "$CUROBO_RECEIPT"; do
  [ -f "$receipt" ] || die "native dependency receipt is absent: $receipt"
done
resolve_cuda_build_target \
  "$DEPLOY_ROOT" "$ISAAC_SIM_PATH" "$CUDA_ROOT" "$CUDA_ARCHITECTURE" "$GPU_INDEX"
CUDA_ROOT="$ROBOTACTILE_CUDA_ROOT"
CUDA_ARCHITECTURE="$ROBOTACTILE_CUDA_ARCHITECTURE"
CUDA_TOOLKIT_VERSION="$ROBOTACTILE_CUDA_TOOLKIT_VERSION"
CUDA_NVCC_IDENTITY="$ROBOTACTILE_CUDA_NVCC_IDENTITY"

for receipt in "$TACEX_RECEIPT" "$UIPC_RECEIPT" "$CUROBO_RECEIPT"; do
  receipt_matches \
    "$receipt" \
    "status=installed" \
    "cuda_toolkit_path=$CUDA_ROOT" \
    "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
    "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
    "cuda_architecture=sm_$CUDA_ARCHITECTURE" || \
    die "native dependency receipt does not match selected CUDA target: $receipt"
done

TORCH_PROBE="$({
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$ISAAC_SIM_PATH/python.sh" -c \
    'from pathlib import Path; import torch_scatter; print("ROBOTACTILE_TORCH_SCATTER=" + str(Path(torch_scatter.__file__).resolve().parent))'
} 2>&1)" || die "failed to locate torch-scatter package: $TORCH_PROBE"
TORCH_SCATTER_DIR="$(printf '%s\n' "$TORCH_PROBE" | sed -n 's/^ROBOTACTILE_TORCH_SCATTER=//p' | tail -n 1)"
[ -n "$TORCH_SCATTER_DIR" ] || die "torch-scatter package path probe returned no path"

if [ -z "$OUTPUT" ]; then
  OUTPUT="$DEPLOY_ROOT/artifacts/deployment/cuda_native_extensions_sm_${CUDA_ARCHITECTURE}.json"
fi
case "$OUTPUT" in /*) ;; *) die "--output must be an absolute path" ;; esac

"$ISAAC_SIM_PATH/python.sh" "$SCRIPT_DIR/attest_cuda_native_extensions.py" \
  --root "$DEPLOY_ROOT" \
  --cuobjdump "$CUDA_ROOT/bin/cuobjdump" \
  --architecture "$CUDA_ARCHITECTURE" \
  --torch-scatter-dir "$TORCH_SCATTER_DIR" \
  --uipc-dir "$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY/third_party/TacEx/source/tacex_uipc/build/Release/bin" \
  --curobo-dir "$DEPLOY_ROOT/sources/$CUROBO_SOURCE_DIRECTORY/src/curobo/curobolib" \
  --dependency-receipt "$TACEX_RECEIPT" \
  --dependency-receipt "$UIPC_RECEIPT" \
  --dependency-receipt "$CUROBO_RECEIPT" \
  --output "$OUTPUT"

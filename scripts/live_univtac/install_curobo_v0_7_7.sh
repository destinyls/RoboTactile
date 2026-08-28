#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

CUROBO_VERSION="v0.7.7"
CUROBO_COMMIT="$(resolve_external_pin curobo commit_sha)"
CUROBO_REPOSITORY="$(resolve_external_pin curobo repository_url)"
CUROBO_SOURCE_DIRECTORY="$(resolve_external_pin curobo source_directory)"
DEPLOY_ROOT="$(default_deployment_root)"
CUDA_ROOT=""
CUDA_ARCHITECTURE="auto"
GPU_INDEX="0"

usage() {
  cat <<'EOF'
Usage: install_curobo_v0_7_7.sh [--root PATH]
                                    [--cuda-root PATH]
                                    [--cuda-architecture auto|NN]
                                    [--gpu INDEX]

Clones cuRobo at the immutable v0.7.7 commit and installs it through the
verified Isaac Sim standalone Python. Native extensions are compiled for the
selected GPU with a deployment-local CUDA 12.4/12.8 toolkit. No sudo is used.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --cuda-root)
      [ "$#" -ge 2 ] || die "--cuda-root requires a value"
      CUDA_ROOT="$2"
      shift 2
      ;;
    --cuda-architecture)
      [ "$#" -ge 2 ] || die "--cuda-architecture requires a value"
      CUDA_ARCHITECTURE="$2"
      shift 2
      ;;
    --gpu)
      [ "$#" -ge 2 ] || die "--gpu requires a value"
      GPU_INDEX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

validate_cuda_build_options "$CUDA_ROOT" "$CUDA_ARCHITECTURE" "$GPU_INDEX"

initialize_layout "$DEPLOY_ROOT"
acquire_lock "curobo-v0.7.7-install"
trap release_lock EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAAC_INSTALL_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json"
ISAACLAB_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
SOURCE_PATH="$DEPLOY_ROOT/sources/$CUROBO_SOURCE_DIRECTORY"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/curobo_install.json"
TORCH_LIBRARY_PATH="$ISAAC_SIM_PATH/kit/python/lib/python3.10/site-packages/torch/lib"
export ISAAC_SIM_PATH

[ -f "$ISAAC_INSTALL_RECEIPT" ] || die "Isaac Sim install receipt is absent"
[ -f "$ISAACLAB_RECEIPT" ] || die "IsaacLab install receipt is absent"
[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
[ -f "$TORCH_LIBRARY_PATH/libc10.so" ] || \
  die "Isaac Sim PyTorch shared-library directory is incomplete"
if ! receipt_matches \
  "$ISAAC_INSTALL_RECEIPT" \
  "component=isaac_sim" \
  "version=4.5.0" \
  "status=installed"; then
  die "Isaac Sim install receipt is incompatible"
fi
if ! receipt_matches \
  "$ISAACLAB_RECEIPT" \
  "component=isaaclab" \
  "version=v2.1.1" \
  "status=installed"; then
  die "IsaacLab install receipt is incompatible"
fi
ISAAC_SOURCE_SHA256="$(receipt_field "$ISAAC_INSTALL_RECEIPT" source_sha256)"
ISAACLAB_SOURCE_COMMIT="$(receipt_field "$ISAACLAB_RECEIPT" source_commit)"

resolve_cuda_build_target \
  "$DEPLOY_ROOT" "$ISAAC_SIM_PATH" "$CUDA_ROOT" "$CUDA_ARCHITECTURE" "$GPU_INDEX"
CUDA_ROOT="$ROBOTACTILE_CUDA_ROOT"
CUDA_TOOLKIT_VERSION="$ROBOTACTILE_CUDA_TOOLKIT_VERSION"
CUDA_NVCC_IDENTITY="$ROBOTACTILE_CUDA_NVCC_IDENTITY"
CUDA_ARCHITECTURE="$ROBOTACTILE_CUDA_ARCHITECTURE"
CUDA_COMPUTE_CAPABILITY="$ROBOTACTILE_CUDA_COMPUTE_CAPABILITY"

if [ -e "$RECEIPT_PATH" ]; then
  ensure_pinned_git_source \
    "$CUROBO_REPOSITORY" \
    "$CUROBO_COMMIT" \
    "$SOURCE_PATH" \
    "setup.py"
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=curobo" \
    "version=$CUROBO_VERSION" \
    "status=installed" \
    "source_repository=$CUROBO_REPOSITORY" \
    "source_commit=$CUROBO_COMMIT" \
    "isaac_sim_source_sha256=$ISAAC_SOURCE_SHA256" \
    "isaaclab_source_commit=$ISAACLAB_SOURCE_COMMIT" \
    "cuda_toolkit_path=$CUDA_ROOT" \
    "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
    "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
    "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
    "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY"; then
    info "cuRobo v0.7.7 is already installed from the pinned commit"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing cuRobo receipt does not match the pinned installation"
fi

ensure_pinned_git_source \
  "$CUROBO_REPOSITORY" \
  "$CUROBO_COMMIT" \
  "$SOURCE_PATH" \
  "setup.py"

LOG_PATH="$(new_log_path "curobo-v0.7.7-install")"
ISAAC_PYTHON=(
  env
  -u CONDA_PREFIX
  -u CONDA_DEFAULT_ENV
  -u CONDA_PROMPT_MODIFIER
  CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  PATH="$CUDA_ROOT/bin:$ISAAC_SIM_PATH/kit/python/bin:/usr/bin:/bin"
  LD_LIBRARY_PATH="$TORCH_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
  CUDA_HOME="$CUDA_ROOT"
  CUDACXX="$CUDA_ROOT/bin/nvcc"
  CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"
  TORCH_CUDA_ARCH_LIST="$CUDA_COMPUTE_CAPABILITY"
  ROBOTACTILE_EXPECTED_CUROBO_SOURCE="$SOURCE_PATH"
  MAX_JOBS=4
  "$ISAAC_SIM_PATH/python.sh"
)
CUROBO_IMPORT_SMOKE='import importlib.metadata as m, os; from pathlib import Path; '
CUROBO_IMPORT_SMOKE+='import curobo; from curobo.curobolib import geom_cu, '
CUROBO_IMPORT_SMOKE+='kinematics_fused_cu, lbfgs_step_cu, line_search_cu, tensor_step_cu; '
CUROBO_IMPORT_SMOKE+='assert m.version("nvidia-curobo") == "0.7.7"; '
CUROBO_IMPORT_SMOKE+='assert Path(curobo.__file__).resolve().is_relative_to('
CUROBO_IMPORT_SMOKE+='Path(os.environ["ROBOTACTILE_EXPECTED_CUROBO_SOURCE"]).resolve()); '
CUROBO_IMPORT_SMOKE+='print(curobo.__file__)'
if ! run_logged \
  "$LOG_PATH" \
  "${ISAAC_PYTHON[@]}" \
  -m pip install "warp-lang==1.0.0" --no-build-isolation; then
  die "cuRobo warp dependency installation failed; see $LOG_PATH"
fi
NATIVE_EXTENSIONS_REUSED=true
if ! run_logged "$LOG_PATH" "${ISAAC_PYTHON[@]}" -c "$CUROBO_IMPORT_SMOKE"; then
  NATIVE_EXTENSIONS_REUSED=false
  if ! run_logged \
    "$LOG_PATH" \
    "${ISAAC_PYTHON[@]}" \
    -m pip install -e "$SOURCE_PATH" \
    --no-deps --no-build-isolation --force-reinstall; then
    die "cuRobo editable installation failed; see $LOG_PATH"
  fi
  if ! run_logged \
    "$LOG_PATH" \
    "${ISAAC_PYTHON[@]}" \
    -c "$CUROBO_IMPORT_SMOKE"; then
    die "cuRobo import smoke failed; see $LOG_PATH"
  fi
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=curobo" \
  "version=$CUROBO_VERSION" \
  "status=installed" \
  "source_repository=$CUROBO_REPOSITORY" \
  "source_commit=$CUROBO_COMMIT" \
  "source_path=sources/$CUROBO_SOURCE_DIRECTORY" \
  "isaac_sim_source_sha256=$ISAAC_SOURCE_SHA256" \
  "isaaclab_source_commit=$ISAACLAB_SOURCE_COMMIT" \
  "cuda_toolkit_path=$CUDA_ROOT" \
  "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
  "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
  "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
  "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY" \
  "torch_library_path=$(relative_to_deploy_root "$TORCH_LIBRARY_PATH")" \
  "native_extensions_reused=$NATIVE_EXTENSIONS_REUSED" \
  "native_extension_import_smoke=passed" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "cuRobo v0.7.7 installation and import smoke completed"
printf '%s\n' "$RECEIPT_PATH"

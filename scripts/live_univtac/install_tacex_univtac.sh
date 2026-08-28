#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

UNIVTAC_COMMIT="$(resolve_external_pin univtac commit_sha)"
UNIVTAC_REPOSITORY="$(resolve_external_pin univtac repository_url)"
UNIVTAC_SOURCE_DIRECTORY="$(resolve_external_pin univtac source_directory)"
DEPLOY_ROOT="$(default_deployment_root)"
TORCH_SCATTER_VERSION="2.1.2"
CUDA_ROOT=""
CUDA_ARCHITECTURE="auto"
GPU_INDEX="0"

usage() {
  cat <<'EOF'
Usage: install_tacex_univtac.sh [--root PATH]
                                 [--cuda-root PATH]
                                 [--cuda-architecture auto|NN]
                                 [--gpu INDEX]

Installs the TacEx core embedded in the pinned UniVTAC checkout into the
verified Isaac Sim runtime. CUDA architecture is detected from the selected
GPU and must match an explicit NN value. Modified UIPC is installed separately.
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
acquire_lock "tacex-univtac-install"
trap release_lock EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAACLAB_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
UNIVTAC_PATH="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY"
UNIVTAC_RECEIPT="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY.robotactile-install.json"
TACEX_PATH="$UNIVTAC_PATH/third_party/TacEx"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/tacex_install.json"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
[ -f "$ISAACLAB_RECEIPT" ] || die "IsaacLab install receipt is absent"
[ -f "$UNIVTAC_RECEIPT" ] || die "UniVTAC install receipt is absent"
if ! receipt_matches \
  "$ISAACLAB_RECEIPT" \
  "component=isaaclab" \
  "version=v2.1.1" \
  "status=installed" \
  "torch_identity=2.7.0+cu128|12.8"; then
  die "IsaacLab receipt is incompatible with TacEx"
fi
if ! receipt_matches \
  "$UNIVTAC_RECEIPT" \
  "integration_id=univtac" \
  "commit_sha=$UNIVTAC_COMMIT"; then
  die "UniVTAC receipt is incompatible"
fi
ensure_pinned_git_source \
  "$UNIVTAC_REPOSITORY" \
  "$UNIVTAC_COMMIT" \
  "$UNIVTAC_PATH" \
  "third_party/TacEx/source/tacex/setup.py"

resolve_cuda_build_target \
  "$DEPLOY_ROOT" "$ISAAC_SIM_PATH" "$CUDA_ROOT" "$CUDA_ARCHITECTURE" "$GPU_INDEX"
CUDA_ROOT="$ROBOTACTILE_CUDA_ROOT"
CUDA_TOOLKIT_VERSION="$ROBOTACTILE_CUDA_TOOLKIT_VERSION"
CUDA_NVCC_IDENTITY="$ROBOTACTILE_CUDA_NVCC_IDENTITY"
CUDA_ARCHITECTURE="$ROBOTACTILE_CUDA_ARCHITECTURE"
CUDA_COMPUTE_CAPABILITY="$ROBOTACTILE_CUDA_COMPUTE_CAPABILITY"

if [ -e "$RECEIPT_PATH" ]; then
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=tacex" \
    "status=installed" \
    "univtac_source_commit=$UNIVTAC_COMMIT" \
    "torch_identity=2.7.0+cu128|12.8" \
    "cuda_toolkit_path=$CUDA_ROOT" \
    "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
    "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
    "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
    "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY"; then
    info "TacEx is already installed for the pinned UniVTAC checkout"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing TacEx receipt does not match the pinned installation"
fi

LOG_PATH="$(new_log_path "tacex-univtac-install")"
ISAAC_PYTHON=(
  env
  -u CONDA_PREFIX
  -u CONDA_DEFAULT_ENV
  -u CONDA_PROMPT_MODIFIER
  PATH="$CUDA_ROOT/bin:$ISAAC_SIM_PATH/kit/python/bin:/usr/bin:/bin"
  CUDA_HOME="$CUDA_ROOT"
  CUDACXX="$CUDA_ROOT/bin/nvcc"
  CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"
  TORCH_CUDA_ARCH_LIST="$CUDA_COMPUTE_CAPABILITY"
  MAX_JOBS=4
  "$ISAAC_SIM_PATH/python.sh"
)

# A missing receipt cannot prove which architecture an importable extension
# was compiled for. Rebuild from source under the selected target instead of
# blessing a potentially copied sm_86 binary on an sm_80 host.
if ! run_logged \
  "$LOG_PATH" \
  "${ISAAC_PYTHON[@]}" \
  -m pip uninstall -y torch-scatter; then
  die "existing torch-scatter cleanup failed; see $LOG_PATH"
fi
if ! run_logged \
  "$LOG_PATH" \
  "${ISAAC_PYTHON[@]}" \
  -m pip install "torch-scatter==$TORCH_SCATTER_VERSION" \
  --no-deps \
  --no-binary torch-scatter \
  --no-build-isolation \
  --no-cache-dir; then
  die "compatible torch-scatter source build failed; see $LOG_PATH"
fi
if ! run_logged \
  "$LOG_PATH" \
  "${ISAAC_PYTHON[@]}" \
  -m pip install \
  "numpy==1.26.0" \
  "pytorch-kinematics==0.7.6" \
  "transforms3d==0.4.2" \
  "tetgen==0.6.4" \
  --no-build-isolation; then
  die "TacEx runtime dependency installation failed; see $LOG_PATH"
fi

for package in tacex tacex_assets tacex_tasks; do
  if ! run_logged \
    "$LOG_PATH" \
    "${ISAAC_PYTHON[@]}" \
    -m pip install \
    -e "$TACEX_PATH/source/$package" \
    --no-deps \
    --no-build-isolation; then
    die "TacEx package installation failed for $package; see $LOG_PATH"
  fi
done

RUNTIME_PROBE='import importlib.metadata as m; import numpy, torch, torch_scatter; print("{}|{}|{}|{}|{}|{}|{}".format(torch.__version__, torch.version.cuda, numpy.__version__, m.version("torch-scatter"), m.version("tacex"), m.version("tacex-assets"), m.version("tacex-tasks")))'
RUNTIME_IDENTITY="$("${ISAAC_PYTHON[@]}" -c "$RUNTIME_PROBE")"
[ "$RUNTIME_IDENTITY" = "2.7.0+cu128|12.8|1.26.0|2.1.2|0.1.0|0.1.0|0.1.0" ] || \
  die "TacEx runtime identity is incompatible: $RUNTIME_IDENTITY"
if ! run_logged \
  "$LOG_PATH" \
  "${ISAAC_PYTHON[@]}" \
  "$SCRIPT_DIR/smoke_tacex_isaac.py"; then
  die "TacEx Isaac headless import smoke failed; see $LOG_PATH"
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=tacex" \
  "version=univtac-embedded" \
  "status=installed" \
  "univtac_source_repository=$UNIVTAC_REPOSITORY" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "source_path=sources/$UNIVTAC_SOURCE_DIRECTORY/third_party/TacEx" \
  "torch_identity=2.7.0+cu128|12.8" \
  "numpy_identity=1.26.0" \
  "torch_scatter_identity=2.1.2+source.cuda${CUDA_TOOLKIT_VERSION}.sm$CUDA_ARCHITECTURE" \
  "cuda_toolkit_path=$CUDA_ROOT" \
  "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
  "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
  "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
  "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY" \
  "isaac_headless_import_smoke=passed" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "TacEx core installation and import smoke completed"
printf '%s\n' "$RECEIPT_PATH"

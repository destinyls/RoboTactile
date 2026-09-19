#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=../live_univtac/common.sh
source "$SCRIPT_DIR/../live_univtac/common.sh"

DEPLOY_ROOT="$(default_deployment_root)"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
LOCAL_SCRATCH_BASE="${ROBOTACTILE_LOCAL_SCRATCH:-}"
PIP_INDEX_URL="${ROBOTACTILE_FTP1_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_VERSION="25.3"
SETUPTOOLS_VERSION="80.9.0"
WHEEL_VERSION="0.45.1"
LEROBOT_COMMIT="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
PYTORCH_CU128_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_VERSION="2.7.1+cu128"
TORCHVISION_VERSION="0.22.1+cu128"
TRITON_VERSION="3.3.1"
EXPECTED_BOOTSTRAP_IDENTITY="$PIP_VERSION|$SETUPTOOLS_VERSION|$WHEEL_VERSION"
CU128_BUNDLE_PACKAGES=(
  "torch==$TORCH_VERSION"
  "torchvision==$TORCHVISION_VERSION"
  "triton==$TRITON_VERSION"
  "nvidia-cuda-nvrtc-cu12==12.8.61"
  "nvidia-cuda-runtime-cu12==12.8.57"
  "nvidia-cuda-cupti-cu12==12.8.57"
  "nvidia-cudnn-cu12==9.7.1.26"
  "nvidia-cublas-cu12==12.8.3.14"
  "nvidia-cufft-cu12==11.3.3.41"
  "nvidia-curand-cu12==10.3.9.55"
  "nvidia-cusolver-cu12==11.7.2.55"
  "nvidia-cusparse-cu12==12.5.7.53"
  "nvidia-cusparselt-cu12==0.6.3"
  "nvidia-nccl-cu12==2.26.2"
  "nvidia-nvtx-cu12==12.8.55"
  "nvidia-nvjitlink-cu12==12.8.61"
  "nvidia-cufile-cu12==1.13.0.11"
)
EXPECTED_CU128_BUNDLE_IDENTITY="torch==2.7.1+cu128|torchvision==0.22.1+cu128|triton==3.3.1|nvidia-cuda-nvrtc-cu12==12.8.61|nvidia-cuda-runtime-cu12==12.8.57|nvidia-cuda-cupti-cu12==12.8.57|nvidia-cudnn-cu12==9.7.1.26|nvidia-cublas-cu12==12.8.3.14|nvidia-cufft-cu12==11.3.3.41|nvidia-curand-cu12==10.3.9.55|nvidia-cusolver-cu12==11.7.2.55|nvidia-cusparse-cu12==12.5.7.53|nvidia-cusparselt-cu12==0.6.3|nvidia-nccl-cu12==2.26.2|nvidia-nvtx-cu12==12.8.55|nvidia-nvjitlink-cu12==12.8.61|nvidia-cufile-cu12==1.13.0.11"
RESUME_EXISTING_RUNTIME=false

usage() {
  cat <<'EOF'
Usage: install_official_runtime.sh [--root PATH] [--python PYTHON]
                                   [--resume-existing-runtime]

Creates deployment/runtime/ftp1-policy with Python 3.11 and the pinned
official FTP-1 source. It never installs into system Python, the Isaac runtime,
the N0-TWAM runtime, or system CUDA.

The isolated runtime receives the exact PyTorch 2.7.1/torchvision 0.22.1
cu128 bundle required for Blackwell sm_120, then must pass a real CUDA tensor
kernel probe before an install receipt can be written.

Set ROBOTACTILE_LOCAL_SCRATCH to an absolute node-local directory to stage
disposable pip downloads and builds away from a slow shared deployment disk.
ROBOTACTILE_FTP1_PIP_INDEX_URL may override the HTTPS package index; the
default matches the registry frozen by the reviewed upstream uv.lock.

--resume-existing-runtime resumes this script's fixed deployment/runtime/
ftp1-policy environment after an interrupted one-time install. It requires an
exact pinned bootstrap identity and does not accept an arbitrary runtime path.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --python) [ "$#" -ge 2 ] || die "--python requires a value"; SYSTEM_PYTHON="$2"; shift 2 ;;
    --resume-existing-runtime) RESUME_EXISTING_RUNTIME=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
acquire_lock "ftp1-policy-runtime-install"
case "$PIP_INDEX_URL" in
  https://*) ;;
  *) die "ROBOTACTILE_FTP1_PIP_INDEX_URL must use HTTPS" ;;
esac
export PIP_INDEX_URL
unset PIP_EXTRA_INDEX_URL

SCRATCH_ROOT=""
if [ -n "$LOCAL_SCRATCH_BASE" ]; then
  case "$LOCAL_SCRATCH_BASE" in /*) ;; *) die "ROBOTACTILE_LOCAL_SCRATCH must be absolute" ;; esac
  [ -d "$LOCAL_SCRATCH_BASE" ] && [ ! -L "$LOCAL_SCRATCH_BASE" ] || \
    die "ROBOTACTILE_LOCAL_SCRATCH must be a real directory"
  LOCAL_SCRATCH_BASE="$(cd "$LOCAL_SCRATCH_BASE" && pwd -P)"
  SCRATCH_ROOT="$(mktemp -d "$LOCAL_SCRATCH_BASE/robotactile-ftp1.XXXXXX")"
  mkdir "$SCRATCH_ROOT/pip-cache" "$SCRATCH_ROOT/tmp"
  export PIP_CACHE_DIR="$SCRATCH_ROOT/pip-cache"
  export TMPDIR="$SCRATCH_ROOT/tmp"
fi

SOURCE_ROOT="$DEPLOY_ROOT/sources/ftp1-policy"
RUNTIME_ROOT="$DEPLOY_ROOT/runtime/ftp1-policy"
RECEIPT="$DEPLOY_ROOT/artifacts/deployment/ftp1_policy_runtime_install.json"
CONSTRAINTS="$SCRIPT_DIR/requirements-runtime.txt"
TOKENIZER_PROVISIONER="$SCRIPT_DIR/provision_openpi_tokenizer.py"
OPENPI_DATA_HOME="$DEPLOY_ROOT/artifacts/openpi-data/ftp1-policy"
TOKENIZER_PATH="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
TOKENIZER_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/ftp1_policy_openpi_tokenizer_provision.json"
TOKENIZER_SHA256="8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
export OPENPI_DATA_HOME
COMMIT="$(resolve_external_pin ftp1_policy commit_sha)"
REPOSITORY="$(resolve_external_pin ftp1_policy repository_url)"
[ "$COMMIT" = "89fa681d6c014cce28300946b7526db808e0b1c1" ] || \
  die "FTP-1 lock commit is not the reviewed runtime commit"
ensure_pinned_git_source \
  "$REPOSITORY" \
  "$COMMIT" \
  "$SOURCE_ROOT" \
  "src/openpi/policies/ftp1_inference_wrapper.py"

CREATED_RUNTIME=false
discard_created_runtime() {
  case "$RUNTIME_ROOT" in
    "$DEPLOY_ROOT/runtime/ftp1-policy") rm -rf -- "$RUNTIME_ROOT" ;;
    *) die "refusing to clean unexpected FTP-1 runtime path" ;;
  esac
}
cleanup() {
  if [ "$CREATED_RUNTIME" = true ] && [ ! -f "$RECEIPT" ]; then
    discard_created_runtime
  fi
  if [ -n "$SCRATCH_ROOT" ]; then
    case "$SCRATCH_ROOT" in
      "$LOCAL_SCRATCH_BASE"/robotactile-ftp1.*) rm -rf -- "$SCRATCH_ROOT" ;;
      *) die "refusing to clean unexpected FTP-1 scratch path" ;;
    esac
  fi
  release_lock
}
trap cleanup EXIT

if ! "$SYSTEM_PYTHON" "$TOKENIZER_PROVISIONER" --root "$DEPLOY_ROOT" >/dev/null; then
  die "FTP-1 hash-bound OpenPI tokenizer provisioning failed"
fi
[ -f "$TOKENIZER_RECEIPT" ] && [ ! -L "$TOKENIZER_RECEIPT" ] || \
  die "FTP-1 tokenizer provisioner did not produce its receipt"
[ -f "$TOKENIZER_PATH" ] && [ ! -L "$TOKENIZER_PATH" ] || \
  die "FTP-1 tokenizer provisioner did not produce its deployment-local file"
TOKENIZER_RECEIPT_SHA256="$(sha256_file "$TOKENIZER_RECEIPT")"

PROBE='import importlib.metadata as m,torch; from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper; print(f"{m.version(chr(111)+chr(112)+chr(101)+chr(110)+chr(112)+chr(105))}|{torch.__version__}|{torch.version.cuda}|{m.version(chr(109)+chr(115)+chr(103)+chr(112)+chr(97)+chr(99)+chr(107))}|{m.version(chr(112)+chr(121)+chr(122)+chr(109)+chr(113))}")'
BOOTSTRAP_PROBE='import importlib.metadata as m; print(f"{m.version(chr(112)+chr(105)+chr(112))}|{m.version(chr(115)+chr(101)+chr(116)+chr(117)+chr(112)+chr(116)+chr(111)+chr(111)+chr(108)+chr(115))}|{m.version(chr(119)+chr(104)+chr(101)+chr(101)+chr(108))}")'
CU128_BUNDLE_PROBE='import importlib.metadata as m; names=("torch","torchvision","triton","nvidia-cuda-nvrtc-cu12","nvidia-cuda-runtime-cu12","nvidia-cuda-cupti-cu12","nvidia-cudnn-cu12","nvidia-cublas-cu12","nvidia-cufft-cu12","nvidia-curand-cu12","nvidia-cusolver-cu12","nvidia-cusparse-cu12","nvidia-cusparselt-cu12","nvidia-nccl-cu12","nvidia-nvtx-cu12","nvidia-nvjitlink-cu12","nvidia-cufile-cu12"); print("|".join(f"{name}=={m.version(name)}" for name in names))'
CUDA_PROBE='import torch; cuda=torch.version.cuda; arch=sorted(torch.cuda.get_arch_list()); assert cuda=="12.8", f"expected CUDA 12.8, got {cuda}"; assert "sm_120" in arch or "compute_120" in arch, f"Blackwell sm_120 is absent: {arch}"; value=torch.ones((1,),device="cuda"); torch.cuda.synchronize(); assert value.item()==1.0; print(cuda+"|"+",".join(arch)+"|torch.ones(cuda):ok")'
if [ -f "$RECEIPT" ] && receipt_matches \
  "$RECEIPT" \
  "component=ftp1_policy_runtime" \
  "source_commit=$COMMIT" \
  "python_version=3.11" \
  "bootstrap_identity=$EXPECTED_BOOTSTRAP_IDENTITY" \
  "cu128_bundle_identity=$EXPECTED_CU128_BUNDLE_IDENTITY" \
  "openpi_data_home=artifacts/openpi-data/ftp1-policy" \
  "tokenizer_sha256=$TOKENIZER_SHA256" \
  "tokenizer_receipt_sha256=$TOKENIZER_RECEIPT_SHA256" \
  "status=installed" && \
  [ -x "$RUNTIME_ROOT/bin/python" ] && \
  PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/src" \
    "$RUNTIME_ROOT/bin/python" -c "$PROBE" >/dev/null 2>&1 && \
  [ "$("$RUNTIME_ROOT/bin/python" -c "$CU128_BUNDLE_PROBE" 2>/dev/null)" = \
    "$EXPECTED_CU128_BUNDLE_IDENTITY" ] && \
  "$RUNTIME_ROOT/bin/python" -c "$CUDA_PROBE" >/dev/null 2>&1; then
  info "official FTP-1 runtime is already installed"
  printf '%s\n' "$RECEIPT"
  exit 0
fi
if [ -e "$RECEIPT" ] || [ -L "$RECEIPT" ]; then
  die "existing FTP-1 runtime/receipt is incomplete or incompatible"
fi

RESUMING_RUNTIME=false
BOOTSTRAP_IDENTITY=""
if [ "$RESUME_EXISTING_RUNTIME" = true ]; then
  [ ! -L "$RUNTIME_ROOT" ] || \
    die "--resume-existing-runtime refuses a symlink runtime root"
  [ -d "$RUNTIME_ROOT" ] || \
    die "--resume-existing-runtime requires the existing fixed FTP-1 runtime"
  [ -x "$RUNTIME_ROOT/bin/python" ] || \
    die "--resume-existing-runtime requires runtime/ftp1-policy/bin/python"
  if ! BOOTSTRAP_IDENTITY="$("$RUNTIME_ROOT/bin/python" -c "$BOOTSTRAP_PROBE")"; then
    die "--resume-existing-runtime could not probe the existing bootstrap"
  fi
  [ "$BOOTSTRAP_IDENTITY" = "$EXPECTED_BOOTSTRAP_IDENTITY" ] || \
    die "--resume-existing-runtime bootstrap identity is incompatible: $BOOTSTRAP_IDENTITY"
  RESUMING_RUNTIME=true
elif [ -e "$RUNTIME_ROOT" ] || [ -L "$RUNTIME_ROOT" ]; then
  die "existing FTP-1 runtime/receipt is incomplete or incompatible"
fi

LOG_PATH="$(new_log_path "ftp1-policy-runtime-install")"
PYTHON="$RUNTIME_ROOT/bin/python"
if [ "$RESUMING_RUNTIME" = true ]; then
  ENVIRONMENT_BACKEND="resumed_existing"
  info "resuming the interrupted official FTP-1 runtime installation"
else
  CREATED_RUNTIME=true
  ENVIRONMENT_BACKEND="venv"
  SYSTEM_PYTHON_VERSION="$($SYSTEM_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$SYSTEM_PYTHON_VERSION" = "3.11" ]; then
    if ! run_logged "$LOG_PATH" "$SYSTEM_PYTHON" -m venv "$RUNTIME_ROOT"; then
      discard_created_runtime
      ENVIRONMENT_BACKEND="micromamba"
    fi
  else
    ENVIRONMENT_BACKEND="micromamba"
  fi
  if [ "$ENVIRONMENT_BACKEND" = "micromamba" ]; then
    discard_created_runtime
    MICROMAMBA="$DEPLOY_ROOT/runtime/micromamba/bin/micromamba"
    [ -x "$MICROMAMBA" ] || \
      die "Python 3.11 is unavailable and deployment-local micromamba is absent; see $LOG_PATH"
    if ! run_logged \
      "$LOG_PATH" \
      env \
      HOME="$DEPLOY_ROOT/runtime/home" \
      MAMBA_ROOT_PREFIX="$DEPLOY_ROOT/runtime/mamba-root" \
      "$MICROMAMBA" create \
      --no-rc \
      --override-channels \
      --channel-priority strict \
      --channel conda-forge \
      --prefix "$RUNTIME_ROOT" \
      "python=3.11.11" \
      "pip=$PIP_VERSION" \
      "setuptools=$SETUPTOOLS_VERSION" \
      "wheel=$WHEEL_VERSION" \
      --yes; then
      die "FTP-1 micromamba environment creation failed; see $LOG_PATH"
    fi
  fi

  PYTHON="$RUNTIME_ROOT/bin/python"
  if [ "$ENVIRONMENT_BACKEND" = "venv" ]; then
    if ! run_logged "$LOG_PATH" "$PYTHON" -m pip install --upgrade \
      "pip==$PIP_VERSION" \
      "setuptools==$SETUPTOOLS_VERSION" \
      "wheel==$WHEEL_VERSION"; then
      die "FTP-1 bootstrap installation failed; see $LOG_PATH"
    fi
  fi
  BOOTSTRAP_IDENTITY="$("$PYTHON" -c "$BOOTSTRAP_PROBE")"
  [ "$BOOTSTRAP_IDENTITY" = "$EXPECTED_BOOTSTRAP_IDENTITY" ] || \
    die "FTP-1 bootstrap identity is incompatible: $BOOTSTRAP_IDENTITY"
fi

# PyTorch's cu128 index carries one reviewed 17-package bundle: torch,
# torchvision, triton, and the 14 CUDA runtime wheels declared by torch 2.7.1.
# Install every member explicitly with no transitive resolution before the
# upstream dependency phase so a default-index cu126 build cannot be selected.
if ! run_logged "$LOG_PATH" "$PYTHON" -m pip install \
  --no-compile \
  --no-deps \
  --index-url "$PYTORCH_CU128_INDEX_URL" \
  "${CU128_BUNDLE_PACKAGES[@]}"; then
  die "FTP-1 PyTorch cu128 bundle installation failed; see $LOG_PATH"
fi

# pip cannot interpret the workspace/Git source table in upstream pyproject.toml,
# so install those two frozen sources explicitly before resolving openpi itself.
if ! run_logged "$LOG_PATH" env GIT_LFS_SKIP_SMUDGE=1 "$PYTHON" -m pip install \
  --no-compile \
  --constraint "$CONSTRAINTS" \
  "$SOURCE_ROOT/packages/openpi-client" \
  "pytest==9.0.3" \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@$LEROBOT_COMMIT"; then
  die "FTP-1 pinned workspace dependencies failed; see $LOG_PATH"
fi
if ! run_logged "$LOG_PATH" env GIT_LFS_SKIP_SMUDGE=1 "$PYTHON" -m pip install \
  --no-compile \
  --constraint "$CONSTRAINTS" \
  "$SOURCE_ROOT"; then
  die "FTP-1 upstream runtime installation failed; see $LOG_PATH"
fi

TRANSFORMERS_ROOT="$($PYTHON -c 'import pathlib,transformers; print(pathlib.Path(transformers.__file__).resolve().parent)')"
if ! run_logged "$LOG_PATH" cp -R \
  "$SOURCE_ROOT/src/openpi/models_pytorch/transformers_replace/." \
  "$TRANSFORMERS_ROOT/"; then
  die "FTP-1 pinned transformers overlay failed; see $LOG_PATH"
fi

if ! BUNDLE_IDENTITY="$("$PYTHON" -c "$CU128_BUNDLE_PROBE" 2>>"$LOG_PATH")"; then
  die "FTP-1 could not read the installed cu128 bundle; see $LOG_PATH"
fi
[ "$BUNDLE_IDENTITY" = "$EXPECTED_CU128_BUNDLE_IDENTITY" ] || \
  die "FTP-1 installed cu128 bundle identity is incompatible: $BUNDLE_IDENTITY"
if ! CUDA_PROBE_IDENTITY="$("$PYTHON" -c "$CUDA_PROBE" 2>>"$LOG_PATH")"; then
  die "FTP-1 Blackwell CUDA kernel probe failed; see $LOG_PATH"
fi
IFS='|' read -r TORCH_CUDA_VERSION TORCH_CUDA_ARCH_LIST TORCH_CUDA_KERNEL_PROBE \
  <<< "$CUDA_PROBE_IDENTITY"
[ "$TORCH_CUDA_VERSION" = "12.8" ] || die "FTP-1 CUDA identity must be 12.8"
[ "$TORCH_CUDA_KERNEL_PROBE" = "torch.ones(cuda):ok" ] || \
  die "FTP-1 CUDA tensor probe did not complete"

RUNTIME_IDENTITY="$(PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/src" "$PYTHON" -c "$PROBE")"
LOG_SHA256="$(sha256_file "$LOG_PATH")"
CONSTRAINTS_SHA256="$(sha256_file "$CONSTRAINTS")"
write_receipt "$RECEIPT" \
  "component=ftp1_policy_runtime" \
  "source_commit=$COMMIT" \
  "status=installed" \
  "python_version=3.11" \
  "environment_backend=$ENVIRONMENT_BACKEND" \
  "bootstrap_identity=$BOOTSTRAP_IDENTITY" \
  "runtime_identity=$RUNTIME_IDENTITY" \
  "pytorch_index_url=$PYTORCH_CU128_INDEX_URL" \
  "cu128_bundle_package_count=17" \
  "cu128_bundle_identity=$BUNDLE_IDENTITY" \
  "torch_cuda_version=$TORCH_CUDA_VERSION" \
  "torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST" \
  "torch_cuda_kernel_probe=$TORCH_CUDA_KERNEL_PROBE" \
  "lerobot_commit=$LEROBOT_COMMIT" \
  "constraints_sha256=$CONSTRAINTS_SHA256" \
  "openpi_data_home=artifacts/openpi-data/ftp1-policy" \
  "tokenizer_path=artifacts/openpi-data/ftp1-policy/big_vision/paligemma_tokenizer.model" \
  "tokenizer_sha256=$TOKENIZER_SHA256" \
  "tokenizer_receipt=artifacts/deployment/ftp1_policy_openpi_tokenizer_provision.json" \
  "tokenizer_receipt_sha256=$TOKENIZER_RECEIPT_SHA256" \
  "node_local_scratch_used=$([ -n "$SCRATCH_ROOT" ] && printf true || printf false)" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256" \
  "runtime_root=runtime/ftp1-policy" \
  "system_python_modified=false" \
  "system_cuda_modified=false" \
  "isaac_runtime_modified=false" \
  "n0_runtime_modified=false"
CREATED_RUNTIME=false

info "official FTP-1 isolated runtime installation completed"
printf '%s\n' "$RECEIPT"

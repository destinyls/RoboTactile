#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

UNIVTAC_COMMIT="$(resolve_external_pin univtac commit_sha)"
UNIVTAC_REPOSITORY="$(resolve_external_pin univtac repository_url)"
UNIVTAC_SOURCE_DIRECTORY="$(resolve_external_pin univtac source_directory)"
VCPKG_COMMIT="ce613c41372b23b1f51333815feb3edd87ef8a8b"
VCPKG_MANIFEST_BASELINE="b2cb0da531c2f1f740045bfe7c4dac59f0b2b69c"
VCPKG_REPOSITORY="https://github.com/microsoft/vcpkg.git"
TINYGLTF_ARCHIVE_SHA512="6dbcff3ea602d0aa45ddd87a87d32ab5ab5453901891dbccfbc660746fe11c5bd814d6f74707244351dd6326e17f6d9ad7c384417db126122cc4a2cba20b205c"
MICROMAMBA_VERSION="2.9.0"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/2.9.0"
MICROMAMBA_SHA256="8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd"
DEPLOY_ROOT="$(default_deployment_root)"
MICROMAMBA_ARCHIVE=""
TINYGLTF_ARCHIVE=""
CUDA_ROOT=""
CUDA_ARCHITECTURE="auto"
GPU_INDEX="0"

usage() {
  cat <<'EOF'
Usage: install_tacex_uipc_univtac.sh [--root PATH]
                                     [--micromamba-archive PATH]
                                     [--tinygltf-archive PATH]
                                     [--cuda-root PATH]
                                     [--cuda-architecture auto|NN]
                                     [--gpu INDEX]

Builds the modified UIPC embedded in the pinned UniVTAC/TacEx source. The
native compiler, package cache, vcpkg source, build tree, and Python package
remain below the RoboTactile deployment root. No sudo or base-env activation
is used.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --micromamba-archive)
      [ "$#" -ge 2 ] || die "--micromamba-archive requires a value"
      MICROMAMBA_ARCHIVE="$2"
      shift 2
      ;;
    --tinygltf-archive)
      [ "$#" -ge 2 ] || die "--tinygltf-archive requires a value"
      TINYGLTF_ARCHIVE="$2"
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
acquire_lock "tacex-uipc-univtac-install"
trap release_lock EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAACLAB_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
TACEX_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_install.json"
UNIVTAC_PATH="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY"
UNIVTAC_RECEIPT="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY.robotactile-install.json"
UIPC_PATH="$UNIVTAC_PATH/third_party/TacEx/source/tacex_uipc"
VCPKG_PATH="$DEPLOY_ROOT/sources/vcpkg-2025.04.09"
TOOLCHAIN_PATH="$DEPLOY_ROOT/runtime/uipc-toolchain"
MICROMAMBA_PATH="$DEPLOY_ROOT/runtime/micromamba/bin/micromamba"
LOCK_PATH="$ROBOTACTILE_REPOSITORY_ROOT/requirements/uipc-toolchain-linux-64.lock.txt"
VCPKG_OVERLAY_ROOT="$ROBOTACTILE_REPOSITORY_ROOT/integrations/vcpkg-overlay-ports"
CPPTRACE_OVERLAY_PATH="$VCPKG_OVERLAY_ROOT/cpptrace"
TINYGLTF_OVERLAY_PATH="$VCPKG_OVERLAY_ROOT/tinygltf"
TINYGLTF_CACHE_PATH="$DEPLOY_ROOT/cache/vcpkg-downloads/syoyo-tinygltf-v2.9.3.tar.gz"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/tacex_uipc_install.json"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
[ -f "$ISAACLAB_RECEIPT" ] || die "IsaacLab install receipt is absent"
[ -f "$TACEX_RECEIPT" ] || die "TacEx install receipt is absent"
[ -f "$UNIVTAC_RECEIPT" ] || die "UniVTAC install receipt is absent"
[ -f "$LOCK_PATH" ] || die "UIPC toolchain lock is absent: $LOCK_PATH"
if ! receipt_matches \
  "$ISAACLAB_RECEIPT" \
  "component=isaaclab" \
  "version=v2.1.1" \
  "status=installed" \
  "torch_identity=2.7.0+cu128|12.8"; then
  die "IsaacLab receipt is incompatible with modified UIPC"
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
  "third_party/TacEx/source/tacex_uipc/setup.py"

resolve_cuda_build_target \
  "$DEPLOY_ROOT" "$ISAAC_SIM_PATH" "$CUDA_ROOT" "$CUDA_ARCHITECTURE" "$GPU_INDEX"
CUDA_ROOT="$ROBOTACTILE_CUDA_ROOT"
CUDA_TOOLKIT_VERSION="$ROBOTACTILE_CUDA_TOOLKIT_VERSION"
CUDA_NVCC_IDENTITY="$ROBOTACTILE_CUDA_NVCC_IDENTITY"
CUDA_ARCHITECTURE="$ROBOTACTILE_CUDA_ARCHITECTURE"
CUDA_COMPUTE_CAPABILITY="$ROBOTACTILE_CUDA_COMPUTE_CAPABILITY"
if ! receipt_matches \
  "$TACEX_RECEIPT" \
  "component=tacex" \
  "status=installed" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "cuda_toolkit_path=$CUDA_ROOT" \
  "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
  "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
  "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
  "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY"; then
  die "TacEx receipt is incompatible with the selected modified UIPC CUDA target"
fi

if [ -e "$RECEIPT_PATH" ]; then
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=tacex_uipc" \
    "version=0.1.0" \
    "status=installed" \
    "univtac_source_commit=$UNIVTAC_COMMIT" \
    "vcpkg_source_commit=$VCPKG_COMMIT" \
    "vcpkg_manifest_baseline=$VCPKG_MANIFEST_BASELINE" \
    "cpptrace_overlay_version=0.8.3" \
    "tinygltf_overlay_version=2.9.3" \
    "tinygltf_archive_sha512=$TINYGLTF_ARCHIVE_SHA512" \
    "micromamba_identity=$MICROMAMBA_VERSION|$MICROMAMBA_SHA256" \
    "cuda_toolkit_path=$CUDA_ROOT" \
    "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
    "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
    "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
    "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY"; then
    info "modified UIPC is already installed for the pinned UniVTAC checkout"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing modified UIPC receipt does not match the pinned installation"
fi

require_command curl
require_command git
require_command tar

if [ -z "$MICROMAMBA_ARCHIVE" ]; then
  MICROMAMBA_ARCHIVE="$DEPLOY_ROOT/runtime/micromamba-download/micromamba-$MICROMAMBA_VERSION.tar.bz2"
  if [ ! -e "$MICROMAMBA_ARCHIVE" ]; then
    mkdir -p "$(dirname "$MICROMAMBA_ARCHIVE")"
    archive_stage="$(mktemp "$DEPLOY_ROOT/runtime/tmp/micromamba.XXXXXX.tar.bz2")"
    if ! curl --fail --location --retry 3 --output "$archive_stage" "$MICROMAMBA_URL"; then
      rm -f "$archive_stage"
      die "failed to download pinned micromamba archive"
    fi
    if [ "$(sha256_file "$archive_stage")" != "$MICROMAMBA_SHA256" ]; then
      rm -f "$archive_stage"
      die "downloaded micromamba archive failed SHA-256 verification"
    fi
    mv "$archive_stage" "$MICROMAMBA_ARCHIVE"
  fi
fi
[ -f "$MICROMAMBA_ARCHIVE" ] || die "micromamba archive is absent"
[ "$(sha256_file "$MICROMAMBA_ARCHIVE")" = "$MICROMAMBA_SHA256" ] || \
  die "micromamba archive SHA-256 mismatch"

if [ ! -x "$MICROMAMBA_PATH" ]; then
  mkdir -p "$(dirname "$MICROMAMBA_PATH")"
  micromamba_stage="$(mktemp "$DEPLOY_ROOT/runtime/tmp/micromamba-bin.XXXXXX")"
  if ! tar -xOf "$MICROMAMBA_ARCHIVE" bin/micromamba > "$micromamba_stage"; then
    rm -f "$micromamba_stage"
    die "failed to extract micromamba binary"
  fi
  chmod 0755 "$micromamba_stage"
  mv "$micromamba_stage" "$MICROMAMBA_PATH"
fi
[ "$($MICROMAMBA_PATH --version)" = "$MICROMAMBA_VERSION" ] || \
  die "micromamba version is incompatible"

LOG_PATH="$(new_log_path "tacex-uipc-univtac-install")"
if [ ! -x "$TOOLCHAIN_PATH/bin/x86_64-conda-linux-gnu-g++" ]; then
  if ! run_logged \
    "$LOG_PATH" \
    env \
    HOME="$DEPLOY_ROOT/runtime/home" \
    MAMBA_ROOT_PREFIX="$DEPLOY_ROOT/runtime/mamba-root" \
    "$MICROMAMBA_PATH" create \
    --no-rc \
    --override-channels \
    --channel-priority strict \
    --prefix "$TOOLCHAIN_PATH" \
    --file "$LOCK_PATH" \
    --yes; then
    die "deployment-local UIPC toolchain creation failed; see $LOG_PATH"
  fi
fi
CMAKE_IDENTITY="$($TOOLCHAIN_PATH/bin/cmake --version | head -n 1)"
GCC_IDENTITY="$($TOOLCHAIN_PATH/bin/x86_64-conda-linux-gnu-gcc --version | head -n 1)"
GXX_IDENTITY="$($TOOLCHAIN_PATH/bin/x86_64-conda-linux-gnu-g++ --version | head -n 1)"
[ "$CMAKE_IDENTITY" = "cmake version 3.26.4" ] || die "CMake identity drifted"
case "$GCC_IDENTITY" in *"11.4.0") ;; *) die "GCC identity drifted" ;; esac
case "$GXX_IDENTITY" in *"11.4.0") ;; *) die "G++ identity drifted" ;; esac

if [ ! -e "$VCPKG_PATH" ]; then
  ensure_pinned_git_source \
    "$VCPKG_REPOSITORY" \
    "$VCPKG_COMMIT" \
    "$VCPKG_PATH" \
    "bootstrap-vcpkg.sh"
else
  [ -d "$VCPKG_PATH/.git" ] || die "existing vcpkg source is not a Git checkout"
  [ "$(git -C "$VCPKG_PATH" rev-parse HEAD)" = "$VCPKG_COMMIT" ] || \
    die "existing vcpkg source is at the wrong commit"
  git -C "$VCPKG_PATH" diff --quiet || die "vcpkg source has tracked changes"
  git -C "$VCPKG_PATH" diff --cached --quiet || die "vcpkg source has staged changes"
fi
if [ ! -x "$VCPKG_PATH/vcpkg" ]; then
  if ! run_logged \
    "$LOG_PATH" \
    env \
    HOME="$DEPLOY_ROOT/runtime/home" \
    VCPKG_DISABLE_METRICS=1 \
    "$VCPKG_PATH/bootstrap-vcpkg.sh" -disableMetrics; then
    die "pinned vcpkg bootstrap failed; see $LOG_PATH"
  fi
fi
[ -f "$CPPTRACE_OVERLAY_PATH/portfile.cmake" ] || \
  die "RoboTactile cpptrace overlay port is absent"
grep -q '"version": "0.8.3"' "$CPPTRACE_OVERLAY_PATH/vcpkg.json" || \
  die "RoboTactile cpptrace overlay version drifted"
[ -f "$TINYGLTF_OVERLAY_PATH/portfile.cmake" ] || \
  die "RoboTactile tinygltf overlay port is absent"
grep -q '"version": "2.9.3"' "$TINYGLTF_OVERLAY_PATH/vcpkg.json" || \
  die "RoboTactile tinygltf overlay version drifted"
grep -q "$TINYGLTF_ARCHIVE_SHA512" "$TINYGLTF_OVERLAY_PATH/portfile.cmake" || \
  die "RoboTactile tinygltf archive digest drifted"

mkdir -p "$DEPLOY_ROOT/cache/vcpkg-downloads"
if [ -n "$TINYGLTF_ARCHIVE" ]; then
  [ -f "$TINYGLTF_ARCHIVE" ] || die "tinygltf archive is absent"
  [ "$(sha512_file "$TINYGLTF_ARCHIVE")" = "$TINYGLTF_ARCHIVE_SHA512" ] || \
    die "tinygltf archive SHA-512 mismatch"
  if [ ! -e "$TINYGLTF_CACHE_PATH" ]; then
    tinygltf_stage="$(mktemp "$DEPLOY_ROOT/runtime/tmp/tinygltf.XXXXXX.tar.gz")"
    cp "$TINYGLTF_ARCHIVE" "$tinygltf_stage"
    mv "$tinygltf_stage" "$TINYGLTF_CACHE_PATH"
  fi
fi
if [ -e "$TINYGLTF_CACHE_PATH" ]; then
  [ -f "$TINYGLTF_CACHE_PATH" ] || die "tinygltf cache target is not a file"
  [ "$(sha512_file "$TINYGLTF_CACHE_PATH")" = "$TINYGLTF_ARCHIVE_SHA512" ] || \
    die "cached tinygltf archive SHA-512 mismatch"
fi

mkdir -p \
  "$DEPLOY_ROOT/cache/vcpkg-binary"
BUILD_ENV=(
  env
  -u CONDA_PREFIX
  -u CONDA_DEFAULT_ENV
  -u CONDA_PROMPT_MODIFIER
  HOME="$DEPLOY_ROOT/runtime/home"
  TMPDIR="$DEPLOY_ROOT/runtime/tmp"
  PATH="$TOOLCHAIN_PATH/bin:$ISAAC_SIM_PATH/kit/python/bin:$CUDA_ROOT/bin:/usr/bin:/bin"
  LD_LIBRARY_PATH="$TOOLCHAIN_PATH/lib:${LD_LIBRARY_PATH:-}"
  CC="$TOOLCHAIN_PATH/bin/x86_64-conda-linux-gnu-gcc"
  CXX="$TOOLCHAIN_PATH/bin/x86_64-conda-linux-gnu-g++"
  CUDAHOSTCXX="$TOOLCHAIN_PATH/bin/x86_64-conda-linux-gnu-g++"
  CUDACXX="$CUDA_ROOT/bin/nvcc"
  CUDA_HOME="$CUDA_ROOT"
  CMAKE_GENERATOR=Ninja
  CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"
  CMAKE_TOOLCHAIN_FILE="$VCPKG_PATH/scripts/buildsystems/vcpkg.cmake"
  VCPKG_ROOT="$VCPKG_PATH"
  VCPKG_FORCE_SYSTEM_BINARIES=1
  VCPKG_OVERLAY_PORTS="$VCPKG_OVERLAY_ROOT"
  VCPKG_DOWNLOADS="$DEPLOY_ROOT/cache/vcpkg-downloads"
  VCPKG_BINARY_SOURCES="clear;files,$DEPLOY_ROOT/cache/vcpkg-binary,readwrite"
  VCPKG_DISABLE_METRICS=1
)
if ! run_logged \
  "$LOG_PATH" \
  "${BUILD_ENV[@]}" \
  "$ISAAC_SIM_PATH/python.sh" \
  -m pip install \
  "pybind11==2.13.6" \
  "mypy==1.11.2" \
  "wildmeshing==0.4.1" \
  --no-build-isolation; then
  die "modified UIPC Python dependency installation failed; see $LOG_PATH"
fi
# A prior configure can retain another toolkit or GPU architecture in
# CMakeCache. A missing matching receipt never authorizes reusing that native
# binary, even when it remains importable on the current host.
CMAKE_CACHE_PATH="$UIPC_PATH/build/CMakeCache.txt"
if [ -f "$CMAKE_CACHE_PATH" ] && {
  ! grep -q "CMAKE_TOOLCHAIN_FILE:FILEPATH=$VCPKG_PATH/scripts/buildsystems/vcpkg.cmake" \
    "$CMAKE_CACHE_PATH" ||
  ! grep -Fq "CMAKE_CUDA_COMPILER:FILEPATH=$CUDA_ROOT/bin/nvcc" \
    "$CMAKE_CACHE_PATH" ||
  ! grep -Eq "CMAKE_CUDA_ARCHITECTURES:[^=]+=$CUDA_ARCHITECTURE$" \
    "$CMAKE_CACHE_PATH"
}; then
  rm -f "$CMAKE_CACHE_PATH"
  rm -rf "$UIPC_PATH/build/CMakeFiles"
fi
# The embedded generator returns success when vcpkg.json already exists,
# which disables manifest installation even after an interrupted build.
# Removing only this generated file makes retries restore missing ports.
rm -f "$UIPC_PATH/build/vcpkg.json"
if ! run_logged \
  "$LOG_PATH" \
  "${BUILD_ENV[@]}" \
  "$ISAAC_SIM_PATH/python.sh" \
  -m pip install -e "$UIPC_PATH" \
  --no-deps --no-build-isolation --verbose --force-reinstall; then
  die "modified UIPC native build failed; see $LOG_PATH"
fi
if ! run_logged \
  "$LOG_PATH" \
  "${BUILD_ENV[@]}" \
  "$ISAAC_SIM_PATH/python.sh" \
  "$SCRIPT_DIR/smoke_tacex_uipc_isaac.py"; then
  die "modified UIPC Isaac headless import smoke failed; see $LOG_PATH"
fi

RUNTIME_IDENTITY="$(
  "${BUILD_ENV[@]}" "$ISAAC_SIM_PATH/python.sh" -c \
    'import importlib.metadata as m; import numpy, torch; print("{}|{}|{}|{}".format(m.version("tacex-uipc"), torch.__version__, torch.version.cuda, numpy.__version__))'
)"
[ "$RUNTIME_IDENTITY" = "0.1.0|2.7.0+cu128|12.8|1.26.0" ] || \
  die "modified UIPC runtime identity is incompatible: $RUNTIME_IDENTITY"
LOG_SHA256="$(sha256_file "$LOG_PATH")"
LOCK_SHA256="$(sha256_file "$LOCK_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=tacex_uipc" \
  "version=0.1.0" \
  "status=installed" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "source_path=sources/$UNIVTAC_SOURCE_DIRECTORY/third_party/TacEx/source/tacex_uipc" \
  "vcpkg_source_repository=$VCPKG_REPOSITORY" \
  "vcpkg_source_commit=$VCPKG_COMMIT" \
  "vcpkg_manifest_baseline=$VCPKG_MANIFEST_BASELINE" \
  "cpptrace_overlay_version=0.8.3" \
  "tinygltf_overlay_version=2.9.3" \
  "tinygltf_archive_sha512=$TINYGLTF_ARCHIVE_SHA512" \
  "vcpkg_force_system_binaries=true" \
  "micromamba_identity=$MICROMAMBA_VERSION|$MICROMAMBA_SHA256" \
  "toolchain_lock_sha256=$LOCK_SHA256" \
  "toolchain_identity=$CMAKE_IDENTITY|gcc-11.4.0|gxx-11.4.0" \
  "runtime_identity=$RUNTIME_IDENTITY" \
  "cuda_toolkit_path=$CUDA_ROOT" \
  "cuda_toolkit_version=$CUDA_TOOLKIT_VERSION" \
  "cuda_nvcc_identity=$CUDA_NVCC_IDENTITY" \
  "cuda_architecture=sm_$CUDA_ARCHITECTURE" \
  "cuda_compute_capability=$CUDA_COMPUTE_CAPABILITY" \
  "isaac_headless_import_smoke=passed" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "modified UIPC installation and Isaac headless import smoke completed"
printf '%s\n' "$RECEIPT_PATH"

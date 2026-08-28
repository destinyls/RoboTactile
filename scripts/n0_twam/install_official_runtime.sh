#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../live_univtac/common.sh
source "$SCRIPT_DIR/../live_univtac/common.sh"

DEPLOY_ROOT="$(default_deployment_root)"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
PIP_VERSION="25.1.1"
SETUPTOOLS_VERSION="80.9.0"
WHEEL_VERSION="0.45.1"
EXPECTED_BOOTSTRAP_IDENTITY="$PIP_VERSION|$SETUPTOOLS_VERSION|$WHEEL_VERSION"
MAMBA_EXTRACT_THREADS="${ROBOTACTILE_MAMBA_EXTRACT_THREADS:-4}"

usage() {
  cat <<'EOF'
Usage: install_official_runtime.sh [--root PATH] [--python PYTHON]
                                   [--mamba-extract-threads N]

Installs the pinned official N0-TWAM source and dependencies into
deployment/runtime/n0-twam. It does not modify system Python or system CUDA.
The micromamba fallback limits package extraction/compile workers to avoid a
metadata storm on shared filesystems (default: 4, maximum: 32).
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --python) [ "$#" -ge 2 ] || die "--python requires a value"; SYSTEM_PYTHON="$2"; shift 2 ;;
    --mamba-extract-threads) [ "$#" -ge 2 ] || die "--mamba-extract-threads requires a value"; MAMBA_EXTRACT_THREADS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$MAMBA_EXTRACT_THREADS" in
  ''|*[!0-9]*) die "--mamba-extract-threads must be an integer" ;;
esac
[ "$MAMBA_EXTRACT_THREADS" -ge 1 ] && [ "$MAMBA_EXTRACT_THREADS" -le 32 ] || \
  die "--mamba-extract-threads must be between 1 and 32"

initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
acquire_lock "n0-twam-runtime-install"
CREATED_RUNTIME=false
cleanup() {
  if [ "$CREATED_RUNTIME" = true ] && [ ! -f "$RECEIPT" ]; then
    discard_created_runtime
  fi
  release_lock
}
discard_created_runtime() {
  case "$RUNTIME_ROOT" in
    "$DEPLOY_ROOT/runtime/n0-twam") rm -rf -- "$RUNTIME_ROOT" ;;
    *) die "refusing to clean unexpected N0 runtime path" ;;
  esac
}
trap cleanup EXIT

SOURCE_ROOT="$DEPLOY_ROOT/sources/N0-TWAM"
RUNTIME_ROOT="$DEPLOY_ROOT/runtime/n0-twam"
RECEIPT="$DEPLOY_ROOT/artifacts/deployment/n0_twam_runtime_install.json"
COMMIT="$(resolve_external_pin n0_twam commit_sha)"
REPOSITORY="$(resolve_external_pin n0_twam repository_url)"
ensure_pinned_git_source "$REPOSITORY" "$COMMIT" "$SOURCE_ROOT" "n0_twam/n0_twam_server.py"

PROBE='import importlib.metadata as m,torch; print(f"{m.version(chr(110)+chr(48)+chr(45)+chr(116)+chr(119)+chr(97)+chr(109))}|{torch.__version__}|{torch.version.cuda}")'
BOOTSTRAP_PROBE='import importlib.metadata as m; print(f"{m.version(chr(112)+chr(105)+chr(112))}|{m.version(chr(115)+chr(101)+chr(116)+chr(117)+chr(112)+chr(116)+chr(111)+chr(111)+chr(108)+chr(115))}|{m.version(chr(119)+chr(104)+chr(101)+chr(101)+chr(108))}")'
if [ -f "$RECEIPT" ] && receipt_matches \
  "$RECEIPT" \
  "component=n0_twam_runtime" \
  "source_commit=$COMMIT" \
  "bootstrap_identity=$EXPECTED_BOOTSTRAP_IDENTITY" \
  "status=installed" && \
  [ -x "$RUNTIME_ROOT/bin/python" ] && \
  "$RUNTIME_ROOT/bin/python" -c "$PROBE" >/dev/null 2>&1; then
  info "official N0-TWAM runtime is already installed"
  printf '%s\n' "$RECEIPT"
  exit 0
fi
if [ -e "$RUNTIME_ROOT" ] || [ -e "$RECEIPT" ]; then
  die "existing N0-TWAM runtime/receipt is incomplete or incompatible"
fi

LOG_PATH="$(new_log_path "n0-twam-runtime-install")"
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
    die "Python 3.11 venv is unavailable and deployment-local micromamba is absent; see $LOG_PATH"
  if ! run_logged \
    "$LOG_PATH" \
    env \
    HOME="$DEPLOY_ROOT/runtime/home" \
    MAMBA_ROOT_PREFIX="$DEPLOY_ROOT/runtime/mamba-root" \
    MAMBA_EXTRACT_THREADS="$MAMBA_EXTRACT_THREADS" \
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
    die "N0-TWAM micromamba environment creation failed; see $LOG_PATH"
  fi
fi
PYTHON="$RUNTIME_ROOT/bin/python"
if [ "$ENVIRONMENT_BACKEND" = "venv" ]; then
  if ! run_logged "$LOG_PATH" "$PYTHON" -m pip install --upgrade \
    "pip==$PIP_VERSION" \
    "setuptools==$SETUPTOOLS_VERSION" \
    "wheel==$WHEEL_VERSION"; then
    die "N0-TWAM bootstrap dependency installation failed; see $LOG_PATH"
  fi
fi
BOOTSTRAP_IDENTITY="$("$PYTHON" -c "$BOOTSTRAP_PROBE")"
[ "$BOOTSTRAP_IDENTITY" = "$EXPECTED_BOOTSTRAP_IDENTITY" ] || \
  die "N0-TWAM bootstrap identity is incompatible: $BOOTSTRAP_IDENTITY"
if ! run_logged "$LOG_PATH" "$PYTHON" -m pip install "$SOURCE_ROOT" \
  "huggingface-hub==0.36.0"; then
  die "N0-TWAM runtime installation failed; see $LOG_PATH"
fi
if ! run_logged "$LOG_PATH" "$PYTHON" -m pip install --no-deps \
  "$ROBOTACTILE_REPOSITORY_ROOT"; then
  die "RoboTactile client installation failed; see $LOG_PATH"
fi
RUNTIME_IDENTITY="$("$PYTHON" -c "$PROBE")"
LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt "$RECEIPT" \
  "component=n0_twam_runtime" \
  "source_commit=$COMMIT" \
  "status=installed" \
  "environment_backend=$ENVIRONMENT_BACKEND" \
  "system_python_version=$SYSTEM_PYTHON_VERSION" \
  "mamba_extract_threads=$MAMBA_EXTRACT_THREADS" \
  "bootstrap_identity=$BOOTSTRAP_IDENTITY" \
  "runtime_identity=$RUNTIME_IDENTITY" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256" \
  "system_python_modified=false" \
  "system_cuda_modified=false"
CREATED_RUNTIME=false

info "official N0-TWAM isolated runtime installation completed"
printf '%s\n' "$RECEIPT"

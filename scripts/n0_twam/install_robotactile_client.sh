#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../live_univtac/common.sh
source "$SCRIPT_DIR/../live_univtac/common.sh"

BENCHMARK_VERSION="0.6.0"
DEPLOY_ROOT="$(default_deployment_root)"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
WHEEL_INPUT=""

usage() {
  cat <<'EOF'
Usage: install_robotactile_client.sh [--root PATH] [--wheel PATH]

Installs the current source-manifest-bound RoboTactile client into the isolated
official N0-TWAM runtime. It does not modify system Python or system CUDA.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --wheel) [ "$#" -ge 2 ] || die "--wheel requires a value"; WHEEL_INPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
acquire_lock "n0-twam-robotactile-client-install"
BUILD_DIRECTORY=""
cleanup() {
  safe_remove_temporary_directory "$BUILD_DIRECTORY"
  release_lock
}
trap cleanup EXIT

RUNTIME_ROOT="$DEPLOY_ROOT/runtime/n0-twam"
RUNTIME_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/n0_twam_runtime_install.json"
SOURCE_MANIFEST_PATH="$ROBOTACTILE_REPOSITORY_ROOT/release/source_manifest.sha256"
PYTHON="$RUNTIME_ROOT/bin/python"
[ -x "$PYTHON" ] || die "official N0-TWAM runtime is absent"
[ -f "$RUNTIME_RECEIPT" ] || die "official N0-TWAM runtime receipt is absent"
[ -f "$SOURCE_MANIFEST_PATH" ] || die "RoboTactile source manifest is absent"
if ! receipt_matches \
  "$RUNTIME_RECEIPT" \
  "component=n0_twam_runtime" \
  "status=installed"; then
  die "official N0-TWAM runtime receipt is incompatible"
fi
if ! "$SYSTEM_PYTHON" \
  "$ROBOTACTILE_REPOSITORY_ROOT/scripts/release/update_source_manifest.py" \
  --project-root "$ROBOTACTILE_REPOSITORY_ROOT" \
  --check >/dev/null; then
  die "RoboTactile source manifest is stale"
fi
if [ -z "$WHEEL_INPUT" ] && \
  ! "$SYSTEM_PYTHON" -c 'import hatchling' >/dev/null 2>&1; then
  die "the selected build Python does not provide pinned hatchling"
fi
if [ -n "$WHEEL_INPUT" ]; then
  [ -f "$WHEEL_INPUT" ] || die "provided RoboTactile wheel is absent"
  case "$(basename "$WHEEL_INPUT")" in
    "robotactile_benchmark-$BENCHMARK_VERSION-py3-none-any.whl") ;;
    *) die "provided RoboTactile wheel filename is incompatible" ;;
  esac
fi

SOURCE_MANIFEST_SHA256="$(sha256_file "$SOURCE_MANIFEST_PATH")"
RECEIPT="$DEPLOY_ROOT/artifacts/deployment/n0_twam_robotactile_client_install-${SOURCE_MANIFEST_SHA256:0:16}.json"
RUNTIME_PROBE='import hashlib,importlib.metadata as m; from robotactile_benchmark.resources import load_source_manifest; print("{}|{}".format(m.version("robotactile-benchmark"), hashlib.sha256(load_source_manifest().encode("utf-8")).hexdigest()))'
EXPECTED_IDENTITY="$BENCHMARK_VERSION|$SOURCE_MANIFEST_SHA256"
if [ -f "$RECEIPT" ] && receipt_matches \
  "$RECEIPT" \
  "component=n0_twam_robotactile_client" \
  "version=$BENCHMARK_VERSION" \
  "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
  "status=installed" && \
  [ "$("$PYTHON" -c "$RUNTIME_PROBE")" = "$EXPECTED_IDENTITY" ]; then
  info "current RoboTactile client is already installed in the N0 runtime"
  printf '%s\n' "$RECEIPT"
  exit 0
fi
[ ! -e "$RECEIPT" ] || die "existing N0 client receipt is incompatible"

LOG_PATH="$(new_log_path "n0-twam-robotactile-client-install")"
BUILD_DIRECTORY="$(mktemp -d "$DEPLOY_ROOT/runtime/tmp/n0-client-wheel.XXXXXX")"
if [ -z "$WHEEL_INPUT" ]; then
  if ! run_logged \
    "$LOG_PATH" \
    "$SYSTEM_PYTHON" -m hatchling build \
    --target wheel \
    --directory "$BUILD_DIRECTORY"; then
    die "RoboTactile wheel build failed; see $LOG_PATH"
  fi
  WHEEL_PATH="$BUILD_DIRECTORY/robotactile_benchmark-$BENCHMARK_VERSION-py3-none-any.whl"
else
  WHEEL_PATH="$WHEEL_INPUT"
fi
[ -f "$WHEEL_PATH" ] || die "expected RoboTactile wheel is absent"
WHEEL_SHA256="$(sha256_file "$WHEEL_PATH")"
if ! run_logged \
  "$LOG_PATH" \
  "$PYTHON" -m pip install \
  --no-deps \
  --force-reinstall \
  "$WHEEL_PATH"; then
  die "RoboTactile N0 client installation failed; see $LOG_PATH"
fi
RUNTIME_IDENTITY="$("$PYTHON" -c "$RUNTIME_PROBE")"
[ "$RUNTIME_IDENTITY" = "$EXPECTED_IDENTITY" ] || \
  die "RoboTactile N0 client identity is incompatible: $RUNTIME_IDENTITY"

write_receipt "$RECEIPT" \
  "component=n0_twam_robotactile_client" \
  "version=$BENCHMARK_VERSION" \
  "status=installed" \
  "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
  "wheel_sha256=$WHEEL_SHA256" \
  "runtime_identity=$RUNTIME_IDENTITY" \
  "runtime_receipt_sha256=$(sha256_file "$RUNTIME_RECEIPT")" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$(sha256_file "$LOG_PATH")" \
  "system_python_modified=false" \
  "system_cuda_modified=false"

info "source-bound RoboTactile client installation completed"
printf '%s\n' "$RECEIPT"

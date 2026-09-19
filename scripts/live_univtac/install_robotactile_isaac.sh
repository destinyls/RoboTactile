#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

BENCHMARK_VERSION="0.6.0"
DEPLOY_ROOT="$(default_deployment_root)"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
WHEEL_INPUT=""
WHEEL_MANIFEST_MEMBER="robotactile_benchmark/source_manifest.sha256"

usage() {
  cat <<'EOF'
Usage: install_robotactile_isaac.sh [--root PATH] [--wheel PATH]

Without --wheel, builds the current source-manifest-bound RoboTactile wheel.
With --wheel, derives the source identity from the packaged manifest instead.
The selected wheel is installed into standalone Isaac Sim Python; the invoking
Python environment is read only.
EOF
}

wheel_source_manifest_sha256() {
  local wheel_path="$1"
  "$SYSTEM_PYTHON" - "$wheel_path" "$WHEEL_MANIFEST_MEMBER" <<'PY'
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

wheel_path = Path(sys.argv[1])
member_name = sys.argv[2]
with zipfile.ZipFile(wheel_path) as archive:
    members = tuple(
        info for info in archive.infolist() if info.filename == member_name
    )
    if len(members) != 1:
        raise SystemExit(
            f"wheel must contain exactly one {member_name}; found {len(members)}"
        )
    payload = archive.read(members[0])
if not payload:
    raise SystemExit("packaged source manifest is empty")
payload.decode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --wheel)
      [ "$#" -ge 2 ] || die "--wheel requires a value"
      WHEEL_INPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
acquire_lock "robotactile-isaac-install"
BUILD_DIRECTORY=""
cleanup() {
  safe_remove_temporary_directory "$BUILD_DIRECTORY"
  release_lock
}
trap cleanup EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAACLAB_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
SOURCE_MANIFEST_PATH="$ROBOTACTILE_REPOSITORY_ROOT/release/source_manifest.sha256"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
[ -f "$ISAACLAB_RECEIPT" ] || die "IsaacLab install receipt is absent"
if ! receipt_matches \
  "$ISAACLAB_RECEIPT" \
  "component=isaaclab" \
  "version=v2.1.1" \
  "status=installed"; then
  die "IsaacLab receipt is incompatible"
fi
if [ -n "$WHEEL_INPUT" ]; then
  [ -f "$WHEEL_INPUT" ] || die "provided RoboTactile wheel is absent"
  case "$(basename "$WHEEL_INPUT")" in
    "robotactile_benchmark-$BENCHMARK_VERSION-py3-none-any.whl") ;;
    *) die "provided RoboTactile wheel filename is incompatible" ;;
  esac
  if ! SOURCE_MANIFEST_SHA256="$(
    wheel_source_manifest_sha256 "$WHEEL_INPUT"
  )"; then
    die "provided RoboTactile wheel has no valid packaged source manifest"
  fi
else
  [ -f "$SOURCE_MANIFEST_PATH" ] || die "RoboTactile source manifest is absent"
  if ! "$SYSTEM_PYTHON" \
    "$ROBOTACTILE_REPOSITORY_ROOT/scripts/release/update_source_manifest.py" \
    --project-root "$ROBOTACTILE_REPOSITORY_ROOT" \
    --check >/dev/null; then
    die "RoboTactile source manifest is stale"
  fi
  if ! "$SYSTEM_PYTHON" -c 'import hatchling' >/dev/null 2>&1; then
    die "the selected build Python does not provide pinned hatchling"
  fi
  SOURCE_MANIFEST_SHA256="$(sha256_file "$SOURCE_MANIFEST_PATH")"
fi

RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/robotactile_isaac_install-${SOURCE_MANIFEST_SHA256:0:16}.json"
ISAAC_PYTHON=(
  env
  -u CONDA_PREFIX
  -u CONDA_DEFAULT_ENV
  -u CONDA_PROMPT_MODIFIER
  "$ISAAC_SIM_PATH/python.sh"
)
RUNTIME_PROBE='import hashlib,importlib.metadata as m,numpy; from robotactile_benchmark.resources import load_source_manifest; print("{}|{}|{}".format(m.version("robotactile-benchmark"), hashlib.sha256(load_source_manifest().encode("utf-8")).hexdigest(), numpy.__version__))'

if [ -e "$RECEIPT_PATH" ]; then
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=robotactile_isaac" \
    "version=$BENCHMARK_VERSION" \
    "status=installed" \
    "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" && \
    [ "$("${ISAAC_PYTHON[@]}" -c "$RUNTIME_PROBE")" = \
      "$BENCHMARK_VERSION|$SOURCE_MANIFEST_SHA256|1.26.0" ]; then
    info "RoboTactile is already installed into the pinned Isaac runtime"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing RoboTactile Isaac receipt does not match requested source identity"
fi

LOG_PATH="$(new_log_path "robotactile-isaac-install")"
if [ -z "$WHEEL_INPUT" ]; then
  BUILD_DIRECTORY="$(mktemp -d "$DEPLOY_ROOT/runtime/tmp/robotactile-wheel.XXXXXX")"
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
  "${ISAAC_PYTHON[@]}" \
  -m pip install \
  --no-deps \
  --force-reinstall \
  "$WHEEL_PATH"; then
  die "RoboTactile Isaac wheel installation failed; see $LOG_PATH"
fi
RUNTIME_IDENTITY="$("${ISAAC_PYTHON[@]}" -c "$RUNTIME_PROBE")"
[ "$RUNTIME_IDENTITY" = \
  "$BENCHMARK_VERSION|$SOURCE_MANIFEST_SHA256|1.26.0" ] || \
  die "RoboTactile Isaac runtime identity is incompatible: $RUNTIME_IDENTITY"

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=robotactile_isaac" \
  "version=$BENCHMARK_VERSION" \
  "status=installed" \
  "source_manifest_sha256=$SOURCE_MANIFEST_SHA256" \
  "wheel_sha256=$WHEEL_SHA256" \
  "runtime_identity=$RUNTIME_IDENTITY" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "RoboTactile wheel installation into Isaac completed"
printf '%s\n' "$RECEIPT_PATH"

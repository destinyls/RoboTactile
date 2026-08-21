#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

CUROBO_VERSION="v0.7.7"
CUROBO_COMMIT="0a50de1ba72db304195d59d9d0b1ed269696047f"
CUROBO_REPOSITORY="https://github.com/NVlabs/curobo.git"
DEPLOY_ROOT="/data1/yanglei/robotactile_univtac_20260821"

usage() {
  cat <<'EOF'
Usage: install_curobo_v0_7_7.sh [--root PATH]

Clones cuRobo at the immutable v0.7.7 commit and installs it through the
verified Isaac Sim standalone Python. No sudo is used.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
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
acquire_lock "curobo-v0.7.7-install"
trap release_lock EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAAC_INSTALL_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json"
ISAACLAB_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
SOURCE_PATH="$DEPLOY_ROOT/src/curobo"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/curobo_install.json"
export ISAAC_SIM_PATH

[ -f "$ISAAC_INSTALL_RECEIPT" ] || die "Isaac Sim install receipt is absent"
[ -f "$ISAACLAB_RECEIPT" ] || die "IsaacLab install receipt is absent"
[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
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
    "isaaclab_source_commit=$ISAACLAB_SOURCE_COMMIT"; then
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
if ! run_logged \
  "$LOG_PATH" \
  "$ISAAC_SIM_PATH/python.sh" \
  -m pip install "warp-lang==1.0.0" --no-build-isolation; then
  die "cuRobo warp dependency installation failed; see $LOG_PATH"
fi
if ! run_logged \
  "$LOG_PATH" \
  "$ISAAC_SIM_PATH/python.sh" \
  -m pip install -e "$SOURCE_PATH" --no-build-isolation; then
  die "cuRobo editable installation failed; see $LOG_PATH"
fi
if ! run_logged \
  "$LOG_PATH" \
  "$ISAAC_SIM_PATH/python.sh" \
  -c "import curobo; print(curobo.__file__)"; then
  die "cuRobo import smoke failed; see $LOG_PATH"
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=curobo" \
  "version=$CUROBO_VERSION" \
  "status=installed" \
  "source_repository=$CUROBO_REPOSITORY" \
  "source_commit=$CUROBO_COMMIT" \
  "source_path=src/curobo" \
  "isaac_sim_source_sha256=$ISAAC_SOURCE_SHA256" \
  "isaaclab_source_commit=$ISAACLAB_SOURCE_COMMIT" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "cuRobo v0.7.7 installation and import smoke completed"
printf '%s\n' "$RECEIPT_PATH"

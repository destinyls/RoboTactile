#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ISAACLAB_VERSION="v2.1.1"
ISAACLAB_COMMIT="$(resolve_external_pin isaaclab commit_sha)"
ISAACLAB_REPOSITORY="$(resolve_external_pin isaaclab repository_url)"
ISAACLAB_SOURCE_DIRECTORY="$(resolve_external_pin isaaclab source_directory)"
DEPLOY_ROOT="$(default_deployment_root)"

usage() {
  cat <<'EOF'
Usage: install_isaaclab_v2_1_1.sh [--root PATH]

Clones IsaacLab at the immutable commit behind v2.1.1 and installs it against
the verified Isaac Sim 4.5.0 standalone runtime. No sudo is used.
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
acquire_lock "isaaclab-v2.1.1-install"
trap release_lock EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAAC_INSTALL_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json"
SOURCE_PATH="$DEPLOY_ROOT/sources/$ISAACLAB_SOURCE_DIRECTORY"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
export ISAAC_SIM_PATH

[ -f "$ISAAC_INSTALL_RECEIPT" ] || die "Isaac Sim install receipt is absent"
[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
if ! receipt_matches \
  "$ISAAC_INSTALL_RECEIPT" \
  "component=isaac_sim" \
  "version=4.5.0" \
  "status=installed"; then
  die "Isaac Sim install receipt is incompatible"
fi
ISAAC_SOURCE_SHA256="$(receipt_field "$ISAAC_INSTALL_RECEIPT" source_sha256)"

if [ -e "$RECEIPT_PATH" ]; then
  ensure_pinned_git_source \
    "$ISAACLAB_REPOSITORY" \
    "$ISAACLAB_COMMIT" \
    "$SOURCE_PATH" \
    "isaaclab.sh"
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=isaaclab" \
    "version=$ISAACLAB_VERSION" \
    "status=installed" \
    "source_repository=$ISAACLAB_REPOSITORY" \
    "source_commit=$ISAACLAB_COMMIT" \
    "isaac_sim_source_sha256=$ISAAC_SOURCE_SHA256"; then
    info "IsaacLab v2.1.1 is already installed from the pinned commit"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing IsaacLab receipt does not match the pinned installation"
fi

ensure_pinned_git_source \
  "$ISAACLAB_REPOSITORY" \
  "$ISAACLAB_COMMIT" \
  "$SOURCE_PATH" \
  "isaaclab.sh"

ISAAC_LINK="$SOURCE_PATH/_isaac_sim"
if [ -L "$ISAAC_LINK" ]; then
  [ "$(readlink "$ISAAC_LINK")" = "$ISAAC_SIM_PATH" ] || \
    die "refusing to replace an incompatible IsaacLab _isaac_sim symlink"
elif [ -e "$ISAAC_LINK" ]; then
  die "refusing to replace an existing IsaacLab _isaac_sim path"
else
  ln -s "$ISAAC_SIM_PATH" "$ISAAC_LINK"
fi

LOG_PATH="$(new_log_path "isaaclab-v2.1.1-install")"
if ! run_logged \
  "$LOG_PATH" \
  "$ISAAC_SIM_PATH/python.sh" \
  -m pip install "flatdict==4.0.1" --no-build-isolation; then
  die "IsaacLab prerequisite installation failed; see $LOG_PATH"
fi
if ! (
  cd "$SOURCE_PATH"
  run_logged "$LOG_PATH" bash ./isaaclab.sh --install
); then
  die "IsaacLab installation failed; see $LOG_PATH"
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=isaaclab" \
  "version=$ISAACLAB_VERSION" \
  "status=installed" \
  "source_repository=$ISAACLAB_REPOSITORY" \
  "source_commit=$ISAACLAB_COMMIT" \
  "source_path=sources/$ISAACLAB_SOURCE_DIRECTORY" \
  "isaac_sim_source_sha256=$ISAAC_SOURCE_SHA256" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "IsaacLab v2.1.1 installation completed"
printf '%s\n' "$RECEIPT_PATH"

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
DEPENDENCY_CONSTRAINTS="$ROBOTACTILE_REPOSITORY_ROOT/requirements/isaaclab-v2.1.1-python310.lock.txt"
EXPECTED_PACKAGE_IDENTITY="0.41.3|0.2.2|1.0.7|0.1.8|0.10.36"
PACKAGE_IDENTITY_PROBE='import importlib.metadata as m; isaaclab_package_identity=(m.version("isaaclab"),m.version("isaaclab_assets"),m.version("isaaclab_mimic"),m.version("isaaclab_rl"),m.version("isaaclab_tasks")); print("|".join(isaaclab_package_identity))'
PIP_DEFAULT_TIMEOUT_SECONDS="600"
PIP_RETRIES="10"
PIP_FIND_LINKS_VALUE="${PIP_FIND_LINKS:-}"
WHEELHOUSE_PREINSTALLED="false"
if [ -n "$PIP_FIND_LINKS_VALUE" ]; then
  WHEELHOUSE_PREINSTALLED="true"
fi

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
[ -f "$DEPENDENCY_CONSTRAINTS" ] || die "IsaacLab dependency constraints are absent"
DEPENDENCY_CONSTRAINTS_SHA256="$(sha256_file "$DEPENDENCY_CONSTRAINTS")"
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
    "isaac_sim_source_sha256=$ISAAC_SOURCE_SHA256" \
    "pip_default_timeout_seconds=$PIP_DEFAULT_TIMEOUT_SECONDS" \
    "pip_retries=$PIP_RETRIES" \
    "pip_find_links=$PIP_FIND_LINKS_VALUE" \
    "wheelhouse_preinstalled=$WHEELHOUSE_PREINSTALLED" && \
    [ "$("$ISAAC_SIM_PATH/python.sh" -c "$PACKAGE_IDENTITY_PROBE")" = \
      "$EXPECTED_PACKAGE_IDENTITY" ]; then
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
  env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_PROMPT_MODIFIER \
  PIP_DEFAULT_TIMEOUT="$PIP_DEFAULT_TIMEOUT_SECONDS" \
  PIP_RETRIES="$PIP_RETRIES" \
  PIP_FIND_LINKS="$PIP_FIND_LINKS_VALUE" \
  "$ISAAC_SIM_PATH/python.sh" \
  -m pip install "flatdict==4.0.1" --no-build-isolation; then
  die "IsaacLab prerequisite installation failed; see $LOG_PATH"
fi
if [ -n "$PIP_FIND_LINKS_VALUE" ]; then
  if ! run_logged \
    "$LOG_PATH" \
    env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    PIP_DEFAULT_TIMEOUT="$PIP_DEFAULT_TIMEOUT_SECONDS" \
    PIP_RETRIES="$PIP_RETRIES" \
    "$ISAAC_SIM_PATH/python.sh" \
    -m pip install \
    --no-index \
    --find-links "$PIP_FIND_LINKS_VALUE" \
    --requirement "$DEPENDENCY_CONSTRAINTS" \
    --no-build-isolation; then
    die "IsaacLab wheelhouse preinstallation failed; see $LOG_PATH"
  fi
fi
if ! (
  cd "$SOURCE_PATH"
  # IsaacLab prefers an active Conda environment over its `_isaac_sim` link.
  # Clear Conda selection variables so a caller's shell cannot redirect the
  # installation into a user or system environment. A capable TERM also keeps
  # the upstream `tabs` call from failing in non-interactive SSH sessions.
  # UniVTAC needs the extensions, not optional RL frameworks: the default
  # `all` extra can replace the pinned Torch/CUDA pair with newer wheels.
  run_logged \
    "$LOG_PATH" \
    env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    PIP_CONSTRAINT="$DEPENDENCY_CONSTRAINTS" \
    PIP_DEFAULT_TIMEOUT="$PIP_DEFAULT_TIMEOUT_SECONDS" \
    PIP_RETRIES="$PIP_RETRIES" \
    PIP_FIND_LINKS="$PIP_FIND_LINKS_VALUE" \
    TERM=xterm-256color \
    bash ./isaaclab.sh --install none
); then
  die "IsaacLab installation failed; see $LOG_PATH"
fi

TORCH_IDENTITY="$(
  env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    "$ISAAC_SIM_PATH/python.sh" -c \
    'import torch; print(f"{torch.__version__}|{torch.version.cuda}")'
)"
[ "$TORCH_IDENTITY" = "2.7.0+cu128|12.8" ] || \
  die "IsaacLab installed an incompatible Torch/CUDA pair: $TORCH_IDENTITY"
PACKAGE_IDENTITY="$("$ISAAC_SIM_PATH/python.sh" -c "$PACKAGE_IDENTITY_PROBE")"
[ "$PACKAGE_IDENTITY" = "$EXPECTED_PACKAGE_IDENTITY" ] || \
  die "IsaacLab package installation is incomplete: ${PACKAGE_IDENTITY:-absent}"

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
  "dependency_constraints_path=requirements/isaaclab-v2.1.1-python310.lock.txt" \
  "dependency_constraints_sha256=$DEPENDENCY_CONSTRAINTS_SHA256" \
  "pip_default_timeout_seconds=$PIP_DEFAULT_TIMEOUT_SECONDS" \
  "pip_retries=$PIP_RETRIES" \
  "pip_find_links=$PIP_FIND_LINKS_VALUE" \
  "wheelhouse_preinstalled=$WHEELHOUSE_PREINSTALLED" \
  "package_identity=$PACKAGE_IDENTITY" \
  "torch_identity=$TORCH_IDENTITY" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "IsaacLab v2.1.1 installation completed"
printf '%s\n' "$RECEIPT_PATH"

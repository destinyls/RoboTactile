#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ISAAC_SIM_VERSION="4.5.0"
DEPLOY_ROOT="/data1/yanglei/robotactile_univtac_20260821"
ARCHIVE=""
EXPECTED_SHA256=""
STAGE_DIR=""

usage() {
  cat <<'EOF'
Usage: install_isaac_sim_4_5.sh --archive ZIP --sha256 HEX [--root PATH]

Installs the verified Isaac Sim 4.5.0 standalone archive without sudo and
without writing to the invoking user's HOME. Existing foreign installations
are never overwritten.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --archive)
      [ "$#" -ge 2 ] || die "--archive requires a value"
      ARCHIVE="$2"
      shift 2
      ;;
    --sha256)
      [ "$#" -ge 2 ] || die "--sha256 requires a value"
      EXPECTED_SHA256="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$ARCHIVE" ] || die "--archive is required"
[ -f "$ARCHIVE" ] || die "Isaac Sim archive is unavailable: $ARCHIVE"
[ -n "$EXPECTED_SHA256" ] || die "--sha256 is required"
validate_sha256 "$EXPECTED_SHA256"
EXPECTED_SHA256="$(printf '%s' "$EXPECTED_SHA256" | tr 'A-F' 'a-f')"

initialize_layout "$DEPLOY_ROOT"
require_command unzip
require_command python3
acquire_lock "isaac-sim-4.5.0-install"

cleanup() {
  if [ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR" ]; then
    safe_remove_temporary_directory "$STAGE_DIR"
  fi
  release_lock
}
trap cleanup EXIT

INSTALL_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/isaac_sim_install.json"
INCOMPLETE_MARKER="$INSTALL_PATH/.robotactile-install-incomplete"
ACTUAL_SHA256="$(sha256_file "$ARCHIVE")"

if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  die "SHA-256 mismatch for $ARCHIVE: expected $EXPECTED_SHA256, got $ACTUAL_SHA256"
fi

if [ -e "$RECEIPT_PATH" ]; then
  [ -d "$INSTALL_PATH" ] || die "receipt exists but Isaac Sim directory is absent"
  [ -f "$INSTALL_PATH/post_install.sh" ] || die "installed post_install.sh is absent"
  [ -x "$INSTALL_PATH/python.sh" ] || die "installed python.sh is absent or not executable"
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=isaac_sim" \
    "version=$ISAAC_SIM_VERSION" \
    "status=installed" \
    "source_sha256=$ACTUAL_SHA256" \
    "install_path=runtime/isaac-sim-4.5.0"; then
    info "Isaac Sim 4.5.0 is already installed with the same archive SHA-256"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing Isaac Sim receipt does not match the requested archive"
fi

if [ -e "$INSTALL_PATH" ]; then
  if [ ! -f "$INCOMPLETE_MARKER" ]; then
    die "refusing to overwrite foreign Isaac Sim destination: $INSTALL_PATH"
  fi
  MARKER_SHA256="$(sed -n '1p' "$INCOMPLETE_MARKER")"
  [ "$MARKER_SHA256" = "$ACTUAL_SHA256" ] || \
    die "refusing to resume an installation created from a different archive"
  [ -f "$INSTALL_PATH/post_install.sh" ] || die "incomplete post_install.sh is absent"
  [ -x "$INSTALL_PATH/python.sh" ] || die "incomplete python.sh is absent or not executable"
  info "resuming verified incomplete Isaac Sim post-install"
else
  STAGE_DIR="$(mktemp -d "$DEPLOY_ROOT/runtime/tmp/isaac-sim-extract.XXXXXX")"
  python3 - "$ARCHIVE" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1])
with zipfile.ZipFile(archive) as bundle:
    if not bundle.infolist():
        raise SystemExit("Isaac Sim archive is empty")
    for member in bundle.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive member: {member.filename!r}")
PY
  unzip -q "$ARCHIVE" -d "$STAGE_DIR"
  PAYLOAD_DIR="$(python3 - "$STAGE_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

stage = Path(sys.argv[1])
candidates = [
    path.parent
    for path in stage.glob("**/post_install.sh")
    if (path.parent / "python.sh").is_file()
]
if len(candidates) != 1:
    raise SystemExit(
        f"expected one Isaac Sim payload, found {len(candidates)} under {stage}"
    )
print(candidates[0])
PY
)"
  [ -d "$PAYLOAD_DIR" ] || die "Isaac Sim payload directory is absent"
  [ -x "$PAYLOAD_DIR/python.sh" ] || die "archive python.sh is not executable"
  mv "$PAYLOAD_DIR" "$INSTALL_PATH"
  printf '%s\n' "$ACTUAL_SHA256" > "$INCOMPLETE_MARKER"
fi

LOG_PATH="$(new_log_path "isaac-sim-post-install")"
if ! (
  cd "$INSTALL_PATH"
  run_logged "$LOG_PATH" bash ./post_install.sh
); then
  die "Isaac Sim post_install failed; see $LOG_PATH"
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=isaac_sim" \
  "version=$ISAAC_SIM_VERSION" \
  "status=installed" \
  "source_archive=$(basename "$ARCHIVE")" \
  "source_sha256=$ACTUAL_SHA256" \
  "install_path=runtime/isaac-sim-4.5.0" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"
rm -f "$INCOMPLETE_MARKER"

info "Isaac Sim 4.5.0 post-install completed"
printf '%s\n' "$RECEIPT_PATH"

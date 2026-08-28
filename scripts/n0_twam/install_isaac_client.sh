#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../live_univtac/common.sh
source "$SCRIPT_DIR/../live_univtac/common.sh"

DEPLOY_ROOT="$(default_deployment_root)"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    -h|--help) printf 'Usage: %s [--root PATH]\n' "$(basename "$0")"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

initialize_layout "$DEPLOY_ROOT"
acquire_lock "n0-twam-isaac-client-install"
trap release_lock EXIT
ISAAC_PYTHON="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh"
RECEIPT="$DEPLOY_ROOT/artifacts/deployment/n0_twam_isaac_client_install.json"
[ -x "$ISAAC_PYTHON" ] || die "Isaac Sim python.sh is absent"
PROBE='import msgpack,websockets; print(f"{msgpack.__version__}|{websockets.__version__}")'
if [ -f "$RECEIPT" ] && receipt_matches \
  "$RECEIPT" \
  "component=n0_twam_isaac_client" \
  "version=msgpack-1.1.1_websockets-15.0.1" \
  "status=installed" && \
  [ "$("$ISAAC_PYTHON" -c "$PROBE")" = "1.1.1|15.0.1" ]; then
  info "official N0-TWAM Isaac client dependencies are already installed"
  printf '%s\n' "$RECEIPT"
  exit 0
fi
[ ! -e "$RECEIPT" ] || die "existing N0 Isaac client receipt is incompatible"
LOG_PATH="$(new_log_path "n0-twam-isaac-client-install")"
if ! run_logged "$LOG_PATH" "$ISAAC_PYTHON" -m pip install \
  "msgpack==1.1.1" "websockets==15.0.1"; then
  die "N0 Isaac client dependency installation failed; see $LOG_PATH"
fi
[ "$("$ISAAC_PYTHON" -c "$PROBE")" = "1.1.1|15.0.1" ] || \
  die "N0 Isaac client dependency probe failed"
write_receipt "$RECEIPT" \
  "component=n0_twam_isaac_client" \
  "version=msgpack-1.1.1_websockets-15.0.1" \
  "status=installed" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$(sha256_file "$LOG_PATH")" \
  "system_python_modified=false" \
  "isaac_runtime_only=true"
printf '%s\n' "$RECEIPT"

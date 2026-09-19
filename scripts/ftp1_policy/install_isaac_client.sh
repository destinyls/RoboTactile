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
acquire_lock "ftp1-policy-isaac-client-install"
trap release_lock EXIT
ISAAC_PYTHON="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0/python.sh"
RECEIPT="$DEPLOY_ROOT/artifacts/deployment/ftp1_policy_isaac_client_install.json"
[ -x "$ISAAC_PYTHON" ] || die "Isaac Sim python.sh is absent"
PROBE='import importlib.metadata as m; print(m.version("msgpack")+"|"+m.version("pyzmq"))'
if [ -f "$RECEIPT" ] && receipt_matches \
  "$RECEIPT" \
  "component=ftp1_policy_isaac_client" \
  "version=msgpack-1.1.1_pyzmq-27.1.0" \
  "status=installed" && \
  [ "$("$ISAAC_PYTHON" -c "$PROBE")" = "1.1.1|27.1.0" ]; then
  info "official FTP-1 Isaac client dependencies are already installed"
  printf '%s\n' "$RECEIPT"
  exit 0
fi
[ ! -e "$RECEIPT" ] || die "existing FTP-1 Isaac client receipt is incompatible"
LOG_PATH="$(new_log_path "ftp1-policy-isaac-client-install")"
if ! run_logged "$LOG_PATH" "$ISAAC_PYTHON" -m pip install \
  "msgpack==1.1.1" "pyzmq==27.1.0"; then
  die "FTP-1 Isaac client dependency installation failed; see $LOG_PATH"
fi
[ "$("$ISAAC_PYTHON" -c "$PROBE")" = "1.1.1|27.1.0" ] || \
  die "FTP-1 Isaac client dependency probe failed"
write_receipt "$RECEIPT" \
  "component=ftp1_policy_isaac_client" \
  "version=msgpack-1.1.1_pyzmq-27.1.0" \
  "status=installed" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$(sha256_file "$LOG_PATH")" \
  "system_python_modified=false" \
  "system_cuda_modified=false" \
  "n0_runtime_modified=false" \
  "ftp1_server_runtime_modified=false" \
  "isaac_runtime_only=true"
printf '%s\n' "$RECEIPT"

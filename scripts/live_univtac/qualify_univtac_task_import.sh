#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

UNIVTAC_COMMIT="$(resolve_external_pin univtac commit_sha)"
UNIVTAC_REPOSITORY="$(resolve_external_pin univtac repository_url)"
UNIVTAC_SOURCE_DIRECTORY="$(resolve_external_pin univtac source_directory)"
DEPLOY_ROOT="$(default_deployment_root)"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
TASK_ID="pull_out_key"
REGISTRY_PATH="$ROBOTACTILE_REPOSITORY_ROOT/configs/univtac/tasks_v1.json"
REGISTRY_RESOURCE_SHA256="6f8d58b8efce09f1c8d3f1a97a9780ba722b748b0d23086b7051a8bf75272084"

usage() {
  cat <<'EOF'
Usage: qualify_univtac_task_import.sh [--root PATH] [--task TASK_ID]

Launches Isaac Sim headlessly and imports one frozen, source-bound TaskCfg. It
does not construct the task environment, execute physics, or report success.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      [ "$#" -ge 2 ] || die "--root requires a value"
      DEPLOY_ROOT="$2"
      shift 2
      ;;
    --task)
      [ "$#" -ge 2 ] || die "--task requires a value"
      TASK_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$TASK_ID" in
  ''|*[!A-Za-z0-9_]*) die "task ID may contain only letters, digits, and underscore" ;;
esac
require_command "$SYSTEM_PYTHON"
[ -f "$REGISTRY_PATH" ] || die "frozen UniVTAC task registry is absent"
[ "$(sha256_file "$REGISTRY_PATH")" = "$REGISTRY_RESOURCE_SHA256" ] || \
  die "frozen UniVTAC task registry hash mismatch"
mapfile -t TASK_FIELDS < <("$SYSTEM_PYTHON" - "$REGISTRY_PATH" "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tasks = document.get("tasks")
expected_ids = {
    "grasp_classify", "insert_HDMI", "insert_hole", "insert_tube",
    "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf",
}
if not isinstance(tasks, list) or len(tasks) != 8:
    raise SystemExit(1)
if any(not isinstance(item, dict) for item in tasks):
    raise SystemExit(1)
if {item.get("task_id") for item in tasks} != expected_ids:
    raise SystemExit(1)
matches = [item for item in tasks if item.get("task_id") == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(1)
entry = matches[0]
module = entry.get("module_name")
digest = entry.get("task_source_sha256")
if module != f"envs.{sys.argv[2]}":
    raise SystemExit(1)
if not isinstance(digest, str) or len(digest) != 64:
    raise SystemExit(1)
if any(character not in "0123456789abcdef" for character in digest):
    raise SystemExit(1)
print(module)
print(module.replace(".", "/") + ".py")
print(digest)
PY
)
[ "${#TASK_FIELDS[@]}" -eq 3 ] || die "task is absent from the frozen UniVTAC registry"
TASK_MODULE="${TASK_FIELDS[0]}"
TASK_SOURCE_RELATIVE="${TASK_FIELDS[1]}"
TASK_SOURCE_SHA256="${TASK_FIELDS[2]}"

initialize_layout "$DEPLOY_ROOT"
acquire_lock "univtac-task-import-$TASK_ID-qualification"
RESULT_DIRECTORY=""
cleanup() {
  safe_remove_temporary_directory "$RESULT_DIRECTORY"
  release_lock
}
trap cleanup EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
ISAACLAB_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/isaaclab_install.json"
TACEX_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_install.json"
UIPC_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_uipc_install.json"
UNIVTAC_PATH="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY"
UNIVTAC_RECEIPT="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY.robotactile-install.json"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/univtac_task_import_${TASK_ID}_v3.json"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
for receipt in \
  "$ISAACLAB_RECEIPT" \
  "$TACEX_RECEIPT" \
  "$UIPC_RECEIPT" \
  "$UNIVTAC_RECEIPT"; do
  [ -f "$receipt" ] || die "required deployment receipt is absent: $receipt"
done
receipt_matches \
  "$UIPC_RECEIPT" \
  "component=tacex_uipc" \
  "status=installed" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "cpptrace_overlay_version=0.8.3" \
  "tinygltf_overlay_version=2.9.3" || \
  die "modified UIPC receipt is incompatible"
ensure_pinned_git_source \
  "$UNIVTAC_REPOSITORY" \
  "$UNIVTAC_COMMIT" \
  "$UNIVTAC_PATH" \
  "$TASK_SOURCE_RELATIVE"
[ "$(sha256_file "$UNIVTAC_PATH/$TASK_SOURCE_RELATIVE")" = \
  "$TASK_SOURCE_SHA256" ] || die "pinned UniVTAC task source hash mismatch"

if [ -e "$RECEIPT_PATH" ]; then
  if receipt_matches \
    "$RECEIPT_PATH" \
    "component=univtac_task_import" \
    "version=$TASK_ID-v3" \
    "status=qualified" \
    "univtac_source_commit=$UNIVTAC_COMMIT" \
    "registry_resource_sha256=$REGISTRY_RESOURCE_SHA256" \
    "task_id=$TASK_ID" \
    "task_module=$TASK_MODULE" \
    "task_source_sha256=$TASK_SOURCE_SHA256" \
    "headless_extensions=omni.ui" \
    "task_instantiated=false" \
    "simulator_steps_executed=0" \
    "evidence_level=isaac_headless_task_config_import_v3"; then
    info "UniVTAC $TASK_ID task import is already qualified"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "existing UniVTAC task import receipt does not match"
fi

LOG_PATH="$(new_log_path "univtac-task-import-$TASK_ID-qualification")"
RESULT_DIRECTORY="$(mktemp -d "$DEPLOY_ROOT/runtime/tmp/task-import.XXXXXX")"
RESULT_PATH="$RESULT_DIRECTORY/result.json"
if ! run_logged \
  "$LOG_PATH" \
  env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_PROMPT_MODIFIER \
  ROBOTACTILE_UNIVTAC_ROOT="$UNIVTAC_PATH" \
  ROBOTACTILE_TASK_SMOKE_RESULT="$RESULT_PATH" \
  "$ISAAC_SIM_PATH/python.sh" \
  "$SCRIPT_DIR/smoke_univtac_task_isaac.py" \
  --task "$TASK_ID"; then
  die "UniVTAC task configuration import failed; see $LOG_PATH"
fi
[ -f "$RESULT_PATH" ] || die "UniVTAC task smoke result is absent"
if ! "$SYSTEM_PYTHON" - "$RESULT_PATH" "$TASK_ID" "$TASK_MODULE" \
  "$TASK_SOURCE_SHA256" "$REGISTRY_RESOURCE_SHA256" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "config_class": "TaskCfg",
    "status": "passed",
    "task_class": "Task",
    "task_id": sys.argv[2],
    "task_instantiated": False,
    "task_module": sys.argv[3],
    "task_source_sha256": sys.argv[4],
    "registry_resource_sha256": sys.argv[5],
}
if document.get("headless_extensions") != ["omni.ui"]:
    raise SystemExit(1)
if any(document.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
PY
then
  die "UniVTAC task smoke result contract failed"
fi

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=univtac_task_import" \
  "version=$TASK_ID-v3" \
  "status=qualified" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "registry_resource_sha256=$REGISTRY_RESOURCE_SHA256" \
  "task_id=$TASK_ID" \
  "task_module=$TASK_MODULE" \
  "task_source_sha256=$TASK_SOURCE_SHA256" \
  "task_config_class=TaskCfg" \
  "headless_extensions=omni.ui" \
  "task_instantiated=false" \
  "simulator_steps_executed=0" \
  "evidence_level=isaac_headless_task_config_import_v3" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

info "UniVTAC $TASK_ID task configuration import qualified"
printf '%s\n' "$RECEIPT_PATH"

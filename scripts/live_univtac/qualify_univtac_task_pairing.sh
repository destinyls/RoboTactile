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
GPU_INDEX="0"
INITIAL_SEED="17"
EXOGENOUS_SEED="29"
TIMEOUT_SECONDS="900"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
TASK_ID="pull_out_key"
ACTION_SPEC="qpos8_next_step"
REGISTRY_PATH="$ROBOTACTILE_REPOSITORY_ROOT/configs/univtac/tasks_v1.json"
REGISTRY_RESOURCE_SHA256="6f8d58b8efce09f1c8d3f1a97a9780ba722b748b0d23086b7051a8bf75272084"

usage() {
  cat <<'EOF'
Usage: qualify_univtac_task_pairing.sh [OPTIONS]

Options:
  --root PATH              Deployment root.
  --task TASK_ID           Frozen UniVTAC task (default: pull_out_key).
  --action-spec SPEC       qpos8_next_step or ee8_absolute.
  --gpu INDEX              Physical NVIDIA GPU index (default: 0).
  --initial-seed INTEGER   UniVTAC construction/reset seed (default: 17).
  --exogenous-seed INTEGER Matched-trial seed (default: 29).
  --timeout-seconds VALUE  Hard wall timeout, at least 60 (default: 900).
  --run-id ID              No-clobber receipt identity.

Runs one real frozen-task reset, one safe simulator action, and one exact
in-process snapshot replay. It loads no learned policy and evaluates no task
success; the receipt is reset-equivalence evidence, not Isaac qualification.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --task) [ "$#" -ge 2 ] || die "--task requires a value"; TASK_ID="$2"; shift 2 ;;
    --action-spec) [ "$#" -ge 2 ] || die "--action-spec requires a value"; ACTION_SPEC="$2"; shift 2 ;;
    --gpu) [ "$#" -ge 2 ] || die "--gpu requires a value"; GPU_INDEX="$2"; shift 2 ;;
    --initial-seed) [ "$#" -ge 2 ] || die "--initial-seed requires a value"; INITIAL_SEED="$2"; shift 2 ;;
    --exogenous-seed) [ "$#" -ge 2 ] || die "--exogenous-seed requires a value"; EXOGENOUS_SEED="$2"; shift 2 ;;
    --timeout-seconds) [ "$#" -ge 2 ] || die "--timeout-seconds requires a value"; TIMEOUT_SECONDS="$2"; shift 2 ;;
    --run-id) [ "$#" -ge 2 ] || die "--run-id requires a value"; RUN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$GPU_INDEX" "$INITIAL_SEED" "$EXOGENOUS_SEED" "$TIMEOUT_SECONDS"; do
  case "$value" in
    ''|*[!0-9]*) die "GPU, seeds, and timeout must be non-negative integers" ;;
  esac
done
[ "$TIMEOUT_SECONDS" -ge 60 ] || die "timeout must be at least 60 seconds"
case "$RUN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "run ID contains unsupported characters" ;;
esac

case "$TASK_ID" in ''|*[!A-Za-z0-9_]*) die "task ID contains unsupported characters" ;; esac
case "$ACTION_SPEC" in
  qpos8_next_step|ee8_absolute) ;;
  *) die "action spec must be qpos8_next_step or ee8_absolute" ;;
esac
ACTION_MODE="qpos"
[ "$ACTION_SPEC" = "ee8_absolute" ] && ACTION_MODE="ee"
require_command "$SYSTEM_PYTHON"
[ -f "$REGISTRY_PATH" ] || die "frozen UniVTAC task registry is absent"
[ "$(sha256_file "$REGISTRY_PATH")" = "$REGISTRY_RESOURCE_SHA256" ] || \
  die "frozen UniVTAC task registry hash mismatch"
mapfile -t TASK_FIELDS < <("$SYSTEM_PYTHON" - "$REGISTRY_PATH" "$TASK_ID" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
tasks = d.get("tasks")
ids = {"grasp_classify", "insert_HDMI", "insert_hole", "insert_tube", "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf"}
if not isinstance(tasks, list) or len(tasks) != 8 or any(not isinstance(x, dict) for x in tasks) or {x.get("task_id") for x in tasks} != ids:
    raise SystemExit(1)
matches = [x for x in tasks if x.get("task_id") == sys.argv[2]]
if len(matches) != 1 or matches[0].get("module_name") != f"envs.{sys.argv[2]}":
    raise SystemExit(1)
digest = matches[0].get("task_source_sha256")
if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    raise SystemExit(1)
print(matches[0]["module_name"].replace(".", "/") + ".py")
print(digest)
PY
)
[ "${#TASK_FIELDS[@]}" -eq 2 ] || die "task is absent from the frozen UniVTAC registry"
TASK_SOURCE_RELATIVE="${TASK_FIELDS[0]}"
TASK_SOURCE_SHA256="${TASK_FIELDS[1]}"
CONTRACT_RUN_ID="$TASK_ID-$ACTION_SPEC-$RUN_ID"

initialize_layout "$DEPLOY_ROOT"
require_command nvidia-smi
require_command timeout
acquire_lock "univtac-task-pairing-$CONTRACT_RUN_ID"
RESULT_DIRECTORY=""
cleanup() {
  safe_remove_temporary_directory "$RESULT_DIRECTORY"
  release_lock
}
trap cleanup EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
UNIVTAC_PATH="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY"
SOURCE_MANIFEST_PATH="$ROBOTACTILE_REPOSITORY_ROOT/release/source_manifest.sha256"
[ -f "$SOURCE_MANIFEST_PATH" ] || die "RoboTactile source manifest is absent"
CURRENT_SOURCE_MANIFEST_SHA256="$(sha256_file "$SOURCE_MANIFEST_PATH")"
validate_sha256 "$CURRENT_SOURCE_MANIFEST_SHA256"
ROBOTACTILE_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/robotactile_isaac_install-${CURRENT_SOURCE_MANIFEST_SHA256:0:16}.json"
UIPC_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_uipc_install.json"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/univtac_task_pairing_${CONTRACT_RUN_ID}.json"
SMOKE_PATH="$SCRIPT_DIR/smoke_univtac_task_pairing_isaac.py"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
for path in "$ROBOTACTILE_RECEIPT" "$UIPC_RECEIPT" "$SMOKE_PATH"; do
  [ -f "$path" ] || die "required pairing input is absent: $path"
done
[ ! -e "$RECEIPT_PATH" ] || die "refusing to overwrite pairing receipt"
receipt_matches "$ROBOTACTILE_RECEIPT" \
  "component=robotactile_isaac" "version=0.4.0" "status=installed" \
  "source_manifest_sha256=$CURRENT_SOURCE_MANIFEST_SHA256" || \
  die "RoboTactile Isaac receipt is incompatible"
receipt_matches "$UIPC_RECEIPT" "component=tacex_uipc" "status=installed" || \
  die "TacEx UIPC receipt is incompatible"
ensure_pinned_git_source \
  "$UNIVTAC_REPOSITORY" \
  "$UNIVTAC_COMMIT" \
  "$UNIVTAC_PATH" \
  "$TASK_SOURCE_RELATIVE"
[ "$(sha256_file "$UNIVTAC_PATH/$TASK_SOURCE_RELATIVE")" = \
  "$TASK_SOURCE_SHA256" ] || die "pinned UniVTAC task source hash mismatch"

RUNTIME_SOURCE_MANIFEST_SHA256="$(
  receipt_field "$ROBOTACTILE_RECEIPT" source_manifest_sha256
)"
validate_sha256 "$RUNTIME_SOURCE_MANIFEST_SHA256"
RUNTIME_PROBE='import hashlib; from robotactile_benchmark.resources import load_source_manifest; print(hashlib.sha256(load_source_manifest().encode("utf-8")).hexdigest())'
[ "$("$ISAAC_SIM_PATH/python.sh" -c "$RUNTIME_PROBE")" = \
  "$RUNTIME_SOURCE_MANIFEST_SHA256" ] || die "runtime source manifest mismatch"

GPU_IDENTITY="$(nvidia-smi --query-gpu=name,driver_version \
  --format=csv,noheader -i "$GPU_INDEX")"
RESULT_DIRECTORY="$(mktemp -d "$DEPLOY_ROOT/runtime/tmp/task-pairing.XXXXXX")"
RESULT_PATH="$RESULT_DIRECTORY/result.json"
TASK_RUNTIME_DIR="$RESULT_DIRECTORY/runtime"
mkdir "$TASK_RUNTIME_DIR"
LOG_PATH="$(new_log_path "univtac-task-pairing-$CONTRACT_RUN_ID")"
if ! run_logged "$LOG_PATH" timeout --signal=TERM --kill-after=30s \
  "${TIMEOUT_SECONDS}s" env \
  -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONHASHSEED="$INITIAL_SEED" \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  ROBOTACTILE_UNIVTAC_ROOT="$UNIVTAC_PATH" \
  ROBOTACTILE_TASK_RUNTIME_DIR="$TASK_RUNTIME_DIR" \
  ROBOTACTILE_TASK_PAIRING_RESULT="$RESULT_PATH" \
  ROBOTACTILE_INITIAL_SEED="$INITIAL_SEED" \
  ROBOTACTILE_EXOGENOUS_SEED="$EXOGENOUS_SEED" \
  "$ISAAC_SIM_PATH/python.sh" "$SMOKE_PATH" \
  --task "$TASK_ID" --action-spec "$ACTION_SPEC"; then
  die "UniVTAC task pairing qualification failed; see $LOG_PATH"
fi
[ -f "$RESULT_PATH" ] || die "pairing result is absent"

PARSED_PATH="$RESULT_DIRECTORY/parsed.txt"
if ! "$SYSTEM_PYTHON" - "$RESULT_PATH" "$TASK_ID" "$ACTION_SPEC" \
  "$TASK_SOURCE_SHA256" > "$PARSED_PATH" <<'PY'
import json
import math
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if document.get("status") != "passed":
    error_type = document.get("error_type", "unknown_error")
    error_message = document.get("error_message", "no error message")
    print(
        f"pairing smoke reported failure: {error_type}: {error_message}",
        file=sys.stderr,
    )
    raise SystemExit(1)
action_mode = "ee" if sys.argv[3] == "ee8_absolute" else "qpos"
expected = {
    "action_commands_executed": 1,
    "action_mode": action_mode,
    "action_spec": sys.argv[3],
    "divergence_proven": True,
    "evidence_level": "in_process_snapshot_replay_equivalence_v1",
    "policy_loaded": False,
    "robotactile_version": "0.4.0",
    "runtime_close_requested": True,
    "simulator_qualification_claimed": False,
    "status": "passed",
    "task_id": sys.argv[2],
    "task_source_sha256": sys.argv[4],
    "task_success_evaluated": False,
    "witness_count": 2,
}
if any(document.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
receipt = document.get("paired_reset_receipt")
if (
    not isinstance(receipt, dict)
    or receipt.get("all_exact") is not True
    or receipt.get("task_id") != sys.argv[2]
):
    raise SystemExit(1)
if [item.get("reset_mode") for item in receipt.get("witnesses", [])] != [
    "canonical_reset", "snapshot_replay"
]:
    raise SystemExit(1)
duration = document.get("duration_s")
if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
    raise SystemExit(1)
hashes = (
    document.get("canonical_state_sha256"),
    document.get("replay_state_sha256"),
    document.get("divergent_state_sha256"),
    document.get("paired_reset_receipt_sha256"),
)
if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
    raise SystemExit(1)
if hashes[0] != hashes[1] or hashes[0] == hashes[2]:
    raise SystemExit(1)
for value in hashes:
    print(value)
print(format(float(duration), ".6f"))
PY
then
  die "pairing result validation failed; see $LOG_PATH"
fi
mapfile -t FIELDS < "$PARSED_PATH"
[ "${#FIELDS[@]}" -eq 5 ] || die "unexpected pairing result field count"

write_receipt "$RECEIPT_PATH" \
  "component=univtac_task_pairing" \
  "version=$TASK_ID-$ACTION_SPEC-v2" \
  "status=qualified" \
  "evidence_level=in_process_snapshot_replay_equivalence_v1" \
  "evidence_boundary=one_live_reset_one_action_and_one_exact_snapshot_replay_only" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "registry_resource_sha256=$REGISTRY_RESOURCE_SHA256" \
  "task_source_sha256=$TASK_SOURCE_SHA256" \
  "runtime_source_manifest_sha256=$RUNTIME_SOURCE_MANIFEST_SHA256" \
  "requested_run_id=$RUN_ID" \
  "run_id=$CONTRACT_RUN_ID" \
  "contract_run_id=$CONTRACT_RUN_ID" \
  "task_id=$TASK_ID" \
  "action_spec=$ACTION_SPEC" \
  "action_mode=$ACTION_MODE" \
  "gpu_index=$GPU_INDEX" \
  "gpu_identity=$GPU_IDENTITY" \
  "initial_seed=$INITIAL_SEED" \
  "exogenous_seed=$EXOGENOUS_SEED" \
  "canonical_state_sha256=${FIELDS[0]}" \
  "replay_state_sha256=${FIELDS[1]}" \
  "divergent_state_sha256=${FIELDS[2]}" \
  "paired_reset_receipt_sha256=${FIELDS[3]}" \
  "duration_s=${FIELDS[4]}" \
  "witness_count=2" \
  "all_exact=true" \
  "policy_loaded=false" \
  "task_success_evaluated=false" \
  "simulator_qualification_claimed=false" \
  "runtime_process_exited=true" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$(sha256_file "$LOG_PATH")"

info "UniVTAC $TASK_ID/$ACTION_SPEC in-process task pairing qualified"
printf '%s\n' "$RECEIPT_PATH"

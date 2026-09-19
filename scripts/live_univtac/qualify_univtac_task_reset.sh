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
Usage: qualify_univtac_task_reset.sh [OPTIONS]

Options:
  --root PATH              Deployment root.
  --task TASK_ID           Frozen UniVTAC task (default: pull_out_key).
  --action-spec SPEC       qpos8_next_step or ee8_absolute.
  --gpu INDEX              Physical NVIDIA GPU index (default: 0).
  --initial-seed INTEGER   UniVTAC reset seed (default: 17).
  --exogenous-seed INTEGER Reserved matched-trial seed (default: 29).
  --timeout-seconds VALUE  Hard wall timeout, at least 60 (default: 900).
  --run-id ID              No-clobber receipt identity.

Constructs one pinned frozen task, completes reset/pre-move, captures and
validates one multimodal observation, then closes the runtime. It does not load
a policy, execute an action command, run a control cycle, or evaluate success.
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
    -h|--help)
      usage
      exit 0
      ;;
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
  ''|*[!A-Za-z0-9._-]*) die "run ID may contain only letters, digits, dot, dash, underscore" ;;
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
acquire_lock "univtac-task-reset-$CONTRACT_RUN_ID"
RESULT_DIRECTORY=""
cleanup() {
  safe_remove_temporary_directory "$RESULT_DIRECTORY"
  release_lock
}
trap cleanup EXIT

ISAAC_SIM_PATH="$DEPLOY_ROOT/runtime/isaac-sim-4.5.0"
UNIVTAC_PATH="$DEPLOY_ROOT/sources/$UNIVTAC_SOURCE_DIRECTORY"
UIPC_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/tacex_uipc_install.json"
IMPORT_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/univtac_task_import_${TASK_ID}_v3.json"
SOURCE_MANIFEST_PATH="$ROBOTACTILE_REPOSITORY_ROOT/release/source_manifest.sha256"
[ -f "$SOURCE_MANIFEST_PATH" ] || die "RoboTactile source manifest is absent"
CURRENT_SOURCE_MANIFEST_SHA256="$(sha256_file "$SOURCE_MANIFEST_PATH")"
validate_sha256 "$CURRENT_SOURCE_MANIFEST_SHA256"
ROBOTACTILE_RECEIPT="$DEPLOY_ROOT/artifacts/deployment/robotactile_isaac_install-${CURRENT_SOURCE_MANIFEST_SHA256:0:16}.json"
RECEIPT_PATH="$DEPLOY_ROOT/artifacts/deployment/univtac_task_reset_${CONTRACT_RUN_ID}.json"
WITNESS_PATH="$DEPLOY_ROOT/artifacts/deployment/univtac_task_reset_witness_${CONTRACT_RUN_ID}.json"
WITNESS_RELPATH="$(relative_to_deploy_root "$WITNESS_PATH")"
SMOKE_PATH="$SCRIPT_DIR/smoke_univtac_task_reset_isaac.py"

[ -x "$ISAAC_SIM_PATH/python.sh" ] || die "Isaac Sim python.sh is absent"
for path in "$UIPC_RECEIPT" "$IMPORT_RECEIPT" "$ROBOTACTILE_RECEIPT" "$SMOKE_PATH"; do
  [ -f "$path" ] || die "required qualification input is absent: $path"
done
receipt_matches \
  "$UIPC_RECEIPT" \
  "component=tacex_uipc" \
  "status=installed" \
  "univtac_source_commit=$UNIVTAC_COMMIT" || \
  die "modified UIPC receipt is incompatible"
receipt_matches \
  "$IMPORT_RECEIPT" \
  "component=univtac_task_import" \
  "version=$TASK_ID-v3" \
  "status=qualified" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "registry_resource_sha256=$REGISTRY_RESOURCE_SHA256" \
  "task_id=$TASK_ID" \
  "task_source_sha256=$TASK_SOURCE_SHA256" \
  "task_instantiated=false" \
  "simulator_steps_executed=0" || \
  die "UniVTAC task-import qualification receipt is incompatible"
receipt_matches \
  "$ROBOTACTILE_RECEIPT" \
  "component=robotactile_isaac" \
  "version=0.6.0" \
  "status=installed" \
  "source_manifest_sha256=$CURRENT_SOURCE_MANIFEST_SHA256" || \
  die "RoboTactile Isaac receipt is incompatible"
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
RUNTIME_MANIFEST_IDENTITY="$(
  env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    "$ISAAC_SIM_PATH/python.sh" -c "$RUNTIME_PROBE"
)"
[ "$RUNTIME_MANIFEST_IDENTITY" = "$RUNTIME_SOURCE_MANIFEST_SHA256" ] || \
  die "installed RoboTactile source manifest does not match its receipt"

QUALIFIER_SHA256="$(sha256_file "$0")"
SMOKE_SHA256="$(sha256_file "$SMOKE_PATH")"
if [ -e "$RECEIPT_PATH" ]; then
  if [ -f "$WITNESS_PATH" ] && receipt_matches \
    "$RECEIPT_PATH" \
    "component=univtac_task_reset" \
    "version=$TASK_ID-$ACTION_SPEC-v3" \
    "status=qualified" \
    "reset_viable=true" \
    "univtac_source_commit=$UNIVTAC_COMMIT" \
    "runtime_source_manifest_sha256=$RUNTIME_SOURCE_MANIFEST_SHA256" \
    "qualifier_sha256=$QUALIFIER_SHA256" \
    "smoke_sha256=$SMOKE_SHA256" \
    "contract_run_id=$CONTRACT_RUN_ID" \
    "run_id=$CONTRACT_RUN_ID" \
    "task_id=$TASK_ID" \
    "action_spec=$ACTION_SPEC" \
    "task_source_sha256=$TASK_SOURCE_SHA256" \
    "gpu_index=$GPU_INDEX" \
    "initial_seed=$INITIAL_SEED" \
    "construction_seed=$INITIAL_SEED" \
    "process_seed=$INITIAL_SEED" \
    "exogenous_seed=$EXOGENOUS_SEED" \
    "task_instantiated=true" \
    "reset_completed=true" \
    "observation_captured=true" \
    "runtime_close_requested=true" \
    "policy_loaded=false" \
    "action_commands_executed=0" \
    "closed_loop_control_cycles=0" \
    "witness_relpath=$WITNESS_RELPATH" \
    "witness_sha256=$(sha256_file "$WITNESS_PATH")"; then
    info "UniVTAC reset qualification already matches: $CONTRACT_RUN_ID"
    printf '%s\n' "$RECEIPT_PATH"
    exit 0
  fi
  die "refusing to overwrite a different reset qualification receipt"
fi
[ ! -e "$WITNESS_PATH" ] || \
  die "refusing to overwrite a reset qualification witness"

GPU_IDENTITY="$(nvidia-smi \
  --query-gpu=name,driver_version \
  --format=csv,noheader \
  -i "$GPU_INDEX")" || die "nvidia-smi failed for GPU $GPU_INDEX"
[ -n "$GPU_IDENTITY" ] || die "nvidia-smi returned an empty GPU identity"

RESULT_DIRECTORY="$(mktemp -d "$DEPLOY_ROOT/runtime/tmp/task-reset.XXXXXX")"
RESULT_PATH="$RESULT_DIRECTORY/result.json"
TASK_RUNTIME_DIR="$RESULT_DIRECTORY/runtime"
mkdir "$TASK_RUNTIME_DIR"
LOG_PATH="$(new_log_path "univtac-task-reset-$CONTRACT_RUN_ID")"
if ! run_logged \
  "$LOG_PATH" \
  timeout \
  --signal=TERM \
  --kill-after=30s \
  "${TIMEOUT_SECONDS}s" \
  env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_PROMPT_MODIFIER \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  ROBOTACTILE_UNIVTAC_ROOT="$UNIVTAC_PATH" \
  ROBOTACTILE_TASK_RUNTIME_DIR="$TASK_RUNTIME_DIR" \
  ROBOTACTILE_TASK_RESET_RESULT="$RESULT_PATH" \
  ROBOTACTILE_INITIAL_SEED="$INITIAL_SEED" \
  ROBOTACTILE_EXOGENOUS_SEED="$EXOGENOUS_SEED" \
  "$ISAAC_SIM_PATH/python.sh" \
  "$SMOKE_PATH" \
  --task "$TASK_ID" \
  --action-spec "$ACTION_SPEC"; then
  die "UniVTAC task reset qualification failed; see $LOG_PATH"
fi
[ -f "$RESULT_PATH" ] || die "UniVTAC task reset result is absent"

if ! RESULT_FIELDS_PAYLOAD="$("$SYSTEM_PYTHON" - "$RESULT_PATH" "$INITIAL_SEED" \
  "$TASK_ID" "$ACTION_SPEC" "$TASK_SOURCE_SHA256" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
initial_seed = int(sys.argv[2])
action_mode = "ee" if sys.argv[4] == "ee8_absolute" else "qpos"
expected = {
    "action_commands_executed": 0,
    "action_mode": action_mode,
    "action_spec": sys.argv[4],
    "closed_loop_control_cycles": 0,
    "construction_seed": initial_seed,
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "numpy_version": "1.26.0",
    "observation_captured": True,
    "observation_step_index": 0,
    "plan_success": True,
    "policy_loaded": False,
    "pythonhashseed": str(initial_seed),
    "reset_completed": True,
    "robotactile_version": "0.6.0",
    "runtime_close_requested": True,
    "simulator_advanced_during_reset": True,
    "status": "passed",
    "task_id": sys.argv[3],
    "task_instantiated": True,
    "task_source_sha256": sys.argv[5],
    "torch_deterministic_algorithms": True,
}
unexpected = {
    key: {"expected": value, "observed": document.get(key)}
    for key, value in expected.items()
    if document.get(key) != value
}
if unexpected:
    print(
        "reset result contract mismatch: "
        + json.dumps(unexpected, sort_keys=True),
        file=sys.stderr,
    )
    raise SystemExit(1)
contract = document.get("observation_contract")
expected_contract = {
    "proprio": {"dtype": "float32", "shape": [8]},
    "tactile": {
        "left": {"dtype": "uint8", "shape": [240, 320, 3]},
        "right": {"dtype": "uint8", "shape": [240, 320, 3]},
    },
    "vision": {
        "top": {"dtype": "uint8", "shape": [270, 480, 3]},
        "wrist_l": {"dtype": "uint8", "shape": [270, 480, 3]},
    },
}
if contract != expected_contract:
    raise SystemExit(1)
phases = document.get("phase_by_slot")
valid_phases = {"free", "contact_onset", "sustained_contact", "release"}
if not isinstance(phases, dict) or set(phases) != {"left", "right"}:
    raise SystemExit(1)
if not set(phases.values()).issubset(valid_phases):
    raise SystemExit(1)
duration = document.get("reset_duration_s")
native_step = document.get("post_reset_native_step_id")
if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
    raise SystemExit(1)
if isinstance(native_step, bool) or not isinstance(native_step, int) or native_step <= 0:
    raise SystemExit(1)
hash_fields = (
    "config_sha256",
    "handshake_sha256",
    "initial_clean_record_sha256",
    "joint_reorder_witness_sha256",
    "reset_receipt_sha256",
    "simulator_state_sha256",
)
for field in hash_fields:
    value = document.get(field)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(1)
    if any(character not in "0123456789abcdef" for character in value):
        raise SystemExit(1)
for field in hash_fields:
    print(document[field])
print(native_step)
print(format(float(duration), ".6f"))
print(phases["left"])
print(phases["right"])
print(document.get("torch_version", ""))
print(document.get("torch_cuda_version", ""))

reset_diagnostics = document.get("reset_diagnostics")
if not isinstance(reset_diagnostics, dict):
    raise SystemExit("reset diagnostics must be a mapping")
for field in ("success_check", "plan_success", "early_stop"):
    if type(reset_diagnostics.get(field)) is not bool:
        raise SystemExit(f"reset diagnostic {field} must be bool")
task_diagnostics = reset_diagnostics.get("task")
if not isinstance(task_diagnostics, dict):
    raise SystemExit("reset task diagnostics must be a mapping")
assessment = document.get("placement_reset_assessment")
is_insertion = sys.argv[3] in {"insert_hole", "insert_tube"}
if is_insertion:
    if not isinstance(assessment, dict):
        raise SystemExit("insertion placement reset assessment is absent")
    if task_diagnostics.get("placement_reset_assessment") != assessment:
        raise SystemExit("placement reset assessment disagrees with task diagnostics")
    if assessment.get("assessment_schema") != "univtac-insertion-placement-reset-v1":
        raise SystemExit("placement reset assessment schema mismatch")
    if assessment.get("available") is not True:
        raise SystemExit("placement reset assessment is unavailable")
    if assessment.get("current_early_stop") is not reset_diagnostics["early_stop"]:
        raise SystemExit("placement assessment early-stop state disagrees with reset")
    if type(assessment.get("reset_viable")) is not bool:
        raise SystemExit("placement reset viability must be bool")
    failure_phase = assessment.get("failure_phase")
    if failure_phase is not None and not isinstance(failure_phase, str):
        raise SystemExit("placement failure phase must be a string or null")
    summaries = assessment.get("phase_summaries")
    if not isinstance(summaries, list) or len(summaries) > 2:
        raise SystemExit("placement phase summaries must contain at most two phases")
    observed_phases = [
        item.get("phase") for item in summaries if isinstance(item, dict)
    ]
    expected_phases = ["approach_complete", "final_continuation_complete"]
    if observed_phases != expected_phases[: len(observed_phases)]:
        raise SystemExit("placement phase summaries are invalid or out of order")
    if assessment.get("observed_phase_count") != len(observed_phases):
        raise SystemExit("placement observed phase count mismatch")
    if assessment.get("expected_phase_count") != 2:
        raise SystemExit("placement expected phase count mismatch")
    phase_sequence_complete = assessment.get("phase_sequence_complete")
    if type(phase_sequence_complete) is not bool:
        raise SystemExit("placement phase sequence completeness must be bool")
    if phase_sequence_complete is not (len(observed_phases) == 2):
        raise SystemExit("placement phase sequence completeness mismatch")
    for item in summaries:
        for field in (
            "before_inhand_z_bias_m",
            "after_inhand_z_bias_m",
            "inhand_z_bias_delta_m",
        ):
            value = item.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise SystemExit(f"placement phase {field} must be finite")
        for field in (
            "early_stop_before",
            "early_stop_after",
            "threshold_crossed",
            "plan_success_after",
        ):
            if type(item.get(field)) is not bool:
                raise SystemExit(f"placement phase {field} must be bool")
    reset_viable = bool(
        reset_diagnostics["plan_success"]
        and not reset_diagnostics["early_stop"]
        and assessment["reset_viable"]
    )
    if reset_viable and failure_phase is not None:
        raise SystemExit("viable placement reset cannot have a failure phase")
    if not reset_viable and failure_phase is None:
        raise SystemExit("non-viable placement reset must have a failure phase")
    assessment_schema = str(assessment["assessment_schema"])
else:
    if assessment is not None:
        raise SystemExit("non-insertion task unexpectedly has placement assessment")
    reset_viable = bool(
        reset_diagnostics["plan_success"] and not reset_diagnostics["early_stop"]
    )
    failure_phase = None if reset_viable else "post_reset_state"
    assessment_schema = "not_applicable"
print(str(reset_viable).lower())
print("none" if failure_phase is None else failure_phase)
print(assessment_schema)
print(str(reset_diagnostics["success_check"]).lower())
print(str(reset_diagnostics["plan_success"]).lower())
print(str(reset_diagnostics["early_stop"]).lower())
PY
)"; then
  die "reset result contract validation failed; see $LOG_PATH"
fi
mapfile -t RESULT_FIELDS <<<"$RESULT_FIELDS_PAYLOAD"
[ "${#RESULT_FIELDS[@]}" -eq 18 ] || die "unexpected reset result field count"

"$SYSTEM_PYTHON" - "$RESULT_PATH" "$WITNESS_PATH" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
document = json.loads(source.read_text(encoding="utf-8"))
serialized = json.dumps(
    document,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
) + "\n"
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
try:
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, target)
except FileExistsError as error:
    raise SystemExit(f"refusing to overwrite reset witness: {target}") from error
finally:
    temporary.unlink(missing_ok=True)
PY
WITNESS_SHA256="$(sha256_file "$WITNESS_PATH")"
RESET_VIABLE="${RESULT_FIELDS[12]}"
QUALIFICATION_STATUS="failed"
[ "$RESET_VIABLE" = "true" ] && QUALIFICATION_STATUS="qualified"

LOG_SHA256="$(sha256_file "$LOG_PATH")"
write_receipt \
  "$RECEIPT_PATH" \
  "component=univtac_task_reset" \
  "version=$TASK_ID-$ACTION_SPEC-v3" \
  "status=$QUALIFICATION_STATUS" \
  "evidence_level=isaac_headless_task_reset_observation_v2" \
  "evidence_boundary=task_instantiation_reset_placement_viability_and_one_observation_only" \
  "univtac_source_commit=$UNIVTAC_COMMIT" \
  "registry_resource_sha256=$REGISTRY_RESOURCE_SHA256" \
  "task_source_sha256=$TASK_SOURCE_SHA256" \
  "runtime_source_manifest_sha256=$RUNTIME_SOURCE_MANIFEST_SHA256" \
  "qualifier_sha256=$QUALIFIER_SHA256" \
  "smoke_sha256=$SMOKE_SHA256" \
  "requested_run_id=$RUN_ID" \
  "run_id=$CONTRACT_RUN_ID" \
  "contract_run_id=$CONTRACT_RUN_ID" \
  "gpu_index=$GPU_INDEX" \
  "gpu_identity=$GPU_IDENTITY" \
  "initial_seed=$INITIAL_SEED" \
  "construction_seed=$INITIAL_SEED" \
  "process_seed=$INITIAL_SEED" \
  "pythonhashseed=$INITIAL_SEED" \
  "cublas_workspace_config=:4096:8" \
  "torch_deterministic_algorithms=true" \
  "cudnn_deterministic=true" \
  "cudnn_benchmark=false" \
  "exogenous_seed=$EXOGENOUS_SEED" \
  "task_id=$TASK_ID" \
  "action_spec=$ACTION_SPEC" \
  "action_mode=$ACTION_MODE" \
  "task_instantiated=true" \
  "reset_completed=true" \
  "observation_captured=true" \
  "runtime_close_requested=true" \
  "runtime_process_exited=true" \
  "simulator_advanced_during_reset=true" \
  "post_reset_native_step_id=${RESULT_FIELDS[6]}" \
  "reset_duration_s=${RESULT_FIELDS[7]}" \
  "left_contact_phase=${RESULT_FIELDS[8]}" \
  "right_contact_phase=${RESULT_FIELDS[9]}" \
  "reset_viable=$RESET_VIABLE" \
  "placement_failure_phase=${RESULT_FIELDS[13]}" \
  "placement_assessment_schema=${RESULT_FIELDS[14]}" \
  "initial_success_check=${RESULT_FIELDS[15]}" \
  "initial_plan_success=${RESULT_FIELDS[16]}" \
  "initial_early_stop=${RESULT_FIELDS[17]}" \
  "observation_contract=top,wrist_l:uint8[270,480,3];left,right:uint8[240,320,3];proprio:float32[8]" \
  "config_sha256=${RESULT_FIELDS[0]}" \
  "handshake_sha256=${RESULT_FIELDS[1]}" \
  "initial_clean_record_sha256=${RESULT_FIELDS[2]}" \
  "joint_reorder_witness_sha256=${RESULT_FIELDS[3]}" \
  "reset_receipt_sha256=${RESULT_FIELDS[4]}" \
  "simulator_state_sha256=${RESULT_FIELDS[5]}" \
  "torch_version=${RESULT_FIELDS[10]}" \
  "torch_cuda_version=${RESULT_FIELDS[11]}" \
  "policy_loaded=false" \
  "action_commands_executed=0" \
  "closed_loop_control_cycles=0" \
  "task_success_evaluated=false" \
  "witness_relpath=$WITNESS_RELPATH" \
  "witness_sha256=$WITNESS_SHA256" \
  "timeout_seconds=$TIMEOUT_SECONDS" \
  "log_path=$(relative_to_deploy_root "$LOG_PATH")" \
  "log_sha256=$LOG_SHA256"

[ "$RESET_VIABLE" = "true" ] || \
  die "UniVTAC $TASK_ID/$ACTION_SPEC reset is not viable; phase=${RESULT_FIELDS[13]}"
info "UniVTAC $TASK_ID/$ACTION_SPEC task reset and observation qualified"
printf '%s\n' "$RECEIPT_PATH"

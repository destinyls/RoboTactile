#!/usr/bin/env bash

# Qualify every frozen UniVTAC task for the official N0 EE8 contract.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

DEPLOY_ROOT="$(default_deployment_root)"
SYSTEM_PYTHON="${ROBOTACTILE_SYSTEM_PYTHON:-python3}"
GPU_INDEX="0"
ACTION_SPEC="ee8_absolute"
REPETITIONS="1"
MASTER_SEED="20260823"
CAMPAIGN_ID="qualification-v1"
TASKS=(
  grasp_classify
  insert_HDMI
  insert_hole
  insert_tube
  lift_bottle
  lift_can
  pull_out_key
  put_bottle_in_shelf
)

usage() {
  cat <<'EOF'
Usage: qualify_univtac_all_tasks.sh [OPTIONS]

Options:
  --root PATH            Deployment root.
  --gpu INDEX            Physical NVIDIA GPU index (default: 0).
  --action-spec SPEC     qpos8_next_step or ee8_absolute (default: ee8_absolute).
  --repetitions N        Reset/pairing executions per task (default: 1).
  --master-seed INTEGER  Deterministic per-task seed root (default: 20260823).
  --campaign-id ID       No-clobber qualification identity.

This is a simulator/task/action-contract precondition. It loads no policy,
does not execute a closed-loop episode, and does not claim a success rate.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) [ "$#" -ge 2 ] || die "--root requires a value"; DEPLOY_ROOT="$2"; shift 2 ;;
    --gpu) [ "$#" -ge 2 ] || die "--gpu requires a value"; GPU_INDEX="$2"; shift 2 ;;
    --action-spec) [ "$#" -ge 2 ] || die "--action-spec requires a value"; ACTION_SPEC="$2"; shift 2 ;;
    --repetitions) [ "$#" -ge 2 ] || die "--repetitions requires a value"; REPETITIONS="$2"; shift 2 ;;
    --master-seed) [ "$#" -ge 2 ] || die "--master-seed requires a value"; MASTER_SEED="$2"; shift 2 ;;
    --campaign-id) [ "$#" -ge 2 ] || die "--campaign-id requires a value"; CAMPAIGN_ID="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$GPU_INDEX" "$REPETITIONS" "$MASTER_SEED"; do
  case "$value" in ''|*[!0-9]*) die "GPU, repetitions, and master seed must be non-negative integers" ;; esac
done
[ "$REPETITIONS" -ge 1 ] || die "qualification requires at least one execution"
case "$ACTION_SPEC" in qpos8_next_step|ee8_absolute) ;; *) die "unsupported action spec" ;; esac
case "$CAMPAIGN_ID" in
  ''|*[!A-Za-z0-9._-]*) die "campaign ID contains unsupported characters" ;;
esac

initialize_layout "$DEPLOY_ROOT"
require_command "$SYSTEM_PYTHON"
acquire_lock "univtac-all-tasks-$CAMPAIGN_ID"
trap release_lock EXIT

seed_pair() {
  local task_id="$1"
  "$SYSTEM_PYTHON" - "$MASTER_SEED" "$task_id" <<'PY'
import hashlib
import json
import sys

master_seed = int(sys.argv[1])
task_id = sys.argv[2]
for role in ("initial", "exogenous"):
    payload = json.dumps(
        {
            "counter": 0,
            "master_seed": master_seed,
            "namespace": "qualification_v1",
            "ordinal": 0,
            "role": role,
            "task_id": task_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
    if value == 0:
        raise SystemExit("derived zero seed")
    print(value)
PY
}

for task_id in "${TASKS[@]}"; do
  "$SCRIPT_DIR/qualify_univtac_task_import.sh" \
    --root "$DEPLOY_ROOT" \
    --task "$task_id"
done

for task_id in "${TASKS[@]}"; do
  mapfile -t seeds < <(seed_pair "$task_id")
  [ "${#seeds[@]}" -eq 2 ] || die "failed to derive qualification seeds"
  initial_seed="${seeds[0]}"
  exogenous_seed="${seeds[1]}"
  repeat="0"
  while [ "$repeat" -lt "$REPETITIONS" ]; do
    run_id="$CAMPAIGN_ID-r$repeat"
    "$SCRIPT_DIR/qualify_univtac_task_reset.sh" \
      --root "$DEPLOY_ROOT" \
      --task "$task_id" \
      --action-spec "$ACTION_SPEC" \
      --gpu "$GPU_INDEX" \
      --initial-seed "$initial_seed" \
      --exogenous-seed "$exogenous_seed" \
      --run-id "$run_id"
    pairing_receipt="$DEPLOY_ROOT/artifacts/deployment/univtac_task_pairing_${task_id}-${ACTION_SPEC}-${run_id}.json"
    if [ -e "$pairing_receipt" ]; then
      receipt_matches \
        "$pairing_receipt" \
        "component=univtac_task_pairing" \
        "status=qualified" \
        "task_id=$task_id" \
        "action_spec=$ACTION_SPEC" \
        "initial_seed=$initial_seed" \
        "exogenous_seed=$exogenous_seed" \
        "all_exact=true" || die "existing pairing receipt is incompatible"
    else
      "$SCRIPT_DIR/qualify_univtac_task_pairing.sh" \
        --root "$DEPLOY_ROOT" \
        --task "$task_id" \
        --action-spec "$ACTION_SPEC" \
        --gpu "$GPU_INDEX" \
        --initial-seed "$initial_seed" \
        --exogenous-seed "$exogenous_seed" \
        --run-id "$run_id"
    fi
    repeat="$((repeat + 1))"
  done
done

OUTPUT="$DEPLOY_ROOT/artifacts/deployment/univtac_all_tasks_qualification_${CAMPAIGN_ID}.json"
"$SYSTEM_PYTHON" - \
  "$DEPLOY_ROOT" "$OUTPUT" "$CAMPAIGN_ID" "$ACTION_SPEC" \
  "$REPETITIONS" "$MASTER_SEED" "${TASKS[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_receipt(path: Path, expected: dict[str, str]) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"qualification receipt is unavailable: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit("qualification receipt must be an object")
    if any(document.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"qualification receipt identity mismatch: {path}")
    return document


root = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
campaign_id = sys.argv[3]
action_spec = sys.argv[4]
repetitions = int(sys.argv[5])
master_seed = int(sys.argv[6])
tasks = tuple(sys.argv[7:])
action_mode = "ee" if action_spec == "ee8_absolute" else "qpos"
inventory: list[dict[str, object]] = []
observed_seeds: set[int] = set()
for task_id in tasks:
    derived: list[int] = []
    for role in ("initial", "exogenous"):
        payload = json.dumps(
            {
                "counter": 0,
                "master_seed": master_seed,
                "namespace": "qualification_v1",
                "ordinal": 0,
                "role": role,
                "task_id": task_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        derived.append(
            int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
            & 0x7FFFFFFF
        )
    seed_initial, seed_exogenous = derived
    if 0 in derived or observed_seeds.intersection(derived):
        raise SystemExit("qualification seed collision")
    observed_seeds.update(derived)
    import_path = root / f"artifacts/deployment/univtac_task_import_{task_id}_v3.json"
    load_receipt(
        import_path,
        {"component": "univtac_task_import", "status": "qualified", "task_id": task_id},
    )
    reset_paths: list[str] = []
    pairing_paths: list[str] = []
    reset_hashes: list[str] = []
    pairing_hashes: list[str] = []
    reset_invariants: list[tuple[object, ...]] = []
    pairing_invariants: list[tuple[object, ...]] = []
    for repeat in range(repetitions):
        run_id = f"{task_id}-{action_spec}-{campaign_id}-r{repeat}"
        reset_path = root / f"artifacts/deployment/univtac_task_reset_{run_id}.json"
        reset = load_receipt(
            reset_path,
            {
                "action_mode": action_mode,
                "action_spec": action_spec,
                "component": "univtac_task_reset",
                "exogenous_seed": str(seed_exogenous),
                "initial_seed": str(seed_initial),
                "status": "qualified",
                "task_id": task_id,
            },
        )
        reset_invariants.append(
            tuple(
                reset.get(key)
                for key in (
                    "config_sha256",
                    "handshake_sha256",
                    "initial_clean_record_sha256",
                    "joint_reorder_witness_sha256",
                    "reset_receipt_sha256",
                    "simulator_state_sha256",
                    "post_reset_native_step_id",
                    "left_contact_phase",
                    "right_contact_phase",
                )
            )
        )
        pairing_path = root / f"artifacts/deployment/univtac_task_pairing_{run_id}.json"
        pairing = load_receipt(
            pairing_path,
            {
                "action_mode": action_mode,
                "action_spec": action_spec,
                "all_exact": "true",
                "component": "univtac_task_pairing",
                "exogenous_seed": str(seed_exogenous),
                "initial_seed": str(seed_initial),
                "status": "qualified",
                "task_id": task_id,
            },
        )
        pairing_invariants.append(
            tuple(
                pairing.get(key)
                for key in (
                    "canonical_state_sha256",
                    "replay_state_sha256",
                    "divergent_state_sha256",
                )
            )
        )
        if pairing_invariants[-1][0] != pairing_invariants[-1][1]:
            raise SystemExit(f"pairing replay mismatch for {task_id}")
        reset_paths.append(reset_path.relative_to(root).as_posix())
        pairing_paths.append(pairing_path.relative_to(root).as_posix())
        reset_hashes.append(sha256_file(reset_path))
        pairing_hashes.append(sha256_file(pairing_path))
    if len(set(reset_invariants)) != 1:
        raise SystemExit(f"same-seed reset is not repeatable for {task_id}")
    if len(set(pairing_invariants)) != 1:
        raise SystemExit(f"same-seed action pairing is not repeatable for {task_id}")
    inventory.append(
        {
            "exogenous_seed": seed_exogenous,
            "import_receipt_relpath": import_path.relative_to(root).as_posix(),
            "import_receipt_sha256": sha256_file(import_path),
            "initial_seed": seed_initial,
            "pairing_receipt_relpaths": pairing_paths,
            "pairing_receipt_sha256s": pairing_hashes,
            "reset_receipt_relpaths": reset_paths,
            "reset_receipt_sha256s": reset_hashes,
            "task_id": task_id,
        }
    )
document = {
    "action_spec": action_spec,
    "campaign_id": campaign_id,
    "closed_loop_episode_executed": False,
    "evidence_level": "all_tasks_reset_action_contract_qualification_v1",
    "master_seed": master_seed,
    "policy_loaded": False,
    "repetitions": repetitions,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed_derivation": "sha256_signed31_v1",
    "semantic_version": "1.0",
    "success_rate_claimed": False,
    "tasks": inventory,
}
payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise SystemExit(f"refusing to overwrite aggregate qualification: {output}")
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, output)
finally:
    temporary.unlink(missing_ok=True)
print(output)
PY

info "all frozen UniVTAC tasks passed $ACTION_SPEC qualification"
printf '%s\n' "$OUTPUT"

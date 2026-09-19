#!/usr/bin/env bash
# Start one task-specific or mixed8 N0-VTLA run under a persistent supervisor.
set -Eeuo pipefail

die() {
  printf 'N0-VTLA formal launcher error: %s\n' "$*" >&2
  exit 2
}

write_state() {
  local output_path="$1"
  local status_value="$2"
  local exit_code_value="$3"
  local finished_at_value="$4"
  local tmp="${output_path}.tmp.$$"
  printf '{\n  "exit_code": %s,\n  "finished_at_utc": "%s",\n  "protocol_id": "robotactile.n0_vtla.formal_supervisor.v1",\n  "status": "%s"\n}\n' \
    "$exit_code_value" "$finished_at_value" "$status_value" >"$tmp"
  mv "$tmp" "$output_path"
}

if [[ "${1:-}" == "__worker" ]]; then
  [[ $# -eq 5 ]] || die "worker expects SCOPE PROJECT_ROOT RUN_ID SUPERVISOR_DIR"
  readonly scope="$2"
  readonly project_root="$3"
  readonly run_id="$4"
  readonly supervisor_dir="$5"
  readonly runner="$project_root/tooling/hpu_training/docker_runtime.sh"
  readonly num_steps="$([[ "$scope" == "mixed8" ]] && printf 160000 || printf 20000)"
  printf '%s\n' "$BASHPID" >"$supervisor_dir/worker.pid"
  set +e
  bash "$runner" formal "$scope" "$project_root" 8 "$num_steps" "$run_id"
  rc=$?
  set -e
  readonly rc
  readonly finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$rc" -eq 0 ]]; then
    write_state "$supervisor_dir/state.json" completed "$rc" "$finished_at"
  else
    write_state "$supervisor_dir/state.json" failed "$rc" "$finished_at"
  fi
  exit "$rc"
fi

[[ $# -eq 3 ]] || die "expected SCOPE PROJECT_ROOT RUN_ID"
readonly scope="$1"
readonly project_root="$2"
readonly run_id="$3"
case "$scope" in
  grasp_classify|insert_HDMI|insert_hole|insert_tube|lift_bottle|lift_can|pull_out_key|put_bottle_in_shelf|mixed8) ;;
  *) die "invalid training scope" ;;
esac
[[ "$project_root" == /mnt/data/* ]] || die "project root must be under /mnt/data"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$ ]] || die "invalid run ID"
readonly runner="$project_root/tooling/hpu_training/docker_runtime.sh"
[[ -f "$runner" ]] || die "container runtime wrapper missing"

readonly supervisor_parent="$project_root/logs/supervisor"
readonly supervisor_dir="$supervisor_parent/$run_id"
mkdir -p "$supervisor_parent"
mkdir "$supervisor_dir" || die "refusing to reuse supervisor directory: $supervisor_dir"
readonly started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_state "$supervisor_dir/state.json" running -1 ""

nohup setsid "$0" __worker "$scope" "$project_root" "$run_id" "$supervisor_dir" \
  >"$supervisor_dir/launcher.log" 2>&1 </dev/null &
readonly launcher_pid="$!"
printf '%s\n' "$launcher_pid" >"$supervisor_dir/launcher.pid"
readonly num_steps="$([[ "$scope" == "mixed8" ]] && printf 160000 || printf 20000)"
printf '{\n  "global_batch_size": 64,\n  "launcher_pid": %s,\n  "node": "%s",\n  "num_train_steps": %s,\n  "protocol_id": "robotactile.n0_vtla.formal_launch.v2",\n  "run_id": "%s",\n  "started_at_utc": "%s",\n  "task_balanced": %s,\n  "training_scope": "%s"\n}\n' \
  "$launcher_pid" "$(hostname)" "$num_steps" "$run_id" "$started_at" \
  "$([[ "$scope" == "mixed8" ]] && printf true || printf false)" "$scope" >"$supervisor_dir/launch.json"
kill -0 "$launcher_pid" 2>/dev/null || die "supervisor exited during launch"
printf 'launched scope=%s run_id=%s pid=%s supervisor=%s\n' \
  "$scope" "$run_id" "$launcher_pid" "$supervisor_dir"

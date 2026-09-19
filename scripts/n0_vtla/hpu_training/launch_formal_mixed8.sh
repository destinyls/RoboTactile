#!/usr/bin/env bash
# User-facing wrapper for one exposure-matched eight-task checkpoint.
set -Eeuo pipefail

[[ $# -eq 2 ]] || {
  printf 'usage: %s PROJECT_ROOT RUN_ID\n' "$0" >&2
  exit 2
}

readonly project_root="$1"
readonly run_id="$2"
exec bash "$project_root/tooling/hpu_training/launch_formal_task.sh" \
  mixed8 "$project_root" "$run_id"

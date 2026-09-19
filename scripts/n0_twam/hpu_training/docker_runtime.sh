#!/usr/bin/env bash
# Host-side, fail-closed wrapper for the proven Lingchu HCU container runtime.
set -Eeuo pipefail

readonly IMAGE_REF="docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
readonly IMAGE_ID="sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
readonly PLUGIN_DIR="/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
readonly PLUGIN_FILE="$PLUGIN_DIR/librccl-net-shca.so.0.0.0"
readonly PLUGIN_SHA="20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
readonly RUNTIME_PYTHON="/mnt/data/task/n0_twam_track32_franka_20260810/runtime/venv_py310_24ec50d/bin/python"

die() {
  printf 'HCU container runtime error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "expected one encoded payload"
command -v python3 >/dev/null 2>&1 || die "host python3 is required for JSON decoding"
command -v docker >/dev/null 2>&1 || die "docker is not installed on this node"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

mapfile -d '' -t fields < <(
  python3 - "$1" <<'PY'
import base64
import json
import sys

try:
    payload = json.loads(base64.b64decode(sys.argv[1], validate=True))
    values = (
        payload["container_name"],
        payload["container_entrypoint"],
        payload["entrypoint_kind"],
        payload["repo"],
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid runtime payload: {exc}") from exc
for value in values:
    sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")
PY
)
[[ ${#fields[@]} -eq 4 ]] || die "invalid runtime payload field count"

readonly container_name="${fields[0]}"
readonly container_entrypoint="${fields[1]}"
readonly entrypoint_kind="${fields[2]}"
readonly repo="${fields[3]}"

[[ "$container_name" =~ ^robotactile-n0-[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$ ]] || \
  die "invalid no-clobber container name"
[[ "$container_entrypoint" == /mnt/data/* ]] || \
  die "container entrypoint must be on shared /mnt/data"
[[ -f "$container_entrypoint" ]] || die "container entrypoint missing"
[[ "$entrypoint_kind" == "bash" || "$entrypoint_kind" == "python" ]] || \
  die "unsupported container entrypoint kind"
[[ "$repo" == /mnt/data/* && -d "$repo" ]] || \
  die "official repo must be on shared /mnt/data"

readonly actual_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")"
[[ "$actual_image_id" == "$IMAGE_ID" ]] || \
  die "container image ID mismatch: $actual_image_id"
readonly actual_plugin_sha="$(sha256sum "$PLUGIN_FILE" | awk '{print $1}')"
[[ "$actual_plugin_sha" == "$PLUGIN_SHA" ]] || \
  die "RCCL network plugin SHA256 mismatch: $actual_plugin_sha"

for device in \
  /dev/kfd /dev/mkfd /dev/dri /dev/infiniband/rdma_cm \
  /dev/infiniband/uverbs0 /dev/infiniband/uverbs1 \
  /dev/infiniband/uverbs2 /dev/infiniband/uverbs3; do
  [[ -e "$device" ]] || die "required HCU/RDMA device missing: $device"
done
for mount_source in /opt/hyhal /etc/hfm "$PLUGIN_DIR" /mnt/data; do
  [[ -e "$mount_source" ]] || die "required runtime mount missing: $mount_source"
done

if docker container inspect "$container_name" >/dev/null 2>&1; then
  die "container name already exists; refusing to touch it: $container_name"
fi

declare -a workload
if [[ "$entrypoint_kind" == "bash" ]]; then
  workload=(bash "$container_entrypoint" "$1")
else
  workload=("$RUNTIME_PYTHON" "$container_entrypoint" --payload "$1")
fi

container_id=""
if ! container_id="$(
  docker create \
    --name "$container_name" \
    --entrypoint "" \
    --network host \
    --ipc host \
    --device /dev/kfd \
    --device /dev/mkfd \
    --device /dev/dri \
    --device /dev/infiniband/rdma_cm \
    --device /dev/infiniband/uverbs0 \
    --device /dev/infiniband/uverbs1 \
    --device /dev/infiniband/uverbs2 \
    --device /dev/infiniband/uverbs3 \
    --security-opt label=disable \
    --ulimit nofile=1048576:1048576 \
    --ulimit stack=-1:-1 \
    --ulimit memlock=-1:-1 \
    --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
    --mount type=bind,src=/etc/hfm,dst=/etc/hfm,readonly \
    --mount type=bind,src="$PLUGIN_DIR",dst=/opt/n0_twam/rccl_plugin,readonly \
    --mount type=bind,src=/mnt/data,dst=/mnt/data \
    --workdir "$repo" \
    "$IMAGE_REF" \
    "${workload[@]}"
)"; then
  die "docker create failed"
fi
readonly container_id
[[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || die "docker returned an invalid container ID"

cleanup_own_container() {
  local current_id=""
  current_id="$(docker container inspect --format '{{.Id}}' "$container_name" 2>/dev/null || true)"
  if [[ "$current_id" == "$container_id" ]]; then
    docker stop --time 15 "$container_id" >/dev/null 2>&1 || true
    docker rm "$container_id" >/dev/null 2>&1 || true
  fi
}
trap cleanup_own_container EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
docker start --attach "$container_id"
readonly workload_status=$?
set -e
exit "$workload_status"

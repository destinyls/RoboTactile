#!/usr/bin/env bash
# Host-side transient container runtime for one N0-VTLA 4-node rank.
set -Eeuo pipefail

readonly IMAGE_REF="docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
readonly IMAGE_ID="sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
readonly PLUGIN_DIR="/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
readonly PLUGIN_FILE="$PLUGIN_DIR/librccl-net-shca.so.0.0.0"
readonly PLUGIN_SHA256="20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"

die() {
  printf 'N0-VTLA 4-node container runtime error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "expected one base64 JSON payload"
command -v python3 >/dev/null 2>&1 || die "host python3 is required"
command -v docker >/dev/null 2>&1 || die "docker is required"
mapfile -d '' -t fields < <(
  python3 - "$1" <<'PY'
import base64
import json
import sys

try:
    value = json.loads(base64.b64decode(sys.argv[1], validate=True))
    fields = (
        value["container_name"], value["distributed_entrypoint"]["path"],
        value["distributed_entrypoint"]["sha256"], value["project_root"],
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid container payload: {exc}") from exc
for field in fields:
    sys.stdout.buffer.write(str(field).encode() + b"\0")
PY
)
[[ ${#fields[@]} -eq 4 ]] || die "invalid payload field count"
readonly container_name="${fields[0]}" entrypoint="${fields[1]}"
readonly entrypoint_sha="${fields[2]}" project_root="${fields[3]}"
[[ "$container_name" =~ ^robotactile-n0-vtla-4node-[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$ ]] || \
  die "invalid container name"
[[ "$project_root" == /mnt/data/* ]] || die "project root must be below /mnt/data"
[[ "$entrypoint" == /mnt/data/* && -f "$entrypoint" && ! -L "$entrypoint" ]] || \
  die "distributed entrypoint is missing or unsafe"
[[ "$(sha256sum "$entrypoint" | awk '{print $1}')" == "$entrypoint_sha" ]] || \
  die "distributed entrypoint SHA256 mismatch"
[[ -d "$project_root/source/N0-VTLA/.git" ]] || die "official source checkout missing"
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")" == "$IMAGE_ID" ]] || \
  die "vendor image identity mismatch"
[[ "$(sha256sum "$PLUGIN_FILE" | awk '{print $1}')" == "$PLUGIN_SHA256" ]] || \
  die "RCCL network plugin SHA256 mismatch"
for device in \
  /dev/kfd /dev/mkfd /dev/dri /dev/infiniband/rdma_cm \
  /dev/infiniband/uverbs0 /dev/infiniband/uverbs1 \
  /dev/infiniband/uverbs2 /dev/infiniband/uverbs3; do
  [[ -e "$device" ]] || die "required HCU device missing: $device"
done
for mount_source in /opt/hyhal /etc/hfm "$PLUGIN_DIR" /mnt/data; do
  [[ -e "$mount_source" ]] || die "required runtime mount missing: $mount_source"
done
if docker container inspect "$container_name" >/dev/null 2>&1; then
  die "container name already exists: $container_name"
fi

container_id="$(
  docker create \
    --name "$container_name" --entrypoint "" --network host --ipc host \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    --device /dev/infiniband/rdma_cm \
    --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 \
    --device /dev/infiniband/uverbs2 --device /dev/infiniband/uverbs3 \
    --security-opt label=disable \
    --ulimit nofile=1048576:1048576 --ulimit stack=-1:-1 --ulimit memlock=-1:-1 \
    --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
    --mount type=bind,src=/etc/hfm,dst=/etc/hfm,readonly \
    --mount type=bind,src="$PLUGIN_DIR",dst=/opt/n0_vtla/rccl_plugin,readonly \
    --mount type=bind,src=/mnt/data,dst=/mnt/data \
    --workdir "$project_root/source/N0-VTLA" \
    "$IMAGE_REF" bash "$entrypoint" "$1"
)"
readonly container_id
[[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || die "docker returned an invalid ID"
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

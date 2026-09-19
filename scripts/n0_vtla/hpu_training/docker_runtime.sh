#!/usr/bin/env bash
# Host-side isolated container wrapper for one N0-VTLA training scope.
set -Eeuo pipefail

readonly IMAGE_REF="docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
readonly IMAGE_ID="sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
readonly PLUGIN_DIR="/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
readonly PLUGIN_FILE="$PLUGIN_DIR/librccl-net-shca.so.0.0.0"
readonly PLUGIN_SHA256="20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"

die() {
  printf 'N0-VTLA container runtime error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 6 ]] || die "expected MODE SCOPE PROJECT_ROOT NPROC NUM_STEPS RUN_ID"
readonly mode="$1"
readonly scope="$2"
readonly project_root="$3"
readonly nproc="$4"
readonly num_steps="$5"
readonly run_id="$6"
[[ "$project_root" == /mnt/data/* ]] || die "project root must be under /mnt/data"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$ ]] || die "invalid run ID"
readonly entrypoint="$project_root/tooling/hpu_training/rank_entrypoint.sh"
readonly repo="$project_root/source/N0-VTLA"
readonly container_name="robotactile-n0-vtla-$run_id"
[[ -f "$entrypoint" ]] || die "container entrypoint missing"
[[ -d "$repo/.git" ]] || die "official source checkout missing"
command -v docker >/dev/null 2>&1 || die "docker is unavailable"
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")" == "$IMAGE_ID" ]] || die "vendor image identity mismatch"
[[ "$(sha256sum "$PLUGIN_FILE" | awk '{print $1}')" == "$PLUGIN_SHA256" ]] || \
  die "RCCL network plugin SHA256 mismatch"
for device in \
  /dev/kfd /dev/mkfd /dev/dri /dev/infiniband/rdma_cm \
  /dev/infiniband/uverbs0 /dev/infiniband/uverbs1 \
  /dev/infiniband/uverbs2 /dev/infiniband/uverbs3; do
  [[ -e "$device" ]] || die "required HCU device missing: $device"
done
if docker container inspect "$container_name" >/dev/null 2>&1; then
  die "container name already exists: $container_name"
fi

container_id="$(
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
    --mount type=bind,src="$PLUGIN_DIR",dst=/opt/n0_vtla/rccl_plugin,readonly \
    --mount type=bind,src=/mnt/data,dst=/mnt/data \
    --workdir "$repo" \
    "$IMAGE_REF" \
    bash "$entrypoint" "$mode" "$scope" "$project_root" "$nproc" "$num_steps" "$run_id"
)"
readonly container_id
cleanup() {
  local current_id=""
  current_id="$(docker container inspect --format '{{.Id}}' "$container_name" 2>/dev/null || true)"
  if [[ "$current_id" == "$container_id" ]]; then
    docker stop --time 15 "$container_id" >/dev/null 2>&1 || true
    docker rm "$container_id" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
docker start --attach "$container_id"

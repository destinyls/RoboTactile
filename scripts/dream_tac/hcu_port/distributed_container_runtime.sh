#!/usr/bin/env bash
# Host-side, no-clobber runtime for one Dream-Tac HCU node rank.
set -Eeuo pipefail

readonly IMAGE_REF="docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
readonly IMAGE_ID="sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
readonly PLUGIN_DIR="/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
readonly PLUGIN_FILE="$PLUGIN_DIR/librccl-net-shca.so.0.0.0"
readonly PLUGIN_SHA="20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"

die() { printf 'Dream-Tac HCU runtime error: %s\n' "$*" >&2; exit 2; }

[[ $# -eq 1 ]] || die "expected one encoded payload"
command -v python3 >/dev/null 2>&1 || die "host python3 is required"
command -v docker >/dev/null 2>&1 || die "docker is required"

mapfile -d '' -t fields < <(
  python3 - "$1" <<'PY'
import base64, json, sys
try:
    value = json.loads(base64.b64decode(sys.argv[1], validate=True))
    fields = (
        value["container_name"], value["container_entrypoint"],
        value["dream_tac_root"], value["dream_tac_host_root"],
        value["runtime_host_root"], value["output_root"],
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid runtime payload: {exc}") from exc
for field in fields:
    sys.stdout.buffer.write(str(field).encode() + b"\0")
PY
)
[[ ${#fields[@]} -eq 6 ]] || die "invalid payload field count"
readonly container_name="${fields[0]}"
readonly entrypoint="${fields[1]}"
readonly dream_tac_root="${fields[2]}"
readonly dream_tac_host_root="${fields[3]}"
readonly runtime_host_root="${fields[4]}"
readonly output_root="${fields[5]}"

[[ "$container_name" =~ ^robotactile-dream-tac-[a-z0-9][a-z0-9_.-]{2,180}$ ]] || \
  die "invalid container name"
for path in "$entrypoint" "$dream_tac_host_root" "$runtime_host_root" "$output_root"; do
  [[ "$path" == /mnt/data/* ]] || die "shared paths must be below /mnt/data"
done
[[ "$dream_tac_root" == /workspace/* ]] || die "container Dream-Tac root must be below /workspace"
[[ -f "$entrypoint" ]] || die "container entrypoint is missing"
[[ -d "$dream_tac_host_root" ]] || die "host Dream-Tac checkout is missing"
[[ -d "$runtime_host_root" ]] || die "host runtime overlay is missing"
[[ -d "$output_root" ]] || die "output root is missing"

readonly actual_image_id="$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")"
[[ "$actual_image_id" == "$IMAGE_ID" ]] || die "container image ID mismatch"
readonly actual_plugin_sha="$(sha256sum "$PLUGIN_FILE" | awk '{print $1}')"
[[ "$actual_plugin_sha" == "$PLUGIN_SHA" ]] || die "RCCL plugin SHA256 mismatch"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  die "container name already exists; refusing to overwrite it"
fi

for device in \
  /dev/kfd /dev/mkfd /dev/dri /dev/infiniband/rdma_cm \
  /dev/infiniband/uverbs0 /dev/infiniband/uverbs1 \
  /dev/infiniband/uverbs2 /dev/infiniband/uverbs3; do
  [[ -e "$device" ]] || die "required HCU/RDMA device missing: $device"
done
for source in /opt/hyhal /etc/hfm "$PLUGIN_DIR" /mnt/data; do
  [[ -e "$source" ]] || die "required runtime mount missing: $source"
done

container_id=""
if ! container_id="$(
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
    --mount type=bind,src="$PLUGIN_DIR",dst=/opt/robotactile/rccl_plugin,readonly \
    --mount type=bind,src="$dream_tac_host_root",dst="$dream_tac_root",readonly \
    --mount type=bind,src="$runtime_host_root",dst=/probe,readonly \
    --mount type=bind,src=/mnt/data,dst=/mnt/data \
    --workdir "$dream_tac_root" \
    "$IMAGE_REF" bash "$entrypoint" "$1"
)"; then
  die "docker create failed"
fi
readonly container_id
[[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || die "docker returned invalid ID"

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

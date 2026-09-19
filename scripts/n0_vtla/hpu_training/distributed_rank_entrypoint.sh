#!/usr/bin/env bash
# Container-side adapter from one source-bound rank payload to the pinned entrypoint.
set -Eeuo pipefail

die() {
  printf 'N0-VTLA 4-node rank error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "expected one base64 JSON payload"
mapfile -d '' -t fields < <(
  python3 - "$1" <<'PY'
import base64
import json
import sys

try:
    value = json.loads(base64.b64decode(sys.argv[1], validate=True))
    fields = (
        value["mode"], value["run_id"], value["project_root"],
        value["node_rank"], value["node_addr"], value["nnodes"], value["nproc_per_node"],
        value["world_size"], value["master_addr"], value["master_port"],
        value["fresh_base"], value["rank_entrypoint"]["path"],
        value["rank_entrypoint"]["sha256"],
        value["collective_probe"]["path"],
        value["collective_probe"]["sha256"],
        value["launch_request"]["path"],
        value["launch_request"]["sha256"],
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid distributed rank payload: {exc}") from exc
for field in fields:
    sys.stdout.buffer.write(str(field).encode() + b"\0")
PY
)
[[ ${#fields[@]} -eq 17 ]] || die "invalid payload field count"
readonly mode="${fields[0]}" run_id="${fields[1]}" project_root="${fields[2]}"
readonly node_rank="${fields[3]}" node_addr="${fields[4]}" nnodes="${fields[5]}" nproc="${fields[6]}"
readonly world_size="${fields[7]}" master_addr="${fields[8]}" master_port="${fields[9]}"
readonly fresh_base="${fields[10]}" rank_entrypoint="${fields[11]}"
readonly rank_entrypoint_sha="${fields[12]}" collective_probe="${fields[13]}"
readonly collective_probe_sha="${fields[14]}"
readonly launch_request="${fields[15]}" launch_request_sha="${fields[16]}"

[[ "$mode" == preflight || "$mode" == train ]] || die "invalid mode"
[[ "$nnodes" == 4 && "$nproc" == 8 && "$world_size" == 32 ]] || \
  die "topology must be exactly 4 nodes x 8 ranks"
[[ "$node_rank" =~ ^[0-3]$ ]] || die "node rank must be in [0, 3]"
[[ "$master_port" =~ ^[0-9]+$ ]] && (( master_port >= 1024 && master_port <= 65535 )) || \
  die "master port is invalid"
[[ "$fresh_base" == True ]] || die "only fresh-base training is allowed"
[[ "$project_root" == /mnt/data/* ]] || die "project root must be below /mnt/data"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$ ]] || die "invalid run ID"
for source_path in "$rank_entrypoint" "$collective_probe" "$launch_request"; do
  [[ "$source_path" == /mnt/data/* && -f "$source_path" && ! -L "$source_path" ]] || \
    die "source-bound path is missing or unsafe: $source_path"
done
[[ "$(sha256sum "$rank_entrypoint" | awk '{print $1}')" == "$rank_entrypoint_sha" ]] || \
  die "rank entrypoint SHA256 mismatch"
[[ "$(sha256sum "$collective_probe" | awk '{print $1}')" == "$collective_probe_sha" ]] || \
  die "collective probe SHA256 mismatch"
[[ "$(sha256sum "$launch_request" | awk '{print $1}')" == "$launch_request_sha" ]] || \
  die "launch request SHA256 mismatch"
python3 - "$launch_request" "$project_root" "$run_id" "$node_rank" "$node_addr" "$master_addr" "$master_port" <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

request_path = pathlib.Path(sys.argv[1])
project_root = pathlib.Path(sys.argv[2])
run_id = sys.argv[3]
node_rank = int(sys.argv[4])
node_addr = sys.argv[5]
master_addr = sys.argv[6]
master_port = int(sys.argv[7])
request = json.loads(request_path.read_text())
expected = {
    "protocol_id": "robotactile.n0_vtla.formal_mixed8_4node.v1",
    "run_id": run_id,
    "project_root": str(project_root),
    "scope": "mixed8",
    "dataset_split": "train759",
    "validation_data_used": False,
    "fresh_base": True,
    "resume_checkpoint": None,
    "task_balanced": True,
    "nnodes": 4,
    "nproc_per_node": 8,
    "world_size": 32,
    "num_train_steps": 160000,
    "global_batch_size": 64,
    "micro_batch_per_rank": 1,
    "gradient_accumulation_steps": 2,
    "save_interval": 2000,
    "master_addr": master_addr,
    "master_port": master_port,
}
for key, value in expected.items():
    if request.get(key) != value:
        raise SystemExit(f"launch request contract mismatch: {key}")
nodes = request.get("nodes")
if (
    not isinstance(nodes, list)
    or len(nodes) != 4
    or len(set(nodes)) != 4
    or nodes[0] != master_addr
    or nodes[node_rank] != node_addr
):
    raise SystemExit("launch request node-rank map mismatch")
expected_request_path = project_root / "logs/supervisor" / run_id / "request.json"
if request_path != expected_request_path:
    raise SystemExit("launch request path mismatch")
tooling = project_root / "tooling/hpu_training"
paths = {
    "distributed_container_runtime": tooling / "distributed_container_runtime.sh",
    "distributed_rank_entrypoint": tooling / "distributed_rank_entrypoint.sh",
    "distributed_collective_probe": tooling / "distributed_collective_probe.py",
    "gradient_accumulation_patch": tooling / "gradient_accumulation_patch.py",
    "hcu_train_entry": tooling / "hcu_train_entry.py",
    "launch_formal_mixed8_4node": tooling / "launch_formal_mixed8_4node.py",
    "multitask_sampler": tooling / "multitask_sampler.py",
    "rank_entrypoint": tooling / "rank_entrypoint.sh",
    "runtime_patches": tooling / "runtime_patches.py",
    "runtime_shims_sitecustomize": tooling / "runtime_shims/sitecustomize.py",
}
bindings = request.get("source_bindings")
if not isinstance(bindings, dict) or set(bindings) != set(paths):
    raise SystemExit("launch request source bindings are incomplete")
for name, path in paths.items():
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"source-bound training file is unsafe: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if bindings.get(name) != digest:
        raise SystemExit(f"source-bound training SHA256 mismatch: {name}")
base = request.get("base_checkpoint")
expected_base = project_root / "artifacts/base/n0-vtla-base-ec12548"
if not isinstance(base, dict) or base.get("path") != str(expected_base):
    raise SystemExit("fresh-base path contract mismatch")
PY

export ROBOTACTILE_N0_VTLA_DISTRIBUTED_MODE=four_node
export ROBOTACTILE_N0_VTLA_NNODES="$nnodes"
export ROBOTACTILE_N0_VTLA_NODE_RANK="$node_rank"
export ROBOTACTILE_N0_VTLA_MASTER_ADDR="$master_addr"
export ROBOTACTILE_N0_VTLA_MASTER_PORT="$master_port"
export ROBOTACTILE_N0_VTLA_FRESH_BASE=1

if [[ "$mode" == preflight ]]; then
  bash "$rank_entrypoint" preflight mixed8 "$project_root" 8 160000 "$run_id"
  [[ -f /opt/hyhal/env.sh ]] || die "HCU environment is not mounted"
  set +u
  # shellcheck disable=SC1091
  source /opt/hyhal/env.sh
  set -u
  export HIP_VISIBLE_DEVICES="0,1,5,4,2,3,7,6"
  export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
  export NCCL_NET_PLUGIN=shca NCCL_PLUGIN_P2P=ucx NCCL_IB_DISABLE=0
  export NCCL_IB_HCA="shca_0:1,shca_1:1,shca_2:1,shca_3:1"
  export NCCL_NET_GDR_LEVEL=4 NCCL_NET_GDR_READ=1
  export NCCL_SOCKET_IFNAME="ib0,ib3" OMPI_MCA_btl_tcp_if_include="ib0,ib3"
  export GLOO_SOCKET_IFNAME=bond1
  export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2" RCCL_PXN_GPU_BALANCE=1
  export NCCL_TOPO_FILE=/opt/shca_device_driver/topo_lib/built-in-508-topo-input-tj-default.xml
  export NCCL_DYNAMIC_QP_LB_DISABLE=1 NCCL_HYGON_GRAPH_FIX=0
  export NCCL_IB_SPLIT_DATA_ON_QPS=0 NCCL_PXN_DISABLE=0 NCCL_TC_LB_DISABLE=1
  export NCCL_NET_AFFINITY_LEVEL=0 NCCL_TIMEOUT=3600
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=0 TORCH_NCCL_BLOCKING_WAIT=1
  export TORCH_NCCL_ENABLE_MONITORING=0 HSA_FORCE_FINE_GRAIN_PCIE=1
  export LD_LIBRARY_PATH="/opt/n0_vtla/rccl_plugin:/opt/shca_device_driver/ucx-1.18.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  exec python -m torch.distributed.run \
    --nnodes="$nnodes" --node_rank="$node_rank" --nproc-per-node="$nproc" \
    --master_addr="$master_addr" --master_port="$master_port" \
    "$collective_probe"
fi

exec bash "$rank_entrypoint" formal mixed8 "$project_root" 8 160000 "$run_id"

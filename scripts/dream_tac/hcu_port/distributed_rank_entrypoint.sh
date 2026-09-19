#!/usr/bin/env bash
# Container-side source-bound torchrun entrypoint.
set -Eeuo pipefail

die() { printf 'Dream-Tac rank error: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 1 ]] || die "expected one encoded payload"

mapfile -d '' -t fields < <(
  python3 - "$1" <<'PY'
import base64, json, sys
try:
    value = json.loads(base64.b64decode(sys.argv[1], validate=True))
    fixed = (
        value["mode"], value["launch_mode"], value["node_rank"], value["nnodes"],
        value["nproc_per_node"], value["world_size"], value["fsdp_shard_size"],
        value["visible_devices"], value["master_addr"], value["master_port"],
        value["network_interface"], value["max_iter"], value["python_executable"],
        value["hcu_environment_script"]["path"],
        value["hcu_environment_script"]["sha256"], value["pythonpath"],
        value["dream_tac_root"], value["output_root"], value["base_dcp_root"],
        value["collective_probe"]["path"], value["collective_probe"]["sha256"],
        value["franka_dataset"]["path"], value["franka_dataset"]["sha256"],
        value["fsdp_sources"]["fsdp_helper"]["path"],
        value["fsdp_sources"]["fsdp_helper"]["sha256"],
        value["fsdp_sources"]["dtensor_helper"]["path"],
        value["fsdp_sources"]["dtensor_helper"]["sha256"],
        value["fsdp_sources"]["mesh_mode"],
        len(value["overrides"]), *value["overrides"],
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid rank payload: {exc}") from exc
for field in fixed:
    sys.stdout.buffer.write(str(field).encode() + b"\0")
PY
)
[[ ${#fields[@]} -ge 29 ]] || die "invalid rank payload field count"
readonly mode="${fields[0]}" launch_mode="${fields[1]}" node_rank="${fields[2]}"
readonly nnodes="${fields[3]}" nproc="${fields[4]}" world_size="${fields[5]}"
readonly fsdp_shard_size="${fields[6]}" visible_devices="${fields[7]}"
readonly master_addr="${fields[8]}" master_port="${fields[9]}" interface="${fields[10]}"
readonly max_iter="${fields[11]}" python_executable="${fields[12]}"
readonly environment_script="${fields[13]}" environment_sha="${fields[14]}"
readonly pythonpath="${fields[15]}" dream_tac_root="${fields[16]}"
readonly output_root="${fields[17]}" base_dcp_root="${fields[18]}"
readonly collective_probe="${fields[19]}" collective_probe_sha="${fields[20]}"
readonly franka_dataset="${fields[21]}" franka_dataset_sha="${fields[22]}"
readonly fsdp_helper="${fields[23]}" fsdp_helper_sha="${fields[24]}"
readonly dtensor_helper="${fields[25]}" dtensor_helper_sha="${fields[26]}"
readonly fsdp_mesh_mode="${fields[27]}" override_count="${fields[28]}"
[[ "$mode" == preflight || "$mode" == train ]] || die "invalid mode"
[[ "$launch_mode" == distributed-smoke || "$launch_mode" == formal ]] || die "invalid launch mode"
[[ "$nnodes" == 2 && "$nproc" == 8 && "$world_size" == 16 ]] || die "topology must be 2x8"
[[ "$fsdp_shard_size" == 16 ]] || die "FSDP shard size must equal the global world size"
[[ "$node_rank" == 0 || "$node_rank" == 1 ]] || die "node rank must be 0 or 1"
[[ "$visible_devices" == "0,1,5,4,2,3,7,6" ]] || die "HCU device order mismatch"
[[ "$override_count" =~ ^[0-9]+$ ]] || die "override count is invalid"
[[ ${#fields[@]} -eq $((29 + override_count)) ]] || die "override count mismatch"
[[ -x "$python_executable" ]] || die "training Python is not executable"
[[ -f "$environment_script" && ! -L "$environment_script" ]] || die "HCU environment script is unsafe"
[[ "$(sha256sum "$environment_script" | awk '{print $1}')" == "$environment_sha" ]] || \
  die "HCU environment script SHA256 mismatch"
[[ -d "$dream_tac_root" && -d "$base_dcp_root/model" ]] || die "source/base DCP missing"
[[ "$collective_probe" == /mnt/data/* && -f "$collective_probe" && ! -L "$collective_probe" ]] || \
  die "collective probe path is unsafe"
[[ "$(sha256sum "$collective_probe" | awk '{print $1}')" == "$collective_probe_sha" ]] || \
  die "collective probe SHA256 mismatch"
[[ "$franka_dataset" == "$dream_tac_root/cosmos_policy/datasets/franka_dataset.py" ]] || \
  die "Franka dataset path mismatch"
[[ -f "$franka_dataset" && ! -L "$franka_dataset" ]] || die "Franka dataset source is unsafe"
[[ "$(sha256sum "$franka_dataset" | awk '{print $1}')" == "$franka_dataset_sha" ]] || \
  die "Franka dataset SHA256 mismatch"
[[ "$fsdp_mesh_mode" == global_1d_shard_v1 ]] || die "FSDP mesh mode mismatch"
[[ "$fsdp_helper" == "$dream_tac_root/cosmos_policy/_src/imaginaire/utils/fsdp_helper.py" ]] || \
  die "FSDP helper path mismatch"
[[ "$dtensor_helper" == "$dream_tac_root/cosmos_policy/_src/predict2/utils/dtensor_helper.py" ]] || \
  die "DTensor helper path mismatch"
for source_binding in "$fsdp_helper:$fsdp_helper_sha" "$dtensor_helper:$dtensor_helper_sha"; do
  source_path="${source_binding%%:*}"
  source_sha="${source_binding##*:}"
  [[ -f "$source_path" && ! -L "$source_path" ]] || die "distributed source is unsafe"
  [[ "$(sha256sum "$source_path" | awk '{print $1}')" == "$source_sha" ]] || \
    die "distributed source SHA256 mismatch"
done
# shellcheck disable=SC1090
source "$environment_script"

export COSMOS_TACTILE_SELF_ATTN_BACKEND=flashbias_sdpa
export CUDA_VISIBLE_DEVICES="$visible_devices" HIP_VISIBLE_DEVICES="$visible_devices"
export ROCR_VISIBLE_DEVICES="$visible_devices"
export DREAM_TAC_BASE_CHECKPOINT="$base_dcp_root"
export IMAGINAIRE_OUTPUT_ROOT="$output_root" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$pythonpath" ROBOTACTILE_HCU_TRAINING=1 TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=shca NCCL_PLUGIN_P2P=ucx NCCL_IB_DISABLE=0
export NCCL_IB_HCA="shca_0:1,shca_1:1,shca_2:1,shca_3:1"
export NCCL_NET_GDR_LEVEL=4 NCCL_NET_GDR_READ=1
export NCCL_SOCKET_IFNAME="ib0,ib3" OMPI_MCA_btl_tcp_if_include="ib0,ib3"
export GLOO_SOCKET_IFNAME="$interface"
export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2" RCCL_PXN_GPU_BALANCE=1
export NCCL_TOPO_FILE=/opt/shca_device_driver/topo_lib/built-in-508-topo-input-tj-default.xml
export NCCL_DYNAMIC_QP_LB_DISABLE=1 NCCL_HYGON_GRAPH_FIX=0
export NCCL_IB_SPLIT_DATA_ON_QPS=0 NCCL_PXN_DISABLE=0 NCCL_TC_LB_DISABLE=1
export NCCL_NET_AFFINITY_LEVEL=0 NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0 TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ENABLE_MONITORING=0 HSA_FORCE_FINE_GRAIN_PCIE=1
export LD_LIBRARY_PATH="/opt/robotactile/rccl_plugin:/opt/shca_device_driver/ucx-1.18.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$python_executable" - <<'PY'
import json, os, torch
assert torch.version.hip, "PyTorch must be an HCU/HIP build"
assert torch.cuda.is_available(), "HCU compatibility API unavailable"
assert torch.cuda.device_count() == 8, "each node must expose exactly 8 HCUs"
assert torch.distributed.is_nccl_available(), "RCCL backend unavailable"
print(json.dumps({"status": "passed", "hip": torch.version.hip,
                  "visible_devices": torch.cuda.device_count(),
                  "node_rank": int(os.environ.get("GROUP_RANK", "0"))}, sort_keys=True))
PY
if [[ "$mode" == preflight ]]; then
  exec "$python_executable" -m torch.distributed.run \
    --nnodes="$nnodes" --node_rank="$node_rank" --nproc_per_node="$nproc" \
    --master_addr="$master_addr" --master_port="$master_port" \
    "$collective_probe"
fi

declare -a overrides=("${fields[@]:29}")
cd "$dream_tac_root"
exec "$python_executable" -m torch.distributed.run \
  --nnodes="$nnodes" --node_rank="$node_rank" --nproc_per_node="$nproc" \
  --master_addr="$master_addr" --master_port="$master_port" \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
  "${overrides[@]}"

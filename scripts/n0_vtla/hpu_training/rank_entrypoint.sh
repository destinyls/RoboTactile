#!/usr/bin/env bash
# Container-side, source-bound task or mixed8 N0-VTLA entrypoint.
set -Eeuo pipefail

readonly OFFICIAL_COMMIT="03a0ce4d7091ca2354864796770715aa212601b7"
readonly BASE_SHA256="4aafb1da22a2671e637884d175ddc1569344ac5fd2dde8b53f3a34720533051c"
readonly BASE_SIZE_BYTES="8253873284"
readonly DINO_REVISION="f9e44c814b77203eaa57a6bdbbd535f21ede1415"

die() {
  printf 'N0-VTLA rank preflight error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 6 ]] || die "expected MODE SCOPE PROJECT_ROOT NPROC NUM_STEPS RUN_ID"
readonly mode="$1"
readonly scope="$2"
readonly project_root="$3"
readonly nproc="$4"
readonly num_steps="$5"
readonly run_id="$6"
readonly distributed_mode="${ROBOTACTILE_N0_VTLA_DISTRIBUTED_MODE:-single_node}"
case "$distributed_mode" in
  single_node)
    readonly distributed_nnodes=1
    readonly distributed_node_rank=0
    readonly distributed_master_addr="127.0.0.1"
    readonly distributed_master_port=0
    ;;
  four_node)
    readonly distributed_nnodes="${ROBOTACTILE_N0_VTLA_NNODES:-}"
    readonly distributed_node_rank="${ROBOTACTILE_N0_VTLA_NODE_RANK:-}"
    readonly distributed_master_addr="${ROBOTACTILE_N0_VTLA_MASTER_ADDR:-}"
    readonly distributed_master_port="${ROBOTACTILE_N0_VTLA_MASTER_PORT:-}"
    [[ "$distributed_nnodes" == "4" ]] || die "four-node training requires NNODES=4"
    [[ "$distributed_node_rank" =~ ^[0-3]$ ]] || die "invalid four-node rank"
    [[ "$distributed_master_addr" =~ ^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$ ]] || \
      die "invalid four-node master address"
    [[ "$distributed_master_port" =~ ^[0-9]+$ ]] || die "invalid four-node master port"
    (( distributed_master_port >= 1024 && distributed_master_port <= 65535 )) || \
      die "four-node master port is outside 1024..65535"
    [[ "$scope" == "mixed8" ]] || die "four-node training only supports mixed8"
    [[ "$mode" == "preflight" || "$mode" == "formal" ]] || \
      die "four-node mode only supports preflight or formal"
    ;;
  *) die "invalid distributed mode" ;;
esac
[[ "$mode" == "preflight" || "$mode" == "norm" || "$mode" == "smoke" || "$mode" == "formal" ]] || die "invalid mode"
case "$scope" in
  grasp_classify|insert_HDMI|insert_hole|insert_tube|lift_bottle|lift_can|pull_out_key|put_bottle_in_shelf|mixed8) ;;
  *) die "invalid training scope" ;;
esac
[[ "$project_root" == /mnt/data/* ]] || die "project root must be under /mnt/data"
[[ "$nproc" == "1" || "$nproc" == "8" ]] || die "NPROC must be 1 or 8"
[[ "$num_steps" =~ ^[1-9][0-9]*$ ]] || die "NUM_STEPS must be positive"
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$ ]] || die "invalid run ID"
if [[ "$mode" == "smoke" ]]; then
  [[ "$num_steps" == "1" ]] || die "smoke must run exactly one optimizer step"
elif [[ "$mode" == "formal" ]]; then
  [[ "$nproc" == "8" ]] || die "formal recipe requires 8 ranks per node"
  if [[ "$scope" == "mixed8" ]]; then
    [[ "$num_steps" == "160000" ]] || die "mixed8 claim recipe is fixed at 160000 steps"
  else
    [[ "$num_steps" == "20000" ]] || die "task recipe is fixed at 20000 steps"
  fi
fi

readonly repo="$project_root/source/N0-VTLA"
readonly tooling="$project_root/tooling/hpu_training"
readonly overlay="$project_root/runtime/n0_vtla_py310_overlay_v3"
readonly base="$project_root/artifacts/base/n0-vtla-base-ec12548"
readonly assets_base="$project_root/assets"
if [[ "$scope" == "mixed8" ]]; then
  readonly dataset="$project_root/data/univtac_qpos8_train759/mixed8"
  readonly dataset_receipt="$dataset/_robotactile_n0_vtla_qpos8_mixed8_receipt.json"
  readonly asset_id="univtac_mixed8_train759_qpos8"
else
  readonly dataset="$project_root/data/univtac_qpos8_train759/$scope"
  readonly dataset_receipt="$dataset/_robotactile_n0_vtla_qpos8_receipt.json"
  readonly asset_id="univtac_${scope}_train759_qpos8"
fi
readonly norm="$assets_base/sim_single_arm_tactile/$asset_id/norm_stats.json"
readonly norm_receipt="$assets_base/sim_single_arm_tactile/$asset_id/_robotactile_train_only_norm_receipt.json"
readonly hf_home="$project_root/artifacts/hf_cache"
readonly data_home="$project_root/artifacts/n0vtla_data"
readonly checkpoint_root="$project_root/runs/checkpoints"
readonly exp_name="${scope}__${run_id}"
readonly checkpoint_run="$checkpoint_root/sim_single_arm_tactile/$exp_name"
readonly log_root="$project_root/logs/$exp_name"

[[ -f /opt/hyhal/env.sh ]] || die "HCU environment is not mounted"
# shellcheck disable=SC1091
set +u
source /opt/hyhal/env.sh
set -u
[[ -d "$repo/.git" ]] || die "official source checkout missing"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$OFFICIAL_COMMIT" ]] || die "official source commit mismatch"
[[ -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" ]] || die "official source checkout is dirty"
[[ -d "$overlay" ]] || die "repo-local Python overlay missing"
[[ -f "$dataset_receipt" ]] || die "qpos8 train-only receipt missing"
[[ -f "$base/model.safetensors" ]] || die "base checkpoint missing"
[[ "$(stat -c '%s' "$base/model.safetensors")" == "$BASE_SIZE_BYTES" ]] || die "base checkpoint size mismatch"
if [[ "$mode" == "preflight" ]]; then
  [[ "$(sha256sum "$base/model.safetensors" | awk '{print $1}')" == "$BASE_SHA256" ]] || \
    die "base checkpoint SHA256 mismatch"
fi
[[ -f "$data_home/big_vision/paligemma_tokenizer.model" ]] || die "PaliGemma tokenizer cache missing"
[[ -f "$tooling/hcu_train_entry.py" && -f "$tooling/runtime_shims/sitecustomize.py" ]] || die "HCU tooling incomplete"
if [[ "$scope" == "mixed8" ]]; then
  [[ -f "$tooling/multitask_sampler.py" && -f "$tooling/compute_mixed_norm_stats.py" ]] || \
    die "mixed8 tooling incomplete"
fi
if [[ "$mode" != "norm" && "$mode" != "preflight" ]]; then
  [[ -f "$norm" ]] || die "task train-only normalization stats missing"
  [[ -f "$norm_receipt" ]] || die "task train-only normalization receipt missing"
fi

export ROBOTACTILE_ENABLE_N0_VTLA_HCU_SHIMS=1
export ROBOTACTILE_VENDOR_TRAIN_FAST_EXIT=1
export PYTHONPATH="$tooling/runtime_shims:$overlay:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$hf_home"
export TRANSFORMERS_CACHE="$hf_home/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export N0VTLA_DATA_HOME="$data_home"
export VTLA_DATASET_PATH="$dataset"
export VTLA_ASSET_ID="$asset_id"
export VTLA_DEFAULT_PROMPT="$(python -c "import json; print(json.loads(open('$dataset/meta/tasks.jsonl').readline())['task'])")"
export VTLA_PRETRAINED_CHECKPOINT="$base"
export VTLA_ATTN_IMPL=eager
export VTLA_PREFIX_CACHE=0
export VTLA_PREDICTOR_LR_SCALE=0.1
export VTLA_PHASE_A_STEPS=0
export VTLA_PREDICTOR_FREEZE=0
export VTLA_SAMPLE_LOSS_CLIP=0
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export ROBOTACTILE_N0_VTLA_DETACH_VL_CTX=1
export ROBOTACTILE_N0_VTLA_GRAD_ACCUM=1
if [[ "$scope" == "mixed8" ]]; then
  export ROBOTACTILE_N0_VTLA_TASK_BALANCED=1
  export ROBOTACTILE_N0_VTLA_MIXED_RECEIPT="$dataset_receipt"
fi
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_HIP_ALLOC_CONF"
export LD_LIBRARY_PATH="/opt/n0_vtla/rccl_plugin:/opt/shca_device_driver/ucx-1.18.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ucx
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="shca_0:1,shca_1:1,shca_2:1,shca_3:1"
export NCCL_NET_GDR_LEVEL=4
export NCCL_NET_GDR_READ=1
export NCCL_SOCKET_IFNAME="ib0,ib3"
export OMPI_MCA_btl_tcp_if_include="ib0,ib3"
export GLOO_SOCKET_IFNAME=bond1
export RCCL_NET_PLANE="shca_0,shca_3|shca_1,shca_2"
export RCCL_PXN_GPU_BALANCE=1
export NCCL_TOPO_FILE=/opt/shca_device_driver/topo_lib/built-in-508-topo-input-tj-default.xml
export NCCL_DYNAMIC_QP_LB_DISABLE=1
export NCCL_HYGON_GRAPH_FIX=0
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_PXN_DISABLE=0
export NCCL_TC_LB_DISABLE=1
export NCCL_NET_AFFINITY_LEVEL=0
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ENABLE_MONITORING=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
if [[ "$nproc" == "1" ]]; then
  export HIP_VISIBLE_DEVICES=0
  export CUDA_VISIBLE_DEVICES=0
else
  export HIP_VISIBLE_DEVICES="0,1,5,4,2,3,7,6"
  export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
fi

python - "$dataset" "$scope" "$DINO_REVISION" "$nproc" "$mode" "$norm" "$norm_receipt" "$dataset_receipt" <<'PY'
import hashlib
import inspect
import json
import os
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.siglip import check

from n0vtla.shared.normalize import NormStats
from n0vtla.training import config as config_module

dataset = pathlib.Path(sys.argv[1])
scope = sys.argv[2]
revision = sys.argv[3]
required_devices = int(sys.argv[4])
mode = sys.argv[5]
norm = pathlib.Path(sys.argv[6])
norm_receipt_path = pathlib.Path(sys.argv[7])
dataset_receipt_path = pathlib.Path(sys.argv[8])
receipt = json.loads(dataset_receipt_path.read_text())
tasks = [
    "grasp_classify", "insert_HDMI", "insert_hole", "insert_tube",
    "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf",
]
if receipt.get("source_split") != "train" or receipt.get("validation_data_used") is not False:
    raise SystemExit("qpos8 receipt does not prove train-only data")
if scope == "mixed8":
    if (
        receipt.get("protocol_id") != "robotactile.n0_vtla.univtac_qpos8_mixed8_train_only.v1"
        or receipt.get("tasks") != tasks
        or receipt.get("episode_count") != 759
        or receipt.get("frame_count") != 144484
    ):
        raise SystemExit("mixed8 dataset receipt contract mismatch")
elif receipt.get("task") != scope:
    raise SystemExit("task dataset receipt contract mismatch")
for key in ("observation.images.tactile_a", "observation.images.tactile_b"):
    count = len(tuple((dataset / "videos").rglob(f"{key}/episode_*.mp4")))
    if count != receipt["episode_count"]:
        raise SystemExit(f"missing tactile videos for {key}: {count}")
hf_root = pathlib.Path(os.environ["HF_HOME"]) / "hub" / "models--facebook--dinov2-base"
if (hf_root / "refs" / "main").read_text().strip() != revision:
    raise SystemExit("DINOv2 cache revision mismatch")
snapshot = hf_root / "snapshots" / revision
for filename in ("config.json", "model.safetensors"):
    if not (snapshot / filename).is_file():
        raise SystemExit(f"DINOv2 cache artifact missing: {filename}")
AutoConfig.from_pretrained("facebook/dinov2-base", revision=revision, local_files_only=True)
if not torch.cuda.is_available():
    raise SystemExit("vendor HCU is not exposed through torch.cuda")
if torch.cuda.device_count() < required_devices:
    raise SystemExit(f"only {torch.cuda.device_count()} HCU devices are visible; required={required_devices}")
config = config_module.get_config("sim_single_arm_tactile")
if config.data.repo_id != str(dataset):
    raise SystemExit("resolved training config does not use the certified task view")
NormStats(mean=np.zeros(8), std=np.ones(8))
if mode not in ("preflight", "norm"):
    norm_receipt = json.loads(norm_receipt_path.read_text())
    if (
        norm_receipt.get("source_split") != "train"
        or norm_receipt.get("validation_data_used") is not False
        or norm_receipt.get("norm_sha256") != hashlib.sha256(norm.read_bytes()).hexdigest()
    ):
        raise SystemExit("normalization receipt validation failed")
    if scope == "mixed8":
        if norm_receipt.get("training_scope") != "mixed8" or norm_receipt.get("tasks") != tasks:
            raise SystemExit("mixed8 normalization receipt validation failed")
    elif norm_receipt.get("task") != scope:
        raise SystemExit("task normalization receipt validation failed")
if "cond" not in inspect.signature(GemmaRMSNorm.forward).parameters:
    raise SystemExit("transformers replacement is inactive")
if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("transformers replacement self-check failed")
print(f"preflight OK scope={scope} torch={torch.__version__} devices={torch.cuda.device_count()}")
PY

if [[ "$mode" == "preflight" ]]; then
  exit 0
fi
if [[ "$mode" == "norm" ]]; then
  [[ ! -e "$norm" ]] || die "refusing to overwrite existing normalization stats"
  [[ ! -e "$norm_receipt" ]] || die "refusing to overwrite normalization receipt"
  cd "$project_root"
  if [[ "$scope" == "mixed8" ]]; then
    python "$tooling/compute_mixed_norm_stats.py"
  else
    python "$repo/scripts/compute_norm_stats.py" --config-name sim_single_arm_tactile
  fi
  [[ -f "$norm" ]] || die "normalization command produced no artifact"
  python - "$norm" "$norm_receipt" "$dataset_receipt" "$scope" "$OFFICIAL_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

norm = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
dataset_receipt_path = pathlib.Path(sys.argv[3])
scope = sys.argv[4]
official_commit = sys.argv[5]
dataset_receipt = json.loads(dataset_receipt_path.read_text())
if dataset_receipt.get("validation_data_used") is not False or dataset_receipt.get("source_split") != "train":
    raise SystemExit("dataset receipt does not prove train-only normalization input")
payload = {
    "config": "sim_single_arm_tactile",
    "dataset_episode_count": dataset_receipt["episode_count"],
    "dataset_frame_count": dataset_receipt["frame_count"],
    "dataset_receipt_sha256": hashlib.sha256(dataset_receipt_path.read_bytes()).hexdigest(),
    "n0_vtla_commit": official_commit,
    "norm_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
    "source_split": "train",
    "validation_data_used": False,
}
if scope == "mixed8":
    payload.update({
        "protocol_id": "robotactile.n0_vtla.mixed8_train_only_norm.v1",
        "training_scope": "mixed8",
        "tasks": dataset_receipt["tasks"],
        "task_frame_counts": dataset_receipt["task_frame_counts"],
    })
else:
    payload.update({
        "protocol_id": "robotactile.n0_vtla.train_only_norm.v1",
        "task": scope,
    })
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  exit 0
fi

if [[ "$distributed_mode" == "four_node" ]]; then
  readonly train_log="$log_root/train-node-$(printf '%02d' "$distributed_node_rank").log"
  if [[ "$distributed_node_rank" == "0" ]]; then
    [[ ! -e "$log_root" ]] || die "refusing to reuse log directory: $log_root"
    [[ ! -e "$checkpoint_run" ]] || die "refusing to reuse checkpoint directory: $checkpoint_run"
    mkdir "$log_root"
  else
    for _wait_index in $(seq 1 300); do
      [[ -d "$log_root" ]] && break
      sleep 1
    done
    [[ -d "$log_root" ]] || die "rank-0 did not publish the shared log directory"
  fi
  [[ ! -e "$train_log" ]] || die "refusing to reuse node log: $train_log"
else
  [[ ! -e "$log_root" ]] || die "refusing to reuse log directory: $log_root"
  [[ ! -e "$checkpoint_run" ]] || die "refusing to reuse checkpoint directory: $checkpoint_run"
  mkdir "$log_root"
  readonly train_log="$log_root/train.log"
fi
readonly batch_size="$([[ "$nproc" == "1" ]] && printf 1 || printf 64)"
readonly num_workers="$([[ "$mode" == "formal" ]] && printf 8 || printf 2)"
declare -a torchrun_topology
if [[ "$distributed_mode" == "four_node" ]]; then
  torchrun_topology=(
    --nnodes="$distributed_nnodes"
    --node-rank="$distributed_node_rank"
    --nproc-per-node="$nproc"
    --master-addr="$distributed_master_addr"
    --master-port="$distributed_master_port"
    --max-restarts=0
  )
else
  torchrun_topology=(--standalone --nnodes=1 --nproc-per-node="$nproc")
fi
python -m torch.distributed.run \
  "${torchrun_topology[@]}" \
  "$tooling/hcu_train_entry.py" \
  sim_single_arm_tactile \
  --exp-name="$exp_name" \
  --checkpoint-base-dir="$checkpoint_root" \
  --assets-base-dir="$assets_base" \
  --batch-size="$batch_size" \
  --num-workers="$num_workers" \
  --num-train-steps="$num_steps" \
  --log-interval=1 \
  --save-interval=2000 \
  2>&1 | tee "$train_log"

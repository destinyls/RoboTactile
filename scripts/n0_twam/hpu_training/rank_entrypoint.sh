#!/usr/bin/env bash
# Container-side source-bound preflight/training entrypoint.
set -Eeuo pipefail

readonly OFFICIAL_URL="https://github.com/neoteai/N0-TWAM.git"
readonly OFFICIAL_COMMIT="cdd87b6a141667123ad2c25f452478afdb71e287"
readonly ALLOWED_CHANGE="n0_twam/configs/twam_posttrain_cfg.py"
readonly RUNTIME_PYTHON="/mnt/data/task/n0_twam_track32_franka_20260810/runtime/venv_py310_24ec50d/bin/python"
readonly DIFFUSERS_OVERLAY="/mnt/data/task/n0_twam_track32_franka_20260810/deploy/wa_track3_clearup_s21000_20260817/python-overlay"
readonly LEROBOT_OVERLAY="/mnt/data/task/n0_twam_track32_franka_xyzw_20260817_v1/runtime/lerobot_0_3_3_overlay_v4"
readonly IMAGEIO_OVERLAY="/mnt/data/task/n0_twam_track31_retrain_vision_tactile_20260829_v1/runtime/python-overlay-official-cdd87b6-v1"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly RUNTIME_SHIM_DIR="$SCRIPT_DIR/runtime_shims"

die() {
  printf 'rank preflight error: %s\n' "$*" >&2
  exit 2
}

[[ $# -eq 1 ]] || die "expected one encoded payload"
[[ -f /opt/hyhal/env.sh ]] || die "HCU environment is not mounted"
# shellcheck disable=SC1091
source /opt/hyhal/env.sh

mapfile -d '' -t fields < <(
  "$RUNTIME_PYTHON" - "$1" <<'PY'
import base64
import json
import sys

try:
    payload = json.loads(base64.b64decode(sys.argv[1], validate=True))
    nodes = payload["nodes"]
    values = (
        payload["mode"],
        payload["launch_mode"],
        payload["rank"],
        len(nodes),
        payload["processes_per_node"],
        payload["expected_num_steps"],
        payload["visible_devices"],
        payload["fsdp_topology"],
        payload["fsdp_shard_size"],
        payload["master_addr"],
        payload["master_port"],
        payload["repo"],
        payload["save_root"],
        payload["latent_inventory_path"],
        payload["latent_inventory_sha256"],
        payload["config_sha256"],
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid launch payload: {exc}") from exc
for value in values:
    sys.stdout.buffer.write(str(value).encode("utf-8") + b"\0")
PY
)
[[ ${#fields[@]} -eq 16 ]] || die "invalid launch payload field count"

readonly mode="${fields[0]}"
readonly launch_mode="${fields[1]}"
readonly node_rank="${fields[2]}"
readonly nnodes="${fields[3]}"
readonly processes_per_node="${fields[4]}"
readonly expected_num_steps="${fields[5]}"
readonly visible_devices="${fields[6]}"
readonly fsdp_topology="${fields[7]}"
readonly fsdp_shard_size="${fields[8]}"
readonly master_addr="${fields[9]}"
readonly master_port="${fields[10]}"
readonly repo="${fields[11]}"
readonly save_root="${fields[12]}"
readonly latent_inventory_path="${fields[13]}"
readonly expected_latent_inventory_sha="${fields[14]}"
readonly expected_config_sha="${fields[15]}"

[[ "$mode" == "preflight" || "$mode" == "train" ]] || die "invalid mode"
[[ "$node_rank" =~ ^[0-9]+$ && "$node_rank" -lt "$nnodes" ]] || \
  die "node rank is outside the launch topology"
[[ "$expected_num_steps" =~ ^[1-9][0-9]*$ ]] || die "num_steps must be positive"
case "$launch_mode" in
  single-card-smoke)
    [[ "$nnodes" == "1" && "$node_rank" == "0" ]] || \
      die "single-card-smoke requires exactly one rank-0 node"
    [[ "$processes_per_node" == "1" ]] || \
      die "single-card-smoke requires one process"
    [[ "$expected_num_steps" == "1" ]] || \
      die "single-card-smoke requires exactly one optimizer step"
    [[ "$visible_devices" == "0" ]] || die "single-card HCU visibility mismatch"
    [[ "$fsdp_topology" == "global_shard" && "$fsdp_shard_size" == "1" ]] || \
      die "single-card FSDP contract mismatch"
    ;;
  distributed-smoke)
    [[ "$nnodes" == "2" && "$processes_per_node" == "8" ]] || \
      die "distributed-smoke requires 2 nodes x 8 processes"
    [[ "$expected_num_steps" -le 20 ]] || die "distributed smoke is not bounded"
    [[ "$visible_devices" == "0,1,5,4,2,3,7,6" ]] || \
      die "distributed HCU visibility mismatch"
    [[ "$fsdp_topology" == "global_shard" && "$fsdp_shard_size" == "16" ]] || \
      die "distributed official FSDP world-shard contract mismatch"
    ;;
  formal)
    [[ "$nnodes" == "2" && "$processes_per_node" == "8" ]] || \
      die "formal training requires 2 nodes x 8 processes"
    [[ "$visible_devices" == "0,1,5,4,2,3,7,6" ]] || \
      die "formal HCU visibility mismatch"
    [[ "$fsdp_topology" == "global_shard" && "$fsdp_shard_size" == "16" ]] || \
      die "formal official FSDP world-shard contract mismatch"
    ;;
  *) die "unsupported launch mode: $launch_mode" ;;
esac
[[ "$master_port" =~ ^[0-9]+$ ]] || die "master port is not numeric"
[[ "$expected_latent_inventory_sha" =~ ^[0-9a-f]{64}$ ]] || \
  die "invalid latent inventory SHA256"
[[ "$expected_config_sha" =~ ^[0-9a-f]{64}$ ]] || die "invalid config SHA256"
[[ -x "$RUNTIME_PYTHON" ]] || die "proven runtime Python is not executable"
for overlay in "$DIFFUSERS_OVERLAY" "$LEROBOT_OVERLAY" "$IMAGEIO_OVERLAY"; do
  [[ -d "$overlay" ]] || die "required Python overlay missing: $overlay"
done
[[ -f "$RUNTIME_SHIM_DIR/sitecustomize.py" ]] || \
  die "required vendor PyTorch compatibility shim missing"
[[ -f "$RUNTIME_SHIM_DIR/n0_train_compat.py" ]] || \
  die "required vendor FlexAttention compatibility shim missing"
[[ -f "$RUNTIME_SHIM_DIR/flex25_compat.py" ]] || \
  die "required vendor grouped-Flash compatibility layer missing"
[[ -d "$repo/.git" ]] || die "official checkout missing: $repo"

readonly actual_head="$(git -C "$repo" rev-parse HEAD)"
[[ "$actual_head" == "$OFFICIAL_COMMIT" ]] || die "official commit mismatch: $actual_head"
readonly actual_origin="$(git -C "$repo" remote get-url origin)"
[[ "${actual_origin%/}" == "${OFFICIAL_URL%/}" ]] || die "official origin mismatch"
while IFS= read -r status_line; do
  [[ -z "$status_line" || "${status_line:3}" == "$ALLOWED_CHANGE" ]] || \
    die "forbidden upstream worktree change: $status_line"
done < <(git -C "$repo" status --porcelain=v1 --untracked-files=all)

readonly config_path="$repo/$ALLOWED_CHANGE"
[[ -f "$config_path" ]] || die "generated config is missing"
readonly actual_config_sha="$(sha256sum "$config_path" | awk '{print $1}')"
[[ "$actual_config_sha" == "$expected_config_sha" ]] || die "config SHA256 mismatch"
[[ -f "$latent_inventory_path" ]] || die "latent inventory is missing"
readonly actual_latent_inventory_sha="$(sha256sum "$latent_inventory_path" | awk '{print $1}')"
[[ "$actual_latent_inventory_sha" == "$expected_latent_inventory_sha" ]] || \
  die "latent inventory SHA256 mismatch"

export ROBOTACTILE_ENABLE_FSDP2_NAMESPACE_SHIM=1
export ROBOTACTILE_ENABLE_FLEX25_COMPAT=1
export ROBOTACTILE_VENDOR_TRAIN_FAST_EXIT=1
export ROBOTACTILE_GROUPED_FLASH_MAX_QUERY_TOKENS=32768
export N0_FLEX_ATTENTION_BACKEND=grouped_flash_attn
export N0_MOT_CROSS_ATTENTION_BACKEND=flash_attn
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH="$RUNTIME_SHIM_DIR:$DIFFUSERS_OVERLAY:$LEROBOT_OVERLAY:$IMAGEIO_OVERLAY:$repo:$repo/n0_twam${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export N0_EXPECTED_REPO="$repo"
export N0_EXPECTED_SAVE_ROOT="$save_root"
export N0_PREFLIGHT_MODE="$mode"
export N0_EXPECTED_DEVICE_COUNT="$processes_per_node"
export N0_EXPECTED_NUM_STEPS="$expected_num_steps"
export N0_LATENT_INVENTORY_PATH="$latent_inventory_path"
export N0_NODE_RANK="$node_rank"
export HIP_VISIBLE_DEVICES="$visible_devices"
export CUDA_VISIBLE_DEVICES="$visible_devices"
export N0_FSDP_TOPOLOGY="$fsdp_topology"
export N0_FSDP_SHARD_SIZE="$fsdp_shard_size"
export N0_FSDP_HSDP_STATE_DICT_COMPAT=vendor_torch_2_5_1_remove_2d_guard_v1
export LD_LIBRARY_PATH="/opt/n0_twam/rccl_plugin:/opt/shca_device_driver/ucx-1.18.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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

"$RUNTIME_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from n0_twam.configs import TWAM_CONFIGS
from flex25_compat import GroupedAttentionMask, install_vendor_flex25_compat

if fully_shard.__module__ != "torch.distributed._composable.fsdp.fully_shard":
    raise RuntimeError(f"unexpected FSDP2 fully_shard source: {fully_shard.__module__}")
if MixedPrecisionPolicy.__module__ != "torch.distributed._composable.fsdp._fsdp_api":
    raise RuntimeError(
        "unexpected FSDP2 MixedPrecisionPolicy source: "
        f"{MixedPrecisionPolicy.__module__}"
    )

flex_mask_compat = install_vendor_flex25_compat()
from models.model import FlexAttnFunc

probe_length = 128
probe_ids = torch.zeros(probe_length, dtype=torch.long, device="cuda")
probe_frames = torch.arange(probe_length, device="cuda", dtype=torch.long) // 32
probe_noise = torch.cat(
    (
        torch.zeros(probe_length // 2, dtype=torch.long, device="cuda"),
        torch.ones(probe_length // 2, dtype=torch.long, device="cuda"),
    )
)
probe_modality = torch.zeros(probe_length, dtype=torch.long, device="cuda")
probe_mask_mod = FlexAttnFunc._get_mask_mod(
    probe_ids,
    probe_frames,
    probe_noise,
    probe_modality,
    window_size=4,
)
probe_block_mask = FlexAttnFunc.compiled_create_block_mask(
    probe_mask_mod,
    1,
    1,
    probe_length,
    probe_length,
    device=torch.device("cuda"),
    _compile=True,
)
if tuple(probe_block_mask.shape)[-2:] != (probe_length, probe_length):
    raise RuntimeError(
        "vendor FlexAttention block-mask probe returned an unexpected shape: "
        f"{tuple(probe_block_mask.shape)}"
    )
if not isinstance(probe_block_mask, GroupedAttentionMask) or not probe_block_mask.groups:
    raise RuntimeError("vendor grouped-Flash self-mask probe has no groups")
probe_cross_mod = FlexAttnFunc._get_cross_mask_mod(
    probe_ids,
    probe_modality,
    probe_ids,
    probe_ids,
)
probe_cross_mask = FlexAttnFunc.compiled_create_block_mask(
    probe_cross_mod,
    1,
    1,
    probe_length,
    probe_length,
    device=torch.device("cuda"),
    _compile=True,
)
if tuple(probe_cross_mask.shape)[-2:] != (probe_length, probe_length):
    raise RuntimeError(
        "vendor FlexAttention cross-mask probe returned an unexpected shape: "
        f"{tuple(probe_cross_mask.shape)}"
    )
if not isinstance(probe_cross_mask, GroupedAttentionMask) or not probe_cross_mask.groups:
    raise RuntimeError("vendor grouped-Flash cross-mask probe has no groups")

probe_attention = FlexAttnFunc()
probe_attention.set_block_mask(probe_block_mask)
probe_query = torch.randn(
    1,
    probe_length,
    2,
    128,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
probe_key = probe_query.detach().clone().requires_grad_(True)
probe_value = probe_query.detach().clone().requires_grad_(True)
probe_output = probe_attention(probe_query, probe_key, probe_value)
if tuple(probe_output.shape) != tuple(probe_query.shape):
    raise RuntimeError("vendor grouped-Flash forward probe shape mismatch")
probe_output.float().square().mean().backward()
for label, tensor in (
    ("query", probe_query),
    ("key", probe_key),
    ("value", probe_value),
):
    if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
        raise RuntimeError(f"vendor grouped-Flash {label} backward probe failed")

expected_device_count = int(os.environ["N0_EXPECTED_DEVICE_COUNT"])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected_device_count:
    raise RuntimeError(
        "official N0-TWAM CUDA visibility differs from the launch profile; "
        f"found {torch.cuda.device_count()}"
    )
cfg = TWAM_CONFIGS["posttrain"]
dataset = Path(cfg.dataset_path)
if dataset.name != "train759" or "frozen40" in dataset.parts:
    raise RuntimeError(f"training dataset is not certified train759: {dataset}")
if cfg.val_dataset_path is not None:
    raise RuntimeError("val_dataset_path must be None")
if not dataset.exists():
    raise FileNotFoundError(dataset)
for field in (
    "wan22_pretrained_model_name_or_path",
    "empty_emb_path",
    "resume_from",
    "norm_stat_path",
):
    value = Path(getattr(cfg, field))
    if not value.exists():
        raise FileNotFoundError(f"{field}: {value}")
expected = {
    "tactile_optional": False,
    "synthetic_tactile_data": False,
    "use_local_tactile": True,
    "tactile_global_zero": False,
    "tactile_cfg_prob": 0.0,
    "cfg_prob": 0.0,
    "noisy_cond_prob_tactile": 0.0,
    "tactile_diffusion_loss_weight": 1.0,
    "action_per_frame": 4,
    "pi05_action_horizon": 4,
    "max_latent_frames": 5,
}
for field, expected_value in expected.items():
    actual = getattr(cfg, field)
    if actual != expected_value:
        raise RuntimeError(f"Vision+tactile contract mismatch: {field}={actual!r}")
if not cfg.obs_cam_keys or not cfg.tactile_keys:
    raise RuntimeError("Vision and tactile key lists must both be non-empty")
if cfg.num_steps != int(os.environ["N0_EXPECTED_NUM_STEPS"]):
    raise RuntimeError("generated config num_steps differs from the launch profile")
if Path(cfg.save_root) != Path(os.environ["N0_EXPECTED_SAVE_ROOT"]):
    raise RuntimeError("save_root differs from the source-bound launch payload")
if os.environ["N0_PREFLIGHT_MODE"] == "preflight" and Path(cfg.save_root).exists():
    raise FileExistsError(f"no-clobber save_root already exists: {cfg.save_root}")

# Inspect one source-bound Vision/tactile pair per task. The controller already
# validates and stats all 4,554 signed inventory files; this bounded runtime
# witness proves that the actual payload frame_ids consumed by the official
# loader have stride 1 at the certified 10 Hz cadence.
if os.environ["N0_NODE_RANK"] == "0":
    inventory_path = Path(os.environ["N0_LATENT_INVENTORY_PATH"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    episodes = inventory.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("latent inventory has no episode list")
    expected_tasks = {
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    }
    samples = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        task = episode.get("task")
        vision = episode.get("vision")
        tactile = episode.get("tactile")
        if task in expected_tasks and task not in samples:
            if isinstance(vision, dict) and isinstance(tactile, dict):
                if int(vision.get("latent_num_frames", 0)) > 0:
                    samples[task] = (vision, tactile)
    if set(samples) != expected_tasks:
        raise RuntimeError("temporal witness does not cover all eight tasks")

    def load_frame_ids(modality, key):
        outputs = modality.get("outputs")
        if not isinstance(outputs, dict) or key not in outputs:
            raise RuntimeError(f"temporal witness output missing: {key}")
        metadata = outputs[key]
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
            raise RuntimeError(f"temporal witness path is invalid: {key}")
        path = Path(metadata["path"])
        try:
            path.resolve(strict=True).relative_to(dataset.resolve(strict=True))
        except ValueError as exc:
            raise RuntimeError(f"temporal witness escapes train759: {path}") from exc
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        frame_ids = payload.get("frame_ids") if isinstance(payload, dict) else None
        if not isinstance(frame_ids, list) or len(frame_ids) < 2:
            raise RuntimeError(f"temporal witness frame_ids are incomplete: {path}")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in frame_ids
        ):
            raise RuntimeError(f"temporal witness frame_ids are invalid: {path}")
        if any(right - left != 1 for left, right in zip(frame_ids, frame_ids[1:])):
            raise RuntimeError(f"latent frame_stride is not 1: {path}")
        if payload.get("fps") != 10 or payload.get("ori_fps") != 10:
            raise RuntimeError(f"latent payload is not source/target 10 Hz: {path}")
        return frame_ids

    for task, (vision, tactile) in sorted(samples.items()):
        vision_ids = load_frame_ids(vision, "observation.images.top")
        tactile_ids = load_frame_ids(
            tactile, "global:observation.images.tactile_a"
        )
        if vision_ids != tactile_ids:
            raise RuntimeError(f"Vision/tactile frame_ids differ for task {task}")

print(
    f"PREFLIGHT_OK devices={torch.cuda.device_count()} "
    f"dataset={cfg.dataset_path} vision={len(cfg.obs_cam_keys)} "
    f"tactile={len(cfg.tactile_keys)} frame_stride=1 "
    "fsdp2_api=torch.distributed._composable.fsdp "
    f"flex_mask_compat={flex_mask_compat} "
    "attention_forward_backward=grouped_flash"
)
PY

if [[ "$mode" == "preflight" ]]; then
  exit 0
fi

readonly runtime_root="$(dirname "$save_root")/runtime"
readonly runtime_work_dir="$runtime_root/node-$node_rank"
mkdir -p "$runtime_root"
mkdir "$runtime_work_dir"
cd "$runtime_work_dir"
exec "$RUNTIME_PYTHON" -m torch.distributed.run \
  --nnodes "$nnodes" \
  --nproc-per-node "$processes_per_node" \
  --node-rank "$node_rank" \
  --master-addr "$master_addr" \
  --master-port "$master_port" \
  --max-restarts 0 \
  --tee 3 \
  -m n0_train_compat \
  --config-name posttrain \
  --save-root "$save_root"

"""Render the sole permitted change inside the official N0-TWAM checkout."""

from __future__ import annotations

import hashlib
import json
import math
import pprint
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpu_training.contract import (  # noqa: E402
    ALLOWED_OFFICIAL_CHANGE,
    TrainingSpec,
    verify_official_checkout,
)
from hpu_training.latent.contracts import TASKS  # noqa: E402


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _read_norm_vectors(path: Path) -> tuple[list[float], list[float]]:
    payload = _load_object(path, label="normalization stats")
    q01 = payload.get("q01", payload.get("action_q01"))
    q99 = payload.get("q99", payload.get("action_q99"))
    if not isinstance(q01, list) or not isinstance(q99, list):
        raise ValueError("normalization stats must contain q01/q99 vectors")
    if len(q01) != 20 or len(q99) != 20:
        raise ValueError("official absEE training requires 20D q01/q99 vectors")
    low = [float(value) for value in q01]
    high = [float(value) for value in q99]
    if not all(math.isfinite(value) for value in low + high):
        raise ValueError("normalization stats must be finite")
    if any(upper <= lower for lower, upper in zip(low, high)):
        raise ValueError("every q99 value must be greater than q01")
    if low == [-1.0] * 20 and high == [1.0] * 20:
        raise ValueError("refusing the official placeholder normalization stats")
    return low, high


def _validate_per_repo_norm(payload: dict[str, object]) -> None:
    if set(payload) != set(TASKS):
        raise ValueError("per-repo normalization must cover exactly eight task repos")
    for repo_name, value in payload.items():
        if not isinstance(repo_name, str) or not isinstance(value, dict):
            raise ValueError("per-repo normalization entries must be JSON objects")
        q01 = value.get("q01")
        q99 = value.get("q99")
        if not isinstance(q01, list) or not isinstance(q99, list):
            raise ValueError(f"per-repo stats are incomplete for {repo_name!r}")
        if len(q01) != 20 or len(q99) != 20:
            raise ValueError(f"per-repo stats must be 20D for {repo_name!r}")


def render_posttrain_config(
    *, repo: Path, spec: TrainingSpec, save_root: Path
) -> dict[str, object]:
    """Atomically replace the sole allowed official file with a strict config."""
    verify_official_checkout(repo)
    q01, q99 = _read_norm_vectors(spec.norm_stat_path)
    per_repo = _load_object(
        spec.per_repo_norm_stat_path, label="per-repo normalization stats"
    )
    _validate_per_repo_norm(per_repo)
    values = {
        "dataset_path": str(spec.dataset_path),
        "base_model_path": str(spec.base_model_path),
        "released_checkpoint_path": str(spec.released_checkpoint_path),
        "empty_embedding_path": str(spec.empty_embedding_path),
        "norm_stat_path": str(spec.norm_stat_path),
        "save_root": str(save_root),
        "prompt": spec.prompt,
        "q01": q01,
        "q99": q99,
        "per_repo_norm_stat": per_repo,
        "max_latent_frames": spec.max_latent_frames,
        "num_steps": spec.num_steps,
        "save_interval": spec.save_interval,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "batch_size": spec.batch_size,
        "load_worker": spec.load_worker,
        "learning_rate": spec.learning_rate,
        "warmup_steps": spec.warmup_steps,
        "gc_interval": spec.gc_interval,
        "seed": spec.seed,
    }
    literal = pprint.pformat(values, sort_dicts=True, width=88)
    source = f'''# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""RoboTactile-generated train-only configuration for official N0-TWAM.

Generated outside the upstream source tree logic. This is the only upstream file
that the external training workflow is permitted to modify.
"""

from easydict import EasyDict

from .twam_base_cfg import twam_base_cfg

_VALUES = {literal}

cfg = EasyDict(twam_base_cfg.copy())
cfg.__name__ = "Config: official N0-TWAM UniVTAC train759 Vision+tactile"

# Train-only data boundary: no validation loader can be constructed.
cfg.dataset_path = _VALUES["dataset_path"]
cfg.val_dataset_path = None
cfg.val_interval = _VALUES["num_steps"] + 1

# UniVTAC observation and official 20D absolute-EE action schema. The source and
# encoded dataset remain at their real 10 Hz cadence: one latent anchor spans
# exactly four actions; no synthetic 30 Hz duplication is permitted.
cfg.obs_cam_keys = ["observation.images.top", "observation.images.wrist_l"]
cfg.tactile_keys = [
    "observation.images.tactile_a",
    "observation.images.tactile_b",
]
cfg.per_repo_obs_cam_keys = {{}}
cfg.per_repo_tactile_keys = {{}}
cfg.tactile_optional = False
cfg.synthetic_tactile_data = False
cfg.tactile_sensor_id_map = {{key: index for index, key in enumerate(cfg.tactile_keys)}}
cfg.max_tactile_streams = 4
cfg.use_local_tactile = True
cfg.local_tactile_mode = "current"
cfg.tactile_global_zero = False

# The official loader samples a fresh contiguous five-latent window from every
# long episode. This preserves all train759 episodes while bounding attention
# activation memory to the proven 2-node recipe.
cfg.max_latent_frames = _VALUES["max_latent_frames"]

cfg.action_dim = 20
cfg.action_delta_mode = "none"
cfg.pi05_action_horizon = 4
cfg.action_per_frame = 4
cfg.pi05_source_action_is_delta = False
cfg.pi05_state_column = "observation.state"
cfg.pi05_delta_channel_ids = list(range(0, 9)) + list(range(10, 19))
cfg.pi05_rot6d_relative_delta = False
cfg.pi05_condition_first_frame_zero = True
cfg.pi05_condition_first_frame_loss = False
cfg.pi05_invalid_horizon_mask = True
cfg.used_action_channel_ids = list(range(10))
cfg.per_repo_used_action_channel_ids = {{}}
cfg.inverse_used_action_channel_ids = list(range(10)) + [10] * 10
cfg.action_norm_method = "q01q99"
cfg.norm_stat = {{"q01": _VALUES["q01"], "q99": _VALUES["q99"]}}
cfg.norm_stat_path = _VALUES["norm_stat_path"]
cfg.per_repo_norm_stat = _VALUES["per_repo_norm_stat"]

cfg.wan22_pretrained_model_name_or_path = _VALUES["base_model_path"]
cfg.empty_emb_path = _VALUES["empty_embedding_path"]
cfg.resume_from = _VALUES["released_checkpoint_path"]
cfg.save_root = _VALUES["save_root"]
cfg.eval_prompt = _VALUES["prompt"]

cfg.use_mot = True
cfg.mot_cross_attn_experts = ("video", "action")
cfg.mot_warmstart_experts = ("video", "action", "tactile")
cfg.mot_expert_hidden_dim = {{"action": 1024, "tactile": 1024}}
cfg.mot_expert_ffn_dim = {{"action": 4096, "tactile": 4096}}
cfg.use_contact_gate = False

# Vision+tactile always-on: no tactile or joint conditioning dropout.
cfg.tactile_cfg_prob = 0.0
cfg.cfg_prob = 0.0
cfg.noisy_cond_prob_tactile = 0.0
cfg.tactile_diffusion_loss_weight = 1.0

cfg.enable_wandb = False
cfg.num_init_worker = 1
cfg.load_worker = _VALUES["load_worker"]
cfg.batch_size = _VALUES["batch_size"]
cfg.gradient_accumulation_steps = _VALUES["gradient_accumulation_steps"]
cfg.num_steps = _VALUES["num_steps"]
cfg.save_interval = _VALUES["save_interval"]
cfg.gc_interval = _VALUES["gc_interval"]
cfg.learning_rate = _VALUES["learning_rate"]
cfg.lr_schedule = "cosine"
cfg.lr_min_ratio = 0.1
cfg.warmup_steps = _VALUES["warmup_steps"]
cfg.seed = _VALUES["seed"]

assert cfg.dataset_path.endswith("/train759") and cfg.val_dataset_path is None
assert cfg.obs_cam_keys and cfg.tactile_keys
assert cfg.tactile_optional is False and cfg.synthetic_tactile_data is False
assert cfg.use_local_tactile is True and cfg.tactile_global_zero is False
assert cfg.tactile_cfg_prob == cfg.cfg_prob == cfg.noisy_cond_prob_tactile == 0.0
assert cfg.tactile_diffusion_loss_weight == 1.0
assert cfg.max_latent_frames == 5
assert len(cfg.norm_stat["q01"]) == len(cfg.norm_stat["q99"]) == 20
assert cfg.action_dim == 20 and cfg.action_per_frame == 4

twam_posttrain_cfg = cfg
'''
    target = repo / ALLOWED_OFFICIAL_CHANGE
    temporary = target.with_suffix(".py.robotactile-tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(target)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    checkout = verify_official_checkout(repo)
    return {
        "config_path": str(target),
        "config_sha256": digest,
        "official_source": checkout,
        "train_split": "train759",
        "validation_dataset_path": None,
        "vision_tactile_policy": "always_on",
        "source_fps": 10,
        "latent_frame_stride": 1,
        "action_per_frame": 4,
        "max_latent_frames": spec.max_latent_frames,
        "num_steps": spec.num_steps,
    }

"""Build the pinned upstream FTP-1 data and training commands."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .contracts import TASK_CAMERA_MODE, TASK_IDS, FTP1TrainingRequest

REPO_PREFIX: Final[str] = "UniVTAC"


def repository_id() -> str:
    """Return the single assets namespace shared by the joint model."""

    return f"{REPO_PREFIX}_all8_train759_zscore_mix_ftp1"


def experiment_name(request: FTP1TrainingRequest) -> str:
    return request.run_id


def task_zarr_path(request: FTP1TrainingRequest, task: str) -> Path:
    suffix = "all" if TASK_CAMERA_MODE[task] == "head+wrist" else "head"
    return request.zarr_root / task / f"{task}_{suffix}.zarr"


def dataset_config(request: FTP1TrainingRequest) -> dict[str, object]:
    return {
        "datasets": [
            {
                "name": f"UniVTAC_{task}",
                "path": str((request.zarr_root / task).resolve()),
                "use_trajectory_ratio": 1.0,
                "enabled": True,
            }
            for task in TASK_IDS
        ],
        "default_use_trajectory_ratio": 1.0,
        "description": (
            "RoboTactile train759-only joint all-8 FTP-1 dataset; each entry "
            "retains independent normalization and a domain-homogeneous batch schema."
        ),
    }


def parser_command(request: FTP1TrainingRequest, task: str) -> tuple[str, ...]:
    entrypoint = Path(__file__).with_name("rgb_parser_entrypoint.py").resolve()
    return (
        str(request.python_executable),
        str(entrypoint),
        f"--ftp1-root={request.ftp1_root}",
        f"--base-dir={request.staging_root}",
        f"--save-dir={request.zarr_root / task}",
        f"--task-list={task}",
        "--downsample=1",
        "--image-size=224",
    )


def rgb_contract_command(
    request: FTP1TrainingRequest,
    task: str,
    zarr_path: Path,
) -> tuple[str, ...]:
    entrypoint = Path(__file__).with_name("rgb_contract.py").resolve()
    command = [
        str(request.python_executable),
        str(entrypoint),
        f"--ftp1-root={request.ftp1_root}",
        f"--staging-root={request.staging_root}",
        f"--zarr-path={zarr_path}",
        f"--task={task}",
        "--image-size=224",
    ]
    if TASK_CAMERA_MODE[task] == "head+wrist":
        command.append("--use-wrist")
    return tuple(command)


def norm_command(request: FTP1TrainingRequest, config_path: Path) -> tuple[str, ...]:
    repo_id = repository_id()
    entrypoint = Path(__file__).with_name("norm_entrypoint.py").resolve()
    return (
        str(request.python_executable),
        str(entrypoint),
        "ftp1",
        f"--repo_id={repo_id}",
        f"--data.repo-id={repo_id}",
        f"--exp_name={request.run_id}_joint_norm",
        "--use_val_dataset",
        "--create_train_val_split",
        "--val_ratio=0.0",
        "--no-wandb_enabled",
        f"--checkpoint_base_dir={request.checkpoint_base_dir}",
        f"--assets_base_dir={request.assets_base_dir}",
        f"--dataset_config_path={config_path}",
        "--norm_sample_ratio=1.0",
        f"--batch_size={request.batch_size}",
        "--norm_batch_size=64",
        f"--norm_num_workers={request.num_workers}",
        "--action_down_sample_steps=1",
        "--norm_type=zscore",
        "--norm_image_tactile_mode=channel_wise",
        "--contact_detection_threshold_k=3.0",
        "--no-skip_contact_detection",
        "--proprioception_pose_rep=relative",
        "--action_pose_rep=relative",
        "--proprioception_joint_rep=abs",
        "--action_joint_rep=mix",
    )


def training_command(
    request: FTP1TrainingRequest,
    config_path: Path,
    *,
    resume: bool,
) -> tuple[str, ...]:
    repo_id = repository_id()
    command = [
        str(request.python_executable),
        "scripts/zarr_train_ftp1_pytorch.py",
        "ftp1",
        f"--exp_name={experiment_name(request)}",
        f"--repo_id={repo_id}",
        f"--data.repo-id={repo_id}",
        f"--checkpoint_base_dir={request.checkpoint_base_dir}",
        f"--assets_base_dir={request.assets_base_dir}",
        f"--dataset_config_path={config_path}",
        f"--batch_size={request.batch_size}",
        "--action_down_sample_steps=1",
        f"--num_train_steps={request.num_train_steps}",
        f"--log_interval={request.log_interval}",
        f"--val_interval={request.val_interval}",
        "--val_ratio=0.0",
        f"--save_interval={request.save_interval}",
        f"--keep_period={request.keep_period}",
        f"--num_workers={request.num_workers}",
        f"--val_num_workers={request.val_num_workers}",
        "--use_val_dataset",
        "--create_train_val_split",
        "--check_norm_params_snapshot",
        "--norm_type=zscore",
        "--norm_image_tactile_mode=channel_wise",
        "--pytorch_training_precision=bfloat16",
        "--model.state_input_mode=adarms",
        "--model.tactile_expert_variant=gemma_small",
        "--model.use_tactile_input",
        "--non_tactile_dropout_ratio=0.0",
        "--model.tactile_tokenizer_config.no_load_t3_pretrained_checkpoint",
        "--model.tactile_tokenizer_config.no_frozen_shared_chunk",
        "--proprioception_pose_rep=relative",
        "--action_pose_rep=relative",
        "--proprioception_joint_rep=abs",
        "--action_joint_rep=mix",
        "--lr_schedule.warmup_steps=500",
        "--lr_schedule.peak_lr=1e-4",
        f"--lr_schedule.decay_steps={request.num_train_steps}",
        "--lr_schedule.decay_lr=1e-5",
        "--optimizer.b1=0.9",
        "--optimizer.b2=0.95",
        "--optimizer.eps=1e-8",
        "--optimizer.weight_decay=1e-10",
        "--optimizer.clip_gradient_norm=1.0",
        "--lr_decay_till_end",
        "--ema_decay=0.99",
        f"--seed={request.seed}",
        "--no-wandb_enabled" if not request.wandb_enabled else "--wandb_enabled",
        "--use_torch_compile"
        if request.use_torch_compile
        else "--no-use_torch_compile",
        (
            "--gradient_checkpointing_enable"
            if request.gradient_checkpointing
            else "--no-gradient_checkpointing_enable"
        ),
    ]
    if resume:
        command.append("--resume")
    else:
        command.append(f"--pytorch_weight_path={request.base_checkpoint_dir}")
    return tuple(command)


def checkpoint_root(request: FTP1TrainingRequest) -> Path:
    return request.checkpoint_base_dir / "ftp1" / experiment_name(request)


__all__ = [
    "checkpoint_root",
    "dataset_config",
    "experiment_name",
    "norm_command",
    "parser_command",
    "rgb_contract_command",
    "repository_id",
    "task_zarr_path",
    "training_command",
]

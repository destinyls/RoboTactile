"""Resolve exact Dream-Tac 2-node commands and rank payloads."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.dream_tac.training.launch_training import (
    build_dataset_contract,
    build_overrides,
)
from scripts.dream_tac.training.training_request import BoundFile, sha256_regular_file

from .distributed_request import TASKS, DistributedTrainingRequest
from .optimizer_step_request import HcuOptimizerStepRequest
from .receipt import signed_receipt


@dataclass(frozen=True)
class _OverrideRequest:
    materialization_root: Path
    base_checkpoint: BoundFile
    batch_size: int
    num_workers: int
    max_iter: int
    save_iter: int
    run_name: str

    @property
    def resume_checkpoint(self) -> None:
        return None

    @property
    def job_project(self) -> str:
        return "robotactile_dream_tac"

    @property
    def job_group(self) -> str:
        return "univtac_hcu_2node"


def training_overrides(
    request: DistributedTrainingRequest,
    source: HcuOptimizerStepRequest,
) -> tuple[str, ...]:
    """Return the train759/CASA overrides shared by all 16 ranks."""

    adapter = _OverrideRequest(
        materialization_root=source.materialization_root,
        base_checkpoint=source.base_checkpoint,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        max_iter=request.max_iter,
        save_iter=request.save_iter,
        run_name=request.run_name,
    )
    excluded = (
        "checkpoint.load_path=",
        "checkpoint.load_training_state=",
        "checkpoint.strict_resume=",
    )
    common = tuple(
        item for item in build_overrides(adapter) if not item.startswith(excluded)
    )
    return (
        *common,
        "~trainer.callbacks.device_monitor",
        f"trainer.grad_accum_iter={request.grad_accum_iter}",
        f"checkpoint.load_path={source.base_dcp_root}",
        "checkpoint.load_training_state=false",
        "checkpoint.strict_resume=false",
        f"model.config.tokenizer.vae_pth={source.tokenizer_checkpoint.path}",
        f"model.config.fsdp_shard_size={request.fsdp_shard_size}",
        "model_parallel.tensor_model_parallel_size=1",
        "model_parallel.pipeline_model_parallel_size=1",
        "model_parallel.context_parallel_size=1",
        "trainer.distributed_parallelism=fsdp",
        "checkpoint.dcp_async_mode_enabled=false",
    )


def build_rank_payload(
    request: DistributedTrainingRequest,
    source: HcuOptimizerStepRequest,
    *,
    node_rank: int,
    mode: str,
) -> str:
    if mode not in {"preflight", "train"}:
        raise ValueError("rank mode must be preflight or train")
    if node_rank not in {0, 1}:
        raise ValueError("node_rank must be 0 or 1")
    entrypoint = (
        request.robotactile_root
        / "scripts/dream_tac/hcu_port/distributed_rank_entrypoint.sh"
    )
    collective_probe = (
        request.robotactile_root
        / "scripts/dream_tac/hcu_port/distributed_collective_probe.py"
    )
    payload: dict[str, object] = {
        "mode": mode,
        "launch_mode": request.launch_mode,
        "node_rank": node_rank,
        "nnodes": 2,
        "nproc_per_node": request.nproc_per_node,
        "world_size": request.world_size,
        "fsdp_shard_size": request.fsdp_shard_size,
        "visible_devices": request.visible_devices,
        "master_addr": request.master_addr,
        "master_port": request.master_port,
        "network_interface": request.network_interface,
        "max_iter": request.max_iter,
        "python_executable": str(source.python_executable),
        "hcu_environment_script": source.hcu_environment_script.to_dict(),
        "pythonpath": ":".join(
            (
                str(request.robotactile_root),
                str(source.dream_tac_root),
                *(str(path) for path in source.runtime_pythonpath_roots),
            )
        ),
        "dream_tac_root": str(source.dream_tac_root),
        "dream_tac_host_root": str(request.dream_tac_host_root),
        "franka_dataset": {
            "path": str(
                source.dream_tac_root / "cosmos_policy/datasets/franka_dataset.py"
            ),
            "sha256": request.franka_dataset_sha256,
        },
        "fsdp_sources": {
            "mesh_mode": "global_1d_shard_v1",
            "fsdp_helper": {
                "path": str(
                    source.dream_tac_root
                    / "cosmos_policy/_src/imaginaire/utils/fsdp_helper.py"
                ),
                "sha256": request.fsdp_helper_sha256,
            },
            "dtensor_helper": {
                "path": str(
                    source.dream_tac_root
                    / "cosmos_policy/_src/predict2/utils/dtensor_helper.py"
                ),
                "sha256": request.dtensor_helper_sha256,
            },
        },
        "runtime_host_root": str(request.runtime_host_root),
        "output_root": str(request.output_root),
        "base_dcp_root": str(source.base_dcp_root),
        "container_name": (
            f"robotactile-dream-tac-{request.run_name}-{request.request_sha256[:12]}-"
            f"{mode}-rank-{node_rank:02d}"
        ),
        "container_entrypoint": str(entrypoint),
        "collective_probe": {
            "path": str(collective_probe),
            "sha256": sha256_regular_file(collective_probe),
        },
        "overrides": list(training_overrides(request, source)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(encoded).decode("ascii")


def build_resolved_experiment(
    request: DistributedTrainingRequest,
    source: HcuOptimizerStepRequest,
) -> dict[str, object]:
    adapter = _OverrideRequest(
        materialization_root=source.materialization_root,
        base_checkpoint=source.base_checkpoint,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        max_iter=request.max_iter,
        save_iter=request.save_iter,
        run_name=request.run_name,
    )
    return signed_receipt(
        {
            "schema_version": 1,
            "status": "resolved",
            "protocol_id": "dream_tac_hcu_2node_resolved_experiment_v1",
            "request_sha256": request.request_sha256,
            "dataset_split": "train759",
            "tasks": list(TASKS),
            "task_count": 8,
            "dataset_bindings": {
                "dataloader_train.dataset": build_dataset_contract(adapter),
                "dataloader_train.sampler.dataset": build_dataset_contract(adapter),
            },
            "vision_tactile_proprio_always_on": True,
            "franka_dataset": {
                "host_path": str(request.franka_dataset_host_path),
                "sha256": request.franka_dataset_sha256,
                "loading": "lazy_random_access_frames_v1",
            },
            "distributed": {
                "nodes": list(request.nodes),
                "nproc_per_node": request.nproc_per_node,
                "world_size": request.world_size,
                "fsdp_shard_size": request.fsdp_shard_size,
                "fsdp_topology": "global_1d_16way_shard",
                "batch_size_per_rank": request.batch_size,
                "grad_accum_iter": request.grad_accum_iter,
                "effective_global_batch": request.effective_global_batch,
            },
            "checkpoint": {
                "load_path": str(source.base_dcp_root),
                "load_training_state": False,
                "single_rank_optimizer_state_used_as_resume": False,
            },
            "overrides": list(training_overrides(request, source)),
        }
    )


__all__ = ["build_rank_payload", "build_resolved_experiment", "training_overrides"]

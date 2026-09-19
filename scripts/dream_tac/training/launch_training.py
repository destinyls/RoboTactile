"""Launch source-bound Dream-Tac UniVTAC P2/P3 on one NVIDIA CUDA node."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from .identity import (
    load_json_object,
    receipt_sha256,
    signed_payload,
    write_or_verify_json,
)
from .training_request import DONOR_EXPERIMENT, BoundFile, DreamTacTrainingRequest

LaunchMode = Literal["dry_run", "train"]
RESOLVED_EXPERIMENT_NAME: Final[str] = "resolved_experiment.json"


class TrainingOverrideRequest(Protocol):
    """Read-only request surface shared by CUDA and bounded HCU launchers."""

    @property
    def materialization_root(self) -> Path: ...

    @property
    def base_checkpoint(self) -> BoundFile: ...

    @property
    def resume_checkpoint(self) -> BoundFile | None: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def num_workers(self) -> int: ...

    @property
    def max_iter(self) -> int: ...

    @property
    def save_iter(self) -> int: ...

    @property
    def job_project(self) -> str: ...

    @property
    def job_group(self) -> str: ...

    @property
    def run_name(self) -> str: ...


def _boolean(value: bool) -> str:
    return "true" if value else "false"


def build_dataset_contract(
    request: TrainingOverrideRequest,
) -> dict[str, str | bool | int | float]:
    """Return the single UniVTAC dataset contract used by both config paths."""

    dataset_root = request.materialization_root / "dataset"
    return {
        "data_dir": str(dataset_root),
        "chunk_size": 20,
        "t5_text_embeddings_path": str(dataset_root / "t5_embeddings.pkl"),
        "use_tactile": True,
        "use_proprio": True,
        "use_wrist_images": True,
        "use_third_person_images": True,
        "demonstration_sampling_prob": 1.0,
        "success_rollout_sampling_prob": 0.0,
        "return_value_function_returns": False,
    }


def _dataset_overrides(
    request: TrainingOverrideRequest, *, prefix: str
) -> tuple[str, ...]:
    values = build_dataset_contract(request)
    return tuple(
        f"{prefix}.{key}={_boolean(value) if isinstance(value, bool) else value}"
        for key, value in values.items()
    )


def build_resolved_experiment(request: DreamTacTrainingRequest) -> dict[str, object]:
    """Freeze the effective UniVTAC tactile/CASA training contract."""

    checkpoint = request.resume_checkpoint or request.base_checkpoint
    dataset_contract = build_dataset_contract(request)
    return signed_payload(
        {
            "schema_version": 1,
            "protocol_id": "dream_tac_univtac_resolved_experiment_v1",
            "request_sha256": request.request_sha256,
            "phase": request.phase,
            "donor_experiment": DONOR_EXPERIMENT,
            "dataset": {
                "data_dir": str(request.materialization_root / "dataset"),
                "t5_text_embeddings_path": str(
                    request.materialization_root / "dataset" / "t5_embeddings.pkl"
                ),
                "split": "train759_only",
                "chunk_size": 20,
                "use_tactile": True,
                "tactile_streams": [
                    "tactile_rectify_left",
                    "tactile_rectify_right",
                ],
                "tactile_dropout": False,
                "use_proprio": True,
                "use_wrist_images": True,
                "use_third_person_images": True,
                "demonstration_sampling_prob": 1.0,
                "success_rollout_sampling_prob": 0.0,
                "return_value_function_returns": False,
            },
            "dataset_bindings": {
                "dataloader_train.dataset": dict(dataset_contract),
                "dataloader_train.sampler.dataset": dict(dataset_contract),
            },
            "model": {
                "state_t": 12,
                "min_num_conditional_frames": 6,
                "max_num_conditional_frames": 6,
                "chunk_duration": 45,
                "text_dropout_rate": 0.0,
                "use_tactile_self_attn_bias": True,
                "tactile_self_attn_alpha": 2.0,
                "tactile_latent_t_indices": [4, 5, 10, 11],
                "tactile_attn_chunk_q": 32,
                "tactile_self_attn_backend": "flashbias_sdpa",
            },
            "trainer": {
                "max_iter": request.max_iter,
                "run_validation": False,
            },
            "dataloader": {
                "batch_size_per_rank": request.batch_size,
                "num_workers_per_rank": request.num_workers,
                "persistent_workers": request.num_workers > 0,
            },
            "checkpoint": {
                "base_path": str(request.base_checkpoint.path),
                "base_sha256": request.base_checkpoint.sha256,
                "load_path": str(checkpoint.path),
                "load_training_state": request.resume_checkpoint is not None,
                "strict_resume": request.resume_checkpoint is not None,
                "save_iter": request.save_iter,
            },
            "distributed": {
                "nnodes": 1,
                "node_rank": 0,
                "nproc_per_node": request.nproc_per_node,
                "cuda_devices": list(request.cuda_devices),
                "master_addr": "127.0.0.1",
                "master_port": request.master_port,
            },
            "job": {
                "project": request.job_project,
                "group": request.job_group,
                "name": request.run_name,
                "wandb_mode": "disabled",
            },
        },
        field="resolved_experiment_sha256",
    )


def build_overrides(request: TrainingOverrideRequest) -> tuple[str, ...]:
    """Return exact LazyConfig overrides enforced for every P2/P3 launch."""

    checkpoint = request.resume_checkpoint or request.base_checkpoint
    values = (
        f"experiment={DONOR_EXPERIMENT}",
        *_dataset_overrides(request, prefix="dataloader_train.dataset"),
        *_dataset_overrides(request, prefix="dataloader_train.sampler.dataset"),
        f"dataloader_train.batch_size={request.batch_size}",
        f"dataloader_train.num_workers={request.num_workers}",
        f"dataloader_train.persistent_workers={_boolean(request.num_workers > 0)}",
        f"trainer.max_iter={request.max_iter}",
        "trainer.run_validation=false",
        "model.config.conditioner.text.dropout_rate=0.0",
        "model.config.state_t=12",
        "model.config.min_num_conditional_frames=6",
        "model.config.max_num_conditional_frames=6",
        "model.config.tokenizer.chunk_duration=45",
        "model.config.net.use_tactile_self_attn_bias=true",
        "model.config.net.tactile_self_attn_alpha=2.0",
        "model.config.net.tactile_latent_t_indices=[4,5,10,11]",
        "model.config.net.tactile_attn_chunk_q=32",
        f"checkpoint.load_path={checkpoint.path}",
        f"checkpoint.load_training_state={_boolean(request.resume_checkpoint is not None)}",
        f"checkpoint.strict_resume={_boolean(request.resume_checkpoint is not None)}",
        f"checkpoint.save_iter={request.save_iter}",
        "checkpoint.load_from_object_store.enabled=false",
        "checkpoint.save_to_object_store.enabled=false",
        f"job.project={request.job_project}",
        f"job.group={request.job_group}",
        f"job.name={request.run_name}",
        f"job.wandb_id={request.run_name}",
        "job.wandb_mode=disabled",
        "upload_reproducible_setup=false",
    )
    return values


def build_command(
    request: DreamTacTrainingRequest, *, mode: LaunchMode
) -> tuple[str, ...]:
    python = str(request.python_executable)
    # Upstream imports this value as a Python module path. An absolute checkout
    # path (especially one containing ``Dream-Tac``) becomes an invalid module
    # name, while every launch already runs with dream_tac_root as its cwd.
    config = "cosmos_policy/config/config.py"
    training = (
        "-m",
        "cosmos_policy.scripts.train",
        f"--config={config}",
    )
    prefix: tuple[str, ...]
    if mode == "dry_run":
        prefix = (python, *training, "--dryrun")
    else:
        prefix = (
            python,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--node_rank=0",
            f"--nproc_per_node={request.nproc_per_node}",
            "--master_addr=127.0.0.1",
            f"--master_port={request.master_port}",
            *training,
        )
    return (*prefix, "--", *build_overrides(request))


def explicit_environment(request: DreamTacTrainingRequest) -> dict[str, str]:
    """Return only deliberate environment overrides recorded in launch plans."""

    return {
        "COSMOS_TACTILE_SELF_ATTN_BACKEND": "flashbias_sdpa",
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(device) for device in request.cuda_devices
        ),
        "DREAM_TAC_BASE_CHECKPOINT": str(request.base_checkpoint.path),
        "IMAGINAIRE_OUTPUT_ROOT": str(request.output_root),
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(request.dream_tac_root),
    }


def _process_environment(request: DreamTacTrainingRequest) -> dict[str, str]:
    environment = dict(os.environ)
    explicit = explicit_environment(request)
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        explicit["PYTHONPATH"] = f"{explicit['PYTHONPATH']}:{inherited_pythonpath}"
    environment.update(explicit)
    return environment


def build_launch_plan(
    request: DreamTacTrainingRequest,
    *,
    mode: LaunchMode,
    preflight_receipt: Mapping[str, object],
    resolved_experiment: Mapping[str, object],
) -> dict[str, object]:
    return signed_payload(
        {
            "schema_version": 1,
            "status": "planned",
            "protocol_id": "dream_tac_univtac_single_node_launch_v1",
            "mode": mode,
            "request_sha256": request.request_sha256,
            "preflight_receipt_sha256": preflight_receipt["preflight_receipt_sha256"],
            "resolved_experiment_sha256": resolved_experiment[
                "resolved_experiment_sha256"
            ],
            "command": list(build_command(request, mode=mode)),
            "environment": explicit_environment(request),
            "cwd": str(request.dream_tac_root),
            "log_path": str(request.receipt_root / f"{mode}.log"),
        },
        field="launch_plan_sha256",
    )


def _existing_result(
    path: Path, *, request: DreamTacTrainingRequest, mode: LaunchMode
) -> dict[str, object] | None:
    if not path.exists():
        return None
    result = load_json_object(path)
    receipt_sha256(result, field="launch_result_sha256")
    if result.get("request_sha256") != request.request_sha256:
        raise ValueError("existing Dream-Tac launch result request mismatch")
    if result.get("mode") != mode:
        raise ValueError("existing Dream-Tac launch result mode mismatch")
    return result


def execute(request: DreamTacTrainingRequest, *, mode: LaunchMode) -> int:
    """Preflight once, launch once, and refuse ambiguous duplicate execution."""

    from .training_preflight import write_preflight_receipt

    _, preflight = write_preflight_receipt(request)
    resolved = build_resolved_experiment(request)
    resolved_path = request.receipt_root / RESOLVED_EXPERIMENT_NAME
    write_or_verify_json(resolved_path, resolved)
    plan = build_launch_plan(
        request,
        mode=mode,
        preflight_receipt=preflight,
        resolved_experiment=resolved,
    )
    plan_path = request.receipt_root / f"{mode}_launch_plan.json"
    result_path = request.receipt_root / f"{mode}_launch_result.json"
    existing = _existing_result(result_path, request=request, mode=mode)
    if existing is not None:
        return int(cast(int, existing["returncode"]))
    if plan_path.exists():
        raise RuntimeError(
            "launch plan exists without a result; inspect the process/log and use "
            "an explicit content-addressed resume request instead of relaunching"
        )
    write_or_verify_json(plan_path, plan)
    log_path = request.receipt_root / f"{mode}.log"
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite launch log: {log_path}")
    command = build_command(request, mode=mode)
    with log_path.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(
            list(command),
            cwd=request.dream_tac_root,
            env=_process_environment(request),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    from .training_request import sha256_regular_file

    result = signed_payload(
        {
            "schema_version": 1,
            "status": "completed" if completed.returncode == 0 else "failed",
            "protocol_id": "dream_tac_univtac_single_node_launch_result_v1",
            "mode": mode,
            "request_sha256": request.request_sha256,
            "launch_plan_sha256": plan["launch_plan_sha256"],
            "returncode": completed.returncode,
            "log_path": str(log_path),
            "log_sha256": sha256_regular_file(log_path),
        },
        field="launch_result_sha256",
    )
    write_or_verify_json(result_path, result)
    return completed.returncode


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--print-command", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    request = DreamTacTrainingRequest.load(args.request)
    if args.print_command:
        print(
            json.dumps(
                {
                    "command": list(build_command(request, mode="train")),
                    "environment": explicit_environment(request),
                    "gates_checked": False,
                    "writes_performed": False,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.preflight_only:
        from .training_preflight import write_preflight_receipt

        path, receipt = write_preflight_receipt(request)
        print(
            json.dumps(
                {"receipt": str(path), "status": receipt["status"]},
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return execute(request, mode="dry_run" if args.dry_run else "train")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_command",
    "build_dataset_contract",
    "build_launch_plan",
    "build_overrides",
    "build_resolved_experiment",
    "execute",
    "explicit_environment",
    "main",
]

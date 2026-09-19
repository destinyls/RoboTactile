"""Preflight, data preparation, normalization, and task training stages."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

from .artifacts import numeric_checkpoints, verify_signed
from .commands import (
    checkpoint_root,
    dataset_config,
    norm_command,
    parser_command,
    repository_id,
    rgb_contract_command,
    task_zarr_path,
    training_command,
)
from .contracts import (
    EXPECTED_TRAIN_COUNTS,
    FTP1_SOURCE_COMMIT,
    TASK_IDS,
    TRAINING_SCOPE,
    FTP1TrainingRequest,
    canonical_json_sha256,
    source_manifest_payload,
    validate_source_manifest,
)
from .runtime import (
    command_output,
    explicit_environment,
    run_logged,
    sha256_file,
    signed,
    tree_inventory,
    write_or_verify,
)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def preflight(request: FTP1TrainingRequest) -> dict[str, object]:
    root = request.ftp1_root.resolve(strict=True)
    if _git_output(root, "rev-parse", "HEAD") != FTP1_SOURCE_COMMIT:
        raise ValueError("FTP-1 checkout does not match the pinned commit")
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("FTP-1 checkout must remain clean")
    entrypoints = (
        root / "scripts" / "zarr_train_ftp1_pytorch.py",
        root / "scripts" / "zarr_compute_norm_stats.py",
        root / "data_processing" / "parse_data_module" / "parse_data_univtac.py",
    )
    for path in entrypoints:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"FTP-1 entrypoint is unavailable: {path}")
    robotactile_entrypoints = tuple(
        Path(__file__).with_name(name).resolve(strict=True)
        for name in (
            "rgb_parser_entrypoint.py",
            "rgb_contract.py",
            "norm_entrypoint.py",
        )
    )
    python = request.python_executable.resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("FTP-1 Python executable is invalid")
    manifest = source_manifest_payload(request.source_manifest_path)
    records = validate_source_manifest(manifest)
    model = request.base_checkpoint_dir / "model.safetensors"
    if model.is_symlink() or not model.is_file():
        raise ValueError("pinned FTP-1 pretrained model.safetensors is unavailable")
    expected_openpi_root = (root / "src" / "openpi").resolve()
    probe_script = "\n".join(
        (
            "import importlib.util,json,pathlib,torch",
            "import openpi",
            "required=('cv2','h5py','openpi','tyro','zarr')",
            "missing=[name for name in required if importlib.util.find_spec(name) is None]",
            "assert not missing, f'missing modules: {missing}'",
            f"expected_openpi_root=pathlib.Path({str(expected_openpi_root)!r})",
            "actual_openpi=pathlib.Path(openpi.__file__).resolve()",
            "assert actual_openpi.is_relative_to(expected_openpi_root), f'openpi is not source-bound: {actual_openpi}'",
            "assert torch.version.cuda is not None and torch.cuda.is_available()",
            "assert torch.cuda.device_count() == 1",
            "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))},sort_keys=True))",
        )
    )
    probe = json.loads(
        command_output(
            (str(python), "-c", probe_script),
            cwd=root,
            env=explicit_environment(request),
        )
    )
    payload = signed(
        {
            "schema_version": "robotactile-ftp1-training-preflight-v1",
            "status": "passed",
            "request_sha256": request.request_sha256,
            "ftp1_source_commit": FTP1_SOURCE_COMMIT,
            "entrypoints": {
                str(path.relative_to(root)): sha256_file(path) for path in entrypoints
            },
            "robotactile_entrypoints": {
                path.name: sha256_file(path) for path in robotactile_entrypoints
            },
            "source_manifest_sha256": manifest["manifest_sha256"],
            "train_episode_count": len(records),
            "task_count": len(TASK_IDS),
            "base_model_path": str(model),
            "base_model_size_bytes": model.stat().st_size,
            "base_model_sha256": sha256_file(model),
            "vision_tactile_policy": "vision_plus_two_tactile_always_on_v1",
            "environment": probe,
        },
        "preflight_receipt_sha256",
    )
    write_or_verify(request.output_root / "preflight_receipt.json", payload)
    return payload


def stage_sources(request: FTP1TrainingRequest, *, verify_hashes: bool) -> None:
    manifest = source_manifest_payload(request.source_manifest_path)
    records = validate_source_manifest(manifest)
    receipt_path = request.output_root / "dataset" / "staging_receipt.json"
    receipt_exists = receipt_path.exists()
    total_bytes = 0
    counts = {task: 0 for task in TASK_IDS}
    for item in records:
        relative = str(item["relative_path"])
        task = str(item["task"])
        source = request.raw_root / relative
        metadata = source.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"raw source must be a regular file: {source}")
        if metadata.st_size != item.get("size_bytes"):
            raise ValueError(f"raw source size changed: {source}")
        if (
            verify_hashes
            and not receipt_exists
            and sha256_file(source) != item.get("sha256")
        ):
            raise ValueError(f"raw source SHA256 changed: {source}")
        destination = request.staging_root / task / "demo" / "hdf5" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if (
                not destination.is_symlink()
                or destination.resolve(strict=True) != source.resolve()
            ):
                raise FileExistsError(f"staging path differs: {destination}")
        else:
            destination.symlink_to(source.resolve(strict=True))
        counts[task] += 1
        total_bytes += metadata.st_size
    if counts != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"staged train759 counts are invalid: {counts}")
    payload = signed(
        {
            "schema_version": "robotactile-ftp1-train759-staging-v1",
            "status": "complete",
            "request_sha256": request.request_sha256,
            "source_manifest_sha256": manifest["manifest_sha256"],
            "source_split": "train",
            "episode_count": len(records),
            "task_counts": counts,
            "source_bytes": total_bytes,
            "source_sha256_validated": verify_hashes or receipt_exists,
            "staging_root": str(request.staging_root),
        },
        "staging_receipt_sha256",
    )
    write_or_verify(receipt_path, payload)


def _zarr_episode_count(request: FTP1TrainingRequest, path: Path) -> int:
    script = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(request.ftp1_root / 'data_processing')!r})",
            "from common.replay_buffer import ReplayBuffer",
            f"print(ReplayBuffer.create_from_path({str(path)!r}, mode='r').n_episodes)",
        )
    )
    raw = command_output(
        (str(request.python_executable), "-c", script),
        cwd=request.ftp1_root,
        env=explicit_environment(request),
    )
    return int(raw.splitlines()[-1])


def parse_task(request: FTP1TrainingRequest, task: str) -> None:
    receipt = request.output_root / "dataset" / task / "conversion_receipt.json"
    rgb_receipt = request.output_root / "dataset" / task / "rgb_contract_receipt.json"
    zarr_path = task_zarr_path(request, task)
    if receipt.exists():
        if (
            not rgb_receipt.is_file()
            or not zarr_path.is_dir()
            or _zarr_episode_count(request, zarr_path) != EXPECTED_TRAIN_COUNTS[task]
        ):
            raise ValueError(
                f"existing {task} conversion receipt is not backed by train data"
            )
        verify_signed(receipt, "conversion_receipt_sha256")
        verified_rgb = verify_signed(rgb_receipt, "rgb_contract_receipt_sha256")
        if verified_rgb.get("status") != "passed":
            raise ValueError(f"existing {task} Zarr is not RGB-certified")
        return
    if zarr_path.exists():
        raise FileExistsError(f"refusing unreceipted zarr output: {zarr_path}")
    run_logged(
        parser_command(request, task),
        cwd=request.ftp1_root / "data_processing",
        env=explicit_environment(request),
        log_path=request.output_root / "logs" / f"parse-{task}.log",
    )
    count = _zarr_episode_count(request, zarr_path)
    if count != EXPECTED_TRAIN_COUNTS[task]:
        raise ValueError(f"{task} zarr contains {count} episodes")
    rgb_output = command_output(
        rgb_contract_command(request, task, zarr_path),
        cwd=request.ftp1_root,
        env=explicit_environment(request),
    )
    rgb_verification = json.loads(rgb_output.splitlines()[-1])
    if (
        not isinstance(rgb_verification, dict)
        or rgb_verification.get("status") != "passed"
    ):
        raise ValueError(f"{task} Zarr failed the RGB contract")
    rgb_payload = signed(
        {
            "schema_version": "robotactile-ftp1-rgb-contract-v1",
            "request_sha256": request.request_sha256,
            "zarr_path": str(zarr_path),
            "verification_command": list(
                rgb_contract_command(request, task, zarr_path)
            ),
            **rgb_verification,
        },
        "rgb_contract_receipt_sha256",
    )
    write_or_verify(rgb_receipt, rgb_payload)
    payload = signed(
        {
            "schema_version": "robotactile-ftp1-zarr-conversion-v2",
            "status": "complete",
            "request_sha256": request.request_sha256,
            "task_id": task,
            "episode_count": count,
            "zarr_path": str(zarr_path),
            "parser_command": list(parser_command(request, task)),
            "color_contract": "canonical_rgb_exact_v1",
            "rgb_contract_receipt_sha256": rgb_payload["rgb_contract_receipt_sha256"],
        },
        "conversion_receipt_sha256",
    )
    write_or_verify(receipt, payload)


def _require_conversions(
    request: FTP1TrainingRequest,
) -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    for task in TASK_IDS:
        receipt_path = (
            request.output_root / "dataset" / task / "conversion_receipt.json"
        )
        rgb_path = request.output_root / "dataset" / task / "rgb_contract_receipt.json"
        if not receipt_path.is_file() or not rgb_path.is_file():
            raise ValueError(f"{task} is not prepared and RGB-certified")
        receipt = verify_signed(receipt_path, "conversion_receipt_sha256")
        rgb_receipt = verify_signed(rgb_path, "rgb_contract_receipt_sha256")
        if (
            receipt.get("request_sha256") != request.request_sha256
            or receipt.get("episode_count") != EXPECTED_TRAIN_COUNTS[task]
            or receipt.get("color_contract") != "canonical_rgb_exact_v1"
            or rgb_receipt.get("status") != "passed"
        ):
            raise ValueError(f"{task} conversion does not satisfy the joint contract")
        zarr_path = task_zarr_path(request, task)
        if _zarr_episode_count(request, zarr_path) != EXPECTED_TRAIN_COUNTS[task]:
            raise ValueError(f"{task} Zarr episode count changed")
        receipts[task] = receipt
    return receipts


def normalize_joint(request: FTP1TrainingRequest) -> Path:
    conversions = _require_conversions(request)
    config_path = request.output_root / "configs" / "joint_all8.json"
    config = dataset_config(request)
    write_or_verify(config_path, config)
    receipt = request.output_root / "joint" / "normalization_receipt.json"
    if receipt.exists():
        verify_signed(receipt, "normalization_receipt_sha256")
        return config_path
    run_logged(
        norm_command(request, config_path),
        cwd=request.ftp1_root,
        env=explicit_environment(request),
        log_path=request.output_root / "logs" / "norm-joint-all8.log",
    )
    assets = request.assets_base_dir / "ftp1" / repository_id()
    payload = signed(
        {
            "schema_version": "robotactile-ftp1-joint-normalization-v1",
            "status": "complete",
            "request_sha256": request.request_sha256,
            "training_scope": TRAINING_SCOPE,
            "task_ids": list(TASK_IDS),
            "source_split": "train",
            "episode_count": sum(EXPECTED_TRAIN_COUNTS.values()),
            "task_episode_counts": EXPECTED_TRAIN_COUNTS,
            "dataset_config_sha256": canonical_json_sha256(config),
            "conversion_receipt_sha256": {
                task: conversions[task]["conversion_receipt_sha256"]
                for task in TASK_IDS
            },
            "assets_root": str(assets),
            "files": tree_inventory(assets),
            "command": list(norm_command(request, config_path)),
        },
        "normalization_receipt_sha256",
    )
    write_or_verify(receipt, payload)
    return config_path


def train_joint(request: FTP1TrainingRequest, config_path: Path) -> None:
    conversions = _require_conversions(request)
    normalization_path = request.output_root / "joint" / "normalization_receipt.json"
    if not normalization_path.is_file():
        raise ValueError("joint normalization receipt is absent; run prepare first")
    normalization = verify_signed(normalization_path, "normalization_receipt_sha256")
    receipt = request.output_root / "joint" / "training_receipt.json"
    if receipt.exists():
        verify_signed(receipt, "training_receipt_sha256")
        return
    root = checkpoint_root(request)
    checkpoints = numeric_checkpoints(root)
    expected_step = request.num_train_steps - 1
    if not checkpoints or checkpoints[-1][0] < expected_step:
        command = training_command(request, config_path, resume=bool(checkpoints))
        run_logged(
            command,
            cwd=request.ftp1_root,
            env=explicit_environment(request),
            log_path=request.output_root / "logs" / "train-joint-all8.log",
        )
        checkpoints = numeric_checkpoints(root)
    if not checkpoints or checkpoints[-1][0] < expected_step:
        raise ValueError("joint all-8 training did not produce the final checkpoint")
    step, final = checkpoints[-1]
    payload = signed(
        {
            "schema_version": "robotactile-ftp1-joint-training-v1",
            "status": "complete",
            "request_sha256": request.request_sha256,
            "training_scope": TRAINING_SCOPE,
            "joint_model": True,
            "checkpoint_count": 1,
            "task_ids": list(TASK_IDS),
            "source_split": "train",
            "episode_count": sum(EXPECTED_TRAIN_COUNTS.values()),
            "task_episode_counts": EXPECTED_TRAIN_COUNTS,
            "conversion_receipt_sha256": {
                task: conversions[task]["conversion_receipt_sha256"]
                for task in TASK_IDS
            },
            "normalization_receipt_sha256": normalization[
                "normalization_receipt_sha256"
            ],
            "optimizer_step": step,
            "checkpoint_root": str(root),
            "final_checkpoint": str(final),
            "files": tree_inventory(final),
            "vision_tactile_policy": "vision_plus_two_tactile_always_on_v1",
        },
        "training_receipt_sha256",
    )
    write_or_verify(receipt, payload)


__all__ = [
    "normalize_joint",
    "parse_task",
    "preflight",
    "stage_sources",
    "train_joint",
]

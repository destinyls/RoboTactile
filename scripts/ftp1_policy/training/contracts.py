"""Immutable contracts for one joint eight-task FTP-1 UniVTAC run."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, cast

REQUEST_SCHEMA: Final[str] = "robotactile-ftp1-train759-joint-request-v1"
ACCELERATOR_CONTRACT: Final[str] = "nvidia_cuda_single_gpu_v1"
VISION_TACTILE_POLICY: Final[str] = "vision_plus_two_tactile_always_on_v1"
TRAINING_SCOPE: Final[str] = "single_joint_checkpoint_all8_v1"
FTP1_SOURCE_COMMIT: Final[str] = "89fa681d6c014cce28300946b7526db808e0b1c1"
FTP1_BASE_REVISION: Final[str] = "d6e5b73e473d3e70fb5f53132a2b1c35b5031156"
UNIVTAC_DATASET_REVISION: Final[str] = "3d4646a7bc19214830f74ade2818b99117e3720f"
SOURCE_MANIFEST_PROTOCOL: Final[str] = "univtac_train759_absee20_v1"
TASK_IDS: Final[tuple[str, ...]] = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
EXPECTED_TRAIN_COUNTS: Final[dict[str, int]] = {
    task: 94 if task == "grasp_classify" else 95 for task in TASK_IDS
}
TASK_CAMERA_MODE: Final[dict[str, str]] = {
    task: "head+wrist" if task in {"insert_tube", "lift_can"} else "head"
    for task in TASK_IDS
}
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{2,95}")
_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accelerator_contract",
        "assets_base_dir",
        "base_checkpoint_dir",
        "base_checkpoint_revision",
        "batch_size",
        "checkpoint_base_dir",
        "cuda_device",
        "dataset_revision",
        "ftp1_root",
        "ftp1_source_commit",
        "gradient_checkpointing",
        "keep_period",
        "log_interval",
        "num_train_steps",
        "num_workers",
        "openpi_data_home",
        "output_root",
        "python_executable",
        "raw_root",
        "run_id",
        "save_interval",
        "schema_version",
        "seed",
        "source_manifest_path",
        "staging_root",
        "tasks",
        "training_scope",
        "use_torch_compile",
        "val_interval",
        "val_num_workers",
        "vision_tactile_policy",
        "wandb_enabled",
        "zarr_root",
    }
)


def canonical_json_sha256(payload: object) -> str:
    """Return the stable digest used by requests and receipts."""

    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class FTP1TrainingRequest:
    """One source-pinned joint model trained on all eight UniVTAC tasks."""

    run_id: str
    ftp1_root: Path
    python_executable: Path
    source_manifest_path: Path
    raw_root: Path
    staging_root: Path
    zarr_root: Path
    base_checkpoint_dir: Path
    assets_base_dir: Path
    checkpoint_base_dir: Path
    output_root: Path
    openpi_data_home: Path
    tasks: tuple[str, ...]
    cuda_device: int
    seed: int
    num_train_steps: int
    batch_size: int
    num_workers: int
    val_num_workers: int
    log_interval: int
    val_interval: int
    save_interval: int
    keep_period: int
    use_torch_compile: bool
    gradient_checkpointing: bool
    wandb_enabled: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FTP1TrainingRequest:
        if set(value) != _REQUEST_FIELDS:
            missing = sorted(_REQUEST_FIELDS - set(value))
            extra = sorted(set(value) - _REQUEST_FIELDS)
            raise ValueError(
                f"FTP-1 request fields mismatch: missing={missing}, extra={extra}"
            )
        fixed = {
            "schema_version": REQUEST_SCHEMA,
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "ftp1_source_commit": FTP1_SOURCE_COMMIT,
            "base_checkpoint_revision": FTP1_BASE_REVISION,
            "dataset_revision": UNIVTAC_DATASET_REVISION,
            "vision_tactile_policy": VISION_TACTILE_POLICY,
            "training_scope": TRAINING_SCOPE,
        }
        for field, expected in fixed.items():
            if value[field] != expected:
                raise ValueError(f"{field} must equal the pinned value {expected}")
        run_id = value["run_id"]
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a safe lowercase identifier")
        raw_tasks = value["tasks"]
        if not isinstance(raw_tasks, list) or tuple(raw_tasks) != TASK_IDS:
            raise ValueError(
                "formal FTP-1 training must contain the ordered eight tasks"
            )
        booleans = (
            "use_torch_compile",
            "gradient_checkpointing",
            "wandb_enabled",
        )
        if any(not isinstance(value[name], bool) for name in booleans):
            raise ValueError("FTP-1 boolean request fields must be bool")
        request = cls(
            run_id=run_id,
            ftp1_root=_absolute_path(value["ftp1_root"], "ftp1_root"),
            python_executable=_absolute_path(
                value["python_executable"], "python_executable"
            ),
            source_manifest_path=_absolute_path(
                value["source_manifest_path"], "source_manifest_path"
            ),
            raw_root=_absolute_path(value["raw_root"], "raw_root"),
            staging_root=_absolute_path(value["staging_root"], "staging_root"),
            zarr_root=_absolute_path(value["zarr_root"], "zarr_root"),
            base_checkpoint_dir=_absolute_path(
                value["base_checkpoint_dir"], "base_checkpoint_dir"
            ),
            assets_base_dir=_absolute_path(value["assets_base_dir"], "assets_base_dir"),
            checkpoint_base_dir=_absolute_path(
                value["checkpoint_base_dir"], "checkpoint_base_dir"
            ),
            output_root=_absolute_path(value["output_root"], "output_root"),
            openpi_data_home=_absolute_path(
                value["openpi_data_home"], "openpi_data_home"
            ),
            tasks=TASK_IDS,
            cuda_device=_integer(value["cuda_device"], "cuda_device", 0, 255),
            seed=_integer(value["seed"], "seed", 0, 2**31 - 1),
            num_train_steps=_integer(
                value["num_train_steps"], "num_train_steps", 1, 1_000_000
            ),
            batch_size=_integer(value["batch_size"], "batch_size", 1, 256),
            num_workers=_integer(value["num_workers"], "num_workers", 0, 64),
            val_num_workers=_integer(
                value["val_num_workers"], "val_num_workers", 0, 64
            ),
            log_interval=_integer(value["log_interval"], "log_interval", 1, 100_000),
            val_interval=_integer(value["val_interval"], "val_interval", 1, 100_000),
            save_interval=_integer(value["save_interval"], "save_interval", 1, 100_000),
            keep_period=_integer(value["keep_period"], "keep_period", 1, 1_000_000),
            use_torch_compile=cast(bool, value["use_torch_compile"]),
            gradient_checkpointing=cast(bool, value["gradient_checkpointing"]),
            wandb_enabled=cast(bool, value["wandb_enabled"]),
        )
        request._validate_paths()
        return request

    @classmethod
    def load(cls, path: Path) -> FTP1TrainingRequest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("FTP-1 training request must be a JSON object")
        return cls.from_dict(payload)

    def _validate_paths(self) -> None:
        inputs = {
            self.ftp1_root,
            self.raw_root,
            self.base_checkpoint_dir,
            self.source_manifest_path,
        }
        outputs = {
            self.staging_root,
            self.zarr_root,
            self.assets_base_dir,
            self.checkpoint_base_dir,
            self.output_root,
        }
        if inputs & outputs:
            raise ValueError("FTP-1 input and output paths must be disjoint")
        if (
            self.raw_root in self.output_root.parents
            or self.output_root in self.raw_root.parents
        ):
            raise ValueError("output_root and raw_root must not contain each other")

    @property
    def request_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "assets_base_dir": str(self.assets_base_dir),
            "base_checkpoint_dir": str(self.base_checkpoint_dir),
            "base_checkpoint_revision": FTP1_BASE_REVISION,
            "batch_size": self.batch_size,
            "checkpoint_base_dir": str(self.checkpoint_base_dir),
            "cuda_device": self.cuda_device,
            "dataset_revision": UNIVTAC_DATASET_REVISION,
            "ftp1_root": str(self.ftp1_root),
            "ftp1_source_commit": FTP1_SOURCE_COMMIT,
            "gradient_checkpointing": self.gradient_checkpointing,
            "keep_period": self.keep_period,
            "log_interval": self.log_interval,
            "num_train_steps": self.num_train_steps,
            "num_workers": self.num_workers,
            "openpi_data_home": str(self.openpi_data_home),
            "output_root": str(self.output_root),
            "python_executable": str(self.python_executable),
            "raw_root": str(self.raw_root),
            "run_id": self.run_id,
            "save_interval": self.save_interval,
            "schema_version": REQUEST_SCHEMA,
            "seed": self.seed,
            "source_manifest_path": str(self.source_manifest_path),
            "staging_root": str(self.staging_root),
            "tasks": list(self.tasks),
            "training_scope": TRAINING_SCOPE,
            "use_torch_compile": self.use_torch_compile,
            "val_interval": self.val_interval,
            "val_num_workers": self.val_num_workers,
            "vision_tactile_policy": VISION_TACTILE_POLICY,
            "wandb_enabled": self.wandb_enabled,
            "zarr_root": str(self.zarr_root),
        }


def validate_source_manifest(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Return the exact train759 records from a canonical 800-episode manifest."""

    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != canonical_json_sha256(unsigned):
        raise ValueError("UniVTAC source manifest digest is invalid")
    if payload.get("protocol_id") != SOURCE_MANIFEST_PROTOCOL:
        raise ValueError("unsupported UniVTAC source manifest protocol")
    if payload.get("split_counts") != {"frozen": 40, "quarantine": 1, "train": 759}:
        raise ValueError("UniVTAC split counts must be frozen40/quarantine1/train759")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 800:
        raise ValueError("UniVTAC source manifest must contain 800 episodes")
    selected: list[dict[str, object]] = []
    paths: set[str] = set()
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("UniVTAC manifest episode must be an object")
        relative = item.get("relative_path")
        task = item.get("task")
        if not isinstance(relative, str) or relative in paths:
            raise ValueError("UniVTAC manifest paths must be unique strings")
        if task not in TASK_IDS:
            raise ValueError(f"unsupported UniVTAC task in manifest: {task}")
        paths.add(relative)
        if item.get("split") == "train":
            selected.append(dict(item))
    counts = {
        task: sum(item.get("task") == task for item in selected) for task in TASK_IDS
    }
    if counts != EXPECTED_TRAIN_COUNTS or len(selected) != 759:
        raise ValueError(f"per-task train759 counts are invalid: {counts}")
    return tuple(selected)


def source_manifest_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest must be a JSON object")
    validate_source_manifest(payload)
    return payload


__all__ = [
    "ACCELERATOR_CONTRACT",
    "EXPECTED_TRAIN_COUNTS",
    "FTP1_BASE_REVISION",
    "FTP1_SOURCE_COMMIT",
    "FTP1TrainingRequest",
    "REQUEST_SCHEMA",
    "SOURCE_MANIFEST_PROTOCOL",
    "TASK_CAMERA_MODE",
    "TASK_IDS",
    "TRAINING_SCOPE",
    "UNIVTAC_DATASET_REVISION",
    "VISION_TACTILE_POLICY",
    "canonical_json_sha256",
    "source_manifest_payload",
    "validate_source_manifest",
]

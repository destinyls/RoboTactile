"""Content-addressed artifacts for a user-supplied Dream-Tac checkpoint."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.integrations.dream_tac.artifact_files import (
    DreamTacArtifactFile,
    absolute_path,
    checkpoint_file_hashes,
    require_below,
    stable_file_sha256,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock
from robotactile_benchmark.policies.dream_tac import (
    DreamTacGripperMapping,
    dream_tac_gripper_qpos_bounds,
)

SCHEMA_VERSION: Final = "robotactile-dream-tac-artifact-v1"
SERVER_PROTOCOL: Final = "dream-tac-franka-http-v1"
CASA_INFERENCE_CONTRACT: Final = "upstream_http_gate_missing_v1"
WEIGHT_RELEASE_STATUS: Final = "user_supplied_unverified"
ACTION_CONVERSION: Final = "absolute_xyz_rpy_gripper_to_ee8_wxyz_v1"
ACTION_EXECUTION: Final = "full_20_step_chunk_then_reinfer_v1"
CAMERA_MAPPING: Final = "top_cam_front__wrist_l_cam_high_v1"
COLOR_CONTRACT: Final = "uint8_hwc_rgb_png_v1"
PROPRIO_CONTRACT: Final = "ee8_wxyz_to_xyz_rpy_v1"
TACTILE_CONTRACT: Final = "delivered_left_right_rgb_required_v1"
QUATERNION_ORDER: Final = "wxyz"
ACTION_HORIZON: Final = 20
RAW_ACTION_DIM: Final = 7
BENCHMARK_ACTION_DIM: Final = 8
STATE_DIM: Final = 6
IMAGE_SIZE: Final = 224
USE_TACTILE: Final = True

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_REQUIRED_FIELDS = frozenset(
    [
        "action_horizon",
        "action_conversion",
        "action_execution",
        "benchmark_action_dim",
        "bundle_root",
        "casa_inference_contract",
        "camera_mapping",
        "checkpoint_files",
        "checkpoint_root",
        "checkpoint_tree_sha256",
        "control_hz",
        "color_contract",
        "dataset_stats_path",
        "dataset_stats_sha256",
        "experiment_config",
        "external_commit",
        "gripper_mapping",
        "gripper_threshold",
        "image_size",
        "instruction",
        "raw_action_dim",
        "proprio_contract",
        "quaternion_order",
        "schema_version",
        "server_protocol",
        "state_dim",
        "t5_embeddings_path",
        "t5_embeddings_sha256",
        "tactile_contract",
        "task_id",
        "use_tactile",
        "weight_release_status",
    ]
)
_OPTIONAL_FIELDS = frozenset({"gripper_qpos_max", "gripper_qpos_min"})
_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _tree_sha256(files: tuple[DreamTacArtifactFile, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([item.to_dict() for item in files])
    ).hexdigest()


def _checkpoint_inventory(root: Path) -> tuple[DreamTacArtifactFile, ...]:
    return tuple(
        DreamTacArtifactFile(path, sha256)
        for path, sha256 in checkpoint_file_hashes(root)
    )


@dataclass(frozen=True)
class DreamTacArtifactManifest:
    """Exact source, files, prompt, timing, and action conversion contract."""

    task_id: str
    instruction: str
    experiment_config: str
    bundle_root: Path
    checkpoint_root: Path
    checkpoint_files: tuple[DreamTacArtifactFile, ...]
    checkpoint_tree_sha256: str
    dataset_stats_path: Path
    dataset_stats_sha256: str
    t5_embeddings_path: Path
    t5_embeddings_sha256: str
    external_commit: str
    control_hz: float
    gripper_mapping: str
    gripper_threshold: float
    gripper_qpos_min: float | None = None
    gripper_qpos_max: float | None = None
    action_conversion: str = ACTION_CONVERSION
    action_execution: str = ACTION_EXECUTION
    camera_mapping: str = CAMERA_MAPPING
    color_contract: str = COLOR_CONTRACT
    proprio_contract: str = PROPRIO_CONTRACT
    tactile_contract: str = TACTILE_CONTRACT
    quaternion_order: str = QUATERNION_ORDER
    action_horizon: int = ACTION_HORIZON
    raw_action_dim: int = RAW_ACTION_DIM
    benchmark_action_dim: int = BENCHMARK_ACTION_DIM
    state_dim: int = STATE_DIM
    image_size: int = IMAGE_SIZE
    use_tactile: bool = USE_TACTILE
    server_protocol: str = SERVER_PROTOCOL
    casa_inference_contract: str = CASA_INFERENCE_CONTRACT
    weight_release_status: str = WEIGHT_RELEASE_STATUS
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("task_id", "instruction", "experiment_config"):
            _string(getattr(self, name), name)
        root = absolute_path(self.bundle_root, "bundle_root")
        checkpoint = absolute_path(self.checkpoint_root, "checkpoint_root")
        stats = absolute_path(self.dataset_stats_path, "dataset_stats_path")
        embeddings = absolute_path(self.t5_embeddings_path, "t5_embeddings_path")
        for path, name in (
            (checkpoint, "checkpoint_root"),
            (stats, "dataset_stats_path"),
            (embeddings, "t5_embeddings_path"),
        ):
            require_below(root, path, name)
        files = tuple(self.checkpoint_files)
        if not files or tuple(item.path for item in files) != tuple(
            sorted(item.path for item in files)
        ):
            raise ValueError("checkpoint_files must be non-empty and sorted")
        if len({item.path for item in files}) != len(files):
            raise ValueError("checkpoint_files paths must be unique")
        if self.checkpoint_tree_sha256 != _tree_sha256(files):
            raise ValueError("Dream-Tac checkpoint tree identity mismatch")
        expected_commit = load_integration_lock().by_id("dream_tac").commit_sha
        if (
            self.external_commit != expected_commit
            or _COMMIT.fullmatch(expected_commit) is None
        ):
            raise ValueError("Dream-Tac external source commit mismatch")
        mapping = DreamTacGripperMapping(self.gripper_mapping)
        if isinstance(self.control_hz, bool) or not isinstance(
            self.control_hz, (int, float)
        ):
            raise TypeError("control_hz must be a real number")
        if not math.isfinite(float(self.control_hz)) or float(self.control_hz) <= 0.0:
            raise ValueError("control_hz must be positive and finite")
        if isinstance(self.gripper_threshold, bool) or not isinstance(
            self.gripper_threshold, (int, float)
        ):
            raise TypeError("gripper_threshold must be a real number")
        if not math.isfinite(float(self.gripper_threshold)):
            raise ValueError("gripper_threshold must be finite")
        qpos_bounds = dream_tac_gripper_qpos_bounds(
            mapping,
            gripper_qpos_min=self.gripper_qpos_min,
            gripper_qpos_max=self.gripper_qpos_max,
        )
        fixed = {
            "schema_version": (self.schema_version, SCHEMA_VERSION),
            "action_conversion": (self.action_conversion, ACTION_CONVERSION),
            "action_execution": (self.action_execution, ACTION_EXECUTION),
            "camera_mapping": (self.camera_mapping, CAMERA_MAPPING),
            "color_contract": (self.color_contract, COLOR_CONTRACT),
            "proprio_contract": (self.proprio_contract, PROPRIO_CONTRACT),
            "tactile_contract": (self.tactile_contract, TACTILE_CONTRACT),
            "quaternion_order": (self.quaternion_order, QUATERNION_ORDER),
            "action_horizon": (self.action_horizon, ACTION_HORIZON),
            "raw_action_dim": (self.raw_action_dim, RAW_ACTION_DIM),
            "benchmark_action_dim": (
                self.benchmark_action_dim,
                BENCHMARK_ACTION_DIM,
            ),
            "state_dim": (self.state_dim, STATE_DIM),
            "image_size": (self.image_size, IMAGE_SIZE),
            "use_tactile": (self.use_tactile, USE_TACTILE),
            "server_protocol": (self.server_protocol, SERVER_PROTOCOL),
            "casa_inference_contract": (
                self.casa_inference_contract,
                CASA_INFERENCE_CONTRACT,
            ),
            "weight_release_status": (
                self.weight_release_status,
                WEIGHT_RELEASE_STATUS,
            ),
        }
        if any(actual != wanted for actual, wanted in fixed.values()):
            raise ValueError("Dream-Tac fixed runtime contract mismatch")
        for name in (
            "checkpoint_tree_sha256",
            "dataset_stats_sha256",
            "t5_embeddings_sha256",
        ):
            _sha256(getattr(self, name), name)
        object.__setattr__(self, "bundle_root", root)
        object.__setattr__(self, "checkpoint_root", checkpoint)
        object.__setattr__(self, "dataset_stats_path", stats)
        object.__setattr__(self, "t5_embeddings_path", embeddings)
        object.__setattr__(self, "checkpoint_files", files)
        object.__setattr__(self, "control_hz", float(self.control_hz))
        object.__setattr__(self, "gripper_mapping", mapping.value)
        object.__setattr__(self, "gripper_threshold", float(self.gripper_threshold))
        if qpos_bounds is not None:
            object.__setattr__(self, "gripper_qpos_min", qpos_bounds[0])
            object.__setattr__(self, "gripper_qpos_max", qpos_bounds[1])

    @property
    def checkpoint_sha256(self) -> str:
        return self.checkpoint_tree_sha256

    @property
    def normalizer_sha256(self) -> str:
        return self.dataset_stats_sha256

    @property
    def config_sha256(self) -> str:
        identity = {
            "action_conversion": self.action_conversion,
            "action_execution": self.action_execution,
            "action_horizon": self.action_horizon,
            "casa_inference_contract": self.casa_inference_contract,
            "camera_mapping": self.camera_mapping,
            "color_contract": self.color_contract,
            "control_hz": self.control_hz,
            "experiment_config": self.experiment_config,
            "gripper_mapping": self.gripper_mapping,
            "gripper_threshold": self.gripper_threshold,
            "image_size": self.image_size,
            "instruction": self.instruction,
            "proprio_contract": self.proprio_contract,
            "quaternion_order": self.quaternion_order,
            "server_protocol": self.server_protocol,
            "state_dim": self.state_dim,
            "tactile_contract": self.tactile_contract,
            "task_id": self.task_id,
            "use_tactile": self.use_tactile,
        }
        if self.gripper_qpos_min is not None:
            identity["gripper_qpos_min"] = self.gripper_qpos_min
            identity["gripper_qpos_max"] = self.gripper_qpos_max
        return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()

    @property
    def serve_bundle_sha256(self) -> str:
        identity = self.to_dict()
        for name in (
            "bundle_root",
            "checkpoint_root",
            "dataset_stats_path",
            "t5_embeddings_path",
        ):
            identity.pop(name)
        return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()

    @property
    def prompt_manifest_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {"instruction": self.instruction, "task_id": self.task_id}
            )
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        fields = _REQUIRED_FIELDS | {
            name for name in _OPTIONAL_FIELDS if getattr(self, name) is not None
        }
        return {
            field: (
                [item.to_dict() for item in self.checkpoint_files]
                if field == "checkpoint_files"
                else str(getattr(self, field))
                if field.endswith("_path") or field.endswith("_root")
                else getattr(self, field)
            )
            for field in sorted(fields)
        }

    @classmethod
    def from_dict(cls, value: object) -> "DreamTacArtifactManifest":
        if not isinstance(value, Mapping):
            raise ValueError("Dream-Tac artifact manifest fields mismatch")
        fields = set(value)
        if not fields >= _REQUIRED_FIELDS or not fields <= _FIELDS:
            raise ValueError("Dream-Tac artifact manifest fields mismatch")
        document = dict(value)
        for name in (
            "bundle_root",
            "checkpoint_root",
            "dataset_stats_path",
            "t5_embeddings_path",
        ):
            document[name] = Path(_string(document[name], name))
        raw_files = document["checkpoint_files"]
        if not isinstance(raw_files, list):
            raise ValueError("checkpoint_files must be a list")
        document["checkpoint_files"] = tuple(
            DreamTacArtifactFile.from_dict(item) for item in raw_files
        )
        return cls(**cast(dict[str, Any], document))


def build_dream_tac_artifact_manifest(
    *,
    bundle_root: Path,
    checkpoint_root: Path,
    dataset_stats_path: Path,
    t5_embeddings_path: Path,
    task_id: str,
    instruction: str,
    experiment_config: str,
    control_hz: float,
    gripper_mapping: DreamTacGripperMapping,
    gripper_threshold: float,
    gripper_qpos_min: float | None = None,
    gripper_qpos_max: float | None = None,
) -> DreamTacArtifactManifest:
    """Hash every local file required by the upstream Franka server."""

    files = _checkpoint_inventory(checkpoint_root)
    return DreamTacArtifactManifest(
        task_id=task_id,
        instruction=instruction,
        experiment_config=experiment_config,
        bundle_root=Path(bundle_root).absolute(),
        checkpoint_root=Path(checkpoint_root).absolute(),
        checkpoint_files=files,
        checkpoint_tree_sha256=_tree_sha256(files),
        dataset_stats_path=Path(dataset_stats_path).absolute(),
        dataset_stats_sha256=stable_file_sha256(
            dataset_stats_path, "dataset statistics"
        ),
        t5_embeddings_path=Path(t5_embeddings_path).absolute(),
        t5_embeddings_sha256=stable_file_sha256(t5_embeddings_path, "T5 embeddings"),
        external_commit=load_integration_lock().by_id("dream_tac").commit_sha,
        control_hz=control_hz,
        gripper_mapping=DreamTacGripperMapping(gripper_mapping).value,
        gripper_threshold=gripper_threshold,
        gripper_qpos_min=gripper_qpos_min,
        gripper_qpos_max=gripper_qpos_max,
    )


def validate_dream_tac_artifact(manifest: DreamTacArtifactManifest) -> None:
    """Re-hash checkpoint, dataset statistics, and T5 embeddings."""

    current_files = _checkpoint_inventory(manifest.checkpoint_root)
    if current_files != manifest.checkpoint_files:
        raise ValueError("Dream-Tac checkpoint file SHA256 mismatch")
    if _tree_sha256(current_files) != manifest.checkpoint_tree_sha256:
        raise ValueError("Dream-Tac checkpoint tree SHA256 mismatch")
    if stable_file_sha256(manifest.dataset_stats_path, "dataset statistics") != (
        manifest.dataset_stats_sha256
    ):
        raise ValueError("Dream-Tac dataset statistics SHA256 mismatch")
    if stable_file_sha256(manifest.t5_embeddings_path, "T5 embeddings") != (
        manifest.t5_embeddings_sha256
    ):
        raise ValueError("Dream-Tac T5 embeddings SHA256 mismatch")


def load_dream_tac_artifact_manifest(path: Path) -> DreamTacArtifactManifest:
    """Load one canonical Dream-Tac manifest without importing its runtime."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("Dream-Tac manifest must be a non-symlink regular file")
    raw = selected.read_bytes()
    manifest = DreamTacArtifactManifest.from_dict(
        strict_json_bytes(raw, "Dream-Tac artifact manifest")
    )
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise ValueError("Dream-Tac artifact manifest is not canonical")
    return manifest


__all__ = [
    "ACTION_HORIZON",
    "CASA_INFERENCE_CONTRACT",
    "DreamTacArtifactFile",
    "DreamTacArtifactManifest",
    "SCHEMA_VERSION",
    "WEIGHT_RELEASE_STATUS",
    "build_dream_tac_artifact_manifest",
    "load_dream_tac_artifact_manifest",
    "validate_dream_tac_artifact",
]

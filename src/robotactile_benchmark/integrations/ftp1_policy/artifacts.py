"""Content-addressed FTP-1 UniVTAC serving artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock

SCHEMA_VERSION: Final = "robotactile-ftp1-policy-artifact-v1"
CHECKPOINT_REPOSITORY: Final = "MJJJJ1064/ftp1_univtac_finetune"
CHECKPOINT_REVISION: Final = "620ac69b4fffd2341300cfef1b1d224d56710ed3"
CHECKPOINT_STEP: Final = 19999
ACTION_DIM: Final = 120
ACTION_HORIZON: Final = 32
ACTION_REP: Final = "mix"
CHUNK_INDEX_OFFSET: Final = 1
CHUNK_FIRST_N: Final = 20
TEMPORAL_ENSEMBLE_K: Final = 0.01
USE_TACTILE: Final = True
COLOR_CONTRACT: Final = "upstream_passthrough_v1"
WEIGHT_LICENSE_STATUS: Final = "upstream_unspecified"
GEMMA_TERMS_URI: Final = "https://ai.google.dev/gemma/terms"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class FTP1TaskRelease:
    """One task-specific checkpoint published in the official collection."""

    checkpoint_name: str
    domain_name: str
    prompt: str
    camera_route: str


TASK_RELEASES: Final[Mapping[str, FTP1TaskRelease]] = {
    "insert_hole": FTP1TaskRelease(
        "FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1",
        "UniVTAC_insert_hole",
        "insert the stick to the hole.",
        "head",
    ),
    "insert_tube": FTP1TaskRelease(
        "FTP1_UniVTAC_insert_tube_expert_gsmall_ftp1",
        "UniVTAC_insert_tube",
        "insert the tube to the fixed slot.",
        "all",
    ),
    "lift_bottle": FTP1TaskRelease(
        "FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1",
        "UniVTAC_lift_bottle",
        (
            "grasp the bottle and lift it vertically, keeping its final base "
            "within 5 cm of the wall."
        ),
        "head",
    ),
    "lift_can": FTP1TaskRelease(
        "FTP1_UniVTAC_lift_can_expert_gsmall_ftp1",
        "UniVTAC_lift_can",
        "grasp the can and lifts it vertically without slippage.",
        "all",
    ),
    "pull_out_key": FTP1TaskRelease(
        "FTP1_UniVTAC_pull_out_key_expert_gsmall_ftp1",
        "UniVTAC_pull_out_key",
        "pull out the key.",
        "head",
    ),
    "put_bottle_in_shelf": FTP1TaskRelease(
        "FTP1_UniVTAC_put_bottle_expert_gsmall_ftp1",
        "UniVTAC_put_bottle",
        "grasp the bottle, then position it into the shelf cavity.",
        "head",
    ),
}

_MANIFEST_FIELDS = frozenset(
    {
        "action_dim",
        "action_horizon",
        "action_rep",
        "bundle_root",
        "camera_route",
        "checkpoint_name",
        "checkpoint_repository",
        "checkpoint_revision",
        "checkpoint_root",
        "checkpoint_step",
        "chunk_first_n",
        "chunk_index_offset",
        "color_contract",
        "domain_name",
        "external_commit",
        "gemma_terms_uri",
        "hpt_gelsight_path",
        "hpt_gelsight_sha256",
        "hpt_shared_path",
        "hpt_shared_sha256",
        "model_config_path",
        "model_config_sha256",
        "model_path",
        "model_sha256",
        "normalization_files",
        "normalization_root",
        "normalization_tree_sha256",
        "prompt",
        "schema_version",
        "tactile_config_path",
        "tactile_config_sha256",
        "task_id",
        "temporal_ensemble_k",
        "train_config_path",
        "train_config_sha256",
        "use_tactile",
        "weight_license_status",
    }
)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _absolute(value: Path, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path")
    return value.absolute()


def _inside(root: Path, path: Path, name: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(f"{name} must be below bundle_root")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def file_sha256(path: Path, name: str) -> str:
    """Hash one stable, non-symlink regular file."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"FTP-1 {name} must be a non-symlink regular file")
    before = selected.stat()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = selected.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"FTP-1 {name} changed while hashing")
    return digest.hexdigest()


@dataclass(frozen=True)
class FTP1ArtifactFile:
    """One path-independent file identity below the normalization root."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        normalized = PurePosixPath(_string(self.path, "normalization file path"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("normalization file path must be safe and relative")
        object.__setattr__(self, "path", normalized.as_posix())
        object.__setattr__(self, "sha256", _sha256(self.sha256, "file sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> "FTP1ArtifactFile":
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError("normalization file identity fields mismatch")
        return cls(
            path=_string(value["path"], "normalization file path"),
            sha256=_sha256(value["sha256"], "normalization file sha256"),
        )


def _normalization_paths(release: FTP1TaskRelease) -> tuple[str, ...]:
    domain = release.domain_name
    return (
        "action_group_frequency_stats_train.json",
        "dataset_stats.json",
        f"{domain}/contact_detection_thresholds.json",
        f"{domain}/independent_norm_stats_all_t0_zscore.json",
        f"{domain}/train_val_split.json",
        "norm_params_snapshot.json",
        "share_norm_stats_all_t0_zscore.json",
    )


def _tree_sha256(files: tuple[FTP1ArtifactFile, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([item.to_dict() for item in files])
    ).hexdigest()


@dataclass(frozen=True)
class FTP1PolicyArtifactManifest:
    """Exact source, checkpoint tree, and serving contract for one FTP-1 task."""

    task_id: str
    bundle_root: Path
    checkpoint_root: Path
    model_path: Path
    model_sha256: str
    model_config_path: Path
    model_config_sha256: str
    train_config_path: Path
    train_config_sha256: str
    tactile_config_path: Path
    tactile_config_sha256: str
    hpt_gelsight_path: Path
    hpt_gelsight_sha256: str
    hpt_shared_path: Path
    hpt_shared_sha256: str
    normalization_root: Path
    normalization_files: tuple[FTP1ArtifactFile, ...]
    normalization_tree_sha256: str
    external_commit: str
    checkpoint_name: str
    domain_name: str
    prompt: str
    camera_route: str
    checkpoint_repository: str = CHECKPOINT_REPOSITORY
    checkpoint_revision: str = CHECKPOINT_REVISION
    checkpoint_step: int = CHECKPOINT_STEP
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    action_rep: str = ACTION_REP
    chunk_index_offset: int = CHUNK_INDEX_OFFSET
    chunk_first_n: int = CHUNK_FIRST_N
    temporal_ensemble_k: float = TEMPORAL_ENSEMBLE_K
    use_tactile: bool = USE_TACTILE
    color_contract: str = COLOR_CONTRACT
    weight_license_status: str = WEIGHT_LICENSE_STATUS
    gemma_terms_uri: str = GEMMA_TERMS_URI
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            release = TASK_RELEASES[self.task_id]
        except KeyError as error:
            raise ValueError("FTP-1 task has no released UniVTAC checkpoint") from error
        root = _absolute(self.bundle_root, "bundle_root")
        checkpoint = _absolute(self.checkpoint_root, "checkpoint_root")
        expected_checkpoint = root / release.checkpoint_name / str(CHECKPOINT_STEP)
        if checkpoint != expected_checkpoint:
            raise ValueError("FTP-1 checkpoint_root does not match the released layout")
        paths = {
            "model_path": (self.model_path, checkpoint / "model.safetensors"),
            "model_config_path": (
                self.model_config_path,
                checkpoint / "model_config.json",
            ),
            "train_config_path": (
                self.train_config_path,
                checkpoint / "train_config.json",
            ),
            "tactile_config_path": (
                self.tactile_config_path,
                checkpoint / "tactile_input_config_file.json",
            ),
            "hpt_gelsight_path": (
                self.hpt_gelsight_path,
                checkpoint
                / "hpt_tokenizer"
                / "GelSightMini_image_224_224_3.safetensors",
            ),
            "hpt_shared_path": (
                self.hpt_shared_path,
                checkpoint / "hpt_tokenizer" / "shared_image_chunk_encoder.safetensors",
            ),
            "normalization_root": (
                self.normalization_root,
                checkpoint / "normalization",
            ),
        }
        for name, (actual, expected) in paths.items():
            normalized = _absolute(actual, name)
            _inside(root, normalized, name)
            if normalized != expected:
                raise ValueError(f"FTP-1 {name} does not match the released layout")
            object.__setattr__(self, name, normalized)
        expected_files = _normalization_paths(release)
        files = tuple(self.normalization_files)
        if tuple(item.path for item in files) != expected_files:
            raise ValueError("FTP-1 normalization file inventory mismatch")
        if self.normalization_tree_sha256 != _tree_sha256(files):
            raise ValueError("FTP-1 normalization tree identity mismatch")
        expected_commit = load_integration_lock().by_id("ftp1_policy").commit_sha
        if (
            self.external_commit != expected_commit
            or _COMMIT.fullmatch(expected_commit) is None
        ):
            raise ValueError("FTP-1 external source commit mismatch")
        fixed = {
            "schema_version": (self.schema_version, SCHEMA_VERSION),
            "checkpoint_repository": (
                self.checkpoint_repository,
                CHECKPOINT_REPOSITORY,
            ),
            "checkpoint_revision": (self.checkpoint_revision, CHECKPOINT_REVISION),
            "checkpoint_step": (self.checkpoint_step, CHECKPOINT_STEP),
            "checkpoint_name": (self.checkpoint_name, release.checkpoint_name),
            "domain_name": (self.domain_name, release.domain_name),
            "prompt": (self.prompt, release.prompt),
            "camera_route": (self.camera_route, release.camera_route),
            "action_dim": (self.action_dim, ACTION_DIM),
            "action_horizon": (self.action_horizon, ACTION_HORIZON),
            "action_rep": (self.action_rep, ACTION_REP),
            "chunk_index_offset": (self.chunk_index_offset, CHUNK_INDEX_OFFSET),
            "chunk_first_n": (self.chunk_first_n, CHUNK_FIRST_N),
            "temporal_ensemble_k": (
                self.temporal_ensemble_k,
                TEMPORAL_ENSEMBLE_K,
            ),
            "use_tactile": (self.use_tactile, USE_TACTILE),
            "color_contract": (self.color_contract, COLOR_CONTRACT),
            "weight_license_status": (
                self.weight_license_status,
                WEIGHT_LICENSE_STATUS,
            ),
            "gemma_terms_uri": (self.gemma_terms_uri, GEMMA_TERMS_URI),
        }
        if any(actual != wanted for actual, wanted in fixed.values()):
            raise ValueError("FTP-1 released artifact or serving contract mismatch")
        for name in (
            "model_sha256",
            "model_config_sha256",
            "train_config_sha256",
            "tactile_config_sha256",
            "hpt_gelsight_sha256",
            "hpt_shared_sha256",
            "normalization_tree_sha256",
        ):
            _sha256(getattr(self, name), name)
        object.__setattr__(self, "bundle_root", root)
        object.__setattr__(self, "checkpoint_root", checkpoint)
        object.__setattr__(self, "normalization_files", files)

    @property
    def checkpoint_sha256(self) -> str:
        return self.model_sha256

    @property
    def config_sha256(self) -> str:
        return self.serve_bundle_sha256

    @property
    def normalizer_sha256(self) -> str:
        return self.normalization_tree_sha256

    @property
    def serve_bundle_sha256(self) -> str:
        identity = self.to_dict()
        for name in (
            "bundle_root",
            "checkpoint_root",
            "model_path",
            "model_config_path",
            "train_config_path",
            "tactile_config_path",
            "hpt_gelsight_path",
            "hpt_shared_path",
            "normalization_root",
        ):
            identity.pop(name)
        return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()

    @property
    def prompt_manifest_sha256(self) -> str:
        identity = {
            "camera_route": self.camera_route,
            "color_contract": self.color_contract,
            "domain_name": self.domain_name,
            "prompt": self.prompt,
            "task_id": self.task_id,
        }
        return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()

    def transport_metadata(self) -> dict[str, object]:
        """Return the exact identity expected from the isolated policy worker."""

        camera_keys = ["camera_ego_rgb_0"]
        if self.camera_route == "all":
            camera_keys.append("right_wrist_camera_rgb_0")
        return {
            "action_horizon": self.action_horizon,
            "action_rep": "absolute",
            "camera_keys": camera_keys,
            "checkpoint_sha256": self.checkpoint_sha256,
            "chunk_first_n": self.chunk_first_n,
            "chunk_index_offset": self.chunk_index_offset,
            "color_contract": self.color_contract,
            "model": "ftp1_policy",
            "model_action_dim": self.action_dim,
            "output_action_dim": 8,
            "prompt": self.prompt,
            "protocol_version": "robotactile-ftp1-zmq-v2",
            "randomness_contract": "episode_exogenous_seed_v1",
            "serve_bundle_sha256": self.serve_bundle_sha256,
            "source_commit": self.external_commit,
            "task_id": self.task_id,
            "temporal_ensemble_k": self.temporal_ensemble_k,
            "use_tactile": self.use_tactile,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            field: (
                [item.to_dict() for item in self.normalization_files]
                if field == "normalization_files"
                else str(getattr(self, field))
                if field.endswith("_path") or field.endswith("_root")
                else getattr(self, field)
            )
            for field in sorted(_MANIFEST_FIELDS)
        }

    @classmethod
    def from_dict(cls, value: object) -> "FTP1PolicyArtifactManifest":
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise ValueError("FTP-1 artifact manifest fields mismatch")
        document = dict(value)
        for name in (
            "bundle_root",
            "checkpoint_root",
            "model_path",
            "model_config_path",
            "train_config_path",
            "tactile_config_path",
            "hpt_gelsight_path",
            "hpt_shared_path",
            "normalization_root",
        ):
            document[name] = Path(_string(document[name], name))
        files = document["normalization_files"]
        if not isinstance(files, list):
            raise ValueError("normalization_files must be a list")
        document["normalization_files"] = tuple(
            FTP1ArtifactFile.from_dict(item) for item in files
        )
        return cls(**cast(dict[str, Any], document))


def build_ftp1_policy_artifact_manifest(
    *, bundle_root: Path, checkpoint_root: Path, task_id: str
) -> FTP1PolicyArtifactManifest:
    """Hash every serving file for one downloaded official task checkpoint."""

    try:
        release = TASK_RELEASES[task_id]
    except KeyError as error:
        raise ValueError("FTP-1 task has no released UniVTAC checkpoint") from error
    root = Path(bundle_root).absolute()
    checkpoint = Path(checkpoint_root).absolute()
    expected_checkpoint = root / release.checkpoint_name / str(CHECKPOINT_STEP)
    if checkpoint != expected_checkpoint:
        raise ValueError("FTP-1 checkpoint_root does not match the released layout")
    normalization = checkpoint / "normalization"
    files = tuple(
        FTP1ArtifactFile(path, file_sha256(normalization / path, path))
        for path in _normalization_paths(release)
    )
    return FTP1PolicyArtifactManifest(
        task_id=task_id,
        bundle_root=root,
        checkpoint_root=checkpoint,
        model_path=checkpoint / "model.safetensors",
        model_sha256=file_sha256(checkpoint / "model.safetensors", "model"),
        model_config_path=checkpoint / "model_config.json",
        model_config_sha256=file_sha256(
            checkpoint / "model_config.json", "model config"
        ),
        train_config_path=checkpoint / "train_config.json",
        train_config_sha256=file_sha256(
            checkpoint / "train_config.json", "train config"
        ),
        tactile_config_path=checkpoint / "tactile_input_config_file.json",
        tactile_config_sha256=file_sha256(
            checkpoint / "tactile_input_config_file.json", "tactile config"
        ),
        hpt_gelsight_path=(
            checkpoint / "hpt_tokenizer" / "GelSightMini_image_224_224_3.safetensors"
        ),
        hpt_gelsight_sha256=file_sha256(
            checkpoint / "hpt_tokenizer" / "GelSightMini_image_224_224_3.safetensors",
            "GelSightMini tokenizer",
        ),
        hpt_shared_path=(
            checkpoint / "hpt_tokenizer" / "shared_image_chunk_encoder.safetensors"
        ),
        hpt_shared_sha256=file_sha256(
            checkpoint / "hpt_tokenizer" / "shared_image_chunk_encoder.safetensors",
            "shared tactile tokenizer",
        ),
        normalization_root=normalization,
        normalization_files=files,
        normalization_tree_sha256=_tree_sha256(files),
        external_commit=load_integration_lock().by_id("ftp1_policy").commit_sha,
        checkpoint_name=release.checkpoint_name,
        domain_name=release.domain_name,
        prompt=release.prompt,
        camera_route=release.camera_route,
    )


def validate_ftp1_policy_artifact(manifest: FTP1PolicyArtifactManifest) -> None:
    """Re-hash all model, tokenizer, config, and normalization files."""

    checks = (
        (manifest.model_path, "model", manifest.model_sha256),
        (
            manifest.model_config_path,
            "model config",
            manifest.model_config_sha256,
        ),
        (
            manifest.train_config_path,
            "train config",
            manifest.train_config_sha256,
        ),
        (
            manifest.tactile_config_path,
            "tactile config",
            manifest.tactile_config_sha256,
        ),
        (
            manifest.hpt_gelsight_path,
            "GelSightMini tokenizer",
            manifest.hpt_gelsight_sha256,
        ),
        (
            manifest.hpt_shared_path,
            "shared tactile tokenizer",
            manifest.hpt_shared_sha256,
        ),
    )
    for path, name, expected in checks:
        if file_sha256(path, name) != expected:
            raise ValueError(f"FTP-1 {name} SHA256 mismatch")
    current_files = tuple(
        FTP1ArtifactFile(
            item.path, file_sha256(manifest.normalization_root / item.path, item.path)
        )
        for item in manifest.normalization_files
    )
    if current_files != manifest.normalization_files:
        raise ValueError("FTP-1 normalization file SHA256 mismatch")
    if _tree_sha256(current_files) != manifest.normalization_tree_sha256:
        raise ValueError("FTP-1 normalization tree SHA256 mismatch")


def load_ftp1_policy_artifact_manifest(path: Path) -> FTP1PolicyArtifactManifest:
    """Load one canonical manifest without importing the FTP-1 runtime."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("FTP-1 manifest must be a non-symlink regular file")
    raw = selected.read_bytes()
    manifest = FTP1PolicyArtifactManifest.from_dict(
        strict_json_bytes(raw, "FTP-1 artifact manifest")
    )
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise ValueError("FTP-1 artifact manifest is not canonical")
    return manifest


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_REP",
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_REVISION",
    "CHECKPOINT_STEP",
    "CHUNK_FIRST_N",
    "CHUNK_INDEX_OFFSET",
    "COLOR_CONTRACT",
    "FTP1ArtifactFile",
    "FTP1PolicyArtifactManifest",
    "FTP1TaskRelease",
    "GEMMA_TERMS_URI",
    "SCHEMA_VERSION",
    "TASK_RELEASES",
    "TEMPORAL_ENSEMBLE_K",
    "USE_TACTILE",
    "WEIGHT_LICENSE_STATUS",
    "build_ftp1_policy_artifact_manifest",
    "load_ftp1_policy_artifact_manifest",
    "validate_ftp1_policy_artifact",
]

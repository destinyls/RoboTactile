"""Content-addressed official N0-VTLA UniVTAC serving artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock

SCHEMA_VERSION = "robotactile-n0-vtla-artifact-v1"
TASK_ID = "insert_hole"
CONFIG_NAME = "sim_single_arm_tactile"
ASSET_ID = "n0_insert_hole_norm"
CHECKPOINT_REPOSITORY = "NeoteAI/n0_VTLA_insert_hole"
CHECKPOINT_REVISION = "73a514c015c6745a14a3efdca92f25c6cfab5eb5"
TRAINING_PROMPT = "insert hole"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FIELDS = frozenset(
    {
        "asset_id",
        "bundle_root",
        "checkpoint_path",
        "checkpoint_repository",
        "checkpoint_revision",
        "checkpoint_root",
        "checkpoint_sha256",
        "config_name",
        "config_path",
        "config_sha256",
        "external_commit",
        "normalizer_path",
        "normalizer_sha256",
        "schema_version",
        "task_id",
    }
)


def _absolute(value: Path, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path")
    return value.absolute()


def _inside(root: Path, path: Path, name: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(f"{name} must be below bundle_root")


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def file_sha256(path: Path, name: str) -> str:
    """Hash one non-symlink regular file after a stable stat check."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"N0-VTLA {name} must be a non-symlink regular file")
    before = selected.stat()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = selected.stat()

    def identity(item: object) -> tuple[int, int, int, int, int]:
        metadata = cast(Any, item)
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    if identity(before) != identity(after):
        raise ValueError(f"N0-VTLA {name} changed while hashing")
    return digest.hexdigest()


@dataclass(frozen=True)
class N0VTLAArtifactManifest:
    """Exact source, checkpoint, config, and normalization identity."""

    bundle_root: Path
    checkpoint_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    config_path: Path
    config_sha256: str
    normalizer_path: Path
    normalizer_sha256: str
    task_id: str = TASK_ID
    config_name: str = CONFIG_NAME
    asset_id: str = ASSET_ID
    checkpoint_repository: str = CHECKPOINT_REPOSITORY
    checkpoint_revision: str = CHECKPOINT_REVISION
    external_commit: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        root = _absolute(self.bundle_root, "bundle_root")
        checkpoint_root = _absolute(self.checkpoint_root, "checkpoint_root")
        checkpoint = _absolute(self.checkpoint_path, "checkpoint_path")
        config = _absolute(self.config_path, "config_path")
        normalizer = _absolute(self.normalizer_path, "normalizer_path")
        for path, name in (
            (checkpoint_root, "checkpoint_root"),
            (checkpoint, "checkpoint_path"),
            (config, "config_path"),
            (normalizer, "normalizer_path"),
        ):
            _inside(root, path, name)
        if checkpoint != checkpoint_root / "model.safetensors":
            raise ValueError("N0-VTLA checkpoint path must end in model.safetensors")
        if config != checkpoint_root / "config.json":
            raise ValueError("N0-VTLA config path must end in checkpoint/config.json")
        expected_normalizer = checkpoint_root / "assets" / ASSET_ID / "norm_stats.json"
        if normalizer != expected_normalizer:
            raise ValueError("N0-VTLA normalizer path does not match the asset ID")
        expected = load_integration_lock().by_id("n0_vtla").commit_sha
        if self.external_commit != expected or _COMMIT.fullmatch(expected) is None:
            raise ValueError("N0-VTLA external source commit mismatch")
        fixed = {
            "schema_version": (self.schema_version, SCHEMA_VERSION),
            "task_id": (self.task_id, TASK_ID),
            "config_name": (self.config_name, CONFIG_NAME),
            "asset_id": (self.asset_id, ASSET_ID),
            "checkpoint_repository": (
                self.checkpoint_repository,
                CHECKPOINT_REPOSITORY,
            ),
            "checkpoint_revision": (self.checkpoint_revision, CHECKPOINT_REVISION),
        }
        if any(actual != wanted for actual, wanted in fixed.values()):
            raise ValueError("N0-VTLA released artifact identity mismatch")
        for name in ("checkpoint_sha256", "config_sha256", "normalizer_sha256"):
            _sha256(getattr(self, name), name)
        object.__setattr__(self, "bundle_root", root)
        object.__setattr__(self, "checkpoint_root", checkpoint_root)
        object.__setattr__(self, "checkpoint_path", checkpoint)
        object.__setattr__(self, "config_path", config)
        object.__setattr__(self, "normalizer_path", normalizer)

    @property
    def serve_bundle_sha256(self) -> str:
        """Return a path-independent identity for the released serve bundle."""

        identity = {
            "asset_id": self.asset_id,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_name": self.config_name,
            "config_sha256": self.config_sha256,
            "external_commit": self.external_commit,
            "normalizer_sha256": self.normalizer_sha256,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
        }
        return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()

    @property
    def prompt_manifest_sha256(self) -> str:
        """Return the exact prompt/task contract expected by the checkpoint."""

        identity = {
            "config_name": self.config_name,
            "prompt": TRAINING_PROMPT,
            "task_id": self.task_id,
        }
        return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "bundle_root": str(self.bundle_root),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_repository": self.checkpoint_repository,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_root": str(self.checkpoint_root),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_name": self.config_name,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "external_commit": self.external_commit,
            "normalizer_path": str(self.normalizer_path),
            "normalizer_sha256": self.normalizer_sha256,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "N0VTLAArtifactManifest":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("N0-VTLA artifact manifest fields mismatch")
        document = dict(value)
        for name in (
            "bundle_root",
            "checkpoint_root",
            "checkpoint_path",
            "config_path",
            "normalizer_path",
        ):
            document[name] = Path(_string(document[name], name))
        return cls(**cast(dict[str, Any], document))


def build_n0_vtla_artifact_manifest(
    *, bundle_root: Path, checkpoint_root: Path
) -> N0VTLAArtifactManifest:
    """Build a manifest from the downloaded checkpoint's complete serve files."""

    checkpoint = checkpoint_root / "model.safetensors"
    config = checkpoint_root / "config.json"
    normalizer = checkpoint_root / "assets" / ASSET_ID / "norm_stats.json"
    return N0VTLAArtifactManifest(
        bundle_root=bundle_root.absolute(),
        checkpoint_root=checkpoint_root.absolute(),
        checkpoint_path=checkpoint.absolute(),
        checkpoint_sha256=file_sha256(checkpoint, "checkpoint"),
        config_path=config.absolute(),
        config_sha256=file_sha256(config, "checkpoint config"),
        normalizer_path=normalizer.absolute(),
        normalizer_sha256=file_sha256(normalizer, "normalizer"),
        external_commit=load_integration_lock().by_id("n0_vtla").commit_sha,
    )


def validate_n0_vtla_artifact(manifest: N0VTLAArtifactManifest) -> None:
    """Re-hash every runtime file named by a typed manifest."""

    checks = (
        (manifest.checkpoint_path, "checkpoint", manifest.checkpoint_sha256),
        (manifest.config_path, "checkpoint config", manifest.config_sha256),
        (manifest.normalizer_path, "normalizer", manifest.normalizer_sha256),
    )
    for path, name, expected in checks:
        if file_sha256(path, name) != expected:
            raise ValueError(f"N0-VTLA {name} SHA256 mismatch")


def load_n0_vtla_artifact_manifest(path: Path) -> N0VTLAArtifactManifest:
    """Load one canonical artifact manifest without importing N0-VTLA."""

    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("N0-VTLA manifest must be a non-symlink regular file")
    raw = selected.read_bytes()
    manifest = N0VTLAArtifactManifest.from_dict(
        strict_json_bytes(raw, "N0-VTLA artifact manifest")
    )
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise ValueError("N0-VTLA artifact manifest is not canonical")
    return manifest


__all__ = [
    "ASSET_ID",
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_REVISION",
    "CONFIG_NAME",
    "N0VTLAArtifactManifest",
    "SCHEMA_VERSION",
    "TASK_ID",
    "TRAINING_PROMPT",
    "build_n0_vtla_artifact_manifest",
    "load_n0_vtla_artifact_manifest",
    "validate_n0_vtla_artifact",
]

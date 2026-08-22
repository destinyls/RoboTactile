"""External N0-TWAM model-bundle identity and file verification."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.integrations.provenance import load_integration_lock

_SCHEMA_VERSION = "robotactile-n0-artifact-v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "bundle_root",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "external_commit",
        "normalizer_path",
        "normalizer_sha256",
        "schema_version",
    }
)


def _absolute(value: Path, name: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ValueError(f"{name} must be an absolute pathlib.Path")
    return value.absolute()


def _inside(root: Path, path: Path, name: str) -> None:
    if path == root or root not in path.parents:
        raise ValueError(f"{name} must be a file below bundle_root")


def _file_sha256(path: Path, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"N0-TWAM {name} must be a non-symlink regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class N0TWAMArtifactManifest:
    """Absolute external paths with content-addressed model resources."""

    bundle_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    config_path: Path
    config_sha256: str
    normalizer_path: Path
    normalizer_sha256: str
    external_commit: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        root = _absolute(self.bundle_root, "bundle_root")
        object.__setattr__(self, "bundle_root", root)
        for name in ("checkpoint_path", "config_path", "normalizer_path"):
            selected = _absolute(getattr(self, name), name)
            _inside(root, selected, name)
            object.__setattr__(self, name, selected)
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("N0-TWAM artifact schema version mismatch")
        expected_commit = load_integration_lock().by_id("n0_twam").commit_sha
        if self.external_commit != expected_commit:
            raise ValueError("N0-TWAM external commit mismatch")
        for name in ("checkpoint_sha256", "config_sha256", "normalizer_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256")

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_root": str(self.bundle_root),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "external_commit": self.external_commit,
            "normalizer_path": str(self.normalizer_path),
            "normalizer_sha256": self.normalizer_sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "N0TWAMArtifactManifest":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise ValueError("N0-TWAM artifact manifest fields mismatch")
        document = dict(value)
        for field in (
            "bundle_root",
            "checkpoint_path",
            "config_path",
            "normalizer_path",
        ):
            raw = document[field]
            if not isinstance(raw, str):
                raise TypeError(f"{field} must be a string")
            document[field] = Path(raw)
        return cls(
            bundle_root=cast(Path, document["bundle_root"]),
            checkpoint_path=cast(Path, document["checkpoint_path"]),
            checkpoint_sha256=cast(str, document["checkpoint_sha256"]),
            config_path=cast(Path, document["config_path"]),
            config_sha256=cast(str, document["config_sha256"]),
            normalizer_path=cast(Path, document["normalizer_path"]),
            normalizer_sha256=cast(str, document["normalizer_sha256"]),
            external_commit=cast(str, document["external_commit"]),
            schema_version=cast(str, document["schema_version"]),
        )


def build_n0_twam_artifact_manifest(
    *,
    bundle_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    normalizer_path: Path,
) -> N0TWAMArtifactManifest:
    """Hash three explicit N0 resources without importing its runtime."""

    root = Path(bundle_root).absolute()
    checkpoint = Path(checkpoint_path).absolute()
    config = Path(config_path).absolute()
    normalizer = Path(normalizer_path).absolute()
    return N0TWAMArtifactManifest(
        bundle_root=root,
        checkpoint_path=checkpoint,
        checkpoint_sha256=_file_sha256(checkpoint, "checkpoint"),
        config_path=config,
        config_sha256=_file_sha256(config, "config"),
        normalizer_path=normalizer,
        normalizer_sha256=_file_sha256(normalizer, "normalizer"),
        external_commit=load_integration_lock().by_id("n0_twam").commit_sha,
    )


def validate_n0_twam_artifact(
    manifest: N0TWAMArtifactManifest,
) -> Mapping[str, str]:
    if type(manifest) is not N0TWAMArtifactManifest:
        raise TypeError("manifest must be an exact N0TWAMArtifactManifest")
    resources = (
        (manifest.checkpoint_path, manifest.checkpoint_sha256, "checkpoint"),
        (manifest.config_path, manifest.config_sha256, "config"),
        (manifest.normalizer_path, manifest.normalizer_sha256, "normalizer"),
    )
    for path, expected, name in resources:
        if _file_sha256(path, name) != expected:
            raise ValueError(f"N0-TWAM {name} SHA256 mismatch")
    return {
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": manifest.config_sha256,
        "external_commit": manifest.external_commit,
        "normalizer_sha256": manifest.normalizer_sha256,
    }


def load_n0_twam_artifact_manifest(path: Path) -> N0TWAMArtifactManifest:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("N0-TWAM artifact manifest must be a regular file")
    raw = selected.read_bytes()
    value = strict_json_bytes(raw, "N0-TWAM artifact manifest")
    manifest = N0TWAMArtifactManifest.from_dict(value)
    if canonical_json_bytes(manifest.to_dict()) != raw:
        raise ValueError("N0-TWAM artifact manifest is not canonical")
    return manifest


__all__ = [
    "N0TWAMArtifactManifest",
    "build_n0_twam_artifact_manifest",
    "load_n0_twam_artifact_manifest",
    "validate_n0_twam_artifact",
]

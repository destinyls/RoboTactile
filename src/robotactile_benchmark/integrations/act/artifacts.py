"""Hash-bound official ACT artifact manifest codecs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
    OfficialUniVTACACTLoadRequest,
    validate_official_univtac_act_artifact,
)

_ARTIFACT_FAMILY = "official_univtac_policy_last"
_FIELDS = frozenset(
    {
        "artifact_family",
        "artifact_root",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "encoder_path",
        "encoder_sha256",
        "profile",
        "semantic_version",
        "source_path",
        "source_sha256",
        "stats_path",
        "stats_sha256",
        "task_id",
        "upstream_commit",
        "upstream_root",
    }
)

ACTArtifactManifest = OfficialUniVTACACTArtifactManifest
ACTLoadRequest = OfficialUniVTACACTLoadRequest


def _sha256_file(path: Path, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"ACT {name} must be a non-symlink regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_act_artifact_manifest(
    *,
    task_id: str,
    profile: OfficialACTProfile,
    artifact_root: Path,
    upstream_root: Path,
) -> ACTArtifactManifest:
    """Hash the frozen official layout and validate it without loading Torch."""

    normalized = OfficialACTProfile(profile)
    artifact = Path(artifact_root).absolute()
    upstream = Path(upstream_root).absolute()
    profile_root = artifact / task_id / normalized.value
    manifest = ACTArtifactManifest.for_shared_root(
        task_id=task_id,
        profile=normalized,
        artifact_root=artifact,
        upstream_root=upstream,
        checkpoint_sha256=_sha256_file(profile_root / "policy_last.ckpt", "checkpoint"),
        stats_sha256=_sha256_file(profile_root / "dataset_stats.pkl", "stats"),
        encoder_sha256=_sha256_file(artifact / "encoder.pth", "encoder"),
    )
    validate_official_univtac_act_artifact(manifest)
    return manifest


def act_artifact_manifest_to_dict(
    manifest: ACTArtifactManifest,
) -> dict[str, object]:
    if type(manifest) is not ACTArtifactManifest:
        raise TypeError("manifest must be an exact ACTArtifactManifest")
    return {
        "artifact_family": _ARTIFACT_FAMILY,
        "artifact_root": str(manifest.artifact_root),
        "checkpoint_path": str(manifest.checkpoint_path),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_path": str(manifest.config_path),
        "config_sha256": manifest.config_sha256,
        "encoder_path": str(manifest.encoder_path),
        "encoder_sha256": manifest.encoder_sha256,
        "profile": manifest.profile.value,
        "semantic_version": manifest.semantic_version,
        "source_path": str(manifest.source_path),
        "source_sha256": manifest.source_sha256,
        "stats_path": str(manifest.stats_path),
        "stats_sha256": manifest.stats_sha256,
        "task_id": manifest.task_id,
        "upstream_commit": manifest.upstream_commit,
        "upstream_root": str(manifest.upstream_root),
    }


def act_artifact_manifest_from_dict(value: object) -> ACTArtifactManifest:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("ACT artifact manifest fields mismatch")
    document = dict(value)
    if document.pop("artifact_family") != _ARTIFACT_FAMILY:
        raise ValueError("ACT artifact family mismatch")
    for field in (
        "artifact_root",
        "checkpoint_path",
        "config_path",
        "encoder_path",
        "source_path",
        "stats_path",
        "upstream_root",
    ):
        raw = document[field]
        if not isinstance(raw, str):
            raise TypeError(f"ACT {field} must be a string")
        document[field] = Path(raw)
    profile = document["profile"]
    if not isinstance(profile, str):
        raise TypeError("ACT profile must be a string")
    document["profile"] = OfficialACTProfile(profile)
    return ACTArtifactManifest(
        task_id=cast(str, document["task_id"]),
        profile=cast(OfficialACTProfile, document["profile"]),
        artifact_root=cast(Path, document["artifact_root"]),
        checkpoint_path=cast(Path, document["checkpoint_path"]),
        checkpoint_sha256=cast(str, document["checkpoint_sha256"]),
        stats_path=cast(Path, document["stats_path"]),
        stats_sha256=cast(str, document["stats_sha256"]),
        encoder_path=cast(Path, document["encoder_path"]),
        encoder_sha256=cast(str, document["encoder_sha256"]),
        upstream_root=cast(Path, document["upstream_root"]),
        upstream_commit=cast(str, document["upstream_commit"]),
        source_path=cast(Path, document["source_path"]),
        source_sha256=cast(str, document["source_sha256"]),
        config_path=cast(Path, document["config_path"]),
        config_sha256=cast(str, document["config_sha256"]),
        semantic_version=cast(str, document["semantic_version"]),
    )


def load_act_artifact_manifest(path: Path) -> ACTArtifactManifest:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("ACT artifact manifest must be a regular file")
    raw = selected.read_bytes()
    value = strict_json_bytes(raw, "ACT artifact manifest")
    manifest = act_artifact_manifest_from_dict(value)
    if canonical_json_bytes(act_artifact_manifest_to_dict(manifest)) != raw:
        raise ValueError("ACT artifact manifest is not canonical")
    return manifest


__all__ = [
    "ACTArtifactManifest",
    "ACTLoadRequest",
    "act_artifact_manifest_from_dict",
    "act_artifact_manifest_to_dict",
    "build_act_artifact_manifest",
    "load_act_artifact_manifest",
]

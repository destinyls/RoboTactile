"""Resolve ACT runtime arguments from one typed integration config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.integrations.act.artifacts import (
    load_act_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
    load_n0_twam_artifact_manifest,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.integrations.registry import (
    load_model_integration_config,
)


@dataclass(frozen=True)
class ACTRuntimeArtifacts:
    artifact_root: Path
    stats_sha256: str
    encoder_sha256: str
    manifest_path: Path


@dataclass(frozen=True)
class N0RuntimeArtifacts:
    """Validated official N0 serving artifacts selected by one config."""

    manifest: N0TWAMArtifactManifest
    manifest_path: Path


def resolve_act_runtime_artifacts(config_path: Path) -> ACTRuntimeArtifacts:
    """Strict-load the ACT config and its canonical artifact manifest."""

    selected_config = Path(config_path).absolute()
    config = load_model_integration_config("act", selected_config)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = selected_config.parent / manifest_path
    manifest = load_act_artifact_manifest(manifest_path)
    return ACTRuntimeArtifacts(
        artifact_root=manifest.artifact_root,
        stats_sha256=manifest.stats_sha256,
        encoder_sha256=manifest.encoder_sha256,
        manifest_path=manifest_path,
    )


def resolve_n0_runtime_artifacts(config_path: Path) -> N0RuntimeArtifacts:
    """Strict-load and re-hash the official N0 artifact configuration."""

    selected_config = Path(config_path).absolute()
    config = load_model_integration_config("n0_twam", selected_config)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = selected_config.parent / manifest_path
    manifest = load_n0_twam_artifact_manifest(manifest_path)
    validate_n0_twam_artifact(manifest)
    return N0RuntimeArtifacts(manifest=manifest, manifest_path=manifest_path)


__all__ = [
    "ACTRuntimeArtifacts",
    "N0RuntimeArtifacts",
    "resolve_act_runtime_artifacts",
    "resolve_n0_runtime_artifacts",
]

"""Resolve ACT runtime arguments from one typed integration config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.integrations.act.artifacts import (
    ACTArtifactManifest,
    load_act_artifact_manifest,
)
from robotactile_benchmark.integrations.dream_tac.artifacts import (
    DreamTacArtifactManifest,
    load_dream_tac_artifact_manifest,
    validate_dream_tac_artifact,
)
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    FTP1PolicyArtifactManifest,
    load_ftp1_policy_artifact_manifest,
    validate_ftp1_policy_artifact,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
    load_n0_twam_artifact_manifest,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    N0VTLAArtifactManifest,
    load_n0_vtla_artifact_manifest,
    validate_n0_vtla_artifact,
)
from robotactile_benchmark.integrations.registry import (
    load_model_integration_config,
)
from robotactile_benchmark.policies.univtac_official_act_loading import (
    validate_official_univtac_act_artifact,
)


@dataclass(frozen=True)
class ACTRuntimeArtifacts:
    manifest: ACTArtifactManifest
    artifact_root: Path
    stats_sha256: str
    encoder_sha256: str
    manifest_path: Path
    device: str


@dataclass(frozen=True)
class N0RuntimeArtifacts:
    """Validated official N0 serving artifacts selected by one config."""

    manifest: N0TWAMArtifactManifest
    manifest_path: Path


@dataclass(frozen=True)
class DreamTacRuntimeArtifacts:
    """Validated user-supplied Dream-Tac serving artifacts."""

    manifest: DreamTacArtifactManifest
    manifest_path: Path


@dataclass(frozen=True)
class FTP1PolicyRuntimeArtifacts:
    """Validated task-specific FTP-1 serving artifacts."""

    manifest: FTP1PolicyArtifactManifest
    manifest_path: Path


@dataclass(frozen=True)
class N0VTLARuntimeArtifacts:
    """Validated official N0-VTLA serving artifacts."""

    manifest: N0VTLAArtifactManifest
    manifest_path: Path


def resolve_act_runtime_artifacts(config_path: Path) -> ACTRuntimeArtifacts:
    """Strict-load the ACT config and its canonical artifact manifest."""

    selected_config = Path(config_path).absolute()
    config = load_model_integration_config("act", selected_config)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = selected_config.parent / manifest_path
    manifest = load_act_artifact_manifest(manifest_path)
    validate_official_univtac_act_artifact(manifest)
    return ACTRuntimeArtifacts(
        manifest=manifest,
        artifact_root=manifest.artifact_root,
        stats_sha256=manifest.stats_sha256,
        encoder_sha256=manifest.encoder_sha256,
        manifest_path=manifest_path,
        device=config.device,
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


def resolve_dream_tac_runtime_artifacts(
    config_path: Path,
) -> DreamTacRuntimeArtifacts:
    """Strict-load and re-hash one Dream-Tac artifact configuration."""

    selected_config = Path(config_path).absolute()
    config = load_model_integration_config("dream_tac", selected_config)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = selected_config.parent / manifest_path
    manifest = load_dream_tac_artifact_manifest(manifest_path)
    validate_dream_tac_artifact(manifest)
    return DreamTacRuntimeArtifacts(manifest=manifest, manifest_path=manifest_path)


def resolve_ftp1_policy_runtime_artifacts(
    config_path: Path,
) -> FTP1PolicyRuntimeArtifacts:
    """Strict-load and re-hash one FTP-1 artifact configuration."""

    selected_config = Path(config_path).absolute()
    config = load_model_integration_config("ftp1_policy", selected_config)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = selected_config.parent / manifest_path
    manifest = load_ftp1_policy_artifact_manifest(manifest_path)
    validate_ftp1_policy_artifact(manifest)
    return FTP1PolicyRuntimeArtifacts(manifest=manifest, manifest_path=manifest_path)


def resolve_n0_vtla_runtime_artifacts(config_path: Path) -> N0VTLARuntimeArtifacts:
    """Strict-load and re-hash one N0-VTLA artifact configuration."""

    selected_config = Path(config_path).absolute()
    config = load_model_integration_config("n0_vtla", selected_config)
    manifest_path = Path(config.artifact_manifest)
    if not manifest_path.is_absolute():
        manifest_path = selected_config.parent / manifest_path
    manifest = load_n0_vtla_artifact_manifest(manifest_path)
    validate_n0_vtla_artifact(manifest)
    return N0VTLARuntimeArtifacts(manifest=manifest, manifest_path=manifest_path)


__all__ = [
    "ACTRuntimeArtifacts",
    "DreamTacRuntimeArtifacts",
    "FTP1PolicyRuntimeArtifacts",
    "N0RuntimeArtifacts",
    "N0VTLARuntimeArtifacts",
    "resolve_act_runtime_artifacts",
    "resolve_dream_tac_runtime_artifacts",
    "resolve_ftp1_policy_runtime_artifacts",
    "resolve_n0_runtime_artifacts",
    "resolve_n0_vtla_runtime_artifacts",
]

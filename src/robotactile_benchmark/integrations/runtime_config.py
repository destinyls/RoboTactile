"""Resolve ACT runtime arguments from one typed integration config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robotactile_benchmark.integrations.act.artifacts import (
    load_act_artifact_manifest,
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


__all__ = ["ACTRuntimeArtifacts", "resolve_act_runtime_artifacts"]

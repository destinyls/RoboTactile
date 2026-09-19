"""Path-serializable values used by the pull-out-key matrix generator."""

from __future__ import annotations

from typing import Any

from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
)

PULL_OUT_KEY_MATRIX_SEMANTIC_VERSION = "2.0"


def artifact_document(
    manifest: OfficialUniVTACACTArtifactManifest,
) -> dict[str, Any]:
    """Serialize the public fields of one pinned official ACT artifact."""

    return {
        "artifact_family": "official_univtac_policy_last",
        "task_id": manifest.task_id,
        "profile": manifest.profile.value,
        "artifact_root": str(manifest.artifact_root),
        "checkpoint_path": str(manifest.checkpoint_path),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "stats_path": str(manifest.stats_path),
        "stats_sha256": manifest.stats_sha256,
        "encoder_path": str(manifest.encoder_path),
        "encoder_sha256": manifest.encoder_sha256,
        "upstream_root": str(manifest.upstream_root),
        "upstream_commit": manifest.upstream_commit,
        "source_path": str(manifest.source_path),
        "source_sha256": manifest.source_sha256,
        "config_path": str(manifest.config_path),
        "config_sha256": manifest.config_sha256,
        "semantic_version": manifest.semantic_version,
    }


def trial_set_document(
    *,
    initial_seed: int,
    exogenous_seed: int,
    task_registry_id: str,
    upstream_commit: str,
    task_id: str,
    action_horizon: int,
    max_observation_steps: int,
) -> dict[str, Any]:
    """Freeze one paired seed row without claiming a raw dataset hash."""

    return {
        "identity_kind": "frozen_trial_set_manifest",
        "task_registry_id": task_registry_id,
        "upstream_commit": upstream_commit,
        "task_id": task_id,
        "trial_count": 1,
        "initial_seed": initial_seed,
        "exogenous_seed": exogenous_seed,
        "action_horizon": action_horizon,
        "max_observation_steps": max_observation_steps,
        "semantic_version": PULL_OUT_KEY_MATRIX_SEMANTIC_VERSION,
    }

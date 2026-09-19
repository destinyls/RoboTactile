"""Canonical Dream-Tac adapter factory."""

from __future__ import annotations

from collections.abc import Callable

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.integrations.dream_tac.artifacts import (
    DreamTacArtifactManifest,
    validate_dream_tac_artifact,
)
from robotactile_benchmark.policies.dream_tac import (
    DreamTacClient,
    DreamTacGripperMapping,
    OfficialDreamTacPolicy,
)


def load_dream_tac_adapter(
    identity: PolicyIdentity,
    manifest: DreamTacArtifactManifest,
    client_factory: Callable[[], DreamTacClient],
) -> OfficialDreamTacPolicy:
    """Bind every checkpoint/runtime identity before allocating the HTTP client."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if type(manifest) is not DreamTacArtifactManifest:
        raise TypeError("manifest must be an exact DreamTacArtifactManifest")
    if identity.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("Dream-Tac checkpoint identity mismatch")
    if identity.config_sha256 != manifest.config_sha256:
        raise ValueError("Dream-Tac configuration identity mismatch")
    if not callable(client_factory):
        raise TypeError("client_factory must be callable")
    validate_dream_tac_artifact(manifest)
    return OfficialDreamTacPolicy(
        identity,
        client_factory,
        task_id=manifest.task_id,
        instruction=manifest.instruction,
        experiment_config=manifest.experiment_config,
        gripper_mapping=DreamTacGripperMapping(manifest.gripper_mapping),
        action_horizon=manifest.action_horizon,
        gripper_threshold=manifest.gripper_threshold,
        gripper_qpos_min=manifest.gripper_qpos_min,
        gripper_qpos_max=manifest.gripper_qpos_max,
    )


__all__ = ["load_dream_tac_adapter"]

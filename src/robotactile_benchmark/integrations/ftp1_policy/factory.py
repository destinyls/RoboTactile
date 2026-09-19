"""Canonical FTP-1 adapter factory."""

from __future__ import annotations

from collections.abc import Callable

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    FTP1PolicyArtifactManifest,
)
from robotactile_benchmark.policies.ftp1_policy import (
    FTP1PolicyClient,
    OfficialFTP1Policy,
)


def load_ftp1_policy_adapter(
    identity: PolicyIdentity,
    manifest: FTP1PolicyArtifactManifest,
    client_factory: Callable[[], FTP1PolicyClient],
) -> OfficialFTP1Policy:
    """Bind every released serving artifact before allocating the client."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if type(manifest) is not FTP1PolicyArtifactManifest:
        raise TypeError("manifest must be an exact FTP1PolicyArtifactManifest")
    if identity.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("FTP-1 checkpoint identity mismatch")
    if identity.config_sha256 != manifest.config_sha256:
        raise ValueError("FTP-1 serving bundle identity mismatch")
    if not callable(client_factory):
        raise TypeError("client_factory must be callable")
    return OfficialFTP1Policy(
        identity,
        client_factory,
        task_id=manifest.task_id,
    )


__all__ = ["load_ftp1_policy_adapter"]

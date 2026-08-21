"""Canonical N0-TWAM adapter factory."""

from __future__ import annotations

from collections.abc import Callable

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
)
from robotactile_benchmark.policies.n0 import N0Policy, N0PolicyClient


def load_n0_twam_adapter(
    identity: PolicyIdentity,
    manifest: N0TWAMArtifactManifest,
    client_factory: Callable[[], N0PolicyClient],
) -> N0Policy:
    """Bind model identity before constructing the lazy typed client adapter."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if type(manifest) is not N0TWAMArtifactManifest:
        raise TypeError("manifest must be an exact N0TWAMArtifactManifest")
    if identity.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("N0-TWAM checkpoint identity mismatch")
    if identity.config_sha256 != manifest.config_sha256:
        raise ValueError("N0-TWAM config identity mismatch")
    if not callable(client_factory):
        raise TypeError("client_factory must be callable")
    return N0Policy(identity, client_factory)


__all__ = ["load_n0_twam_adapter"]

"""Canonical N0-VTLA adapter factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    N0VTLAArtifactManifest,
)
from robotactile_benchmark.policies.n0_vtla import (
    N0VTLAClient,
    OfficialN0VTLAPolicy,
)
from robotactile_benchmark.policies.tactile_availability import (
    RGBShape,
    TactileAvailabilityMode,
)


def load_n0_vtla_adapter(
    identity: PolicyIdentity,
    manifest: N0VTLAArtifactManifest,
    client_factory: Callable[[], N0VTLAClient],
    *,
    tactile_availability_mode: TactileAvailabilityMode = TactileAvailabilityMode.REQUIRED,
    tactile_zero_shape: Optional[RGBShape] = None,
    execution_profile: Optional[str] = None,
) -> OfficialN0VTLAPolicy:
    """Bind the released artifact identity before allocating the ZMQ client."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if type(manifest) is not N0VTLAArtifactManifest:
        raise TypeError("manifest must be an exact N0VTLAArtifactManifest")
    if identity.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("N0-VTLA checkpoint identity mismatch")
    if identity.config_sha256 != manifest.config_sha256:
        raise ValueError("N0-VTLA config identity mismatch")
    if not callable(client_factory):
        raise TypeError("client_factory must be callable")
    return OfficialN0VTLAPolicy(
        identity,
        client_factory,
        task_id=manifest.task_id,
        tactile_availability_mode=tactile_availability_mode,
        tactile_zero_shape=tactile_zero_shape,
        execution_profile=execution_profile,
    )


__all__ = ["load_n0_vtla_adapter"]

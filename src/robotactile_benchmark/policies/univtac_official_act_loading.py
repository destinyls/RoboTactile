"""Pinned live loader for official UniVTAC ACT ``policy_last`` artifacts.

Manifest and non-allocating validation live in sibling modules. Public imports
remain here for backward compatibility with the original loader API.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Optional

from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.policies.univtac_official_act import (
    OfficialACTProfile,
    OfficialUniVTACACTPolicy,
    _construct_pinned_upstream_runtime,
    _OwnedRuntime,
    _release_runtime,
    _torch_transform,
)
from robotactile_benchmark.policies.univtac_official_act_manifest import (
    OfficialUniVTACACTArtifactManifest,
    OfficialUniVTACACTLoadRequest,
)
from robotactile_benchmark.policies.univtac_official_act_validation import (
    _STAT_KEYS,
    _load_stats,
    _verify_manifest_files,
    validate_official_univtac_act_artifact,
)


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _construct_upstream_runtime(
    manifest: OfficialUniVTACACTArtifactManifest,
    device_name: str,
    torch_module: Any,
) -> Any:
    return _construct_pinned_upstream_runtime(
        source_path=manifest.source_path,
        source_sha256=manifest.source_sha256,
        profile=manifest.profile,
        task_id=manifest.task_id,
        encoder_path=manifest.encoder_path,
        device_name=device_name,
        torch_module=torch_module,
    )


def _validate_identity(
    identity: PolicyIdentity, request: OfficialUniVTACACTLoadRequest
) -> None:
    manifest = request.manifest
    if request.live is not True:
        raise ValueError("official ACT loading requires an explicit live request")
    if request.task_id != manifest.task_id:
        raise ValueError("official ACT request/manifest task mismatch")
    if request.profile is not manifest.profile:
        raise ValueError("official ACT request/manifest profile mismatch")
    if identity.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("official ACT checkpoint identity mismatch")
    if identity.config_sha256 != manifest.config_sha256:
        raise ValueError("official ACT config identity mismatch")
    tactile = manifest.profile is OfficialACTProfile.UNIVTAC
    if (
        identity.action_spec != ACTION_SPEC
        or identity.consumes_tactile is not tactile
        or identity.supports_structural_absence is tactile
    ):
        raise ValueError("official ACT identity/profile capabilities mismatch")


def load_official_univtac_act_policy(
    identity: PolicyIdentity,
    request: OfficialUniVTACACTLoadRequest,
) -> OfficialUniVTACACTPolicy:
    """Load a hash-authorized official ACT only after explicit live opt-in."""

    if type(identity) is not PolicyIdentity:
        raise TypeError("identity must be an exact PolicyIdentity")
    if type(request) is not OfficialUniVTACACTLoadRequest:
        raise TypeError("request must be an exact official ACT load request")
    _validate_identity(identity, request)
    manifest = request.manifest
    _verify_manifest_files(manifest)
    stats = _load_stats(manifest.stats_path)
    torch_module = _import_torch()
    runtime: Optional[Any] = None
    try:
        state = torch_module.load(
            manifest.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, Mapping):
            raise TypeError("official ACT checkpoint must contain a state mapping")
        runtime = _construct_upstream_runtime(
            manifest, request.device_name, torch_module
        )
        model = getattr(runtime, "policy", None)
        load_state_dict = getattr(model, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError("official ACT runtime lacks a policy state boundary")
        load_state_dict(state, strict=True)
        runtime.stats = stats
        owned = _OwnedRuntime(runtime, torch_module)
        runtime = None
        return OfficialUniVTACACTPolicy(
            identity,
            owned,
            artifact_task=manifest.task_id,
            profile=manifest.profile,
            input_transform=_torch_transform(torch_module),
        )
    except Exception:
        _release_runtime(runtime, torch_module)
        raise


__all__ = [
    "OfficialUniVTACACTArtifactManifest",
    "OfficialUniVTACACTLoadRequest",
    "_STAT_KEYS",
    "load_official_univtac_act_policy",
    "validate_official_univtac_act_artifact",
]

"""Official UniVTAC ACT wiring for one unqualified live benchmark run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from robotactile_benchmark.closed_loop.contracts import PolicyIdentity
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.execution.contracts import (
    ArtifactExportStatus,
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveBackendFactory,
    LivePolicyFactory,
    default_live_backend_factory,
    execute_live_univtac_run,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_loading import (
    OfficialUniVTACACTArtifactManifest,
    OfficialUniVTACACTLoadRequest,
    load_official_univtac_act_policy,
)
from robotactile_benchmark.trials import Condition


class OfficialACTPolicyLoader(Protocol):
    """Allocate one hash-checked official policy for a loaded trial."""

    def __call__(
        self,
        identity: PolicyIdentity,
        request: OfficialUniVTACACTLoadRequest,
    ) -> ClosedLoopPolicy: ...


@dataclass(frozen=True)
class OfficialACTLiveBinding:
    """The profile-specific official artifact selected for one live request."""

    manifest: OfficialUniVTACACTArtifactManifest
    load_request: OfficialUniVTACACTLoadRequest


def official_act_profile(condition: Condition) -> OfficialACTProfile:
    """Map benchmark conditions to the matched official ACT profile."""

    normalized = condition if isinstance(condition, Condition) else Condition(condition)
    if normalized is Condition.NO_TOUCH:
        return OfficialACTProfile.VISION_ONLY
    return OfficialACTProfile.UNIVTAC


def build_official_act_live_binding(
    request: LiveUniVTACRunRequest,
    *,
    artifact_root: Path,
    stats_sha256: str,
    encoder_sha256: str,
) -> OfficialACTLiveBinding:
    """Bind request identities to the frozen official ``policy_last`` layout."""

    if type(request) is not LiveUniVTACRunRequest:
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    if request.policy_kind is not LivePolicyKind.ACT:
        raise ValueError("official UniVTAC ACT execution requires policy_kind=act")
    if request.output_dir is None:
        raise ValueError("official UniVTAC ACT execution requires request.output_dir")
    if request.act_device_name is None:
        raise ValueError("official UniVTAC ACT execution requires act_device_name")
    profile = official_act_profile(request.condition)
    manifest = OfficialUniVTACACTArtifactManifest.for_shared_root(
        task_id=request.task_id,
        profile=profile,
        artifact_root=artifact_root,
        upstream_root=request.upstream_root,
        checkpoint_sha256=request.checkpoint_sha256,
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
    )
    if request.config_sha256 != manifest.config_sha256:
        raise ValueError(
            "live request config SHA256 does not match official ACT profile"
        )
    load_request = OfficialUniVTACACTLoadRequest(
        manifest=manifest,
        task_id=request.task_id,
        profile=profile,
        device_name=request.act_device_name,
        live=True,
    )
    return OfficialACTLiveBinding(manifest, load_request)


def make_official_act_policy_factory(
    binding: OfficialACTLiveBinding,
    *,
    policy_loader: Optional[OfficialACTPolicyLoader] = None,
) -> LivePolicyFactory:
    """Create the custom factory that keeps official ACT out of StrictACT gates."""

    if type(binding) is not OfficialACTLiveBinding:
        raise TypeError("binding must be an exact OfficialACTLiveBinding")
    selected_loader = (
        load_official_univtac_act_policy if policy_loader is None else policy_loader
    )
    if not callable(selected_loader):
        raise TypeError("policy_loader must be callable")

    def factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        if loaded.request.task_id != binding.manifest.task_id:
            raise ValueError("loaded task does not match official ACT artifact")
        if loaded.request.checkpoint_sha256 != binding.manifest.checkpoint_sha256:
            raise ValueError("loaded checkpoint does not match official ACT artifact")
        if loaded.request.config_sha256 != binding.manifest.config_sha256:
            raise ValueError("loaded config does not match official ACT artifact")
        return selected_loader(loaded.policy_identity, binding.load_request)

    return factory


def execute_official_act_live_run(
    request: LiveUniVTACRunRequest,
    *,
    artifact_root: Path,
    stats_sha256: str,
    encoder_sha256: str,
    backend_factory: Optional[LiveBackendFactory] = None,
    policy_loader: Optional[OfficialACTPolicyLoader] = None,
) -> LoadedLiveUniVTACArtifact:
    """Execute, export, and independently reload one unqualified live trace."""

    binding = build_official_act_live_binding(
        request,
        artifact_root=artifact_root,
        stats_sha256=stats_sha256,
        encoder_sha256=encoder_sha256,
    )
    result = execute_live_univtac_run(
        request,
        backend_factory=(
            default_live_backend_factory if backend_factory is None else backend_factory
        ),
        policy_factory=make_official_act_policy_factory(
            binding, policy_loader=policy_loader
        ),
        artifact_exporter=write_live_univtac_artifact,
    )
    if (
        result.artifact_export is not ArtifactExportStatus.EXPORTED
        or result.artifact_receipt is None
        or request.output_dir is None
    ):
        raise RuntimeError("official ACT live execution did not export an artifact")
    artifact = load_live_univtac_artifact(request.output_dir)
    if (
        artifact.run_content_sha256 != result.loaded.content_sha256
        or artifact.root_receipt.result_sha256 != result.evidence.result.sha256
        or artifact.root_receipt.simulator_qualification_claimed is not False
    ):
        raise RuntimeError("reloaded official ACT live artifact cross-link mismatch")
    return artifact


def official_act_live_summary(
    artifact: LoadedLiveUniVTACArtifact,
) -> dict[str, object]:
    """Return the canonical, explicitly unqualified CLI summary fields."""

    if type(artifact) is not LoadedLiveUniVTACArtifact:
        raise TypeError("artifact must be an exact LoadedLiveUniVTACArtifact")
    receipt = artifact.root_receipt
    result = artifact.evidence.result
    return {
        "artifact_root_sha256": artifact.external_root_sha256,
        "condition": artifact.trial.condition.value,
        "evidence_level": receipt.evidence_level,
        "simulator_qualification_claimed": receipt.simulator_qualification_claimed,
        "task_id": artifact.trial.task,
        "terminal_status": result.terminal_status.value,
        "validation_passed": result.validation_passed,
    }


__all__ = [
    "OfficialACTLiveBinding",
    "build_official_act_live_binding",
    "execute_official_act_live_run",
    "make_official_act_policy_factory",
    "official_act_live_summary",
    "official_act_profile",
]

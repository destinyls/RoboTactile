"""Official N0-TWAM UniVTAC websocket wiring for live benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple

from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import (
    ArtifactExportStatus,
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.isaac_runtime_attestation import (
    IsaacAttestationRequest,
    build_isaac_runtime_attestation,
)
from robotactile_benchmark.execution.lifecycle_watchdog import LifecycleStageJournal
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExporter,
    LiveBackendFactory,
    LivePolicyFactory,
    default_live_backend_factory,
    execute_live_univtac_run,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.execution.paired_live_univtac import (
    PairedBackendSessionFactory,
    PairedLiveUniVTACExecutionResult,
    default_paired_backend_session_factory,
    execute_paired_live_univtac_runs,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
)
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import OfficialN0Policy
from robotactile_benchmark.transport.n0_official import (
    OfficialN0Client,
    load_official_n0_rpc,
)


@dataclass(frozen=True)
class OfficialN0LiveBinding:
    """Verified model/source/endpoint selection for one live request."""

    manifest: N0TWAMArtifactManifest
    source_root: Path
    host: str
    port: int
    api_key: Optional[str] = None


@dataclass(frozen=True)
class OfficialN0PairedLiveResult:
    """Reloaded artifacts and snapshot-pairing evidence for N0-TWAM."""

    paired: PairedLiveUniVTACExecutionResult
    artifacts: Tuple[LoadedLiveUniVTACArtifact, ...]

    def __post_init__(self) -> None:
        if len(self.paired.executions) != len(self.artifacts):
            raise ValueError("paired N0 artifacts do not cover every execution")
        for execution, artifact in zip(self.paired.executions, self.artifacts):
            if (
                artifact.trial != execution.loaded.trial
                or artifact.root_receipt.result_sha256
                != execution.evidence.result.sha256
            ):
                raise ValueError("paired N0 artifact cross-link mismatch")


def build_official_n0_live_binding(
    request: LiveUniVTACRunRequest,
    *,
    manifest: N0TWAMArtifactManifest,
    source_root: Path,
    host: str,
    port: int,
    api_key: Optional[str] = None,
) -> OfficialN0LiveBinding:
    """Bind all request identities to one released task checkpoint."""

    if type(request) is not LiveUniVTACRunRequest:
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    if request.policy_kind is not LivePolicyKind.N0:
        raise ValueError("official N0 execution requires policy_kind=n0")
    if request.output_dir is None:
        raise ValueError("official N0 execution requires request.output_dir")
    if manifest.task_id != request.task_id:
        raise ValueError("N0 manifest task does not match live request")
    expected = {
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": manifest.config_sha256,
        "n0_source_commit": manifest.external_commit,
        "n0_normalizer_sha256": manifest.normalizer_sha256,
        "n0_serve_bundle_sha256": manifest.serve_bundle_sha256,
        "n0_prompt_manifest_sha256": manifest.prompt_manifest_sha256,
    }
    for name, value in expected.items():
        if getattr(request, name) != value:
            raise ValueError(f"live request {name} does not match N0 manifest")
    selected_source = Path(source_root).absolute()
    source_receipt = verify_external_checkout(
        load_integration_lock().by_id("n0_twam"), selected_source
    )
    if source_receipt.commit_sha != manifest.external_commit:
        raise ValueError("official N0 source commit does not match manifest")
    if not isinstance(host, str) or not host or host.strip() != host:
        raise ValueError("N0 host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("N0 port must be in [1,65535]")
    return OfficialN0LiveBinding(
        manifest=manifest,
        source_root=selected_source,
        host=host,
        port=port,
        api_key=api_key,
    )


def make_official_n0_policy_factory(
    binding: OfficialN0LiveBinding,
) -> LivePolicyFactory:
    """Create a fresh stateful client for every closed-loop condition."""

    if type(binding) is not OfficialN0LiveBinding:
        raise TypeError("binding must be an exact OfficialN0LiveBinding")

    def factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        if loaded.request.task_id != binding.manifest.task_id:
            raise ValueError("loaded task does not match N0 binding")

        def client_factory() -> OfficialN0Client:
            rpc = load_official_n0_rpc(
                source_root=binding.source_root,
                host=binding.host,
                port=binding.port,
                api_key=binding.api_key,
            )
            return OfficialN0Client(rpc)

        return OfficialN0Policy(
            loaded.policy_identity,
            client_factory,
            input_profile=N0_LIVE_UNIVTAC_INPUT_PROFILE,
        )

    return factory


def execute_official_n0_live_run(
    request: LiveUniVTACRunRequest,
    *,
    manifest: N0TWAMArtifactManifest,
    source_root: Path,
    host: str,
    port: int,
    api_key: Optional[str] = None,
    backend_factory: Optional[LiveBackendFactory] = None,
    lifecycle_journal: Optional[LifecycleStageJournal] = None,
    isaac_attestation_request: Optional[IsaacAttestationRequest] = None,
    action_execution_contract: Optional[str] = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> LoadedLiveUniVTACArtifact:
    """Execute, export, and reload one official N0-TWAM live trace."""

    if lifecycle_journal is not None:
        lifecycle_journal.record("source_binding")
    binding = build_official_n0_live_binding(
        request,
        manifest=manifest,
        source_root=source_root,
        host=host,
        port=port,
        api_key=api_key,
    )
    if isaac_attestation_request is not None:
        if lifecycle_journal is None:
            raise ValueError("Isaac attestation requires a lifecycle journal")
        lifecycle_journal.record("runtime_attestation")
        build_isaac_runtime_attestation(
            deployment_root=isaac_attestation_request.deployment_root,
            campaign_id=isaac_attestation_request.campaign_id,
            lifecycle_identity=lifecycle_journal.identity,
            output_path=isaac_attestation_request.output_path,
            qualification_path=isaac_attestation_request.qualification_path,
            n0_server_attestation_path=(
                isaac_attestation_request.n0_server_attestation_path
            ),
            n0_server_attestation_sha256=(
                isaac_attestation_request.n0_server_attestation_sha256
            ),
            request=request,
            manifest=manifest,
            n0_source_root=binding.source_root,
        )
    selected_capture = LiveCaptureProfile(capture_profile)
    artifact_exporter: LiveArtifactExporter = (
        write_live_univtac_artifact
        if selected_capture.is_full_trace
        else partial(
            write_live_univtac_artifact,
            capture_profile=selected_capture,
        )
    )
    result = execute_live_univtac_run(
        request,
        backend_factory=(
            default_live_backend_factory if backend_factory is None else backend_factory
        ),
        policy_factory=make_official_n0_policy_factory(binding),
        artifact_exporter=artifact_exporter,
        lifecycle_journal=lifecycle_journal,
        n0_action_execution_contract=action_execution_contract,
    )
    if (
        result.artifact_export is not ArtifactExportStatus.EXPORTED
        or result.artifact_receipt is None
        or request.output_dir is None
    ):
        raise RuntimeError("official N0 live execution did not export an artifact")
    artifact = load_live_univtac_artifact(request.output_dir)
    if (
        artifact.run_content_sha256 != result.loaded.content_sha256
        or artifact.root_receipt.result_sha256 != result.evidence.result.sha256
        or artifact.capture_profile is not selected_capture
    ):
        raise RuntimeError("reloaded official N0 artifact cross-link mismatch")
    if lifecycle_journal is not None:
        lifecycle_journal.record("artifact_verified")
        lifecycle_journal.record("completed")
    return artifact


def execute_official_n0_paired_live_runs(
    requests: Sequence[LiveUniVTACRunRequest],
    *,
    manifest: N0TWAMArtifactManifest,
    source_root: Path,
    host: str,
    port: int,
    api_key: Optional[str] = None,
    session_factory: PairedBackendSessionFactory = (
        default_paired_backend_session_factory
    ),
) -> OfficialN0PairedLiveResult:
    """Execute matched N0 conditions through one simulator snapshot session."""

    request_tuple = tuple(requests)
    bindings = tuple(
        build_official_n0_live_binding(
            request,
            manifest=manifest,
            source_root=source_root,
            host=host,
            port=port,
            api_key=api_key,
        )
        for request in request_tuple
    )
    by_request = {
        id(request): binding for request, binding in zip(request_tuple, bindings)
    }

    def policy_factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        return make_official_n0_policy_factory(by_request[id(loaded.request)])(loaded)

    paired = execute_paired_live_univtac_runs(
        request_tuple,
        session_factory=session_factory,
        policy_factory=policy_factory,
        artifact_exporter=write_live_univtac_artifact,
    )
    artifacts = []
    for request in request_tuple:
        if request.output_dir is None:
            raise ValueError("paired official N0 request requires output_dir")
        artifacts.append(load_live_univtac_artifact(request.output_dir))
    return OfficialN0PairedLiveResult(paired=paired, artifacts=tuple(artifacts))


__all__ = [
    "OfficialN0LiveBinding",
    "OfficialN0PairedLiveResult",
    "IsaacAttestationRequest",
    "build_official_n0_live_binding",
    "execute_official_n0_live_run",
    "execute_official_n0_paired_live_runs",
    "make_official_n0_policy_factory",
]

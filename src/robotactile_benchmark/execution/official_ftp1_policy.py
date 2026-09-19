"""Source-bound FTP-1 UniVTAC ZMQ wiring for live benchmark runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import (
    ArtifactExportStatus,
    LivePolicyKind,
    LiveUniVTACRunRequest,
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
from robotactile_benchmark.integrations.ftp1_policy.artifacts import (
    FTP1PolicyArtifactManifest,
)
from robotactile_benchmark.integrations.ftp1_policy.factory import (
    load_ftp1_policy_adapter,
)
from robotactile_benchmark.integrations.ftp1_policy.transport import (
    OfficialFTP1PolicyClient,
)
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)


@dataclass(frozen=True)
class OfficialFTP1LiveBinding:
    """Verified source, checkpoint, runtime metadata, and endpoint."""

    manifest: FTP1PolicyArtifactManifest
    source_root: Path
    endpoint: str


@dataclass(frozen=True)
class OfficialFTP1PairedLiveResult:
    """Paired execution plus reloaded no-clobber artifacts."""

    paired: PairedLiveUniVTACExecutionResult
    artifacts: Tuple[LoadedLiveUniVTACArtifact, ...]

    def __post_init__(self) -> None:
        if len(self.paired.executions) != len(self.artifacts):
            raise ValueError("paired FTP-1 artifacts do not cover every execution")


def build_official_ftp1_live_binding(
    request: LiveUniVTACRunRequest,
    *,
    manifest: FTP1PolicyArtifactManifest,
    source_root: Path,
    endpoint: str,
) -> OfficialFTP1LiveBinding:
    """Bind one request to a released task-specific FTP-1 checkpoint."""

    if type(request) is not LiveUniVTACRunRequest:
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    if request.policy_kind is not LivePolicyKind.FTP1_POLICY:
        raise ValueError("official FTP-1 execution requires policy_kind=ftp1_policy")
    if request.output_dir is None:
        raise ValueError("official FTP-1 execution requires request.output_dir")
    if manifest.task_id != request.task_id:
        raise ValueError("FTP-1 manifest task does not match live request")
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
            raise ValueError(f"live request {name} does not match FTP-1 manifest")
    selected_source = Path(source_root).absolute()
    receipt = verify_external_checkout(
        load_integration_lock().by_id("ftp1_policy"), selected_source
    )
    if receipt.commit_sha != manifest.external_commit:
        raise ValueError("FTP-1 source commit does not match manifest")
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
        raise ValueError("FTP-1 endpoint must be a tcp:// address")
    return OfficialFTP1LiveBinding(manifest, selected_source, endpoint)


def make_official_ftp1_policy_factory(
    binding: OfficialFTP1LiveBinding,
) -> LivePolicyFactory:
    """Create one fresh source-bound client for each paired condition."""

    if type(binding) is not OfficialFTP1LiveBinding:
        raise TypeError("binding must be an exact OfficialFTP1LiveBinding")

    def factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        if loaded.request.task_id != binding.manifest.task_id:
            raise ValueError("loaded task does not match FTP-1 binding")
        return load_ftp1_policy_adapter(
            loaded.policy_identity,
            binding.manifest,
            lambda: OfficialFTP1PolicyClient(
                binding.endpoint,
                expected_metadata=binding.manifest.transport_metadata(),
            ),
        )

    return factory


def execute_official_ftp1_live_run(
    request: LiveUniVTACRunRequest,
    *,
    manifest: FTP1PolicyArtifactManifest,
    source_root: Path,
    endpoint: str,
    backend_factory: Optional[LiveBackendFactory] = None,
    lifecycle_journal: Optional[LifecycleStageJournal] = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> LoadedLiveUniVTACArtifact:
    """Execute, export, and reload one FTP-1 live trace."""

    if lifecycle_journal is not None:
        lifecycle_journal.record("source_binding")
    binding = build_official_ftp1_live_binding(
        request,
        manifest=manifest,
        source_root=source_root,
        endpoint=endpoint,
    )
    selected_capture = LiveCaptureProfile(capture_profile)
    exporter: LiveArtifactExporter = (
        write_live_univtac_artifact
        if selected_capture.is_full_trace
        else partial(write_live_univtac_artifact, capture_profile=selected_capture)
    )
    result = execute_live_univtac_run(
        request,
        backend_factory=(
            default_live_backend_factory if backend_factory is None else backend_factory
        ),
        policy_factory=make_official_ftp1_policy_factory(binding),
        artifact_exporter=exporter,
        lifecycle_journal=lifecycle_journal,
    )
    if (
        result.artifact_export is not ArtifactExportStatus.EXPORTED
        or result.artifact_receipt is None
        or request.output_dir is None
    ):
        raise RuntimeError("official FTP-1 execution did not export an artifact")
    artifact = load_live_univtac_artifact(request.output_dir)
    if (
        artifact.run_content_sha256 != result.loaded.content_sha256
        or artifact.root_receipt.result_sha256 != result.evidence.result.sha256
        or artifact.capture_profile is not selected_capture
    ):
        raise RuntimeError("reloaded FTP-1 artifact cross-link mismatch")
    if lifecycle_journal is not None:
        lifecycle_journal.record("artifact_verified")
        lifecycle_journal.record("completed")
    return artifact


def execute_official_ftp1_paired_live_runs(
    requests: Sequence[LiveUniVTACRunRequest],
    *,
    manifest: FTP1PolicyArtifactManifest,
    source_root: Path,
    endpoint: str,
    session_factory: PairedBackendSessionFactory = (
        default_paired_backend_session_factory
    ),
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
    pre_close_publisher: Optional[
        Callable[[OfficialFTP1PairedLiveResult], None]
    ] = None,
) -> OfficialFTP1PairedLiveResult:
    """Execute matched FTP-1 conditions from one simulator snapshot."""

    request_tuple = tuple(requests)
    bindings = tuple(
        build_official_ftp1_live_binding(
            request,
            manifest=manifest,
            source_root=source_root,
            endpoint=endpoint,
        )
        for request in request_tuple
    )
    by_request = {
        id(request): binding for request, binding in zip(request_tuple, bindings)
    }

    def policy_factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        return make_official_ftp1_policy_factory(by_request[id(loaded.request)])(loaded)

    selected_capture = LiveCaptureProfile(capture_profile)
    exporter: LiveArtifactExporter = (
        write_live_univtac_artifact
        if selected_capture.is_full_trace
        else partial(write_live_univtac_artifact, capture_profile=selected_capture)
    )
    published: list[OfficialFTP1PairedLiveResult] = []

    def publish_before_close(result: PairedLiveUniVTACExecutionResult) -> None:
        wrapped = OfficialFTP1PairedLiveResult(
            result,
            _load_verified_artifacts(request_tuple, selected_capture),
        )
        if pre_close_publisher is not None:
            pre_close_publisher(wrapped)
        published.append(wrapped)

    paired = execute_paired_live_univtac_runs(
        request_tuple,
        session_factory=session_factory,
        policy_factory=policy_factory,
        artifact_exporter=exporter,
        pre_close_publisher=(
            publish_before_close if pre_close_publisher is not None else None
        ),
    )
    if published:
        if len(published) != 1 or published[0].paired != paired:
            raise RuntimeError("paired FTP-1 pre-close publication mismatch")
        return published[0]
    return OfficialFTP1PairedLiveResult(
        paired,
        _load_verified_artifacts(request_tuple, selected_capture),
    )


def _load_verified_artifacts(
    requests: Sequence[LiveUniVTACRunRequest],
    capture_profile: LiveCaptureProfile,
) -> Tuple[LoadedLiveUniVTACArtifact, ...]:
    artifacts = []
    for request in requests:
        if request.output_dir is None:
            raise ValueError("paired FTP-1 request requires output_dir")
        artifact = load_live_univtac_artifact(request.output_dir)
        if artifact.capture_profile is not capture_profile:
            raise RuntimeError("paired FTP-1 artifact capture profile mismatch")
        artifacts.append(artifact)
    return tuple(artifacts)


__all__ = [
    "OfficialFTP1LiveBinding",
    "OfficialFTP1PairedLiveResult",
    "build_official_ftp1_live_binding",
    "execute_official_ftp1_live_run",
    "execute_official_ftp1_paired_live_runs",
    "make_official_ftp1_policy_factory",
]

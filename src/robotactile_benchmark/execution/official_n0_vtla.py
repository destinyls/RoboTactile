"""Official N0-VTLA UniVTAC ZMQ wiring for live benchmark runs."""

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
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    N0VTLAArtifactManifest,
)
from robotactile_benchmark.integrations.n0_vtla.factory import load_n0_vtla_adapter
from robotactile_benchmark.integrations.n0_vtla.transport import (
    OfficialN0VTLAClient,
)
from robotactile_benchmark.integrations.provenance import (
    load_integration_lock,
    verify_external_checkout,
)


@dataclass(frozen=True)
class OfficialN0VTLALiveBinding:
    """Verified source, checkpoint, and endpoint for one live request."""

    manifest: N0VTLAArtifactManifest
    source_root: Path
    endpoint: str


@dataclass(frozen=True)
class OfficialN0VTLAPairedLiveResult:
    """Paired execution plus its reloaded no-clobber artifacts."""

    paired: PairedLiveUniVTACExecutionResult
    artifacts: Tuple[LoadedLiveUniVTACArtifact, ...]

    def __post_init__(self) -> None:
        if len(self.paired.executions) != len(self.artifacts):
            raise ValueError("paired N0-VTLA artifacts do not cover every execution")


def build_official_n0_vtla_live_binding(
    request: LiveUniVTACRunRequest,
    *,
    manifest: N0VTLAArtifactManifest,
    source_root: Path,
    endpoint: str,
) -> OfficialN0VTLALiveBinding:
    """Bind one request to the released UniVTAC N0-VTLA policy."""

    if type(request) is not LiveUniVTACRunRequest:
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    if request.policy_kind is not LivePolicyKind.N0_VTLA:
        raise ValueError("official N0-VTLA execution requires policy_kind=n0_vtla")
    if request.output_dir is None:
        raise ValueError("official N0-VTLA execution requires request.output_dir")
    if manifest.task_id != request.task_id:
        raise ValueError("N0-VTLA manifest task does not match live request")
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
            raise ValueError(f"live request {name} does not match N0-VTLA manifest")
    selected_source = Path(source_root).absolute()
    receipt = verify_external_checkout(
        load_integration_lock().by_id("n0_vtla"), selected_source
    )
    if receipt.commit_sha != manifest.external_commit:
        raise ValueError("official N0-VTLA source commit does not match manifest")
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
        raise ValueError("N0-VTLA endpoint must be a tcp:// address")
    return OfficialN0VTLALiveBinding(manifest, selected_source, endpoint)


def make_official_n0_vtla_policy_factory(
    binding: OfficialN0VTLALiveBinding,
) -> LivePolicyFactory:
    """Create one fresh ZMQ client for each paired condition."""

    if type(binding) is not OfficialN0VTLALiveBinding:
        raise TypeError("binding must be an exact OfficialN0VTLALiveBinding")

    def factory(loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy:
        if loaded.request.task_id != binding.manifest.task_id:
            raise ValueError("loaded task does not match N0-VTLA binding")
        return load_n0_vtla_adapter(
            loaded.policy_identity,
            binding.manifest,
            lambda: OfficialN0VTLAClient(binding.endpoint),
            tactile_availability_mode=loaded.request.tactile_availability_mode,
            tactile_zero_shape=loaded.request.tactile_zero_shape,
            execution_profile=loaded.request.n0_vtla_execution_profile,
        )

    return factory


def execute_official_n0_vtla_live_run(
    request: LiveUniVTACRunRequest,
    *,
    manifest: N0VTLAArtifactManifest,
    source_root: Path,
    endpoint: str,
    backend_factory: Optional[LiveBackendFactory] = None,
    lifecycle_journal: Optional[LifecycleStageJournal] = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> LoadedLiveUniVTACArtifact:
    """Execute, export, and reload one official N0-VTLA live trace."""

    if lifecycle_journal is not None:
        lifecycle_journal.record("source_binding")
    binding = build_official_n0_vtla_live_binding(
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
        policy_factory=make_official_n0_vtla_policy_factory(binding),
        artifact_exporter=exporter,
        lifecycle_journal=lifecycle_journal,
    )
    if (
        result.artifact_export is not ArtifactExportStatus.EXPORTED
        or result.artifact_receipt is None
        or request.output_dir is None
    ):
        raise RuntimeError("official N0-VTLA execution did not export an artifact")
    artifact = load_live_univtac_artifact(request.output_dir)
    if (
        artifact.run_content_sha256 != result.loaded.content_sha256
        or artifact.root_receipt.result_sha256 != result.evidence.result.sha256
        or artifact.capture_profile is not selected_capture
    ):
        raise RuntimeError("reloaded official N0-VTLA artifact cross-link mismatch")
    if lifecycle_journal is not None:
        lifecycle_journal.record("artifact_verified")
        lifecycle_journal.record("completed")
    return artifact


def execute_official_n0_vtla_paired_live_runs(
    requests: Sequence[LiveUniVTACRunRequest],
    *,
    manifest: N0VTLAArtifactManifest,
    source_root: Path,
    endpoint: str,
    session_factory: PairedBackendSessionFactory = default_paired_backend_session_factory,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
    pre_close_publisher: Optional[
        Callable[[OfficialN0VTLAPairedLiveResult], None]
    ] = None,
) -> OfficialN0VTLAPairedLiveResult:
    """Execute matched N0-VTLA conditions from one simulator snapshot."""

    request_tuple = tuple(requests)
    bindings = tuple(
        build_official_n0_vtla_live_binding(
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
        binding = by_request[id(loaded.request)]
        return make_official_n0_vtla_policy_factory(binding)(loaded)

    selected_capture = LiveCaptureProfile(capture_profile)
    exporter: LiveArtifactExporter = (
        write_live_univtac_artifact
        if selected_capture.is_full_trace
        else partial(write_live_univtac_artifact, capture_profile=selected_capture)
    )
    published: list[OfficialN0VTLAPairedLiveResult] = []

    def publish_before_close(result: PairedLiveUniVTACExecutionResult) -> None:
        wrapped = OfficialN0VTLAPairedLiveResult(
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
            raise RuntimeError("paired N0-VTLA pre-close publication mismatch")
        return published[0]
    return OfficialN0VTLAPairedLiveResult(
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
            raise ValueError("paired official N0-VTLA request requires output_dir")
        artifact = load_live_univtac_artifact(request.output_dir)
        if artifact.capture_profile is not capture_profile:
            raise RuntimeError("paired N0-VTLA artifact capture profile mismatch")
        artifacts.append(artifact)
    return tuple(artifacts)


__all__ = [
    "OfficialN0VTLALiveBinding",
    "OfficialN0VTLAPairedLiveResult",
    "build_official_n0_vtla_live_binding",
    "execute_official_n0_vtla_live_run",
    "execute_official_n0_vtla_paired_live_runs",
    "make_official_n0_vtla_policy_factory",
]

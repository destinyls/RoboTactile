"""Production wiring for one bounded, unqualified live UniVTAC execution."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_isaac import UniVTACIsaacBackend
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import (
    UNQUALIFIED_EXECUTION_EVIDENCE,
    ArtifactExportStatus,
    LiveExecutionUnavailableError,
    LivePolicyKind,
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_run,
)
from robotactile_benchmark.policies.act_loading import (
    load_matched_no_touch_policy,
    load_strict_act_policy,
)
from robotactile_benchmark.policies.n0 import N0Policy
from robotactile_benchmark.transport.n0_client import N0Client, N0Transport
from robotactile_benchmark.trials import Condition

LIVE_ARTIFACT_EVIDENCE_LEVEL = "unqualified_live_univtac_execution_v1"


class LiveBackendFactory(Protocol):
    """Construct a fresh single-use backend for one loaded request."""

    def __call__(self, loaded: LoadedLiveUniVTACRun) -> SimulationBackend: ...


class LivePolicyFactory(Protocol):
    """Construct a fresh single-use policy for one loaded request."""

    def __call__(self, loaded: LoadedLiveUniVTACRun) -> ClosedLoopPolicy: ...


class N0TransportFactory(Protocol):
    """Acquire a production N0 transport only when N0Policy resets."""

    def __call__(self) -> N0Transport: ...


class LiveArtifactExporter(Protocol):
    """Future hook for a live-specific atomic artifact contract."""

    def __call__(
        self,
        output: Path,
        loaded: LoadedLiveUniVTACRun,
        evidence: ClosedLoopExecutionEvidence,
    ) -> LiveArtifactExportReceipt: ...


@dataclass(frozen=True)
class LiveArtifactExportReceipt:
    """Minimal cross-links required from a future live artifact exporter."""

    evidence_level: str
    run_content_sha256: str
    trial_manifest_sha256: str
    result_sha256: str
    semantic_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.evidence_level != LIVE_ARTIFACT_EVIDENCE_LEVEL:
            raise ValueError("live artifact evidence level mismatch")
        if self.semantic_version != "1.0":
            raise ValueError("live artifact receipt semantic version mismatch")
        for name in (
            "run_content_sha256",
            "trial_manifest_sha256",
            "result_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256")

    @classmethod
    def for_execution(
        cls,
        loaded: LoadedLiveUniVTACRun,
        evidence: ClosedLoopExecutionEvidence,
    ) -> LiveArtifactExportReceipt:
        """Build path-free links; this does not itself write an artifact."""

        return cls(
            evidence_level=LIVE_ARTIFACT_EVIDENCE_LEVEL,
            run_content_sha256=loaded.content_sha256,
            trial_manifest_sha256=loaded.trial.sha256,
            result_sha256=evidence.result.sha256,
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class LiveUniVTACExecutionResult:
    """In-memory execution capture that never self-claims simulator qualification."""

    loaded: LoadedLiveUniVTACRun
    evidence: ClosedLoopExecutionEvidence
    artifact_export: ArtifactExportStatus
    artifact_receipt: Optional[LiveArtifactExportReceipt]
    evidence_level: str = UNQUALIFIED_EXECUTION_EVIDENCE
    simulator_qualification_claimed: bool = False

    def __post_init__(self) -> None:
        if self.evidence_level != UNQUALIFIED_EXECUTION_EVIDENCE:
            raise ValueError("execution evidence level cannot be upgraded")
        if self.simulator_qualification_claimed is not False:
            raise ValueError("Task 6D cannot claim simulator qualification")
        status = (
            self.artifact_export
            if isinstance(self.artifact_export, ArtifactExportStatus)
            else ArtifactExportStatus(self.artifact_export)
        )
        object.__setattr__(self, "artifact_export", status)
        if (status is ArtifactExportStatus.EXPORTED) != (
            self.artifact_receipt is not None
        ):
            raise ValueError("artifact export status and receipt disagree")
        if self.evidence.result.trial_manifest_sha256 != self.loaded.trial.sha256:
            raise ValueError("execution result does not match loaded trial")
        if self.evidence.result.run_spec_sha256 != self.loaded.run_spec.sha256:
            raise ValueError("execution result does not match loaded run spec")
        if self.artifact_receipt is not None and self.artifact_receipt != (
            LiveArtifactExportReceipt.for_execution(self.loaded, self.evidence)
        ):
            raise ValueError("live artifact receipt cross-links do not match execution")


def _live_dependency_preflight() -> None:
    if platform.system() != "Linux":
        raise LiveExecutionUnavailableError(
            "live_univtac_requires_linux",
            "live UniVTAC execution requires Linux; CPU fake evidence is not substituted",
        )
    try:
        isaac_spec = importlib.util.find_spec("isaaclab.app")
    except (ImportError, ModuleNotFoundError, AttributeError) as error:
        raise LiveExecutionUnavailableError(
            "isaaclab_app_unavailable", "isaaclab.app dependency preflight failed"
        ) from error
    if isaac_spec is None:
        raise LiveExecutionUnavailableError(
            "isaaclab_app_unavailable", "isaaclab.app is unavailable"
        )


def default_live_backend_factory(
    loaded: LoadedLiveUniVTACRun,
) -> UniVTACIsaacBackend:
    """Launch AppLauncher-first, then transfer runtime ownership to the backend."""

    _live_dependency_preflight()
    request = loaded.request
    runtime = launch_univtac_runtime(
        loaded.backend_config,
        upstream_root=request.upstream_root,
        runtime_dir=request.runtime_dir,
        launcher_args=request.launcher_args,
        device=request.simulator_device,
    )
    try:
        return UniVTACIsaacBackend(loaded.backend_config, runtime)
    except Exception:
        runtime.close_runtime()
        raise


def default_live_policy_factory(
    loaded: LoadedLiveUniVTACRun,
    *,
    n0_transport_factory: Optional[N0TransportFactory] = None,
) -> ClosedLoopPolicy:
    """Load qualified ACT or construct a lazy typed N0 client boundary."""

    request = loaded.request
    if request.policy_kind is LivePolicyKind.ACT:
        if request.act_device_name is None:
            raise ValueError("ACT request lost its device identity")
        return load_strict_act_policy(
            loaded.policy_identity,
            task=request.task_id,
            device_name=request.act_device_name,
        )
    if n0_transport_factory is None:
        raise LiveExecutionUnavailableError(
            "n0_transport_unavailable",
            "N0 live execution requires an injected production transport factory",
        )

    def client_factory() -> N0Client:
        return N0Client(
            n0_transport_factory(),
            expected_source_commit=_required_n0(request.n0_source_commit, "source"),
            expected_checkpoint_sha256=request.checkpoint_sha256,
            expected_config_sha256=request.config_sha256,
            expected_normalizer_sha256=_required_n0(
                request.n0_normalizer_sha256, "normalizer"
            ),
            expected_serve_bundle_sha256=_required_n0(
                request.n0_serve_bundle_sha256, "serve bundle"
            ),
            expected_prompt_manifest_sha256=_required_n0(
                request.n0_prompt_manifest_sha256, "prompt manifest"
            ),
        )

    return N0Policy(loaded.policy_identity, client_factory)


def _required_n0(value: Optional[str], name: str) -> str:
    if value is None:
        raise ValueError(f"N0 request lost its {name} identity")
    return value


def execute_live_univtac_run(
    request: LiveUniVTACRunRequest,
    *,
    backend_factory: LiveBackendFactory = default_live_backend_factory,
    policy_factory: Optional[LivePolicyFactory] = None,
    n0_transport_factory: Optional[N0TransportFactory] = None,
    artifact_exporter: Optional[LiveArtifactExporter] = None,
) -> LiveUniVTACExecutionResult:
    """Consume fresh resources in the runner and return path-free typed evidence."""

    _preflight_capabilities(request, policy_factory, n0_transport_factory)
    loaded = load_live_univtac_run(request)
    backend: Optional[SimulationBackend] = None
    try:
        backend = backend_factory(loaded)
        policy = (
            policy_factory(loaded)
            if policy_factory is not None
            else default_live_policy_factory(
                loaded, n0_transport_factory=n0_transport_factory
            )
        )
    except Exception:
        if backend is not None:
            backend.close()
        raise
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        policy,
        fault_manifest=loaded.fault_manifest,
        rest_references=loaded.rest_references,
    )
    export_status, receipt = _export_artifact(loaded, evidence, artifact_exporter)
    return LiveUniVTACExecutionResult(
        loaded=loaded,
        evidence=evidence,
        artifact_export=export_status,
        artifact_receipt=receipt,
    )


def _preflight_capabilities(
    request: LiveUniVTACRunRequest,
    policy_factory: Optional[LivePolicyFactory],
    n0_transport_factory: Optional[N0TransportFactory],
) -> None:
    if request.condition is Condition.NO_TOUCH and policy_factory is None:
        load_matched_no_touch_policy(request.matched_no_touch_artifact_path)
    if (
        request.policy_kind is LivePolicyKind.N0
        and policy_factory is None
        and n0_transport_factory is None
    ):
        raise LiveExecutionUnavailableError(
            "n0_transport_unavailable",
            "N0 live execution requires an injected production transport factory",
        )


def _export_artifact(
    loaded: LoadedLiveUniVTACRun,
    evidence: ClosedLoopExecutionEvidence,
    exporter: Optional[LiveArtifactExporter],
) -> tuple[ArtifactExportStatus, Optional[LiveArtifactExportReceipt]]:
    output = loaded.request.output_dir
    if output is None:
        return ArtifactExportStatus.NOT_REQUESTED, None
    if exporter is None:
        return ArtifactExportStatus.UNSUPPORTED_CONTRACT, None
    receipt = exporter(output, loaded, evidence)
    if not isinstance(receipt, LiveArtifactExportReceipt):
        raise TypeError("live artifact exporter must return LiveArtifactExportReceipt")
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError("live artifact exporter did not publish a real directory")
    return ArtifactExportStatus.EXPORTED, receipt

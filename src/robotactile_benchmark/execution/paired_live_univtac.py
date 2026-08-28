"""Single-process snapshot/replay execution for matched UniVTAC conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

from robotactile_benchmark.backends.univtac_factory import launch_univtac_runtime
from robotactile_benchmark.backends.univtac_pairing import (
    PAIRING_EVIDENCE_LEVEL,
    UniVTACPairedBackendSession,
    UniVTACPairedResetReceipt,
    UniVTACPairingError,
)
from robotactile_benchmark.closed_loop.interfaces import (
    SimulationBackend,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExporter,
    LivePolicyFactory,
    LiveUniVTACExecutionResult,
    N0TransportFactory,
    _export_artifact,
    _live_dependency_preflight,
    _preflight_capabilities,
    default_live_policy_factory,
)
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_run,
)
from robotactile_benchmark.trials import Condition, TerminalStatus

PAIRED_EXECUTION_EVIDENCE_LEVEL = "unqualified_paired_live_univtac_execution_v1"
PAIRED_EXECUTION_SEMANTIC_VERSION = "1.0"


class PairedBackendSession(Protocol):
    """Lease single-use backends while retaining one simulator runtime."""

    @property
    def reset_receipt(self) -> UniVTACPairedResetReceipt: ...

    def new_backend(self) -> SimulationBackend: ...

    def close(self) -> None: ...


class PairedBackendSessionFactory(Protocol):
    def __call__(self, loaded: LoadedLiveUniVTACRun) -> PairedBackendSession: ...


@dataclass(frozen=True)
class PairedLiveUniVTACExecutionResult:
    """Matched executions plus a reset-equivalence receipt."""

    executions: Tuple[LiveUniVTACExecutionResult, ...]
    reset_receipt: UniVTACPairedResetReceipt
    witness_indices: Tuple[Optional[int], ...]
    group_content_sha256: str
    evidence_level: str = PAIRED_EXECUTION_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = PAIRED_EXECUTION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if self.evidence_level != PAIRED_EXECUTION_EVIDENCE_LEVEL:
            raise ValueError("paired execution evidence level mismatch")
        if self.simulator_qualification_claimed is not False:
            raise ValueError("paired execution cannot claim simulator qualification")
        if self.semantic_version != PAIRED_EXECUTION_SEMANTIC_VERSION:
            raise ValueError("paired execution semantic version mismatch")
        if not self.executions or len(self.executions) != len(self.witness_indices):
            raise ValueError("paired executions and witness indices must align")
        if not self.reset_receipt.all_exact:
            raise ValueError("paired execution requires an exact reset receipt")
        expected_hash = canonical_hash(
            {
                "namespace": PAIRED_EXECUTION_EVIDENCE_LEVEL,
                "run_content_sha256": tuple(
                    item.loaded.content_sha256 for item in self.executions
                ),
                "result_sha256": tuple(
                    item.evidence.result.sha256 for item in self.executions
                ),
                "reset_receipt_sha256": self.reset_receipt.sha256,
                "witness_indices": self.witness_indices,
            }
        )
        if self.group_content_sha256 != expected_hash:
            raise ValueError("paired execution group hash mismatch")
        self._validate_witness_links()

    def _validate_witness_links(self) -> None:
        for execution, index in zip(self.executions, self.witness_indices):
            initial = execution.evidence.result.initial_state_sha256
            if index is None:
                if (
                    initial is not None
                    or execution.evidence.result.terminal_status
                    is not TerminalStatus.UNSUPPORTED_CONTRACT
                ):
                    raise ValueError("non-reset paired result is not unsupported")
                continue
            witness = self.reset_receipt.witnesses[index]
            if initial != witness.simulator_state_sha256:
                raise ValueError("paired execution/reset witness mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "group_content_sha256": self.group_content_sha256,
            "reset_receipt": self.reset_receipt.to_dict(),
            "reset_receipt_sha256": self.reset_receipt.sha256,
            "semantic_version": self.semantic_version,
            "executions": [
                {
                    "condition": item.loaded.trial.condition.value,
                    "run_content_sha256": item.loaded.content_sha256,
                    "result_sha256": item.evidence.result.sha256,
                    "terminal_status": item.evidence.result.terminal_status.value,
                    "validation_passed": item.evidence.result.validation_passed,
                    "witness_index": witness_index,
                }
                for item, witness_index in zip(self.executions, self.witness_indices)
            ],
        }


def default_paired_backend_session_factory(
    loaded: LoadedLiveUniVTACRun,
) -> UniVTACPairedBackendSession:
    """Launch one AppLauncher runtime and retain it for all paired conditions."""

    _live_dependency_preflight()
    request = loaded.request
    runtime = launch_univtac_runtime(
        loaded.backend_config,
        upstream_root=request.upstream_root,
        runtime_dir=request.runtime_dir,
        initial_seed=loaded.trial.initial_seed,
        launcher_args=request.launcher_args,
        device=request.simulator_device,
    )
    try:
        return UniVTACPairedBackendSession(loaded.backend_config, runtime)
    except Exception:
        runtime.close_runtime()
        raise


def execute_paired_live_univtac_runs(
    requests: Sequence[LiveUniVTACRunRequest],
    *,
    session_factory: PairedBackendSessionFactory = (
        default_paired_backend_session_factory
    ),
    policy_factory: Optional[LivePolicyFactory] = None,
    n0_transport_factory: Optional[N0TransportFactory] = None,
    artifact_exporter: Optional[LiveArtifactExporter] = None,
) -> PairedLiveUniVTACExecutionResult:
    """Run matched requests from one canonical state in one simulator process."""

    request_tuple = tuple(requests)
    for request in request_tuple:
        _preflight_capabilities(request, policy_factory, n0_transport_factory)
    loaded = tuple(load_live_univtac_run(request) for request in request_tuple)
    _validate_paired_group(loaded)
    session = session_factory(loaded[0])
    executions: list[LiveUniVTACExecutionResult] = []
    witness_indices: list[Optional[int]] = []
    try:
        for item in loaded:
            execution, witness_index = _execute_one(
                item,
                session,
                policy_factory=policy_factory,
                n0_transport_factory=n0_transport_factory,
                artifact_exporter=artifact_exporter,
            )
            executions.append(execution)
            witness_indices.append(witness_index)
        receipt = session.reset_receipt
        if not receipt.all_exact:
            raise UniVTACPairingError(
                "reset_equivalence_mismatch",
                "paired execution contains a non-equivalent reset witness",
            )
    finally:
        session.close()
    execution_tuple = tuple(executions)
    index_tuple = tuple(witness_indices)
    group_hash = canonical_hash(
        {
            "namespace": PAIRED_EXECUTION_EVIDENCE_LEVEL,
            "run_content_sha256": tuple(
                item.loaded.content_sha256 for item in execution_tuple
            ),
            "result_sha256": tuple(
                item.evidence.result.sha256 for item in execution_tuple
            ),
            "reset_receipt_sha256": receipt.sha256,
            "witness_indices": index_tuple,
        }
    )
    return PairedLiveUniVTACExecutionResult(
        executions=execution_tuple,
        reset_receipt=receipt,
        witness_indices=index_tuple,
        group_content_sha256=group_hash,
    )


def _execute_one(
    loaded: LoadedLiveUniVTACRun,
    session: PairedBackendSession,
    *,
    policy_factory: Optional[LivePolicyFactory],
    n0_transport_factory: Optional[N0TransportFactory],
    artifact_exporter: Optional[LiveArtifactExporter],
) -> tuple[LiveUniVTACExecutionResult, Optional[int]]:
    before = _witness_count(session)
    backend = session.new_backend()
    try:
        policy = (
            policy_factory(loaded)
            if policy_factory is not None
            else default_live_policy_factory(
                loaded, n0_transport_factory=n0_transport_factory
            )
        )
    except Exception:
        backend.close()
        raise
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        policy,
        fault_manifest=loaded.fault_manifest,
        rest_references=loaded.rest_references,
        initial_state_policy=loaded.request.initial_state_policy,
    )
    after = _witness_count(session)
    if after == before + 1:
        accepted_index = before
        witness_index: Optional[int] = accepted_index
        if not session.reset_receipt.witnesses[accepted_index].exact_match:
            raise UniVTACPairingError(
                "reset_equivalence_mismatch",
                "paired condition failed the exact reset-equivalence gate",
            )
    elif after == before and (
        evidence.result.terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
        and evidence.result.initial_state_sha256 is None
    ):
        witness_index = None
    else:
        raise UniVTACPairingError(
            "paired_reset_incomplete",
            "paired condition did not produce one accepted reset witness",
        )
    export_status, artifact_receipt = _export_artifact(
        loaded, evidence, artifact_exporter
    )
    return (
        LiveUniVTACExecutionResult(
            loaded=loaded,
            evidence=evidence,
            artifact_export=export_status,
            artifact_receipt=artifact_receipt,
        ),
        witness_index,
    )


def _witness_count(session: PairedBackendSession) -> int:
    try:
        return len(session.reset_receipt.witnesses)
    except UniVTACPairingError as error:
        if error.code == "canonical_reset_missing":
            return 0
        raise


def _validate_paired_group(loaded: Tuple[LoadedLiveUniVTACRun, ...]) -> None:
    if len(loaded) < 2:
        raise ValueError("paired live execution requires at least two requests")
    if loaded[0].trial.condition is not Condition.CLEAN:
        raise ValueError("paired live execution must start with the clean condition")
    first = loaded[0]
    if len({item.content_sha256 for item in loaded}) != len(loaded):
        raise ValueError("paired live requests must be content-address unique")
    if any(item.trial.pair_key != first.trial.pair_key for item in loaded):
        raise ValueError("paired live requests do not share one pair key")
    if any(item.backend_config != first.backend_config for item in loaded):
        raise ValueError("paired live requests do not share one backend config")
    invariant_paths = (
        "upstream_root",
        "simulator_device",
        "launcher_args",
        "policy_kind",
    )
    for name in invariant_paths:
        expected = getattr(first.request, name)
        if any(getattr(item.request, name) != expected for item in loaded):
            raise ValueError(f"paired live request field mismatch: {name}")
    outputs = tuple(
        item.request.output_dir
        for item in loaded
        if item.request.output_dir is not None
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("paired live request output directories must be unique")


__all__ = [
    "PAIRED_EXECUTION_EVIDENCE_LEVEL",
    "PAIRED_EXECUTION_SEMANTIC_VERSION",
    "PAIRING_EVIDENCE_LEVEL",
    "PairedBackendSession",
    "PairedBackendSessionFactory",
    "PairedLiveUniVTACExecutionResult",
    "default_paired_backend_session_factory",
    "execute_paired_live_univtac_runs",
]

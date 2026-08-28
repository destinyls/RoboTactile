"""Independent semantic and hash validation for live UniVTAC artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from robotactile_benchmark.closed_loop.artifact_semantics import (
    validate_trial_execution_semantics,
)
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_MAX_TRACE_RECORDS,
    LiveArtifactRootReceipt,
    LiveArtifactValidationError,
)
from robotactile_benchmark.execution.live_artifacts_values import (
    run_content_sha256_from_identity,
    source_binding_sha256,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.trials import Condition, TrialManifest
from robotactile_benchmark.validators import ValidationReport, validate_delivery


def validate_live_write_inputs(
    loaded: LoadedLiveUniVTACRun, evidence: ClosedLoopExecutionEvidence
) -> None:
    if not isinstance(loaded, LoadedLiveUniVTACRun):
        raise TypeError("live exporter requires LoadedLiveUniVTACRun")
    if not isinstance(evidence, ClosedLoopExecutionEvidence):
        raise TypeError("live exporter requires ClosedLoopExecutionEvidence")
    if (
        evidence.result.trial_manifest_sha256 != loaded.trial.sha256
        or evidence.result.run_spec_sha256 != loaded.run_spec.sha256
    ):
        raise LiveArtifactValidationError("live execution does not match loaded run")
    _validate_evidence_semantics(
        loaded.trial,
        loaded.run_spec,
        loaded.fault_manifest,
        loaded.rest_references,
        evidence,
    )


def validate_loaded_live_links(
    identity: Mapping[str, object],
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault: FaultManifest | None,
    rest_references: RestReferenceBundle | None,
    evidence: ClosedLoopExecutionEvidence,
    validation_links: Mapping[str, str | None],
    receipt: LiveArtifactRootReceipt,
) -> None:
    result = evidence.result
    finalization = evidence.finalization
    expected_fault = None if fault is None else fault.sha256
    expected_rest = None if rest_references is None else rest_references.sha256
    checks = (
        canonical_hash(identity) == receipt.request_sha256,
        run_content_sha256_from_identity(identity) == receipt.run_content_sha256,
        source_binding_sha256(identity) == receipt.source_binding_sha256,
        trial.sha256 == receipt.trial_manifest_sha256,
        trial.pair_key == receipt.pair_key,
        run_spec.sha256 == receipt.run_spec_sha256,
        expected_fault == receipt.fault_manifest_sha256,
        expected_rest == receipt.rest_references_sha256,
        result.sha256 == receipt.result_sha256,
        result.terminal_trace_sha256 == receipt.terminal_trace_sha256,
        result.action_trace_sha256 == receipt.action_trace_sha256,
        result.clean_trace_sha256 == receipt.clean_trace_sha256,
        result.delivered_trace_sha256 == receipt.delivered_trace_sha256,
    )
    if not all(checks):
        raise LiveArtifactValidationError("live artifact cross-link mismatch")
    expected_links = {
        "manifest_sha256": None
        if finalization is None
        else finalization.manifest_sha256,
        "clean_trace_sha256": (
            None if finalization is None else finalization.clean_trace_sha256
        ),
        "delivered_trace_sha256": (
            None if finalization is None else finalization.delivered_trace_sha256
        ),
    }
    if dict(validation_links) != expected_links:
        raise LiveArtifactValidationError("live validation links disagree")
    _validate_evidence_semantics(trial, run_spec, fault, rest_references, evidence)


def _validate_evidence_semantics(
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault: FaultManifest | None,
    rest_references: RestReferenceBundle | None,
    evidence: ClosedLoopExecutionEvidence,
) -> None:
    if len(evidence.action_entries) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live action trace exceeds cap")
    if len(evidence.transition_entries) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live transition trace exceeds cap")
    faulted = trial.condition in {Condition.FAULTED, Condition.RESTORED}
    if faulted != (fault is not None) or trial.fault_manifest_sha256 != (
        None if fault is None else fault.sha256
    ):
        raise LiveArtifactValidationError("live trial/fault identity mismatch")
    if rest_references is not None and fault is None:
        raise LiveArtifactValidationError("live rest references require a fault")
    finalization = evidence.finalization
    if finalization is None:
        return
    validate_trial_execution_semantics(
        trial, run_spec, evidence.result, finalization, evidence.action_entries
    )
    if finalization.validation is None:
        if fault is not None:
            raise LiveArtifactValidationError("fault trace is missing validation")
        return
    if fault is None:
        raise LiveArtifactValidationError("identity trace cannot carry validation")
    rerun = validate_delivery(
        finalization.clean_records,
        finalization.delivered_records,
        fault,
        rest_references,
    )
    if _report_hash(rerun) != _report_hash(finalization.validation):
        raise LiveArtifactValidationError("independent live validation disagrees")


def _report_hash(report: ValidationReport) -> str:
    return canonical_hash(
        {
            "passed": report.passed,
            "failure_codes": report.failure_codes,
            "failures": report.failures,
            "metrics": report.metrics,
        }
    )

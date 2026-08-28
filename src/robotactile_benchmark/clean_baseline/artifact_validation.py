"""Cross-link validation for one clean live artifact."""

from __future__ import annotations

from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.execution.live_artifacts_values import (
    live_request_identity,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.trials import Condition


def validate_clean_artifact(
    run: LoadedLiveUniVTACRun,
    spec: CleanCampaignTrialSpec,
    artifact: LoadedLiveUniVTACArtifact,
    *,
    allow_compact: bool = False,
) -> None:
    """Require one strict artifact to preserve the frozen clean identity."""

    if type(allow_compact) is not bool:
        raise TypeError("allow_compact must be bool")
    if not artifact.capture_profile.is_full_trace and not allow_compact:
        raise CleanCampaignError(
            "compact live artifact requires explicit diagnostic aggregation"
        )
    result = artifact.evidence.result
    finalization = artifact.evidence.finalization
    checks = (
        artifact.request_identity == live_request_identity(run),
        artifact.run_content_sha256 == spec.run_content_sha256,
        artifact.trial == run.trial,
        artifact.run_spec == run.run_spec,
        artifact.fault_manifest is None,
        artifact.rest_references is None,
        artifact.trial.condition is Condition.CLEAN,
        artifact.root_receipt.fault_manifest_sha256 is None,
        artifact.root_receipt.rest_references_sha256 is None,
        artifact.root_receipt.simulator_qualification_claimed is False,
        result.trial_manifest_sha256 == spec.trial_manifest_sha256,
        result.pair_key == spec.pair_key,
        result.run_spec_sha256 == spec.run_spec_sha256,
    )
    if not all(checks):
        raise CleanCampaignError("live artifact does not match the frozen clean trial")
    if result.clean_trace_sha256 != result.delivered_trace_sha256:
        raise CleanCampaignError("clean result did not preserve identity delivery")
    if finalization is not None and (
        finalization.validation is not None
        or finalization.manifest_sha256 is not None
        or finalization.clean_trace_sha256 != finalization.delivered_trace_sha256
    ):
        raise CleanCampaignError(
            "clean live artifact did not preserve identity delivery"
        )


__all__ = ["validate_clean_artifact"]

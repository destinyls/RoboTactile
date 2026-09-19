"""Shared strict validation for captured online fault delivery traces."""

from __future__ import annotations

from typing import Optional, Sequence

from robotactile_benchmark.contracts import EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.validators import ValidationReport, validate_delivery


def validate_online_delivery(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    rest_references: Optional[RestReferenceBundle] = None,
) -> ValidationReport:
    """Recompute delivery validity, including the historical A2 resume rule.

    An invalid online episode must remain independently exportable/readable.
    Unlike offline replay, online capture may end before a planned recovery.
    """
    validation = validate_delivery(
        clean_records, delivered_records, manifest, rest_references
    )
    if manifest.operator_id != "A2_frame_erasure":
        return validation
    if manifest.parameters.get("a2_end_policy") == "episode_censored_v1":
        observed_erasures = tuple(
            manifest.start_index + int(offset)
            for offset in manifest.parameters["erased_offsets"]
            if manifest.start_index + int(offset) < len(clean_records)
        )
        right_censored = bool(observed_erasures) and (
            observed_erasures[-1] == len(clean_records) - 1
        )
        resume_observed = (
            bool(observed_erasures) and not right_censored and validation.passed
        )
        if not validation.passed:
            resume_status = "invalid_delivery"
        elif right_censored:
            resume_status = "right_censored"
        elif resume_observed:
            resume_status = "observed_resume"
        else:
            resume_status = "no_erasure_observed"
        return ValidationReport(
            passed=validation.passed,
            failure_codes=validation.failure_codes,
            failures=validation.failures,
            metrics={
                **validation.metrics,
                "a2_end_policy": "episode_censored_v1",
                "a2_resume_status": resume_status,
                "a2_right_censored": right_censored,
                "a2_resume_observed": resume_observed,
            },
        )
    last_erasure = manifest.start_index + int(manifest.parameters["erased_offsets"][-1])
    if last_erasure < len(clean_records) - 1:
        return validation
    code = "A2_RESUME_MISSING"
    if code in validation.failure_codes:
        return validation
    return ValidationReport(
        passed=False,
        failure_codes=validation.failure_codes + (code,),
        failures=validation.failures
        + ("A2 requires a later clean payload to resume after its last erasure",),
        metrics=validation.metrics,
    )

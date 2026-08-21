"""Strict serialization of the validator report and its trace cross-links."""

from __future__ import annotations

from typing import Any

from robotactile_benchmark.closed_loop.artifact_contracts import (
    BUNDLE_SEMANTIC_VERSION,
    ArtifactValidationError,
    require_sha256,
)
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.contracts import freeze_value, thaw_value
from robotactile_benchmark.validators import ValidationReport

_VALIDATION_FIELDS = frozenset(
    {
        "passed",
        "failure_codes",
        "failures",
        "metrics",
        "manifest_sha256",
        "clean_trace_sha256",
        "delivered_trace_sha256",
        "semantic_version",
    }
)


def validation_report_to_dict(
    report: ValidationReport,
    finalization: DeliveryFinalization,
) -> dict[str, object]:
    return {
        "passed": report.passed,
        "failure_codes": list(report.failure_codes),
        "failures": list(report.failures),
        "metrics": thaw_value(report.metrics),
        "manifest_sha256": finalization.manifest_sha256,
        "clean_trace_sha256": finalization.clean_trace_sha256,
        "delivered_trace_sha256": finalization.delivered_trace_sha256,
        "semantic_version": BUNDLE_SEMANTIC_VERSION,
    }


def validation_report_from_dict(
    value: object,
) -> tuple[ValidationReport, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != _VALIDATION_FIELDS:
        raise ArtifactValidationError("validation report fields mismatch")
    document: dict[str, Any] = value
    if document["semantic_version"] != BUNDLE_SEMANTIC_VERSION:
        raise ArtifactValidationError("unsupported validation semantic version")
    if type(document["passed"]) is not bool:
        raise ArtifactValidationError("validation passed must be boolean")
    failure_codes = document["failure_codes"]
    failures = document["failures"]
    metrics = document["metrics"]
    if (
        not isinstance(failure_codes, list)
        or not all(isinstance(item, str) and item for item in failure_codes)
        or len(set(failure_codes)) != len(failure_codes)
        or not isinstance(failures, list)
        or not all(isinstance(item, str) and item for item in failures)
        or not isinstance(metrics, dict)
    ):
        raise ArtifactValidationError("validation report values are malformed")
    report = ValidationReport(
        passed=document["passed"],
        failure_codes=tuple(failure_codes),
        failures=tuple(failures),
        metrics=freeze_value(metrics),
    )
    links = {
        "manifest_sha256": require_sha256(
            document["manifest_sha256"], "validation manifest sha256"
        ),
        "clean_trace_sha256": require_sha256(
            document["clean_trace_sha256"], "validation clean trace sha256"
        ),
        "delivered_trace_sha256": require_sha256(
            document["delivered_trace_sha256"], "validation delivered trace sha256"
        ),
    }
    return report, links

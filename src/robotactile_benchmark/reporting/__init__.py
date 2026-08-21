"""Public scientific reporting surface for RoboTactile matrices."""

from robotactile_benchmark.reporting.aggregation import aggregate_benchmark
from robotactile_benchmark.reporting.bundle import (
    LoadedReportBundle,
    ReportReceipt,
    load_report_bundle,
    write_report_bundle,
)
from robotactile_benchmark.reporting.contracts import OutcomeRecord, ReportingSpec
from robotactile_benchmark.reporting.matrix_adapter import (
    ARTIFACTS_DIRECTORY,
    ArtifactResolver,
    MatrixReportResult,
    content_addressed_artifact_resolver,
    load_matrix_manifest,
    load_matrix_outcomes,
    load_reporting_spec,
    write_matrix_report,
)
from robotactile_benchmark.reporting.recovery_evidence import (
    CLEAN_ENVELOPE_ALIGNMENT,
    POLICY_TASK_PROGRESS_SIGNAL_ID,
    RECOVERY_EVIDENCE_LEVEL,
    RECOVERY_EVIDENCE_SEMANTIC_VERSION,
    REGISTERED_RECOVERY_SIGNAL_IDS,
    RecoveryEvidence,
    RecoveryEvidenceBinding,
    RecoveryEvidenceValidationError,
    load_recovery_evidence,
    recovery_evidence_path,
    write_recovery_evidence,
)
from robotactile_benchmark.reporting.recovery_evidence_matrix import (
    RECOVERY_DIRECTORY,
)
from robotactile_benchmark.reporting.summary_contracts import BenchmarkSummary

__all__ = [
    "BenchmarkSummary",
    "ARTIFACTS_DIRECTORY",
    "CLEAN_ENVELOPE_ALIGNMENT",
    "ArtifactResolver",
    "LoadedReportBundle",
    "MatrixReportResult",
    "OutcomeRecord",
    "POLICY_TASK_PROGRESS_SIGNAL_ID",
    "RECOVERY_DIRECTORY",
    "RECOVERY_EVIDENCE_LEVEL",
    "RECOVERY_EVIDENCE_SEMANTIC_VERSION",
    "REGISTERED_RECOVERY_SIGNAL_IDS",
    "RecoveryEvidence",
    "RecoveryEvidenceBinding",
    "RecoveryEvidenceValidationError",
    "ReportReceipt",
    "ReportingSpec",
    "aggregate_benchmark",
    "content_addressed_artifact_resolver",
    "load_matrix_manifest",
    "load_matrix_outcomes",
    "load_recovery_evidence",
    "load_report_bundle",
    "load_reporting_spec",
    "recovery_evidence_path",
    "write_matrix_report",
    "write_report_bundle",
    "write_recovery_evidence",
]

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
from robotactile_benchmark.reporting.summary_contracts import BenchmarkSummary

__all__ = [
    "BenchmarkSummary",
    "ARTIFACTS_DIRECTORY",
    "ArtifactResolver",
    "LoadedReportBundle",
    "MatrixReportResult",
    "OutcomeRecord",
    "ReportReceipt",
    "ReportingSpec",
    "aggregate_benchmark",
    "content_addressed_artifact_resolver",
    "load_matrix_manifest",
    "load_matrix_outcomes",
    "load_report_bundle",
    "load_reporting_spec",
    "write_matrix_report",
    "write_report_bundle",
]

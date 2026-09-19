"""Deterministic JSON/CSV publication for one N0 robustness summary."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0FaultCampaignError,
    require_sha256,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    discard_staging,
    publish_staging,
    staging_directory,
    write_json,
)

from .reporting import N0FaultCampaignSummary

N0_FAULT_REPORT_EVIDENCE_LEVEL = "derived_from_strict_n0_live_artifacts_v1"
N0_FAULT_REPORT_SEMANTIC_VERSION = "1.0"
REPORT_RECEIPT_PATH = "report_receipt.json"
REPORT_MEMBERS = ("operator_cells.csv", "per_task.csv", "summary.json")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: object) -> str:
    return "" if value is None else str(value)


@dataclass(frozen=True)
class N0FaultReportMember:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.path not in REPORT_MEMBERS:
            raise N0FaultCampaignError("unknown N0 report member")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "member"))
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise N0FaultCampaignError("report member size must be positive")

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class N0FaultReportReceipt:
    summary_sha256: str
    reporting_spec_sha256: str
    source_root_sha256: Tuple[str, ...]
    members: Tuple[N0FaultReportMember, ...]
    evidence_level: str = N0_FAULT_REPORT_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = N0_FAULT_REPORT_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in ("summary_sha256", "reporting_spec_sha256"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        sources = tuple(self.source_root_sha256)
        if not sources or sources != tuple(sorted(set(sources))):
            raise N0FaultCampaignError("report source roots must be unique and sorted")
        for source in sources:
            require_sha256(source, "source root")
        members = tuple(self.members)
        if tuple(item.path for item in members) != REPORT_MEMBERS:
            raise N0FaultCampaignError("report member inventory mismatch")
        if (
            self.evidence_level != N0_FAULT_REPORT_EVIDENCE_LEVEL
            or self.simulator_qualification_claimed
            or self.semantic_version != N0_FAULT_REPORT_SEMANTIC_VERSION
        ):
            raise N0FaultCampaignError("report evidence contract mismatch")
        object.__setattr__(self, "source_root_sha256", sources)
        object.__setattr__(self, "members", members)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "summary_sha256": self.summary_sha256,
            "reporting_spec_sha256": self.reporting_spec_sha256,
            "source_root_sha256": list(self.source_root_sha256),
            "members": [item.to_dict() for item in self.members],
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "semantic_version": self.semantic_version,
        }


@dataclass(frozen=True)
class N0FaultReportWriteResult:
    publication_status: str
    receipt: N0FaultReportReceipt
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        if self.publication_status not in {"created", "already_present"}:
            raise N0FaultCampaignError("unknown report publication status")
        object.__setattr__(
            self,
            "receipt_file_sha256",
            require_sha256(self.receipt_file_sha256, "receipt file"),
        )


def _per_task_csv(summary: N0FaultCampaignSummary) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "task",
            "clean_outcomes",
            "clean_sr",
            "eligible_fault_pairs",
            "scored_cells",
            "fault_sr",
            "degradation_clean_minus_fault",
            "retention_fault_over_clean",
        )
    )
    for task in summary.tasks:
        writer.writerow(
            (
                task.task,
                task.clean_outcome_count,
                _metric(task.clean_success_rate),
                task.eligible_fault_pair_count,
                task.scored_cell_count,
                _metric(task.fault_success_rate),
                _metric(task.degradation),
                _metric(task.retention),
            )
        )
    return stream.getvalue().encode("utf-8")


def _operator_cells_csv(summary: N0FaultCampaignSummary) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "operator_id",
            "axis",
            "severity",
            "native_dose",
            "native_unit",
            "requested",
            "eligible_pairs",
            "clean_sr",
            "fault_sr",
            "degradation_clean_minus_fault",
            "retention_fault_over_clean",
            "ci_lower",
            "ci_upper",
            "mcnemar_p",
            "holm_adjusted_p",
            "statistics_status",
        )
    )
    for cell in summary.operator_cells:
        interval = cell.degradation_interval
        writer.writerow(
            (
                cell.operator_id,
                cell.axis,
                cell.severity_level,
                cell.native_dose,
                cell.native_unit,
                cell.requested_count,
                cell.eligible_pair_count,
                _metric(cell.clean_success_rate),
                _metric(cell.fault_success_rate),
                _metric(cell.degradation),
                _metric(cell.retention),
                _metric(None if interval is None else interval.lower),
                _metric(None if interval is None else interval.upper),
                _metric(cell.mcnemar_p_value),
                _metric(cell.holm_adjusted_p_value),
                cell.statistics_status,
            )
        )
    return stream.getvalue().encode("utf-8")


def _members(summary: N0FaultCampaignSummary) -> dict[str, bytes]:
    return {
        "operator_cells.csv": _operator_cells_csv(summary),
        "per_task.csv": _per_task_csv(summary),
        "summary.json": canonical_json_bytes(summary.to_dict()),
    }


def write_n0_fault_report_bundle(
    output: Path, summary: N0FaultCampaignSummary
) -> N0FaultReportWriteResult:
    """Atomically publish a source-bound report without replacing other bytes."""

    if type(summary) is not N0FaultCampaignSummary:
        raise TypeError("summary must be an exact N0FaultCampaignSummary")
    staging = staging_directory(Path(output))
    try:
        members = _members(summary)
        receipt = N0FaultReportReceipt(
            summary_sha256=summary.sha256,
            reporting_spec_sha256=summary.spec.sha256,
            source_root_sha256=summary.source_root_sha256,
            members=tuple(
                N0FaultReportMember(path, _sha256(payload), len(payload))
                for path, payload in sorted(members.items())
            ),
        )
        for path, payload in members.items():
            with (staging / path).open("xb") as stream:
                stream.write(payload)
        write_json(staging / REPORT_RECEIPT_PATH, receipt.to_dict())
        receipt_raw = (staging / REPORT_RECEIPT_PATH).read_bytes()
        status = publish_staging(staging, Path(output))
        return N0FaultReportWriteResult(status, receipt, _sha256(receipt_raw))
    finally:
        discard_staging(staging)

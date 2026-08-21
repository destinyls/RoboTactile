"""Content-addressed multi-format reporting bundle writer and loader."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from robotactile_benchmark.contracts import canonical_hash, canonical_json
from robotactile_benchmark.reporting.codec import summary_from_dict
from robotactile_benchmark.reporting.contracts import (
    REPORTING_SEMANTIC_VERSION,
    require_sha256,
)
from robotactile_benchmark.reporting.render import rendered_members
from robotactile_benchmark.reporting.summary_contracts import BenchmarkSummary

REPORT_EVIDENCE_LEVEL = "derived_from_verified_closed_loop_results"
REPORT_RECEIPT_NAME = "report_receipt.json"
MAX_REPORT_MEMBER_BYTES = 32 * 1024 * 1024
EXPECTED_MEMBERS = frozenset(
    {
        "native_dose_curves.csv",
        "native_dose_curves.svg",
        "operator_cells.csv",
        "per_task.csv",
        "summary.json",
        "summary_table.tex",
    }
)


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_canonical_json(payload: bytes, name: str) -> object:
    if len(payload) > MAX_REPORT_MEMBER_BYTES:
        raise ValueError(f"{name} exceeds the report member cap")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if payload != _canonical_json_bytes(value):
        raise ValueError(f"{name} is not in canonical JSON form")
    return value


@dataclass(frozen=True)
class ReportMember:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.path not in EXPECTED_MEMBERS:
            raise ValueError("unknown report member")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, "member"))
        if not 0 < self.size_bytes <= MAX_REPORT_MEMBER_BYTES:
            raise ValueError("report member size is outside bounds")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReportMember:
        if not isinstance(value, dict) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("report member fields mismatch")
        return cls(value["path"], value["sha256"], value["size_bytes"])


@dataclass(frozen=True)
class ReportReceipt:
    summary_sha256: str
    reporting_spec_sha256: str
    source_root_sha256: Tuple[str, ...]
    members: Tuple[ReportMember, ...]
    evidence_level: str = REPORT_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = REPORTING_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "summary_sha256", require_sha256(self.summary_sha256, "summary")
        )
        object.__setattr__(
            self,
            "reporting_spec_sha256",
            require_sha256(self.reporting_spec_sha256, "reporting spec"),
        )
        sources = tuple(self.source_root_sha256)
        if not sources or sources != tuple(sorted(set(sources))):
            raise ValueError("receipt source roots must be unique and sorted")
        for source in sources:
            require_sha256(source, "source root")
        members = tuple(self.members)
        paths = tuple(member.path for member in members)
        if paths != tuple(sorted(EXPECTED_MEMBERS)):
            raise ValueError("report member inventory mismatch")
        if self.evidence_level != REPORT_EVIDENCE_LEVEL:
            raise ValueError("report evidence level mismatch")
        if self.simulator_qualification_claimed is not False:
            raise ValueError("derived report cannot claim simulator qualification")
        if self.semantic_version != REPORTING_SEMANTIC_VERSION:
            raise ValueError("report semantic version mismatch")
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
            "members": [member.to_dict() for member in self.members],
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReportReceipt:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("report receipt fields mismatch")
        members = value["members"]
        sources = value["source_root_sha256"]
        if not isinstance(members, list) or not isinstance(sources, list):
            raise ValueError("report receipt inventories must be lists")
        return cls(
            summary_sha256=value["summary_sha256"],
            reporting_spec_sha256=value["reporting_spec_sha256"],
            source_root_sha256=tuple(sources),
            members=tuple(ReportMember.from_dict(item) for item in members),
            evidence_level=value["evidence_level"],
            simulator_qualification_claimed=value["simulator_qualification_claimed"],
            semantic_version=value["semantic_version"],
        )


@dataclass(frozen=True)
class LoadedReportBundle:
    summary: BenchmarkSummary
    receipt: ReportReceipt
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_file_sha256",
            require_sha256(self.receipt_file_sha256, "receipt file"),
        )


def _receipt_for(summary: BenchmarkSummary, members: dict[str, bytes]) -> ReportReceipt:
    return ReportReceipt(
        summary_sha256=summary.sha256,
        reporting_spec_sha256=summary.spec.sha256,
        source_root_sha256=summary.source_root_sha256,
        members=tuple(
            ReportMember(path, _file_sha256(payload), len(payload))
            for path, payload in sorted(members.items())
        ),
    )


def load_report_bundle(output_dir: Path) -> LoadedReportBundle:
    """Strictly load and re-render one report bundle."""

    output_dir = Path(output_dir)
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("report bundle must be a regular directory")
    entries = tuple(sorted(output_dir.iterdir(), key=lambda path: path.name))
    if {path.name for path in entries} != EXPECTED_MEMBERS | {REPORT_RECEIPT_NAME}:
        raise ValueError("report directory inventory mismatch")
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise ValueError("report members must be regular files")
        if path.stat().st_size > MAX_REPORT_MEMBER_BYTES:
            raise ValueError("report member exceeds its byte cap")
    receipt_payload = (output_dir / REPORT_RECEIPT_NAME).read_bytes()
    receipt = ReportReceipt.from_dict(
        _load_canonical_json(receipt_payload, REPORT_RECEIPT_NAME)
    )
    member_payloads = {
        member.path: (output_dir / member.path).read_bytes()
        for member in receipt.members
    }
    for member in receipt.members:
        payload = member_payloads[member.path]
        if len(payload) != member.size_bytes or _file_sha256(payload) != member.sha256:
            raise ValueError("report member content disagrees with receipt")
    summary_payload = member_payloads["summary.json"]
    summary = summary_from_dict(_load_canonical_json(summary_payload, "summary.json"))
    if summary.sha256 != receipt.summary_sha256:
        raise ValueError("summary content hash disagrees with receipt")
    if summary.spec.sha256 != receipt.reporting_spec_sha256:
        raise ValueError("reporting spec hash disagrees with receipt")
    if summary.source_root_sha256 != receipt.source_root_sha256:
        raise ValueError("source roots disagree with receipt")
    if rendered_members(summary) != member_payloads:
        raise ValueError("derived report members disagree with summary")
    return LoadedReportBundle(
        summary=summary,
        receipt=receipt,
        receipt_file_sha256=_file_sha256(receipt_payload),
    )


def write_report_bundle(output_dir: Path, summary: BenchmarkSummary) -> ReportReceipt:
    """Atomically write a deterministic report without overwriting other data."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        try:
            loaded = load_report_bundle(output_dir)
        except ValueError as exc:
            raise ValueError(
                "report output directory is non-empty with unrelated content"
            ) from exc
        if loaded.summary.sha256 != summary.sha256:
            raise ValueError("report output directory is non-empty with other content")
        return loaded.receipt
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    try:
        members = rendered_members(summary)
        receipt = _receipt_for(summary, members)
        for name, payload in members.items():
            (staging / name).write_bytes(payload)
        (staging / REPORT_RECEIPT_NAME).write_bytes(
            _canonical_json_bytes(receipt.to_dict())
        )
        loaded = load_report_bundle(staging)
        if loaded.receipt != receipt:
            raise ValueError("staged report failed receipt verification")
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)

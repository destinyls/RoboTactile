"""Fail-closed bridge from verified matrix artifacts to paper reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, cast

from robotactile_benchmark.closed_loop.artifact_contracts import (
    EVIDENCE_LEVEL as CPU_CLOSED_LOOP_EVIDENCE_LEVEL,
)
from robotactile_benchmark.closed_loop.artifacts import load_closed_loop_bundle
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LIVE_ARTIFACT_EVIDENCE_LEVEL,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.matrix.contracts import MatrixCellSpec, MatrixGridKind
from robotactile_benchmark.matrix.io import (
    MatrixResumeError,
    canonical_matrix_json_bytes,
    load_matrix_summary,
)
from robotactile_benchmark.matrix.manifest import MatrixManifest
from robotactile_benchmark.matrix.results import (
    CellArtifactReference,
    MatrixCellStatus,
)
from robotactile_benchmark.matrix.states import MatrixCellState
from robotactile_benchmark.reporting.aggregation import aggregate_benchmark
from robotactile_benchmark.reporting.bundle import ReportReceipt, write_report_bundle
from robotactile_benchmark.reporting.contracts import OutcomeRecord, ReportingSpec
from robotactile_benchmark.reporting.summary_contracts import BenchmarkSummary
from robotactile_benchmark.trials import TerminalStatus, TrialManifest

ARTIFACTS_DIRECTORY = "artifacts"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_SPEC_BYTES = 1024 * 1024
_NORMAL_STATUSES = frozenset(
    {
        TerminalStatus.SUCCESS,
        TerminalStatus.TASK_FAILURE,
        TerminalStatus.EARLY_STOP,
        TerminalStatus.TIMEOUT,
    }
)

ArtifactResolver = Callable[[CellArtifactReference], Path]


@dataclass(frozen=True)
class _VerifiedArtifact:
    """Common typed view returned only after an artifact-specific strict load."""

    trial: TrialManifest
    fault_manifest: Optional[FaultManifest]
    result: ClosedLoopTrialResult
    root_receipt_sha256: str
    evidence_level: str


@dataclass(frozen=True)
class MatrixReportResult:
    """Source-bound result of one matrix-to-report export."""

    manifest_sha256: str
    outcome_count: int
    summary: BenchmarkSummary
    receipt: ReportReceipt

    def to_cli_dict(self) -> dict[str, object]:
        return {
            "evidence_level": self.receipt.evidence_level,
            "manifest_sha256": self.manifest_sha256,
            "outcome_count": self.outcome_count,
            "report_receipt_sha256": self.receipt.sha256,
            "reporting_spec_sha256": self.summary.spec.sha256,
            "simulator_qualification_claimed": (
                self.receipt.simulator_qualification_claimed
            ),
            "summary_sha256": self.summary.sha256,
        }


def load_matrix_manifest(path: Path) -> MatrixManifest:
    """Strictly reconstruct a canonical matrix manifest from a regular file."""

    try:
        document = _load_canonical_document(
            path, _MAX_MANIFEST_BYTES, "matrix manifest"
        )
        return MatrixManifest.from_dict(document)
    except (TypeError, ValueError) as error:
        raise MatrixResumeError("matrix manifest failed typed validation") from error


def load_reporting_spec(path: Path) -> ReportingSpec:
    """Strictly reconstruct the frozen reporting choices used by the CLI."""

    document = _load_canonical_document(path, _MAX_SPEC_BYTES, "reporting spec")
    try:
        return ReportingSpec.from_dict(document)
    except (TypeError, ValueError) as error:
        raise ValueError("reporting spec failed typed validation") from error


def content_addressed_artifact_resolver(matrix_output: Path) -> ArtifactResolver:
    """Resolve shared artifacts by their externally pinned root receipt hash."""

    artifact_store = Path(matrix_output) / ARTIFACTS_DIRECTORY

    def resolve(reference: CellArtifactReference) -> Path:
        return artifact_store / reference.root_receipt_sha256

    return resolve


def load_matrix_outcomes(
    matrix_manifest: Path,
    matrix_output: Path,
    spec: ReportingSpec,
    *,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> Tuple[OutcomeRecord, ...]:
    """Load every matrix receipt and derive outcomes from verified result bytes."""

    if not isinstance(spec, ReportingSpec):
        raise TypeError("spec must be a ReportingSpec")
    manifest = load_matrix_manifest(matrix_manifest)
    return _load_outcomes_for_manifest(
        manifest,
        matrix_output,
        spec,
        artifact_resolver=artifact_resolver,
    )


def _load_outcomes_for_manifest(
    manifest: MatrixManifest,
    matrix_output: Path,
    spec: ReportingSpec,
    *,
    artifact_resolver: Optional[ArtifactResolver],
) -> Tuple[OutcomeRecord, ...]:
    summary = load_matrix_summary(Path(matrix_output), manifest)
    if manifest.kind is not MatrixGridKind.PRIMARY:
        raise ValueError("focused matrix reporting is not registered")
    if any(cell.trial.base_system_id != spec.system_id for cell in manifest.cells):
        raise ValueError("matrix base system does not match reporting spec")
    resolver = artifact_resolver or content_addressed_artifact_resolver(matrix_output)
    outcomes = tuple(
        _outcome_from_cell(
            cell,
            state,
            spec,
            resolver,
        )
        for cell, state in zip(manifest.cells, summary.cells)
    )
    if len(outcomes) != len(manifest.cells):
        raise MatrixResumeError("matrix outcome inventory is incomplete")
    return outcomes


def write_matrix_report(
    matrix_manifest: Path,
    matrix_output: Path,
    reporting_spec: Path,
    report_output: Path,
    *,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> MatrixReportResult:
    """Regenerate a deterministic report from a fully materialized primary matrix."""

    manifest = load_matrix_manifest(matrix_manifest)
    spec = load_reporting_spec(reporting_spec)
    outcomes = _load_outcomes_for_manifest(
        manifest,
        matrix_output,
        spec,
        artifact_resolver=artifact_resolver,
    )
    summary = aggregate_benchmark(outcomes, spec)
    receipt = write_report_bundle(Path(report_output), summary)
    return MatrixReportResult(
        manifest_sha256=manifest.sha256,
        outcome_count=len(outcomes),
        summary=summary,
        receipt=receipt,
    )


def _outcome_from_cell(
    cell: MatrixCellSpec,
    state: MatrixCellState,
    spec: ReportingSpec,
    resolver: ArtifactResolver,
) -> OutcomeRecord:
    if (
        state.cell_sha256 != cell.sha256
        or state.trial_manifest_sha256 != cell.trial.sha256
    ):
        raise MatrixResumeError("matrix state does not match its materialized cell")
    if state.artifact is None:
        return _receipt_only_outcome(cell, state, spec)
    artifact_path = Path(resolver(state.artifact))
    verified = _load_verified_artifact(artifact_path, state.artifact)
    _validate_artifact_crosslinks(cell, state, verified)
    result = verified.result
    return OutcomeRecord(
        system_id=spec.system_id,
        task=cell.task,
        pair_key=cell.pair_key,
        condition=cell.trial.condition,
        terminal_status=result.terminal_status,
        score_eligible=result.score_eligible,
        score_success=result.score_success,
        source_root_sha256=verified.root_receipt_sha256,
        operator_id=cell.operator_id,
        severity_level=cell.severity_level,
    )


def _receipt_only_outcome(
    cell: MatrixCellSpec, state: MatrixCellState, spec: ReportingSpec
) -> OutcomeRecord:
    if state.receipt_sha256 is None:
        raise MatrixResumeError("terminal matrix state lost its receipt")
    if state.status is MatrixCellStatus.UNSUPPORTED:
        terminal = TerminalStatus.UNSUPPORTED_CONTRACT
        eligible = False
        success = None
    elif state.status is MatrixCellStatus.CRASH:
        terminal = TerminalStatus.CRASH
        eligible = True
        success = False
    else:
        raise ValueError("completed/validator matrix cells require a strict artifact")
    return OutcomeRecord(
        system_id=spec.system_id,
        task=cell.task,
        pair_key=cell.pair_key,
        condition=cell.trial.condition,
        terminal_status=terminal,
        score_eligible=eligible,
        score_success=success,
        source_root_sha256=state.receipt_sha256,
        operator_id=cell.operator_id,
        severity_level=cell.severity_level,
    )


def _load_verified_artifact(
    path: Path, reference: CellArtifactReference
) -> _VerifiedArtifact:
    if reference.evidence_level == LIVE_ARTIFACT_EVIDENCE_LEVEL:
        live_loaded = load_live_univtac_artifact(path)
        return _VerifiedArtifact(
            trial=live_loaded.trial,
            fault_manifest=live_loaded.fault_manifest,
            result=live_loaded.evidence.result,
            root_receipt_sha256=live_loaded.root_receipt_sha256,
            evidence_level=live_loaded.root_receipt.evidence_level,
        )
    if reference.evidence_level == CPU_CLOSED_LOOP_EVIDENCE_LEVEL:
        cpu_loaded = load_closed_loop_bundle(path)
        return _VerifiedArtifact(
            trial=cpu_loaded.trial,
            fault_manifest=cpu_loaded.fault_manifest,
            result=cpu_loaded.result,
            root_receipt_sha256=cpu_loaded.root_receipt_sha256,
            evidence_level=cpu_loaded.root_receipt.evidence_level,
        )
    raise ValueError(
        "matrix artifact evidence level is not a supported loadable bundle type"
    )


def _validate_artifact_crosslinks(
    cell: MatrixCellSpec, state: MatrixCellState, artifact: _VerifiedArtifact
) -> None:
    reference = state.artifact
    if reference is None:
        raise ValueError("matrix artifact reference disappeared")
    if artifact.root_receipt_sha256 != reference.root_receipt_sha256:
        raise ValueError("loaded artifact root disagrees with matrix receipt")
    if artifact.evidence_level != reference.evidence_level:
        raise ValueError("loaded artifact evidence level disagrees with matrix receipt")
    if artifact.result.sha256 != reference.result_sha256:
        raise ValueError("loaded result hash disagrees with matrix receipt")
    if (
        artifact.trial != cell.trial
        or artifact.result.trial_manifest_sha256 != cell.trial.sha256
    ):
        raise ValueError("loaded artifact trial disagrees with matrix cell")
    if artifact.result.pair_key != cell.pair_key:
        raise ValueError("loaded artifact pair key disagrees with matrix cell")
    if artifact.fault_manifest != cell.fault_manifest:
        raise ValueError("loaded fault manifest disagrees with matrix cell")
    expected = {
        MatrixCellStatus.COMPLETED: _NORMAL_STATUSES,
        MatrixCellStatus.CRASH: frozenset({TerminalStatus.CRASH}),
        MatrixCellStatus.UNSUPPORTED: frozenset({TerminalStatus.UNSUPPORTED_CONTRACT}),
        MatrixCellStatus.VALIDATOR_REJECTED: frozenset(
            {TerminalStatus.VALIDATOR_REJECTED}
        ),
    }.get(state.status)
    if expected is None or artifact.result.terminal_status not in expected:
        raise ValueError("loaded terminal status disagrees with matrix receipt")
    if state.failure_code != artifact.result.failure_code:
        raise ValueError("loaded failure code disagrees with matrix receipt")


def _load_canonical_document(path: Path, limit: int, label: str) -> object:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    before = path.stat()
    if not 0 < before.st_size <= limit:
        raise ValueError(f"{label} size is outside bounds")
    raw = path.read_bytes()
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != before.st_size:
        raise ValueError(f"{label} changed during read")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"{label} has duplicate JSON keys")
            document[key] = value
        return document

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} has non-finite constant {token}")

    try:
        document = cast(
            object,
            json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=pairs_hook,
                parse_constant=reject_constant,
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if canonical_matrix_json_bytes(document) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return document

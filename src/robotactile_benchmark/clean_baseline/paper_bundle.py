"""Bind a complete Clean summary to simulator qualification and paper exports."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Tuple, cast

from robotactile_benchmark.clean_baseline.aggregation import (
    VerifiedCleanArtifact,
    load_clean_artifact_inventory,
)
from robotactile_benchmark.clean_baseline.attempts import CleanAttemptDisposition
from robotactile_benchmark.clean_baseline.contracts import (
    CleanCampaignError,
    CleanCampaignManifest,
)
from robotactile_benchmark.clean_baseline.io import (
    load_clean_baseline_summary,
    load_clean_campaign_manifest,
    read_canonical_json_file,
    write_canonical_no_clobber,
)
from robotactile_benchmark.clean_baseline.qualification import (
    QUALIFICATION_V3_SEMANTIC_VERSION,
    VerifiedAllTaskQualification,
    verify_all_task_qualification,
)
from robotactile_benchmark.clean_baseline.summary import CleanBaselineSummary
from robotactile_benchmark.clean_baseline.summary_values import (
    CleanTaskSummary,
    bootstrap_to_dict,
    wilson_to_dict,
)
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.runtime_source import RuntimeSourceBinding

PAPER_RESULT_EVIDENCE_LEVEL = "qualified_clean_closed_loop_result_v2"
PAPER_RESULT_V3_EVIDENCE_LEVEL = "dual_attested_clean_closed_loop_result_v3"


@dataclass(frozen=True)
class PaperResultBundle:
    receipt_path: Path
    receipt_sha256: str
    csv_path: Path
    latex_path: Path
    paper_claim_eligible: bool
    blockers: Tuple[str, ...]


@dataclass(frozen=True)
class _RuntimeEvidence:
    blocker: str | None
    task_source_bindings: Tuple[tuple[str, RuntimeSourceBinding], ...] = ()
    valid_attempt_attestations: Tuple[Mapping[str, object], ...] = ()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_paper_result_bundle(
    *,
    deployment_root: Path,
    summary_path: Path,
    qualification_path: Path,
    output_directory: Path,
) -> PaperResultBundle:
    """Create deterministic JSON, CSV, and LaTeX evidence from verified inputs."""

    root = Path(deployment_root).resolve(strict=True)
    summary_absolute = Path(summary_path).absolute()
    try:
        summary_relative = summary_absolute.relative_to(root)
    except ValueError as error:
        raise CleanCampaignError(
            "clean summary must remain below deployment root"
        ) from error
    if summary_relative.parts[:1] != ("outputs",) or summary_absolute.is_symlink():
        raise CleanCampaignError("clean summary must be a regular outputs file")
    summary = load_clean_baseline_summary(summary_path)
    qualification = verify_all_task_qualification(root, qualification_path)
    expected_action = (
        "ee8_absolute"
        if summary.policy_kind is LivePolicyKind.N0
        else "qpos8_next_step"
    )
    blockers = list(summary.paper_claim_blockers)
    runtime_evidence = _verify_runtime_evidence(root, summary, qualification)
    if not qualification.source_bound:
        blockers.append("qualification_not_source_bound")
    else:
        if qualification.campaign_manifest_sha256 != summary.campaign_manifest_sha256:
            blockers.append("qualification_campaign_manifest_mismatch")
        if runtime_evidence.blocker is not None:
            blockers.append(runtime_evidence.blocker)
        else:
            blockers = [item for item in blockers if item != "simulator_not_qualified"]
    if qualification.action_spec != expected_action:
        blockers.append("qualification_action_spec_mismatch")
    summary_tasks = tuple(item.task for item in summary.task_summaries)
    if set(summary_tasks) != set(qualification.tasks):
        blockers.append("qualification_task_scope_mismatch")
    blockers = list(dict.fromkeys(blockers))
    csv_payload = _render_csv(summary)
    latex_payload = _render_latex(summary)
    output = Path(output_directory).absolute()
    try:
        relative_output = output.relative_to(root / "outputs")
    except ValueError as error:
        raise CleanCampaignError("paper bundle must remain below outputs/") from error
    if not relative_output.parts:
        raise CleanCampaignError("paper bundle output must be a child directory")
    csv_path = output / "clean_baseline_table.csv"
    latex_path = output / "clean_baseline_table.tex"
    _write_bytes_no_clobber(csv_path, csv_payload)
    _write_bytes_no_clobber(latex_path, latex_payload)
    qualification_absolute = qualification.path
    receipt_path = output / "paper_result.json"
    paper_v3 = qualification.semantic_version == QUALIFICATION_V3_SEMANTIC_VERSION
    receipt: dict[str, object] = {
        "bootstrap_interval": bootstrap_to_dict(summary.task_stratified_bootstrap),
        "campaign_id": summary.campaign_id,
        "campaign_manifest_sha256": summary.campaign_manifest_sha256,
        "csv_relpath": csv_path.relative_to(root).as_posix(),
        "csv_sha256": _sha256_bytes(csv_payload),
        "eligible_trial_count": summary.eligible_trial_count,
        "evidence_level": (
            PAPER_RESULT_V3_EVIDENCE_LEVEL if paper_v3 else PAPER_RESULT_EVIDENCE_LEVEL
        ),
        "failure_count": summary.failure_count,
        "latex_relpath": latex_path.relative_to(root).as_posix(),
        "latex_sha256": _sha256_bytes(latex_payload),
        "macro_task_success_rate": summary.macro_task_success_rate,
        "paper_claim_blockers": blockers,
        "paper_claim_eligible": not blockers,
        "planned_trial_count": summary.planned_trial_count,
        "policy_kind": summary.policy_kind.value,
        "pooled_success_rate": summary.eligible_success_rate,
        "pooled_wilson": wilson_to_dict(summary.pooled_wilson),
        "protocol_id": summary.protocol_id.value,
        "qualification_action_spec": qualification.action_spec,
        "qualification_campaign_id": qualification.campaign_id,
        "qualification_campaign_manifest_sha256": (
            qualification.campaign_manifest_sha256
        ),
        "qualification_observation_parity_sha256s": list(
            qualification.observation_parity_sha256s
        ),
        "qualification_relpath": qualification_absolute.relative_to(root).as_posix(),
        "qualification_repetitions": qualification.repetitions,
        "qualification_semantic_version": qualification.semantic_version,
        "qualification_sha256": qualification.sha256,
        "qualification_source_binding": (
            None
            if qualification.source_binding is None
            else qualification.source_binding.to_dict()
        ),
        "qualification_source_bound": qualification.source_bound,
        "semantic_version": "3.0" if paper_v3 else "2.0",
        "success_count": summary.success_count,
        "summary_relpath": summary_absolute.relative_to(root).as_posix(),
        "summary_sha256": summary.sha256,
        "task_registry_sha256": summary.task_registry_sha256,
        "task_rows": [_task_row(item) for item in summary.task_summaries],
    }
    if paper_v3:
        receipt.update(
            {
                "qualification_task_source_bindings": {
                    task: source.to_dict()
                    for task, source in runtime_evidence.task_source_bindings
                },
                "valid_attempt_attestations": list(
                    runtime_evidence.valid_attempt_attestations
                ),
            }
        )
    write_canonical_no_clobber(receipt_path, receipt)
    return PaperResultBundle(
        receipt_path=receipt_path,
        receipt_sha256=_sha256_file(receipt_path),
        csv_path=csv_path,
        latex_path=latex_path,
        paper_claim_eligible=not blockers,
        blockers=tuple(blockers),
    )


def _verify_runtime_evidence(
    root: Path,
    summary: CleanBaselineSummary,
    qualification: VerifiedAllTaskQualification,
) -> _RuntimeEvidence:
    if qualification.semantic_version != QUALIFICATION_V3_SEMANTIC_VERSION:
        return _RuntimeEvidence("qualification_runtime_binding_unbound")
    if len(qualification.tasks) != len(qualification.task_source_bindings):
        return _RuntimeEvidence("qualification_runtime_binding_unbound")
    task_sources = tuple(zip(qualification.tasks, qualification.task_source_bindings))
    try:
        manifest = _qualification_campaign_manifest(root, qualification)
        inventory = load_clean_artifact_inventory(
            root,
            manifest,
            allow_compact=False,
        )
    except (OSError, TypeError, ValueError):
        return _RuntimeEvidence("qualification_runtime_binding_mismatch", task_sources)
    if any(not item.artifact.has_full_trace for item in inventory.artifacts):
        return _RuntimeEvidence("qualification_runtime_binding_mismatch", task_sources)
    if not _summary_inventory_matches(summary, manifest.sha256, inventory):
        return _RuntimeEvidence("qualification_runtime_binding_mismatch", task_sources)
    valid = tuple(
        sorted(
            (
                item
                for item in inventory.artifacts
                if item.disposition is CleanAttemptDisposition.VALID_OUTCOME
            ),
            key=lambda item: item.trial_spec.ordinal,
        )
    )
    if len(valid) != summary.eligible_trial_count:
        return _RuntimeEvidence("qualification_runtime_binding_mismatch", task_sources)
    source_by_task = dict(task_sources)
    if any(not _attempt_is_source_bound(item) for item in valid):
        return _RuntimeEvidence("qualification_runtime_binding_unbound", task_sources)
    if any(
        item.qualification_sha256 != qualification.sha256
        or item.runtime_source_binding != source_by_task.get(item.trial_spec.task)
        for item in valid
    ):
        return _RuntimeEvidence("qualification_runtime_binding_mismatch", task_sources)
    return _RuntimeEvidence(
        None,
        task_sources,
        tuple(_attempt_attestation_row(item) for item in valid),
    )


def _qualification_campaign_manifest(
    root: Path,
    qualification: VerifiedAllTaskQualification,
) -> CleanCampaignManifest:
    document, _ = read_canonical_json_file(qualification.path, "all-task qualification")
    relative = document.get("campaign_manifest_relpath")
    if not isinstance(relative, str) or "\\" in relative:
        raise CleanCampaignError("qualification campaign manifest path is invalid")
    member = PurePosixPath(relative)
    if (
        member.is_absolute()
        or not member.parts
        or member.parts[0] != "requests"
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise CleanCampaignError("qualification campaign manifest path is unsafe")
    path = root / Path(*member.parts)
    manifest = load_clean_campaign_manifest(path)
    if (
        qualification.campaign_manifest_sha256 is None
        or manifest.sha256 != qualification.campaign_manifest_sha256
    ):
        raise CleanCampaignError("qualification campaign manifest mismatch")
    return manifest


def _summary_inventory_matches(
    summary: CleanBaselineSummary,
    manifest_sha256: str,
    inventory: object,
) -> bool:
    from robotactile_benchmark.clean_baseline.aggregation import CleanArtifactInventory
    from robotactile_benchmark.clean_baseline.summary import (
        CleanCandidateClassification,
    )

    if not isinstance(inventory, CleanArtifactInventory):
        return False
    provenance = inventory.candidate_provenance
    artifact_hashes = tuple(
        cast(str, item.artifact_root_sha256)
        for item in provenance
        if item.artifact_root_sha256 is not None
    )
    attempt_hashes = tuple(
        cast(str, item.attempt_receipt_sha256)
        for item in provenance
        if item.artifact_root_sha256 is not None
    )
    exception_hashes = tuple(
        cast(str, item.attempt_receipt_sha256)
        for item in provenance
        if item.classification is CleanCandidateClassification.EXCEPTION_REPLACED
        and item.attempt_receipt_sha256 is not None
    )
    return (
        summary.campaign_manifest_sha256 == manifest_sha256
        and inventory.campaign_manifest_sha256 == manifest_sha256
        and summary.artifact_root_sha256s == artifact_hashes
        and summary.attempt_receipt_sha256s == attempt_hashes
        and summary.exception_attempt_receipt_sha256s == exception_hashes
        and not inventory.missing_trials
        and not inventory.protocol_invalid_trials
    )


def _attempt_is_source_bound(item: VerifiedCleanArtifact) -> bool:
    return (
        item.attempt_semantic_version == "3.0"
        and item.qualification_relpath is not None
        and item.qualification_sha256 is not None
        and item.runtime_source_binding is not None
        and item.isaac_attestation_relpath is not None
        and item.isaac_attestation_sha256 is not None
        and item.n0_server_attestation_relpath is not None
        and item.n0_server_attestation_sha256 is not None
    )


def _attempt_attestation_row(item: VerifiedCleanArtifact) -> Mapping[str, object]:
    assert item.qualification_sha256 is not None
    assert item.isaac_attestation_sha256 is not None
    assert item.n0_server_attestation_sha256 is not None
    return {
        "artifact_root_sha256": item.artifact.external_root_sha256,
        "attempt_receipt_sha256": item.attempt_receipt_sha256,
        "isaac_attestation_sha256": item.isaac_attestation_sha256,
        "n0_server_attestation_sha256": item.n0_server_attestation_sha256,
        "ordinal": item.trial_spec.ordinal,
        "qualification_sha256": item.qualification_sha256,
        "task": item.trial_spec.task,
    }


def _task_row(item: CleanTaskSummary) -> dict[str, object]:
    return {
        "eligible": item.eligible,
        "failures": item.failures,
        "success_rate": item.eligible_success_rate,
        "successes": item.successes,
        "task": item.task,
        "wilson": wilson_to_dict(item.wilson),
    }


def _render_csv(summary: CleanBaselineSummary) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ("task", "successes", "trials", "success_rate", "ci_low", "ci_high")
    )
    for item in summary.task_summaries:
        interval = item.wilson
        writer.writerow(
            (
                item.task,
                item.successes,
                item.eligible,
                _number(item.eligible_success_rate),
                _number(None if interval is None else interval.lower),
                _number(None if interval is None else interval.upper),
            )
        )
    pooled = summary.pooled_wilson
    writer.writerow(
        (
            "pooled",
            summary.success_count,
            summary.eligible_trial_count,
            _number(summary.eligible_success_rate),
            _number(None if pooled is None else pooled.lower),
            _number(None if pooled is None else pooled.upper),
        )
    )
    return stream.getvalue().encode("utf-8")


def _render_latex(summary: CleanBaselineSummary) -> bytes:
    rows = ["Task & Success & Trials & SR (95\\% CI) \\\\", "\\midrule"]
    for item in summary.task_summaries:
        interval = item.wilson
        estimate = _percent(item.eligible_success_rate)
        bounds = (
            "--"
            if interval is None
            else f"[{_percent(interval.lower)}, {_percent(interval.upper)}]"
        )
        task = item.task.replace("_", "\\_")
        rows.append(
            f"{task} & {item.successes} & {item.eligible} & {estimate} {bounds} \\\\"
        )
    rows.extend(
        (
            "\\midrule",
            "Macro & -- & "
            f"{summary.eligible_trial_count} & "
            f"{_percent(summary.macro_task_success_rate)} \\\\",
        )
    )
    payload = "\n".join(
        (
            "\\begin{tabular}{lrrl}",
            "\\toprule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        )
    )
    return payload.encode("utf-8")


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}\\%"


def _write_bytes_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CleanCampaignError("paper export cannot be a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different paper export")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PAPER_RESULT_EVIDENCE_LEVEL",
    "PAPER_RESULT_V3_EVIDENCE_LEVEL",
    "PaperResultBundle",
    "VerifiedAllTaskQualification",
    "build_paper_result_bundle",
    "verify_all_task_qualification",
]

"""Execute one hash-bound N0 fault pair from a generated campaign bundle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from robotactile_benchmark.backends.univtac_contracts import (
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.official_n0 import (
    OfficialN0PairedLiveResult,
    execute_official_n0_paired_live_runs,
)
from robotactile_benchmark.execution.paired_receipt_io import (
    write_paired_execution_receipt,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCellDisposition,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    load_n0_fault_campaign_bundle,
    write_json,
)
from robotactile_benchmark.trials import Condition

N0_FAULT_PAIR_RUN_SEMANTIC_VERSION = "1.0"
N0_FAULT_PAIR_RUN_EVIDENCE_LEVEL = "unqualified_paired_n0_fault_execution_v1"


@dataclass(frozen=True)
class N0FaultPairRunReceipt:
    """Exact campaign, request, reset, capture, and artifact links for one pair."""

    campaign_id: str
    campaign_manifest_sha256: str
    generation_receipt_file_sha256: str
    task: str
    pair_key: str
    request_file_sha256s: Tuple[str, ...]
    group_content_sha256: str
    paired_receipt_file_sha256: str
    artifact_root_sha256s: Tuple[str, ...]
    capture_profile: str
    action_execution_contract: str
    evidence_level: str = N0_FAULT_PAIR_RUN_EVIDENCE_LEVEL
    simulator_qualification_claimed: bool = False
    semantic_version: str = N0_FAULT_PAIR_RUN_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        require_nonempty(self.campaign_id, "campaign_id")
        require_nonempty(self.task, "task")
        for name in (
            "campaign_manifest_sha256",
            "generation_receipt_file_sha256",
            "pair_key",
            "group_content_sha256",
            "paired_receipt_file_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if len(self.request_file_sha256s) < 2:
            raise N0FaultCampaignError("pair run requires Clean plus Faulted requests")
        for value in self.request_file_sha256s:
            require_sha256(value, "request_file_sha256")
        if len(self.artifact_root_sha256s) != len(self.request_file_sha256s):
            raise N0FaultCampaignError("artifact roots must cover every live request")
        for value in self.artifact_root_sha256s:
            require_sha256(value, "artifact_root_sha256")
        LiveCaptureProfile(self.capture_profile)
        require_nonempty(self.action_execution_contract, "action_execution_contract")
        if (
            self.evidence_level != N0_FAULT_PAIR_RUN_EVIDENCE_LEVEL
            or self.simulator_qualification_claimed
            or self.semantic_version != N0_FAULT_PAIR_RUN_SEMANTIC_VERSION
        ):
            raise N0FaultCampaignError("pair run evidence contract mismatch")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "generation_receipt_file_sha256": self.generation_receipt_file_sha256,
            "task": self.task,
            "pair_key": self.pair_key,
            "request_file_sha256s": list(self.request_file_sha256s),
            "group_content_sha256": self.group_content_sha256,
            "paired_receipt_file_sha256": self.paired_receipt_file_sha256,
            "artifact_root_sha256s": list(self.artifact_root_sha256s),
            "capture_profile": self.capture_profile,
            "action_execution_contract": self.action_execution_contract,
            "evidence_level": self.evidence_level,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "semantic_version": self.semantic_version,
        }


def select_live_pair_cells(
    cells: Tuple[N0FaultCampaignCellSpec, ...], pair_key: Optional[str]
) -> Tuple[N0FaultCampaignCellSpec, ...]:
    """Select one complete executable pair in deterministic request order."""

    pair_keys = sorted({cell.pair_key for cell in cells})
    selected_key = pair_key
    if selected_key is None:
        if len(pair_keys) != 1:
            raise N0FaultCampaignError(
                "multi-pair campaign execution requires an explicit pair_key"
            )
        selected_key = pair_keys[0]
    if selected_key not in pair_keys:
        raise N0FaultCampaignError("requested pair_key is absent from campaign")
    selected = [
        cell
        for cell in cells
        if cell.pair_key == selected_key
        and cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
    ]
    clean = [cell for cell in selected if cell.condition is Condition.CLEAN]
    faults = [cell for cell in selected if cell.condition is Condition.FAULTED]
    if len(clean) != 1 or not faults:
        raise N0FaultCampaignError("selected pair lacks Clean or executable faults")
    return (clean[0], *sorted(faults, key=_fault_order))


def _fault_order(cell: N0FaultCampaignCellSpec) -> tuple[str, int]:
    assert cell.operator_id is not None
    assert cell.severity_level is not None
    return cell.operator_id, cell.severity_level


def run_n0_fault_pair(
    campaign_root: Path,
    *,
    integration_config: Path,
    n0_source_root: Path,
    host: str,
    port: int,
    pair_key: Optional[str] = None,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
    action_execution_contract: str = N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    receipt_dir: Optional[Path] = None,
) -> N0FaultPairRunReceipt:
    """Run one Clean/Faulted pair once; existing evidence is never overwritten."""

    loaded = load_n0_fault_campaign_bundle(campaign_root)
    cells = select_live_pair_cells(loaded.manifest.cells, pair_key)
    task = cells[0].task
    if any(cell.task != task for cell in cells):
        raise N0FaultCampaignError("one pair cannot span multiple tasks")
    requests = tuple(
        load_live_univtac_request(loaded.root / str(cell.request_relpath))
        for cell in cells
    )
    for request in requests:
        output = request.output_dir
        if output is None:
            raise N0FaultCampaignError("fault campaign request has no output_dir")
        if output.is_symlink() or (
            output.exists() and (not output.is_dir() or any(output.iterdir()))
        ):
            raise FileExistsError("refusing to rerun a pair with existing artifacts")
    output_root = (
        loaded.root / "executions" / task / cells[0].pair_key
        if receipt_dir is None
        else Path(receipt_dir).expanduser().absolute()
    )
    pair_receipt_path = output_root / "paired_execution_receipt.json"
    run_receipt_path = output_root / "pair_run_receipt.json"
    if run_receipt_path.exists() or run_receipt_path.is_symlink():
        raise FileExistsError("pair run receipt already exists")
    runtime = resolve_n0_runtime_artifacts(integration_config)
    if runtime.manifest.task_id != task:
        raise N0FaultCampaignError("integration config task does not match pair")
    published: list[N0FaultPairRunReceipt] = []

    def publish_before_close(result: OfficialN0PairedLiveResult) -> None:
        written = write_paired_execution_receipt(pair_receipt_path, result.paired)
        receipt = N0FaultPairRunReceipt(
            campaign_id=loaded.manifest.campaign_id,
            campaign_manifest_sha256=loaded.manifest.sha256,
            generation_receipt_file_sha256=loaded.receipt_file_sha256,
            task=task,
            pair_key=cells[0].pair_key,
            request_file_sha256s=tuple(_request_sha256(cell) for cell in cells),
            group_content_sha256=result.paired.group_content_sha256,
            paired_receipt_file_sha256=written.file_sha256,
            artifact_root_sha256s=tuple(
                artifact.external_root_sha256 for artifact in result.artifacts
            ),
            capture_profile=LiveCaptureProfile(capture_profile).value,
            action_execution_contract=action_execution_contract,
        )
        write_json(run_receipt_path, receipt.to_dict())
        published.append(receipt)

    result = execute_official_n0_paired_live_runs(
        requests,
        manifest=runtime.manifest,
        source_root=Path(n0_source_root).expanduser().absolute(),
        host=host,
        port=port,
        api_key=os.environ.get("N0_TWAM_API_KEY"),
        action_execution_contract=action_execution_contract,
        capture_profile=capture_profile,
        pre_close_publisher=publish_before_close,
    )
    if not published:
        publish_before_close(result)
    if len(published) != 1:
        raise RuntimeError("N0 fault pair receipt publication count mismatch")
    return published[0]


def _request_sha256(cell: N0FaultCampaignCellSpec) -> str:
    if cell.request_file_sha256 is None:
        raise N0FaultCampaignError("live cell lost its request hash")
    return cell.request_file_sha256


__all__ = [
    "N0_FAULT_PAIR_RUN_EVIDENCE_LEVEL",
    "N0_FAULT_PAIR_RUN_SEMANTIC_VERSION",
    "N0FaultPairRunReceipt",
    "run_n0_fault_pair",
    "select_live_pair_cells",
]

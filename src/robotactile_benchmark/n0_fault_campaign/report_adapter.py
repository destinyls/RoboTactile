"""Strict bridge from one generated N0 campaign to reportable outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
)
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCellDisposition,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    LoadedN0FaultCampaign,
    load_n0_fault_campaign_bundle,
    read_json,
)
from robotactile_benchmark.reporting.contracts import OutcomeRecord
from robotactile_benchmark.trials import Condition, TerminalStatus

from .reporting import N0FaultReportingSpec


@dataclass(frozen=True)
class LoadedN0CampaignOutcomes:
    """Executed outcomes plus explicit unmaterialized live cells."""

    campaign_root: Path
    campaign_id: str
    campaign_manifest_sha256: str
    system_id: str
    outcomes: Tuple[OutcomeRecord, ...]
    missing_live_cell_sha256s: Tuple[str, ...]
    planned_cell_count: int

    def __post_init__(self) -> None:
        if self.planned_cell_count != len(self.outcomes) + len(
            self.missing_live_cell_sha256s
        ):
            raise N0FaultCampaignError("outcome inventory does not cover campaign")

    @property
    def complete(self) -> bool:
        return not self.missing_live_cell_sha256s


def _campaign_system_id(campaign: LoadedN0FaultCampaign) -> str:
    ids = {
        load_live_univtac_request(
            campaign.root / str(cell.request_relpath)
        ).base_system_id
        for cell in campaign.manifest.cells
        if cell.condition is Condition.CLEAN
    }
    if len(ids) != 1:
        raise N0FaultCampaignError("campaign Clean cells do not share one system")
    return next(iter(ids))


def _validate_artifact_tree(campaign: LoadedN0FaultCampaign) -> None:
    base = campaign.root / "artifacts"
    if not base.exists():
        return
    expected = tuple(
        campaign.root / str(cell.artifact_relpath)
        for cell in campaign.manifest.cells
        if cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
    )
    for path in base.rglob("*"):
        if path.is_symlink():
            raise N0FaultCampaignError("campaign artifacts cannot contain symlinks")
        inside_leaf = any(path == root or root in path.parents for root in expected)
        known_parent = path.is_dir() and any(path in root.parents for root in expected)
        if not inside_leaf and not known_parent:
            raise N0FaultCampaignError("campaign contains an unknown artifact member")


def _validate_loaded_cell(
    cell: N0FaultCampaignCellSpec, system_id: str, artifact_path: Path
) -> OutcomeRecord:
    loaded = load_live_univtac_artifact(artifact_path)
    trial = loaded.trial
    result = loaded.evidence.result
    fault = loaded.fault_manifest
    if (
        trial.base_system_id != system_id
        or trial.sha256 != cell.trial_manifest_sha256
        or trial.task != cell.task
        or trial.pair_key != cell.pair_key
        or trial.condition is not cell.condition
        or result.trial_manifest_sha256 != cell.trial_manifest_sha256
        or result.pair_key != cell.pair_key
    ):
        raise N0FaultCampaignError("live artifact differs from campaign cell")
    if cell.condition is Condition.CLEAN:
        if fault is not None or cell.operator_id is not None:
            raise N0FaultCampaignError("Clean artifact contains fault metadata")
    elif (
        fault is None
        or fault.sha256 != cell.fault_manifest_sha256
        or fault.operator_id != cell.operator_id
        or fault.severity_level != cell.severity_level
    ):
        raise N0FaultCampaignError("Faulted artifact differs from campaign cell")
    return OutcomeRecord(
        system_id=system_id,
        task=cell.task,
        pair_key=cell.pair_key,
        condition=cell.condition,
        terminal_status=result.terminal_status,
        score_eligible=result.score_eligible,
        score_success=result.score_success,
        source_root_sha256=loaded.root_receipt_sha256,
        operator_id=cell.operator_id,
        severity_level=cell.severity_level,
    )


def _unsupported_outcome(
    cell: N0FaultCampaignCellSpec, system_id: str
) -> OutcomeRecord:
    if cell.unsupported_receipt_sha256 is None:
        raise N0FaultCampaignError("unsupported cell lost its receipt hash")
    return OutcomeRecord(
        system_id=system_id,
        task=cell.task,
        pair_key=cell.pair_key,
        condition=Condition.FAULTED,
        terminal_status=TerminalStatus.UNSUPPORTED_CONTRACT,
        score_eligible=False,
        score_success=None,
        source_root_sha256=cell.unsupported_receipt_sha256,
        operator_id=cell.operator_id,
        severity_level=cell.severity_level,
    )


def load_n0_campaign_outcomes(campaign_root: Path) -> LoadedN0CampaignOutcomes:
    """Strict-load artifacts; absent live outputs remain explicit missing cells."""

    campaign = load_n0_fault_campaign_bundle(Path(campaign_root))
    system_id = _campaign_system_id(campaign)
    _validate_artifact_tree(campaign)
    outcomes = []
    missing = []
    for cell in campaign.manifest.cells:
        if cell.disposition is N0FaultCellDisposition.UNSUPPORTED_CONTRACT:
            outcomes.append(_unsupported_outcome(cell, system_id))
            continue
        artifact = campaign.root / str(cell.artifact_relpath)
        if not artifact.exists():
            missing.append(canonical_hash(cell.to_dict()))
            continue
        outcomes.append(_validate_loaded_cell(cell, system_id, artifact))
    return LoadedN0CampaignOutcomes(
        campaign_root=campaign.root,
        campaign_id=campaign.manifest.campaign_id,
        campaign_manifest_sha256=campaign.manifest.sha256,
        system_id=system_id,
        outcomes=tuple(outcomes),
        missing_live_cell_sha256s=tuple(sorted(missing)),
        planned_cell_count=campaign.manifest.cell_count,
    )


def reporting_spec_for_campaign(
    loaded: LoadedN0CampaignOutcomes, *, bootstrap_seed: int
) -> N0FaultReportingSpec:
    campaign = load_n0_fault_campaign_bundle(loaded.campaign_root)
    if campaign.manifest.sha256 != loaded.campaign_manifest_sha256:
        raise N0FaultCampaignError("loaded outcome manifest binding drifted")
    operators = campaign.manifest.operator_ids
    live_fault_operators = {
        cell.operator_id
        for cell in campaign.manifest.cells
        if cell.condition is Condition.FAULTED
        and cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
    }
    registry_ids = _campaign_severity_registry_ids(campaign)
    if len(registry_ids) != 1:
        raise N0FaultCampaignError("campaign fault manifests mix severity registries")
    return N0FaultReportingSpec(
        system_id=loaded.system_id,
        supported_operator_ids=tuple(
            item for item in operators if item in live_fault_operators
        ),
        contract_operator_ids=operators,
        severity_levels=campaign.manifest.severity_levels,
        bootstrap_seed=bootstrap_seed,
        severity_registry=next(iter(registry_ids)),
    )


def _campaign_severity_registry_ids(
    campaign: LoadedN0FaultCampaign,
) -> set[str]:
    """Strictly recover the profile identity bound by every fault manifest."""

    return {
        FaultManifest.from_dict(
            read_json(
                campaign.root / str(cell.fault_manifest_relpath),
                "campaign fault manifest",
            )
        ).severity_registry
        for cell in campaign.manifest.cells
        if cell.fault_manifest_relpath is not None
    }

"""Contract tests for official ACT Clean/Faulted campaigns."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from robotactile_benchmark.act_fault_campaign.contracts import (
    ACT_FAULT_TEMPLATE_SEED_DERIVATION,
    ACT_UNSUPPORTED_OPERATOR_IDS,
    ACTFaultCampaignCellSpec,
    ACTFaultCampaignError,
    ACTFaultCampaignManifest,
    ACTFaultCellDisposition,
    ACTUnsupportedContractSpec,
)
from robotactile_benchmark.act_fault_campaign.io import (
    ACTFaultCampaignGenerationReceipt,
)
from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition, TerminalStatus

PAIR = "a" * 64
TRIAL = "b" * 64
FAULT = "c" * 64
REQUEST = "d" * 64
UNSUPPORTED = "e" * 64
TASK = "insert_tube"
ROOT = Path(__file__).parents[1]


def _cell(
    ordinal: int,
    operator_id: str | None,
    severity: int | None = None,
) -> ACTFaultCampaignCellSpec:
    if operator_id is None:
        label = "clean"
        condition = Condition.CLEAN
        disposition = ACTFaultCellDisposition.LIVE_REQUEST
    else:
        label = f"{operator_id}/s{severity}"
        condition = Condition.FAULTED
        disposition = (
            ACTFaultCellDisposition.UNSUPPORTED_CONTRACT
            if operator_id in ACT_UNSUPPORTED_OPERATOR_IDS
            else ACTFaultCellDisposition.LIVE_REQUEST
        )
    live = disposition is ACTFaultCellDisposition.LIVE_REQUEST
    return ACTFaultCampaignCellSpec(
        ordinal=ordinal,
        task=TASK,
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        condition=condition,
        disposition=disposition,
        operator_id=operator_id,
        severity_level=severity,
        operator_template_seed=None if operator_id is None else 19 + ordinal,
        request_relpath=f"requests/{TASK}/{PAIR}/{label}.json" if live else None,
        artifact_relpath=f"artifacts/{TASK}/{PAIR}/{label}" if live else None,
        request_file_sha256=REQUEST if live else None,
        trial_manifest_sha256=TRIAL,
        fault_manifest_sha256=None if operator_id is None else FAULT,
        fault_manifest_relpath=(
            None
            if operator_id is None
            else f"fault_manifests/{TASK}/{PAIR}/{label}.json"
        ),
        unsupported_receipt_relpath=(
            f"unsupported_contracts/{TASK}/{PAIR}/{label}.json" if not live else None
        ),
        unsupported_receipt_sha256=UNSUPPORTED if not live else None,
    )


def _manifest() -> ACTFaultCampaignManifest:
    operators = tuple(sorted(CORE_OPERATOR_IDS))
    cells = (_cell(0, None),) + tuple(
        _cell(index, operator, 3) for index, operator in enumerate(operators, start=1)
    )
    return ACTFaultCampaignManifest(
        campaign_id="act-pilot",
        policy_kind=LivePolicyKind.ACT,
        profile=OfficialACTProfile.UNIVTAC,
        operator_ids=operators,
        severity_levels=(3,),
        operator_template_seed_derivation=ACT_FAULT_TEMPLATE_SEED_DERIVATION,
        pair_count=1,
        cell_count=15,
        live_request_count=13,
        unsupported_contract_count=2,
        rest_reference_bindings={
            TASK: {
                "artifact_relpath": f"rest_references/{TASK}",
                "artifact_root_sha256": "f" * 64,
                "rest_reference_sha256": "9" * 64,
            }
        },
        cells=cells,
    )


def test_manifest_round_trip_is_full_act_only_grid() -> None:
    manifest = _manifest()
    reloaded = ACTFaultCampaignManifest.from_dict(manifest.to_dict())

    assert reloaded == manifest
    assert reloaded.sha256 == manifest.sha256
    assert reloaded.profile is OfficialACTProfile.UNIVTAC
    assert reloaded.cell_count == 15
    assert reloaded.live_request_count == 13
    assert reloaded.unsupported_contract_count == 2
    assert {
        cell.operator_id
        for cell in reloaded.cells
        if cell.disposition is ACTFaultCellDisposition.UNSUPPORTED_CONTRACT
    } == ACT_UNSUPPORTED_OPERATOR_IDS
    with pytest.raises(FrozenInstanceError):
        manifest.campaign_id = "changed"  # type: ignore[misc]


def test_contract_forbids_no_touch_wrong_profile_and_live_a1() -> None:
    with pytest.raises(ACTFaultCampaignError, match="clean/faulted"):
        replace(_cell(0, None), condition=Condition.NO_TOUCH)
    with pytest.raises(ACTFaultCampaignError, match="official ACT univtac"):
        replace(_manifest(), profile=OfficialACTProfile.VISION_ONLY)
    a1 = next(
        cell for cell in _manifest().cells if cell.operator_id == "A1_stream_absence"
    )
    with pytest.raises(ACTFaultCampaignError, match="unsupported_contract"):
        replace(
            a1,
            disposition=ACTFaultCellDisposition.LIVE_REQUEST,
            request_relpath=f"requests/{TASK}/{PAIR}/A1_stream_absence/s3.json",
            artifact_relpath=f"artifacts/{TASK}/{PAIR}/A1_stream_absence/s3",
            request_file_sha256=REQUEST,
            unsupported_receipt_relpath=None,
            unsupported_receipt_sha256=None,
        )


def test_unsupported_receipt_cannot_claim_execution_or_black_substitution() -> None:
    receipt = ACTUnsupportedContractSpec(
        campaign_id="act-pilot",
        task=TASK,
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        operator_id="A2_frame_erasure",
        severity_level=3,
        operator_template_seed=29,
        fault_manifest_sha256=FAULT,
        trial_manifest_sha256=TRIAL,
    )

    assert receipt.terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
    assert ACTUnsupportedContractSpec.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ACTFaultCampaignError, match="evidence mismatch"):
        replace(receipt, black_frame_substitution_used=True)


def test_act_campaign_schemas_match_typed_top_level_contracts() -> None:
    expected = {
        "act_fault_campaign_manifest.schema.json": set(
            ACTFaultCampaignManifest.__dataclass_fields__
        ),
        "act_fault_campaign_generation_receipt.schema.json": set(
            ACTFaultCampaignGenerationReceipt.__dataclass_fields__
        ),
        "act_fault_campaign_unsupported_contract.schema.json": set(
            ACTUnsupportedContractSpec.__dataclass_fields__
        ),
    }
    for filename, fields in expected.items():
        schema = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        assert set(schema["required"]) == fields
        assert set(schema["properties"]) == fields

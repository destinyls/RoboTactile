"""Unit tests for immutable N0 fault campaign contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0_FAULT_TEMPLATE_SEED_DERIVATION,
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCampaignManifest,
    N0FaultCellDisposition,
    N0UnsupportedContractSpec,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    N0FaultCampaignGenerationReceipt,
)
from robotactile_benchmark.trials import Condition, TerminalStatus

PAIR = "a" * 64
TRIAL = "b" * 64
FAULT = "c" * 64
REQUEST = "d" * 64
UNSUPPORTED = "e" * 64
ROOT = Path(__file__).parents[1]


def _clean() -> N0FaultCampaignCellSpec:
    return N0FaultCampaignCellSpec(
        ordinal=0,
        task="insert_tube",
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        condition=Condition.CLEAN,
        disposition=N0FaultCellDisposition.LIVE_REQUEST,
        operator_id=None,
        severity_level=None,
        operator_template_seed=None,
        request_relpath=f"requests/insert_tube/{PAIR}/clean.json",
        artifact_relpath=f"artifacts/insert_tube/{PAIR}/clean",
        request_file_sha256=REQUEST,
        trial_manifest_sha256=TRIAL,
        fault_manifest_sha256=None,
        fault_manifest_relpath=None,
        unsupported_receipt_relpath=None,
        unsupported_receipt_sha256=None,
    )


def _unsupported() -> N0FaultCampaignCellSpec:
    return N0FaultCampaignCellSpec(
        ordinal=1,
        task="insert_tube",
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        condition=Condition.FAULTED,
        disposition=N0FaultCellDisposition.UNSUPPORTED_CONTRACT,
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_template_seed=19,
        request_relpath=None,
        artifact_relpath=None,
        request_file_sha256=None,
        trial_manifest_sha256="f" * 64,
        fault_manifest_sha256=FAULT,
        fault_manifest_relpath=f"fault_manifests/{PAIR}/{FAULT}.json",
        unsupported_receipt_relpath=(
            f"unsupported_contracts/insert_tube/{PAIR}/A1_stream_absence/s3.json"
        ),
        unsupported_receipt_sha256=UNSUPPORTED,
    )


def test_manifest_is_immutable_and_round_trips_exactly() -> None:
    manifest = N0FaultCampaignManifest(
        campaign_id="n0-pilot",
        policy_kind=LivePolicyKind.N0,
        operator_ids=("A1_stream_absence",),
        severity_levels=(3,),
        operator_template_seed_derivation=N0_FAULT_TEMPLATE_SEED_DERIVATION,
        pair_count=1,
        cell_count=2,
        live_request_count=1,
        unsupported_contract_count=1,
        rest_reference_bindings={},
        cells=(_clean(), _unsupported()),
    )

    reloaded = N0FaultCampaignManifest.from_dict(manifest.to_dict())

    assert reloaded == manifest
    assert reloaded.sha256 == manifest.sha256
    with pytest.raises(FrozenInstanceError):
        manifest.campaign_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.rest_reference_bindings["task"] = {}  # type: ignore[index]


def test_unsupported_receipt_forbids_execution_and_black_frame_substitution() -> None:
    receipt = N0UnsupportedContractSpec(
        campaign_id="n0-pilot",
        task="insert_tube",
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        operator_id="A2_frame_erasure",
        severity_level=5,
        operator_template_seed=29,
        fault_manifest_sha256=FAULT,
        trial_manifest_sha256=TRIAL,
    )

    assert receipt.terminal_status is TerminalStatus.UNSUPPORTED_CONTRACT
    assert receipt.live_request_generated is False
    assert receipt.simulator_execution_claimed is False
    assert receipt.black_frame_substitution_used is False
    assert N0UnsupportedContractSpec.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(N0FaultCampaignError, match="evidence mismatch"):
        replace(receipt, black_frame_substitution_used=True)


def test_manifest_rejects_missing_clean_and_unsupported_live_request() -> None:
    unsupported = _unsupported()
    with pytest.raises(N0FaultCampaignError, match="one Clean"):
        N0FaultCampaignManifest(
            campaign_id="n0-pilot",
            policy_kind=LivePolicyKind.N0,
            operator_ids=("A1_stream_absence",),
            severity_levels=(3,),
            operator_template_seed_derivation=N0_FAULT_TEMPLATE_SEED_DERIVATION,
            pair_count=1,
            cell_count=1,
            live_request_count=0,
            unsupported_contract_count=1,
            rest_reference_bindings={},
            cells=(replace(unsupported, ordinal=0),),
        )
    with pytest.raises(N0FaultCampaignError, match="only A1/A2"):
        replace(unsupported, operator_id="F5_contact_shape_distortion")


def test_manifest_requires_rest_binding_for_rest_dependent_operator() -> None:
    faulted = replace(
        _unsupported(),
        operator_id="F2_spatial_sensitivity_loss",
        disposition=N0FaultCellDisposition.LIVE_REQUEST,
        request_relpath=f"requests/insert_tube/{PAIR}/F2/s3.json",
        artifact_relpath=f"artifacts/insert_tube/{PAIR}/F2/s3",
        request_file_sha256=REQUEST,
        unsupported_receipt_relpath=None,
        unsupported_receipt_sha256=None,
    )
    with pytest.raises(N0FaultCampaignError, match="rest-reference bindings"):
        N0FaultCampaignManifest(
            campaign_id="n0-pilot",
            policy_kind=LivePolicyKind.N0,
            operator_ids=("F2_spatial_sensitivity_loss",),
            severity_levels=(3,),
            operator_template_seed_derivation=N0_FAULT_TEMPLATE_SEED_DERIVATION,
            pair_count=1,
            cell_count=2,
            live_request_count=2,
            unsupported_contract_count=0,
            rest_reference_bindings={},
            cells=(_clean(), faulted),
        )


def test_campaign_schemas_match_typed_top_level_contracts() -> None:
    expected = {
        "n0_fault_campaign_manifest.schema.json": set(
            N0FaultCampaignManifest.__dataclass_fields__
        ),
        "n0_fault_campaign_generation_receipt.schema.json": set(
            N0FaultCampaignGenerationReceipt.__dataclass_fields__
        ),
        "n0_fault_campaign_unsupported_contract.schema.json": set(
            N0UnsupportedContractSpec.__dataclass_fields__
        ),
    }
    for filename, fields in expected.items():
        schema = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        assert set(schema["required"]) == fields
        assert set(schema["properties"]) == fields

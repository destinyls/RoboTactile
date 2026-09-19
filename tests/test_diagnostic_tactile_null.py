"""All-zero black-frame tactile-null ablation contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.closed_loop.delivery import OnlineFaultSession
from robotactile_benchmark.constants import (
    DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.n0_fault_campaign import (
    N0FaultCampaignError,
    N0FaultCampaignGenerationSpec,
)
from robotactile_benchmark.resources import load_severity_registry
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.severity import (
    DIAGNOSTIC_TACTILE_NULL_VALUES,
    severity_value,
)


def _manifest(length: int = 12) -> FaultManifest:
    return FaultManifest(
        operator_id="F1_global_response_drift",
        severity_level=5,
        operator_seed=20260829,
        start_index=0,
        stop_index=length,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={},
        severity_registry=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    )


def test_tactile_null_replaces_both_streams_from_first_observation() -> None:
    clean = make_synthetic_episode()
    manifest = _manifest(len(clean))
    result = apply_fault(clean, manifest)

    assert result.validation.passed, result.validation.failures
    assert result.validation.metrics["achieved_dose"] == pytest.approx(1.0)
    assert result.validation.metrics["black_frame_max_abs_value"] == 0.0
    assert manifest.parameters["ablation_id"] == "tactile_null_black_frame_v1"
    assert manifest.parameters["response_domain"] == "absolute_black_frame"
    assert "rest_reference_sha256" not in manifest.parameters
    assert manifest.parameters["temporal_path"] == "immediate_step"
    for index, (source, delivered) in enumerate(zip(clean, result.records)):
        assert np.array_equal(
            source.observation.vision["top"],
            delivered.observation.vision["top"],
        )
        assert np.array_equal(source.observation.proprio, delivered.observation.proprio)
        for slot_id in manifest.sensor_slots:
            source_payload = source.observation.sensor(slot_id).payload
            sensor = delivered.observation.sensor(slot_id)
            assert source_payload is not None
            assert sensor.payload is not None
            assert sensor.payload.dtype == source_payload.dtype
            assert sensor.payload.shape == source_payload.shape
            assert np.count_nonzero(sensor.payload) == 0
            assert delivered.provenance_for(slot_id).source_index == index
            assert delivered.provenance_for(slot_id).active_fault_ids == (
                manifest.operator_id,
            )


def test_tactile_null_streaming_matches_batch_reference_exactly() -> None:
    clean = make_synthetic_episode()
    manifest = _manifest(len(clean))
    batch = apply_fault(clean, manifest)
    session = OnlineFaultSession(manifest)
    streamed = tuple(session.deliver(record) for record in clean)
    finalization = session.finalize()

    assert finalization.validation is not None
    assert finalization.validation.passed
    assert tuple(item.delivered_record_sha256 for item in streamed) == tuple(
        item.delivered_record_sha256 for item in batch.records
    )


@pytest.mark.parametrize(
    ("operator_id", "start_index", "sensor_slots", "message"),
    (
        ("F2_spatial_sensitivity_loss", 0, ("left", "right"), "only permits F1"),
        ("F1_global_response_drift", 1, ("left", "right"), "observation zero"),
        ("F1_global_response_drift", 0, ("left",), "both sensor slots"),
    ),
)
def test_tactile_null_rejects_non_ablation_manifests(
    operator_id: str,
    start_index: int,
    sensor_slots: tuple[str, ...],
    message: str,
) -> None:
    parameters: dict[str, object] = {}
    with pytest.raises(ValueError, match=message):
        FaultManifest(
            operator_id=operator_id,
            severity_level=5,
            operator_seed=1,
            start_index=start_index,
            stop_index=12,
            sensor_slots=sensor_slots,
            observability=Observability.BLIND,
            parameters=parameters,
            severity_registry=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
        )


def test_tactile_null_resource_and_campaign_contract_are_explicit(
    tmp_path: Path,
) -> None:
    registry = load_severity_registry(DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID)
    assert registry["calibration_status"].startswith("all-zero black-frame")
    assert {
        operator_id: entry["value"] for operator_id, entry in registry["paths"].items()
    } == DIAGNOSTIC_TACTILE_NULL_VALUES
    assert (
        severity_value(
            "F1_global_response_drift",
            5,
            registry_id=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
        )
        == 0.0
    )

    spec = N0FaultCampaignGenerationSpec(
        campaign_id="n0-tactile-null",
        base_clean_request_paths=(tmp_path / "clean.json",),
        operator_ids=("F1_global_response_drift",),
        severity_levels=(5,),
        operator_seed_master=17,
        fault_start_index=0,
        fault_stop_index=600,
        rest_reference_artifacts={},
        severity_registry=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
    )
    assert spec.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID

    with pytest.raises(N0FaultCampaignError, match="only F1"):
        N0FaultCampaignGenerationSpec(
            campaign_id="invalid-null",
            base_clean_request_paths=(tmp_path / "clean.json",),
            operator_ids=("F2_spatial_sensitivity_loss",),
            severity_levels=(5,),
            operator_seed_master=17,
            fault_start_index=0,
            fault_stop_index=600,
            rest_reference_artifacts={},
            severity_registry=DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID,
        )

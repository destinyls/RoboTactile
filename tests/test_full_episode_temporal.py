"""Opt-in causal temporal schedules without changing legacy profile instances."""

from dataclasses import replace

import numpy as np
import pytest
from test_optical14_profile import optical_fixture

from robotactile_benchmark.constants import (
    DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    OPTICAL_MARKER_REGISTRY_ID,
    OPTICAL_MARKER_REGISTRY_IDS,
    OPTICAL_MARKER_STRESS_REGISTRY_ID,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.contracts import build_evaluation_record
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.runtime import apply_fault
from robotactile_benchmark.streaming.session import StreamingFaultSession
from robotactile_benchmark.validators import validate_delivery

OPERATORS = (
    "T1_fixed_source_delay",
    "T2_held_last_freeze",
    "T3_inter_sensor_skew",
)


def make_manifest(
    operator: str,
    *,
    registry: str = OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    start: int = 0,
    stop: int = 26,
    schedule: object = "full_episode_v1",
) -> FaultManifest:
    return FaultManifest(
        operator_id=operator,
        severity_level=5,
        operator_seed=23,
        start_index=start,
        stop_index=stop,
        sensor_slots=("left", "right"),
        observability=Observability.BLIND,
        parameters={
            **(
                {"sample_period_s": 0.1}
                if registry in OPTICAL_MARKER_REGISTRY_IDS
                else {}
            ),
            "temporal_schedule": schedule,
        },
        severity_registry=registry,
    )


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize(
    "registry",
    [
        OPTICAL_MARKER_REGISTRY_ID,
        OPTICAL_MARKER_STRESS_REGISTRY_ID,
        OPTICAL_MARKER_EXTREME_REGISTRY_ID,
    ],
)
def test_full_episode_causal_routes_boundaries_and_stream_batch(operator, registry):
    clean, _ = optical_fixture(0.1)
    manifest = make_manifest(operator, registry=registry)
    batch = apply_fault(clean, manifest)
    assert batch.validation.passed, batch.validation
    session = StreamingFaultSession(manifest)
    streamed = tuple(session.deliver_one(record) for record in clean)
    assert [r.delivered_record_sha256 for r in streamed] == [
        r.delivered_record_sha256 for r in batch.records
    ]
    for index in range(manifest.stop_index):
        for slot in manifest.sensor_slots:
            if operator == OPERATORS[1]:
                source = 0
            elif operator == OPERATORS[0]:
                source = max(0, index - manifest.parameters["lag_frames"])
            else:
                source = (
                    max(0, index - manifest.parameters["skew_frames"])
                    if slot == "right"
                    else index
                )
            actual = streamed[index].provenance_for(slot)
            assert actual.source_index == source <= index
            assert actual.active_fault_ids == (operator,)
            assert streamed[index].observation.sensor(slot).delivery_index == index
            np.testing.assert_array_equal(
                streamed[index].observation.sensor(slot).payload,
                clean[source].observation.sensor(slot).payload,
            )
    for slot in manifest.sensor_slots:
        np.testing.assert_array_equal(
            streamed[0].observation.sensor(slot).payload,
            clean[0].observation.sensor(slot).payload,
        )
    for index in range(manifest.stop_index, len(clean)):
        assert (
            streamed[index].delivered_record_sha256
            == clean[index].delivered_record_sha256
        )


def test_full_freeze_outlasts_baseline_and_reports_actual_duration():
    manifest = make_manifest(OPERATORS[1])
    assert manifest.parameters["requested_duration_s"] == 1.5
    assert manifest.parameters["hold_policy"] == "until_episode_end"
    assert manifest.parameters["effective_hold_duration_s"] == pytest.approx(2.6)
    assert manifest.parameters["hold_duration_frames"] == 26
    assert manifest.parameters["source_index_map"] == (0,) * 26


def test_t1_startup_is_explicit_and_full_lag_begins_only_after_history_exists():
    manifest = make_manifest(OPERATORS[0])
    assert manifest.parameters["startup_policy"] == "hold_first"
    assert manifest.parameters["steady_state_start_index"] == 15
    sources = manifest.parameters["source_index_map"]
    assert all(i - sources[i] < 15 for i in range(15))
    assert all(i - sources[i] == 15 for i in range(15, 26))


@pytest.mark.parametrize("operator", OPERATORS)
def test_roundtrip_and_reparameterized_window_keep_opt_in(operator):
    manifest = make_manifest(operator)
    assert FaultManifest.from_dict(manifest.to_dict()).sha256 == manifest.sha256
    assert manifest.reparameterized().sha256 == manifest.sha256
    longer = manifest.reparameterized(stop_index=40)
    assert longer.parameters["temporal_schedule"] == "full_episode_v1"
    assert longer.sha256 != manifest.sha256
    if operator == OPERATORS[1]:
        assert longer.parameters["source_index_map"] == (0,) * 40
        assert longer.parameters["effective_hold_duration_s"] == 4.0


@pytest.mark.parametrize("schedule", [None, False, 1, "episode", {}])
def test_invalid_schedule_rejected(schedule):
    with pytest.raises(ValueError, match="temporal_schedule"):
        make_manifest(OPERATORS[0], schedule=schedule)


def test_schedule_requires_optical_temporal_and_start_zero():
    with pytest.raises(ValueError, match="optical temporal"):
        make_manifest(OPERATORS[1], registry=SEVERITY_REGISTRY_ID)
    with pytest.raises(ValueError, match="optical temporal"):
        make_manifest("F1_global_response_drift")
    with pytest.raises(ValueError, match="start_index=0"):
        make_manifest(OPERATORS[1], start=1)


def test_default_freeze_retains_fifteen_frame_recovery_contract():
    manifest = make_manifest(OPERATORS[1])
    data = manifest.to_dict()
    data["parameters"] = {"sample_period_s": 0.1}
    legacy = FaultManifest.from_dict(data)
    assert "temporal_schedule" not in legacy.parameters
    assert "effective_hold_duration_s" not in legacy.parameters
    assert legacy.parameters["hold_duration_frames"] == 15
    assert legacy.parameters["source_index_map"] == (0,) * 15 + tuple(range(15, 26))
    clean, _ = optical_fixture(0.1)
    session = StreamingFaultSession(legacy)
    delivered = tuple(session.deliver_one(record) for record in clean)
    assert delivered[0].provenance_for("right").active_fault_ids == ()
    assert delivered[15].delivered_record_sha256 == clean[15].delivered_record_sha256
    data["operator_id"] = OPERATORS[0]
    with pytest.raises(ValueError, match="warm-up"):
        FaultManifest.from_dict(data)


@pytest.mark.parametrize("index", [1, 20])
def test_independent_validator_rejects_wrong_startup_or_steady_source(index):
    clean, _ = optical_fixture(0.1)
    manifest = make_manifest(OPERATORS[0])
    delivered = list(apply_fault(clean, manifest).records)
    record = delivered[index]
    delivered[index] = build_evaluation_record(
        record.observation,
        tuple(replace(p, source_index=index) for p in record.provenance),
    )
    assert not validate_delivery(clean, tuple(delivered), manifest).passed


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("start", (1, 4, 8))
@pytest.mark.parametrize(
    "registry",
    [
        OPTICAL_MARKER_REGISTRY_ID,
        OPTICAL_MARKER_STRESS_REGISTRY_ID,
        OPTICAL_MARKER_EXTREME_REGISTRY_ID,
        SEVERITY_REGISTRY_ID,
        DIAGNOSTIC_STRESS_MAX_REGISTRY_ID,
    ],
)
def test_window_to_end_routes_from_nonzero_onset_through_stop(
    operator, registry, start
):
    clean, _ = optical_fixture(0.1)
    manifest = make_manifest(
        operator,
        registry=registry,
        start=start,
        schedule="window_to_end_v1",
    )
    batch = apply_fault(clean, manifest)
    assert batch.validation.passed, batch.validation
    session = StreamingFaultSession(manifest)
    streamed = tuple(session.deliver_one(record) for record in clean)
    assert [record.delivered_record_sha256 for record in streamed] == [
        record.delivered_record_sha256 for record in batch.records
    ]
    assert all(
        streamed[index].delivered_record_sha256 == clean[index].delivered_record_sha256
        for index in range(manifest.start_index)
    )
    for index in range(manifest.start_index, manifest.stop_index):
        for slot in manifest.sensor_slots:
            if operator == OPERATORS[0]:
                source = max(0, index - manifest.parameters["lag_frames"])
            elif operator == OPERATORS[1]:
                source = manifest.start_index - 1
            else:
                source = (
                    max(0, index - manifest.parameters["skew_frames"])
                    if slot == "right"
                    else index
                )
            assert streamed[index].provenance_for(slot).source_index == source
            assert streamed[index].provenance_for(slot).active_fault_ids == (operator,)
    if operator in (OPERATORS[0], OPERATORS[2]):
        assert manifest.parameters["startup_policy"] == "hold_first"
        assert (
            manifest.parameters["steady_state_start_index"]
            == manifest.parameters[
                "lag_frames" if operator == OPERATORS[0] else "skew_frames"
            ]
        )
    if operator == OPERATORS[0]:
        assert manifest.parameters["source_index_map"][0] < manifest.start_index
    if operator == OPERATORS[1]:
        assert manifest.parameters["hold_policy"] == "until_window_end"
        assert (
            manifest.parameters["hold_duration_frames"] == manifest.stop_index - start
        )
    assert FaultManifest.from_dict(manifest.to_dict()).sha256 == manifest.sha256
    assert manifest.reparameterized().sha256 == manifest.sha256


def test_window_to_end_requires_nonzero_onset():
    with pytest.raises(ValueError, match="start_index>0"):
        make_manifest(OPERATORS[1], schedule="window_to_end_v1")


@pytest.mark.parametrize("operator", (OPERATORS[0], OPERATORS[2]))
@pytest.mark.parametrize("start", (1, 8))
def test_window_to_end_cold_start_reaches_fixed_lag(operator, start):
    manifest = make_manifest(operator, start=start, schedule="window_to_end_v1")
    lag = manifest.parameters[
        "lag_frames" if operator == OPERATORS[0] else "skew_frames"
    ]
    sources = manifest.parameters["source_index_map"]
    if operator == OPERATORS[2]:
        sources = sources["right"]
    assert sources[0] == 0
    assert sources[lag - start] == 0
    assert sources[lag - start + 1] == 1
    assert sources[-1] == manifest.stop_index - 1 - lag
    assert manifest.parameters["startup_policy"] == "hold_first"
    assert manifest.parameters["steady_state_start_index"] == lag


@pytest.mark.parametrize("operator", OPERATORS)
def test_window_to_end_validator_rejects_corrupted_final_delivery(operator):
    clean, _ = optical_fixture(0.1)
    manifest = make_manifest(operator, start=4, schedule="window_to_end_v1")
    delivered = list(apply_fault(clean, manifest).records)
    index = manifest.stop_index - 1
    record = delivered[index]
    delivered[index] = build_evaluation_record(
        record.observation,
        tuple(
            replace(provenance, source_index=index)
            if provenance.slot_id == "right"
            else provenance
            for provenance in record.provenance
        ),
    )
    assert not validate_delivery(clean, tuple(delivered), manifest).passed

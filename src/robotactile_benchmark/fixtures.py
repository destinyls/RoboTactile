"""Small deterministic fixtures for contract and replay validation."""

from __future__ import annotations

from typing import Any, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.constants import LOGICAL_STEP_SECONDS, SENSOR_SLOTS
from robotactile_benchmark.contracts import (
    ContactPhase,
    EvaluationRecord,
    ObservationRecord,
    SensorObservation,
    SensorProvenance,
    array_sha256,
    build_evaluation_record,
    canonical_hash,
)
from robotactile_benchmark.rest_references import (
    ReferenceSplit,
    RestReferenceBundle,
)


def make_synthetic_rest_references() -> RestReferenceBundle:
    """Create a development-split rest bundle for bounded synthetic tests."""

    return RestReferenceBundle(
        reference_id="synthetic-dev-rest-v1",
        dataset_split=ReferenceSplit.SYNTHETIC_DEVELOPMENT,
        split_manifest_sha256=canonical_hash("synthetic-split-manifest-v1"),
        source_artifact_sha256=canonical_hash("synthetic-development-artifact-v1"),
        no_contact_predicate_id="synthetic-free-phase-v1",
        no_contact_validation_sha256=canonical_hash(
            "synthetic-no-contact-validation-v1"
        ),
        no_contact_verified=True,
        qualified_record_ids={
            slot_id: f"synthetic-free-{slot_id}-0" for slot_id in SENSOR_SLOTS
        },
        calibration_sha256={
            slot_id: canonical_hash("calibration-v1") for slot_id in SENSOR_SLOTS
        },
        payloads={
            slot_id: _tactile_frame(0, sensor_index)
            for sensor_index, slot_id in enumerate(SENSOR_SLOTS)
        },
    )


def _phase(index: int) -> ContactPhase:
    if index < 3:
        return ContactPhase.FREE
    if index == 3:
        return ContactPhase.ONSET
    if index < 8:
        return ContactPhase.SUSTAINED
    if index == 8:
        return ContactPhase.RELEASE
    return ContactPhase.FREE


def _tactile_frame(index: int, sensor_index: int, size: int = 32) -> NDArray[Any]:
    base = np.full((size, size, 3), 35 + sensor_index * 8, dtype=np.float32)
    marker_offset = 2 + sensor_index
    for row in range(marker_offset, size, 4):
        for column in range(marker_offset, size, 4):
            base[row, column, :] += 105.0
    phase = _phase(index)
    if phase in {ContactPhase.ONSET, ContactPhase.SUSTAINED}:
        yy, xx = np.mgrid[0:size, 0:size]
        center_x = size * (0.45 + 0.08 * sensor_index)
        center_y = size * 0.52
        radius = 3.0 + min(index - 3, 3) * 0.65
        contact = np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * radius**2)
        )
        base[..., 0] += contact * 105.0
        base[..., 1] += contact * 70.0
        base[..., 2] += contact * 40.0
    elif phase is ContactPhase.RELEASE:
        base[7:9, 7:9, :] += 2.0
    return cast(NDArray[Any], np.clip(np.rint(base), 0, 255).astype(np.uint8))


def make_synthetic_episode(length: int = 12) -> Tuple[EvaluationRecord, ...]:
    """Create a clean episode with free/onset/contact/release phases."""

    if length < 10:
        raise ValueError("synthetic episode length must be at least 10")
    records = []
    calibration_hash = canonical_hash("calibration-v1")
    for index in range(length):
        time_s = index * LOGICAL_STEP_SECONDS
        sensors = []
        provenance = []
        for sensor_index, slot_id in enumerate(SENSOR_SLOTS):
            payload = _tactile_frame(index, sensor_index)
            sensors.append(
                SensorObservation(
                    slot_id=slot_id,
                    payload=payload,
                    payload_present=True,
                    declared_validity=True,
                    delivery_index=index,
                    delivery_time_s=time_s,
                    visible_source_time_s=None,
                    frame_id=f"{slot_id}_raw",
                    calibration_id="calibration-v1",
                )
            )
            provenance.append(
                SensorProvenance(
                    slot_id=slot_id,
                    physical_source_id=slot_id,
                    source_index=index,
                    source_time_s=time_s,
                    payload_sha256=array_sha256(payload),
                    calibration_sha256=calibration_hash,
                    phase=_phase(index),
                )
            )
        vision = {
            "top": np.full((12, 16, 3), min(255, 45 + index), dtype=np.uint8),
            "wrist_l": np.full((12, 16, 3), min(255, 65 + index), dtype=np.uint8),
        }
        observation = ObservationRecord(
            episode_id="synthetic-episode-v1",
            task="insert_HDMI",
            seed=1001,
            step_index=index,
            tactile=tuple(sensors),
            vision=vision,
            proprio=np.linspace(0.0, 0.8, 9, dtype=np.float32) + index * 0.001,
        )
        records.append(build_evaluation_record(observation, tuple(provenance)))
    return tuple(records)

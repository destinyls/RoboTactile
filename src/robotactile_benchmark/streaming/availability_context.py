"""Single-record availability and context fault delivery."""

from __future__ import annotations

from dataclasses import replace
from typing import AbstractSet, cast

import numpy as np

from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    array_sha256,
    canonical_hash,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators.base import (
    baseline_for,
    fault_declared_validity,
    replace_delivery,
)
from robotactile_benchmark.rest_references import RestReferenceBundle


def apply_availability(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    affected_indices: AbstractSet[int],
) -> EvaluationRecord:
    """Remove registered payloads at one scheduled delivery index."""

    index = clean_record.observation.step_index
    if index not in affected_indices:
        return clean_record
    output = clean_record
    declared = manifest.observability is Observability.DECLARED
    for slot_id in manifest.sensor_slots:
        sensor = output.observation.sensor(slot_id).without_payload(declared)
        provenance = replace(
            output.provenance_for(slot_id),
            source_index=None,
            source_time_s=None,
            payload_sha256=None,
            active_fault_ids=(manifest.operator_id,),
        )
        output = replace_delivery(output, sensor, provenance)
    return output


def apply_context(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
    affected_indices: AbstractSet[int],
) -> EvaluationRecord:
    """Apply one C1 or C2 delivery at the current benchmark step."""

    if manifest.operator_id == "C1_sensor_identity_misrouting":
        return _apply_c1(clean_record, manifest, affected_indices)
    if manifest.operator_id == "C2_frame_misregistration":
        return _apply_c2(clean_record, manifest, rest_references)
    raise KeyError(f"unknown streaming context operator: {manifest.operator_id}")


def _apply_c1(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    affected_indices: AbstractSet[int],
) -> EvaluationRecord:
    if clean_record.observation.step_index not in affected_indices:
        return clean_record
    if set(manifest.sensor_slots) != set(SENSOR_SLOTS):
        raise ValueError("C1 requires both left and right sensor slots")
    output = clean_record
    route_map = manifest.parameters["route_map"]
    for target_slot in SENSOR_SLOTS:
        source_slot = str(route_map[target_slot])
        target = output.observation.sensor(target_slot)
        source = clean_record.observation.sensor(source_slot)
        source_provenance = clean_record.provenance_for(source_slot)
        routed = replace(
            source,
            slot_id=target_slot,
            delivery_index=target.delivery_index,
            delivery_time_s=target.delivery_time_s,
            declared_validity=fault_declared_validity(target, manifest),
            frame_id=target.frame_id,
            calibration_id=target.calibration_id,
        )
        provenance = replace(
            source_provenance,
            slot_id=target_slot,
            payload_sha256=array_sha256(routed.payload),
            active_fault_ids=(manifest.operator_id,),
        )
        output = replace_delivery(output, routed, provenance)
    return output


def _apply_c2(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
) -> EvaluationRecord:
    index = clean_record.observation.step_index
    if not manifest.active(index):
        return clean_record
    if manifest.parameters.get("realization") != "registered_pixels":
        raise ValueError(
            "C2 requires realization='registered_pixels'; raw-RGB-only tracks are N/A"
        )
    shift_x, shift_y = (
        int(value) for value in manifest.parameters["translation_xy_px"]
    )
    if shift_x < 0 or shift_y < 0 or (shift_x == 0 and shift_y == 0):
        raise ValueError("reference C2 requires one non-zero positive integer shift")
    output = clean_record
    for slot_id in manifest.sensor_slots:
        sensor = output.observation.sensor(slot_id)
        if sensor.payload is None:
            raise ValueError("C2 requires a present payload")
        baseline = baseline_for(slot_id, manifest, rest_references)
        payload = _baseline_fill_shift(sensor.payload, baseline, shift_x, shift_y)
        calibration_id = f"{sensor.calibration_id}:misregistered:{shift_x},{shift_y}px"
        routed = replace(
            sensor,
            payload=payload,
            declared_validity=fault_declared_validity(sensor, manifest),
            calibration_id=calibration_id,
        )
        provenance = replace(
            output.provenance_for(slot_id),
            payload_sha256=array_sha256(payload),
            calibration_sha256=canonical_hash(calibration_id),
            active_fault_ids=(manifest.operator_id,),
        )
        output = replace_delivery(output, routed, provenance)
    return output


def _baseline_fill_shift(
    payload: Array,
    baseline: Array,
    shift_x: int,
    shift_y: int,
) -> Array:
    output = cast(Array, np.asarray(baseline).copy())
    height, width = payload.shape[:2]
    if shift_x >= width or shift_y >= height:
        return output
    output[shift_y:, shift_x:, :] = payload[: height - shift_y, : width - shift_x, :]
    return output

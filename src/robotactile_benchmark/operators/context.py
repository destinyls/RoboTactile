"""Sensor identity and calibration-frame context faults."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    array_sha256,
    canonical_hash,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators.base import (
    OperatorSpec,
    baseline_for,
    fault_declared_validity,
    replace_delivery,
)
from robotactile_benchmark.operators.registry import register_operator
from robotactile_benchmark.rest_references import RestReferenceBundle


def _baseline_fill_shift(
    payload: Array, baseline: Array, shift_x: int, shift_y: int
) -> Array:
    output = np.asarray(baseline).copy()
    height, width = payload.shape[:2]
    if shift_x >= width or shift_y >= height:
        return output
    output[shift_y:, shift_x:, :] = payload[: height - shift_y, : width - shift_x, :]
    return output


@register_operator("C1_sensor_identity_misrouting")
class SensorIdentityMisroutingOperator:
    spec = OperatorSpec(
        "C1_sensor_identity_misrouting",
        "context",
        "misrouted_window_fraction",
        False,
        "engineering_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        if set(manifest.sensor_slots) != set(SENSOR_SLOTS):
            raise ValueError("C1 requires both left and right sensor slots")
        output = list(clean_records)
        affected_indices = {
            manifest.start_index + int(offset)
            for offset in manifest.parameters["affected_offsets"]
        }
        route_map = manifest.parameters["route_map"]
        for index, clean in enumerate(clean_records):
            if index not in affected_indices:
                continue
            current = output[index]
            for target_slot in SENSOR_SLOTS:
                source_slot = str(route_map[target_slot])
                target = current.observation.sensor(target_slot)
                source = clean.observation.sensor(source_slot)
                source_provenance = clean.provenance_for(source_slot)
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
                current = replace_delivery(current, routed, provenance)
            output[index] = current
        return tuple(output)


@register_operator("C2_frame_misregistration")
class FrameMisregistrationOperator:
    spec = OperatorSpec(
        "C2_frame_misregistration",
        "context",
        "reprojection_displacement_pixels",
        False,
        "mechanism_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        realization = manifest.parameters.get("realization")
        if realization != "registered_pixels":
            raise ValueError(
                "C2 requires realization='registered_pixels'; raw-RGB-only tracks are N/A"
            )
        shift_x, shift_y = (
            int(value) for value in manifest.parameters["translation_xy_px"]
        )
        if shift_x < 0 or shift_y < 0 or (shift_x == 0 and shift_y == 0):
            raise ValueError(
                "reference C2 requires one non-zero positive integer shift"
            )
        output = list(clean_records)
        for index, _clean in enumerate(clean_records):
            if not manifest.active(index):
                continue
            for slot_id in manifest.sensor_slots:
                sensor = output[index].observation.sensor(slot_id)
                if sensor.payload is None:
                    raise ValueError("C2 requires a present payload")
                baseline = baseline_for(slot_id, manifest, rest_references)
                payload = _baseline_fill_shift(
                    sensor.payload, baseline, shift_x, shift_y
                )
                calibration_id = (
                    f"{sensor.calibration_id}:misregistered:{shift_x},{shift_y}px"
                )
                routed = replace(
                    sensor,
                    payload=payload,
                    declared_validity=fault_declared_validity(sensor, manifest),
                    calibration_id=calibration_id,
                )
                provenance = replace(
                    output[index].provenance_for(slot_id),
                    payload_sha256=array_sha256(payload),
                    calibration_sha256=canonical_hash(calibration_id),
                    active_fault_ids=(manifest.operator_id,),
                )
                output[index] = replace_delivery(output[index], routed, provenance)
        return tuple(output)

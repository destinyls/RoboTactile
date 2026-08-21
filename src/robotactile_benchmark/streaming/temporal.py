"""O(1) causal temporal routing for one delivered record."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from robotactile_benchmark.contracts import EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators.base import (
    fault_declared_validity,
    replace_delivery,
)

SourceLookup = Callable[[int], EvaluationRecord]


def apply_temporal(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    source_lookup: SourceLookup,
) -> EvaluationRecord:
    """Route one current delivery to its registered causal source records."""

    index = clean_record.observation.step_index
    if not manifest.active(index):
        return clean_record
    offset = index - manifest.start_index
    if manifest.operator_id in {
        "T1_fixed_source_delay",
        "T2_held_last_freeze",
    }:
        source_index = int(manifest.parameters["source_index_map"][offset])
        if manifest.operator_id == "T2_held_last_freeze" and source_index == index:
            return clean_record
        output = clean_record
        source_record = source_lookup(source_index)
        for slot_id in manifest.sensor_slots:
            output = _route_source(output, source_record, slot_id, manifest)
        return output
    if manifest.operator_id == "T3_inter_sensor_skew":
        output = clean_record
        source_map = manifest.parameters["source_index_map"]
        for slot_id in manifest.sensor_slots:
            source_index = int(source_map[slot_id][offset])
            output = _route_source(
                output, source_lookup(source_index), slot_id, manifest
            )
        return output
    raise KeyError(f"unknown streaming temporal operator: {manifest.operator_id}")


def maximum_source_age(manifest: FaultManifest) -> int:
    """Return the bounded causal history required by one temporal manifest."""

    if manifest.operator_id in {
        "T1_fixed_source_delay",
        "T2_held_last_freeze",
    }:
        source_map = tuple(
            int(value) for value in manifest.parameters["source_index_map"]
        )
        delivery = range(manifest.start_index, manifest.stop_index)
        return _validate_source_ages(delivery, source_map)
    if manifest.operator_id == "T3_inter_sensor_skew":
        source_map = manifest.parameters["source_index_map"]
        maximum = 0
        for slot_id in manifest.sensor_slots:
            delivery = range(manifest.start_index, manifest.stop_index)
            sources = tuple(int(value) for value in source_map[slot_id])
            maximum = max(maximum, _validate_source_ages(delivery, sources))
        return maximum
    return 0


def _validate_source_ages(
    delivery_indices: range, source_indices: tuple[int, ...]
) -> int:
    delivery = tuple(delivery_indices)
    if len(delivery) != len(source_indices):
        raise ValueError("temporal source map length disagrees with fault window")
    ages = tuple(
        delivery_index - source_index
        for delivery_index, source_index in zip(delivery, source_indices)
    )
    if any(source < 0 for source in source_indices) or any(age < 0 for age in ages):
        raise ValueError("temporal source map must be causal and non-negative")
    return max(ages, default=0)


def _route_source(
    output_record: EvaluationRecord,
    source_record: EvaluationRecord,
    slot_id: str,
    manifest: FaultManifest,
) -> EvaluationRecord:
    current_sensor = output_record.observation.sensor(slot_id)
    source_sensor = source_record.observation.sensor(slot_id)
    source_provenance = source_record.provenance_for(slot_id)
    sensor = replace(
        source_sensor,
        slot_id=slot_id,
        delivery_index=current_sensor.delivery_index,
        delivery_time_s=current_sensor.delivery_time_s,
        declared_validity=fault_declared_validity(current_sensor, manifest),
    )
    provenance = replace(
        source_provenance,
        slot_id=slot_id,
        active_fault_ids=(manifest.operator_id,),
    )
    return replace_delivery(output_record, sensor, provenance)

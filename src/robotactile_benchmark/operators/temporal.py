"""Causal temporal source mappings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Optional, Tuple

from robotactile_benchmark.contracts import EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators.base import (
    OperatorSpec,
    fault_declared_validity,
    replace_delivery,
)
from robotactile_benchmark.operators.registry import register_operator
from robotactile_benchmark.rest_references import RestReferenceBundle


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


@register_operator("T1_fixed_source_delay")
class FixedSourceDelayOperator:
    spec = OperatorSpec(
        "T1_fixed_source_delay", "temporal", "lag_frames", False, "engineering_proxy"
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        source_map = tuple(
            int(value) for value in manifest.parameters["source_index_map"]
        )
        output = list(clean_records)
        for offset, index in enumerate(
            range(manifest.start_index, min(manifest.stop_index, len(clean_records)))
        ):
            source_index = source_map[offset]
            for slot_id in manifest.sensor_slots:
                output[index] = _route_source(
                    output[index], clean_records[source_index], slot_id, manifest
                )
        return tuple(output)


@register_operator("T2_held_last_freeze")
class HeldLastFreezeOperator:
    spec = OperatorSpec(
        "T2_held_last_freeze",
        "temporal",
        "held_duration_frames",
        True,
        "engineering_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        output = list(clean_records)
        source_map = tuple(
            int(value) for value in manifest.parameters["source_index_map"]
        )
        for offset, index in enumerate(
            range(manifest.start_index, min(manifest.stop_index, len(clean_records)))
        ):
            source_index = source_map[offset]
            if source_index == index:
                continue
            for slot_id in manifest.sensor_slots:
                output[index] = _route_source(
                    output[index], clean_records[source_index], slot_id, manifest
                )
        return tuple(output)


@register_operator("T3_inter_sensor_skew")
class InterSensorSkewOperator:
    spec = OperatorSpec(
        "T3_inter_sensor_skew",
        "temporal",
        "source_index_gap",
        True,
        "direct_observation",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        output = list(clean_records)
        source_map = manifest.parameters["source_index_map"]
        for offset, index in enumerate(
            range(manifest.start_index, min(manifest.stop_index, len(clean_records)))
        ):
            for slot_id in manifest.sensor_slots:
                source_index = int(source_map[slot_id][offset])
                output[index] = _route_source(
                    output[index], clean_records[source_index], slot_id, manifest
                )
        return tuple(output)

"""Availability operators preserve structural absence semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Optional, Tuple

from robotactile_benchmark.contracts import EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators.base import OperatorSpec, replace_delivery
from robotactile_benchmark.operators.registry import register_operator
from robotactile_benchmark.rest_references import RestReferenceBundle


def _absence(
    clean_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    erased_indices: Sequence[int],
) -> Tuple[EvaluationRecord, ...]:
    output = list(clean_records)
    declared = manifest.observability is Observability.DECLARED
    erased = set(erased_indices)
    for index, record in enumerate(clean_records):
        if index not in erased:
            continue
        for slot_id in manifest.sensor_slots:
            sensor = record.observation.sensor(slot_id).without_payload(declared)
            provenance = replace(
                record.provenance_for(slot_id),
                source_index=None,
                source_time_s=None,
                payload_sha256=None,
                active_fault_ids=(manifest.operator_id,),
            )
            output[index] = replace_delivery(output[index], sensor, provenance)
    return tuple(output)


@register_operator("A1_stream_absence")
class StreamAbsenceOperator:
    spec = OperatorSpec(
        "A1_stream_absence",
        "availability",
        "affected_window_fraction",
        False,
        "engineering_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        offsets = tuple(int(value) for value in manifest.parameters["affected_offsets"])
        return _absence(
            clean_records,
            manifest,
            [manifest.start_index + offset for offset in offsets],
        )


@register_operator("A2_frame_erasure")
class FrameErasureOperator:
    spec = OperatorSpec(
        "A2_frame_erasure",
        "availability",
        "erased_frame_fraction",
        False,
        "engineering_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        erased = [
            manifest.start_index + int(offset)
            for offset in manifest.parameters["erased_offsets"]
        ]
        return _absence(clean_records, manifest, erased)

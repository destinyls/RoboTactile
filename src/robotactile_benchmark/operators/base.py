"""Operator specifications and transformation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Optional, Protocol, Tuple, cast

import numpy as np

from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    SensorObservation,
    SensorProvenance,
    array_sha256,
    build_evaluation_record,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.rest_references import RestReferenceBundle


@dataclass(frozen=True)
class OperatorSpec:
    """Static registry metadata for one canonical operator."""

    operator_id: str
    family: str
    native_unit: str
    stateful: bool
    evidence_tier: str


class FaultOperator(Protocol):
    """Deterministic batch operator protocol."""

    spec: OperatorSpec

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]: ...


def clip_like(reference: Array, normalized: Array) -> Array:
    """Clip a normalized float image and restore the reference dtype."""

    normalized = np.clip(normalized, 0.0, 1.0)
    if np.issubdtype(reference.dtype, np.integer):
        return cast(Array, np.rint(normalized * 255.0).astype(reference.dtype))
    return cast(Array, normalized.astype(reference.dtype))


def normalize(payload: Array) -> Array:
    """Normalize raw tactile RGB to [0, 1]."""

    array: Array = payload.astype(np.float32)
    if np.issubdtype(payload.dtype, np.integer) or float(array.max(initial=0.0)) > 1.5:
        array /= 255.0
    return cast(Array, np.clip(array, 0.0, 1.0))


def baseline_for(
    slot_id: str,
    manifest: FaultManifest,
    rest_references: Optional[RestReferenceBundle],
) -> Array:
    """Load a development-split rest reference bound by its manifest hash."""

    if rest_references is None:
        raise ValueError("this operator requires a frozen rest-reference bundle")
    if manifest.parameters["rest_reference_sha256"] != rest_references.sha256:
        raise ValueError("rest-reference bundle does not match the fault manifest")
    return rest_references.payload_for(slot_id)


def replace_delivery(
    record: EvaluationRecord,
    sensor: SensorObservation,
    provenance: SensorProvenance,
) -> EvaluationRecord:
    """Replace one slot while preserving the clean record hash."""

    observation = record.observation.replace_sensor(sensor)
    items = tuple(
        provenance if item.slot_id == provenance.slot_id else item
        for item in record.provenance
    )
    return build_evaluation_record(
        observation=observation,
        provenance=items,
        clean_record_sha256=record.clean_record_sha256,
    )


def mark_provenance(
    provenance: SensorProvenance,
    manifest: FaultManifest,
    payload: Optional[Array] = None,
) -> SensorProvenance:
    """Attach the active operator and an optional delivered payload hash."""

    return replace(
        provenance,
        active_fault_ids=tuple(
            sorted(set(provenance.active_fault_ids + (manifest.operator_id,)))
        ),
        payload_sha256=array_sha256(payload)
        if payload is not None
        else provenance.payload_sha256,
    )


def fault_declared_validity(
    sensor: SensorObservation, manifest: FaultManifest
) -> Optional[bool]:
    """Expose a current health alert only in the declared condition."""

    if manifest.observability is Observability.DECLARED:
        return False
    return sensor.declared_validity

"""Delivery validation is a hard gate before benchmark scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from robotactile_benchmark.constants import DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID
from robotactile_benchmark.contracts import (
    EvaluationRecord,
    array_sha256,
    canonical_hash,
    delivered_hash,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operator_validators import validate_operator_semantics
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.severity import severity_value


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable validator outcome."""

    passed: bool
    failure_codes: Tuple[str, ...]
    failures: Tuple[str, ...]
    metrics: Mapping[str, Any]


def _failure(code: str, message: str, codes: List[str], failures: List[str]) -> None:
    if code not in codes:
        codes.append(code)
        failures.append(message)


def _window_indices(manifest: FaultManifest, episode_length: int) -> Tuple[int, ...]:
    stop_index = min(manifest.stop_index, episode_length)
    return tuple(range(manifest.start_index, max(manifest.start_index, stop_index)))


def _availability_indices(
    manifest: FaultManifest, episode_length: int
) -> Tuple[int, ...]:
    if manifest.operator_id == "A2_frame_erasure":
        return tuple(
            manifest.start_index + int(offset)
            for offset in manifest.parameters["erased_offsets"]
        )
    return tuple(
        manifest.start_index + int(offset)
        for offset in manifest.parameters["affected_offsets"]
    )


def _slot_delivery_hash(record: EvaluationRecord, slot_id: str) -> str:
    return canonical_hash(
        {
            "sensor": record.observation.sensor(slot_id),
            "provenance": record.provenance_for(slot_id),
        }
    )


def _slot_matches_clean(
    clean: EvaluationRecord, delivered: EvaluationRecord, slot_id: str
) -> bool:
    return _slot_delivery_hash(clean, slot_id) == _slot_delivery_hash(
        delivered, slot_id
    )


def _non_tactile_hash(record: EvaluationRecord) -> str:
    observation = record.observation
    return canonical_hash(
        {
            "episode_id": observation.episode_id,
            "task": observation.task,
            "seed": observation.seed,
            "step_index": observation.step_index,
            "vision": observation.vision,
            "proprio": observation.proprio,
        }
    )


def validate_delivery(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    rest_references: Optional[RestReferenceBundle] = None,
) -> ValidationReport:
    """Validate causality, window identity, payload hashes, and operator semantics."""

    codes: List[str] = []
    failures: List[str] = []
    metrics: Dict[str, Any] = {
        "operator_id": manifest.operator_id,
        "native_dose": severity_value(
            manifest.operator_id,
            manifest.severity_level,
            registry_id=manifest.severity_registry,
        ),
        "affected_records": 0,
    }
    if manifest.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID:
        metrics.update(
            diagnostic_ablation="tactile_null_black_frame_v1",
            paper_s1_s5_claim=False,
        )
    if len(clean_records) != len(delivered_records):
        _failure(
            "LENGTH_MISMATCH", "clean and delivered lengths differ", codes, failures
        )
        return ValidationReport(False, tuple(codes), tuple(failures), metrics)
    if (
        manifest.severity_registry == DIAGNOSTIC_TACTILE_NULL_REGISTRY_ID
        and manifest.stop_index < len(clean_records)
    ):
        _failure(
            "TACTILE_NULL_WINDOW_NOT_FULL_TRACE",
            "tactile-null delivery stopped before the observed trace ended",
            codes,
            failures,
        )

    availability_indices = set()
    if manifest.operator_id in {"A1_stream_absence", "A2_frame_erasure"}:
        availability_indices = set(
            _availability_indices(manifest, len(delivered_records))
        )

    for index, (clean, delivered) in enumerate(zip(clean_records, delivered_records)):
        clean_hash = delivered_hash(clean.observation, clean.provenance)
        recomputed_hash = delivered_hash(delivered.observation, delivered.provenance)
        if delivered.clean_record_sha256 != clean_hash:
            _failure(
                "CLEAN_REFERENCE_MISMATCH",
                f"record {index} does not reference its paired clean record",
                codes,
                failures,
            )
        if recomputed_hash != delivered.delivered_record_sha256:
            _failure(
                "STALE_DELIVERED_HASH",
                f"record {index} has a cached hash that disagrees with its content",
                codes,
                failures,
            )
        if not manifest.active(index):
            if recomputed_hash != clean_hash:
                _failure(
                    "OUTSIDE_WINDOW_CHANGED",
                    f"record {index} changed outside the registered window",
                    codes,
                    failures,
                )
            continue
        if recomputed_hash != delivered_hash(clean.observation, clean.provenance):
            metrics["affected_records"] += 1
        if _non_tactile_hash(delivered) != _non_tactile_hash(clean):
            _failure(
                "WRITE_SET_VIOLATION",
                "tactile operator changed vision, proprioception, or episode metadata",
                codes,
                failures,
            )
        for slot_id in {item.slot_id for item in clean.observation.tactile} - set(
            manifest.sensor_slots
        ):
            if not _slot_matches_clean(clean, delivered, slot_id):
                _failure(
                    "WRITE_SET_VIOLATION",
                    "operator changed an undeclared tactile slot",
                    codes,
                    failures,
                )
        for slot_id in manifest.sensor_slots:
            sensor = delivered.observation.sensor(slot_id)
            provenance = delivered.provenance_for(slot_id)
            fault_is_active = manifest.operator_id in provenance.active_fault_ids
            if (
                not _slot_matches_clean(clean, delivered, slot_id)
                and not fault_is_active
            ):
                _failure(
                    "MISSING_FAULT_PROVENANCE",
                    "changed tactile delivery omits the active operator identifier",
                    codes,
                    failures,
                )
            if fault_is_active:
                clean_validity = clean.observation.sensor(slot_id).declared_validity
                expected_fault_validity: Optional[bool]
                if manifest.observability is Observability.DECLARED:
                    expected_fault_validity = False
                elif sensor.payload_present:
                    expected_fault_validity = clean_validity
                else:
                    expected_fault_validity = None
                if sensor.declared_validity is not expected_fault_validity:
                    _failure(
                        "OBSERVABILITY_MISMATCH",
                        "validity metadata disagrees with the fault observability mode",
                        codes,
                        failures,
                    )
            if provenance.source_index is not None and provenance.source_index > index:
                _failure(
                    "FUTURE_SOURCE", "future source index delivered", codes, failures
                )
            if (
                sensor.payload is not None
                and array_sha256(sensor.payload) != provenance.payload_sha256
            ):
                _failure(
                    "SOURCE_PAYLOAD_MISMATCH",
                    "payload hash disagrees with source provenance",
                    codes,
                    failures,
                )
            expected_absence = index in availability_indices
            if expected_absence:
                if sensor.payload_present or sensor.payload is not None:
                    _failure(
                        "ABSENCE_NOT_STRUCTURAL",
                        "Availability delivery used pixels instead of structural absence",
                        codes,
                        failures,
                    )
                if (
                    provenance.source_index is not None
                    or provenance.source_time_s is not None
                    or provenance.payload_sha256 is not None
                ):
                    _failure(
                        "ABSENCE_SOURCE_NOT_NULL",
                        "structurally absent payload retains source provenance",
                        codes,
                        failures,
                    )
                expected_validity = (
                    False if manifest.observability is Observability.DECLARED else None
                )
                if sensor.declared_validity is not expected_validity:
                    _failure(
                        "OBSERVABILITY_MISMATCH",
                        "Availability validity metadata disagrees with observability mode",
                        codes,
                        failures,
                    )
            elif manifest.operator_id.startswith("A") and not _slot_matches_clean(
                clean, delivered, slot_id
            ):
                _failure(
                    "NATIVE_FOOTPRINT_VIOLATION",
                    "Availability operator changed a slot outside its native footprint",
                    codes,
                    failures,
                )
            elif (
                not manifest.operator_id.startswith("A") and not sensor.payload_present
            ):
                _failure(
                    "NON_AVAILABILITY_REMOVED_PAYLOAD",
                    "Fidelity/Temporal/Context operator removed a payload",
                    codes,
                    failures,
                )

    if metrics["affected_records"] == 0:
        _failure(
            "INACTIVE_OPERATOR",
            "operator changed no registered record",
            codes,
            failures,
        )

    validate_operator_semantics(
        clean_records,
        delivered_records,
        manifest,
        codes,
        failures,
        metrics,
        rest_references=rest_references,
    )

    return ValidationReport(
        passed=not codes,
        failure_codes=tuple(codes),
        failures=tuple(failures),
        metrics=metrics,
    )

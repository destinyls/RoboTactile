"""Operator-specific delivery predicates and achieved-dose measurements."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any, List, Optional

from robotactile_benchmark.contracts import (
    EvaluationRecord,
    canonical_hash,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.signature_validators import measure_signature


def _failure(code: str, message: str, codes: List[str], failures: List[str]) -> None:
    if code not in codes:
        codes.append(code)
        failures.append(message)


def _slot_hash(record: EvaluationRecord, slot_id: str) -> str:
    return canonical_hash(
        {
            "sensor": record.observation.sensor(slot_id),
            "provenance": record.provenance_for(slot_id),
        }
    )


def _slot_matches_clean(
    clean: EvaluationRecord, delivered: EvaluationRecord, slot_id: str
) -> bool:
    return _slot_hash(clean, slot_id) == _slot_hash(delivered, slot_id)


def _validate_routed_source(
    clean_records: Sequence[EvaluationRecord],
    delivered: EvaluationRecord,
    target_slot: str,
    expected_source_index: int,
    expected_source_slot: str,
    codes: List[str],
    failures: List[str],
) -> None:
    provenance = delivered.provenance_for(target_slot)
    expected = clean_records[expected_source_index].provenance_for(expected_source_slot)
    if provenance.source_index != expected_source_index:
        _failure(
            "SOURCE_INDEX_MISMATCH",
            "routed source index does not match the operator contract",
            codes,
            failures,
        )
    if provenance.source_time_s != expected.source_time_s:
        _failure(
            "SOURCE_TIME_MISMATCH",
            "routed source time does not match the clean source record",
            codes,
            failures,
        )
    if provenance.payload_sha256 != expected.payload_sha256:
        _failure(
            "SOURCE_PAYLOAD_MISMATCH",
            "routed payload hash does not match the clean source record",
            codes,
            failures,
        )
    if provenance.physical_source_id != expected.physical_source_id:
        _failure(
            "SOURCE_IDENTITY_MISMATCH",
            "routed physical source identity does not match the clean source record",
            codes,
            failures,
        )


def _validate_pixel_signature(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    codes: List[str],
    failures: List[str],
    metrics: MutableMapping[str, Any],
    rest_references: Optional[RestReferenceBundle],
) -> None:
    measurement = measure_signature(
        clean_records, delivered_records, manifest, rest_references=rest_references
    )
    metrics["achieved_dose"] = measurement.achieved_dose
    metrics["achieved_dose_unit"] = measurement.unit
    metrics.update(measurement.diagnostics)
    if measurement.achieved_dose <= 0.0:
        _failure(
            "SIGNATURE_NOT_DELIVERED",
            "operator changed metadata but delivered no observable tactile signature",
            codes,
            failures,
        )
    if not measurement.matched:
        _failure(
            "OPERATOR_SIGNATURE_MISMATCH",
            "delivered signature does not match the registered operator and severity",
            codes,
            failures,
        )
    for failure_code in measurement.failure_codes:
        _failure(
            failure_code,
            (
                "operator delivered no observable tactile signature"
                if failure_code == "SIGNATURE_NOT_DELIVERED"
                else "required diagnostic frames are absent for the operator validator"
            ),
            codes,
            failures,
        )
    if manifest.operator_id != "C2_frame_misregistration":
        return
    calibration_changed = any(
        delivered_records[index].provenance_for(slot_id).calibration_sha256
        != clean_records[index].provenance_for(slot_id).calibration_sha256
        for index in range(
            manifest.start_index,
            min(manifest.stop_index, len(delivered_records)),
        )
        for slot_id in manifest.sensor_slots
    )
    if not calibration_changed:
        _failure(
            "CALIBRATION_PROVENANCE_UNCHANGED",
            "C2 changed pixels without recording calibration provenance",
            codes,
            failures,
        )


def _validate_fixed_delay(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    codes: List[str],
    failures: List[str],
) -> None:
    source_map = tuple(int(value) for value in manifest.parameters["source_index_map"])
    for offset, index in enumerate(
        range(manifest.start_index, min(manifest.stop_index, len(delivered_records)))
    ):
        for slot_id in manifest.sensor_slots:
            provenance = delivered_records[index].provenance_for(slot_id)
            if provenance.source_index != source_map[offset]:
                _failure(
                    "SOURCE_PAYLOAD_MISMATCH",
                    "T1 source index does not match the registered payload lag",
                    codes,
                    failures,
                )
                continue
            _validate_routed_source(
                clean_records,
                delivered_records[index],
                slot_id,
                provenance.source_index,
                slot_id,
                codes,
                failures,
            )


def _validate_held_last(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    codes: List[str],
    failures: List[str],
) -> None:
    configured_duration = int(manifest.parameters["hold_duration_frames"])
    freeze_stop = min(manifest.stop_index, manifest.start_index + configured_duration)
    held_index = int(manifest.parameters["held_source_index"])
    for slot_id in manifest.sensor_slots:
        indices = {
            delivered_records[index].provenance_for(slot_id).source_index
            for index in range(
                manifest.start_index, min(freeze_stop, len(delivered_records))
            )
        }
        if len(indices) != 1:
            _failure(
                "FREEZE_NOT_HELD",
                "T2 source index advanced during freeze",
                codes,
                failures,
            )
        for index in range(
            manifest.start_index, min(freeze_stop, len(delivered_records))
        ):
            _validate_routed_source(
                clean_records,
                delivered_records[index],
                slot_id,
                held_index,
                slot_id,
                codes,
                failures,
            )
        for index in range(
            freeze_stop, min(manifest.stop_index, len(delivered_records))
        ):
            if not _slot_matches_clean(
                clean_records[index], delivered_records[index], slot_id
            ):
                _failure(
                    "NATIVE_FOOTPRINT_VIOLATION",
                    "T2 continued holding after its registered duration",
                    codes,
                    failures,
                )


def _validate_inter_sensor_skew(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    codes: List[str],
    failures: List[str],
    metrics: MutableMapping[str, Any],
) -> None:
    gaps = []
    source_map = manifest.parameters["source_index_map"]
    for index in range(
        manifest.start_index, min(manifest.stop_index, len(delivered_records))
    ):
        source_indices = [
            delivered_records[index].provenance_for(slot).source_index
            for slot in manifest.sensor_slots
        ]
        if any(source_index is None for source_index in source_indices):
            _failure(
                "SOURCE_INDEX_MISMATCH",
                "T3 requires a concrete source index on every routed stream",
                codes,
                failures,
            )
            continue
        concrete_indices = [
            source_index for source_index in source_indices if source_index is not None
        ]
        gaps.append(max(concrete_indices) - min(concrete_indices))
        for slot_id in manifest.sensor_slots:
            expected_source_index = int(
                source_map[slot_id][index - manifest.start_index]
            )
            _validate_routed_source(
                clean_records,
                delivered_records[index],
                slot_id,
                expected_source_index,
                slot_id,
                codes,
                failures,
            )
    metrics["max_source_index_skew"] = max(gaps) if gaps else 0
    if not gaps or max(gaps) <= 0:
        _failure("NO_INTER_SENSOR_SKEW", "T3 produced no source skew", codes, failures)


def _validate_identity_misrouting(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    codes: List[str],
    failures: List[str],
    metrics: MutableMapping[str, Any],
) -> None:
    mismatches = 0
    affected_indices = {
        manifest.start_index + int(offset)
        for offset in manifest.parameters["affected_offsets"]
    }
    route_map = manifest.parameters["route_map"]
    for index in sorted(affected_indices):
        if index >= len(delivered_records):
            continue
        for slot in manifest.sensor_slots:
            expected_source_slot = str(route_map[slot])
            if delivered_records[index].provenance_for(slot).physical_source_id != slot:
                mismatches += 1
            _validate_routed_source(
                clean_records,
                delivered_records[index],
                slot,
                index,
                expected_source_slot,
                codes,
                failures,
            )
    for index in range(
        manifest.start_index, min(manifest.stop_index, len(delivered_records))
    ):
        if index in affected_indices:
            continue
        for slot_id in manifest.sensor_slots:
            if not _slot_matches_clean(
                clean_records[index], delivered_records[index], slot_id
            ):
                _failure(
                    "NATIVE_FOOTPRINT_VIOLATION",
                    "C1 continued misrouting after its registered exposure",
                    codes,
                    failures,
                )
    metrics["identity_mismatches"] = mismatches
    if mismatches == 0:
        _failure(
            "NO_IDENTITY_MISROUTING", "C1 preserved every identity", codes, failures
        )


def validate_operator_semantics(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    codes: List[str],
    failures: List[str],
    metrics: MutableMapping[str, Any],
    rest_references: Optional[RestReferenceBundle] = None,
) -> None:
    """Apply the exact predicate for one registered non-Availability operator."""

    operator_id = manifest.operator_id
    if operator_id.startswith("F") or operator_id == "C2_frame_misregistration":
        _validate_pixel_signature(
            clean_records,
            delivered_records,
            manifest,
            codes,
            failures,
            metrics,
            rest_references,
        )
    if operator_id == "T1_fixed_source_delay":
        _validate_fixed_delay(
            clean_records, delivered_records, manifest, codes, failures
        )
    elif operator_id == "T2_held_last_freeze":
        _validate_held_last(clean_records, delivered_records, manifest, codes, failures)
    elif operator_id == "T3_inter_sensor_skew":
        _validate_inter_sensor_skew(
            clean_records, delivered_records, manifest, codes, failures, metrics
        )
    elif operator_id == "C1_sensor_identity_misrouting":
        _validate_identity_misrouting(
            clean_records, delivered_records, manifest, codes, failures, metrics
        )

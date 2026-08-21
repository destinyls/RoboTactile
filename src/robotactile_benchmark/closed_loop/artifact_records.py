"""Canonical JSON views for action traces and evaluator delivery records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple

from robotactile_benchmark.closed_loop.artifact_arrays import (
    ArrayFileStore,
    load_array,
)
from robotactile_benchmark.closed_loop.artifact_contracts import (
    BUNDLE_SEMANTIC_VERSION,
    ArtifactValidationError,
    require_sha256,
)
from robotactile_benchmark.closed_loop.capture import ActionTraceEntry
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.contracts import (
    ContactPhase,
    EvaluationRecord,
    ObservationRecord,
    SensorObservation,
    SensorProvenance,
    build_evaluation_record,
)
from robotactile_benchmark.validators import ValidationReport

_SENSOR_FIELDS = frozenset(
    {
        "slot_id",
        "payload",
        "payload_present",
        "declared_validity",
        "delivery_index",
        "delivery_time_s",
        "visible_source_time_s",
        "frame_id",
        "calibration_id",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {"episode_id", "task", "seed", "step_index", "tactile", "vision", "proprio"}
)
_PROVENANCE_FIELDS = frozenset(
    {
        "slot_id",
        "physical_source_id",
        "source_index",
        "source_time_s",
        "payload_sha256",
        "calibration_sha256",
        "phase",
        "active_fault_ids",
    }
)
_RECORD_FIELDS = frozenset(
    {"observation", "provenance", "clean_record_sha256", "delivered_record_sha256"}
)


def _mapping(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ArtifactValidationError(f"{name} fields mismatch")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactValidationError(f"{name} must be a non-negative integer")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{name} must be a JSON number")
    return float(value)


def _optional_real(value: object, name: str) -> Optional[float]:
    return None if value is None else _real(value, name)


def _optional_bool(value: object, name: str) -> Optional[bool]:
    if value is not None and type(value) is not bool:
        raise ArtifactValidationError(f"{name} must be boolean or null")
    return value


def action_trace_to_dict(
    entries: Sequence[ActionTraceEntry],
    arrays: ArrayFileStore,
) -> dict[str, object]:
    return {
        "entries": [
            {
                "action_plan_sha256": entry.action_plan_sha256,
                "source_step_index": entry.source_step_index,
                "executed_actions": arrays.add(entry.executed_actions),
            }
            for entry in entries
        ],
        "semantic_version": BUNDLE_SEMANTIC_VERSION,
    }


def action_trace_from_dict(
    value: object,
    files: Mapping[str, bytes],
    referenced_paths: set[str],
) -> Tuple[ActionTraceEntry, ...]:
    document = _mapping(
        value,
        frozenset({"entries", "semantic_version"}),
        "action trace",
    )
    if document["semantic_version"] != BUNDLE_SEMANTIC_VERSION:
        raise ArtifactValidationError("unsupported action trace semantic version")
    entries = document["entries"]
    if not isinstance(entries, list):
        raise ArtifactValidationError("action entries must be a list")
    loaded = []
    for value_entry in entries:
        entry = _mapping(
            value_entry,
            frozenset({"action_plan_sha256", "source_step_index", "executed_actions"}),
            "action entry",
        )
        loaded.append(
            ActionTraceEntry(
                action_plan_sha256=require_sha256(
                    entry["action_plan_sha256"], "action plan sha256"
                ),
                source_step_index=_integer(
                    entry["source_step_index"], "source step index"
                ),
                executed_actions=load_array(
                    entry["executed_actions"], files, referenced_paths
                ),
            )
        )
    return tuple(loaded)


def delivery_trace_to_dict(
    finalization: DeliveryFinalization,
    arrays: ArrayFileStore,
) -> dict[str, object]:
    return {
        "clean_records": [
            _record_to_dict(record, arrays) for record in finalization.clean_records
        ],
        "delivered_records": [
            _record_to_dict(record, arrays) for record in finalization.delivered_records
        ],
        "clean_trace_sha256": finalization.clean_trace_sha256,
        "delivered_trace_sha256": finalization.delivered_trace_sha256,
        "manifest_sha256": finalization.manifest_sha256,
        "semantic_version": BUNDLE_SEMANTIC_VERSION,
    }


def delivery_trace_from_dict(
    value: object,
    validation: ValidationReport,
    files: Mapping[str, bytes],
    referenced_paths: set[str],
) -> DeliveryFinalization:
    document = _mapping(
        value,
        frozenset(
            {
                "clean_records",
                "delivered_records",
                "clean_trace_sha256",
                "delivered_trace_sha256",
                "manifest_sha256",
                "semantic_version",
            }
        ),
        "delivery trace",
    )
    if document["semantic_version"] != BUNDLE_SEMANTIC_VERSION:
        raise ArtifactValidationError("unsupported delivery trace semantic version")
    clean = document["clean_records"]
    delivered = document["delivered_records"]
    if not isinstance(clean, list) or not isinstance(delivered, list):
        raise ArtifactValidationError("delivery records must be lists")
    manifest_hash = require_sha256(document["manifest_sha256"], "manifest sha256")
    return DeliveryFinalization(
        clean_records=tuple(
            _record_from_dict(item, files, referenced_paths) for item in clean
        ),
        delivered_records=tuple(
            _record_from_dict(item, files, referenced_paths) for item in delivered
        ),
        validation=validation,
        clean_trace_sha256=require_sha256(
            document["clean_trace_sha256"], "clean trace sha256"
        ),
        delivered_trace_sha256=require_sha256(
            document["delivered_trace_sha256"], "delivered trace sha256"
        ),
        manifest_sha256=manifest_hash,
    )


def _record_to_dict(
    record: EvaluationRecord,
    arrays: ArrayFileStore,
) -> dict[str, object]:
    observation = record.observation
    return {
        "observation": {
            "episode_id": observation.episode_id,
            "task": observation.task,
            "seed": observation.seed,
            "step_index": observation.step_index,
            "tactile": [
                {
                    "slot_id": sensor.slot_id,
                    "payload": None
                    if sensor.payload is None
                    else arrays.add(sensor.payload),
                    "payload_present": sensor.payload_present,
                    "declared_validity": sensor.declared_validity,
                    "delivery_index": sensor.delivery_index,
                    "delivery_time_s": sensor.delivery_time_s,
                    "visible_source_time_s": sensor.visible_source_time_s,
                    "frame_id": sensor.frame_id,
                    "calibration_id": sensor.calibration_id,
                }
                for sensor in observation.tactile
            ],
            "vision": {
                name: arrays.add(array) for name, array in observation.vision.items()
            },
            "proprio": arrays.add(observation.proprio),
        },
        "provenance": [
            {
                "slot_id": item.slot_id,
                "physical_source_id": item.physical_source_id,
                "source_index": item.source_index,
                "source_time_s": item.source_time_s,
                "payload_sha256": item.payload_sha256,
                "calibration_sha256": item.calibration_sha256,
                "phase": item.phase.value,
                "active_fault_ids": list(item.active_fault_ids),
            }
            for item in record.provenance
        ],
        "clean_record_sha256": record.clean_record_sha256,
        "delivered_record_sha256": record.delivered_record_sha256,
    }


def _record_from_dict(
    value: object,
    files: Mapping[str, bytes],
    referenced_paths: set[str],
) -> EvaluationRecord:
    record = _mapping(value, _RECORD_FIELDS, "evaluation record")
    observation = _observation_from_dict(record["observation"], files, referenced_paths)
    provenance_values = record["provenance"]
    if not isinstance(provenance_values, list):
        raise ArtifactValidationError("provenance must be a list")
    provenance = tuple(_provenance_from_dict(item) for item in provenance_values)
    clean_hash = require_sha256(record["clean_record_sha256"], "clean record sha256")
    delivered_hash = require_sha256(
        record["delivered_record_sha256"], "delivered record sha256"
    )
    rebuilt = build_evaluation_record(observation, provenance, clean_hash)
    if rebuilt.delivered_record_sha256 != delivered_hash:
        raise ArtifactValidationError("evaluation record cached hash is stale")
    return rebuilt


def _observation_from_dict(
    value: object,
    files: Mapping[str, bytes],
    referenced_paths: set[str],
) -> ObservationRecord:
    observation = _mapping(value, _OBSERVATION_FIELDS, "observation")
    tactile = observation["tactile"]
    vision = observation["vision"]
    if not isinstance(tactile, list) or not isinstance(vision, dict) or not vision:
        raise ArtifactValidationError("observation tactile/vision fields are malformed")
    if any(not isinstance(name, str) or not name for name in vision):
        raise ArtifactValidationError("vision names must be non-empty strings")
    return ObservationRecord(
        episode_id=_string(observation["episode_id"], "episode id"),
        task=_string(observation["task"], "task"),
        seed=_integer(observation["seed"], "seed"),
        step_index=_integer(observation["step_index"], "step index"),
        tactile=tuple(
            _sensor_from_dict(item, files, referenced_paths) for item in tactile
        ),
        vision={
            name: load_array(descriptor, files, referenced_paths)
            for name, descriptor in vision.items()
        },
        proprio=load_array(observation["proprio"], files, referenced_paths),
    )


def _sensor_from_dict(
    value: object,
    files: Mapping[str, bytes],
    referenced_paths: set[str],
) -> SensorObservation:
    sensor = _mapping(value, _SENSOR_FIELDS, "sensor observation")
    payload = sensor["payload"]
    if type(sensor["payload_present"]) is not bool:
        raise ArtifactValidationError("payload_present must be boolean")
    return SensorObservation(
        slot_id=_string(sensor["slot_id"], "slot id"),
        payload=(
            None if payload is None else load_array(payload, files, referenced_paths)
        ),
        payload_present=sensor["payload_present"],
        declared_validity=_optional_bool(
            sensor["declared_validity"], "declared validity"
        ),
        delivery_index=_integer(sensor["delivery_index"], "delivery index"),
        delivery_time_s=_real(sensor["delivery_time_s"], "delivery time"),
        visible_source_time_s=_optional_real(
            sensor["visible_source_time_s"], "visible source time"
        ),
        frame_id=_string(sensor["frame_id"], "frame id"),
        calibration_id=_string(sensor["calibration_id"], "calibration id"),
    )


def _provenance_from_dict(value: object) -> SensorProvenance:
    item = _mapping(value, _PROVENANCE_FIELDS, "sensor provenance")
    source_index = item["source_index"]
    payload_hash = item["payload_sha256"]
    active = item["active_fault_ids"]
    if not isinstance(active, list) or not all(
        isinstance(fault_id, str) and fault_id for fault_id in active
    ):
        raise ArtifactValidationError("active fault IDs must be a string list")
    if payload_hash is not None:
        payload_hash = require_sha256(payload_hash, "provenance payload sha256")
    return SensorProvenance(
        slot_id=_string(item["slot_id"], "provenance slot id"),
        physical_source_id=_string(item["physical_source_id"], "physical source id"),
        source_index=(
            None if source_index is None else _integer(source_index, "source index")
        ),
        source_time_s=_optional_real(item["source_time_s"], "source time"),
        payload_sha256=payload_hash,
        calibration_sha256=require_sha256(
            item["calibration_sha256"], "calibration sha256"
        ),
        phase=ContactPhase(_string(item["phase"], "contact phase")),
        active_fault_ids=tuple(active),
    )

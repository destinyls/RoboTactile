"""Canonical action and observation/provenance records for live artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Tuple

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
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_ARTIFACT_SEMANTIC_VERSION,
    LIVE_MAX_TRACE_RECORDS,
    LiveArtifactValidationError,
    require_live_sha256,
    require_optional_live_sha256,
)
from robotactile_benchmark.execution.live_artifacts_fs import LiveBundleSnapshot
from robotactile_benchmark.execution.live_artifacts_io import (
    LiveArrayWriter,
    load_live_array,
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
        raise LiveArtifactValidationError(f"{name} fields mismatch")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LiveArtifactValidationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LiveArtifactValidationError(f"{name} must be a non-negative integer")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiveArtifactValidationError(f"{name} must be a JSON number")
    return float(value)


def _optional_real(value: object, name: str) -> float | None:
    return None if value is None else _real(value, name)


def _optional_bool(value: object, name: str) -> bool | None:
    if value is not None and type(value) is not bool:
        raise LiveArtifactValidationError(f"{name} must be boolean or null")
    return value


def action_trace_to_live_dict(
    entries: Sequence[ActionTraceEntry], arrays: LiveArrayWriter
) -> dict[str, object]:
    if len(entries) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live action trace exceeds record cap")
    return {
        "entries": [
            {
                "action_plan_sha256": entry.action_plan_sha256,
                "source_step_index": entry.source_step_index,
                "executed_actions": arrays.add(entry.executed_actions),
            }
            for entry in entries
        ],
        "semantic_version": LIVE_ARTIFACT_SEMANTIC_VERSION,
    }


def action_trace_from_live_dict(
    value: object,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> Tuple[ActionTraceEntry, ...]:
    document = _mapping(
        value, frozenset({"entries", "semantic_version"}), "live action trace"
    )
    if document["semantic_version"] != LIVE_ARTIFACT_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live action trace version mismatch")
    entries = document["entries"]
    if not isinstance(entries, list) or len(entries) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live action entries are outside bounds")
    loaded = []
    for raw_entry in entries:
        entry = _mapping(
            raw_entry,
            frozenset({"action_plan_sha256", "source_step_index", "executed_actions"}),
            "live action entry",
        )
        loaded.append(
            ActionTraceEntry(
                action_plan_sha256=require_live_sha256(
                    entry["action_plan_sha256"], "action plan"
                ),
                source_step_index=_integer(entry["source_step_index"], "source step"),
                executed_actions=load_live_array(
                    entry["executed_actions"], snapshot, referenced_paths
                ),
            )
        )
    return tuple(loaded)


def delivery_trace_to_live_dict(
    finalization: DeliveryFinalization | None, arrays: LiveArrayWriter
) -> dict[str, object]:
    if finalization is None:
        return {
            "finalization": None,
            "semantic_version": LIVE_ARTIFACT_SEMANTIC_VERSION,
        }
    if len(finalization.clean_records) > LIVE_MAX_TRACE_RECORDS:
        raise LiveArtifactValidationError("live observation trace exceeds record cap")
    return {
        "finalization": {
            "clean_records": [
                _record_to_dict(record, arrays) for record in finalization.clean_records
            ],
            "delivered_records": [
                _record_to_dict(record, arrays)
                for record in finalization.delivered_records
            ],
            "clean_trace_sha256": finalization.clean_trace_sha256,
            "delivered_trace_sha256": finalization.delivered_trace_sha256,
            "manifest_sha256": finalization.manifest_sha256,
        },
        "semantic_version": LIVE_ARTIFACT_SEMANTIC_VERSION,
    }


def delivery_trace_from_live_dict(
    value: object,
    validation: ValidationReport | None,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> DeliveryFinalization | None:
    document = _mapping(
        value,
        frozenset({"finalization", "semantic_version"}),
        "live delivery trace",
    )
    if document["semantic_version"] != LIVE_ARTIFACT_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live delivery trace version mismatch")
    raw = document["finalization"]
    if raw is None:
        if validation is not None:
            raise LiveArtifactValidationError("validation cannot outlive finalization")
        return None
    finalization = _mapping(
        raw,
        frozenset(
            {
                "clean_records",
                "delivered_records",
                "clean_trace_sha256",
                "delivered_trace_sha256",
                "manifest_sha256",
            }
        ),
        "live finalization",
    )
    clean = finalization["clean_records"]
    delivered = finalization["delivered_records"]
    if (
        not isinstance(clean, list)
        or not isinstance(delivered, list)
        or len(clean) > LIVE_MAX_TRACE_RECORDS
        or len(delivered) > LIVE_MAX_TRACE_RECORDS
    ):
        raise LiveArtifactValidationError("live observation lists are outside bounds")
    return DeliveryFinalization(
        clean_records=tuple(
            _record_from_dict(item, snapshot, referenced_paths) for item in clean
        ),
        delivered_records=tuple(
            _record_from_dict(item, snapshot, referenced_paths) for item in delivered
        ),
        validation=validation,
        clean_trace_sha256=require_live_sha256(
            finalization["clean_trace_sha256"], "clean trace"
        ),
        delivered_trace_sha256=require_live_sha256(
            finalization["delivered_trace_sha256"], "delivered trace"
        ),
        manifest_sha256=require_optional_live_sha256(
            finalization["manifest_sha256"], "finalization manifest"
        ),
    )


def _record_to_dict(
    record: EvaluationRecord, arrays: LiveArrayWriter
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
                    "payload": (
                        None if sensor.payload is None else arrays.add(sensor.payload)
                    ),
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
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> EvaluationRecord:
    record = _mapping(value, _RECORD_FIELDS, "live evaluation record")
    observation = _observation_from_dict(
        record["observation"], snapshot, referenced_paths
    )
    raw_provenance = record["provenance"]
    if not isinstance(raw_provenance, list):
        raise LiveArtifactValidationError("live provenance must be a list")
    provenance = tuple(_provenance_from_dict(item) for item in raw_provenance)
    clean_hash = require_live_sha256(record["clean_record_sha256"], "clean record")
    delivered_hash = require_live_sha256(
        record["delivered_record_sha256"], "delivered record"
    )
    rebuilt = build_evaluation_record(observation, provenance, clean_hash)
    if rebuilt.delivered_record_sha256 != delivered_hash:
        raise LiveArtifactValidationError("live record cached hash is stale")
    return rebuilt


def _observation_from_dict(
    value: object,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> ObservationRecord:
    observation = _mapping(value, _OBSERVATION_FIELDS, "live observation")
    tactile = observation["tactile"]
    vision = observation["vision"]
    if not isinstance(tactile, list) or not isinstance(vision, dict) or not vision:
        raise LiveArtifactValidationError("live observation modalities are malformed")
    if any(not isinstance(name, str) or not name for name in vision):
        raise LiveArtifactValidationError("live vision names must be non-empty")
    return ObservationRecord(
        episode_id=_string(observation["episode_id"], "episode id"),
        task=_string(observation["task"], "task"),
        seed=_integer(observation["seed"], "seed"),
        step_index=_integer(observation["step_index"], "step index"),
        tactile=tuple(
            _sensor_from_dict(item, snapshot, referenced_paths) for item in tactile
        ),
        vision={
            name: load_live_array(descriptor, snapshot, referenced_paths)
            for name, descriptor in vision.items()
        },
        proprio=load_live_array(observation["proprio"], snapshot, referenced_paths),
    )


def _sensor_from_dict(
    value: object,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> SensorObservation:
    sensor = _mapping(value, _SENSOR_FIELDS, "live sensor observation")
    if type(sensor["payload_present"]) is not bool:
        raise LiveArtifactValidationError("payload_present must be boolean")
    payload = sensor["payload"]
    return SensorObservation(
        slot_id=_string(sensor["slot_id"], "slot id"),
        payload=(
            None
            if payload is None
            else load_live_array(payload, snapshot, referenced_paths)
        ),
        payload_present=sensor["payload_present"],
        declared_validity=_optional_bool(sensor["declared_validity"], "validity"),
        delivery_index=_integer(sensor["delivery_index"], "delivery index"),
        delivery_time_s=_real(sensor["delivery_time_s"], "delivery time"),
        visible_source_time_s=_optional_real(
            sensor["visible_source_time_s"], "visible source time"
        ),
        frame_id=_string(sensor["frame_id"], "frame id"),
        calibration_id=_string(sensor["calibration_id"], "calibration id"),
    )


def _provenance_from_dict(value: object) -> SensorProvenance:
    item = _mapping(value, _PROVENANCE_FIELDS, "live sensor provenance")
    active = item["active_fault_ids"]
    if not isinstance(active, list) or not all(
        isinstance(fault_id, str) and fault_id for fault_id in active
    ):
        raise LiveArtifactValidationError("active fault IDs must be strings")
    source_index = item["source_index"]
    return SensorProvenance(
        slot_id=_string(item["slot_id"], "provenance slot"),
        physical_source_id=_string(item["physical_source_id"], "physical source"),
        source_index=(
            None if source_index is None else _integer(source_index, "source index")
        ),
        source_time_s=_optional_real(item["source_time_s"], "source time"),
        payload_sha256=require_optional_live_sha256(
            item["payload_sha256"], "provenance payload"
        ),
        calibration_sha256=require_live_sha256(
            item["calibration_sha256"], "provenance calibration"
        ),
        phase=ContactPhase(_string(item["phase"], "contact phase")),
        active_fault_ids=tuple(active),
    )

"""Immutable model-boundary observations and evaluator-only provenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Optional, Tuple, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import TypeAlias

from robotactile_benchmark.constants import SENSOR_SLOTS

Array: TypeAlias = NDArray[Any]


def _require_integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _require_finite_real(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


class ContactPhase(str, Enum):
    """Evaluator-private contact phase."""

    FREE = "free"
    ONSET = "contact_onset"
    SUSTAINED = "sustained_contact"
    RELEASE = "release"


def freeze_array(value: Array, dtype: Optional[DTypeLike] = None) -> Array:
    """Defensively copy an array and mark it read-only."""

    array = cast(Array, np.ascontiguousarray(value, dtype=dtype).copy())
    array.setflags(write=False)
    return array


def freeze_value(value: Any) -> Any:
    """Freeze the strict deterministic JSON domain used by manifests."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("parameters mapping keys must be strings")
        return MappingProxyType(
            {key: freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("parameters floats must be finite")
        return value
    raise TypeError(
        "parameters values must be JSON scalars, lists, tuples, or mappings"
    )


def thaw_value(value: Any) -> Any:
    """Convert frozen values into JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): thaw_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def array_sha256(value: Optional[Array]) -> Optional[str]:
    """Hash array dtype, shape, and bytes."""

    if value is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": array_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("canonical values must use the deterministic JSON domain")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical values must contain finite floats")
    return value


def canonical_json(value: Any) -> str:
    """Serialize a value with deterministic ordering and separators."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_hash(value: Any) -> str:
    """Return a stable SHA256 for a contract value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SensorObservation:
    """The model-visible delivery for one tactile input slot."""

    slot_id: str
    payload: Optional[Array]
    payload_present: bool
    declared_validity: Optional[bool]
    delivery_index: int
    delivery_time_s: float
    visible_source_time_s: Optional[float]
    frame_id: str
    calibration_id: str

    def __post_init__(self) -> None:
        if self.slot_id not in SENSOR_SLOTS:
            raise ValueError(f"unknown tactile slot: {self.slot_id}")
        if self.payload_present != (self.payload is not None):
            raise ValueError("payload_present must equal whether payload is present")
        delivery_index = _require_integer(self.delivery_index, "delivery_index")
        delivery_time_s = _require_finite_real(self.delivery_time_s, "delivery_time_s")
        visible_source_time_s = self.visible_source_time_s
        if visible_source_time_s is not None:
            visible_source_time_s = _require_finite_real(
                visible_source_time_s, "visible_source_time_s"
            )
        if (
            visible_source_time_s is not None
            and visible_source_time_s > delivery_time_s + 1e-12
        ):
            raise ValueError("visible source time cannot be in the future")
        if self.payload is not None:
            payload = freeze_array(self.payload)
            if payload.ndim != 3 or payload.shape[-1] != 3:
                raise ValueError("tactile payload must be HWC with three channels")
            if not np.isfinite(payload).all():
                raise ValueError("tactile payload must be finite")
            if payload.dtype != np.uint8:
                raise TypeError("tactile payload must use canonical uint8 RGB encoding")
            object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "delivery_index", delivery_index)
        object.__setattr__(self, "delivery_time_s", delivery_time_s)
        object.__setattr__(self, "visible_source_time_s", visible_source_time_s)

    def without_payload(self, declared: bool) -> SensorObservation:
        """Create structural absence while preserving delivery metadata."""

        return replace(
            self,
            payload=None,
            payload_present=False,
            declared_validity=False if declared else None,
            visible_source_time_s=None,
        )


@dataclass(frozen=True)
class ObservationRecord:
    """A policy-visible multi-modal observation."""

    episode_id: str
    task: str
    seed: int
    step_index: int
    tactile: Tuple[SensorObservation, ...]
    vision: Mapping[str, Array]
    proprio: Array

    def __post_init__(self) -> None:
        seed = _require_integer(self.seed, "seed")
        step_index = _require_integer(self.step_index, "step_index")
        slots = tuple(sensor.slot_id for sensor in self.tactile)
        if len(slots) != len(set(slots)) or set(slots) != set(SENSOR_SLOTS):
            raise ValueError("tactile slots must exactly cover left and right")
        frozen_vision = {
            str(key): freeze_array(value) for key, value in self.vision.items()
        }
        if any(not np.isfinite(value).all() for value in frozen_vision.values()):
            raise ValueError("vision arrays must be finite")
        frozen_proprio = freeze_array(self.proprio)
        if not np.isfinite(frozen_proprio).all():
            raise ValueError("proprioception must be finite")
        object.__setattr__(self, "vision", MappingProxyType(frozen_vision))
        object.__setattr__(self, "proprio", frozen_proprio)
        object.__setattr__(self, "tactile", tuple(self.tactile))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "step_index", step_index)

    def sensor(self, slot_id: str) -> SensorObservation:
        for sensor in self.tactile:
            if sensor.slot_id == slot_id:
                return sensor
        raise KeyError(f"unknown tactile slot: {slot_id}")

    def replace_sensor(self, replacement: SensorObservation) -> ObservationRecord:
        found = False
        sensors = []
        for sensor in self.tactile:
            if sensor.slot_id == replacement.slot_id:
                sensors.append(replacement)
                found = True
            else:
                sensors.append(sensor)
        if not found:
            raise KeyError(f"unknown tactile slot: {replacement.slot_id}")
        return replace(self, tactile=tuple(sensors))


@dataclass(frozen=True)
class SensorProvenance:
    """Evaluator-only physical source metadata."""

    slot_id: str
    physical_source_id: str
    source_index: Optional[int]
    source_time_s: Optional[float]
    payload_sha256: Optional[str]
    calibration_sha256: str
    phase: ContactPhase
    active_fault_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.slot_id not in SENSOR_SLOTS:
            raise ValueError(f"unknown provenance slot: {self.slot_id}")
        if not self.physical_source_id:
            raise ValueError("physical source ID must be non-empty")
        source_index = self.source_index
        source_time_s = self.source_time_s
        if source_index is not None:
            source_index = _require_integer(source_index, "source_index")
        if source_time_s is not None:
            source_time_s = _require_finite_real(source_time_s, "source_time_s")
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "source_time_s", source_time_s)
        object.__setattr__(self, "active_fault_ids", tuple(self.active_fault_ids))


@dataclass(frozen=True)
class EvaluationRecord:
    """Delivered record paired with evaluator-only provenance and hashes."""

    observation: ObservationRecord
    provenance: Tuple[SensorProvenance, ...]
    clean_record_sha256: str
    delivered_record_sha256: str

    def provenance_for(self, slot_id: str) -> SensorProvenance:
        for item in self.provenance:
            if item.slot_id == slot_id:
                return item
        raise KeyError(f"unknown provenance slot: {slot_id}")

    def with_observation(self, observation: ObservationRecord) -> EvaluationRecord:
        provenance = []
        for item in self.provenance:
            payload = observation.sensor(item.slot_id).payload
            provenance.append(replace(item, payload_sha256=array_sha256(payload)))
        return build_evaluation_record(
            observation=observation,
            provenance=tuple(provenance),
            clean_record_sha256=self.clean_record_sha256,
        )


def delivered_hash(
    observation: ObservationRecord, provenance: Sequence[SensorProvenance]
) -> str:
    """Hash the complete delivered record, including evaluator provenance."""

    return canonical_hash({"observation": observation, "provenance": tuple(provenance)})


def build_evaluation_record(
    observation: ObservationRecord,
    provenance: Tuple[SensorProvenance, ...],
    clean_record_sha256: Optional[str] = None,
) -> EvaluationRecord:
    """Validate source causality and construct one evaluation record."""

    provenance = tuple(provenance)
    provenance_slots = tuple(item.slot_id for item in provenance)
    if (
        len(provenance_slots) != len(set(provenance_slots))
        or set(provenance_slots) != set(SENSOR_SLOTS)
        or set(provenance_slots) != {item.slot_id for item in observation.tactile}
    ):
        raise ValueError("observation and provenance slots must match")
    for item in provenance:
        delivery = observation.sensor(item.slot_id)
        if (
            item.source_index is not None
            and item.source_index > delivery.delivery_index
        ):
            raise ValueError("future source index is not causal")
        if (
            item.source_time_s is not None
            and item.source_time_s > delivery.delivery_time_s + 1e-12
        ):
            raise ValueError("future source time is not causal")
    current_hash = delivered_hash(observation, provenance)
    return EvaluationRecord(
        observation=observation,
        provenance=provenance,
        clean_record_sha256=clean_record_sha256 or current_hash,
        delivered_record_sha256=current_hash,
    )

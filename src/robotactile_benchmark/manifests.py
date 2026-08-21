"""Immutable, fully materialized machine-readable fault manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, cast

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    FAULT_MANIFEST_SEMANTIC_VERSION,
    OPERATOR_IMPLEMENTATION_VERSION,
    SENSOR_SLOTS,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.contracts import canonical_hash, freeze_value, thaw_value
from robotactile_benchmark.operator_parameters import (
    materialize_operator_parameters,
    rematerialization_inputs,
)

_MANIFEST_FIELDS = frozenset(
    {
        "operator_id",
        "severity_level",
        "operator_seed",
        "start_index",
        "stop_index",
        "sensor_slots",
        "observability",
        "parameters",
        "semantic_version",
        "implementation_version",
        "severity_registry",
    }
)


def _strict_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return cast(int, value)


def _strict_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


class Observability(str, Enum):
    """Whether a payload validity signal is visible to the model."""

    BLIND = "blind"
    DECLARED = "declared"


@dataclass(frozen=True)
class FaultManifest:
    """One atomic operator instance scheduled on one clean episode."""

    operator_id: str
    severity_level: int
    operator_seed: int
    start_index: int
    stop_index: int
    sensor_slots: Tuple[str, ...]
    observability: Observability
    parameters: Mapping[str, Any]
    semantic_version: str = FAULT_MANIFEST_SEMANTIC_VERSION
    implementation_version: str = OPERATOR_IMPLEMENTATION_VERSION
    severity_registry: str = SEVERITY_REGISTRY_ID

    def __post_init__(self) -> None:
        operator_id = _strict_string(self.operator_id, "operator_id")
        severity_level = _strict_integer(self.severity_level, "severity_level")
        operator_seed = _strict_integer(self.operator_seed, "operator_seed")
        start_index = _strict_integer(self.start_index, "start_index")
        stop_index = _strict_integer(self.stop_index, "stop_index")
        if operator_id not in CORE_OPERATOR_IDS:
            raise ValueError(f"unknown core operator: {operator_id}")
        if self.semantic_version != FAULT_MANIFEST_SEMANTIC_VERSION:
            raise ValueError("unsupported fault-manifest semantic version")
        if self.implementation_version != OPERATOR_IMPLEMENTATION_VERSION:
            raise ValueError("unsupported operator implementation version")
        if self.severity_registry != SEVERITY_REGISTRY_ID:
            raise ValueError("unsupported severity registry")
        if not 1 <= severity_level <= 5:
            raise ValueError("severity level must be in [1, 5]")
        if operator_seed < 0:
            raise ValueError("operator seed must be non-negative")
        if start_index < 0 or stop_index <= start_index:
            raise ValueError("fault window must satisfy 0 <= start < stop")
        if not isinstance(self.sensor_slots, (tuple, list)) or any(
            not isinstance(slot, str) for slot in self.sensor_slots
        ):
            raise TypeError("sensor_slots must be a string sequence")
        slots = tuple(self.sensor_slots)
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("sensor scope must be non-empty and unique")
        unknown = set(slots) - set(SENSOR_SLOTS)
        if unknown:
            raise ValueError(f"unknown sensor slot: {sorted(unknown)}")
        if operator_id == "C1_sensor_identity_misrouting" and set(slots) != set(
            SENSOR_SLOTS
        ):
            raise ValueError("C1 requires both left and right sensor slots")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        parameters = materialize_operator_parameters(
            operator_id,
            severity_level,
            operator_seed,
            start_index,
            stop_index,
            slots,
            self.parameters,
        )
        observability = (
            self.observability
            if isinstance(self.observability, Observability)
            else Observability(self.observability)
        )
        object.__setattr__(self, "severity_level", severity_level)
        object.__setattr__(self, "operator_seed", operator_seed)
        object.__setattr__(self, "start_index", start_index)
        object.__setattr__(self, "stop_index", stop_index)
        object.__setattr__(self, "sensor_slots", slots)
        object.__setattr__(self, "observability", observability)
        object.__setattr__(self, "parameters", freeze_value(parameters))

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def active(self, index: int) -> bool:
        return self.start_index <= index < self.stop_index

    def reparameterized(
        self,
        *,
        severity_level: Optional[int] = None,
        stop_index: Optional[int] = None,
    ) -> FaultManifest:
        """Build a fresh registered instance after severity/window changes."""

        target_severity = (
            self.severity_level if severity_level is None else severity_level
        )
        target_stop = self.stop_index if stop_index is None else stop_index
        preserve_schedule = (
            self.operator_id == "A2_frame_erasure"
            and severity_level is None
            and target_stop == self.stop_index
        )
        return FaultManifest(
            operator_id=self.operator_id,
            severity_level=target_severity,
            operator_seed=self.operator_seed,
            start_index=self.start_index,
            stop_index=target_stop,
            sensor_slots=self.sensor_slots,
            observability=self.observability,
            parameters=rematerialization_inputs(
                self.operator_id,
                self.parameters,
                preserve_erasure_schedule=preserve_schedule,
            ),
            semantic_version=self.semantic_version,
            implementation_version=self.implementation_version,
            severity_registry=self.severity_registry,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "severity_level": self.severity_level,
            "operator_seed": self.operator_seed,
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "sensor_slots": list(self.sensor_slots),
            "observability": self.observability.value,
            "parameters": thaw_value(self.parameters),
            "semantic_version": self.semantic_version,
            "implementation_version": self.implementation_version,
            "severity_registry": self.severity_registry,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FaultManifest:
        if set(value) != _MANIFEST_FIELDS:
            missing = sorted(_MANIFEST_FIELDS - set(value))
            extra = sorted(set(value) - _MANIFEST_FIELDS)
            raise ValueError(
                f"fault manifest fields mismatch: missing={missing}, extra={extra}"
            )
        sensor_slots = value["sensor_slots"]
        if not isinstance(sensor_slots, (list, tuple)) or any(
            not isinstance(item, str) for item in sensor_slots
        ):
            raise TypeError("sensor_slots must be a string sequence")
        parameters = value["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        return cls(
            operator_id=_strict_string(value["operator_id"], "operator_id"),
            severity_level=_strict_integer(value["severity_level"], "severity_level"),
            operator_seed=_strict_integer(value["operator_seed"], "operator_seed"),
            start_index=_strict_integer(value["start_index"], "start_index"),
            stop_index=_strict_integer(value["stop_index"], "stop_index"),
            sensor_slots=tuple(sensor_slots),
            observability=Observability(
                _strict_string(value["observability"], "observability")
            ),
            parameters=dict(parameters),
            semantic_version=_strict_string(
                value["semantic_version"], "semantic_version"
            ),
            implementation_version=_strict_string(
                value["implementation_version"], "implementation_version"
            ),
            severity_registry=_strict_string(
                value["severity_registry"], "severity_registry"
            ),
        )

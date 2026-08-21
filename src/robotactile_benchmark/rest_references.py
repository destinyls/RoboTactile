"""Self-attested, immutable no-contact reference payload contracts.

Real benchmark qualification additionally requires an external artifact-store
verifier; only the package's synthetic fixture is locally trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import Array, canonical_hash


class ReferenceSplit(str, Enum):
    """Splits allowed to supply calibration references."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    SYNTHETIC_DEVELOPMENT = "synthetic-development"


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


@dataclass(frozen=True)
class FrozenPayload:
    """Byte-backed image payload whose internal state cannot be made writable."""

    dtype: str
    shape: Tuple[int, ...]
    data_hex: str

    @classmethod
    def from_array(cls, value: Array) -> FrozenPayload:
        array = np.ascontiguousarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("rest reference payloads must be HWC RGB")
        if array.dtype != np.uint8:
            raise ValueError("rest reference payloads must be canonical uint8")
        return cls(str(array.dtype), tuple(array.shape), array.tobytes().hex())

    def to_array(self) -> Array:
        array = np.frombuffer(bytes.fromhex(self.data_hex), dtype=np.dtype(self.dtype))
        output = np.ascontiguousarray(array.reshape(self.shape)).copy()
        output.setflags(write=False)
        return output


@dataclass(frozen=True)
class RestReferenceBundle:
    """References naming split/predicate artifacts without verifying their store."""

    reference_id: str
    dataset_split: ReferenceSplit
    split_manifest_sha256: str
    source_artifact_sha256: str
    no_contact_predicate_id: str
    no_contact_validation_sha256: str
    no_contact_verified: bool
    qualified_record_ids: Mapping[str, str]
    calibration_sha256: Mapping[str, str]
    payloads: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("rest reference ID must be non-empty")
        if not isinstance(self.dataset_split, ReferenceSplit):
            object.__setattr__(
                self, "dataset_split", ReferenceSplit(self.dataset_split)
            )
        for value, name in (
            (self.split_manifest_sha256, "split manifest"),
            (self.source_artifact_sha256, "source artifact"),
            (self.no_contact_validation_sha256, "no-contact validation artifact"),
        ):
            _require_sha256(value, name)
        if not self.no_contact_predicate_id:
            raise ValueError("no-contact predicate ID must be non-empty")
        if type(self.no_contact_verified) is not bool or not self.no_contact_verified:
            raise ValueError("rest reference must pass the no-contact predicate")
        if set(self.qualified_record_ids) != set(SENSOR_SLOTS) or any(
            not isinstance(value, str) or not value
            for value in self.qualified_record_ids.values()
        ):
            raise ValueError("qualified record IDs must cover left and right")
        if set(self.calibration_sha256) != set(SENSOR_SLOTS):
            raise ValueError("rest calibration hashes must cover left and right")
        for calibration_hash in self.calibration_sha256.values():
            _require_sha256(calibration_hash, "rest calibration value")
        if set(self.payloads) != set(SENSOR_SLOTS):
            raise ValueError("rest reference payloads must cover left and right")
        if set(self.qualified_record_ids.values()) & {self.reference_id}:
            raise ValueError("reference ID cannot masquerade as a qualified record ID")
        frozen_payloads = {
            slot_id: payload
            if isinstance(payload, FrozenPayload)
            else FrozenPayload.from_array(payload)
            for slot_id, payload in self.payloads.items()
        }
        if len({payload.shape for payload in frozen_payloads.values()}) != 1:
            raise ValueError("rest reference payloads must share one shape")
        object.__setattr__(
            self,
            "qualified_record_ids",
            MappingProxyType(dict(self.qualified_record_ids)),
        )
        object.__setattr__(
            self,
            "calibration_sha256",
            MappingProxyType(dict(self.calibration_sha256)),
        )
        object.__setattr__(self, "payloads", MappingProxyType(frozen_payloads))

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    def payload_for(self, slot_id: str) -> Array:
        try:
            payload = self.payloads[slot_id]
        except KeyError as exc:
            raise KeyError(f"unknown rest reference slot: {slot_id}") from exc
        if not isinstance(payload, FrozenPayload):
            raise TypeError("rest reference payload store is not frozen")
        return payload.to_array()

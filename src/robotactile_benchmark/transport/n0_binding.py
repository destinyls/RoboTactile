"""Immutable snapshots that detect transport-side N0 request mutation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash, freeze_array


def _immutable_array(value: Array) -> Array:
    contiguous = np.ascontiguousarray(value)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    return immutable


def _freeze_snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("N0 request keys must be strings")
        return MappingProxyType(
            {key: _freeze_snapshot(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_snapshot(item) for item in value)
    return value


def _wire_copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return freeze_array(value)
    if isinstance(value, Mapping):
        return {key: _wire_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_wire_copy(item) for item in value)
    return value


@dataclass(frozen=True)
class RequestBinding:
    """A canonical immutable request retained across one transport call."""

    snapshot: Mapping[str, object]
    request_digest: str

    def __post_init__(self) -> None:
        frozen = cast(Mapping[str, object], _freeze_snapshot(self.snapshot))
        if frozen.get("request_digest") != self.request_digest:
            raise ValueError("N0 request binding digest field mismatch")
        unsigned = {
            key: value for key, value in frozen.items() if key != "request_digest"
        }
        if canonical_hash(unsigned) != self.request_digest:
            raise ValueError("N0 request binding content does not match its digest")
        object.__setattr__(self, "snapshot", frozen)

    @classmethod
    def from_unsigned(cls, unsigned: Mapping[str, object]) -> "RequestBinding":
        if "request_digest" in unsigned:
            raise ValueError("unsigned N0 request cannot contain request_digest")
        frozen_unsigned = cast(Mapping[str, object], _freeze_snapshot(unsigned))
        digest = canonical_hash(frozen_unsigned)
        snapshot = cast(
            Mapping[str, object],
            _freeze_snapshot({**frozen_unsigned, "request_digest": digest}),
        )
        return cls(snapshot=snapshot, request_digest=digest)

    def wire_request(self) -> dict[str, object]:
        """Return a disposable deep copy; the snapshot never reaches transport."""

        return cast(dict[str, object], _wire_copy(self.snapshot))

    def validate_returned_request(self, request: Mapping[str, object]) -> None:
        """Reject top-level or nested mutation performed during transport."""

        if set(request) != set(self.snapshot):
            raise RuntimeError("N0 transport mutated bound request fields")
        if request.get("request_digest") != self.request_digest:
            raise RuntimeError("N0 transport mutated bound request digest")
        unsigned = {
            key: value for key, value in request.items() if key != "request_digest"
        }
        if canonical_hash(unsigned) != self.request_digest:
            raise RuntimeError("N0 transport mutated bound request content")
        if canonical_hash(request) != canonical_hash(self.snapshot):
            raise RuntimeError("N0 transport mutated the immutable request binding")

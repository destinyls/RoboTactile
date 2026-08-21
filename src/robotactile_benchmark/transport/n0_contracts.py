"""Strict immutable handshake contract for the N0 qpos8 serving boundary."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from typing import Any, Tuple

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash, freeze_array

SCHEMA_VERSION = "robotactile-n0-v1"
ACTION_SPEC = "qpos8_next_step"
ACTION_SEMANTICS = "absolute"
NATIVE_ACTION_SHAPE = (8, 2, 4)
SIMULATOR_ACTION_SHAPE = (8, 8)
CAMERA_KEYS = ("top", "wrist_l")
TACTILE_KEYS = ("tactile_a", "tactile_b")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def require_string(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def require_sha256(value: Any, name: str) -> str:
    text = require_string(value, name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _require_exact_sequence(
    value: Any, expected: Tuple[Any, ...], name: str
) -> Tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(value)
    if len(normalized) != len(expected) or any(
        type(item) is not type(expected_item)
        for item, expected_item in zip(normalized, expected)
    ):
        raise TypeError(f"{name} values must have exact canonical types")
    if normalized != expected:
        raise ValueError(f"{name} must equal {expected}")
    return normalized


def _require_bounds(value: Any, name: str) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != 8:
        raise ValueError(f"{name} must contain eight values")
    result = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise TypeError(f"{name} values must be real")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} values must be finite")
        result.append(normalized)
    return tuple(result)


def validate_rpc_ack(
    request: Mapping[str, object],
    response: Mapping[str, object],
    payload_fields: Tuple[str, ...],
) -> None:
    """Require an exact ACK and bind every common transaction field."""

    if not isinstance(response, Mapping):
        raise TypeError("N0 acknowledgement must be a mapping")
    common = {
        "schema_version",
        "op",
        "request_id",
        "episode_id",
        "step_index",
        "transaction_id",
        "server_epoch",
        "request_digest",
        "status",
    }
    if set(response) != common | set(payload_fields):
        raise RuntimeError("N0 acknowledgement fields do not match the exact contract")
    string_fields = common - {"status", "step_index"}
    for name in string_fields:
        if type(request[name]) is not str or type(response[name]) is not str:
            raise TypeError(f"N0 acknowledgement {name} must be an exact string")
    if type(request["step_index"]) is not int:
        raise TypeError("N0 request step_index must be an exact integer")
    if type(response["step_index"]) is not int:
        raise TypeError("N0 acknowledgement step_index must be an exact integer")
    if type(response["status"]) is not str or response["status"] != "ok":
        raise RuntimeError("N0 acknowledgement status is not ok")
    for name in common - {"status"}:
        if canonical_hash(response[name]) != canonical_hash(request[name]):
            label = name.replace("_", " ")
            raise RuntimeError(f"N0 acknowledgement has wrong {label}")


def _require_image_shapes(
    value: Any, name: str
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != 2:
        raise ValueError(f"{name} must contain two shapes")
    normalized: list[Tuple[int, ...]] = []
    for shape in value:
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
            raise TypeError(f"{name} entries must be shape sequences")
        if len(shape) != 3:
            raise ValueError(f"{name} entries must be exact HWC shapes")
        dimensions: list[int] = []
        for dimension in shape:
            if type(dimension) is not int:
                raise TypeError(f"{name} dimensions must be exact integers")
            if dimension <= 0:
                raise ValueError(f"{name} dimensions must be positive")
            dimensions.append(dimension)
        if dimensions[-1] != 3:
            raise ValueError(f"{name} entries must be RGB shapes")
        normalized.append(tuple(dimensions))
    return cast_image_shapes(normalized)


def cast_image_shapes(
    value: list[Tuple[int, ...]],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Narrow shapes after the exact runtime checks above."""

    first, second = value
    return (
        (first[0], first[1], first[2]),
        (second[0], second[1], second[2]),
    )


def _immutable_array(value: Array) -> Array:
    """Copy into a bytes-backed array whose write flag cannot be re-enabled."""

    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _freeze_rgb(value: Any, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be exact uint8 HWC RGB")
    return _immutable_array(value)


def _freeze_qpos8(value: Any, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.float32 or value.shape != (8,):
        raise ValueError(f"{name} must be exact float32 shape (8,)")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return _immutable_array(value)


@dataclass(frozen=True)
class N0GroundingFrame:
    """One policy-visible post-action frame authorized for N0 grounding."""

    step_index: int
    top: Array
    wrist_l: Array
    tactile_a: Array
    tactile_b: Array
    proprio: Array
    frame_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_index",
            require_nonnegative_int(self.step_index, "grounding step_index"),
        )
        for name in ("top", "wrist_l", "tactile_a", "tactile_b"):
            object.__setattr__(self, name, _freeze_rgb(getattr(self, name), name))
        object.__setattr__(
            self, "proprio", _freeze_qpos8(self.proprio, "grounding proprio")
        )
        object.__setattr__(self, "frame_sha256", canonical_hash(self._unsigned()))

    def _unsigned(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "top": self.top,
            "wrist_l": self.wrist_l,
            "tactile_a": self.tactile_a,
            "tactile_b": self.tactile_b,
            "proprio": self.proprio,
        }

    def to_wire(self) -> dict[str, object]:
        """Return a second defensive copy suitable for request binding."""

        unsigned = {
            key: freeze_array(value) if isinstance(value, np.ndarray) else value
            for key, value in self._unsigned().items()
        }
        return {**unsigned, "frame_sha256": self.frame_sha256}


@dataclass(frozen=True)
class N0Handshake:
    """Exact frozen server identity, tensor layout, and cache semantics."""

    schema_version: str
    source_commit: str
    checkpoint_sha256: str
    config_sha256: str
    normalizer_sha256: str
    serve_bundle_sha256: str
    action_spec: str
    action_semantics: str
    action_dim: int
    native_action_shape: Tuple[int, ...]
    simulator_action_shape: Tuple[int, ...]
    camera_keys: Tuple[str, ...]
    tactile_keys: Tuple[str, ...]
    camera_shapes: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    tactile_shapes: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    tactile_optional: bool
    action_lower_bounds: Tuple[float, ...]
    action_upper_bounds: Tuple[float, ...]
    prompt_manifest_sha256: str
    requires_commit: bool
    cold_seed_mode: str
    no_cold_frame_skip: bool
    server_epoch: str

    def __post_init__(self) -> None:
        schema_version = require_string(self.schema_version, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        source_commit = require_string(self.source_commit, "source_commit")
        if _GIT_COMMIT.fullmatch(source_commit) is None:
            raise ValueError("source_commit must be a lowercase 40-character commit")
        object.__setattr__(self, "source_commit", source_commit)
        for name in (
            "checkpoint_sha256",
            "config_sha256",
            "normalizer_sha256",
            "serve_bundle_sha256",
            "prompt_manifest_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        action_spec = require_string(self.action_spec, "action_spec")
        action_semantics = require_string(self.action_semantics, "action_semantics")
        if action_spec != ACTION_SPEC or action_semantics != ACTION_SEMANTICS:
            raise ValueError("N0 must serve absolute qpos8_next_step actions")
        if type(self.action_dim) is not int:
            raise TypeError("action_dim must be an exact integer")
        if require_nonnegative_int(self.action_dim, "action_dim") != 8:
            raise ValueError("N0 action_dim must equal 8")
        for name, expected in (
            ("native_action_shape", NATIVE_ACTION_SHAPE),
            ("simulator_action_shape", SIMULATOR_ACTION_SHAPE),
            ("camera_keys", CAMERA_KEYS),
            ("tactile_keys", TACTILE_KEYS),
        ):
            object.__setattr__(
                self, name, _require_exact_sequence(getattr(self, name), expected, name)
            )
        object.__setattr__(
            self,
            "camera_shapes",
            _require_image_shapes(self.camera_shapes, "camera_shapes"),
        )
        object.__setattr__(
            self,
            "tactile_shapes",
            _require_image_shapes(self.tactile_shapes, "tactile_shapes"),
        )
        if _require_bool(self.tactile_optional, "tactile_optional"):
            raise ValueError("frozen N0 requires tactile inputs")
        lower = _require_bounds(self.action_lower_bounds, "action_lower_bounds")
        upper = _require_bounds(self.action_upper_bounds, "action_upper_bounds")
        if any(low >= high for low, high in zip(lower, upper)):
            raise ValueError("N0 action lower bounds must be below upper bounds")
        object.__setattr__(self, "action_lower_bounds", lower)
        object.__setattr__(self, "action_upper_bounds", upper)
        if not _require_bool(self.requires_commit, "requires_commit"):
            raise ValueError("N0 serving must require commit")
        if require_string(self.cold_seed_mode, "cold_seed_mode") != "free":
            raise ValueError("N0 cold_seed_mode must be free")
        if not _require_bool(self.no_cold_frame_skip, "no_cold_frame_skip"):
            raise ValueError("N0 row zero may not be skipped")
        object.__setattr__(
            self, "server_epoch", require_string(self.server_epoch, "server_epoch")
        )

    @classmethod
    def from_mapping(
        cls,
        document: Mapping[str, Any],
        *,
        expected_source_commit: str,
        expected_checkpoint_sha256: str,
        expected_config_sha256: str,
        expected_normalizer_sha256: str,
        expected_serve_bundle_sha256: str,
        expected_prompt_manifest_sha256: str,
    ) -> "N0Handshake":
        """Parse only an exact document and bind every artifact identity."""

        if not isinstance(document, Mapping):
            raise TypeError("N0 handshake must be a mapping")
        expected_fields = {field.name for field in fields(cls)}
        if set(document) != expected_fields:
            raise ValueError("N0 handshake fields do not match the exact contract")
        expected_hashes = {
            "source_commit": expected_source_commit,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "config_sha256": expected_config_sha256,
            "normalizer_sha256": expected_normalizer_sha256,
            "serve_bundle_sha256": expected_serve_bundle_sha256,
            "prompt_manifest_sha256": expected_prompt_manifest_sha256,
        }
        for name, expected in expected_hashes.items():
            if document[name] != expected:
                raise ValueError(f"N0 handshake {name} does not match the trial")
        return cls(
            schema_version=document["schema_version"],
            source_commit=document["source_commit"],
            checkpoint_sha256=document["checkpoint_sha256"],
            config_sha256=document["config_sha256"],
            normalizer_sha256=document["normalizer_sha256"],
            serve_bundle_sha256=document["serve_bundle_sha256"],
            action_spec=document["action_spec"],
            action_semantics=document["action_semantics"],
            action_dim=document["action_dim"],
            native_action_shape=tuple(document["native_action_shape"]),
            simulator_action_shape=tuple(document["simulator_action_shape"]),
            camera_keys=tuple(document["camera_keys"]),
            tactile_keys=tuple(document["tactile_keys"]),
            camera_shapes=tuple(document["camera_shapes"]),
            tactile_shapes=tuple(document["tactile_shapes"]),
            tactile_optional=document["tactile_optional"],
            action_lower_bounds=tuple(document["action_lower_bounds"]),
            action_upper_bounds=tuple(document["action_upper_bounds"]),
            prompt_manifest_sha256=document["prompt_manifest_sha256"],
            requires_commit=document["requires_commit"],
            cold_seed_mode=document["cold_seed_mode"],
            no_cold_frame_skip=document["no_cold_frame_skip"],
            server_epoch=document["server_epoch"],
        )


def validate_action_bounds(actions: Array, handshake: N0Handshake) -> None:
    """Reject any simulator action outside the handshake's qpos8 bounds."""

    lower = np.asarray(handshake.action_lower_bounds, dtype=np.float32)
    upper = np.asarray(handshake.action_upper_bounds, dtype=np.float32)
    if np.any(actions < lower[None, :]) or np.any(actions > upper[None, :]):
        raise ValueError("N0 action exceeds handshake bounds")

"""Validated wire capabilities shared by the N0-TWAM adapter and transport."""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from robotactile_benchmark.contracts import Array


def require_integer(value: Any, name: str, minimum: int = 0) -> int:
    """Validate an integer without silently accepting bools or floats."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def canonical_rgb(value: Array, name: str) -> Array:
    """Require the exact uint8 HWC boundary consumed by N0 preprocessing."""

    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be canonical uint8 HWC RGB")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous")
    return array


class AdapterStateError(RuntimeError):
    """Raised when infer/commit ordering would corrupt the model cache."""


class UnsupportedAvailabilityError(RuntimeError):
    """Raised when a frozen tactile-required model receives no payload."""


@dataclass(frozen=True)
class N0Handshake:
    """Versioned server capabilities required by the benchmark adapter."""

    schema_version: str
    checkpoint_sha256: str
    config_sha256: str
    action_spec: str
    action_dim: int
    camera_keys: Tuple[str, ...]
    tactile_keys: Tuple[str, ...]
    frame_chunk_size: int
    action_per_frame: int
    minimum_commit_keyframes: int
    requires_commit: bool
    action_lower_bounds: Tuple[float, ...]
    action_upper_bounds: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_keys", tuple(self.camera_keys))
        object.__setattr__(self, "tactile_keys", tuple(self.tactile_keys))
        object.__setattr__(self, "action_lower_bounds", tuple(self.action_lower_bounds))
        object.__setattr__(self, "action_upper_bounds", tuple(self.action_upper_bounds))
        for field_name in (
            "action_dim",
            "frame_chunk_size",
            "action_per_frame",
            "minimum_commit_keyframes",
        ):
            object.__setattr__(
                self,
                field_name,
                require_integer(getattr(self, field_name), field_name, minimum=1),
            )
        if self.schema_version != "robotactile-n0-v1":
            raise ValueError("unsupported N0 handshake schema version")
        sha_pattern = re.compile(r"^[0-9a-f]{64}$")
        if not sha_pattern.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be lowercase SHA256")
        if not sha_pattern.fullmatch(self.config_sha256):
            raise ValueError("config_sha256 must be lowercase SHA256")
        if self.minimum_commit_keyframes < 3:
            raise ValueError("N0 warm grounding requires at least three keyframes")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("N0 camera keys must be non-empty and unique")
        if not self.tactile_keys or len(set(self.tactile_keys)) != len(
            self.tactile_keys
        ):
            raise ValueError("N0 tactile keys must be non-empty and unique")
        if not isinstance(self.requires_commit, bool) or not self.requires_commit:
            raise ValueError("Track 3.1 N0 adapter requires stateful commit")
        if (
            len(self.action_lower_bounds) != self.action_dim
            or len(self.action_upper_bounds) != self.action_dim
        ):
            raise ValueError("N0 action bounds must match action_dim")
        lower = np.asarray(self.action_lower_bounds, dtype=np.float64)
        upper = np.asarray(self.action_upper_bounds, dtype=np.float64)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("N0 action bounds must be finite")
        if np.any(lower >= upper):
            raise ValueError("N0 action lower bounds must be below upper bounds")

    def to_dict(self) -> Dict[str, Any]:
        """Return the machine-readable handshake fields."""

        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "action_spec": self.action_spec,
            "action_dim": self.action_dim,
            "camera_keys": self.camera_keys,
            "tactile_keys": self.tactile_keys,
            "frame_chunk_size": self.frame_chunk_size,
            "action_per_frame": self.action_per_frame,
            "minimum_commit_keyframes": self.minimum_commit_keyframes,
            "requires_commit": self.requires_commit,
            "action_lower_bounds": self.action_lower_bounds,
            "action_upper_bounds": self.action_upper_bounds,
        }

    @classmethod
    def from_server_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        expected_checkpoint_sha256: str,
        expected_config_sha256: str,
    ) -> N0Handshake:
        """Construct capabilities only from an exact patched-server payload."""

        expected_fields = {
            "schema_version",
            "checkpoint_sha256",
            "config_sha256",
            "action_spec",
            "action_dim",
            "camera_keys",
            "tactile_keys",
            "frame_chunk_size",
            "action_per_frame",
            "minimum_commit_keyframes",
            "requires_commit",
            "action_lower_bounds",
            "action_upper_bounds",
        }
        if set(metadata) != expected_fields:
            raise ValueError(
                "N0 server metadata fields do not match the typed contract"
            )
        if metadata["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError("N0 server checkpoint does not match the trial manifest")
        if metadata["config_sha256"] != expected_config_sha256:
            raise ValueError("N0 server config does not match the trial manifest")
        return cls(
            schema_version=metadata["schema_version"],
            checkpoint_sha256=metadata["checkpoint_sha256"],
            config_sha256=metadata["config_sha256"],
            action_spec=metadata["action_spec"],
            action_dim=metadata["action_dim"],
            camera_keys=tuple(metadata["camera_keys"]),
            tactile_keys=tuple(metadata["tactile_keys"]),
            frame_chunk_size=metadata["frame_chunk_size"],
            action_per_frame=metadata["action_per_frame"],
            minimum_commit_keyframes=metadata["minimum_commit_keyframes"],
            requires_commit=metadata["requires_commit"],
            action_lower_bounds=tuple(metadata["action_lower_bounds"]),
            action_upper_bounds=tuple(metadata["action_upper_bounds"]),
        )

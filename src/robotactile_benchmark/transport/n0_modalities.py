"""Handshake-bound N0 infer and grounding modality validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Tuple

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.transport.n0_contracts import (
    N0GroundingFrame,
    N0Handshake,
    require_sha256,
)


def _validate_rgb_shape(
    value: object, expected: Tuple[int, int, int], name: str
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(f"{name} must be exact uint8 shape {expected}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous")


def deep_freeze_array(value: Array) -> Array:
    """Copy into bytes-backed storage whose write flag cannot be re-enabled."""

    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def validate_encoded_observation(
    observation: Mapping[str, object], handshake: N0Handshake
) -> None:
    """Validate the exact server-reconstructible infer observation domain."""

    expected_fields = {"vision", "tactile", "proprio", "observation_digest"}
    if set(observation) != expected_fields:
        raise ValueError("N0 encoded observation fields do not match the contract")
    vision = observation["vision"]
    tactile = observation["tactile"]
    if not isinstance(vision, Mapping) or set(vision) != set(handshake.camera_keys):
        raise ValueError("N0 encoded camera keys do not match the handshake")
    if not isinstance(tactile, Mapping) or set(tactile) != set(handshake.tactile_keys):
        raise ValueError("N0 encoded tactile keys do not match the handshake")
    for key, shape in zip(handshake.camera_keys, handshake.camera_shapes):
        _validate_rgb_shape(vision[key], shape, f"N0 camera {key}")
    for key, shape in zip(handshake.tactile_keys, handshake.tactile_shapes):
        _validate_rgb_shape(tactile[key], shape, f"N0 tactile {key}")
    proprio = observation["proprio"]
    if not isinstance(proprio, np.ndarray):
        raise TypeError("N0 observation proprio must be a numpy array")
    if proprio.dtype != np.float32 or proprio.shape != (8,):
        raise ValueError("N0 observation proprio must be exact float32 shape (8,)")
    if not np.isfinite(proprio).all():
        raise ValueError("N0 observation proprio must be finite")
    digest = require_sha256(observation["observation_digest"], "observation_digest")
    domain = {"vision": vision, "tactile": tactile, "proprio": proprio}
    if digest != canonical_hash(domain):
        raise ValueError("N0 observation digest does not match encoded modalities")


def require_qpos_anchor(observation: Mapping[str, object]) -> Array:
    """Return a defensive finite qpos8 snapshot from a validated observation."""

    value = observation["proprio"]
    if not isinstance(value, np.ndarray):
        raise TypeError("N0 observation proprio must be a numpy array")
    if value.dtype != np.float32 or value.shape != (8,):
        raise ValueError("N0 observation proprio must be exact float32 shape (8,)")
    if not np.isfinite(value).all():
        raise ValueError("N0 observation proprio must be finite")
    return deep_freeze_array(value)


def validate_grounding_frame_shapes(
    frames: Tuple[N0GroundingFrame, ...], handshake: N0Handshake
) -> None:
    """Bind every commit frame to handshake-frozen modality shapes and hash."""

    for frame in frames:
        _validate_rgb_shape(frame.top, handshake.camera_shapes[0], "grounding top")
        _validate_rgb_shape(
            frame.wrist_l, handshake.camera_shapes[1], "grounding wrist_l"
        )
        _validate_rgb_shape(
            frame.tactile_a, handshake.tactile_shapes[0], "grounding tactile_a"
        )
        _validate_rgb_shape(
            frame.tactile_b, handshake.tactile_shapes[1], "grounding tactile_b"
        )
        unsigned = {
            key: value
            for key, value in frame.to_wire().items()
            if key != "frame_sha256"
        }
        if canonical_hash(unsigned) != frame.frame_sha256:
            raise ValueError("N0 grounding frame digest mismatch")

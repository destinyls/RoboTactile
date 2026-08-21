"""Bit-exact N0 native-action to simulator-qpos8 conversion."""

from __future__ import annotations

import numpy as np

from robotactile_benchmark.contracts import Array

NATIVE_ACTION_SHAPE = (8, 2, 4)
SIMULATOR_ACTION_SHAPE = (8, 8)


def _require_float32(value: object, shape: tuple[int, ...], name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must use exact float32")
    if value.shape != shape:
        raise ValueError(f"{name} must have exact shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def native_to_simulator_actions(native: object) -> Array:
    """Convert [C,F,H] to frame-major, action-slot-major [F*H,C]."""

    array = _require_float32(native, NATIVE_ACTION_SHAPE, "native action")
    flat = np.transpose(array, (1, 2, 0)).reshape(SIMULATOR_ACTION_SHAPE).copy()
    flat.setflags(write=False)
    return flat


def simulator_to_native_actions(actions: object) -> Array:
    """Reverse the frozen layout without changing float32 action bits."""

    array = _require_float32(actions, SIMULATOR_ACTION_SHAPE, "simulator actions")
    native = np.transpose(array.reshape(2, 4, 8), (2, 0, 1)).copy()
    native.setflags(write=False)
    return native

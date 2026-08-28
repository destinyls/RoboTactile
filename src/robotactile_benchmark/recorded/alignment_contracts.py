"""Typed trajectory views for expert-to-live diagnostic alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from robotactile_benchmark.contracts import Array, ContactPhase, freeze_array


def _source_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("source_sha256 must be lowercase SHA256")


def _rgb(value: object, name: str) -> Array:
    image = freeze_array(value, np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be uint8 HWC RGB")
    return image


@dataclass(frozen=True)
class AlignmentFrame:
    """One source-hashed HDF5 row used by a live state-match probe."""

    source_sha256: str
    index: int
    native_step: int
    state: Array
    joint9: Array
    top_rgb: Array
    wrist_rgb: Array
    left_rgb: Array
    right_rgb: Array
    left_depth: Array
    right_depth: Array

    def __post_init__(self) -> None:
        _source_sha256(self.source_sha256)
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
            or isinstance(self.native_step, bool)
            or not isinstance(self.native_step, int)
            or self.native_step < 0
        ):
            raise ValueError(
                "frame index and native_step must be non-negative integers"
            )
        state = freeze_array(self.state, np.float32)
        joint9 = freeze_array(self.joint9, np.float32)
        if (
            state.shape != (8,)
            or joint9.shape != (9,)
            or not np.isfinite(state).all()
            or not np.isfinite(joint9).all()
        ):
            raise ValueError("alignment frame state/joint shapes are invalid")
        if not np.isclose(np.linalg.norm(state[3:7]), 1.0, atol=1e-4, rtol=0.0):
            raise ValueError("alignment frame quaternion must have unit norm")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "joint9", joint9)
        for name in ("top_rgb", "wrist_rgb", "left_rgb", "right_rgb"):
            object.__setattr__(self, name, _rgb(getattr(self, name), name))
        for name in ("left_depth", "right_depth"):
            depth = freeze_array(getattr(self, name), np.float32)
            if depth.ndim != 2 or not np.isfinite(depth).all():
                raise ValueError(f"{name} must be finite float32 HW")
            object.__setattr__(self, name, depth)

    def streams(self) -> dict[str, Array]:
        """Return checkpoint-facing stream names without changing pixels."""

        return {
            "top": self.top_rgb,
            "wrist_l": self.wrist_rgb,
            "tactile_a": self.left_rgb,
            "tactile_b": self.right_rgb,
        }

    def depths(self) -> dict[str, Array]:
        """Return physical sensor names for evaluator-only depth evidence."""

        return {"tactile_a": self.left_depth, "tactile_b": self.right_depth}


@dataclass(frozen=True)
class AlignmentTrajectory:
    """One full EE8 trajectory plus initial visual/tactile alignment witnesses."""

    trajectory_id: str
    source_sha256: str
    states: Array
    native_steps: Tuple[int, ...]
    left_phases: Tuple[ContactPhase, ...]
    right_phases: Tuple[ContactPhase, ...]
    initial_top_rgb: Array
    initial_wrist_rgb: Array
    initial_left_rgb: Array
    initial_right_rgb: Array

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        _source_sha256(self.source_sha256)
        states = freeze_array(self.states, np.float32)
        if states.ndim != 2 or states.shape[1] != 8 or states.shape[0] < 2:
            raise ValueError("alignment states must have shape [N,8] with N >= 2")
        if not np.isfinite(states).all():
            raise ValueError("alignment states must be finite")
        norms = np.linalg.norm(states[:, 3:7], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
            raise ValueError("alignment quaternions must be unit length")
        count = states.shape[0]
        native_steps = tuple(self.native_steps)
        if len(native_steps) != count or any(
            current <= previous
            for previous, current in zip(native_steps, native_steps[1:])
        ):
            raise ValueError(
                "native steps must be dense-length and strictly increasing"
            )
        left = tuple(self.left_phases)
        right = tuple(self.right_phases)
        if len(left) != count or len(right) != count:
            raise ValueError("phase traces must match the state count")
        for name in (
            "initial_top_rgb",
            "initial_wrist_rgb",
            "initial_left_rgb",
            "initial_right_rgb",
        ):
            object.__setattr__(self, name, _rgb(getattr(self, name), name))
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "native_steps", native_steps)
        object.__setattr__(self, "left_phases", left)
        object.__setattr__(self, "right_phases", right)

    @property
    def count(self) -> int:
        """Return the number of trajectory states."""

        return int(self.states.shape[0])


__all__ = ["AlignmentFrame", "AlignmentTrajectory"]

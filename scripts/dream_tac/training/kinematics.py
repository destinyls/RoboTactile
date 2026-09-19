"""Causal UniVTAC WXYZ-pose conversion for Dream-Tac Franka training."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

FloatArray: TypeAlias = npt.NDArray[np.float32]


def continuous_quaternions_wxyz(quaternions: npt.ArrayLike) -> FloatArray:
    """Normalize signs so adjacent WXYZ quaternions stay on one hemisphere."""

    values = np.asarray(quaternions)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternions must be float32 shape [N,4] in WXYZ order")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("quaternion sequence must be finite and non-empty")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("every WXYZ quaternion must have unit norm")
    result = values.copy()
    for index in range(1, result.shape[0]):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= np.float32(-1.0)
    return np.ascontiguousarray(result, dtype=np.float32)


def _quaternion_to_euler_xyz(quaternion: FloatArray) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quaternion_sequence_to_unwrapped_euler_xyz(
    quaternions: npt.ArrayLike,
) -> FloatArray:
    """Convert a continuous WXYZ episode and unwrap each XYZ Euler axis."""

    continuous = continuous_quaternions_wxyz(quaternions)
    wrapped = np.asarray(
        [_quaternion_to_euler_xyz(row) for row in continuous], dtype=np.float64
    )
    return np.ascontiguousarray(np.unwrap(wrapped, axis=0), dtype=np.float32)


def pack_dream_tac_trajectories(
    ee: npt.ArrayLike,
    joint: npt.ArrayLike,
    *,
    usable_start: int = 0,
    usable_stop: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return qpos6[t] and strict action7[t]=pose/gripper[t+1]."""

    ee_rows = np.asarray(ee)
    joint_rows = np.asarray(joint)
    if ee_rows.dtype != np.float32 or ee_rows.ndim != 2 or ee_rows.shape[1] != 7:
        raise ValueError("embodiment/ee must be float32 [N,7]")
    if (
        joint_rows.dtype != np.float32
        or joint_rows.ndim != 2
        or joint_rows.shape[1] < 8
        or joint_rows.shape[0] != ee_rows.shape[0]
    ):
        raise ValueError("embodiment/joint must be float32 [N,D>=8]")
    stop = int(ee_rows.shape[0]) if usable_stop is None else usable_stop
    if not 0 <= usable_start < stop <= ee_rows.shape[0] or stop - usable_start < 2:
        raise ValueError("usable source interval must contain at least two rows")
    selected = ee_rows[usable_start:stop]
    euler = quaternion_sequence_to_unwrapped_euler_xyz(selected[:, 3:7])
    poses = np.concatenate((selected[:, :3], euler), axis=1).astype(
        np.float32, copy=False
    )
    gripper = joint_rows[usable_start + 1 : stop, 7:8]
    qpos = np.ascontiguousarray(poses[:-1], dtype=np.float32)
    action = np.ascontiguousarray(
        np.concatenate((poses[1:], gripper), axis=1), dtype=np.float32
    )
    if not np.isfinite(qpos).all() or not np.isfinite(action).all():
        raise ValueError("Dream-Tac qpos/action contains non-finite values")
    return qpos, action


__all__ = [
    "continuous_quaternions_wxyz",
    "pack_dream_tac_trajectories",
    "quaternion_sequence_to_unwrapped_euler_xyz",
]

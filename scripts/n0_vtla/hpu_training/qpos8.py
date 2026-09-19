"""UniVTAC HDF5 to official N0-VTLA qpos8 supervision."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import numpy.typing as npt

from .contracts import SourceRecord


def pack_next_step_qpos8(
    joint: npt.ArrayLike,
    *,
    usable_start: int = 0,
    usable_stop: int | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Return state[t] and action[t]=state[t+1] using joint[:8]."""

    rows = np.asarray(joint)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] < 8
        or not np.isfinite(rows).all()
    ):
        raise ValueError("embodiment/joint must be finite float32 [N,D>=8]")
    stop = len(rows) if usable_stop is None else usable_stop
    if not 0 <= usable_start < stop <= len(rows) or stop - usable_start < 2:
        raise ValueError("usable source interval must contain at least two rows")
    qpos = np.ascontiguousarray(rows[usable_start:stop, :8], dtype=np.float32)
    return qpos[:-1], qpos[1:]


def load_episode_qpos8(
    record: SourceRecord,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read one identity-checked source episode and enforce temporal continuity."""

    h5py: Any = importlib.import_module("h5py")
    with h5py.File(record.path, "r") as root:
        if "embodiment/joint" not in root:
            raise ValueError(f"missing embodiment/joint: {record.relative_path}")
        joint = np.asarray(root["embodiment/joint"][:])
        if "step" in root:
            steps = np.asarray(root["step"][:])
            if steps.shape != (len(joint),) or not np.issubdtype(
                steps.dtype, np.integer
            ):
                raise ValueError(f"invalid step vector: {record.relative_path}")
            discontinuities = tuple(int(i) for i in np.flatnonzero(np.diff(steps) <= 0))
            expected = (227,) if record.usable_source_range == (0, 228) else ()
            if discontinuities != expected:
                raise ValueError(
                    f"unexpected step discontinuities for {record.relative_path}: "
                    f"{discontinuities}"
                )
        start, stop = record.usable_source_range or (0, len(joint))
        return pack_next_step_qpos8(joint, usable_start=start, usable_stop=stop)


def vector_stats(values: npt.ArrayLike) -> dict[str, object]:
    """Return LeRobot-compatible per-dimension summary statistics."""

    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) == 0:
        raise ValueError("qpos8 stats require non-empty [N,8] rows")
    return {
        "min": rows.min(axis=0).astype(float).tolist(),
        "max": rows.max(axis=0).astype(float).tolist(),
        "mean": rows.mean(axis=0, dtype=np.float64).astype(float).tolist(),
        "std": rows.std(axis=0, dtype=np.float64).astype(float).tolist(),
        "count": [len(rows)],
    }

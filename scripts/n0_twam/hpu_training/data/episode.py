"""UniVTAC HDF5 reading and official N0 20D absEE frame conversion."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from .contracts import LIFT_CAN_62_SHA256, SOURCE_FPS, SourceEpisode

FloatArray: TypeAlias = npt.NDArray[np.float32]
UInt8Image: TypeAlias = npt.NDArray[np.uint8]

EE_PATH: Final[str] = "embodiment/ee"
JOINT_PATH: Final[str] = "embodiment/joint"
STEP_PATH: Final[str] = "step"
IMAGE_PATHS: Final[tuple[tuple[str, str, tuple[int, int, int]], ...]] = (
    ("observation.images.top", "observation/head/rgb", (270, 480, 3)),
    ("observation.images.wrist_l", "observation/wrist/rgb", (270, 480, 3)),
    (
        "observation.images.tactile_a",
        "tactile/left_gsmini/rgb_marker",
        (240, 320, 3),
    ),
    (
        "observation.images.tactile_b",
        "tactile/right_gsmini/rgb_marker",
        (240, 320, 3),
    ),
)


@dataclass(frozen=True)
class EpisodeLayout:
    """Validated source interval used for strict next-row supervision."""

    raw_length: int
    usable_start: int
    usable_stop: int

    @property
    def converted_length(self) -> int:
        return self.usable_stop - self.usable_start - 1


def _h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise RuntimeError(
            "UniVTAC conversion requires h5py in the data-preparation runtime"
        ) from exc


def _source_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = os.lstat(path)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def verify_source_identity(record: SourceEpisode) -> tuple[int, int, int, int]:
    """Check that a source still matches the manifest's stable file identity."""

    path = record.path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source is not a regular non-symlink file: {path}")
    identity = _source_identity(path)
    if identity[2:] != (record.size_bytes, record.mtime_ns):
        raise ValueError(f"source changed after split freeze: {record.relative_path}")
    return identity


def _quat_wxyz_to_matrix(quaternion: npt.ArrayLike) -> npt.NDArray[np.float64]:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must be finite wxyz shape (4,)")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8 or not np.isclose(norm, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("UniVTAC wxyz quaternion must have unit norm")
    w, x, y, z = value / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def ee7_gripper_to_state20(ee7: npt.ArrayLike, gripper: object) -> FloatArray:
    """Encode ``xyz + quat(wxyz) + joint[7]`` as official single-arm state20."""

    ee = np.asarray(ee7)
    grip = np.asarray(gripper)
    if ee.dtype != np.float32 or ee.shape != (7,) or not np.isfinite(ee).all():
        raise ValueError("UniVTAC EE row must be finite float32 [xyz, quat(wxyz)]")
    if grip.ndim != 0 or not np.isfinite(grip):
        raise ValueError("UniVTAC gripper must be a finite scalar")
    rotation = _quat_wxyz_to_matrix(ee[3:7])
    state: FloatArray = np.zeros(20, dtype=np.float32)
    state[:3] = ee[:3]
    # Official N0 convention: first rotation-matrix column xyz, then the second.
    state[3:9] = rotation[:, :2].T.reshape(-1)
    state[9] = np.float32(grip)
    return state


def pack_next_step_trajectories(
    ee: npt.ArrayLike,
    joint: npt.ArrayLike,
    *,
    usable_start: int = 0,
    usable_stop: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return state[t] and strict action[t]=state[t+1] in 20D absEE form."""

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
    encoded = np.stack(
        [
            ee7_gripper_to_state20(ee_rows[index], joint_rows[index, 7])
            for index in range(usable_start, stop)
        ]
    ).astype(np.float32, copy=False)
    return (
        np.ascontiguousarray(encoded[:-1]),
        np.ascontiguousarray(encoded[1:]),
    )


def decode_legacy_rgb(payload: object) -> UInt8Image:
    """Decode the UniVTAC writer's OpenCV-encoded RGB byte convention."""

    if isinstance(payload, np.ndarray) and payload.ndim == 3:
        image = payload
    else:
        if isinstance(payload, np.ndarray) and payload.ndim == 1:
            compressed: UInt8Image = payload.astype(np.uint8, copy=False)
        elif isinstance(payload, (bytes, bytearray, np.bytes_)):
            compressed = np.frombuffer(bytes(payload), dtype=np.uint8)
        else:
            raise TypeError(f"unsupported image payload: {type(payload)!r}")
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "compressed UniVTAC images require opencv-python"
            ) from exc
        image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not decode a UniVTAC image")
        # UniVTAC passed simulator RGB directly to cv2.imencode.  The numeric
        # OpenCV decode output therefore restores the original RGB ordering.
    result = np.asarray(image)
    if result.dtype != np.uint8 or result.ndim != 3 or result.shape[-1] != 3:
        raise ValueError("decoded UniVTAC image must be uint8 HWC RGB")
    return cast(UInt8Image, np.ascontiguousarray(result, dtype=np.uint8))


def _steps(root: Mapping[str, Any], length: int) -> npt.NDArray[np.int64]:
    if STEP_PATH not in root:
        return np.arange(length, dtype=np.int64)
    raw = np.asarray(root[STEP_PATH][:])
    if raw.shape != (length,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"step must be an integer vector with length {length}")
    return raw.astype(np.int64, copy=False)


def inspect_hdf5_root(root: Mapping[str, Any], record: SourceEpisode) -> EpisodeLayout:
    """Validate one open HDF5 root and resolve the sole temporal exception."""

    required = (EE_PATH, JOINT_PATH) + tuple(path for _, path, _ in IMAGE_PATHS)
    missing = tuple(path for path in required if path not in root)
    if missing:
        raise ValueError(f"missing UniVTAC HDF5 datasets: {missing}")
    ee = root[EE_PATH]
    joint = root[JOINT_PATH]
    if ee.dtype != np.dtype(np.float32) or len(ee.shape) != 2 or ee.shape[1] != 7:
        raise ValueError("embodiment/ee must be float32 [N,7]")
    length = int(ee.shape[0])
    if (
        joint.dtype != np.dtype(np.float32)
        or len(joint.shape) != 2
        or joint.shape[0] != length
        or joint.shape[1] < 8
    ):
        raise ValueError("embodiment/joint must be float32 [N,D>=8]")
    for _, path, expected_shape in IMAGE_PATHS:
        dataset = root[path]
        if int(dataset.shape[0]) != length:
            raise ValueError(f"image row count does not match EE rows: {path}")
        first = decode_legacy_rgb(dataset[0])
        if first.shape != expected_shape:
            raise ValueError(f"image shape mismatch for {path}: {first.shape}")
    steps = _steps(root, length)
    discontinuities = tuple(int(index) for index in np.flatnonzero(np.diff(steps) <= 0))
    if record.task == "lift_can" and record.episode_id == 62:
        if (
            record.sha256 != LIFT_CAN_62_SHA256
            or record.relative_path != "lift_can/clean/62.hdf5"
            or length != 256
            or discontinuities != (227,)
            or int(steps[227]) != 594
            or int(steps[228]) != 586
        ):
            raise ValueError("lift_can/62 does not match its pinned prefix contract")
        return EpisodeLayout(raw_length=length, usable_start=0, usable_stop=228)
    if discontinuities:
        raise ValueError(
            f"step must be strictly increasing for {record.relative_path}: "
            f"{discontinuities}"
        )
    if length < 2:
        raise ValueError("source episode must contain at least two rows")
    return EpisodeLayout(raw_length=length, usable_start=0, usable_stop=length)


def iter_episode_frames(
    record: SourceEpisode,
) -> Iterator[tuple[dict[str, object], float]]:
    """Yield official LeRobot rows while preserving source cadence and pixels."""

    opened_identity = verify_source_identity(record)
    h5py = _h5py()
    with h5py.File(record.path, "r") as root:
        layout = inspect_hdf5_root(root, record)
        state, action = pack_next_step_trajectories(
            np.asarray(root[EE_PATH][:]),
            np.asarray(root[JOINT_PATH][:]),
            usable_start=layout.usable_start,
            usable_stop=layout.usable_stop,
        )
        for local_index, source_index in enumerate(
            range(layout.usable_start, layout.usable_stop - 1)
        ):
            frame: dict[str, object] = {
                "observation.state": state[local_index],
                "action": action[local_index],
            }
            for feature_name, path, expected_shape in IMAGE_PATHS:
                image = decode_legacy_rgb(root[path][source_index])
                if image.shape != expected_shape:
                    raise ValueError(
                        f"image shape changed at {record.relative_path}:{source_index}"
                    )
                frame[feature_name] = image
            yield frame, local_index / float(SOURCE_FPS)
    if _source_identity(record.path) != opened_identity:
        raise ValueError(f"source changed during conversion: {record.relative_path}")


__all__ = [
    "EE_PATH",
    "IMAGE_PATHS",
    "JOINT_PATH",
    "EpisodeLayout",
    "decode_legacy_rgb",
    "ee7_gripper_to_state20",
    "inspect_hdf5_root",
    "iter_episode_frames",
    "pack_next_step_trajectories",
    "verify_source_identity",
]

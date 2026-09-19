"""Strict HDF5/MP4 export for the official Dream-Tac Franka loader."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ContextManager, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from scripts.n0_twam.hpu_training.data.contracts import SourceEpisode, sha256_file
from scripts.n0_twam.hpu_training.data.episode import (
    EE_PATH,
    JOINT_PATH,
    decode_legacy_rgb,
    inspect_hdf5_root,
    verify_source_identity,
)

from .contracts import SOURCE_FPS, VIDEO_SOURCES
from .kinematics import FloatArray, pack_dream_tac_trajectories

UInt8Image: TypeAlias = npt.NDArray[np.uint8]
VideoWriter: TypeAlias = Callable[[Iterable[UInt8Image], Path, int], int]
Hdf5Writer: TypeAlias = Callable[
    [Path, FloatArray, FloatArray, Mapping[str, str], str, str], None
]
RootOpener: TypeAlias = Callable[[Path], ContextManager[Mapping[str, Any]]]


def _h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise RuntimeError("Dream-Tac materialization requires h5py") from exc


@contextmanager
def open_hdf5_root(path: Path) -> Iterator[Mapping[str, Any]]:
    handle = _h5py().File(path, "r")
    try:
        yield cast(Mapping[str, Any], handle)
    finally:
        handle.close()


def write_mp4(frames: Iterable[UInt8Image], path: Path, fps: int) -> int:
    """Write one RGB stream with OpenCV's upstream-compatible MP4 path."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite video: {path}")
    if fps != SOURCE_FPS:
        raise ValueError(f"Dream-Tac videos must use {SOURCE_FPS} Hz")
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError("Dream-Tac MP4 export requires opencv-python") from exc
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Dream-Tac video cannot be empty") from exc
    first_value = _checked_rgb(first, expected_shape=None)
    height, width, _ = first_value.shape
    frame_shape = (int(height), int(width), 3)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV could not open Dream-Tac video: {path}")
    count = 0
    try:
        for frame in itertools.chain((first_value,), iterator):
            value = _checked_rgb(frame, expected_shape=frame_shape)
            writer.write(cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        writer.release()
    if count <= 0 or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Dream-Tac video export failed: {path}")
    return count


def _checked_rgb(
    frame: object, *, expected_shape: tuple[int, int, int] | None
) -> UInt8Image:
    value = np.asarray(frame)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Dream-Tac video frames must be uint8 HWC RGB")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError("Dream-Tac video frame shape changed within a stream")
    return cast(UInt8Image, np.ascontiguousarray(value, dtype=np.uint8))


def write_official_hdf5(
    path: Path,
    qpos: FloatArray,
    action: FloatArray,
    video_paths: Mapping[str, str],
    prompt: str,
    task_id: str,
) -> None:
    """Write the exact fields consumed by upstream ``FrankaDataset``."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite HDF5: {path}")
    if qpos.dtype != np.float32 or qpos.ndim != 2 or qpos.shape[1] != 6:
        raise ValueError("Dream-Tac qpos must be float32 [T,6]")
    if action.dtype != np.float32 or action.shape != (qpos.shape[0], 7):
        raise ValueError("Dream-Tac action must be float32 [T,7]")
    if set(video_paths) != {name for name, _ in VIDEO_SOURCES}:
        raise ValueError("Dream-Tac HDF5 requires all four video paths")
    relative_action = np.zeros_like(action)
    if action.shape[0] > 1:
        relative_action[:-1] = action[1:] - action[:-1]
        relative_action[-1] = relative_action[-2]
    h5py = _h5py()
    with h5py.File(path, "x", rdcc_nbytes=2 * 1024**2) as root:
        root.attrs["task_name"] = prompt
        root.attrs["robotactile_task_id"] = task_id
        root.attrs["robotactile_source_fps"] = SOURCE_FPS
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos, dtype=np.float32)
        paths = observations.create_group("video_paths")
        for stream_name, _ in VIDEO_SOURCES:
            paths.create_dataset(stream_name, data=video_paths[stream_name].encode())
        root.create_dataset("action", data=action, dtype=np.float32)
        root.create_dataset("relative_action", data=relative_action, dtype=np.float32)


def verify_official_hdf5(
    path: Path, *, prompt: str, task_id: str, converted_length: int
) -> None:
    """Verify the upstream loader fields without trusting a conversion receipt."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Dream-Tac HDF5 is not a regular file: {path}")
    with _h5py().File(path, "r") as root:
        if root.attrs.get("task_name") != prompt:
            raise ValueError("Dream-Tac HDF5 task_name does not match the T5 key")
        if root.attrs.get("robotactile_task_id") != task_id:
            raise ValueError("Dream-Tac HDF5 task identity mismatch")
        qpos = root["observations/qpos"]
        action = root["action"]
        if qpos.dtype != np.dtype(np.float32) or qpos.shape != (converted_length, 6):
            raise ValueError("Dream-Tac HDF5 qpos contract mismatch")
        if action.dtype != np.dtype(np.float32) or action.shape != (
            converted_length,
            7,
        ):
            raise ValueError("Dream-Tac HDF5 action contract mismatch")
        paths = root["observations/video_paths"]
        if set(paths.keys()) != {name for name, _ in VIDEO_SOURCES}:
            raise ValueError("Dream-Tac HDF5 four-stream path contract mismatch")


def read_dream_tac_trajectories(
    record: SourceEpisode, *, root_opener: RootOpener = open_hdf5_root
) -> tuple[FloatArray, FloatArray]:
    """Read only the source-bound qpos/action trajectories for statistics."""

    opened_identity = verify_source_identity(record)
    with root_opener(record.path) as root:
        layout = inspect_hdf5_root(root, record)
        qpos, action = pack_dream_tac_trajectories(
            np.asarray(root[EE_PATH][:]),
            np.asarray(root[JOINT_PATH][:]),
            usable_start=layout.usable_start,
            usable_stop=layout.usable_stop,
        )
    if verify_source_identity(record) != opened_identity:
        raise ValueError(f"source changed during conversion: {record.relative_path}")
    return qpos, action


def export_episode(
    record: SourceEpisode,
    *,
    staging_root: Path,
    prompt: str,
    video_writer: VideoWriter = write_mp4,
    hdf5_writer: Hdf5Writer = write_official_hdf5,
    root_opener: RootOpener = open_hdf5_root,
) -> dict[str, object]:
    """Export one T-1 episode while the immutable source is held open."""

    opened_identity = verify_source_identity(record)
    staging_root.mkdir(parents=True, exist_ok=False)
    prefix = f"episode_{record.episode_id:03d}"
    video_names = {
        stream_name: f"{prefix}_{stream_name}.mp4" for stream_name, _ in VIDEO_SOURCES
    }
    frame_counts: dict[str, int] = {}
    with root_opener(record.path) as root:
        layout = inspect_hdf5_root(root, record)
        qpos, action = pack_dream_tac_trajectories(
            np.asarray(root[EE_PATH][:]),
            np.asarray(root[JOINT_PATH][:]),
            usable_start=layout.usable_start,
            usable_stop=layout.usable_stop,
        )
        for stream_name, source_path in VIDEO_SOURCES:
            indices = range(layout.usable_start, layout.usable_stop - 1)
            frames = (decode_legacy_rgb(root[source_path][index]) for index in indices)
            frame_counts[stream_name] = video_writer(
                frames, staging_root / video_names[stream_name], SOURCE_FPS
            )
        if set(frame_counts.values()) != {layout.converted_length}:
            raise ValueError("Dream-Tac video streams are not aligned to T-1")
        hdf5_name = f"{prefix}.hdf5"
        hdf5_writer(
            staging_root / hdf5_name,
            qpos,
            action,
            video_names,
            prompt,
            record.task,
        )
    if verify_source_identity(record) != opened_identity:
        raise ValueError(f"source changed during conversion: {record.relative_path}")
    files: list[dict[str, object]] = [
        {
            "path": hdf5_name,
            "sha256": sha256_file(staging_root / hdf5_name),
            "kind": "hdf5",
        }
    ]
    files.extend(
        {
            "path": video_names[name],
            "sha256": sha256_file(staging_root / video_names[name]),
            "kind": "video",
            "stream": name,
            "frame_count": frame_counts[name],
        }
        for name, _ in VIDEO_SOURCES
    )
    return {
        "raw_frame_count": layout.raw_length,
        "converted_frame_count": layout.converted_length,
        "usable_source_range": [layout.usable_start, layout.usable_stop],
        "files": files,
    }


__all__ = [
    "Hdf5Writer",
    "RootOpener",
    "VideoWriter",
    "export_episode",
    "open_hdf5_root",
    "read_dream_tac_trajectories",
    "verify_official_hdf5",
    "write_mp4",
    "write_official_hdf5",
]

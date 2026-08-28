"""Strict readers and atomic writer for expert alignment diagnostics."""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Tuple, cast

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.adapters.univtac import ContactPhaseState
from robotactile_benchmark.backends.univtac_registry import build_config
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import Array, ContactPhase
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
)
from robotactile_benchmark.recorded.alignment_contracts import (
    AlignmentFrame,
    AlignmentTrajectory,
)

_HDF5_FIELDS = (
    "step",
    "embodiment/ee",
    "embodiment/joint",
    "observation/head/rgb",
    "observation/wrist/rgb",
    "tactile/left_gsmini/rgb",
    "tactile/left_gsmini/depth",
    "tactile/right_gsmini/rgb",
    "tactile/right_gsmini/depth",
)


class ExpertAlignmentIOError(ValueError):
    """An alignment input or output violates its strict contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as error:
        raise ExpertAlignmentIOError("expert alignment requires h5py") from error


def _decode_rgb(value: object, name: str) -> Array:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as error:
        raise ExpertAlignmentIOError("expert alignment requires Pillow") from error
    raw = bytes(value) if isinstance(value, (bytes, np.bytes_)) else None
    if raw is None:
        raise ExpertAlignmentIOError(f"{name} must contain encoded image bytes")
    try:
        with image_module.open(io.BytesIO(raw)) as image:
            return cast(Array, np.asarray(image.convert("RGB"), dtype=np.uint8))
    except (OSError, ValueError) as error:
        raise ExpertAlignmentIOError(f"cannot decode {name}") from error


def _phase_trace(depth: Any, task_id: str) -> Tuple[ContactPhase, ...]:
    config = build_config(task_id, action_spec=EE8_ACTION_SPEC)
    state = ContactPhaseState(False)
    phases = []
    for start in range(0, int(depth.shape[0]), 32):
        chunk = np.asarray(depth[start : start + 32], dtype=np.float32)
        if chunk.ndim != 3 or not np.isfinite(chunk).all():
            raise ExpertAlignmentIOError("expert tactile depth must be finite [N,H,W]")
        indentations = np.maximum(
            0.0,
            config.aliases.far_plane_mm - np.min(chunk, axis=(1, 2)),
        )
        for indentation in indentations:
            phase, state = config.phase_tracker.update(float(indentation), state)
            phases.append(phase)
    return tuple(phases)


def load_expert_hdf5_trajectory(path: Path, task_id: str) -> AlignmentTrajectory:
    """Load a complete expert HDF5 trajectory without decoding every image."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ExpertAlignmentIOError("expert HDF5 must be a non-symlink regular file")
    before = source.stat()
    h5py = _h5py()
    with h5py.File(source, "r") as root:
        try:
            datasets = {name: root[name] for name in _HDF5_FIELDS}
        except KeyError as error:
            raise ExpertAlignmentIOError(
                "expert HDF5 is missing a required field"
            ) from error
        counts = {int(value.shape[0]) for value in datasets.values()}
        if len(counts) != 1:
            raise ExpertAlignmentIOError("expert HDF5 fields have inconsistent lengths")
        count = counts.pop()
        ee = np.asarray(datasets["embodiment/ee"], dtype=np.float32)
        joint = np.asarray(datasets["embodiment/joint"], dtype=np.float32)
        steps = np.asarray(datasets["step"], dtype=np.int64)
        if ee.shape != (count, 7) or joint.shape != (count, 9):
            raise ExpertAlignmentIOError("expert embodiment shapes are invalid")
        if count < 2 or steps.shape != (count,) or not np.all(np.diff(steps) > 0):
            raise ExpertAlignmentIOError("expert steps must be strictly increasing")
        states = np.concatenate((ee, joint[:, -2:-1]), axis=1).astype(
            np.float32, copy=False
        )
        left_phases = _phase_trace(datasets["tactile/left_gsmini/depth"], task_id)
        right_phases = _phase_trace(datasets["tactile/right_gsmini/depth"], task_id)
        images = {
            name: _decode_rgb(datasets[field][0], name)
            for name, field in (
                ("top", "observation/head/rgb"),
                ("wrist", "observation/wrist/rgb"),
                ("left", "tactile/left_gsmini/rgb"),
                ("right", "tactile/right_gsmini/rgb"),
            )
        }
    source_sha256 = _sha256_file(source)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExpertAlignmentIOError("expert HDF5 changed while being read")
    return AlignmentTrajectory(
        trajectory_id=f"expert:{source.name}",
        source_sha256=source_sha256,
        states=states,
        native_steps=tuple(int(value) for value in steps),
        left_phases=left_phases,
        right_phases=right_phases,
        initial_top_rgb=images["top"],
        initial_wrist_rgb=images["wrist"],
        initial_left_rgb=images["left"],
        initial_right_rgb=images["right"],
    )


def load_expert_hdf5_joint9_trajectory(path: Path) -> tuple[Array, str]:
    """Load all canonical joint rows for official sequential qpos replay."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ExpertAlignmentIOError("expert HDF5 must be a non-symlink regular file")
    before = source.stat()
    h5py = _h5py()
    with h5py.File(source, "r") as root:
        try:
            joint9 = np.asarray(root["embodiment/joint"], dtype=np.float32)
        except KeyError as error:
            raise ExpertAlignmentIOError(
                "expert HDF5 is missing embodiment/joint"
            ) from error
    if joint9.ndim != 2 or joint9.shape[1] != 9 or not np.isfinite(joint9).all():
        raise ExpertAlignmentIOError("expert joints must be finite float32 [N,9]")
    source_sha256 = _sha256_file(source)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExpertAlignmentIOError("expert HDF5 changed while being read")
    result = np.ascontiguousarray(joint9, dtype=np.float32)
    result.setflags(write=False)
    return result, source_sha256


def load_expert_hdf5_actor_poses(
    path: Path,
    actor_names: tuple[str, ...],
    *,
    index: int = 0,
) -> tuple[dict[str, Array], str]:
    """Load source-hashed actor poses for an explicit simulator alignment probe."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ExpertAlignmentIOError("expert HDF5 must be a non-symlink regular file")
    if (
        not actor_names
        or len(actor_names) != len(set(actor_names))
        or any(not name or "/" in name for name in actor_names)
    ):
        raise ExpertAlignmentIOError("actor names must be unique HDF5 path components")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ExpertAlignmentIOError("expert actor pose index must be non-negative")
    before = source.stat()
    h5py = _h5py()
    poses: dict[str, Array] = {}
    with h5py.File(source, "r") as root:
        for name in actor_names:
            try:
                dataset = root[f"actor/{name}"]
            except KeyError as error:
                raise ExpertAlignmentIOError(
                    f"expert HDF5 is missing actor/{name}"
                ) from error
            if dataset.ndim != 2 or dataset.shape[1] != 7 or index >= dataset.shape[0]:
                raise ExpertAlignmentIOError(f"actor/{name} pose shape is invalid")
            pose = np.asarray(dataset[index], dtype=np.float32)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ExpertAlignmentIOError(f"actor/{name} pose must be finite [7]")
            if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4, rtol=0.0):
                raise ExpertAlignmentIOError(
                    f"actor/{name} quaternion must have unit norm"
                )
            frozen = np.ascontiguousarray(pose, dtype=np.float32)
            frozen.setflags(write=False)
            poses[name] = frozen
    source_sha256 = _sha256_file(source)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExpertAlignmentIOError("expert HDF5 changed while being read")
    return poses, source_sha256


def load_expert_hdf5_frame(path: Path, index: int) -> AlignmentFrame:
    """Load one exact HDF5 row for a state-matched visual comparison."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise ExpertAlignmentIOError("expert HDF5 must be a non-symlink regular file")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ExpertAlignmentIOError("expert frame index must be non-negative")
    before = source.stat()
    h5py = _h5py()
    with h5py.File(source, "r") as root:
        try:
            datasets = {name: root[name] for name in _HDF5_FIELDS}
        except KeyError as error:
            raise ExpertAlignmentIOError(
                "expert HDF5 is missing a required field"
            ) from error
        counts = {int(value.shape[0]) for value in datasets.values()}
        if len(counts) != 1 or index >= counts.pop():
            raise ExpertAlignmentIOError("expert frame index is outside the episode")
        ee = np.asarray(datasets["embodiment/ee"][index], dtype=np.float32)
        joint = np.asarray(datasets["embodiment/joint"][index], dtype=np.float32)
        if ee.shape != (7,) or joint.shape != (9,):
            raise ExpertAlignmentIOError("expert frame embodiment shapes are invalid")
        state = np.concatenate((ee, joint[-2:-1])).astype(np.float32, copy=False)
        images = {
            name: _decode_rgb(datasets[field][index], name)
            for name, field in (
                ("top", "observation/head/rgb"),
                ("wrist", "observation/wrist/rgb"),
                ("left", "tactile/left_gsmini/rgb"),
                ("right", "tactile/right_gsmini/rgb"),
            )
        }
        depths = {
            "left": np.asarray(
                datasets["tactile/left_gsmini/depth"][index], dtype=np.float32
            ),
            "right": np.asarray(
                datasets["tactile/right_gsmini/depth"][index], dtype=np.float32
            ),
        }
        native_step = int(datasets["step"][index])
    source_sha256 = _sha256_file(source)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExpertAlignmentIOError("expert HDF5 changed while being read")
    return AlignmentFrame(
        source_sha256=source_sha256,
        index=index,
        native_step=native_step,
        state=state,
        joint9=joint,
        top_rgb=images["top"],
        wrist_rgb=images["wrist"],
        left_rgb=images["left"],
        right_rgb=images["right"],
        left_depth=depths["left"],
        right_depth=depths["right"],
    )


def load_live_alignment_trajectory(path: Path) -> tuple[AlignmentTrajectory, int]:
    """Load a verified live artifact as a trajectory plus executed action count."""

    loaded = load_live_univtac_artifact(path)
    finalization = loaded.evidence.finalization
    if finalization is None or len(finalization.clean_records) < 2:
        raise ExpertAlignmentIOError("live artifact lacks a complete clean trajectory")
    records = finalization.clean_records
    initial = records[0].observation
    left = initial.sensor("left").payload
    right = initial.sensor("right").payload
    if left is None or right is None:
        raise ExpertAlignmentIOError("live initial tactile payloads are absent")
    try:
        top = initial.vision["top"]
        wrist = initial.vision["wrist_l"]
    except KeyError as error:
        raise ExpertAlignmentIOError(
            "live initial camera payloads are absent"
        ) from error
    trajectory = AlignmentTrajectory(
        trajectory_id=f"live:{loaded.trial.sha256[:16]}",
        source_sha256=loaded.root_receipt_sha256,
        states=np.stack([item.observation.proprio for item in records]).astype(
            np.float32, copy=False
        ),
        native_steps=tuple(item.observation.step_index for item in records),
        left_phases=tuple(item.provenance_for("left").phase for item in records),
        right_phases=tuple(item.provenance_for("right").phase for item in records),
        initial_top_rgb=top,
        initial_wrist_rgb=wrist,
        initial_left_rgb=left,
        initial_right_rgb=right,
    )
    executed = sum(
        int(entry.executed_actions.shape[0]) for entry in loaded.evidence.action_entries
    )
    return trajectory, executed


def write_alignment_report(path: Path, report: object) -> str:
    """Atomically create an immutable canonical JSON report."""

    output = Path(path).absolute()
    payload = canonical_json_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ExpertAlignmentIOError("alignment output cannot be a symlink")
    if output.exists():
        if not output.is_file() or output.read_bytes() != payload:
            raise FileExistsError("alignment output already differs")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


__all__ = [
    "ExpertAlignmentIOError",
    "load_expert_hdf5_actor_poses",
    "load_expert_hdf5_frame",
    "load_expert_hdf5_joint9_trajectory",
    "load_expert_hdf5_trajectory",
    "load_live_alignment_trajectory",
    "write_alignment_report",
]

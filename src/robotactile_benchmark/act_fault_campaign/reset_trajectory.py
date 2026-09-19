"""Strict persistence for source-bound ACT pre-move trajectories."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.backends.univtac_reset_trajectory import (
    UniVTACPreMoveTrajectory,
    UniVTACPreMoveTrajectoryError,
)
from robotactile_benchmark.backends.univtac_reset_witness import (
    UniVTACResetReference,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)

MAX_ACT_RESET_TRAJECTORY_BYTES = 16 * 1024 * 1024

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ACTResetTrajectoryArtifactError(ValueError):
    """An ACT reset-trajectory path or JSON artifact failed validation."""


def build_act_trajectory_replay_reference(
    source_reference: UniVTACResetReference,
    trajectory: UniVTACPreMoveTrajectory,
) -> UniVTACResetReference:
    """Anchor reset qualification to the exact captured trajectory endpoint.

    The successful Clean-derived reference remains the immutable provenance
    source.  Replay changes only the expected live reset endpoint and tightens
    the joint tolerance; all task, seed, data, model, and run identities remain
    those of the successful Clean artifact.
    """

    if type(source_reference) is not UniVTACResetReference:
        raise TypeError("source_reference must be an exact UniVTACResetReference")
    if type(trajectory) is not UniVTACPreMoveTrajectory:
        raise TypeError("trajectory must be an exact UniVTACPreMoveTrajectory")
    identity_matches = (
        trajectory.task_id == source_reference.task_id
        and trajectory.initial_seed == source_reference.initial_seed
        and trajectory.exogenous_seed == source_reference.exogenous_seed
        and trajectory.pair_key == source_reference.pair_key
        and trajectory.dataset_sha256 == source_reference.dataset_sha256
        and trajectory.checkpoint_sha256 == source_reference.checkpoint_sha256
        and trajectory.config_sha256 == source_reference.config_sha256
        and trajectory.source_run_content_sha256
        == source_reference.source_run_content_sha256
    )
    if (
        trajectory.reset_reference_sha256 != source_reference.sha256
        or not identity_matches
    ):
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory identity differs from source reference"
        )
    return replace(
        source_reference,
        expected_simulator_state_sha256=(trajectory.capture_simulator_state_sha256),
        expected_native_step=trajectory.capture_native_step,
        expected_qpos8=trajectory.capture_qpos8,
        qpos_atol=1e-5,
    )


def act_reset_trajectory_relpath(task: str, pair_key: str) -> str:
    """Return the canonical campaign-relative path for one task/seed pair."""

    if not isinstance(task, str) or _SAFE_IDENTIFIER.fullmatch(task) is None:
        raise ACTResetTrajectoryArtifactError("task must be a safe identifier")
    if not isinstance(pair_key, str) or _SHA256.fullmatch(pair_key) is None:
        raise ACTResetTrajectoryArtifactError("pair_key must be a lowercase SHA256")
    return f"reset_trajectories/{task}/{pair_key}.json"


def write_act_reset_trajectory(
    path: Path,
    trajectory: UniVTACPreMoveTrajectory,
) -> bool:
    """Atomically publish canonical JSON without replacing existing bytes."""

    if type(trajectory) is not UniVTACPreMoveTrajectory:
        raise TypeError("trajectory must be an exact UniVTACPreMoveTrajectory")
    target = Path(path).expanduser().absolute()
    try:
        payload = canonical_json_bytes(trajectory.to_dict())
    except (TypeError, ValueError) as error:
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory is outside canonical JSON"
        ) from error
    if not 1 <= len(payload) <= MAX_ACT_RESET_TRAJECTORY_BYTES:
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory size is outside bounds"
        )
    _reject_symlink_components(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    if target.exists() or target.is_symlink():
        return _verify_existing(target, payload)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return _verify_existing(target, payload)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_act_reset_trajectory(path: Path) -> UniVTACPreMoveTrajectory:
    """Load one stable canonical trajectory through a non-symlink path."""

    source = Path(path).expanduser().absolute()
    _reject_symlink_components(source)
    if not source.is_file():
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory must be a regular non-symlink file"
        )
    try:
        before = source.stat()
        if not 1 <= before.st_size <= MAX_ACT_RESET_TRAJECTORY_BYTES:
            raise ACTResetTrajectoryArtifactError(
                "ACT reset trajectory size is outside bounds"
            )
        raw = source.read_bytes()
        after = source.stat()
    except OSError as error:
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory cannot be read"
        ) from error
    if (
        len(raw) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory changed while being read"
        )
    try:
        document = strict_json_bytes(raw, "ACT reset trajectory")
        if not isinstance(document, Mapping):
            raise ACTResetTrajectoryArtifactError(
                "ACT reset trajectory must contain a JSON object"
            )
        trajectory = UniVTACPreMoveTrajectory.from_dict(document)
    except (TypeError, ValueError, UniVTACPreMoveTrajectoryError) as error:
        if isinstance(error, ACTResetTrajectoryArtifactError):
            raise
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory is not strict canonical JSON"
        ) from error
    if canonical_json_bytes(trajectory.to_dict()) != raw:
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory does not round-trip canonically"
        )
    return trajectory


def _verify_existing(path: Path, payload: bytes) -> bool:
    _reject_symlink_components(path)
    if not path.is_file():
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory output is not a regular file"
        )
    try:
        current = path.read_bytes()
    except OSError as error:
        raise ACTResetTrajectoryArtifactError(
            "ACT reset trajectory output cannot be read"
        ) from error
    if current != payload:
        raise FileExistsError("refusing to replace a different ACT reset trajectory")
    return False


def _reject_symlink_components(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ACTResetTrajectoryArtifactError(
                "ACT reset trajectory path cannot traverse a symlink"
            )


__all__ = [
    "ACTResetTrajectoryArtifactError",
    "act_reset_trajectory_relpath",
    "build_act_trajectory_replay_reference",
    "load_act_reset_trajectory",
    "write_act_reset_trajectory",
]

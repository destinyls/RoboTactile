"""Capture and replay source-bound UniVTAC ``pre_move`` planner segments."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Tuple, cast

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.contracts import canonical_hash

UNIVTAC_PRE_MOVE_TRAJECTORY_SCHEMA = "univtac-pre-move-trajectory-v3"
UNIVTAC_PRE_MOVE_TRAJECTORY_SEMANTIC_VERSION = "3.0"

_EXPECTED_KINDS = ("gripper", "arm", "gripper", "arm")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
PRE_MOVE_CAPTURE_MODE = "reset_only_source_proximity_guided_trajectory_v3"


class UniVTACPreMoveTrajectoryError(ValueError):
    """A pre-move trajectory or its live consumption violated the contract."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise UniVTACPreMoveTrajectoryError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise UniVTACPreMoveTrajectoryError(f"{name} must be a lowercase SHA256")
    return value


def _seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise UniVTACPreMoveTrajectoryError(f"{name} must be non-negative integer")
    return int(value)


def _hold_steps(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        message = "post_reset_hold_steps must be an integer between 0 and 2"
        raise UniVTACPreMoveTrajectoryError(message)
    result = int(value)
    if result < 0 or result > 2:
        raise UniVTACPreMoveTrajectoryError(
            "post_reset_hold_steps must be between 0 and 2"
        )
    return result


def _source_qpos8_error(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        message = "source_qpos8_max_abs_error must be a real number"
        raise UniVTACPreMoveTrajectoryError(message)
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 0.002:
        message = "source_qpos8_max_abs_error must be finite and between 0 and 0.002"
        raise UniVTACPreMoveTrajectoryError(message)
    return result


def _vector(value: object, name: str, width: int) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UniVTACPreMoveTrajectoryError(f"{name} must be a sequence")
    if len(value) != width:
        raise UniVTACPreMoveTrajectoryError(f"{name} must have width {width}")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise UniVTACPreMoveTrajectoryError(f"{name} must contain real numbers")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise UniVTACPreMoveTrajectoryError(f"{name} must be finite")
        result.append(normalized)
    return tuple(result)


def _matrix(value: object, name: str, width: int) -> Tuple[Tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise UniVTACPreMoveTrajectoryError(f"{name} must be a sequence")
    result = tuple(
        _vector(row, f"{name}[{index}]", width) for index, row in enumerate(value)
    )
    if not result:
        raise UniVTACPreMoveTrajectoryError(f"{name} must contain at least one row")
    return result


@dataclass(frozen=True)
class UniVTACPreMoveSegment:
    """One dense gripper or arm plan executed by upstream ``move``."""

    kind: str
    request_fingerprint: str
    start_joint9: Tuple[float, ...]
    position: Tuple[Tuple[float, ...], ...]
    velocity: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.kind not in {"gripper", "arm"}:
            raise UniVTACPreMoveTrajectoryError("segment kind must be gripper or arm")
        fingerprint = _sha256(self.request_fingerprint, "request_fingerprint")
        object.__setattr__(self, "request_fingerprint", fingerprint)
        start_joint9 = _vector(self.start_joint9, "start_joint9", 9)
        object.__setattr__(self, "start_joint9", start_joint9)
        width = 1 if self.kind == "gripper" else 7
        object.__setattr__(self, "position", _matrix(self.position, "position", width))
        object.__setattr__(self, "velocity", _matrix(self.velocity, "velocity", width))
        if len(self.position) != len(self.velocity):
            raise UniVTACPreMoveTrajectoryError(
                "segment position/velocity lengths differ"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "request_fingerprint": self.request_fingerprint,
            "start_joint9": list(self.start_joint9),
            "position": [list(row) for row in self.position],
            "velocity": [list(row) for row in self.velocity],
        }

    @classmethod
    def from_dict(cls, value: object) -> UniVTACPreMoveSegment:
        if not isinstance(value, Mapping):
            raise UniVTACPreMoveTrajectoryError("segment must be a mapping")
        if set(value) != set(cls.__dataclass_fields__):
            raise UniVTACPreMoveTrajectoryError("segment fields mismatch")
        return cls(
            kind=cast(Any, value["kind"]),
            request_fingerprint=cast(Any, value["request_fingerprint"]),
            start_joint9=cast(Any, value["start_joint9"]),
            position=cast(Any, value["position"]),
            velocity=cast(Any, value["velocity"]),
        )


@dataclass(frozen=True)
class UniVTACPreMoveTrajectory:
    """Four source-bound plans for ACT/QPOS8 ``grasp_classify`` reset."""

    task_id: str
    action_spec: str
    initial_seed: int
    exogenous_seed: int
    pair_key: str
    dataset_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    source_run_content_sha256: str
    reset_reference_sha256: str
    capture_reset_receipt_sha256: str
    capture_simulator_state_sha256: str
    capture_native_step: int
    capture_qpos8: Tuple[float, ...]
    source_qpos8_max_abs_error: float
    upstream_commit: str
    task_source_sha256: str
    post_reset_hold_steps: int
    segments: Tuple[UniVTACPreMoveSegment, ...]
    capture_mode: str = PRE_MOVE_CAPTURE_MODE
    schema: str = UNIVTAC_PRE_MOVE_TRAJECTORY_SCHEMA
    semantic_version: str = UNIVTAC_PRE_MOVE_TRAJECTORY_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        if self.task_id != "grasp_classify":
            message = "pre-move replay currently supports only grasp_classify"
            raise UniVTACPreMoveTrajectoryError(message)
        if self.action_spec != QPOS8_ACTION_SPEC:
            raise UniVTACPreMoveTrajectoryError("pre-move replay requires QPOS8")
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(self, name, _seed(getattr(self, name), name))
        hold_steps = _hold_steps(self.post_reset_hold_steps)
        object.__setattr__(self, "post_reset_hold_steps", hold_steps)
        native_step = _seed(self.capture_native_step, "capture_native_step")
        object.__setattr__(self, "capture_native_step", native_step)
        qpos8 = _vector(self.capture_qpos8, "capture_qpos8", 8)
        object.__setattr__(self, "capture_qpos8", qpos8)
        qpos8_error = _source_qpos8_error(self.source_qpos8_max_abs_error)
        object.__setattr__(self, "source_qpos8_max_abs_error", qpos8_error)
        for name in (
            "pair_key",
            "dataset_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "source_run_content_sha256",
            "reset_reference_sha256",
            "capture_reset_receipt_sha256",
            "capture_simulator_state_sha256",
            "task_source_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        commit_valid = isinstance(self.upstream_commit, str) and _GIT_COMMIT.fullmatch(
            self.upstream_commit
        )
        if not commit_valid:
            message = "upstream_commit must be a lowercase 40-character Git commit"
            raise UniVTACPreMoveTrajectoryError(message)
        segments = tuple(self.segments)
        if any(not isinstance(item, UniVTACPreMoveSegment) for item in segments):
            raise UniVTACPreMoveTrajectoryError("segments must contain segment objects")
        if tuple(segment.kind for segment in segments) != _EXPECTED_KINDS:
            raise UniVTACPreMoveTrajectoryError(
                "segments must be ordered gripper, arm, gripper, arm"
            )
        object.__setattr__(self, "segments", segments)
        if self.capture_mode != PRE_MOVE_CAPTURE_MODE:
            raise UniVTACPreMoveTrajectoryError("trajectory capture mode mismatch")
        if self.schema != UNIVTAC_PRE_MOVE_TRAJECTORY_SCHEMA:
            raise UniVTACPreMoveTrajectoryError("trajectory schema mismatch")
        if self.semantic_version != UNIVTAC_PRE_MOVE_TRAJECTORY_SEMANTIC_VERSION:
            raise UniVTACPreMoveTrajectoryError("trajectory semantic version mismatch")

    @property
    def sha256(self) -> str:
        """Return the canonical hash of the complete trajectory document."""

        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON-safe representation."""

        return {
            "task_id": self.task_id,
            "action_spec": self.action_spec,
            "initial_seed": self.initial_seed,
            "exogenous_seed": self.exogenous_seed,
            "pair_key": self.pair_key,
            "dataset_sha256": self.dataset_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "source_run_content_sha256": self.source_run_content_sha256,
            "reset_reference_sha256": self.reset_reference_sha256,
            "capture_reset_receipt_sha256": self.capture_reset_receipt_sha256,
            "capture_simulator_state_sha256": self.capture_simulator_state_sha256,
            "capture_native_step": self.capture_native_step,
            "capture_qpos8": list(self.capture_qpos8),
            "source_qpos8_max_abs_error": self.source_qpos8_max_abs_error,
            "upstream_commit": self.upstream_commit,
            "task_source_sha256": self.task_source_sha256,
            "post_reset_hold_steps": self.post_reset_hold_steps,
            "segments": [segment.to_dict() for segment in self.segments],
            "capture_mode": self.capture_mode,
            "schema": self.schema,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> UniVTACPreMoveTrajectory:
        """Load the exact trajectory document emitted by :meth:`to_dict`."""

        if not isinstance(value, Mapping):
            raise UniVTACPreMoveTrajectoryError("trajectory must be a mapping")
        if set(value) != set(cls.__dataclass_fields__):
            raise UniVTACPreMoveTrajectoryError("trajectory fields mismatch")
        raw_segments = value["segments"]
        if not isinstance(raw_segments, list):
            raise UniVTACPreMoveTrajectoryError("segments must be a JSON array")
        document = dict(value)
        document["segments"] = tuple(
            UniVTACPreMoveSegment.from_dict(item) for item in raw_segments
        )
        return cls(**cast(Any, document))


class PreMoveTrajectoryCapture:
    """Mutable one-reset capture lifecycle returned by the installer."""

    def __init__(self, task: Any, manager: Any) -> None:
        self._task = task
        self._manager = manager
        self._active = False
        self._complete = False
        self._reset_seed: int | None = None
        self._post_reset_hold_steps = 0
        self._segments: list[UniVTACPreMoveSegment] = []

    def _begin(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
        self._reset_seed = _seed(
            kwargs.get("seed", args[0] if args else None), "reset seed"
        )
        self._segments = []
        self._post_reset_hold_steps = 0
        self._complete = False
        self._active = True

    def _validate_segments(self) -> None:
        if tuple(item.kind for item in self._segments) != _EXPECTED_KINDS:
            raise UniVTACPreMoveTrajectoryError("reset planner sequence mismatch")

    def _end(self, post_reset_hold_steps: int) -> None:
        self._active = False
        self._validate_segments()
        self._post_reset_hold_steps = _hold_steps(post_reset_hold_steps)
        self._complete = True

    def _abort(self) -> None:
        self._segments = []
        self._post_reset_hold_steps = 0
        self._complete = False
        self._active = False

    def _record(self, segment: UniVTACPreMoveSegment) -> None:
        if not self._active or getattr(self._task, "in_pre_move", None) is not True:
            return
        index = len(self._segments)
        if index >= 4 or segment.kind != _EXPECTED_KINDS[index]:
            raise UniVTACPreMoveTrajectoryError("pre_move planner order mismatch")
        self._segments.append(segment)

    def finish(
        self,
        *,
        task_id: str,
        action_spec: str,
        initial_seed: int,
        exogenous_seed: int,
        pair_key: str,
        dataset_sha256: str,
        checkpoint_sha256: str,
        config_sha256: str,
        source_run_content_sha256: str,
        reset_reference_sha256: str,
        capture_reset_receipt_sha256: str,
        capture_simulator_state_sha256: str,
        capture_native_step: int,
        capture_qpos8: Tuple[float, ...],
        source_qpos8_max_abs_error: float,
        upstream_commit: str,
        task_source_sha256: str,
    ) -> UniVTACPreMoveTrajectory:
        """Freeze the last complete reset under its formal identity."""
        if not self._complete or self._reset_seed is None:
            raise UniVTACPreMoveTrajectoryError("no complete reset has been captured")
        if _seed(initial_seed, "initial_seed") != self._reset_seed:
            raise UniVTACPreMoveTrajectoryError("captured reset seed mismatch")
        return UniVTACPreMoveTrajectory(
            task_id=task_id,
            action_spec=action_spec,
            initial_seed=initial_seed,
            exogenous_seed=exogenous_seed,
            pair_key=pair_key,
            dataset_sha256=dataset_sha256,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=config_sha256,
            source_run_content_sha256=source_run_content_sha256,
            reset_reference_sha256=reset_reference_sha256,
            capture_reset_receipt_sha256=capture_reset_receipt_sha256,
            capture_simulator_state_sha256=capture_simulator_state_sha256,
            capture_native_step=capture_native_step,
            capture_qpos8=capture_qpos8,
            source_qpos8_max_abs_error=source_qpos8_max_abs_error,
            upstream_commit=upstream_commit,
            task_source_sha256=task_source_sha256,
            post_reset_hold_steps=self._post_reset_hold_steps,
            segments=tuple(self._segments),
        )


def install_pre_move_trajectory_capture(
    task: Any, expected_native_step: int | None = None
) -> PreMoveTrajectoryCapture:
    """Lazily install the live capture hook without import cycles."""
    from robotactile_benchmark.backends.univtac_reset_trajectory_runtime import (
        install_pre_move_trajectory_capture as install,
    )

    return install(task, expected_native_step=expected_native_step)


def install_pre_move_trajectory_replay(
    task: Any, trajectory: UniVTACPreMoveTrajectory | None
) -> bool:
    """Lazily install the live replay hook without import cycles."""
    from robotactile_benchmark.backends.univtac_reset_trajectory_runtime import (
        install_pre_move_trajectory_replay as install,
    )

    return install(task, trajectory)


__all__ = [
    "PreMoveTrajectoryCapture",
    "PRE_MOVE_CAPTURE_MODE",
    "UNIVTAC_PRE_MOVE_TRAJECTORY_SCHEMA",
    "UNIVTAC_PRE_MOVE_TRAJECTORY_SEMANTIC_VERSION",
    "UniVTACPreMoveSegment",
    "UniVTACPreMoveTrajectory",
    "UniVTACPreMoveTrajectoryError",
    "install_pre_move_trajectory_capture",
    "install_pre_move_trajectory_replay",
]

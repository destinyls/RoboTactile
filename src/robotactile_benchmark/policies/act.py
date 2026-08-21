"""Closed-loop adapters for the qualified strict ACT runtime boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Optional, Protocol

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    ActionPlan,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, ObservationRecord

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


class StrictACTRuntime(Protocol):
    """The only model boundary exposed by StrictACTRuntime."""

    def reset(self) -> None:
        """Reset temporal aggregation."""

        ...

    def get_action(self, observation: Mapping[str, object]) -> Array:
        """Advance the model once and return one qpos8 action."""

        ...


RuntimeInputTransform = Callable[[Mapping[str, Array]], Mapping[str, object]]


class _PolicyState(str, Enum):
    NEW = "new"
    READY = "ready"
    AWAITING_COMMIT = "awaiting_commit"
    ABORTED = "aborted"
    CLOSED = "closed"


def _identity_input(value: Mapping[str, Array]) -> Mapping[str, object]:
    return value


def _require_rgb(value: object, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be exact uint8 HWC RGB")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous")
    return value


def _linear_resample_matrix(source_size: int, target_size: int) -> Array:
    scale = source_size / target_size
    support = max(scale, 1.0)
    matrix = np.zeros((target_size, source_size), dtype=np.float32)
    for target_index in range(target_size):
        center = (target_index + 0.5) * scale - 0.5
        first = max(0, int(np.ceil(center - support)))
        last = min(source_size - 1, int(np.floor(center + support)))
        source_indices = np.arange(first, last + 1, dtype=np.int64)
        weights = np.maximum(0.0, 1.0 - np.abs(source_indices - center) / support)
        weights /= weights.sum()
        matrix[target_index, first : last + 1] = weights.astype(np.float32)
    return matrix


def _resize_chw_bilinear(value: Array, size: int = 256) -> Array:
    """Match torchvision's antialiased tensor bilinear Resize exactly."""

    _, source_height, source_width = value.shape
    vertical = _linear_resample_matrix(source_height, size)
    horizontal = _linear_resample_matrix(source_width, size)
    resized_height = np.einsum("oh,chw->cow", vertical, value, optimize=True)
    resized = np.einsum("pw,cow->cop", horizontal, resized_height, optimize=True)
    return np.ascontiguousarray(resized, dtype=np.float32)


def preprocess_camera(value: object) -> Array:
    """Match ToTensor -> Resize(256) -> ImageNet normalization."""

    rgb = _require_rgb(value, "camera top")
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(
        (_resize_chw_bilinear(chw) - _IMAGENET_MEAN) / _IMAGENET_STD
    )


def preprocess_tactile(value: object, slot_id: str) -> Array:
    """Match ToTensor -> Resize(256), with no ImageNet normalization."""

    rgb = _require_rgb(value, f"tactile {slot_id}")
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / np.float32(255.0)
    return _resize_chw_bilinear(chw)


class _BaseACTPolicy:
    """Shared lifecycle without importing torch or upstream ACT modules."""

    def __init__(
        self,
        identity: PolicyIdentity,
        runtime: StrictACTRuntime,
        *,
        artifact_task: str,
        input_transform: RuntimeInputTransform = _identity_input,
    ) -> None:
        if identity.action_spec != ACTION_SPEC:
            raise ValueError("ACT policy requires qpos8_next_step")
        if not identity.consumes_tactile or identity.supports_structural_absence:
            raise ValueError(
                "strict ACT requires tactile and rejects structural absence"
            )
        if not isinstance(artifact_task, str) or not artifact_task:
            raise ValueError("ACT artifact task must be a non-empty string")
        self.identity = identity
        self._runtime = runtime
        self._artifact_task = artifact_task
        self._input_transform = input_transform
        self._state = _PolicyState.NEW
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending_plan: Optional[ActionPlan] = None

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._state is _PolicyState.CLOSED:
            raise RuntimeError("closed ACT policy cannot reset")
        if type(context) is not PolicyEpisodeContext:
            raise TypeError("context must be an exact PolicyEpisodeContext")
        if context.action_spec != self.identity.action_spec:
            raise ValueError("context action spec does not match ACT identity")
        if context.task != self._artifact_task:
            raise ValueError("context task does not match ACT artifact task")
        self._runtime.reset()
        self._context = context
        self._pending_plan = None
        self._state = _PolicyState.READY

    def _validate_observation(self, observation: ObservationRecord) -> None:
        if type(observation) is not ObservationRecord:
            raise TypeError("ACT infer accepts only exact ObservationRecord")
        if self._state is not _PolicyState.READY or self._context is None:
            raise RuntimeError("ACT infer requires a fresh reset and completed commit")
        if (
            observation.episode_id != self._context.episode_id
            or observation.task != self._context.task
            or observation.seed != self._context.initial_seed
        ):
            raise ValueError("ACT observation identity does not match reset context")
        if observation.proprio.shape != (8,):
            raise ValueError(
                "ACT proprio must have exact shape (8,); slicing is forbidden"
            )
        if observation.proprio.dtype != np.float32:
            raise TypeError("ACT proprio must use float32")
        if not np.isfinite(observation.proprio).all():
            raise ValueError("ACT proprio must be finite")

    def _encode(self, observation: ObservationRecord) -> Mapping[str, Array]:
        self._validate_observation(observation)
        if "top" not in observation.vision:
            raise ValueError("ACT observation requires vision['top']")
        encoded: dict[str, Array] = {
            "cam_high": preprocess_camera(observation.vision["top"]),
            "qpos": np.ascontiguousarray(observation.proprio.copy()),
        }
        for slot_id, runtime_key in (("left", "tac_left"), ("right", "tac_right")):
            sensor = observation.sensor(slot_id)
            if not sensor.payload_present or sensor.payload is None:
                raise ValueError(
                    "tactile ACT does not support structural tactile absence"
                )
            encoded[runtime_key] = preprocess_tactile(sensor.payload, slot_id)
        return encoded

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        encoded = self._encode(observation)
        runtime_input = self._input_transform(encoded)
        self._state = _PolicyState.ABORTED
        action = self._runtime.get_action(runtime_input)
        if not isinstance(action, np.ndarray):
            raise TypeError("StrictACTRuntime output must be a numpy array")
        if action.dtype != np.float32:
            raise TypeError("StrictACTRuntime output must use exact float32")
        if action.shape != (1, 8):
            raise ValueError("StrictACTRuntime output must have exact shape (1, 8)")
        if not np.isfinite(action).all():
            raise ValueError("StrictACTRuntime output must be finite")
        plan = ActionPlan(ACTION_SPEC, observation.step_index, action)
        self._pending_plan = plan
        self._state = _PolicyState.AWAITING_COMMIT
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        if (
            self._state is not _PolicyState.AWAITING_COMMIT
            or self._pending_plan is None
        ):
            raise RuntimeError("ACT commit requires one pending inference")
        if type(execution) is not PolicyExecution:
            raise TypeError("execution must be an exact PolicyExecution")
        if execution.action_plan_sha256 != self._pending_plan.sha256:
            raise ValueError("ACT execution does not match the pending action plan")
        expected = self._pending_plan.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("ACT executed actions do not match the pending plan")
        self._pending_plan = None
        self._state = _PolicyState.READY

    def abort(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("abort reason code must be non-empty")
        if self._state is _PolicyState.CLOSED:
            return
        self._pending_plan = None
        self._state = _PolicyState.ABORTED

    def close(self) -> None:
        if self._state is _PolicyState.CLOSED:
            return
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()
        self._pending_plan = None
        self._context = None
        self._state = _PolicyState.CLOSED


class StrictACTPolicy(_BaseACTPolicy):
    """Tactile-required wrapper around StrictACTRuntime.get_action()."""

    def __init__(
        self,
        identity: PolicyIdentity,
        runtime: StrictACTRuntime,
        *,
        artifact_task: str,
        input_transform: RuntimeInputTransform = _identity_input,
    ) -> None:
        super().__init__(
            identity,
            runtime,
            artifact_task=artifact_task,
            input_transform=input_transform,
        )

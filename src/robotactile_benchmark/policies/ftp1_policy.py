"""Official FTP-1 adapter for the six released UniVTAC checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional, Protocol

import numpy as np

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, ObservationRecord, freeze_array

FTP1_PROTOCOL = "robotactile.ftp1.v1"
ACTION_DIM = 8
EXECUTED_ACTION_STEPS = 1

_TASK_PROMPTS = {
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_bottle": (
        "grasp the bottle and lift it vertically, keeping its final base within "
        "5 cm of the wall."
    ),
    "lift_can": "grasp the can and lifts it vertically without slippage.",
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": (
        "grasp the bottle, then position it into the shelf cavity."
    ),
}
_WRIST_TASKS = frozenset({"insert_tube", "lift_can"})


class FTP1PolicyClient(Protocol):
    """Dependency-light operations exposed by the isolated FTP-1 runtime."""

    def reset(self, *, task_id: str, prompt: str, seed: int) -> object: ...

    def infer(self, observation: Mapping[str, object]) -> Array: ...

    def close(self) -> None: ...


def ftp1_supported_tasks() -> tuple[str, ...]:
    """Return tasks with public task-specific checkpoints in fixed order."""

    return tuple(_TASK_PROMPTS)


def ftp1_training_prompt(task_id: str) -> str:
    """Return the exact prompt used by the released upstream evaluator."""

    try:
        return _TASK_PROMPTS[task_id]
    except KeyError as error:
        raise KeyError(
            f"FTP-1 has no released UniVTAC checkpoint for {task_id}"
        ) from error


def ftp1_uses_wrist_camera(task_id: str) -> bool:
    """Return the pinned upstream camera routing for one released task."""

    ftp1_training_prompt(task_id)
    return task_id in _WRIST_TASKS


def _rgb(value: object, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be exact uint8 HWC with three channels")
    return freeze_array(value, np.uint8)


def ftp1_wire_observation(
    observation: ObservationRecord,
    *,
    task_id: str,
    prompt: str,
    use_wrist: Optional[bool] = None,
) -> dict[str, object]:
    """Build the canonical RPC value before upstream resize/state packing."""

    if observation.proprio.dtype != np.float32 or observation.proprio.shape != (8,):
        raise ValueError("FTP-1 proprio must be exact float32 shape (8,)")
    if "top" not in observation.vision:
        raise ValueError("FTP-1 requires vision['top']")
    wire: dict[str, object] = {
        "task": task_id,
        "prompt": prompt,
        "top": _rgb(observation.vision["top"], "vision.top"),
        "qpos8": freeze_array(observation.proprio, np.float32),
    }
    if ftp1_uses_wrist_camera(task_id) if use_wrist is None else use_wrist:
        if "wrist_l" not in observation.vision:
            raise ValueError(f"FTP-1 task {task_id} requires vision['wrist_l']")
        wire["wrist"] = _rgb(observation.vision["wrist_l"], "vision.wrist_l")
    for slot_id in ("left", "right"):
        sensor = observation.sensor(slot_id)
        if not sensor.payload_present or sensor.payload is None:
            raise ValueError("FTP-1 requires both tactile payloads")
        wire[slot_id] = _rgb(sensor.payload, f"tactile.{slot_id}")
    return wire


class OfficialFTP1Policy:
    """One-action qpos8 policy matching upstream per-step temporal ensemble."""

    def __init__(
        self,
        identity: PolicyIdentity,
        client_factory: Callable[[], FTP1PolicyClient],
        *,
        task_id: str,
        training_prompt: Optional[str] = None,
        use_wrist: Optional[bool] = None,
    ) -> None:
        if type(identity) is not PolicyIdentity:
            raise TypeError("identity must be an exact PolicyIdentity")
        if identity.action_spec != QPOS8_ACTION_SPEC:
            raise ValueError("FTP-1 requires qpos8_next_step")
        if not identity.consumes_tactile or identity.supports_structural_absence:
            raise ValueError("FTP-1 requires two present tactile payloads")
        if training_prompt is None:
            ftp1_training_prompt(task_id)
        elif (
            not training_prompt
            or training_prompt.strip() != training_prompt
            or type(use_wrist) is not bool
        ):
            raise ValueError("retrained FTP-1 needs an exact prompt and camera route")
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        self.identity = identity
        self._client_factory = client_factory
        self._task_id = task_id
        self._prompt = training_prompt or ftp1_training_prompt(task_id)
        self._use_wrist = use_wrist
        self._client: Optional[FTP1PolicyClient] = None
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending: Optional[ActionPlan] = None
        self._closed = False

    def _client_instance(self) -> FTP1PolicyClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("closed FTP-1 policy cannot reset")
        if type(context) is not PolicyEpisodeContext:
            raise TypeError("context must be an exact PolicyEpisodeContext")
        prompt = self._prompt
        if (
            context.action_spec != QPOS8_ACTION_SPEC
            or context.task != self._task_id
            or context.instruction != prompt
        ):
            raise ValueError("FTP-1 reset task/action/prompt contract mismatch")
        self._client_instance().reset(
            task_id=self._task_id,
            prompt=prompt,
            seed=context.exogenous_seed,
        )
        self._context = context
        self._pending = None

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        context = self._context
        if context is None:
            raise RuntimeError("FTP-1 infer requires reset")
        if self._pending is not None:
            raise RuntimeError("FTP-1 infer requires prior commit")
        if (
            observation.episode_id != context.episode_id
            or observation.task != context.task
            or observation.seed != context.initial_seed
        ):
            raise ValueError("FTP-1 observation identity mismatch")
        action = self._client_instance().infer(
            ftp1_wire_observation(
                observation,
                task_id=self._task_id,
                prompt=self._prompt,
                use_wrist=self._use_wrist,
            )
        )
        if action.dtype != np.float32 or action.shape != (ACTION_DIM,):
            raise ValueError("FTP-1 service must return float32 shape (8,)")
        actions = freeze_array(action.reshape(1, ACTION_DIM), np.float32)
        plan = ActionPlan(QPOS8_ACTION_SPEC, observation.step_index, actions)
        self._pending = plan
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        if self._pending is None:
            raise RuntimeError("FTP-1 commit requires pending inference")
        if type(execution) is not PolicyExecution:
            raise TypeError("execution must be an exact PolicyExecution")
        if execution.action_plan_sha256 != self._pending.sha256:
            raise ValueError("FTP-1 execution does not match action plan")
        expected = self._pending.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("FTP-1 executed actions differ from its plan")
        if (
            execution.terminal_signal is BackendSignal.RUNNING
            and execution.executed_actions.shape != (1, ACTION_DIM)
        ):
            raise RuntimeError("FTP-1 running commit requires exactly one action")
        self._pending = None

    def abort(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("abort reason code must be non-empty")
        if self._client is not None:
            self._client.close()
        self._client = None
        self._context = None
        self._pending = None

    def close(self) -> None:
        if self._closed:
            return
        if self._client is not None:
            self._client.close()
        self._client = None
        self._context = None
        self._pending = None
        self._closed = True


__all__ = [
    "ACTION_DIM",
    "EXECUTED_ACTION_STEPS",
    "FTP1_PROTOCOL",
    "FTP1PolicyClient",
    "OfficialFTP1Policy",
    "ftp1_supported_tasks",
    "ftp1_training_prompt",
    "ftp1_uses_wrist_camera",
    "ftp1_wire_observation",
]

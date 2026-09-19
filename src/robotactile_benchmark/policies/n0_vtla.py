"""Official N0-VTLA policy adapter for the released UniVTAC checkpoint."""

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
from robotactile_benchmark.policies.n0_vtla_execution import n0_vtla_execution_steps
from robotactile_benchmark.policies.tactile_availability import (
    RGBShape,
    TactileAvailabilityMode,
    normalize_zero_shape,
    validate_availability_config,
    zero_fill_observation,
)

ACTION_HORIZON = 50
PADDED_ACTION_DIM = 32
RAW_ACTION_DIM = 8
SUPPORTED_TASK = "insert_hole"
TRAINING_PROMPT = "insert hole"


class N0VTLAClient(Protocol):
    """Operations required from the official ZMQ transport."""

    def reset(self) -> None: ...

    def infer(self, observation: Mapping[str, object]) -> Array: ...

    def close(self) -> None: ...


def n0_vtla_training_prompt(task_id: str) -> str:
    """Return the released checkpoint's exact serving prompt."""

    if task_id != SUPPORTED_TASK:
        raise KeyError(f"official N0-VTLA has no released UniVTAC policy for {task_id}")
    return TRAINING_PROMPT


def _rgb(value: object, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be exact uint8 HWC RGB")
    return freeze_array(value, np.uint8)


def _wire_observation(
    observation: ObservationRecord,
    prompt: str,
    mode: TactileAvailabilityMode = TactileAvailabilityMode.REQUIRED,
    zero_shape: Optional[RGBShape] = None,
) -> dict[str, object]:
    if observation.proprio.dtype != np.float32 or observation.proprio.shape != (8,):
        raise ValueError("N0-VTLA proprio must be exact float32 shape (8,)")
    if not {"top", "wrist_l"}.issubset(observation.vision):
        raise ValueError("N0-VTLA requires vision['top'] and vision['wrist_l']")
    if mode is TactileAvailabilityMode.ZERO_FILL:
        if zero_shape is None:
            raise ValueError("zero-fill requires a declared tactile shape")
        observation = zero_fill_observation(observation, zero_shape)
    tactile: dict[str, Array] = {}
    for slot_id in ("left", "right"):
        sensor = observation.sensor(slot_id)
        if not sensor.payload_present or sensor.payload is None:
            if mode is TactileAvailabilityMode.NATIVE_MISSING:
                continue
            raise ValueError("N0-VTLA requires both tactile payloads")
        tactile[slot_id] = _rgb(sensor.payload, f"tactile.{slot_id}")
    message: dict[str, object] = {
        "state": freeze_array(observation.proprio, np.float32),
        "prompt": prompt,
        "observation/image": _rgb(observation.vision["top"], "vision.top"),
        "observation/wrist_image": _rgb(
            observation.vision["wrist_l"], "vision.wrist_l"
        ),
    }
    message.update(
        {f"observation/{slot}_tactile": image for slot, image in tactile.items()}
    )
    if mode is not TactileAvailabilityMode.REQUIRED:
        message["robotactile_tactile_protocol"] = mode.value
    return message


class OfficialN0VTLAPolicy:
    """50-step qpos adapter compatible with RoboTactile's closed-loop runner."""

    def __init__(
        self,
        identity: PolicyIdentity,
        client_factory: Callable[[], N0VTLAClient],
        *,
        task_id: str = SUPPORTED_TASK,
        training_prompt: Optional[str] = None,
        tactile_availability_mode: TactileAvailabilityMode = TactileAvailabilityMode.REQUIRED,
        tactile_zero_shape: Optional[RGBShape] = None,
        execution_profile: Optional[str] = None,
    ) -> None:
        if type(identity) is not PolicyIdentity:
            raise TypeError("identity must be an exact PolicyIdentity")
        if identity.action_spec != QPOS8_ACTION_SPEC:
            raise ValueError("official N0-VTLA requires qpos8_next_step")
        mode = TactileAvailabilityMode(tactile_availability_mode)
        shape = normalize_zero_shape(tactile_zero_shape)
        validate_availability_config(mode, shape, "n0_vtla")
        if not identity.consumes_tactile or identity.supports_structural_absence != (
            mode is not TactileAvailabilityMode.REQUIRED
        ):
            raise ValueError("official N0-VTLA requires two tactile streams")
        self._availability_mode = mode
        self._zero_shape = shape
        self._execute_steps = n0_vtla_execution_steps(execution_profile, task_id)
        if training_prompt is None and task_id != SUPPORTED_TASK:
            raise ValueError("released N0-VTLA checkpoint supports only insert_hole")
        if training_prompt is not None and (
            not training_prompt or training_prompt.strip() != training_prompt
        ):
            raise ValueError("retrained N0-VTLA requires an exact nonempty prompt")
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        self.identity = identity
        self._client_factory = client_factory
        self._task_id = task_id
        self._prompt = training_prompt or n0_vtla_training_prompt(task_id)
        self._client: Optional[N0VTLAClient] = None
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending: Optional[ActionPlan] = None
        self._closed = False

    def _client_instance(self) -> N0VTLAClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("closed N0-VTLA policy cannot reset")
        if type(context) is not PolicyEpisodeContext:
            raise TypeError("context must be an exact PolicyEpisodeContext")
        if context.action_spec != QPOS8_ACTION_SPEC or context.task != self._task_id:
            raise ValueError("N0-VTLA reset task/action contract mismatch")
        if context.instruction != self._prompt:
            raise ValueError("N0-VTLA reset prompt must match the released checkpoint")
        self._client_instance().reset()
        self._context = context
        self._pending = None

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        context = self._context
        if context is None:
            raise RuntimeError("N0-VTLA infer requires reset")
        if self._pending is not None:
            raise RuntimeError("N0-VTLA infer requires prior commit")
        if (
            observation.episode_id != context.episode_id
            or observation.task != context.task
            or observation.seed != context.initial_seed
        ):
            raise ValueError("N0-VTLA observation identity mismatch")
        native = self._client_instance().infer(
            _wire_observation(
                observation, self._prompt, self._availability_mode, self._zero_shape
            )
        )
        if native.dtype != np.float32 or native.shape != (
            ACTION_HORIZON,
            PADDED_ACTION_DIM,
        ):
            raise ValueError("N0-VTLA server must return float32 shape (50, 32)")
        # The checkpoint predicts joint deltas internally, but the official policy
        # applies AbsoluteActions against the supplied state before serve_zmq pads
        # the result. Adding proprio again here would double-apply that transform.
        actions = freeze_array(native[:, :RAW_ACTION_DIM], np.float32)
        plan = ActionPlan(QPOS8_ACTION_SPEC, observation.step_index, actions)
        self._pending = plan
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        if self._pending is None:
            raise RuntimeError("N0-VTLA commit requires pending inference")
        if type(execution) is not PolicyExecution:
            raise TypeError("execution must be an exact PolicyExecution")
        if execution.action_plan_sha256 != self._pending.sha256:
            raise ValueError("N0-VTLA execution does not match action plan")
        expected = self._pending.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("N0-VTLA executed actions differ from its plan")
        executed_count = execution.executed_actions.shape[0]
        if not 1 <= executed_count <= self._execute_steps:
            raise RuntimeError("N0-VTLA commit exceeds the configured execution prefix")
        if (
            execution.terminal_signal is BackendSignal.RUNNING
            and executed_count != self._execute_steps
        ):
            raise RuntimeError(
                f"N0-VTLA running commit requires the full {self._execute_steps}-step chunk"
            )
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
    "ACTION_HORIZON",
    "N0VTLAClient",
    "OfficialN0VTLAPolicy",
    "SUPPORTED_TASK",
    "TRAINING_PROMPT",
    "n0_vtla_training_prompt",
]

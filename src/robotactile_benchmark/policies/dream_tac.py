"""Dream-Tac adapter for the upstream Franka HTTP inference contract."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Optional, Protocol

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop.contracts import (
    ActionPlan,
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, ObservationRecord, freeze_array

ACTION_HORIZON = 20
RAW_ACTION_DIM = 7
EE_ACTION_DIM = 8
GRIPPER_CLOSE_THRESHOLD = 0.5
OPEN_GRIPPER_QPOS = 0.04
CLOSED_GRIPPER_QPOS = 0.0
TACTILE_GATE_MEDIAN = 0.002
TACTILE_GATE_MAD = 0.001
TACTILE_GATE_SIGMOID_K = 4.0
TACTILE_GATE_MIN = 0.15
TACTILE_GATE_MAX = 1.0


class DreamTacGripperMapping(str, Enum):
    """Explicit resolution of the contradictory upstream gripper convention."""

    GREATER_IS_CLOSED = "greater_than_threshold_is_closed_v1"
    GREATER_IS_OPEN = "greater_than_threshold_is_open_v1"
    DIRECT_QPOS = "direct_qpos_v1"


class DreamTacClient(Protocol):
    """Operations required from the pinned upstream HTTP server."""

    def reset(
        self, *, experiment_config: str, action_horizon: int, use_tactile: bool
    ) -> None: ...

    def infer(self, observation: Mapping[str, object]) -> Array: ...

    def close(self) -> None: ...


def _rgb(value: object, name: str) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be exact uint8 HWC RGB")
    return freeze_array(value, np.uint8)


def quaternion_wxyz_to_euler_xyz(quaternion: Array) -> Array:
    """Convert a normalized WXYZ quaternion to upstream XYZ Euler angles."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must be finite shape (4,)")
    norm = float(np.linalg.norm(value))
    if not math.isclose(norm, 1.0, abs_tol=1e-4, rel_tol=0.0):
        raise ValueError("quaternion must have unit norm")
    w, x, y, z = value / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return freeze_array(np.asarray((roll, pitch, yaw), dtype=np.float32), np.float32)


def unwrap_euler_xyz(current: Array, previous: Array | None) -> Array:
    """Continue wrapped XYZ Euler angles across an episode like training."""

    value = np.asarray(current, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("current Euler angles must be finite shape (3,)")
    if previous is None:
        return freeze_array(value.astype(np.float32), np.float32)
    reference = np.asarray(previous, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("previous Euler angles must be finite shape (3,)")
    continued = np.unwrap(np.stack((reference, value), axis=0), axis=0)[1]
    return freeze_array(continued.astype(np.float32), np.float32)


def dream_tac_tactile_self_attn_gate(
    left_current: Array,
    right_current: Array,
    left_previous: Array | None,
    right_previous: Array | None,
) -> float:
    """Reproduce the train759 tactile-delta gate for one observation step."""

    current_left = _rgb(left_current, "tactile.left.current")
    current_right = _rgb(right_current, "tactile.right.current")
    if (left_previous is None) != (right_previous is None):
        raise ValueError("previous tactile gate inputs must be both present or absent")
    raw = 0.0
    if left_previous is not None and right_previous is not None:
        previous_left = _rgb(left_previous, "tactile.left.previous")
        previous_right = _rgb(right_previous, "tactile.right.previous")
        if (
            previous_left.shape != current_left.shape
            or previous_right.shape != current_right.shape
        ):
            raise ValueError("consecutive tactile frames must preserve sensor shape")
        left_delta = float(
            np.abs(
                current_left.astype(np.float32) - previous_left.astype(np.float32)
            ).mean()
            / 255.0
        )
        right_delta = float(
            np.abs(
                current_right.astype(np.float32) - previous_right.astype(np.float32)
            ).mean()
            / 255.0
        )
        raw = max(left_delta, right_delta)
    z = TACTILE_GATE_SIGMOID_K * (raw - TACTILE_GATE_MEDIAN) / (TACTILE_GATE_MAD + 1e-6)
    z = max(-30.0, min(30.0, z))
    gate_01 = 1.0 / (1.0 + math.exp(-z))
    return TACTILE_GATE_MIN + (TACTILE_GATE_MAX - TACTILE_GATE_MIN) * gate_01


def euler_xyz_to_quaternion_wxyz(euler: Array) -> Array:
    """Convert upstream XYZ Euler angles to a normalized WXYZ quaternion."""

    value = np.asarray(euler, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Euler angles must be finite shape (3,)")
    roll, pitch, yaw = value
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    quaternion = np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    return freeze_array(quaternion.astype(np.float32), np.float32)


def dream_tac_wire_observation(
    observation: ObservationRecord,
    *,
    instruction: str,
    previous_euler: Array | None = None,
    tactile_self_attn_gate: float | None = None,
) -> dict[str, object]:
    """Build the official two-camera, two-tactile, 6D-pose request."""

    if observation.proprio.dtype != np.float32 or observation.proprio.shape != (8,):
        raise ValueError("Dream-Tac proprio must be exact float32 EE8 shape (8,)")
    if not {"top", "wrist_l"}.issubset(observation.vision):
        raise ValueError("Dream-Tac requires vision['top'] and vision['wrist_l']")
    euler = unwrap_euler_xyz(
        quaternion_wxyz_to_euler_xyz(observation.proprio[3:7]), previous_euler
    )
    state = freeze_array(np.concatenate((observation.proprio[:3], euler)), np.float32)
    tactile: dict[str, Array] = {}
    for slot_id in ("left", "right"):
        sensor = observation.sensor(slot_id)
        if not sensor.payload_present or sensor.payload is None:
            raise ValueError("Dream-Tac requires both tactile payloads")
        tactile[slot_id] = _rgb(sensor.payload, f"tactile.{slot_id}")
    gate = (
        dream_tac_tactile_self_attn_gate(tactile["left"], tactile["right"], None, None)
        if tactile_self_attn_gate is None
        else tactile_self_attn_gate
    )
    if isinstance(gate, bool) or not isinstance(gate, (int, float)):
        raise TypeError("tactile_self_attn_gate must be a real number")
    gate = float(gate)
    if not math.isfinite(gate) or not TACTILE_GATE_MIN <= gate <= TACTILE_GATE_MAX:
        raise ValueError("tactile_self_attn_gate is outside the training range")
    return {
        "cam_front": _rgb(observation.vision["top"], "vision.top"),
        "cam_high": _rgb(observation.vision["wrist_l"], "vision.wrist_l"),
        "instruction": instruction,
        "state": state,
        "tactile_left": tactile["left"],
        "tactile_right": tactile["right"],
        "tactile_self_attn_gate": gate,
    }


def dream_tac_actions_to_ee8(
    actions: Array,
    *,
    gripper_mapping: DreamTacGripperMapping,
    gripper_threshold: float = GRIPPER_CLOSE_THRESHOLD,
    gripper_qpos_min: float | None = None,
    gripper_qpos_max: float | None = None,
) -> Array:
    """Convert upstream XYZ/RPY/gripper chunks to UniVTAC EE8."""

    if isinstance(gripper_threshold, bool) or not isinstance(
        gripper_threshold, (int, float)
    ):
        raise TypeError("gripper_threshold must be a real number")
    if not math.isfinite(float(gripper_threshold)):
        raise ValueError("gripper_threshold must be finite")
    value = np.asarray(actions)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != RAW_ACTION_DIM:
        raise ValueError("Dream-Tac actions must be float32 shape [H, 7]")
    if not np.isfinite(value).all():
        raise ValueError("Dream-Tac actions must be finite")
    converted = np.empty((value.shape[0], EE_ACTION_DIM), dtype=np.float32)
    converted[:, :3] = value[:, :3]
    for index, action in enumerate(value):
        converted[index, 3:7] = euler_xyz_to_quaternion_wxyz(action[3:6])
    mapping = DreamTacGripperMapping(gripper_mapping)
    # The recorded training extrema are provenance, not actuator limits.  The
    # upstream diffusion decoder may produce finite values outside that sample
    # support because its normalized output is intentionally not clipped.
    dream_tac_gripper_qpos_bounds(
        mapping, gripper_qpos_min=gripper_qpos_min, gripper_qpos_max=gripper_qpos_max
    )
    if mapping is DreamTacGripperMapping.DIRECT_QPOS:
        converted[:, 7] = np.clip(value[:, 6], CLOSED_GRIPPER_QPOS, OPEN_GRIPPER_QPOS)
        return freeze_array(converted, np.float32)

    # The threshold remains part of the v1 artifact for compatibility. It is
    # unused by DIRECT_QPOS, which keeps continuous qpos semantics after the
    # physical-envelope projection above.
    greater = value[:, 6] > float(gripper_threshold)
    closed = (
        greater if mapping is DreamTacGripperMapping.GREATER_IS_CLOSED else ~greater
    )
    converted[:, 7] = np.where(closed, CLOSED_GRIPPER_QPOS, OPEN_GRIPPER_QPOS)
    return freeze_array(converted, np.float32)


def dream_tac_gripper_qpos_bounds(
    mapping: DreamTacGripperMapping,
    *,
    gripper_qpos_min: float | None,
    gripper_qpos_max: float | None,
) -> tuple[float, float] | None:
    if (gripper_qpos_min is None) != (gripper_qpos_max is None):
        raise ValueError("gripper qpos training range requires both min and max")
    if gripper_qpos_min is None or gripper_qpos_max is None:
        return None
    if mapping is not DreamTacGripperMapping.DIRECT_QPOS:
        raise ValueError("gripper qpos training range requires direct_qpos_v1")
    for value, name in (
        (gripper_qpos_min, "gripper_qpos_min"),
        (gripper_qpos_max, "gripper_qpos_max"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    lower, upper = float(gripper_qpos_min), float(gripper_qpos_max)
    if lower > upper:
        raise ValueError("gripper_qpos_min must not exceed gripper_qpos_max")
    return lower, upper


class OfficialDreamTacPolicy:
    """Full-chunk EE8 adapter around Dream-Tac's official Franka server."""

    def __init__(
        self,
        identity: PolicyIdentity,
        client_factory: Callable[[], DreamTacClient],
        *,
        task_id: str,
        instruction: str,
        experiment_config: str,
        gripper_mapping: DreamTacGripperMapping,
        action_horizon: int = ACTION_HORIZON,
        gripper_threshold: float = GRIPPER_CLOSE_THRESHOLD,
        gripper_qpos_min: float | None = None,
        gripper_qpos_max: float | None = None,
    ) -> None:
        if type(identity) is not PolicyIdentity:
            raise TypeError("identity must be an exact PolicyIdentity")
        if identity.action_spec != EE8_ACTION_SPEC:
            raise ValueError("Dream-Tac requires ee8_absolute")
        if not identity.consumes_tactile or identity.supports_structural_absence:
            raise ValueError("Dream-Tac requires two present tactile payloads")
        for value, name in (
            (task_id, "task_id"),
            (instruction, "instruction"),
            (experiment_config, "experiment_config"),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty string")
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        if isinstance(gripper_threshold, bool) or not isinstance(
            gripper_threshold, (int, float)
        ):
            raise TypeError("gripper_threshold must be a real number")
        if not math.isfinite(float(gripper_threshold)):
            raise ValueError("gripper_threshold must be finite")
        mapping = DreamTacGripperMapping(gripper_mapping)
        qpos_bounds = dream_tac_gripper_qpos_bounds(
            mapping,
            gripper_qpos_min=gripper_qpos_min,
            gripper_qpos_max=gripper_qpos_max,
        )
        self.identity = identity
        self._client_factory = client_factory
        self._task_id = task_id
        self._instruction = instruction
        self._experiment_config = experiment_config
        self._gripper_mapping = mapping
        self._action_horizon = int(action_horizon)
        self._gripper_threshold = float(gripper_threshold)
        self._gripper_qpos_min = None if qpos_bounds is None else qpos_bounds[0]
        self._gripper_qpos_max = None if qpos_bounds is None else qpos_bounds[1]
        self._client: Optional[DreamTacClient] = None
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending: Optional[ActionPlan] = None
        self._continuous_euler: Optional[Array] = None
        self._previous_tactile: Optional[tuple[Array, Array]] = None
        self._tactile_self_attn_gate = dream_tac_tactile_self_attn_gate(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            None,
            None,
        )
        self._closed = False

    def _client_instance(self) -> DreamTacClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("closed Dream-Tac policy cannot reset")
        if type(context) is not PolicyEpisodeContext:
            raise TypeError("context must be an exact PolicyEpisodeContext")
        if (
            context.action_spec != EE8_ACTION_SPEC
            or context.task != self._task_id
            or context.instruction != self._instruction
        ):
            raise ValueError("Dream-Tac reset task/action/prompt contract mismatch")
        self._client_instance().reset(
            experiment_config=self._experiment_config,
            action_horizon=self._action_horizon,
            use_tactile=True,
        )
        self._context = context
        self._pending = None
        self._continuous_euler = None
        self._previous_tactile = None
        self._tactile_self_attn_gate = dream_tac_tactile_self_attn_gate(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            None,
            None,
        )

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        context = self._context
        if context is None:
            raise RuntimeError("Dream-Tac infer requires reset")
        if self._pending is not None:
            raise RuntimeError("Dream-Tac infer requires prior commit")
        if (
            observation.episode_id != context.episode_id
            or observation.task != context.task
            or observation.seed != context.initial_seed
        ):
            raise ValueError("Dream-Tac observation identity mismatch")
        wire = dream_tac_wire_observation(
            observation,
            instruction=self._instruction,
            previous_euler=self._continuous_euler,
            tactile_self_attn_gate=self._tactile_self_attn_gate,
        )
        native = self._client_instance().infer(wire)
        if native.dtype != np.float32 or native.shape != (
            self._action_horizon,
            RAW_ACTION_DIM,
        ):
            raise ValueError(
                "Dream-Tac server action chunk does not match the artifact contract"
            )
        actions = dream_tac_actions_to_ee8(
            native,
            gripper_mapping=self._gripper_mapping,
            gripper_threshold=self._gripper_threshold,
            gripper_qpos_min=self._gripper_qpos_min,
            gripper_qpos_max=self._gripper_qpos_max,
        )
        plan = ActionPlan(EE8_ACTION_SPEC, observation.step_index, actions)
        state = np.asarray(wire["state"], dtype=np.float32)
        self._continuous_euler = freeze_array(state[3:].copy(), np.float32)
        self._previous_tactile = (
            freeze_array(np.asarray(wire["tactile_left"]).copy(), np.uint8),
            freeze_array(np.asarray(wire["tactile_right"]).copy(), np.uint8),
        )
        self._pending = plan
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        if self._pending is None:
            raise RuntimeError("Dream-Tac commit requires pending inference")
        if type(execution) is not PolicyExecution:
            raise TypeError("execution must be an exact PolicyExecution")
        if execution.action_plan_sha256 != self._pending.sha256:
            raise ValueError("Dream-Tac execution does not match action plan")
        expected = self._pending.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("Dream-Tac executed actions differ from its plan")
        if (
            execution.terminal_signal is BackendSignal.RUNNING
            and execution.executed_actions.shape
            != (self._action_horizon, EE_ACTION_DIM)
        ):
            raise RuntimeError(
                "Dream-Tac running commit requires the full action chunk"
            )
        for observation in execution.delivered_observations:
            raw_euler = quaternion_wxyz_to_euler_xyz(observation.proprio[3:7])
            self._continuous_euler = unwrap_euler_xyz(raw_euler, self._continuous_euler)
            left_sensor = observation.sensor("left")
            right_sensor = observation.sensor("right")
            if (
                not left_sensor.payload_present
                or left_sensor.payload is None
                or not right_sensor.payload_present
                or right_sensor.payload is None
            ):
                raise ValueError("Dream-Tac commit requires both tactile payloads")
            previous = self._previous_tactile
            self._tactile_self_attn_gate = dream_tac_tactile_self_attn_gate(
                left_sensor.payload,
                right_sensor.payload,
                None if previous is None else previous[0],
                None if previous is None else previous[1],
            )
            self._previous_tactile = (
                freeze_array(np.asarray(left_sensor.payload).copy(), np.uint8),
                freeze_array(np.asarray(right_sensor.payload).copy(), np.uint8),
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
        self._continuous_euler = None
        self._previous_tactile = None

    def close(self) -> None:
        if self._closed:
            return
        if self._client is not None:
            self._client.close()
        self._client = None
        self._context = None
        self._pending = None
        self._continuous_euler = None
        self._previous_tactile = None
        self._closed = True


__all__ = [
    "ACTION_HORIZON",
    "DreamTacClient",
    "DreamTacGripperMapping",
    "OfficialDreamTacPolicy",
    "dream_tac_actions_to_ee8",
    "dream_tac_gripper_qpos_bounds",
    "dream_tac_tactile_self_attn_gate",
    "dream_tac_wire_observation",
    "euler_xyz_to_quaternion_wxyz",
    "quaternion_wxyz_to_euler_xyz",
    "unwrap_euler_xyz",
]

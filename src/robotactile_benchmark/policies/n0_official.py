"""Official N0-TWAM UniVTAC policy adapter over its websocket protocol."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
from robotactile_benchmark.contracts import (
    Array,
    ObservationRecord,
    freeze_array,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0InputProfile,
    prepare_n0_image,
)
from robotactile_benchmark.transport.n0_official import OfficialN0CommitTransform

NATIVE_ACTION_SHAPE = (20, 2, 12)
SLOTS_PER_FRAME = 12
KEYFRAMES_PER_FRAME = 4

_TRAINING_PROMPTS = {
    "insert_tube": "Insert a tube into a tilted fixture",
    "insert_hole": "Precision peg-in-hole insertion",
    "insert_HDMI": "Insert an HDMI connector into a port",
    "grasp_classify": "Grasp an object and classify it by tactile texture",
    "pull_out_key": "Untwist and extract a key from a lock",
    "lift_can": "Rotate a lying can so it stands upright",
    "lift_bottle": "Grasp and lift a bottle off a surface near a wall",
    "put_bottle_in_shelf": ("Reorient a bottle upright and place it on a shelf"),
}


class OfficialN0PolicyClient(Protocol):
    """Stateful operations required from the official websocket client."""

    def reset(self, *, prompt: str, seed: int) -> None: ...

    def infer(self, observation: Mapping[str, object]) -> Array: ...

    def commit(
        self,
        *,
        video_keyframes: Sequence[Mapping[str, Array]],
        tactile_keyframes: Sequence[Mapping[str, Array]],
        inferred_action: Array,
        native_action: Array,
        action_transform: OfficialN0CommitTransform,
        current_state: Array,
        prompt: str,
    ) -> None: ...

    def discard_terminal(self) -> None: ...

    def close(self) -> None: ...


def n0_training_prompt(task_id: str) -> str:
    """Return the released checkpoint's verbatim prompt for one UniVTAC task."""

    try:
        return _TRAINING_PROMPTS[task_id]
    except KeyError as error:
        raise KeyError(f"official N0 has no UniVTAC prompt for {task_id}") from error


def _quat_wxyz_to_matrix(quaternion: Array) -> Array:
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("EE quaternion must be finite shape (4,)")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("EE quaternion norm is zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: Array) -> Array:
    matrix64 = np.asarray(matrix, dtype=np.float64)
    if matrix64.shape != (3, 3) or not np.isfinite(matrix64).all():
        raise ValueError("rotation matrix must be finite shape (3,3)")
    trace = float(np.trace(matrix64))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix64[2, 1] - matrix64[1, 2]) / scale
        y = (matrix64[0, 2] - matrix64[2, 0]) / scale
        z = (matrix64[1, 0] - matrix64[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix64)))
        if index == 0:
            scale = (
                np.sqrt(1.0 + matrix64[0, 0] - matrix64[1, 1] - matrix64[2, 2]) * 2.0
            )
            w = (matrix64[2, 1] - matrix64[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix64[0, 1] + matrix64[1, 0]) / scale
            z = (matrix64[0, 2] + matrix64[2, 0]) / scale
        elif index == 1:
            scale = (
                np.sqrt(1.0 + matrix64[1, 1] - matrix64[0, 0] - matrix64[2, 2]) * 2.0
            )
            w = (matrix64[0, 2] - matrix64[2, 0]) / scale
            x = (matrix64[0, 1] + matrix64[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix64[1, 2] + matrix64[2, 1]) / scale
        else:
            scale = (
                np.sqrt(1.0 + matrix64[2, 2] - matrix64[0, 0] - matrix64[1, 1]) * 2.0
            )
            w = (matrix64[1, 0] - matrix64[0, 1]) / scale
            x = (matrix64[0, 2] + matrix64[2, 0]) / scale
            y = (matrix64[1, 2] + matrix64[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)


def ee8_to_state20(ee8: object) -> Array:
    """Encode UniVTAC [xyz, quat(wxyz), grip] as N0's rot6d state20."""

    value = np.asarray(ee8)
    if value.dtype != np.float32 or value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError("N0 proprio must be finite float32 shape (8,)")
    rotation = _quat_wxyz_to_matrix(value[3:7])
    state = np.zeros(20, dtype=np.float32)
    state[:3] = value[:3]
    state[3:9] = rotation[:, :2].T.reshape(-1)
    state[9] = value[7]
    return freeze_array(state, np.float32)


def _rot6d10_to_ee8(value: Array) -> Array:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (10,) or not np.isfinite(vector).all():
        raise ValueError("N0 arm action must be finite shape (10,)")
    first = vector[3:6]
    second = vector[6:9]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-8:
        raise ValueError("N0 rot6d first column is degenerate")
    basis_first = first / first_norm
    second = second - float(basis_first @ second) * basis_first
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1e-8:
        raise ValueError("N0 rot6d second column is degenerate")
    basis_second = second / second_norm
    basis_third = np.cross(basis_first, basis_second)
    rotation = np.stack((basis_first, basis_second, basis_third), axis=1)
    result = np.empty(8, dtype=np.float32)
    result[:3] = vector[:3]
    result[3:7] = _matrix_to_quat_wxyz(rotation)
    result[7] = vector[9]
    return result


def native_to_ee8_actions(native: object, *, cold_chunk: bool) -> Array:
    """Convert official [20,2,12] output to executable frame-major EE8 rows."""

    value = np.asarray(native)
    if (
        value.dtype != np.float32
        or value.shape != NATIVE_ACTION_SHAPE
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"N0 native action must be float32 {NATIVE_ACTION_SHAPE}")
    start_frame = 1 if cold_chunk else 0
    rows = [
        _rot6d10_to_ee8(value[:10, frame, slot])
        for frame in range(start_frame, value.shape[1])
        for slot in range(value.shape[2])
    ]
    return freeze_array(np.stack(rows), np.float32)


def _wire_observation(
    observation: ObservationRecord,
    *,
    input_profile: N0InputProfile,
) -> tuple[dict[str, Array], dict[str, Array], Array]:
    if set(observation.vision) != {"top", "wrist_l"}:
        raise ValueError("official N0 vision keys must be top and wrist_l")
    vision = {
        f"observation.images.{key}": prepare_n0_image(
            observation.vision[key],
            profile=input_profile,
            name=f"vision.{key}",
        )
        for key in ("top", "wrist_l")
    }
    tactile: dict[str, Array] = {}
    for slot_id, key in (("left", "tactile_a"), ("right", "tactile_b")):
        sensor = observation.sensor(slot_id)
        if not sensor.payload_present or sensor.payload is None:
            raise ValueError("official N0 requires both tactile streams")
        tactile[f"observation.images.{key}"] = prepare_n0_image(
            sensor.payload,
            profile=input_profile,
            name=f"tactile.{slot_id}",
        )
    return vision, tactile, ee8_to_state20(observation.proprio)


class OfficialN0Policy:
    """First-class adapter for the released UniVTAC-delta checkpoint."""

    def __init__(
        self,
        identity: PolicyIdentity,
        client_factory: Callable[[], OfficialN0PolicyClient],
        *,
        input_profile: N0InputProfile,
    ) -> None:
        if identity.action_spec != EE8_ACTION_SPEC:
            raise ValueError("official N0 policy requires ee8_absolute")
        if not identity.consumes_tactile or identity.supports_structural_absence:
            raise ValueError("official N0 requires tactile without structural absence")
        if type(input_profile) is not N0InputProfile:
            raise TypeError("input_profile must be an exact N0InputProfile")
        self.identity = identity
        self.input_profile = input_profile
        self._client_factory = client_factory
        self._client: Optional[OfficialN0PolicyClient] = None
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending_plan: Optional[ActionPlan] = None
        self._pending_native: Optional[Array] = None
        self._pending_state: Optional[Array] = None
        self._cold_chunk = True
        self._closed = False

    def _client_instance(self) -> OfficialN0PolicyClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("closed official N0 policy cannot reset")
        if context.action_spec != EE8_ACTION_SPEC:
            raise ValueError("official N0 reset action spec mismatch")
        prompt = n0_training_prompt(context.task)
        if context.instruction != prompt:
            raise ValueError("official N0 reset prompt must match training verbatim")
        self._client_instance().reset(prompt=prompt, seed=context.exogenous_seed)
        self._context = context
        self._pending_plan = None
        self._pending_native = None
        self._pending_state = None
        self._cold_chunk = True

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        if self._context is None:
            raise RuntimeError("official N0 infer requires reset")
        if self._pending_plan is not None:
            raise RuntimeError("official N0 infer requires prior commit")
        if (
            observation.episode_id != self._context.episode_id
            or observation.task != self._context.task
            or observation.seed != self._context.initial_seed
        ):
            raise ValueError("official N0 observation identity mismatch")
        vision, tactile, current_state = _wire_observation(
            observation,
            input_profile=self.input_profile,
        )
        prompt = n0_training_prompt(observation.task)
        native = self._client_instance().infer(
            {
                "obs": vision,
                "tactile": tactile,
                "current_state": current_state.tolist(),
                "prompt": prompt,
            }
        )
        actions = native_to_ee8_actions(native, cold_chunk=self._cold_chunk)
        plan = ActionPlan(EE8_ACTION_SPEC, observation.step_index, actions)
        self._pending_plan = plan
        self._pending_native = native
        self._pending_state = current_state
        self._cold_chunk = False
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        context = self._context
        if (
            self._pending_plan is None
            or self._pending_native is None
            or self._pending_state is None
            or context is None
        ):
            raise RuntimeError("official N0 commit requires pending inference")
        if execution.action_plan_sha256 != self._pending_plan.sha256:
            raise ValueError("official N0 execution does not match action plan")
        expected = self._pending_plan.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("official N0 executed action mismatch")
        client = self._client_instance()
        if execution.terminal_signal is not BackendSignal.RUNNING:
            client.discard_terminal()
        else:
            if execution.executed_actions.shape != self._pending_plan.actions.shape:
                raise RuntimeError("official N0 running commit requires a full chunk")
            delivered = execution.delivered_observations
            if len(delivered) != self._pending_plan.actions.shape[0]:
                raise RuntimeError("official N0 grounding observation count mismatch")
            every = SLOTS_PER_FRAME // KEYFRAMES_PER_FRAME
            selected = tuple(
                delivered[index] for index in range(every - 1, len(delivered), every)
            )
            wire = tuple(
                _wire_observation(
                    item,
                    input_profile=self.input_profile,
                )
                for item in selected
            )
            client.commit(
                video_keyframes=tuple(item[0] for item in wire),
                tactile_keyframes=tuple(item[1] for item in wire),
                inferred_action=self._pending_native,
                native_action=self._pending_native,
                action_transform=OfficialN0CommitTransform.IDENTITY,
                current_state=self._pending_state,
                prompt=n0_training_prompt(context.task),
            )
        self._pending_plan = None
        self._pending_native = None
        self._pending_state = None

    def abort(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("abort reason code must be non-empty")
        if self._client is not None:
            self._client.close()
        self._client = None
        self._context = None
        self._pending_plan = None
        self._pending_native = None
        self._pending_state = None

    def close(self) -> None:
        if self._closed:
            return
        if self._client is not None:
            self._client.close()
        self._closed = True
        self._context = None


__all__ = [
    "OfficialN0Policy",
    "OfficialN0PolicyClient",
    "ee8_to_state20",
    "n0_training_prompt",
    "native_to_ee8_actions",
]

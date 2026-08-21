"""Closed-loop N0 policy adapter over the typed stateful client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Optional, Protocol, Tuple, cast

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    ActionPlan,
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.contracts import Array, ObservationRecord, canonical_hash
from robotactile_benchmark.transport.n0_client import N0ClientState
from robotactile_benchmark.transport.n0_codec import native_to_simulator_actions
from robotactile_benchmark.transport.n0_contracts import (
    N0GroundingFrame,
    N0Handshake,
)


class N0PolicyClient(Protocol):
    """Client operations used by the policy without exposing its transport."""

    state: N0ClientState

    @property
    def handshake_identity(self) -> N0Handshake: ...

    def handshake(self) -> None: ...

    def reset(self, episode_id: str, prompt: str, task: str, seed: int) -> None: ...

    def infer(self, step_index: int, observation: Mapping[str, object]) -> Array: ...

    def commit(
        self,
        *,
        step_index: int,
        native_action: Array,
        executed_actions: Array,
        grounding_frames: Tuple[N0GroundingFrame, ...],
    ) -> None: ...

    def terminal(self) -> None: ...

    def abort(self, reason_code: str) -> None: ...

    def close(self) -> None: ...


def _require_rgb(
    value: object, name: str, expected_shape: Tuple[int, int, int]
) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.uint8 or value.shape != expected_shape:
        raise ValueError(f"{name} must be exact uint8 shape {expected_shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_client_identity(
    policy_identity: PolicyIdentity, handshake: N0Handshake
) -> None:
    if handshake.checkpoint_sha256 != policy_identity.checkpoint_sha256:
        raise ValueError("N0 handshake checkpoint identity mismatch")
    if handshake.config_sha256 != policy_identity.config_sha256:
        raise ValueError("N0 handshake config identity mismatch")
    if handshake.action_spec != policy_identity.action_spec:
        raise ValueError("N0 handshake action specification mismatch")


class N0Policy:
    """Tactile-required qpos8 N0 policy with a lazily acquired server lease."""

    def __init__(
        self,
        identity: PolicyIdentity,
        client_factory: Callable[[], N0PolicyClient],
    ) -> None:
        if identity.action_spec != ACTION_SPEC:
            raise ValueError("N0 policy requires qpos8_next_step")
        if not identity.consumes_tactile or identity.supports_structural_absence:
            raise ValueError(
                "frozen N0 requires tactile and rejects structural absence"
            )
        self.identity = identity
        self._client_factory = client_factory
        self._client: Optional[N0PolicyClient] = None
        self._context: Optional[PolicyEpisodeContext] = None
        self._pending_plan: Optional[ActionPlan] = None
        self._pending_native: Optional[Array] = None
        self._closed = False

    def _get_client(self) -> N0PolicyClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def reset(self, context: PolicyEpisodeContext) -> None:
        if self._closed:
            raise RuntimeError("closed N0 policy cannot reset")
        if type(context) is not PolicyEpisodeContext:
            raise TypeError("context must be an exact PolicyEpisodeContext")
        if context.action_spec != self.identity.action_spec:
            raise ValueError("context action spec does not match N0 identity")
        client = self._get_client()
        if client.state is N0ClientState.CONNECTED:
            client.handshake()
        _validate_client_identity(self.identity, client.handshake_identity)
        client.reset(
            context.episode_id,
            context.instruction,
            context.task,
            context.initial_seed,
        )
        self._context = context
        self._pending_plan = None
        self._pending_native = None

    def _encode(self, observation: ObservationRecord) -> Mapping[str, object]:
        if type(observation) is not ObservationRecord:
            raise TypeError("N0 infer accepts only exact ObservationRecord")
        if self._context is None:
            raise RuntimeError("N0 infer requires reset")
        if (
            observation.episode_id != self._context.episode_id
            or observation.task != self._context.task
            or observation.seed != self._context.initial_seed
        ):
            raise ValueError("N0 observation identity does not match reset context")
        if observation.proprio.shape != (8,):
            raise ValueError(
                "N0 proprio must have exact shape (8,); slicing is forbidden"
            )
        if observation.proprio.dtype != np.float32:
            raise TypeError("N0 proprio must use exact float32")
        if not np.isfinite(observation.proprio).all():
            raise ValueError("N0 proprio must be finite")
        if set(observation.vision) != {"top", "wrist_l"}:
            raise ValueError("N0 vision keys must be exactly top and wrist_l")
        handshake = self._get_client().handshake_identity
        vision = {
            name: _require_rgb(
                observation.vision[name], f"vision {name}", expected_shape
            )
            for name, expected_shape in zip(
                handshake.camera_keys, handshake.camera_shapes
            )
        }
        tactile: dict[str, Array] = {}
        for (slot_id, key), expected_shape in zip(
            (("left", "tactile_a"), ("right", "tactile_b")),
            handshake.tactile_shapes,
        ):
            sensor = observation.sensor(slot_id)
            if not sensor.payload_present or sensor.payload is None:
                raise ValueError(
                    "frozen N0 does not support structural tactile absence"
                )
            tactile[key] = _require_rgb(
                sensor.payload, f"tactile {slot_id}", expected_shape
            )
        encoded: dict[str, object] = {
            "vision": vision,
            "tactile": tactile,
            "proprio": observation.proprio,
        }
        return {**encoded, "observation_digest": canonical_hash(encoded)}

    def _grounding_frame(self, observation: ObservationRecord) -> N0GroundingFrame:
        """Route one delivered model-visible observation into commit authority."""

        encoded = self._encode(observation)
        vision = encoded["vision"]
        tactile = encoded["tactile"]
        if not isinstance(vision, Mapping) or not isinstance(tactile, Mapping):
            raise TypeError("N0 encoded grounding modalities must be mappings")
        return N0GroundingFrame(
            step_index=observation.step_index,
            top=vision["top"],
            wrist_l=vision["wrist_l"],
            tactile_a=tactile["tactile_a"],
            tactile_b=tactile["tactile_b"],
            proprio=cast(Array, encoded["proprio"]),
        )

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        if self._pending_plan is not None:
            raise RuntimeError("N0 infer requires the prior execution to be committed")
        payload = self._encode(observation)
        native = self._get_client().infer(observation.step_index, payload)
        actions = native_to_simulator_actions(native)
        plan = ActionPlan(ACTION_SPEC, observation.step_index, actions)
        self._pending_native = native
        self._pending_plan = plan
        return plan

    def commit(self, execution: PolicyExecution) -> None:
        if self._pending_plan is None or self._pending_native is None:
            raise RuntimeError("N0 commit requires one pending inference")
        if type(execution) is not PolicyExecution:
            raise TypeError("execution must be an exact PolicyExecution")
        if execution.action_plan_sha256 != self._pending_plan.sha256:
            raise ValueError("N0 execution does not match the pending action plan")
        expected = self._pending_plan.actions[: execution.executed_actions.shape[0]]
        if not np.array_equal(execution.executed_actions, expected):
            raise ValueError("N0 executed actions do not match the pending plan")
        client = self._get_client()
        if execution.terminal_signal is not BackendSignal.RUNNING:
            client.terminal()
        else:
            if (
                execution.executed_actions.shape != (8, 8)
                or len(execution.delivered_observations) != 8
            ):
                raise RuntimeError(
                    "N0 running execution requires the complete eight-action chunk"
                )
            expected_steps = tuple(
                range(
                    self._pending_plan.source_step_index + 1,
                    self._pending_plan.source_step_index + 9,
                )
            )
            delivered_steps = tuple(
                observation.step_index
                for observation in execution.delivered_observations
            )
            if delivered_steps != expected_steps:
                raise ValueError("N0 running delivered observation steps must be dense")
            grounding_frames = tuple(
                self._grounding_frame(observation)
                for observation in execution.delivered_observations
            )
            client.commit(
                step_index=self._pending_plan.source_step_index,
                native_action=self._pending_native,
                executed_actions=execution.executed_actions,
                grounding_frames=grounding_frames,
            )
        self._pending_plan = None
        self._pending_native = None

    def abort(self, reason_code: str) -> None:
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("abort reason code must be non-empty")
        if self._client is not None:
            self._client.abort(reason_code)
        self._pending_plan = None
        self._pending_native = None
        self._context = None

    def close(self) -> None:
        if self._closed:
            return
        if self._client is not None:
            self._client.close()
        self._pending_plan = None
        self._pending_native = None
        self._context = None
        self._closed = True

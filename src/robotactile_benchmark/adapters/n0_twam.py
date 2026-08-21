"""Typed qpos8 request contract for N0-TWAM Track 3.1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from robotactile_benchmark.adapters.n0_contracts import (
    AdapterStateError,
    N0Handshake,
    UnsupportedAvailabilityError,
    canonical_rgb,
    require_integer,
)
from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    array_sha256,
    canonical_hash,
)


class _State(str, Enum):
    NEW = "new"
    RESET_PENDING_ACK = "reset_pending_ack"
    READY = "ready"
    INFER_PENDING_ACK = "infer_pending_ack"
    AWAITING_COMMIT = "awaiting_commit"
    COMMIT_PENDING_ACK = "commit_pending_ack"
    ABORTED = "aborted"


class N0TwamQpos8Adapter:
    """Build N0 requests while enforcing the stateful cache transaction."""

    def __init__(self, handshake: N0Handshake) -> None:
        if handshake.action_spec != "qpos8_next_step":
            raise ValueError("N0 action spec must be qpos8_next_step")
        if handshake.action_dim != 8 or handshake.action_per_frame != 4:
            raise ValueError(
                "N0 qpos8 handshake must declare dim=8 and action_per_frame=4"
            )
        if handshake.frame_chunk_size != 2:
            raise ValueError("N0 Track 3.1 serve contract requires frame_chunk_size=2")
        expected_camera_keys = (
            "observation.images.top",
            "observation.images.wrist_l",
        )
        expected_tactile_keys = (
            "observation.images.tactile_a",
            "observation.images.tactile_b",
        )
        if handshake.camera_keys != expected_camera_keys:
            raise ValueError("N0 camera keys must match the Track 3.1 serve config")
        if handshake.tactile_keys != expected_tactile_keys:
            raise ValueError("N0 tactile keys must match the Track 3.1 serve config")
        self._handshake = handshake
        self._action_lower_bounds = np.asarray(
            handshake.action_lower_bounds, dtype=np.float32
        )[:, None, None]
        self._action_upper_bounds = np.asarray(
            handshake.action_upper_bounds, dtype=np.float32
        )[:, None, None]
        self._state = _State.NEW
        self._episode_id = ""
        self._infer_step_index: Optional[int] = None
        self._pending_transaction_id: Optional[str] = None
        self._pending_request_sha256: Optional[str] = None
        self._pending_action_sha256: Optional[str] = None
        self._pending_infer_anchor: Optional[Array] = None
        self._pending_commit_last_step: Optional[int] = None
        self._episode_task = ""
        self._episode_seed: Optional[int] = None
        self._prompt_sha256 = ""
        self._last_committed_step: Optional[int] = None
        self._transaction_counter = 0

    def _start_transaction(self, operation: str, step_index: Optional[int]) -> str:
        self._transaction_counter += 1
        transaction_id = canonical_hash(
            {
                "episode_id": self._episode_id,
                "counter": self._transaction_counter,
                "operation": operation,
                "step_index": step_index,
            }
        )
        self._pending_transaction_id = transaction_id
        return transaction_id

    @staticmethod
    def success_response(request: Mapping[str, Any], **payload: Any) -> Dict[str, Any]:
        """Build the typed ACK shape expected from the patched N0 server."""

        unsigned_request = {
            key: value for key, value in request.items() if key != "request_sha256"
        }
        request_sha256 = request.get("request_sha256")
        if request_sha256 != canonical_hash(unsigned_request):
            raise AdapterStateError("request content no longer matches its digest")
        return {
            "status": "ok",
            "op": request["op"],
            "transaction_id": request["transaction_id"],
            "episode_id": request["episode_id"],
            "request_sha256": request_sha256,
            **payload,
        }

    def _acknowledge(self, response: Mapping[str, Any], operation: str) -> None:
        if response.get("status") != "ok":
            raise AdapterStateError(f"{operation} server acknowledgement is not ok")
        if response.get("op") != operation:
            raise AdapterStateError(f"{operation} acknowledgement has the wrong op")
        if response.get("transaction_id") != self._pending_transaction_id:
            raise AdapterStateError(
                f"{operation} acknowledgement has the wrong transaction ID"
            )
        if response.get("episode_id") != self._episode_id:
            raise AdapterStateError(
                f"{operation} acknowledgement has the wrong episode ID"
            )
        if response.get("request_sha256") != self._pending_request_sha256:
            raise AdapterStateError(
                f"{operation} acknowledgement has the wrong request digest"
            )

    def _request_envelope(
        self, operation: str, step_index: Optional[int]
    ) -> Dict[str, Any]:
        transaction_id = self._start_transaction(operation, step_index)
        return {
            "schema_version": self._handshake.schema_version,
            "op": operation,
            "transaction_id": transaction_id,
            "episode_id": self._episode_id,
            "step_id": step_index,
        }

    def _bind_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Bind an ACK to the exact typed request, including array contents."""

        request_sha256 = canonical_hash(request)
        self._pending_request_sha256 = request_sha256
        return {**request, "request_sha256": request_sha256}

    def reset(
        self, episode_id: str, prompt: str, seed: int, task: str = "unspecified"
    ) -> Dict[str, Any]:
        if self._state in {
            _State.RESET_PENDING_ACK,
            _State.INFER_PENDING_ACK,
            _State.AWAITING_COMMIT,
            _State.COMMIT_PENDING_ACK,
        }:
            raise AdapterStateError(
                "reset cannot discard an unacknowledged transaction; fail it first"
            )
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        if not prompt or not task:
            raise ValueError("prompt and task must be non-empty")
        seed = require_integer(seed, "seed")
        if seed > 0x7FFFFFFF:
            raise ValueError("seed must fit N0-TWAM's signed 31-bit episode seed")
        self._episode_id = episode_id
        self._episode_task = task
        self._episode_seed = seed
        self._prompt_sha256 = canonical_hash(prompt)
        self._infer_step_index = None
        self._last_committed_step = None
        self._pending_action_sha256 = None
        self._pending_infer_anchor = None
        self._pending_commit_last_step = None
        self._state = _State.RESET_PENDING_ACK
        return self._bind_request(
            {
                **self._request_envelope("reset", None),
                "reset": True,
                "prompt": prompt,
                "prompt_sha256": self._prompt_sha256,
                "task": task,
                "seed": seed,
            }
        )

    def ack_reset(self, response: Mapping[str, Any]) -> None:
        """Accept a typed reset ACK before the first inference request."""

        if self._state is not _State.RESET_PENDING_ACK:
            raise AdapterStateError("reset acknowledgement requires a pending reset")
        self._acknowledge(response, "reset")
        self._pending_transaction_id = None
        self._pending_request_sha256 = None
        self._state = _State.READY

    def fail_reset(self) -> None:
        """Abort an episode whose server-side reset did not complete."""

        if self._state is not _State.RESET_PENDING_ACK:
            raise AdapterStateError("reset failure requires a pending reset")
        self._state = _State.ABORTED

    def _policy_payload(
        self, record: EvaluationRecord
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        if record.observation.episode_id != self._episode_id:
            raise AdapterStateError("record episode does not match the reset episode")
        if record.observation.task != self._episode_task:
            raise AdapterStateError("record task does not match the reset task")
        if record.observation.seed != self._episode_seed:
            raise AdapterStateError("record seed does not match the reset seed")
        vision = {
            server_key: canonical_rgb(
                record.observation.vision[local_key], f"vision {local_key}"
            )
            for local_key, server_key in zip(
                ("top", "wrist_l"), self._handshake.camera_keys
            )
        }
        tactile: Dict[str, Array] = {}
        for slot_id, key in zip(("left", "right"), self._handshake.tactile_keys):
            sensor = record.observation.sensor(slot_id)
            if not sensor.payload_present or sensor.payload is None:
                raise UnsupportedAvailabilityError(
                    "frozen N0-TWAM requires both tactile payloads; absence is unsupported"
                )
            tactile[key] = canonical_rgb(sensor.payload, f"tactile {slot_id}")
        return vision, tactile

    def build_infer(self, record: EvaluationRecord) -> Dict[str, Any]:
        if self._state is not _State.READY:
            raise AdapterStateError("infer requires reset or a completed commit")
        vision, tactile = self._policy_payload(record)
        if (
            self._last_committed_step is not None
            and record.observation.step_index != self._last_committed_step
        ):
            raise AdapterStateError(
                "next infer must use the last committed observation step"
            )
        self._infer_step_index = record.observation.step_index
        anchor = np.asarray(record.observation.proprio[:8], dtype=np.float32).copy()
        anchor.setflags(write=False)
        self._pending_infer_anchor = anchor
        self._state = _State.INFER_PENDING_ACK
        return self._bind_request(
            {
                **self._request_envelope("infer", record.observation.step_index),
                "obs": vision,
                "tactile": tactile,
                "current_state": anchor,
            }
        )

    def decode_action(self, action: Array) -> Array:
        array = np.asarray(action, dtype=np.float32)
        expected = (
            self._handshake.action_dim,
            self._handshake.frame_chunk_size,
            self._handshake.action_per_frame,
        )
        if array.shape != expected:
            raise ValueError("N0 qpos8 action must have shape [8, F, 4]")
        if not np.isfinite(array).all():
            raise ValueError("N0 qpos8 action must be finite")
        if np.any(array < self._action_lower_bounds) or np.any(
            array > self._action_upper_bounds
        ):
            raise ValueError("N0 qpos8 action exceeds the frozen physical bounds")
        return np.asarray(array.copy(), dtype=np.float32)

    def ack_infer(self, response: Mapping[str, Any]) -> Array:
        """Validate the typed infer response and bind its qpos8 action."""

        if self._state is not _State.INFER_PENDING_ACK:
            raise AdapterStateError("infer acknowledgement requires a pending request")
        self._acknowledge(response, "infer")
        if "action" not in response:
            raise AdapterStateError("infer acknowledgement omitted the action")
        action = self.decode_action(np.asarray(response["action"]))
        self._pending_action_sha256 = array_sha256(action)
        self._pending_transaction_id = None
        self._pending_request_sha256 = None
        self._state = _State.AWAITING_COMMIT
        return action

    def build_commit(
        self, action: Array, keyframes: Sequence[EvaluationRecord]
    ) -> Dict[str, Any]:
        if self._state is not _State.AWAITING_COMMIT:
            raise AdapterStateError("commit requires exactly one preceding infer")
        if len(keyframes) < self._handshake.minimum_commit_keyframes:
            raise ValueError(
                "commit requires at least "
                f"{self._handshake.minimum_commit_keyframes} executed keyframes"
            )
        step_indices = tuple(record.observation.step_index for record in keyframes)
        if any(right <= left for left, right in zip(step_indices, step_indices[1:])):
            raise ValueError("commit keyframe indices must be strictly increasing")
        if self._infer_step_index is None or step_indices[0] <= self._infer_step_index:
            raise ValueError("commit keyframes must be captured after infer")
        decoded = self.decode_action(action)
        if array_sha256(decoded) != self._pending_action_sha256:
            raise AdapterStateError(
                "commit action does not match the acknowledged infer"
            )
        observations = []
        tactile_frames = []
        for record in keyframes:
            vision, tactile = self._policy_payload(record)
            observations.append(vision)
            tactile_frames.append(tactile)
        self._state = _State.COMMIT_PENDING_ACK
        if self._pending_infer_anchor is None:
            raise AdapterStateError("commit lost the infer anchor state")
        self._pending_commit_last_step = step_indices[-1]
        decoded.setflags(write=False)
        return self._bind_request(
            {
                **self._request_envelope("commit", step_indices[-1]),
                "compute_kv_cache": True,
                "obs": observations,
                "tactile": tactile_frames,
                "state": decoded,
                "current_state": self._pending_infer_anchor,
                "action_anchor_state": self._pending_infer_anchor,
                "action_format": "qpos8_next_step",
            }
        )

    def ack_commit(self, response: Mapping[str, Any]) -> None:
        """Mark one server-confirmed cache commit as complete."""

        if self._state is not _State.COMMIT_PENDING_ACK:
            raise AdapterStateError("commit acknowledgement requires a pending request")
        self._acknowledge(response, "commit")
        self._infer_step_index = None
        self._last_committed_step = self._pending_commit_last_step
        self._pending_commit_last_step = None
        self._pending_transaction_id = None
        self._pending_request_sha256 = None
        self._pending_action_sha256 = None
        self._pending_infer_anchor = None
        self._state = _State.READY

    def fail_infer(self) -> None:
        """Invalidate the episode after an infer transport or server failure."""

        if self._state is not _State.INFER_PENDING_ACK:
            raise AdapterStateError("infer failure requires an outstanding request")
        self._state = _State.ABORTED

    def fail_commit(self) -> None:
        """Invalidate the episode after a failed non-idempotent cache commit."""

        if self._state is not _State.COMMIT_PENDING_ACK:
            raise AdapterStateError("commit failure requires a pending request")
        self._state = _State.ABORTED

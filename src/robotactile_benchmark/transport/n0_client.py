"""Typed stateful N0 client with fail-closed transaction binding."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from enum import Enum
from typing import Optional, Protocol, Tuple, cast

from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.transport.n0_binding import RequestBinding
from robotactile_benchmark.transport.n0_codec import native_to_simulator_actions
from robotactile_benchmark.transport.n0_commit import (
    N0CommitAuthority,
    N0PreparedCommit,
)
from robotactile_benchmark.transport.n0_contracts import (
    SCHEMA_VERSION,
    N0GroundingFrame,
    N0Handshake,
    require_nonnegative_int,
    require_string,
    validate_action_bounds,
    validate_rpc_ack,
)
from robotactile_benchmark.transport.n0_modalities import (
    deep_freeze_array,
    require_qpos_anchor,
    validate_encoded_observation,
    validate_grounding_frame_shapes,
)


class N0Transport(Protocol):
    """The minimal request/reply boundary implemented by a live gateway."""

    def connect(self) -> Mapping[str, object]: ...

    def call(self, request: Mapping[str, object]) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class N0ClientState(str, Enum):
    CONNECTED = "connected"
    HANDSHAKEN = "handshaken"
    RESET_REQUIRED = "reset_required"
    RESET_PENDING = "reset_pending"
    READY = "ready"
    INFER_PENDING = "infer_pending"
    AWAITING_EXECUTION = "awaiting_execution"
    PREPARE_COMMIT_PENDING = "prepare_commit_pending"
    FINALIZE_COMMIT_PENDING = "finalize_commit_pending"
    INDETERMINATE = "indeterminate"
    CLOSED = "closed"


class N0Client:
    """One-session N0 cache transaction with commit-at-most-once semantics."""

    def __init__(
        self,
        transport: N0Transport,
        *,
        expected_source_commit: str,
        expected_checkpoint_sha256: str,
        expected_config_sha256: str,
        expected_normalizer_sha256: str,
        expected_serve_bundle_sha256: str,
        expected_prompt_manifest_sha256: str,
    ) -> None:
        self._transport = transport
        self._expected = {
            "expected_source_commit": expected_source_commit,
            "expected_checkpoint_sha256": expected_checkpoint_sha256,
            "expected_config_sha256": expected_config_sha256,
            "expected_normalizer_sha256": expected_normalizer_sha256,
            "expected_serve_bundle_sha256": expected_serve_bundle_sha256,
            "expected_prompt_manifest_sha256": expected_prompt_manifest_sha256,
        }
        self.state = N0ClientState.CONNECTED
        self._handshake: Optional[N0Handshake] = None
        self._episode_id = ""
        self._request_counter = 0
        self._transaction_counter = 0
        self._reset_generation: Optional[int] = None
        self._cache_epoch = ""
        self._cache_position = 0
        self._pending_native: Optional[Array] = None
        self._pending_anchor: Optional[Array] = None
        self._pending_infer_transaction_id: Optional[str] = None
        self._pending_step: Optional[int] = None
        self._next_infer_step = 0

    @property
    def handshake_identity(self) -> N0Handshake:
        """Expose only the validated immutable server identity."""

        if self._handshake is None:
            raise RuntimeError("N0 client has not completed handshake")
        return self._handshake

    def handshake(self) -> None:
        if self.state is not N0ClientState.CONNECTED:
            raise RuntimeError("N0 handshake is allowed exactly once")
        try:
            document = self._transport.connect()
            self._handshake = N0Handshake.from_mapping(document, **self._expected)
        except Exception:
            self.state = N0ClientState.CLOSED
            with suppress(Exception):
                self._transport.close()
            raise
        self.state = N0ClientState.HANDSHAKEN
        self.state = N0ClientState.RESET_REQUIRED

    def _identity(self, operation: str, step_index: int) -> dict[str, object]:
        handshake = self.handshake_identity
        self._request_counter += 1
        self._transaction_counter += 1
        request_id = canonical_hash(
            {
                "namespace": "robotactile.n0.request.v1",
                "server_epoch": handshake.server_epoch,
                "counter": self._request_counter,
            }
        )
        transaction_id = canonical_hash(
            {
                "namespace": "robotactile.n0.transaction.v1",
                "episode_id": self._episode_id,
                "operation": operation,
                "step_index": step_index,
                "counter": self._transaction_counter,
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "op": operation,
            "request_id": request_id,
            "episode_id": self._episode_id,
            "step_index": step_index,
            "transaction_id": transaction_id,
            "server_epoch": handshake.server_epoch,
        }

    def _binding(
        self, operation: str, step_index: int, payload: Mapping[str, object]
    ) -> RequestBinding:
        identity = self._identity(operation, step_index)
        if set(identity) & set(payload):
            raise RuntimeError("N0 payload may not override transaction identity")
        return RequestBinding.from_unsigned({**identity, **payload})

    def _call(self, binding: RequestBinding) -> Mapping[str, object]:
        request = binding.wire_request()
        response = self._transport.call(request)
        binding.validate_returned_request(request)
        return response

    def _clear_pending(self) -> None:
        self._pending_native = None
        self._pending_anchor = None
        self._pending_infer_transaction_id = None
        self._pending_step = None

    def _accept_reset_ack(
        self, binding: RequestBinding, response: Mapping[str, object]
    ) -> tuple[int, str]:
        validate_rpc_ack(
            binding.snapshot,
            response,
            ("reset_generation", "cache_epoch", "cache_position"),
        )
        generation = require_nonnegative_int(
            response["reset_generation"], "reset_generation"
        )
        cache_epoch = require_string(response["cache_epoch"], "cache_epoch")
        position = require_nonnegative_int(response["cache_position"], "cache_position")
        if position != 0:
            raise RuntimeError("N0 reset acknowledgement cache position must be zero")
        if self._cache_epoch and cache_epoch == self._cache_epoch:
            raise RuntimeError("N0 reset must return a new cache epoch")
        if self._reset_generation is not None and generation <= self._reset_generation:
            raise RuntimeError("N0 reset generation must strictly increase")
        return generation, cache_epoch

    def reset(self, episode_id: str, prompt: str, task: str, seed: int) -> None:
        if self.state not in {
            N0ClientState.RESET_REQUIRED,
            N0ClientState.INDETERMINATE,
        }:
            raise RuntimeError(
                "N0 reset requires reset_required or indeterminate state"
            )
        self._episode_id = require_string(episode_id, "episode_id")
        prompt = require_string(prompt, "prompt")
        task = require_string(task, "task")
        seed = require_nonnegative_int(seed, "seed")
        operation = (
            "hard_reset" if self.state is N0ClientState.INDETERMINATE else "reset"
        )
        self.state = N0ClientState.RESET_PENDING
        try:
            binding = self._binding(
                operation,
                0,
                {
                    "prompt": prompt,
                    "prompt_sha256": canonical_hash(prompt),
                    "task": task,
                    "seed": seed,
                },
            )
            response = self._call(binding)
            generation, cache_epoch = self._accept_reset_ack(binding, response)
        except Exception:
            self.state = N0ClientState.INDETERMINATE
            raise
        self._reset_generation = generation
        self._cache_epoch = cache_epoch
        self._cache_position = 0
        self._clear_pending()
        self._next_infer_step = 0
        self.state = N0ClientState.READY

    def infer(self, step_index: int, observation: Mapping[str, object]) -> Array:
        if self.state is not N0ClientState.READY:
            raise RuntimeError("N0 infer requires ready state")
        step_index = require_nonnegative_int(step_index, "step_index")
        if step_index != self._next_infer_step:
            raise ValueError(
                f"N0 infer step must equal the next decision step {self._next_infer_step}"
            )
        if not isinstance(observation, Mapping) or not observation:
            raise ValueError("N0 observation payload must be a non-empty mapping")
        validate_encoded_observation(observation, self.handshake_identity)
        anchor = require_qpos_anchor(observation)
        self.state = N0ClientState.INFER_PENDING
        try:
            binding = self._binding(
                "infer",
                step_index,
                {
                    "observation": observation,
                    "reset_generation": self._reset_generation,
                    "cache_epoch": self._cache_epoch,
                    "cache_position": self._cache_position,
                },
            )
            response = self._call(binding)
            validate_rpc_ack(
                binding.snapshot,
                response,
                (
                    "native_action",
                    "native_action_sha256",
                    "pre_commit_cache_position",
                ),
            )
            native = cast(Array, response["native_action"])
            simulator = native_to_simulator_actions(native)
            if response["native_action_sha256"] != canonical_hash(native):
                raise RuntimeError("N0 native action digest mismatch")
            position = require_nonnegative_int(
                response["pre_commit_cache_position"],
                "pre_commit_cache_position",
            )
            if position != self._cache_position:
                raise RuntimeError(
                    "N0 infer reported a stale or advanced cache position"
                )
            validate_action_bounds(simulator, self.handshake_identity)
        except Exception:
            self.state = N0ClientState.RESET_REQUIRED
            raise
        self._pending_native = deep_freeze_array(native)
        self._pending_anchor = anchor
        self._pending_infer_transaction_id = cast(
            str, binding.snapshot["transaction_id"]
        )
        self._pending_step = step_index
        self.state = N0ClientState.AWAITING_EXECUTION
        return self._pending_native

    def commit(
        self,
        *,
        step_index: int,
        native_action: Array,
        executed_actions: Array,
        grounding_frames: Tuple[N0GroundingFrame, ...],
    ) -> None:
        if self.state is not N0ClientState.AWAITING_EXECUTION:
            raise RuntimeError("N0 commit requires one pending inference")
        if (
            self._pending_native is None
            or self._pending_anchor is None
            or self._pending_infer_transaction_id is None
            or self._pending_step is None
            or self._reset_generation is None
        ):
            raise RuntimeError("N0 pending transaction authority is unavailable")
        if require_nonnegative_int(step_index, "step_index") != self._pending_step:
            raise ValueError("N0 commit step does not match infer")
        frames = tuple(grounding_frames)
        validate_grounding_frame_shapes(frames, self.handshake_identity)
        authority = N0CommitAuthority.build(
            pending_action=self._pending_native,
            native_action=native_action,
            executed_actions=executed_actions,
            grounding_frames=frames,
            infer_qpos_anchor=self._pending_anchor,
            source_step=self._pending_step,
        )
        self.state = N0ClientState.PREPARE_COMMIT_PENDING
        try:
            prepare_binding = self._binding(
                "prepare_commit",
                step_index,
                authority.prepare_payload(
                    infer_transaction_id=self._pending_infer_transaction_id,
                    reset_generation=self._reset_generation,
                    cache_epoch=self._cache_epoch,
                    cache_position=self._cache_position,
                ),
            )
            prepare_response = self._call(prepare_binding)
            prepared = N0PreparedCommit.accept(
                prepare_binding, prepare_response, self._cache_position
            )
        except Exception:
            self.state = N0ClientState.RESET_REQUIRED
            raise
        self.state = N0ClientState.FINALIZE_COMMIT_PENDING
        try:
            finalize_binding = self._binding(
                "finalize_commit",
                step_index,
                prepared.finalize_payload(
                    reset_generation=self._reset_generation,
                    cache_epoch=self._cache_epoch,
                    cache_position=self._cache_position,
                ),
            )
            finalize_response = self._call(finalize_binding)
            new_position = prepared.accept_finalize(
                finalize_binding, finalize_response, self._cache_position
            )
        except Exception:
            self.state = N0ClientState.INDETERMINATE
            raise
        self._cache_position = new_position
        self._next_infer_step = self._pending_step + 8
        self._clear_pending()
        self.state = N0ClientState.READY

    def terminal(self) -> None:
        if self.state is not N0ClientState.AWAITING_EXECUTION:
            raise RuntimeError("N0 terminal discard requires a pending inference")
        self._clear_pending()
        self._next_infer_step = 0
        self.state = N0ClientState.RESET_REQUIRED

    def abort(self, reason_code: str) -> None:
        if self.state in {N0ClientState.CONNECTED, N0ClientState.CLOSED}:
            return
        if self.state is N0ClientState.INDETERMINATE:
            return
        reason = require_string(reason_code, "reason_code")
        pending_step = 0 if self._pending_step is None else self._pending_step
        try:
            binding = self._binding("abort", pending_step, {"reason_code": reason})
            response = self._call(binding)
            generation, cache_epoch = self._accept_reset_ack(binding, response)
        except Exception:
            self.state = N0ClientState.INDETERMINATE
            raise
        self._reset_generation = generation
        self._cache_epoch = cache_epoch
        self._cache_position = 0
        self._next_infer_step = 0
        self._clear_pending()
        self.state = N0ClientState.RESET_REQUIRED

    def close(self) -> None:
        if self.state is N0ClientState.CLOSED:
            return
        self._transport.close()
        self._clear_pending()
        self.state = N0ClientState.CLOSED

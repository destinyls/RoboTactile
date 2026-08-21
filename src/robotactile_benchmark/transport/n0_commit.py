"""Validated grounding authority for one complete N0 cache commit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.transport.n0_binding import RequestBinding
from robotactile_benchmark.transport.n0_codec import native_to_simulator_actions
from robotactile_benchmark.transport.n0_contracts import (
    N0GroundingFrame,
    require_nonnegative_int,
    require_sha256,
    validate_rpc_ack,
)
from robotactile_benchmark.transport.n0_modalities import deep_freeze_array

_AUTHORITY_FIELDS = (
    "infer_transaction_id",
    "pending_action_sha256",
    "executed_actions_sha256",
    "grounding_frames_sha256",
    "infer_qpos_anchor_sha256",
)


def _require_anchor(value: object) -> Array:
    if not isinstance(value, np.ndarray):
        raise TypeError("N0 infer qpos anchor must be a numpy array")
    if value.dtype != np.float32 or value.shape != (8,):
        raise ValueError("N0 infer qpos anchor must be exact float32 shape (8,)")
    if not np.isfinite(value).all():
        raise ValueError("N0 infer qpos anchor must be finite")
    return deep_freeze_array(value)


@dataclass(frozen=True)
class N0CommitAuthority:
    """Deep-frozen action, observations, and infer anchor for one commit."""

    pending_action: Array
    executed_actions: Array
    grounding_frames: Tuple[N0GroundingFrame, ...]
    infer_qpos_anchor: Array
    pending_action_sha256: str
    executed_actions_sha256: str
    grounding_frames_sha256: str
    infer_qpos_anchor_sha256: str

    @classmethod
    def build(
        cls,
        *,
        pending_action: Array,
        native_action: Array,
        executed_actions: Array,
        grounding_frames: Tuple[N0GroundingFrame, ...],
        infer_qpos_anchor: Array,
        source_step: int,
    ) -> "N0CommitAuthority":
        expected = native_to_simulator_actions(pending_action)
        native_to_simulator_actions(native_action)
        if not np.array_equal(native_action, pending_action):
            raise ValueError("N0 commit native action does not match pending infer")
        if (
            not isinstance(executed_actions, np.ndarray)
            or executed_actions.dtype != np.float32
            or executed_actions.shape != (8, 8)
            or not np.isfinite(executed_actions).all()
            or not np.array_equal(executed_actions, expected)
        ):
            raise ValueError(
                "N0 running commit requires complete exact float32 actions"
            )
        frames = tuple(grounding_frames)
        if len(frames) != 8 or any(
            type(frame) is not N0GroundingFrame for frame in frames
        ):
            raise ValueError("N0 running commit requires eight exact grounding frames")
        expected_steps = tuple(range(source_step + 1, source_step + 9))
        if tuple(frame.step_index for frame in frames) != expected_steps:
            raise ValueError("N0 grounding frame steps must be dense after infer")
        pending = deep_freeze_array(pending_action)
        executed = deep_freeze_array(executed_actions)
        anchor = _require_anchor(infer_qpos_anchor)
        wire_frames = tuple(frame.to_wire() for frame in frames)
        return cls(
            pending_action=pending,
            executed_actions=executed,
            grounding_frames=frames,
            infer_qpos_anchor=anchor,
            pending_action_sha256=canonical_hash(pending),
            executed_actions_sha256=canonical_hash(executed),
            grounding_frames_sha256=canonical_hash(wire_frames),
            infer_qpos_anchor_sha256=canonical_hash(anchor),
        )

    def prepare_payload(
        self,
        *,
        infer_transaction_id: str,
        reset_generation: int,
        cache_epoch: str,
        cache_position: int,
    ) -> dict[str, object]:
        """Build exact authority fields consumed by the N0 gateway."""

        return {
            "reset_generation": reset_generation,
            "cache_epoch": cache_epoch,
            "pre_commit_cache_position": cache_position,
            "infer_transaction_id": infer_transaction_id,
            "pending_action_sha256": self.pending_action_sha256,
            "executed_actions": self.executed_actions,
            "executed_actions_sha256": self.executed_actions_sha256,
            "grounding_frames": tuple(
                frame.to_wire() for frame in self.grounding_frames
            ),
            "grounding_frames_sha256": self.grounding_frames_sha256,
            "infer_qpos_anchor": self.infer_qpos_anchor,
            "infer_qpos_anchor_sha256": self.infer_qpos_anchor_sha256,
        }


def prepare_token_for_request(request: Mapping[str, object]) -> str:
    """Derive the exact one-use server stage token from a bound request."""

    return canonical_hash(
        {
            "namespace": "robotactile.n0.prepare.v1",
            "server_epoch": request["server_epoch"],
            "episode_id": request["episode_id"],
            "step_index": request["step_index"],
            "prepare_transaction_id": request["transaction_id"],
            "prepare_request_digest": request["request_digest"],
        }
    )


def _require_echo(
    binding: RequestBinding, response: Mapping[str, object], name: str
) -> str:
    value = require_sha256(response[name], name)
    if value != binding.snapshot[name]:
        raise RuntimeError(f"N0 acknowledgement has wrong {name}")
    return value


@dataclass(frozen=True)
class N0PreparedCommit:
    """Validated stage authority carried into one finalize transaction."""

    prepare_transaction_id: str
    prepare_token: str
    infer_transaction_id: str
    pending_action_sha256: str
    executed_actions_sha256: str
    grounding_frames_sha256: str
    infer_qpos_anchor_sha256: str

    @classmethod
    def accept(
        cls,
        binding: RequestBinding,
        response: Mapping[str, object],
        cache_position: int,
    ) -> "N0PreparedCommit":
        fields = (*_AUTHORITY_FIELDS, "prepare_token", "staged_cache_position")
        validate_rpc_ack(binding.snapshot, response, fields)
        echoed = {
            name: _require_echo(binding, response, name) for name in _AUTHORITY_FIELDS
        }
        token = require_sha256(response["prepare_token"], "prepare_token")
        if token != prepare_token_for_request(binding.snapshot):
            raise RuntimeError("N0 prepare acknowledgement has wrong stage token")
        position = require_nonnegative_int(
            response["staged_cache_position"], "staged_cache_position"
        )
        if position != cache_position:
            raise RuntimeError("N0 prepare acknowledgement has wrong cache position")
        return cls(
            prepare_transaction_id=require_sha256(
                binding.snapshot["transaction_id"], "prepare_transaction_id"
            ),
            prepare_token=token,
            infer_transaction_id=echoed["infer_transaction_id"],
            pending_action_sha256=echoed["pending_action_sha256"],
            executed_actions_sha256=echoed["executed_actions_sha256"],
            grounding_frames_sha256=echoed["grounding_frames_sha256"],
            infer_qpos_anchor_sha256=echoed["infer_qpos_anchor_sha256"],
        )

    def finalize_payload(
        self,
        *,
        reset_generation: int,
        cache_epoch: str,
        cache_position: int,
    ) -> dict[str, object]:
        """Build the digest-only request that authorizes one engine commit."""

        return {
            "reset_generation": reset_generation,
            "cache_epoch": cache_epoch,
            "pre_commit_cache_position": cache_position,
            "prepare_transaction_id": self.prepare_transaction_id,
            "prepare_token": self.prepare_token,
            "infer_transaction_id": self.infer_transaction_id,
            "pending_action_sha256": self.pending_action_sha256,
            "executed_actions_sha256": self.executed_actions_sha256,
            "grounding_frames_sha256": self.grounding_frames_sha256,
            "infer_qpos_anchor_sha256": self.infer_qpos_anchor_sha256,
        }

    def accept_finalize(
        self,
        binding: RequestBinding,
        response: Mapping[str, object],
        cache_position: int,
    ) -> int:
        fields = (
            "prepare_transaction_id",
            "prepare_token",
            *_AUTHORITY_FIELDS,
            "new_cache_position",
        )
        validate_rpc_ack(binding.snapshot, response, fields)
        for name in fields[:-1]:
            _require_echo(binding, response, name)
        new_position = require_nonnegative_int(
            response["new_cache_position"], "new_cache_position"
        )
        if new_position != cache_position + 1:
            raise RuntimeError("N0 finalize cache position must advance exactly once")
        return new_position

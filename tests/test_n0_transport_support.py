from __future__ import annotations

import unittest
from typing import Mapping

import numpy as np

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.transport.n0_client import N0Client
from robotactile_benchmark.transport.n0_contracts import N0GroundingFrame


def handshake_document() -> dict[str, object]:
    return {
        "schema_version": "robotactile-n0-v1",
        "source_commit": "9" * 40,
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "normalizer_sha256": "c" * 64,
        "serve_bundle_sha256": "d" * 64,
        "action_spec": "qpos8_next_step",
        "action_semantics": "absolute",
        "action_dim": 8,
        "native_action_shape": [8, 2, 4],
        "simulator_action_shape": [8, 8],
        "camera_keys": ["top", "wrist_l"],
        "tactile_keys": ["tactile_a", "tactile_b"],
        "camera_shapes": [[3, 4, 3], [3, 4, 3]],
        "tactile_shapes": [[3, 4, 3], [3, 4, 3]],
        "tactile_optional": False,
        "action_lower_bounds": [-3.0] * 7 + [0.0],
        "action_upper_bounds": [3.0] * 7 + [0.04],
        "prompt_manifest_sha256": "e" * 64,
        "requires_commit": True,
        "cold_seed_mode": "free",
        "no_cold_frame_skip": True,
        "server_epoch": "epoch-001",
    }


def strict_ack(request: Mapping[str, object], **payload: object) -> dict[str, object]:
    return {
        "schema_version": request["schema_version"],
        "op": request["op"],
        "request_id": request["request_id"],
        "episode_id": request["episode_id"],
        "step_index": request["step_index"],
        "transaction_id": request["transaction_id"],
        "server_epoch": request["server_epoch"],
        "request_digest": request["request_digest"],
        "status": "ok",
        **payload,
    }


def prepare_token(request: Mapping[str, object]) -> str:
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


def infer_observation(value: float = 0.0) -> dict[str, object]:
    encoded: dict[str, object] = {
        "vision": {
            "top": np.full((3, 4, 3), 10, dtype=np.uint8),
            "wrist_l": np.full((3, 4, 3), 20, dtype=np.uint8),
        },
        "tactile": {
            "tactile_a": np.full((3, 4, 3), 30, dtype=np.uint8),
            "tactile_b": np.full((3, 4, 3), 40, dtype=np.uint8),
        },
        "proprio": np.full(8, value, dtype=np.float32),
    }
    return {**encoded, "observation_digest": canonical_hash(encoded)}


def grounding_frames() -> tuple[N0GroundingFrame, ...]:
    return tuple(
        N0GroundingFrame(
            step_index=step,
            top=np.full((3, 4, 3), 10 + step, dtype=np.uint8),
            wrist_l=np.full((3, 4, 3), 20 + step, dtype=np.uint8),
            tactile_a=np.full((3, 4, 3), 30 + step, dtype=np.uint8),
            tactile_b=np.full((3, 4, 3), 40 + step, dtype=np.uint8),
            proprio=np.full(8, step / 100.0, dtype=np.float32),
        )
        for step in range(1, 9)
    )


class FakeTransport:
    def __init__(self) -> None:
        self.connect_count = 0
        self.calls: list[Mapping[str, object]] = []
        self.finalize_timeout = False
        self.generation = 1
        self.cache_position = 0
        self.prepare_count = 0
        self.finalize_count = 0
        self.engine_commit_count = 0
        self.staged_token: str | None = None

    def connect(self) -> Mapping[str, object]:
        self.connect_count += 1
        return handshake_document()

    def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
        self.calls.append(request)
        op = request["op"]
        if op in {"reset", "hard_reset"}:
            self.generation += 1
            self.cache_position = 0
            self.staged_token = None
            return strict_ack(
                request,
                reset_generation=self.generation,
                cache_epoch=f"cache-{self.generation}",
                cache_position=0,
            )
        if op == "infer":
            native = np.zeros((8, 2, 4), dtype=np.float32)
            return strict_ack(
                request,
                native_action=native,
                native_action_sha256=canonical_hash(native),
                pre_commit_cache_position=self.cache_position,
            )
        if op == "prepare_commit":
            self.prepare_count += 1
            self.staged_token = prepare_token(request)
            return strict_ack(
                request,
                infer_transaction_id=request["infer_transaction_id"],
                pending_action_sha256=request["pending_action_sha256"],
                executed_actions_sha256=request["executed_actions_sha256"],
                grounding_frames_sha256=request["grounding_frames_sha256"],
                infer_qpos_anchor_sha256=request["infer_qpos_anchor_sha256"],
                prepare_token=self.staged_token,
                staged_cache_position=self.cache_position,
            )
        if op == "finalize_commit":
            self.finalize_count += 1
            if request["prepare_token"] != self.staged_token:
                raise RuntimeError("wrong staged token")
            self.cache_position += 1
            self.engine_commit_count += 1
            self.staged_token = None
            if self.finalize_timeout:
                raise TimeoutError("lost finalize commit ack")
            return strict_ack(
                request,
                prepare_transaction_id=request["prepare_transaction_id"],
                prepare_token=request["prepare_token"],
                infer_transaction_id=request["infer_transaction_id"],
                pending_action_sha256=request["pending_action_sha256"],
                executed_actions_sha256=request["executed_actions_sha256"],
                grounding_frames_sha256=request["grounding_frames_sha256"],
                infer_qpos_anchor_sha256=request["infer_qpos_anchor_sha256"],
                new_cache_position=self.cache_position,
            )
        if op == "abort":
            self.generation += 1
            self.cache_position = 0
            self.staged_token = None
            return strict_ack(
                request,
                reset_generation=self.generation,
                cache_epoch=f"cache-{self.generation}",
                cache_position=0,
            )
        raise AssertionError(f"unexpected op: {op}")

    def close(self) -> None:
        return None


class WrongFieldTransport(FakeTransport):
    def __init__(
        self, field: str, wrong_value: object, *, operation: str | None = None
    ) -> None:
        super().__init__()
        self._field = field
        self._wrong_value = wrong_value
        self._operation = operation

    def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
        response = dict(super().call(request))
        if self._operation is None or request["op"] == self._operation:
            response[self._field] = self._wrong_value
        return response


class N0ClientTestCase(unittest.TestCase):
    def make_client(self, transport: FakeTransport) -> N0Client:
        return N0Client(
            transport,
            expected_source_commit="9" * 40,
            expected_checkpoint_sha256="a" * 64,
            expected_config_sha256="b" * 64,
            expected_normalizer_sha256="c" * 64,
            expected_serve_bundle_sha256="d" * 64,
            expected_prompt_manifest_sha256="e" * 64,
        )

from __future__ import annotations

import unittest
from typing import Mapping, cast

import numpy as np
from test_n0_transport_support import (
    FakeTransport,
    N0ClientTestCase,
    grounding_frames,
    handshake_document,
    infer_observation,
)

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.transport.n0_client import N0ClientState
from robotactile_benchmark.transport.n0_contracts import N0Handshake


class N0ClientLifecycleTests(N0ClientTestCase):
    def test_reset_infer_commit_infer_happy_path(self) -> None:
        transport = FakeTransport()
        client = self.make_client(transport)
        client.handshake()
        self.assertEqual(client.handshake_identity.checkpoint_sha256, "a" * 64)
        with self.assertRaises(AttributeError):
            client.handshake_identity = N0Handshake.from_mapping(  # type: ignore[misc]
                handshake_document(),
                expected_source_commit="9" * 40,
                expected_checkpoint_sha256="a" * 64,
                expected_config_sha256="b" * 64,
                expected_normalizer_sha256="c" * 64,
                expected_serve_bundle_sha256="d" * 64,
                expected_prompt_manifest_sha256="e" * 64,
            )
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        native = client.infer(0, infer_observation())
        client.commit(
            step_index=0,
            native_action=native,
            executed_actions=np.transpose(native, (1, 2, 0)).reshape(8, 8),
            grounding_frames=grounding_frames(),
        )
        client.infer(8, infer_observation(0.08))

        self.assertEqual(client.state, N0ClientState.AWAITING_EXECUTION)
        self.assertEqual(
            [request["op"] for request in transport.calls],
            ["reset", "infer", "prepare_commit", "finalize_commit", "infer"],
        )
        self.assertEqual(transport.calls[1]["step_index"], 0)
        self.assertEqual(transport.calls[2]["step_index"], 0)
        self.assertEqual(
            transport.calls[2]["infer_transaction_id"],
            transport.calls[1]["transaction_id"],
        )
        self.assertNotEqual(
            transport.calls[2]["transaction_id"],
            transport.calls[1]["transaction_id"],
        )
        prepare = transport.calls[2]
        self.assertEqual(
            set(prepare),
            {
                "schema_version",
                "op",
                "request_id",
                "episode_id",
                "step_index",
                "transaction_id",
                "server_epoch",
                "request_digest",
                "reset_generation",
                "cache_epoch",
                "pre_commit_cache_position",
                "infer_transaction_id",
                "pending_action_sha256",
                "executed_actions",
                "executed_actions_sha256",
                "grounding_frames",
                "grounding_frames_sha256",
                "infer_qpos_anchor",
                "infer_qpos_anchor_sha256",
            },
        )
        self.assertEqual(np.asarray(prepare["executed_actions"]).shape, (8, 8))
        self.assertEqual(len(cast(tuple[object, ...], prepare["grounding_frames"])), 8)
        np.testing.assert_array_equal(
            prepare["infer_qpos_anchor"], np.zeros(8, dtype=np.float32)
        )
        self.assertEqual(prepare["pending_action_sha256"], canonical_hash(native))
        self.assertEqual(
            prepare["executed_actions_sha256"],
            canonical_hash(prepare["executed_actions"]),
        )
        self.assertEqual(
            prepare["grounding_frames_sha256"],
            canonical_hash(prepare["grounding_frames"]),
        )
        self.assertEqual(
            prepare["infer_qpos_anchor_sha256"],
            canonical_hash(prepare["infer_qpos_anchor"]),
        )
        finalize = transport.calls[3]
        self.assertEqual(
            set(finalize),
            {
                "schema_version",
                "op",
                "request_id",
                "episode_id",
                "step_index",
                "transaction_id",
                "server_epoch",
                "request_digest",
                "reset_generation",
                "cache_epoch",
                "pre_commit_cache_position",
                "prepare_transaction_id",
                "prepare_token",
                "infer_transaction_id",
                "pending_action_sha256",
                "executed_actions_sha256",
                "grounding_frames_sha256",
                "infer_qpos_anchor_sha256",
            },
        )
        self.assertEqual(finalize["prepare_transaction_id"], prepare["transaction_id"])
        self.assertEqual(
            finalize["infer_transaction_id"], prepare["infer_transaction_id"]
        )
        for field in (
            "pending_action_sha256",
            "executed_actions_sha256",
            "grounding_frames_sha256",
            "infer_qpos_anchor_sha256",
        ):
            self.assertEqual(finalize[field], prepare[field])
        self.assertEqual(transport.prepare_count, 1)
        self.assertEqual(transport.finalize_count, 1)
        self.assertEqual(transport.engine_commit_count, 1)
        for request in transport.calls:
            unsigned = {
                key: value for key, value in request.items() if key != "request_digest"
            }
            self.assertEqual(request["request_digest"], canonical_hash(unsigned))

    def test_failed_handshake_closes_lease_and_cannot_reconnect(self) -> None:
        class InvalidHandshakeTransport(FakeTransport):
            def __init__(self) -> None:
                super().__init__()
                self.close_count = 0

            def connect(self) -> Mapping[str, object]:
                document = dict(super().connect())
                document["unknown"] = "field"
                return document

            def close(self) -> None:
                self.close_count += 1

        transport = InvalidHandshakeTransport()
        client = self.make_client(transport)
        with self.assertRaises(ValueError):
            client.handshake()
        with self.assertRaises(RuntimeError):
            client.handshake()
        self.assertEqual(transport.connect_count, 1)
        self.assertEqual(transport.close_count, 1)

    def test_infer_step_is_exact_and_rejected_before_rpc(self) -> None:
        transport = FakeTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        with self.assertRaisesRegex(ValueError, "step"):
            client.infer(1, infer_observation())
        self.assertEqual([request["op"] for request in transport.calls], ["reset"])

        native = client.infer(0, infer_observation())
        client.commit(
            step_index=0,
            native_action=native,
            executed_actions=np.transpose(native, (1, 2, 0)).reshape(8, 8),
            grounding_frames=grounding_frames(),
        )
        with self.assertRaisesRegex(ValueError, "step"):
            client.infer(7, infer_observation(0.08))
        self.assertEqual(
            [request["op"] for request in transport.calls],
            ["reset", "infer", "prepare_commit", "finalize_commit"],
        )

    def test_pending_native_action_cannot_be_made_writeable(self) -> None:
        transport = FakeTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)

        native = client.infer(0, infer_observation())

        self.assertFalse(native.flags.writeable)
        with self.assertRaises(ValueError):
            native.setflags(write=True)

    def test_lost_commit_ack_is_never_retried_and_requires_hard_reset(self) -> None:
        transport = FakeTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        native = client.infer(0, infer_observation())
        transport.finalize_timeout = True
        with self.assertRaises(TimeoutError):
            client.commit(
                step_index=0,
                native_action=native,
                executed_actions=np.transpose(native, (1, 2, 0)).reshape(8, 8),
                grounding_frames=grounding_frames(),
            )
        self.assertEqual(client.state, N0ClientState.INDETERMINATE)
        self.assertEqual(transport.prepare_count, 1)
        self.assertEqual(transport.finalize_count, 1)
        self.assertEqual(transport.engine_commit_count, 1)
        with self.assertRaises(RuntimeError):
            client.infer(8, infer_observation(0.08))

        transport.finalize_timeout = False
        client.reset("episode-2", "insert HDMI", "insert_HDMI", 8)
        self.assertEqual(transport.calls[-1]["op"], "hard_reset")
        self.assertEqual(client.state, N0ClientState.READY)

    def test_wrong_ack_identity_fails_closed(self) -> None:
        class WrongAckTransport(FakeTransport):
            def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
                response = dict(super().call(request))
                response["transaction_id"] = "stale-transaction"
                return response

        client = self.make_client(WrongAckTransport())
        client.handshake()
        with self.assertRaisesRegex(RuntimeError, "transaction"):
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)


if __name__ == "__main__":
    unittest.main()

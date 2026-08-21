from __future__ import annotations

import unittest
from typing import Mapping, cast

import numpy as np
from test_n0_transport_support import (
    FakeTransport,
    N0ClientTestCase,
    grounding_frames,
    infer_observation,
)

from robotactile_benchmark.transport.n0_client import N0ClientState
from robotactile_benchmark.transport.n0_contracts import N0GroundingFrame


class N0ClientAuthorityTests(N0ClientTestCase):
    def test_infer_rejects_nonfinite_qpos_anchor_before_rpc(self) -> None:
        transport = FakeTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        observation = infer_observation()
        observation["proprio"] = np.full(8, np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            client.infer(0, observation)
        self.assertEqual([request["op"] for request in transport.calls], ["reset"])

    def test_infer_rejects_unbound_digest_and_shape_before_rpc(self) -> None:
        for mutation in ("digest", "shape"):
            transport = FakeTransport()
            client = self.make_client(transport)
            client.handshake()
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
            observation = infer_observation()
            if mutation == "digest":
                observation["observation_digest"] = "f" * 64
            else:
                vision = cast(dict[str, object], observation["vision"])
                vision["top"] = np.zeros((4, 4, 3), dtype=np.uint8)
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "digest|shape"),
            ):
                client.infer(0, observation)
            self.assertEqual([request["op"] for request in transport.calls], ["reset"])

    def test_duplicate_commit_is_rejected_before_a_second_rpc(self) -> None:
        transport = FakeTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        native = client.infer(0, infer_observation())
        arguments = {
            "step_index": 0,
            "native_action": native,
            "executed_actions": np.transpose(native, (1, 2, 0)).reshape(8, 8),
            "grounding_frames": grounding_frames(),
        }
        client.commit(**arguments)
        with self.assertRaisesRegex(RuntimeError, "pending"):
            client.commit(**arguments)
        self.assertEqual(transport.prepare_count, 1)
        self.assertEqual(transport.finalize_count, 1)
        self.assertEqual(transport.engine_commit_count, 1)

    def test_commit_authority_preflight_rejects_without_transport_effects(self) -> None:
        invalid_arguments: tuple[dict[str, object], ...] = (
            {"native_action": np.ones((8, 2, 4), dtype=np.float32)},
            {"executed_actions": np.zeros((7, 8), dtype=np.float32)},
            {
                "grounding_frames": (
                    N0GroundingFrame(
                        step_index=9,
                        top=np.zeros((3, 4, 3), dtype=np.uint8),
                        wrist_l=np.zeros((3, 4, 3), dtype=np.uint8),
                        tactile_a=np.zeros((3, 4, 3), dtype=np.uint8),
                        tactile_b=np.zeros((3, 4, 3), dtype=np.uint8),
                        proprio=np.zeros(8, dtype=np.float32),
                    ),
                    *grounding_frames()[1:],
                )
            },
        )
        for replacement in invalid_arguments:
            transport = FakeTransport()
            client = self.make_client(transport)
            client.handshake()
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
            native = client.infer(0, infer_observation())
            arguments: dict[str, object] = {
                "step_index": 0,
                "native_action": native,
                "executed_actions": np.transpose(native, (1, 2, 0)).reshape(8, 8),
                "grounding_frames": grounding_frames(),
            }
            with (
                self.subTest(replacement=set(replacement)),
                self.assertRaises(ValueError),
            ):
                client.commit(**{**arguments, **replacement})  # type: ignore[arg-type]
            self.assertEqual(
                [request["op"] for request in transport.calls], ["reset", "infer"]
            )
            self.assertEqual(client.state, N0ClientState.AWAITING_EXECUTION)

    def test_abort_requires_a_new_cache_epoch(self) -> None:
        class ReusedCacheEpochTransport(FakeTransport):
            def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
                response = dict(super().call(request))
                if request["op"] == "abort":
                    response["cache_epoch"] = "cache-2"
                return response

        transport = ReusedCacheEpochTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        with self.assertRaisesRegex(RuntimeError, "cache epoch"):
            client.abort("test_abort")
        self.assertEqual(client.state, N0ClientState.INDETERMINATE)


if __name__ == "__main__":
    unittest.main()

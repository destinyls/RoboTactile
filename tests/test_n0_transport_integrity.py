from __future__ import annotations

import unittest
from typing import Mapping, cast

import numpy as np
from test_n0_transport_support import (
    FakeTransport,
    N0ClientTestCase,
    WrongFieldTransport,
    grounding_frames,
    infer_observation,
)

from robotactile_benchmark.transport.n0_client import N0ClientState


class N0ClientIntegrityTests(N0ClientTestCase):
    def test_every_common_ack_identity_field_is_bound(self) -> None:
        replacements: tuple[tuple[str, object], ...] = (
            ("schema_version", "wrong-schema"),
            ("op", "infer"),
            ("request_id", "f" * 64),
            ("episode_id", "wrong-episode"),
            ("episode_id", np.str_("episode-1")),
            ("step_index", 99),
            ("step_index", False),
            ("step_index", np.int64(0)),
            ("transaction_id", "f" * 64),
            ("server_epoch", "stale-epoch"),
            ("request_digest", "f" * 64),
            ("status", "error"),
        )
        for field, wrong_value in replacements:
            transport = WrongFieldTransport(field, wrong_value)
            client = self.make_client(transport)
            client.handshake()
            with (
                self.subTest(field=field, wrong_value=wrong_value),
                self.assertRaises((TypeError, RuntimeError)),
            ):
                client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
            self.assertEqual(len(transport.calls), 1)

    def test_transport_cannot_mutate_bound_request_after_digesting(self) -> None:
        class MutatingTransport(FakeTransport):
            def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
                mutable = cast(dict[str, object], request)
                if request["op"] == "reset":
                    mutable["episode_id"] = "mutated-episode"
                return super().call(request)

        transport = MutatingTransport()
        client = self.make_client(transport)
        client.handshake()
        with self.assertRaisesRegex(RuntimeError, "mutat|digest|binding"):
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(client.state, N0ClientState.INDETERMINATE)

    def test_nested_array_mutation_is_detected_after_transport_returns(self) -> None:
        class NestedMutationTransport(FakeTransport):
            def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
                if request["op"] == "infer":
                    mutable = cast(dict[str, object], request)
                    observation = cast(dict[str, object], mutable["observation"])
                    observation["proprio"] = np.ones(8, dtype=np.float32)
                return super().call(request)

        transport = NestedMutationTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        with self.assertRaisesRegex(RuntimeError, "mutat|digest|binding"):
            client.infer(0, infer_observation())
        self.assertEqual(client.state, N0ClientState.RESET_REQUIRED)

    def test_in_place_nested_array_mutation_is_detected(self) -> None:
        class InPlaceMutationTransport(FakeTransport):
            def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
                if request["op"] == "infer":
                    observation = cast(Mapping[str, object], request["observation"])
                    proprio = cast(np.ndarray, observation["proprio"])
                    proprio.setflags(write=True)
                    proprio.fill(1.0)
                return super().call(request)

        transport = InPlaceMutationTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        with self.assertRaisesRegex(RuntimeError, "mutat|digest|binding"):
            client.infer(0, infer_observation())
        self.assertEqual(client.state, N0ClientState.RESET_REQUIRED)

    def test_mutated_prepare_never_reaches_engine_or_finalize(self) -> None:
        class CoordinatedPrepareMutationTransport(FakeTransport):
            def call(self, request: Mapping[str, object]) -> Mapping[str, object]:
                if request["op"] == "prepare_commit":
                    cast(dict[str, object], request)["episode_id"] = "mutated"
                return super().call(request)

        transport = CoordinatedPrepareMutationTransport()
        client = self.make_client(transport)
        client.handshake()
        client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
        native = client.infer(0, infer_observation())
        with self.assertRaisesRegex(RuntimeError, "mutat|digest|binding"):
            client.commit(
                step_index=0,
                native_action=native,
                executed_actions=np.transpose(native, (1, 2, 0)).reshape(8, 8),
                grounding_frames=grounding_frames(),
            )

        self.assertEqual(client.state, N0ClientState.RESET_REQUIRED)
        self.assertEqual(transport.prepare_count, 1)
        self.assertEqual(transport.finalize_count, 0)
        self.assertEqual(transport.engine_commit_count, 0)
        self.assertEqual(
            [request["op"] for request in transport.calls],
            ["reset", "infer", "prepare_commit"],
        )

    def test_operation_specific_ack_fields_are_all_bound(self) -> None:
        reset_tampering: tuple[tuple[str, object], ...] = (
            ("reset_generation", "bad"),
            ("cache_epoch", ""),
            ("cache_position", 1),
        )
        for field, wrong_value in reset_tampering:
            transport = WrongFieldTransport(field, wrong_value)
            client = self.make_client(transport)
            client.handshake()
            with (
                self.subTest(op="reset", field=field),
                self.assertRaises((TypeError, ValueError, RuntimeError)),
            ):
                client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)

        for field, wrong_value in (
            ("native_action", np.zeros((8, 2, 3), dtype=np.float32)),
            ("native_action_sha256", "f" * 64),
            ("pre_commit_cache_position", 1),
        ):
            transport = WrongFieldTransport(field, wrong_value, operation="infer")
            client = self.make_client(transport)
            client.handshake()
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
            with (
                self.subTest(op="infer", field=field),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                client.infer(0, infer_observation())

        authority_fields = (
            "infer_transaction_id",
            "pending_action_sha256",
            "executed_actions_sha256",
            "grounding_frames_sha256",
            "infer_qpos_anchor_sha256",
        )
        for field, wrong_value in (
            *((field, "f" * 64) for field in authority_fields),
            ("prepare_token", "f" * 64),
            ("staged_cache_position", 7),
        ):
            transport = WrongFieldTransport(
                field, wrong_value, operation="prepare_commit"
            )
            client = self.make_client(transport)
            client.handshake()
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
            native = client.infer(0, infer_observation())
            with (
                self.subTest(op="prepare_commit", field=field),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                client.commit(
                    step_index=0,
                    native_action=native,
                    executed_actions=np.transpose(native, (1, 2, 0)).reshape(8, 8),
                    grounding_frames=grounding_frames(),
                )
            self.assertEqual(client.state, N0ClientState.RESET_REQUIRED)
            self.assertEqual(transport.finalize_count, 0)
            self.assertEqual(transport.engine_commit_count, 0)

        finalize_fields = (
            "prepare_transaction_id",
            "prepare_token",
            *authority_fields,
        )
        for field, wrong_value in (
            *((field, "f" * 64) for field in finalize_fields),
            ("new_cache_position", 7),
        ):
            transport = WrongFieldTransport(
                field, wrong_value, operation="finalize_commit"
            )
            client = self.make_client(transport)
            client.handshake()
            client.reset("episode-1", "insert HDMI", "insert_HDMI", 7)
            native = client.infer(0, infer_observation())
            with (
                self.subTest(op="finalize_commit", field=field),
                self.assertRaises(RuntimeError),
            ):
                client.commit(
                    step_index=0,
                    native_action=native,
                    executed_actions=np.transpose(native, (1, 2, 0)).reshape(8, 8),
                    grounding_frames=grounding_frames(),
                )
            self.assertEqual(client.state, N0ClientState.INDETERMINATE)
            self.assertEqual(transport.finalize_count, 1)
            self.assertEqual(transport.engine_commit_count, 1)


if __name__ == "__main__":
    unittest.main()

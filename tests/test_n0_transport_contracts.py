from __future__ import annotations

import unittest
from typing import Mapping, cast

import numpy as np
from test_n0_transport_support import handshake_document

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.transport.n0_binding import RequestBinding
from robotactile_benchmark.transport.n0_contracts import (
    N0GroundingFrame,
    N0Handshake,
)


class N0HandshakeTests(unittest.TestCase):
    def test_handshake_is_exact_and_hash_bound(self) -> None:
        document = handshake_document()
        handshake = N0Handshake.from_mapping(
            document,
            expected_source_commit="9" * 40,
            expected_checkpoint_sha256="a" * 64,
            expected_config_sha256="b" * 64,
            expected_normalizer_sha256="c" * 64,
            expected_serve_bundle_sha256="d" * 64,
            expected_prompt_manifest_sha256="e" * 64,
        )
        self.assertEqual(handshake.server_epoch, "epoch-001")
        self.assertEqual(handshake.camera_shapes, ((3, 4, 3), (3, 4, 3)))
        self.assertEqual(handshake.tactile_shapes, ((3, 4, 3), (3, 4, 3)))
        for invalid in (
            {},
            {**document, "unknown": 1},
            {**document, "checkpoint_sha256": "f" * 64},
            {**document, "camera_shapes": [[3, 4, 4], [3, 4, 3]]},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                N0Handshake.from_mapping(
                    invalid,
                    expected_source_commit="9" * 40,
                    expected_checkpoint_sha256="a" * 64,
                    expected_config_sha256="b" * 64,
                    expected_normalizer_sha256="c" * 64,
                    expected_serve_bundle_sha256="d" * 64,
                    expected_prompt_manifest_sha256="e" * 64,
                )

    def test_grounding_frame_is_deep_frozen_and_self_hashed(self) -> None:
        top = np.full((3, 4, 3), 11, dtype=np.uint8)
        proprio = np.arange(8, dtype=np.float32)
        frame = N0GroundingFrame(
            step_index=1,
            top=top,
            wrist_l=np.full((3, 4, 3), 21, dtype=np.uint8),
            tactile_a=np.full((3, 4, 3), 31, dtype=np.uint8),
            tactile_b=np.full((3, 4, 3), 41, dtype=np.uint8),
            proprio=proprio,
        )
        top.fill(99)
        proprio.fill(99.0)

        self.assertEqual(int(frame.top[0, 0, 0]), 11)
        self.assertEqual(float(frame.proprio[0]), 0.0)
        self.assertFalse(frame.top.flags.writeable)
        self.assertFalse(frame.proprio.flags.writeable)
        with self.assertRaises(ValueError):
            frame.top.setflags(write=True)
        with self.assertRaises(ValueError):
            frame.proprio.setflags(write=True)
        wire = frame.to_wire()
        self.assertEqual(
            set(wire),
            {
                "step_index",
                "top",
                "wrist_l",
                "tactile_a",
                "tactile_b",
                "proprio",
                "frame_sha256",
            },
        )
        unsigned = {key: value for key, value in wire.items() if key != "frame_sha256"}
        self.assertEqual(frame.frame_sha256, canonical_hash(unsigned))

    def test_grounding_frame_rejects_bad_rgb_and_proprio(self) -> None:
        base = {
            "step_index": 1,
            "top": np.zeros((3, 4, 3), dtype=np.uint8),
            "wrist_l": np.zeros((3, 4, 3), dtype=np.uint8),
            "tactile_a": np.zeros((3, 4, 3), dtype=np.uint8),
            "tactile_b": np.zeros((3, 4, 3), dtype=np.uint8),
            "proprio": np.zeros(8, dtype=np.float32),
        }
        for replacement in (
            {"top": np.zeros((3, 4, 3), dtype=np.float32)},
            {"tactile_a": np.zeros((3, 4), dtype=np.uint8)},
            {"proprio": np.zeros(9, dtype=np.float32)},
            {"proprio": np.full(8, np.nan, dtype=np.float32)},
        ):
            with (
                self.subTest(replacement=replacement),
                self.assertRaises((TypeError, ValueError)),
            ):
                N0GroundingFrame(**{**base, **replacement})

    def test_request_binding_is_deep_immutable_and_defensively_copied(self) -> None:
        source = np.arange(8, dtype=np.float32)
        unsigned: dict[str, object] = {"op": "infer", "nested": {"qpos": source}}
        binding = RequestBinding.from_unsigned(unsigned)
        source.fill(99.0)

        nested = cast(Mapping[str, object], binding.snapshot["nested"])
        qpos = cast(np.ndarray, nested["qpos"])
        np.testing.assert_array_equal(qpos, np.arange(8, dtype=np.float32))
        self.assertFalse(qpos.flags.writeable)
        with self.assertRaises(ValueError):
            qpos.setflags(write=True)
        with self.assertRaises(TypeError):
            cast(dict[str, object], binding.snapshot)["op"] = "mutated"
        wire = binding.wire_request()
        wire_nested = cast(Mapping[str, object], wire["nested"])
        self.assertFalse(cast(np.ndarray, wire_nested["qpos"]).flags.writeable)


if __name__ == "__main__":
    unittest.main()

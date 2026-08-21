from __future__ import annotations

import unittest

import numpy as np

from robotactile_benchmark.transport.n0_codec import (
    native_to_simulator_actions,
    simulator_to_native_actions,
)


class N0CodecTests(unittest.TestCase):
    def test_native_action_maps_frame_then_slot_and_keeps_row_zero(self) -> None:
        native = np.arange(64, dtype=np.float32).reshape(8, 2, 4)
        expected = np.asarray(
            [
                [0, 8, 16, 24, 32, 40, 48, 56],
                [1, 9, 17, 25, 33, 41, 49, 57],
                [2, 10, 18, 26, 34, 42, 50, 58],
                [3, 11, 19, 27, 35, 43, 51, 59],
                [4, 12, 20, 28, 36, 44, 52, 60],
                [5, 13, 21, 29, 37, 45, 53, 61],
                [6, 14, 22, 30, 38, 46, 54, 62],
                [7, 15, 23, 31, 39, 47, 55, 63],
            ],
            dtype=np.float32,
        )

        flat = native_to_simulator_actions(native)

        np.testing.assert_array_equal(flat, expected)
        self.assertEqual(flat[0].tolist(), expected[0].tolist())
        self.assertFalse(flat.flags.writeable)

    def test_reverse_conversion_is_bit_exact(self) -> None:
        native = np.linspace(-0.7, 0.7, 64, dtype=np.float32).reshape(8, 2, 4)
        recovered = simulator_to_native_actions(native_to_simulator_actions(native))
        self.assertEqual(recovered.tobytes(), native.tobytes())

    def test_codec_rejects_wrong_shape_dtype_and_nonfinite_values(self) -> None:
        for value in (
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 2, 4), dtype=np.float64),
            np.full((8, 2, 4), np.nan, dtype=np.float32),
        ):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                native_to_simulator_actions(value)


if __name__ == "__main__":
    unittest.main()

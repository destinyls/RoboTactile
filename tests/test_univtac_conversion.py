from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from robotactile_benchmark.adapters.univtac import ContactPhaseState
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_conversion import (
    UniVTACConversionError,
    _tactile_rgb_array,
    convert_raw_observation,
    cuda_like_to_numpy,
    joint9_to_qpos8,
    reorder_joint9,
    validate_action_batch,
)


class StrictCudaLike:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value
        self.calls: list[str] = []

    def _call(self, expected: str) -> StrictCudaLike:
        wanted = ("detach", "cpu", "contiguous", "numpy")[len(self.calls)]
        if expected != wanted:
            raise AssertionError(f"expected {wanted}, received {expected}")
        self.calls.append(expected)
        return self

    def detach(self) -> StrictCudaLike:
        return self._call("detach")

    def cpu(self) -> StrictCudaLike:
        return self._call("cpu")

    def contiguous(self) -> StrictCudaLike:
        return self._call("contiguous")

    def numpy(self) -> np.ndarray:
        self._call("numpy")
        return self._value


def _raw(
    native_step: int = 417,
    *,
    live_joint_names: tuple[str, ...],
    cuda_like: bool = True,
) -> dict[str, Any]:
    canonical = tuple(f"panda_joint{index}" for index in range(1, 8)) + (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    values = {name: float(index) / 10.0 for index, name in enumerate(canonical[:7])}
    values.update(
        {
            "panda_finger_joint1": 0.02,
            "panda_finger_joint2": 0.02,
        }
    )

    def wrapped(value: np.ndarray) -> Any:
        return StrictCudaLike(value) if cuda_like else value

    return {
        "step": native_step,
        "observation": {
            "head": {"rgb": wrapped(np.full((270, 480, 3), 20, np.uint8))},
            "wrist": {"rgb": wrapped(np.full((270, 480, 3), 30, np.uint8))},
        },
        "tactile": {
            "left_tactile": {
                "rgb_marker": wrapped(np.full((240, 320, 3), 60, np.uint8)),
                "depth": wrapped(np.full((240, 320), 34.0, np.float32)),
            },
            "right_tactile": {
                "rgb_marker": wrapped(np.full((240, 320, 3), 70, np.uint8)),
                "depth": wrapped(np.full((240, 320), 32.5, np.float32)),
            },
        },
        "embodiment": {
            "joint": wrapped(
                np.asarray(
                    [values[name] for name in live_joint_names], dtype=np.float32
                )
            )
        },
    }


class UniVTACConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_univtac_backend_config("pull_out_key")
        self.live_names = tuple(reversed(self.config.canonical_joint_names))
        self.handshake = self.config.expected_handshake(self.live_names)
        self.phase_states = {
            "left": ContactPhaseState(False),
            "right": ContactPhaseState(False),
        }

    def test_cuda_like_values_require_exact_conversion_order(self) -> None:
        value = StrictCudaLike(np.arange(4, dtype=np.float32))

        converted = cuda_like_to_numpy(value, "joint")

        np.testing.assert_array_equal(converted, np.arange(4, dtype=np.float32))
        self.assertEqual(value.calls, ["detach", "cpu", "contiguous", "numpy"])

    def test_joint9_reorders_and_qpos8_uses_upstream_first_finger(self) -> None:
        raw_values = np.asarray(
            [0.02, 0.02, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            dtype=np.float32,
        )

        reordered = reorder_joint9(
            raw_values,
            self.live_names,
            self.config.canonical_joint_names,
        )

        np.testing.assert_allclose(
            reordered,
            np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02, 0.02]),
        )
        self.assertEqual(reordered.shape, (9,))
        qpos8 = joint9_to_qpos8(reordered)
        self.assertEqual(qpos8.shape, (8,))
        np.testing.assert_allclose(
            qpos8,
            np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02]),
        )

        asymmetric = reorder_joint9(
            np.asarray(
                [0.01, 0.02, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                dtype=np.float32,
            ),
            self.live_names,
            self.config.canonical_joint_names,
        )
        self.assertAlmostEqual(float(asymmetric[-2]), 0.02)
        self.assertAlmostEqual(float(asymmetric[-1]), 0.01)
        self.assertAlmostEqual(float(joint9_to_qpos8(asymmetric)[-1]), 0.02)

    def test_duplicate_missing_and_unexpected_joint_names_fail_closed(self) -> None:
        canonical = self.config.canonical_joint_names
        cases = (
            canonical[:-1] + (canonical[-2],),
            canonical[:-1],
            canonical[:-1] + ("unexpected_joint",),
        )
        for live_names in cases:
            with (
                self.subTest(live_names=live_names),
                self.assertRaisesRegex(
                    (UniVTACContractError, UniVTACConversionError),
                    "joint",
                ),
            ):
                reorder_joint9(
                    np.zeros(len(live_names), dtype=np.float32),
                    live_names,
                    canonical,
                )

    def test_dense_benchmark_step_keeps_distinct_native_step_diagnostic(self) -> None:
        raw = _raw(live_joint_names=self.live_names)

        converted = convert_raw_observation(
            raw,
            config=self.config,
            handshake=self.handshake,
            phase_states=self.phase_states,
            episode_id="episode-1",
            task_id="pull_out_key",
            initial_seed=11,
            benchmark_step=0,
        )

        self.assertEqual(converted.native_step_id, 417)
        self.assertEqual(converted.record.observation.step_index, 0)
        self.assertEqual(converted.record.provenance_for("left").source_index, 0)
        self.assertEqual(converted.record.observation.proprio.shape, (8,))
        self.assertNotEqual(
            converted.simulator_state_sha256,
            converted.joint_reorder_witness_sha256,
        )
        left_phase = converted.record.provenance_for("left").phase.value
        right_phase = converted.record.provenance_for("right").phase.value
        self.assertEqual((left_phase, right_phase), ("free", "contact_onset"))
        leaves = (
            raw["observation"]["head"]["rgb"],
            raw["observation"]["wrist"]["rgb"],
            raw["tactile"]["left_tactile"]["rgb_marker"],
            raw["tactile"]["left_tactile"]["depth"],
            raw["tactile"]["right_tactile"]["rgb_marker"],
            raw["tactile"]["right_tactile"]["depth"],
            raw["embodiment"]["joint"],
        )
        self.assertTrue(
            all(
                leaf.calls == ["detach", "cpu", "contiguous", "numpy"]
                for leaf in leaves
            )
        )

    def test_tacex_float32_tactile_rgb_is_cast_exactly_for_both_sensors(
        self,
    ) -> None:
        raw = _raw(live_joint_names=self.live_names, cuda_like=False)
        left = np.full(self.config.tactile_rgb_shape, 60.0, dtype=np.float32)
        right = np.full(self.config.tactile_rgb_shape, 255.0, dtype=np.float32)
        right[0, 0] = np.asarray([0.0, 127.0, 255.0], dtype=np.float32)
        raw["tactile"]["left_tactile"]["rgb_marker"] = left
        raw["tactile"]["right_tactile"]["rgb_marker"] = right

        converted = self._convert(raw)

        left_payload = converted.record.observation.sensor("left").payload
        right_payload = converted.record.observation.sensor("right").payload
        self.assertEqual(left_payload.dtype, np.dtype(np.uint8))
        self.assertEqual(right_payload.dtype, np.dtype(np.uint8))
        np.testing.assert_array_equal(left_payload, left.astype(np.uint8))
        np.testing.assert_array_equal(right_payload, right.astype(np.uint8))

    def test_uint8_and_float32_marker_payloads_are_exactly_equivalent(self) -> None:
        shape = self.config.tactile_rgb_shape
        linear = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
        left_marker = ((linear * 37 + 11) % 256).astype(np.uint8)
        right_marker = ((linear * 19 + 73) % 256).astype(np.uint8)
        left_uint8 = np.asfortranarray(left_marker)
        right_uint8 = np.asfortranarray(right_marker)
        left_float32 = np.asfortranarray(left_marker.astype(np.float32))
        right_float32 = np.asfortranarray(right_marker.astype(np.float32))
        inputs = (left_uint8, right_uint8, left_float32, right_float32)
        snapshots = tuple(value.copy() for value in inputs)

        bridged = tuple(
            _tactile_rgb_array(value, "test tactile RGB", shape) for value in inputs
        )
        for source, output, expected in zip(
            inputs,
            bridged,
            (left_marker, right_marker, left_marker, right_marker),
        ):
            self.assertEqual(output.dtype, np.dtype(np.uint8))
            self.assertTrue(output.flags.c_contiguous)
            self.assertFalse(np.shares_memory(output, source))
            np.testing.assert_array_equal(output, expected)

        raw_uint8 = _raw(live_joint_names=self.live_names, cuda_like=False)
        raw_float32 = _raw(live_joint_names=self.live_names, cuda_like=False)
        for raw, left, right in (
            (raw_uint8, left_uint8, right_uint8),
            (raw_float32, left_float32, right_float32),
        ):
            raw["tactile"]["left_tactile"]["rgb_marker"] = left
            raw["tactile"]["right_tactile"]["rgb_marker"] = right

        converted_uint8 = self._convert(raw_uint8)
        converted_float32 = self._convert(raw_float32)
        for slot, uint8_source, float32_source in (
            ("left", left_uint8, left_float32),
            ("right", right_uint8, right_float32),
        ):
            uint8_payload = converted_uint8.record.observation.sensor(slot).payload
            float32_payload = converted_float32.record.observation.sensor(slot).payload
            np.testing.assert_array_equal(uint8_payload, float32_payload)
            self.assertFalse(np.shares_memory(uint8_payload, uint8_source))
            self.assertFalse(np.shares_memory(float32_payload, float32_source))
            self.assertEqual(
                converted_uint8.record.provenance_for(slot).payload_sha256,
                converted_float32.record.provenance_for(slot).payload_sha256,
            )

        self.assertEqual(
            converted_uint8.record.clean_record_sha256,
            converted_float32.record.clean_record_sha256,
        )
        self.assertEqual(
            converted_uint8.record.delivered_record_sha256,
            converted_float32.record.delivered_record_sha256,
        )
        self.assertEqual(
            converted_uint8.simulator_state_sha256,
            converted_float32.simulator_state_sha256,
        )
        for source, snapshot in zip(inputs, snapshots):
            np.testing.assert_array_equal(source, snapshot)

    def test_tactile_rgb_bridge_rejects_invalid_values_on_both_sensors(self) -> None:
        shape = self.config.tactile_rgb_shape
        invalid_cases = (
            (np.full(shape, 60.0, dtype=np.float64), "field_dtype"),
            (np.full(shape, 60.5, dtype=np.float32), "field_fractional"),
            (np.full(shape, -1.0, dtype=np.float32), "field_range"),
            (np.full(shape, 256.0, dtype=np.float32), "field_range"),
            (np.full(shape, np.nan, dtype=np.float32), "field_nonfinite"),
            (np.full(shape, np.inf, dtype=np.float32), "field_nonfinite"),
            (np.zeros((shape[0] - 1, *shape[1:]), dtype=np.float32), "field_shape"),
        )
        for sensor_name in ("left_tactile", "right_tactile"):
            for invalid, expected_code in invalid_cases:
                with self.subTest(sensor=sensor_name, code=expected_code):
                    raw = _raw(live_joint_names=self.live_names, cuda_like=False)
                    raw["tactile"][sensor_name]["rgb_marker"] = invalid
                    with self.assertRaises(UniVTACConversionError) as raised:
                        self._convert(raw)
                    self.assertEqual(raised.exception.code, expected_code)

    def test_camera_rgb_does_not_use_the_tactile_float32_bridge(self) -> None:
        for camera_name in ("head", "wrist"):
            with self.subTest(camera=camera_name):
                raw = _raw(live_joint_names=self.live_names, cuda_like=False)
                raw["observation"][camera_name]["rgb"] = np.zeros(
                    (270, 480, 3), dtype=np.float32
                )
                with self.assertRaises(UniVTACConversionError) as raised:
                    self._convert(raw)
                self.assertEqual(raised.exception.code, "field_dtype")

    def test_alias_shape_and_nonfinite_errors_fail_closed(self) -> None:
        missing = _raw(live_joint_names=self.live_names, cuda_like=False)
        del missing["tactile"]["left_tactile"]
        with self.assertRaisesRegex(UniVTACConversionError, "left_tactile"):
            self._convert(missing)

        malformed = _raw(live_joint_names=self.live_names, cuda_like=False)
        malformed["observation"]["head"]["rgb"] = np.zeros((10, 10, 3), np.uint8)
        with self.assertRaisesRegex(UniVTACConversionError, "head RGB shape"):
            self._convert(malformed)

        nonfinite = _raw(live_joint_names=self.live_names, cuda_like=False)
        nonfinite["tactile"]["right_tactile"]["depth"][0, 0] = np.nan
        with self.assertRaisesRegex(UniVTACConversionError, "finite"):
            self._convert(nonfinite)

    def test_action_batch_is_exact_float32_finite_bounded_and_immutable(self) -> None:
        valid = np.zeros((3, 8), dtype=np.float32)
        valid[:, 3] = -1.0
        accepted = validate_action_batch(valid, self.config)
        self.assertFalse(accepted.flags.writeable)

        cases = (
            np.zeros((8,), dtype=np.float32),
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((2, 7), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float64),
            np.full((2, 8), np.nan, dtype=np.float32),
            np.full((2, 8), 100.0, dtype=np.float32),
        )
        for actions in cases:
            with (
                self.subTest(shape=actions.shape, dtype=actions.dtype),
                self.assertRaises(UniVTACConversionError),
            ):
                validate_action_batch(actions, self.config)

    def _convert(self, raw: dict[str, Any]):
        return convert_raw_observation(
            raw,
            config=self.config,
            handshake=self.handshake,
            phase_states=self.phase_states,
            episode_id="episode-1",
            task_id="pull_out_key",
            initial_seed=11,
            benchmark_step=0,
        )


if __name__ == "__main__":
    unittest.main()

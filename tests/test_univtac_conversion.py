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

    def test_joint9_reorders_by_names_and_validates_shared_gripper(self) -> None:
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

        with self.assertRaisesRegex(UniVTACConversionError, "shared gripper"):
            reorder_joint9(
                np.asarray(
                    [0.02, 0.01, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    dtype=np.float32,
                ),
                self.live_names,
                self.config.canonical_joint_names,
            )

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

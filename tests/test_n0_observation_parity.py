from __future__ import annotations

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam.observation_parity import (
    action_tracking_summary,
    color_order_witness,
    summarize_image_stream,
    temporal_contract_summary,
)


def _frames(rgb: tuple[int, int, int], count: int = 3) -> np.ndarray:
    value = np.empty((count, 4, 5, 3), dtype=np.uint8)
    value[...] = rgb
    return value


def test_image_summary_exposes_duplicate_and_dynamic_frames() -> None:
    static = _frames((10, 20, 30))
    dynamic = static.copy()
    dynamic[1, 1, 1] = (20, 30, 40)

    static_summary = summarize_image_stream(static)
    dynamic_summary = summarize_image_stream(dynamic)

    assert static_summary.unique_frame_count == 1
    assert static_summary.temporal_mae_mean == 0.0
    assert dynamic_summary.unique_frame_count == 2
    assert dynamic_summary.temporal_mae_mean > 0.0
    assert dynamic_summary.spatial_gradient_mean > 0.0


def test_color_order_witness_prefers_direct_rgb_or_reversal() -> None:
    training = _frames((200, 100, 20))

    identity = color_order_witness(training, training.copy())
    reversed_input = color_order_witness(training, training[..., ::-1].copy())

    assert identity.preferred_transform == "identity"
    assert identity.identity_rgb_mean_mae == 0.0
    assert reversed_input.preferred_transform == "reverse_rgb"
    assert reversed_input.reverse_rgb_mean_mae == 0.0


def test_temporal_contract_separates_nominal_and_physical_rates() -> None:
    report = temporal_contract_summary(
        np.asarray((100, 102, 104, 106), dtype=np.int64),
        checkpoint_fps=30,
        sim_hz=120,
        live_native_steps=np.asarray((500, 512, 524, 536), dtype=np.int64),
    )

    assert report["training_physical_action_hz"] == 60.0
    assert report["training_physical_keyframe_hz"] == 20.0
    assert report["checkpoint_nominal_keyframe_hz"] == 10.0
    assert report["live_physical_action_hz"] == 10.0
    assert report["live_to_training_physical_action_hz_ratio"] == pytest.approx(1 / 6)


def test_temporal_contract_counts_live_decimation_as_physics_time() -> None:
    report = temporal_contract_summary(
        np.asarray((100, 102, 104, 106), dtype=np.int64),
        checkpoint_fps=30,
        sim_hz=120,
        live_native_steps=np.asarray((500, 501, 502, 503), dtype=np.int64),
        live_physics_steps_per_native_step=2,
    )

    assert report["live_physical_action_hz"] == 60.0
    assert report["live_to_training_physical_action_hz_ratio"] == 1.0


def test_action_tracking_measures_endpoint_not_target_path_quality() -> None:
    target = np.asarray(
        [
            (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.01),
            (0.2, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02),
        ],
        dtype=np.float32,
    )
    observed = target.copy()
    observed[1, 0] += np.float32(0.001)
    observed[1, 7] += np.float32(0.0002)

    report = action_tracking_summary(target, observed)

    assert report["count"] == 2
    assert report["translation_l2_m"]["maximum"] == pytest.approx(0.001, rel=1e-5)
    assert report["gripper_abs_m"]["maximum"] == pytest.approx(0.0002, rel=1e-5)
    assert report["rotation_geodesic_deg"]["maximum"] == 0.0


@pytest.mark.parametrize(
    "value",
    (
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((2, 3, 3, 3), dtype=np.float32),
    ),
)
def test_image_summary_rejects_wrong_contract(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8"):
        summarize_image_stream(value)

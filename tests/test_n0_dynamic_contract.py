from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam.dynamic_contract import (
    cadence_gate,
    joint_state_error,
    registered_image_metrics,
    select_nearest_state_match,
    state_match_at_index,
    stream_freshness,
    tactile_depth_summary,
)


def _states() -> np.ndarray:
    states = np.zeros((4, 8), dtype=np.float32)
    states[:, 0] = (0.40, 0.45, 0.50, 0.55)
    states[:, 3] = 1.0
    states[:, 7] = 0.02
    return states


def test_state_match_is_quaternion_sign_invariant_and_reserves_successor() -> None:
    live = _states()[2].copy()
    live[3:7] *= -1.0

    match = select_nearest_state_match(_states(), live)

    assert match.index == 2
    assert match.successor_index == 3
    assert match.passed is True
    assert match.rotation_geodesic_deg == 0.0


def test_state_match_exposes_failed_translation_gate() -> None:
    live = _states()[1].copy()
    live[1] = np.float32(0.01)

    match = select_nearest_state_match(_states(), live)

    assert match.index == 1
    assert match.translation_l2_m == pytest.approx(0.01)
    assert match.passed is False


def test_state_match_at_index_preserves_selected_expert_identity() -> None:
    live = _states()[2].copy()

    match = state_match_at_index(_states(), live, 2)

    assert match.index == 2
    assert match.successor_index == 3
    assert match.passed is True


def test_dynamic_contract_reports_joint_stream_and_cadence_evidence() -> None:
    reference = np.zeros(9, dtype=np.float32)
    observed = reference.copy()
    observed[4] = np.float32(0.01)
    errors = joint_state_error(reference, observed)
    assert errors["maximum_abs"] == pytest.approx(0.01)

    before = {
        name: np.zeros((4, 5, 3), dtype=np.uint8)
        for name in ("top", "wrist_l", "tactile_a", "tactile_b")
    }
    after = {name: value.copy() for name, value in before.items()}
    after["top"][0, 0] = (1, 2, 3)
    freshness = stream_freshness(before, after)
    assert freshness["top"]["changed"] is True
    assert freshness["wrist_l"]["changed"] is False

    cadence = cadence_gate(
        {
            "native_step_delta": 2,
            "physics_step_delta": 2,
            "n0_fixed_cadence": {
                "render_contract": "one_endpoint_render_no_intermediate_render_v1",
                "status": "Success",
                "stock_move_loop_used": False,
            },
        },
        sim_hz=120,
        decimation=1,
        physics_steps_per_action=2,
    )
    assert cadence["passed"] is True
    assert cadence["action_endpoint_hz"] == 60.0
    assert cadence["feedback_keyframe_hz"] == 20.0
    assert cadence["required"]["render_contract"] == (
        "one_endpoint_render_no_intermediate_render_v1"
    )


def test_registered_pixels_and_tactile_depth_are_numerical_evidence() -> None:
    reference = np.zeros((8, 9, 3), dtype=np.uint8)
    reference[:, 4:] = 200
    observed = reference.copy()
    image_metrics = registered_image_metrics(reference, observed)
    assert image_metrics["identical"] is True
    assert image_metrics["psnr_db"] is None
    assert image_metrics["edge_f1"] == 1.0
    assert image_metrics["border_px"] == 0
    assert image_metrics["evaluated_shape"] == [8, 9, 3]

    depth = np.full((4, 5), 30.0, dtype=np.float32)
    depth[1:3, 2:4] = 29.0
    tactile = tactile_depth_summary(depth, far_plane_mm=30.0, contact_threshold_mm=0.5)
    assert tactile["contact_area_pixels"] == 4
    assert tactile["deformation_centroid_yx"] == pytest.approx([1.5, 2.5])


def test_registered_image_metrics_excludes_the_requested_border() -> None:
    reference = np.zeros((8, 9, 3), dtype=np.uint8)
    observed = reference.copy()
    observed[[0, -1], :, :] = 255
    observed[:, [0, -1], :] = 255

    complete = registered_image_metrics(reference, observed)
    interior = registered_image_metrics(reference, observed, border_px=1)

    assert complete["identical"] is False
    assert interior["identical"] is True
    assert interior["mae"] == 0.0
    assert interior["psnr_db"] is None
    assert interior["border_px"] == 1
    assert interior["evaluated_shape"] == [6, 7, 3]


@pytest.mark.parametrize("value", (True, -1, 1.5, "1"))
def test_registered_image_metrics_rejects_invalid_border_type_or_value(
    value: object,
) -> None:
    image = np.zeros((8, 9, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative integer"):
        registered_image_metrics(image, image, border_px=cast(int, value))


@pytest.mark.parametrize("border_px", (4, 5))
def test_registered_image_metrics_rejects_border_without_an_interior(
    border_px: int,
) -> None:
    image = np.zeros((8, 9, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="removes the complete"):
        registered_image_metrics(image, image, border_px=border_px)

from __future__ import annotations

import numpy as np
import pytest

from robotactile_benchmark.recorded.n0_temporal_diagnostics import (
    ChunkHorizonSlice,
    build_cold_expert_native_action,
    build_expert_native_action,
    causal_phase_id,
    causal_phase_truth_table,
    check_first_warm_tactile_temporal_invariant,
    chunk_horizon_slice,
    horizon_action_metrics,
    select_warm_keyframe_indices,
    validate_emitted_horizons,
)


def _ee8_rows(count: int) -> np.ndarray:
    rows = np.zeros((count, 8), dtype=np.float32)
    rows[:, 0] = np.arange(count, dtype=np.float32) / 100.0
    rows[:, 3] = 1.0
    rows[:, 7] = np.arange(count, dtype=np.float32) / 1000.0
    return rows


def _state20(value: object) -> object:
    ee8 = np.asarray(value, dtype=np.float32)
    state = np.zeros(20, dtype=np.float32)
    state[:8] = ee8
    return state


def test_expert_history_builder_packs_two_selected_horizons_frame_major() -> None:
    expert = _ee8_rows(40)
    native = build_expert_native_action(
        expert,
        frame_start_indices=(2, 18),
        ee8_to_state20=_state20,
    )

    assert native.shape == (20, 2, 12)
    assert native.dtype == np.float32
    assert not native.flags.writeable
    assert native[0, 0].tolist() == pytest.approx(expert[2:14, 0].tolist())
    assert native[0, 1].tolist() == pytest.approx(expert[18:30, 0].tolist())
    assert native[7, 1, -1] == pytest.approx(expert[29, 7])


def test_expert_history_builder_rejects_uncovered_or_bad_state20() -> None:
    with pytest.raises(ValueError, match="cover both"):
        build_expert_native_action(
            _ee8_rows(20),
            frame_start_indices=(0, 12),
            ee8_to_state20=_state20,
        )

    def bad_encoder(value: object) -> object:
        del value
        return np.zeros(19, dtype=np.float32)

    with pytest.raises(ValueError, match=r"float32 \[20\]"):
        build_expert_native_action(
            _ee8_rows(24),
            frame_start_indices=(0, 12),
            ee8_to_state20=bad_encoder,
        )


def test_cold_expert_builder_repeats_anchor_then_packs_anchor_horizon() -> None:
    expert = _ee8_rows(30)
    native = build_cold_expert_native_action(
        expert,
        anchor_index=7,
        ee8_to_state20=_state20,
    )

    assert native.shape == (20, 2, 12)
    assert np.all(native[0, 0] == expert[7, 0])
    assert native[0, 1].tolist() == pytest.approx(expert[7:19, 0].tolist())
    assert native[7, 1].tolist() == pytest.approx(expert[7:19, 7].tolist())
    assert not native.flags.writeable


def test_cold_expert_builder_rejects_uncovered_horizon() -> None:
    with pytest.raises(ValueError, match="cold 12-step"):
        build_cold_expert_native_action(
            _ee8_rows(15),
            anchor_index=4,
            ee8_to_state20=_state20,
        )


def test_anchor_174_to_target_186_selects_four_causal_keyframes() -> None:
    assert select_warm_keyframe_indices(anchor_index=174, target_index=186) == (
        177,
        180,
        183,
        186,
    )
    with pytest.raises(ValueError, match="target_index"):
        select_warm_keyframe_indices(anchor_index=174, target_index=185)


def test_cold_and_warm_horizon_slices_match_runtime_execution() -> None:
    full = _ee8_rows(24)
    cold = chunk_horizon_slice("cold")
    warm = chunk_horizon_slice("warm")

    assert (cold.flat_start, cold.flat_stop) == (12, 24)
    assert (warm.flat_start, warm.flat_stop) == (0, 24)
    assert np.array_equal(cold.select(full), full[12:24])
    assert np.array_equal(warm.select(full), full)
    assert validate_emitted_horizons(full[12:24], "cold").shape == (12, 8)
    assert validate_emitted_horizons(full, "warm").shape == (24, 8)
    with pytest.raises(ValueError, match=r"cold.*\(12, 8\)"):
        validate_emitted_horizons(full, "cold")
    with pytest.raises(ValueError, match="disagrees"):
        ChunkHorizonSlice("cold", 0, 2)


def test_horizon_metrics_reports_each_row_and_aggregate() -> None:
    target = _ee8_rows(3)
    predicted = target.copy()
    predicted[1, 0] += 0.1
    predicted[1, 3:7] = np.asarray(
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=np.float32
    )
    predicted[1, 7] += 0.2
    predicted[2, 0] += 0.2
    predicted[2, 3:7] *= -1.0

    metrics = horizon_action_metrics(predicted, target)

    assert tuple(row.horizon_index for row in metrics.horizons) == (0, 1, 2)
    assert metrics.horizons[1].translation_l2_m == pytest.approx(0.1)
    assert metrics.horizons[1].rotation_geodesic_deg == pytest.approx(90.0)
    assert metrics.horizons[1].gripper_abs == pytest.approx(0.2)
    assert metrics.horizons[2].rotation_geodesic_deg == pytest.approx(0.0)
    assert metrics.aggregate.translation_mean_l2_m == pytest.approx(0.1)
    assert metrics.aggregate.translation_max_l2_m == pytest.approx(0.2)
    assert metrics.aggregate.rotation_max_geodesic_deg == pytest.approx(90.0)


def test_horizon_metrics_preserves_absolute_indices_for_a_slice() -> None:
    metrics = horizon_action_metrics(
        _ee8_rows(12), _ee8_rows(12), horizon_start=3, horizon_stop=6
    )
    assert tuple(row.horizon_index for row in metrics.horizons) == (3, 4, 5)
    assert metrics.aggregate.horizon_count == 3


def test_causal_phase_truth_table_exposes_even_vision_tactile_and_odd_action() -> None:
    assert causal_phase_id("video", 3) == 6
    assert causal_phase_id("tactile", 3) == 6
    assert causal_phase_id("action", 3) == 7
    table = {
        (row.query_modality, row.key_modality): row
        for row in causal_phase_truth_table(3)
    }

    action_to_tactile = table[("action", "tactile")]
    assert action_to_tactile.training_noisy_to_clean is True
    assert action_to_tactile.serve_query_to_clean_cache is True
    tactile_to_action = table[("tactile", "action")]
    assert tactile_to_action.training_clean_to_clean is False
    assert tactile_to_action.serve_query_to_current_noisy is False
    video_to_tactile = table[("video", "tactile")]
    assert video_to_tactile.training_noisy_to_clean is False
    assert video_to_tactile.training_noisy_to_noisy is True
    assert video_to_tactile.serve_query_to_current_noisy is True


def test_first_warm_tactile_checker_reports_f2_f2_f1_mismatch() -> None:
    mismatch = check_first_warm_tactile_temporal_invariant(
        video_frames=2, action_frames=2, tactile_frames=1
    )

    assert mismatch.passed is False
    assert mismatch.violation_codes == (
        "FIRST_WARM_TACTILE_FRAME_COUNT",
        "FIRST_WARM_MODALITY_FRAME_MISMATCH",
    )
    assert "video F=2, action F=2, tactile F=1" in mismatch.detail
    assert check_first_warm_tactile_temporal_invariant(
        video_frames=2, action_frames=2, tactile_frames=2
    ).passed

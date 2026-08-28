from __future__ import annotations

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam.train_serve_gap import (
    array_parity_metrics,
    freeze_float32,
    sampled_source_indices,
    temporal_chunk_slices,
)


def test_temporal_chunk_slices_matches_live_cold_then_warm_contract() -> None:
    assert temporal_chunk_slices(9) == ((0, 1), (1, 5), (5, 9))


@pytest.mark.parametrize("frame_count", (0, 2, 6, 8))
def test_temporal_chunk_slices_rejects_non_wan_history(frame_count: int) -> None:
    with pytest.raises(ValueError):
        temporal_chunk_slices(frame_count)


def test_sampled_source_indices_ends_at_anchor() -> None:
    assert sampled_source_indices(
        anchor_index=186, keyframe_count=9, source_stride=3
    ) == (162, 165, 168, 171, 174, 177, 180, 183, 186)


def test_sampled_source_indices_rejects_history_before_episode() -> None:
    with pytest.raises(ValueError, match="before the episode"):
        sampled_source_indices(anchor_index=3, keyframe_count=9, source_stride=3)


def test_array_parity_metrics_reports_scale_aware_errors() -> None:
    reference = np.asarray((1.0, 2.0, 3.0), dtype=np.float32)
    candidate = np.asarray((1.0, 1.0, 4.0), dtype=np.float32)
    metrics = array_parity_metrics(reference, candidate)
    assert metrics.shape == (3,)
    assert metrics.mean_abs == pytest.approx(2.0 / 3.0)
    assert metrics.root_mean_square == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics.max_abs == 1.0
    assert 0.0 < metrics.relative_l2 < 1.0


def test_array_parity_metrics_rejects_shape_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        array_parity_metrics(np.zeros((2,)), np.zeros((3,)))
    with pytest.raises(ValueError, match="finite"):
        array_parity_metrics(np.zeros((2,)), np.asarray((0.0, np.nan)))


def test_freeze_float32_returns_immutable_canonical_array() -> None:
    result = freeze_float32((1.0, 2.0))
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert not result.flags.writeable

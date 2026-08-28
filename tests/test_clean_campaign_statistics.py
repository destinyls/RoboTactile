"""Dependency-light statistics used by clean campaign summaries."""

from __future__ import annotations

import pytest

from robotactile_benchmark.reporting.statistics import (
    task_stratified_success_bootstrap,
    wilson_score_interval,
)


def test_wilson_interval_matches_known_small_sample_value() -> None:
    interval = wilson_score_interval(5, 10, confidence_level=0.95)

    assert interval.estimate == 0.5
    assert interval.lower == pytest.approx(0.2365930905)
    assert interval.upper == pytest.approx(0.7634069095)
    assert interval.success_count == 5
    assert interval.trial_count == 10


def test_task_stratified_success_bootstrap_is_seeded_and_equal_weighted() -> None:
    outcomes = {
        "large": (True, True, True, False, False, False),
        "small": (True, False),
    }

    first = task_stratified_success_bootstrap(
        outcomes,
        n_resamples=500,
        confidence_level=0.95,
        seed=17,
    )
    second = task_stratified_success_bootstrap(
        outcomes,
        n_resamples=500,
        confidence_level=0.95,
        seed=17,
    )

    assert first == second
    assert first.estimate == 0.5
    assert first.task_count == 2
    assert first.pair_count == 8


@pytest.mark.parametrize(("successes", "trials"), ((-1, 2), (3, 2), (0, 0)))
def test_wilson_rejects_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        wilson_score_interval(successes, trials, confidence_level=0.95)

"""Dependency-light paired statistics for benchmark reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _confidence(value: object) -> float:
    result = _finite(value, "confidence_level")
    if not 0.0 < result < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    return result


@dataclass(frozen=True)
class IntervalEstimate:
    """One deterministic percentile interval and its sampling inventory."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    valid_resamples: int
    task_count: int
    pair_count: int


@dataclass(frozen=True)
class McNemarResult:
    """Exact two-sided McNemar result for paired binary outcomes."""

    reference_only_successes: int
    comparison_only_successes: int
    p_value: float
    paired_success_delta: float
    pair_count: int


@dataclass(frozen=True)
class KendallResult:
    """Kendall tau-b with explicit pair counts."""

    tau: float
    concordant: int
    discordant: int
    tied_x_only: int
    tied_y_only: int
    pair_count: int


def task_stratified_paired_bootstrap(
    differences_by_task: Mapping[str, Sequence[float]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> IntervalEstimate:
    """Bootstrap pairs within task, then macro-average tasks equally."""

    n_resamples = _positive_int(n_resamples, "n_resamples")
    confidence_level = _confidence(confidence_level)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if not differences_by_task:
        raise ValueError("differences_by_task must be non-empty")
    normalized: dict[str, NDArray[np.float64]] = {}
    for task in sorted(differences_by_task):
        if not isinstance(task, str) or not task:
            raise ValueError("task names must be non-empty strings")
        values = np.asarray(
            [
                _finite(value, f"differences_by_task[{task!r}]")
                for value in differences_by_task[task]
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError("each task stratum must contain at least one pair")
        normalized[task] = values
    estimate = float(np.mean([values.mean() for values in normalized.values()]))
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    samples = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        task_means = []
        for values in normalized.values():
            selected = generator.integers(0, values.size, size=values.size)
            task_means.append(float(values[selected].mean()))
        samples[index] = float(np.mean(task_means))
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, (alpha, 1.0 - alpha), method="linear")
    return IntervalEstimate(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        valid_resamples=n_resamples,
        task_count=len(normalized),
        pair_count=sum(values.size for values in normalized.values()),
    )


def exact_mcnemar(
    reference: Sequence[bool], comparison: Sequence[bool]
) -> McNemarResult:
    """Compute the exact two-sided binomial McNemar test."""

    if len(reference) != len(comparison):
        raise ValueError("paired outcomes must have the same length")
    if not reference:
        raise ValueError("paired outcomes must be non-empty")
    if any(type(item) is not bool for item in (*reference, *comparison)):
        raise TypeError("McNemar outcomes must be exact booleans")
    reference_only = sum(a and not b for a, b in zip(reference, comparison))
    comparison_only = sum(not a and b for a, b in zip(reference, comparison))
    discordant = reference_only + comparison_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(reference_only, comparison_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    delta = (sum(comparison) - sum(reference)) / len(reference)
    return McNemarResult(
        reference_only_successes=reference_only,
        comparison_only_successes=comparison_only,
        p_value=float(p_value),
        paired_success_delta=float(delta),
        pair_count=len(reference),
    )


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values for one preregistered family."""

    if not p_values:
        raise ValueError("p_values must be non-empty")
    normalized = {}
    for name, value in p_values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("hypothesis names must be non-empty strings")
        scalar = _finite(value, f"p_values[{name!r}]")
        if not 0.0 <= scalar <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
        normalized[name] = scalar
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    running = 0.0
    adjusted: dict[str, float] = {}
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> KendallResult:
    """Compute Kendall tau-b without an optional SciPy dependency."""

    if len(x) != len(y):
        raise ValueError("Kendall inputs must have the same length")
    if len(x) < 2:
        raise ValueError("Kendall inputs require at least two observations")
    x_values = tuple(_finite(value, "Kendall x") for value in x)
    y_values = tuple(_finite(value, "Kendall y") for value in y)
    concordant = discordant = tied_x = tied_y = 0
    for left in range(len(x_values) - 1):
        for right in range(left + 1, len(x_values)):
            x_sign = (x_values[right] > x_values[left]) - (
                x_values[right] < x_values[left]
            )
            y_sign = (y_values[right] > y_values[left]) - (
                y_values[right] < y_values[left]
            )
            if x_sign == 0 and y_sign == 0:
                continue
            if x_sign == 0:
                tied_x += 1
            elif y_sign == 0:
                tied_y += 1
            elif x_sign == y_sign:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x) * (concordant + discordant + tied_y)
    )
    if denominator == 0.0:
        raise ValueError("Kendall tau-b is undefined for the supplied ties")
    return KendallResult(
        tau=(concordant - discordant) / denominator,
        concordant=concordant,
        discordant=discordant,
        tied_x_only=tied_x,
        tied_y_only=tied_y,
        pair_count=len(x_values),
    )


def bootstrap_kendall_tau_b(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> IntervalEstimate:
    """Return a paired percentile interval for a jointly qualified ranking."""

    estimate = kendall_tau_b(x, y).tau
    n_resamples = _positive_int(n_resamples, "n_resamples")
    confidence_level = _confidence(confidence_level)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    x_values = tuple(_finite(value, "Kendall x") for value in x)
    y_values = tuple(_finite(value, "Kendall y") for value in y)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    sampled: list[float] = []
    for _ in range(n_resamples):
        indices = generator.integers(0, len(x_values), size=len(x_values))
        try:
            sampled.append(
                kendall_tau_b(
                    tuple(x_values[index] for index in indices),
                    tuple(y_values[index] for index in indices),
                ).tau
            )
        except ValueError:
            continue
    if not sampled:
        raise ValueError("no defined Kendall bootstrap resamples")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(sampled, (alpha, 1.0 - alpha), method="linear")
    return IntervalEstimate(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        valid_resamples=len(sampled),
        task_count=0,
        pair_count=len(x_values),
    )

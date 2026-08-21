"""Small analytic metrics matching the paper's registered semantics."""

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _strict_index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def psnr(prediction: NDArray[Any], target: NDArray[Any]) -> float:
    """Compute PSNR for arrays already registered to [0, 1]."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("PSNR arrays must have identical shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("PSNR arrays must be finite")
    if (
        prediction.min() < 0
        or prediction.max() > 1
        or target.min() < 0
        or target.max() > 1
    ):
        raise ValueError("PSNR arrays must lie in [0, 1]")
    mse = float(np.mean((prediction - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def tactile_gain_retention(
    clean: float, no_touch: float, fault: float, minimum_gain: float = 1e-8
) -> float:
    """Return TGR only when the matched clean tactile gain is positive."""

    clean = _finite_real(clean, "clean")
    no_touch = _finite_real(no_touch, "no_touch")
    fault = _finite_real(fault, "fault")
    minimum_gain = _finite_real(minimum_gain, "minimum_gain")
    if minimum_gain <= 0.0:
        raise ValueError("minimum_gain must be positive")
    gain = clean - no_touch
    if gain <= minimum_gain:
        raise ValueError("clean tactile gain is not sufficiently positive")
    return (fault - no_touch) / gain


def recovery_lag(
    quality: Sequence[float],
    clean_envelope: Sequence[float],
    fault_stop_index: int,
    tolerance: float,
    consecutive_steps: int,
) -> Optional[int]:
    """Find the first post-restoration run inside the clean envelope."""

    quality_values = tuple(
        _finite_real(value, f"quality[{index}]") for index, value in enumerate(quality)
    )
    envelope_values = tuple(
        _finite_real(value, f"clean_envelope[{index}]")
        for index, value in enumerate(clean_envelope)
    )
    fault_stop_index = _strict_index(fault_stop_index, "fault_stop_index")
    consecutive_steps = _strict_index(consecutive_steps, "consecutive_steps")
    tolerance = _finite_real(tolerance, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if len(quality_values) != len(envelope_values):
        raise ValueError("quality and clean envelope lengths must match")
    if consecutive_steps <= 0:
        raise ValueError("consecutive_steps must be positive")
    if fault_stop_index < 0 or fault_stop_index >= len(quality_values):
        raise ValueError("fault_stop_index must identify the first restored sample")
    start = fault_stop_index
    for index in range(start, len(quality_values) - consecutive_steps + 1):
        if all(
            abs(quality_values[offset] - envelope_values[offset]) <= tolerance
            for offset in range(index, index + consecutive_steps)
        ):
            return index - fault_stop_index
    return None

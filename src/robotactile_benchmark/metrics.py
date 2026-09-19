"""Small analytic metrics matching the paper's registered semantics."""

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


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

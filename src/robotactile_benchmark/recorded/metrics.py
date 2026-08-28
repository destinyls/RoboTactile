"""Action-space metrics for recorded N0-TWAM robustness evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from robotactile_benchmark.contracts import Array


@dataclass(frozen=True)
class ActionErrorMetrics:
    """Translation, orientation, and gripper errors over one EE8 horizon."""

    horizon_start: int
    horizon_stop: int
    translation_mean_l2_m: float
    translation_final_l2_m: float
    rotation_mean_geodesic_deg: float
    rotation_final_geodesic_deg: float
    gripper_mean_abs: float
    gripper_final_abs: float

    def to_dict(self) -> dict[str, Union[float, int]]:
        return {
            "horizon_start": self.horizon_start,
            "horizon_stop": self.horizon_stop,
            "translation_mean_l2_m": self.translation_mean_l2_m,
            "translation_final_l2_m": self.translation_final_l2_m,
            "rotation_mean_geodesic_deg": self.rotation_mean_geodesic_deg,
            "rotation_final_geodesic_deg": self.rotation_final_geodesic_deg,
            "gripper_mean_abs": self.gripper_mean_abs,
            "gripper_final_abs": self.gripper_final_abs,
        }


def _actions(value: Array, name: str) -> Array:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[1] != 8
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite float32 [H,8]")
    norms = np.linalg.norm(array[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(f"{name} quaternion rows must have unit norm")
    return array


def action_metrics(
    predicted: Array,
    target: Array,
    *,
    horizon_start: int = 0,
    horizon_stop: Optional[int] = None,
) -> ActionErrorMetrics:
    """Measure contract-faithful EE8 errors on a selected half-open horizon."""

    prediction = _actions(predicted, "predicted actions")
    reference = _actions(target, "target actions")
    if prediction.shape != reference.shape:
        raise ValueError("predicted and target action shapes must match")
    stop = prediction.shape[0] if horizon_stop is None else horizon_stop
    if not 0 <= horizon_start < stop <= prediction.shape[0]:
        raise ValueError("metric horizon is outside the action chunk")
    selected_prediction = prediction[horizon_start:stop]
    selected_reference = reference[horizon_start:stop]
    translation = np.linalg.norm(
        selected_prediction[:, :3] - selected_reference[:, :3], axis=1
    )
    dots = np.abs(
        np.sum(selected_prediction[:, 3:7] * selected_reference[:, 3:7], axis=1)
    )
    rotation = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    gripper = np.abs(selected_prediction[:, 7] - selected_reference[:, 7])
    return ActionErrorMetrics(
        horizon_start=horizon_start,
        horizon_stop=stop,
        translation_mean_l2_m=float(np.mean(translation)),
        translation_final_l2_m=float(translation[-1]),
        rotation_mean_geodesic_deg=float(np.mean(rotation)),
        rotation_final_geodesic_deg=float(rotation[-1]),
        gripper_mean_abs=float(np.mean(gripper)),
        gripper_final_abs=float(gripper[-1]),
    )


def finite_metrics(value: ActionErrorMetrics) -> bool:
    """Return whether every reported metric is a finite scalar."""

    return all(
        math.isfinite(item)
        for item in (
            value.translation_mean_l2_m,
            value.translation_final_l2_m,
            value.rotation_mean_geodesic_deg,
            value.rotation_final_geodesic_deg,
            value.gripper_mean_abs,
            value.gripper_final_abs,
        )
    )

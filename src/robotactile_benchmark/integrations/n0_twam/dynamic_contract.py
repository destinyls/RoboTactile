"""Pure contracts for the N0-to-UniVTAC one-step dynamic probe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from robotactile_benchmark.contracts import Array, array_sha256

TRANSLATION_GATE_M = 0.002
ROTATION_GATE_DEG = 1.0
GRIPPER_GATE_M = 0.0002
_STREAMS = ("top", "wrist_l", "tactile_a", "tactile_b")


def _ee8_rows(value: object, name: str) -> Array:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != 8
        or rows.shape[0] < 1
        or not np.isfinite(rows).all()
    ):
        raise ValueError(f"{name} must be finite float32 [N,8]")
    norms = np.linalg.norm(rows[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError(f"{name} quaternions must have unit norm")
    return rows


@dataclass(frozen=True)
class StateMatch:
    """Nearest expert row under the pre-registered robot-state gates."""

    index: int
    successor_index: int
    score: float
    translation_l2_m: float
    rotation_geodesic_deg: float
    gripper_abs_m: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "gates": {
                "gripper_abs_m_max": GRIPPER_GATE_M,
                "rotation_geodesic_deg_max": ROTATION_GATE_DEG,
                "translation_l2_m_max": TRANSLATION_GATE_M,
            },
            "gripper_abs_m": self.gripper_abs_m,
            "index": self.index,
            "passed": self.passed,
            "rotation_geodesic_deg": self.rotation_geodesic_deg,
            "score": self.score,
            "successor_index": self.successor_index,
            "translation_l2_m": self.translation_l2_m,
        }


def select_nearest_state_match(expert_states: object, live_state: object) -> StateMatch:
    """Select one HDF5 row while reserving its successor as a safe target."""

    expert = _ee8_rows(expert_states, "expert_states")
    if expert.shape[0] < 2:
        raise ValueError("expert_states must contain a successor row")
    live = _ee8_rows(np.asarray(live_state)[None, :], "live_state")[0]
    candidates = expert[:-1]
    translation = np.linalg.norm(candidates[:, :3] - live[:3], axis=1)
    dots = np.abs(np.sum(candidates[:, 3:7] * live[None, 3:7], axis=1))
    rotation = np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
    gripper = np.abs(candidates[:, 7] - live[7])
    scores = np.sqrt(
        np.square(translation / TRANSLATION_GATE_M)
        + np.square(rotation / ROTATION_GATE_DEG)
        + np.square(gripper / GRIPPER_GATE_M)
    )
    index = int(np.argmin(scores))
    translation_error = float(translation[index])
    rotation_error = float(rotation[index])
    gripper_error = float(gripper[index])
    return StateMatch(
        index=index,
        successor_index=index + 1,
        score=float(scores[index]),
        translation_l2_m=translation_error,
        rotation_geodesic_deg=rotation_error,
        gripper_abs_m=gripper_error,
        passed=(
            translation_error <= TRANSLATION_GATE_M
            and rotation_error <= ROTATION_GATE_DEG
            and gripper_error <= GRIPPER_GATE_M
        ),
    )


def state_match_at_index(
    expert_states: object, live_state: object, index: int
) -> StateMatch:
    """Evaluate one pre-selected HDF5 row under the same robot-state gates."""

    expert = _ee8_rows(expert_states, "expert_states")
    if expert.shape[0] < 2:
        raise ValueError("expert_states must contain a successor row")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(expert) - 1
    ):
        raise ValueError("index must select an expert row with a successor")
    live = _ee8_rows(np.asarray(live_state)[None, :], "live_state")[0]
    local_match = select_nearest_state_match(expert[index : index + 2], live)
    return StateMatch(
        index=index,
        successor_index=index + 1,
        score=local_match.score,
        translation_l2_m=local_match.translation_l2_m,
        rotation_geodesic_deg=local_match.rotation_geodesic_deg,
        gripper_abs_m=local_match.gripper_abs_m,
        passed=local_match.passed,
    )


def joint_state_error(reference: object, live: object) -> dict[str, object]:
    """Report joint-space mismatch without turning it into a hidden gate."""

    expected = np.asarray(reference)
    observed = np.asarray(live)
    if (
        expected.dtype != np.float32
        or observed.dtype != np.float32
        or expected.shape != (9,)
        or observed.shape != (9,)
        or not np.isfinite(expected).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("joint states must be finite float32 [9]")
    absolute = np.abs(expected - observed)
    return {
        "absolute_error": [float(item) for item in absolute],
        "maximum_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
    }


def _edge_mask(image: Array, threshold: float = 20.0) -> Array:
    gray = image.astype(np.float32).mean(axis=2)
    gradient = np.zeros_like(gray, dtype=np.float32)
    gradient[:, 1:] = np.maximum(gradient[:, 1:], np.abs(gray[:, 1:] - gray[:, :-1]))
    gradient[1:, :] = np.maximum(gradient[1:, :], np.abs(gray[1:, :] - gray[:-1, :]))
    return gradient >= threshold


def _dilate(mask: Array) -> Array:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return cast(
        Array,
        np.logical_or.reduce(
            tuple(
                padded[y : y + mask.shape[0], x : x + mask.shape[1]]
                for y in range(3)
                for x in range(3)
            )
        ),
    )


def registered_image_metrics(
    reference: object,
    candidate: object,
    *,
    border_px: int = 0,
) -> dict[str, object]:
    """Compute same-shape pixel and edge evidence inside an optional border."""

    if isinstance(border_px, bool) or not isinstance(border_px, int) or border_px < 0:
        raise ValueError("border_px must be a non-negative integer")
    expected = np.asarray(reference)
    observed = np.asarray(candidate)
    if (
        expected.dtype != np.uint8
        or observed.dtype != np.uint8
        or expected.ndim != 3
        or expected.shape[-1] != 3
        or observed.shape != expected.shape
    ):
        raise ValueError("registered images must be equal-shape uint8 HWC RGB")
    if border_px * 2 >= expected.shape[0] or border_px * 2 >= expected.shape[1]:
        raise ValueError("border_px removes the complete registered image")
    if border_px:
        expected = expected[border_px:-border_px, border_px:-border_px]
        observed = observed[border_px:-border_px, border_px:-border_px]
    difference = expected.astype(np.float32) - observed.astype(np.float32)
    mse = float(np.mean(np.square(difference)))
    reference_edges = _edge_mask(expected)
    candidate_edges = _edge_mask(observed)
    precision = float(
        np.sum(candidate_edges & _dilate(reference_edges))
        / max(int(np.sum(candidate_edges)), 1)
    )
    recall = float(
        np.sum(reference_edges & _dilate(candidate_edges))
        / max(int(np.sum(reference_edges)), 1)
    )
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )

    def centroid(mask: Array) -> tuple[float, float] | None:
        locations = np.argwhere(mask)
        if locations.size == 0:
            return None
        mean = locations.mean(axis=0)
        return float(mean[0]), float(mean[1])

    reference_centroid = centroid(reference_edges)
    candidate_centroid = centroid(candidate_edges)
    centroid_distance = (
        None
        if reference_centroid is None or candidate_centroid is None
        else float(
            np.linalg.norm(
                np.asarray(reference_centroid) - np.asarray(candidate_centroid)
            )
        )
    )
    return {
        "border_px": border_px,
        "edge_centroid_distance_px": centroid_distance,
        "edge_f1": f1,
        "edge_metric": "gray_gradient_threshold20_tolerance1_v1",
        "evaluated_shape": [int(value) for value in expected.shape],
        "identical": mse == 0.0,
        "mae": float(np.mean(np.abs(difference))),
        "mse": mse,
        "psnr_db": None if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse))),
    }


def tactile_depth_summary(
    depth: object, *, far_plane_mm: float, contact_threshold_mm: float
) -> dict[str, object]:
    """Summarize physical indentation without assuming bilateral symmetry."""

    value = np.asarray(depth)
    if (
        value.dtype != np.float32
        or value.ndim != 2
        or not np.isfinite(value).all()
        or not np.isfinite(far_plane_mm)
        or not np.isfinite(contact_threshold_mm)
        or contact_threshold_mm < 0.0
    ):
        raise ValueError("tactile depth contract is invalid")
    indentation = np.maximum(np.float32(0.0), np.float32(far_plane_mm) - value)
    mask = indentation >= np.float32(contact_threshold_mm)
    locations = np.argwhere(mask)
    centroid = None
    if locations.size:
        weights = indentation[mask].astype(np.float64)
        centroid = [
            float(np.average(locations[:, axis], weights=weights)) for axis in range(2)
        ]
    gradients = (
        np.abs(np.diff(indentation, axis=0)).mean()
        + np.abs(np.diff(indentation, axis=1)).mean()
    ) / 2.0
    return {
        "contact_area_fraction": float(np.mean(mask)),
        "contact_area_pixels": int(np.sum(mask)),
        "deformation_centroid_yx": centroid,
        "indentation_max_mm": float(np.max(indentation)),
        "indentation_mean_mm": float(np.mean(indentation)),
        "spatial_gradient_mean_mm": float(gradients),
    }


def stream_freshness(
    before: Mapping[str, Array], after: Mapping[str, Array]
) -> dict[str, object]:
    """Compare delivered stream content independently of camera counters."""

    if set(before) != set(_STREAMS) or set(after) != set(_STREAMS):
        raise ValueError("dynamic probe stream inventory mismatch")
    result: dict[str, object] = {}
    for name in _STREAMS:
        first = np.asarray(before[name])
        second = np.asarray(after[name])
        if (
            first.dtype != np.uint8
            or second.dtype != np.uint8
            or first.ndim != 3
            or first.shape[-1] != 3
            or second.shape != first.shape
        ):
            raise ValueError(f"{name} streams must be equal-shape uint8 HWC RGB")
        first_sha256 = array_sha256(first)
        second_sha256 = array_sha256(second)
        result[name] = {
            "changed": first_sha256 != second_sha256,
            "mae": float(
                np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32)))
            ),
            "post_sha256": second_sha256,
            "pre_sha256": first_sha256,
            "shape": list(first.shape),
        }
    return result


def cadence_gate(
    diagnostics: Mapping[str, object],
    *,
    sim_hz: int,
    decimation: int,
    physics_steps_per_action: int,
    action_rows_per_keyframe: int = 3,
) -> dict[str, object]:
    """Evaluate observed physics deltas against the explicitly supplied cadence."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (
            sim_hz,
            decimation,
            physics_steps_per_action,
            action_rows_per_keyframe,
        )
    ):
        raise ValueError("cadence parameters must be positive integers")
    native_steps_per_action, remainder = divmod(physics_steps_per_action, decimation)
    native_delta = diagnostics.get("native_step_delta")
    physics_delta = diagnostics.get("physics_step_delta")
    fixed = diagnostics.get("n0_fixed_cadence")
    if not isinstance(fixed, Mapping):
        fixed = {}
    passed = (
        remainder == 0
        and native_delta == native_steps_per_action
        and physics_delta == physics_steps_per_action
        and fixed.get("stock_move_loop_used") is False
        and fixed.get("render_contract")
        == "one_endpoint_render_no_intermediate_render_v1"
        and fixed.get("status") == "Success"
    )
    endpoint_hz = float(sim_hz / physics_steps_per_action)
    return {
        "action_endpoint_hz": endpoint_hz,
        "action_rows_per_keyframe": action_rows_per_keyframe,
        "feedback_keyframe_hz": endpoint_hz / action_rows_per_keyframe,
        "fixed_cadence_plan": dict(fixed),
        "native_step_delta": native_delta,
        "passed": passed,
        "physics_step_delta": physics_delta,
        "required": {
            "action_endpoint_hz": endpoint_hz,
            "feedback_keyframe_hz": endpoint_hz / action_rows_per_keyframe,
            "native_step_delta": native_steps_per_action,
            "physics_step_delta": physics_steps_per_action,
            "render_contract": "one_endpoint_render_no_intermediate_render_v1",
            "stock_move_loop_used": False,
        },
    }


__all__ = [
    "GRIPPER_GATE_M",
    "ROTATION_GATE_DEG",
    "StateMatch",
    "TRANSLATION_GATE_M",
    "cadence_gate",
    "joint_state_error",
    "registered_image_metrics",
    "select_nearest_state_match",
    "state_match_at_index",
    "stream_freshness",
    "tactile_depth_summary",
]

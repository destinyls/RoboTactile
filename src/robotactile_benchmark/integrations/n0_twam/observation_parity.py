"""Numerical evidence for N0 UniVTAC training-versus-live parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.contracts import Array, array_sha256


@dataclass(frozen=True)
class ImageStreamSummary:
    """Content and stability statistics for one RGB stream."""

    frame_count: int
    shape: Tuple[int, int, int]
    dtype: str
    rgb_mean: Tuple[float, float, float]
    rgb_std: Tuple[float, float, float]
    spatial_gradient_mean: float
    temporal_mae_mean: float
    unique_frame_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "frame_count": self.frame_count,
            "rgb_mean": list(self.rgb_mean),
            "rgb_std": list(self.rgb_std),
            "shape": list(self.shape),
            "spatial_gradient_mean": self.spatial_gradient_mean,
            "temporal_mae_mean": self.temporal_mae_mean,
            "unique_frame_count": self.unique_frame_count,
        }


@dataclass(frozen=True)
class ColorOrderWitness:
    """Channel-statistic evidence for identity versus R/B reversal."""

    identity_rgb_mean_mae: float
    reverse_rgb_mean_mae: float
    preferred_transform: str

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_rgb_mean_mae": self.identity_rgb_mean_mae,
            "preferred_transform": self.preferred_transform,
            "reverse_rgb_mean_mae": self.reverse_rgb_mean_mae,
            "reverse_minus_identity_mae": (
                self.reverse_rgb_mean_mae - self.identity_rgb_mean_mae
            ),
        }


def _rgb_frames(value: object, name: str) -> Array:
    frames = np.asarray(value)
    if (
        frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[0] < 1
        or frames.shape[-1] != 3
    ):
        raise ValueError(f"{name} must be non-empty uint8 [T,H,W,3]")
    return np.ascontiguousarray(frames)


def summarize_image_stream(value: object) -> ImageStreamSummary:
    """Summarize color, spatial high frequency, and temporal variation."""

    frames = _rgb_frames(value, "image stream")
    numeric = frames.astype(np.float32)
    horizontal = np.abs(np.diff(numeric, axis=2)).mean()
    vertical = np.abs(np.diff(numeric, axis=1)).mean()
    temporal = (
        0.0 if frames.shape[0] == 1 else float(np.abs(np.diff(numeric, axis=0)).mean())
    )
    flattened = numeric.reshape(-1, 3)
    rgb_mean = flattened.mean(axis=0)
    rgb_std = flattened.std(axis=0)
    return ImageStreamSummary(
        frame_count=int(frames.shape[0]),
        shape=(int(frames.shape[1]), int(frames.shape[2]), int(frames.shape[3])),
        dtype=str(frames.dtype),
        rgb_mean=(float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])),
        rgb_std=(float(rgb_std[0]), float(rgb_std[1]), float(rgb_std[2])),
        spatial_gradient_mean=float((horizontal + vertical) / 2.0),
        temporal_mae_mean=temporal,
        unique_frame_count=len({array_sha256(frame) for frame in frames}),
    )


def color_order_witness(training: object, live: object) -> ColorOrderWitness:
    """Compare per-frame RGB means without assuming pixel registration."""

    reference = _rgb_frames(training, "training frames")
    candidate = _rgb_frames(live, "live frames")
    count = min(reference.shape[0], candidate.shape[0])
    reference_mean = reference[:count].astype(np.float32).mean(axis=(1, 2))
    candidate_mean = candidate[:count].astype(np.float32).mean(axis=(1, 2))
    identity = float(np.abs(candidate_mean - reference_mean).mean())
    reverse = float(np.abs(candidate_mean[..., ::-1] - reference_mean).mean())
    preferred = "identity" if identity <= reverse else "reverse_rgb"
    return ColorOrderWitness(identity, reverse, preferred)


def temporal_contract_summary(
    training_native_steps: object,
    *,
    checkpoint_fps: int,
    sim_hz: int,
    live_native_steps: Optional[object],
    action_rows_per_keyframe: int = 3,
    live_physics_steps_per_native_step: int = 1,
) -> dict[str, object]:
    """Separate nominal checkpoint FPS from physical simulator cadence."""

    for value, name in (
        (checkpoint_fps, "checkpoint_fps"),
        (sim_hz, "sim_hz"),
        (action_rows_per_keyframe, "action_rows_per_keyframe"),
        (
            live_physics_steps_per_native_step,
            "live_physics_steps_per_native_step",
        ),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    training = np.asarray(training_native_steps)
    if (
        training.ndim != 1
        or training.size < 2
        or not np.issubdtype(training.dtype, np.integer)
    ):
        raise ValueError("training_native_steps must be an integer vector")
    training_delta = np.diff(training.astype(np.int64))
    if np.any(training_delta <= 0):
        raise ValueError("training native steps must increase")
    training_median = float(np.median(training_delta))
    training_physical_hz = float(sim_hz / training_median)
    result: dict[str, object] = {
        "action_rows_per_keyframe": action_rows_per_keyframe,
        "checkpoint_nominal_action_hz": float(checkpoint_fps),
        "checkpoint_nominal_keyframe_hz": float(
            checkpoint_fps / action_rows_per_keyframe
        ),
        "sim_hz": sim_hz,
        "live_physics_steps_per_native_step": (live_physics_steps_per_native_step),
        "training_native_step_delta": _delta_summary(training_delta),
        "training_physical_action_hz": training_physical_hz,
        "training_physical_keyframe_hz": float(
            training_physical_hz / action_rows_per_keyframe
        ),
        "training_nominal_to_physical_action_hz_ratio": float(
            checkpoint_fps / training_physical_hz
        ),
    }
    if live_native_steps is None:
        result.update(
            {
                "live_native_step_delta": None,
                "live_physical_action_hz": None,
                "live_physical_keyframe_hz": None,
                "live_to_training_physical_action_hz_ratio": None,
            }
        )
        return result
    live = np.asarray(live_native_steps)
    if live.ndim != 1 or live.size < 2 or not np.issubdtype(live.dtype, np.integer):
        raise ValueError("live_native_steps must be an integer vector")
    live_delta = np.diff(live.astype(np.int64))
    if np.any(live_delta <= 0):
        raise ValueError("live native steps must increase")
    live_hz = float(
        sim_hz / (float(np.median(live_delta)) * live_physics_steps_per_native_step)
    )
    result.update(
        {
            "live_native_step_delta": _delta_summary(live_delta),
            "live_physical_action_hz": live_hz,
            "live_physical_keyframe_hz": float(live_hz / action_rows_per_keyframe),
            "live_to_training_physical_action_hz_ratio": float(
                live_hz / training_physical_hz
            ),
        }
    )
    return result


def _delta_summary(value: Array) -> dict[str, float]:
    numeric = np.asarray(value, dtype=np.float64)
    return {
        "maximum": float(np.max(numeric)),
        "median": float(np.median(numeric)),
        "minimum": float(np.min(numeric)),
        "p95": float(np.percentile(numeric, 95)),
    }


def action_tracking_summary(targets: object, observations: object) -> dict[str, object]:
    """Measure whether UniVTAC reached each model-issued absolute EE8 target."""

    target = np.asarray(targets)
    observed = np.asarray(observations)
    if (
        target.dtype != np.float32
        or observed.dtype != np.float32
        or target.ndim != 2
        or target.shape[1:] != (8,)
        or observed.shape != target.shape
        or target.shape[0] < 1
        or not np.isfinite(target).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("targets and observations must be finite float32 [N,8]")
    translation = np.linalg.norm(target[:, :3] - observed[:, :3], axis=1)
    gripper = np.abs(target[:, 7] - observed[:, 7])
    target_quat = target[:, 3:7].astype(np.float64)
    observed_quat = observed[:, 3:7].astype(np.float64)
    denominator = np.linalg.norm(target_quat, axis=1) * np.linalg.norm(
        observed_quat, axis=1
    )
    if np.any(denominator <= 1e-8):
        raise ValueError("action tracking quaternions must be non-degenerate")
    cosine = np.abs(np.sum(target_quat * observed_quat, axis=1) / denominator)
    rotation = np.degrees(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "count": int(target.shape[0]),
        "gripper_abs_m": _error_summary(gripper),
        "rotation_geodesic_deg": _error_summary(rotation),
        "translation_l2_m": _error_summary(translation),
    }


def _error_summary(value: Array) -> dict[str, float]:
    numeric = np.asarray(value, dtype=np.float64)
    return {
        "maximum": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "p50": float(np.percentile(numeric, 50)),
        "p95": float(np.percentile(numeric, 95)),
    }


__all__ = [
    "ColorOrderWitness",
    "ImageStreamSummary",
    "action_tracking_summary",
    "color_order_witness",
    "summarize_image_stream",
    "temporal_contract_summary",
]

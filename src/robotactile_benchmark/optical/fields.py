"""Deterministic optical-field decomposition, without moving or cloning markers.

Dark-mask detection is a documented GelSight RGB proxy, not ground-truth marker
segmentation. Inpainting estimates shading under the dots; it does not infer
metric geometry. Only current-frame marker pixels are restored to delivered RGB.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from robotactile_benchmark.contracts import Array


def box_mean(values: Array, radius: int) -> Array:
    """Edge-padded box filter with O(HW) integral sums, for 2D or HWC input."""
    pad = ((radius, radius), (radius, radius)) + ((0, 0),) * (values.ndim - 2)
    padded = np.pad(values.astype(np.float64), pad, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)) + ((0, 0),) * (values.ndim - 2))
    integral = integral.cumsum(0).cumsum(1)
    size = 2 * radius + 1
    return cast(
        Array,
        (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )
        / size**2,
    )


def marker_mask(payload: Array, settings: Mapping[str, Any]) -> Array:
    """Protect dark marker cores plus a small guard; no coordinate transform."""
    threshold = int(settings["marker_max_channel"])
    mask = np.max(payload, axis=-1) <= threshold
    for _ in range(int(settings["marker_guard_px"])):
        padded = np.pad(mask, 1)
        mask = np.logical_or.reduce(
            [
                padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return cast(Array, mask)


def optical_field(payload: Array, settings: Mapping[str, Any]) -> tuple[Array, Array]:
    """Estimate marker-free shading using only nearby non-marker samples."""
    if payload.dtype != np.uint8 or payload.ndim != 3 or payload.shape[-1] != 3:
        raise ValueError("optical_marker_v1 requires uint8 HWC RGB")
    mask = marker_mask(payload, settings)
    current = payload.astype(np.float64) / 255.0
    radius = max(
        int(settings["minimum_inpaint_radius_px"]),
        int(round(min(payload.shape[:2]) * float(settings["inpaint_radius_fraction"]))),
    )
    valid = (~mask).astype(np.float64)
    weight = box_mean(valid, radius)
    numerator = box_mean(current * valid[..., None], radius)
    estimate = np.divide(
        numerator, weight[..., None], out=current.copy(), where=weight[..., None] > 1e-8
    )
    # A large uniformly dark region is not silently invented as a marker field.
    if np.any(mask & (weight <= 1e-8)):
        raise ValueError(
            "marker mask has no local shading support; sensor profile is inapplicable"
        )
    field = current.copy()
    field[mask] = estimate[mask]
    return field, mask


def restore_current_markers(payload: Array, field: Array, mask: Array) -> Array:
    result = np.rint(np.clip(field, 0.0, 1.0) * 255.0).astype(np.uint8)
    result[mask] = payload[mask]
    return cast(Array, result)


def spatial_weight(
    shape: tuple[int, ...], parameters: Mapping[str, Any], *, dead_patch: bool = False
) -> Array:
    height, width = shape[:2]
    yy, xx = np.mgrid[:height, :width]
    cx, cy = (float(v) for v in parameters["center_xy"])
    radius = np.hypot(xx - cx * (width - 1), yy - cy * (height - 1)) / min(
        height, width
    )
    if dead_patch:
        core = float(parameters["radius_fraction"])
        support = core + float(parameters["edge_feather_fraction"])
    else:
        core = float(parameters["core_radius_fraction"])
        support = float(parameters["support_radius_fraction"])
    phase = np.clip((radius - core) / (support - core), 0.0, 1.0)
    return cast(Array, 0.5 * (1.0 + np.cos(np.pi * phase)))

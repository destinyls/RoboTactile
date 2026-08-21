"""Deterministic single-frame transforms for fidelity operators F5 and F7."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np

from robotactile_benchmark.contracts import Array


def compact_horizontal_warp(
    payload: Array,
    center_xy: Tuple[float, float],
    support_radius_fraction: float,
    displacement_px: float,
) -> Array:
    """Backward-warp one current frame with a compact, fold-free field.

    Every delivered pixel samples exactly one spatial neighborhood from the
    current frame.  No rest frame or second marker lattice enters the result.
    Pixels outside the compact support are copied byte-for-byte.
    """

    height, width = payload.shape[:2]
    radius = min(height, width) * support_radius_fraction
    if radius <= 0.0:
        raise ValueError("warp support radius must be positive")
    center_x = center_xy[0] * (width - 1)
    center_y = center_xy[1] * (height - 1)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    radius_squared = ((xx - center_x) / radius) ** 2 + ((yy - center_y) / radius) ** 2
    support = radius_squared < 1.0
    weight = np.zeros((height, width), dtype=np.float64)
    weight[support] = (1.0 - radius_squared[support]) ** 2
    source_x = xx - displacement_px * weight
    if np.any(source_x[support] < 0.0) or np.any(source_x[support] > width - 1):
        raise ValueError("F5 source map leaves the current RGB frame")
    source_step = np.diff(source_x, axis=1)
    if source_step.size and float(source_step.min()) <= 0.0:
        raise ValueError("F5 source map folds at this image size and dose")

    output = payload.copy()
    if not np.any(support):
        return output
    left = np.floor(source_x).astype(np.int64)
    right = np.minimum(left + 1, width - 1)
    fraction = (source_x - left)[..., None]
    rows = np.arange(height, dtype=np.int64)[:, None]
    normalized = _normalize_float64(payload)
    sampled = (1.0 - fraction) * normalized[rows, left] + fraction * normalized[
        rows, right
    ]
    restored = _restore_like(payload, sampled)
    output[support] = restored[support]
    return output


def reflected_box_anchor(
    payload: Array,
    radius_fraction: float,
    minimum_radius_px: int,
) -> Array:
    """Return a float64 same-frame box anchor with reflect padding."""

    normalized = _normalize_float64(payload)
    height, width = payload.shape[:2]
    maximum_radius = (min(height, width) - 1) // 2
    requested = max(
        minimum_radius_px,
        int(np.floor(radius_fraction * min(height, width) + 0.5)),
    )
    radius = min(maximum_radius, requested)
    if radius < 1:
        raise ValueError("F7 input is too small for a reflected box anchor")

    row_indices = _reflected_indices(height, radius)
    column_indices = _reflected_indices(width, radius)
    vertical = np.zeros_like(normalized, dtype=np.float64)
    for row_index in row_indices:
        vertical += np.take(normalized, row_index, axis=0)
    vertical /= float(len(row_indices))
    anchor = np.zeros_like(vertical, dtype=np.float64)
    for column_index in column_indices:
        anchor += np.take(vertical, column_index, axis=1)
    anchor /= float(len(column_indices))
    return anchor


def same_frame_response_compression(
    payload: Array,
    response_knee: float,
    plateau_width_ratio: float,
    anchor_radius_fraction: float,
    minimum_anchor_radius_px: int,
) -> Array:
    """Compress high same-frame RGB detail without importing a rest lattice."""

    if response_knee <= 0.0 or plateau_width_ratio <= 0.0:
        raise ValueError("F7 response parameters must be positive")
    current = _normalize_float64(payload)
    anchor = reflected_box_anchor(
        payload, anchor_radius_fraction, minimum_anchor_radius_px
    )
    detail = current - anchor
    magnitude = np.linalg.norm(detail, axis=-1, keepdims=True)
    plateau = response_knee * plateau_width_ratio
    excess = np.maximum(magnitude - response_knee, 0.0)
    compressed = np.where(
        magnitude <= response_knee,
        magnitude,
        response_knee + plateau * excess / (plateau + excess),
    )
    scale = np.divide(
        compressed,
        magnitude,
        out=np.ones_like(compressed),
        where=magnitude > 1e-12,
    )
    return _restore_like(payload, anchor + scale * detail)


def _reflected_indices(length: int, radius: int) -> Tuple[Array, ...]:
    base = np.arange(length, dtype=np.int64)
    period = 2 * (length - 1)
    indices = []
    for offset in range(-radius, radius + 1):
        raw = np.mod(base + offset, period)
        indices.append(np.where(raw < length, raw, period - raw))
    return tuple(indices)


def _normalize_float64(payload: Array) -> Array:
    array = payload.astype(np.float64)
    if np.issubdtype(payload.dtype, np.integer) or float(array.max(initial=0.0)) > 1.5:
        array /= 255.0
    return cast(Array, np.clip(array, 0.0, 1.0))


def _restore_like(reference: Array, normalized: Array) -> Array:
    bounded = np.clip(normalized, 0.0, 1.0)
    if np.issubdtype(reference.dtype, np.integer):
        return cast(Array, np.rint(bounded * 255.0).astype(reference.dtype))
    return cast(Array, bounded.astype(reference.dtype))

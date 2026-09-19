"""Independent reference realization for pixel-signature validation.

This module intentionally does not import the operator implementations.  It is
a second, manifest-driven implementation used only by the delivery gate so a
bug in an injector cannot validate itself by replaying the same code path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Optional, Tuple, cast

import numpy as np

from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import RestReferenceBundle

PayloadKey = Tuple[int, str]


def _normalize(payload: Array) -> Array:
    array = payload.astype(np.float32)
    if np.issubdtype(payload.dtype, np.integer) or float(array.max(initial=0)) > 1.5:
        array /= 255.0
    return cast(Array, np.clip(array, 0.0, 1.0))


def _restore(reference: Array, value: Array) -> Array:
    bounded = np.clip(value, 0.0, 1.0)
    if np.issubdtype(reference.dtype, np.integer):
        return cast(Array, np.rint(bounded * 255.0).astype(reference.dtype))
    return cast(Array, bounded.astype(reference.dtype))


def _grid(payload: Array) -> Tuple[Array, Array]:
    yy, xx = np.mgrid[0 : payload.shape[0], 0 : payload.shape[1]]
    return yy.astype(np.float32), xx.astype(np.float32)


def _baseline(
    slot_id: str,
    manifest: FaultManifest,
    references: RestReferenceBundle,
) -> Array:
    if manifest.parameters["rest_reference_sha256"] != references.sha256:
        raise ValueError("rest-reference bundle does not match the fault manifest")
    return references.payload_for(slot_id)


def _f1(
    payload: Array,
    baseline: Array,
    manifest: FaultManifest,
    index: int,
) -> Array:
    if manifest.parameters.get("response_domain") == "absolute_black_frame":
        return np.zeros_like(payload)
    target = float(manifest.parameters["target_gain"])
    if manifest.parameters["temporal_path"] == "immediate_step":
        rest = _normalize(baseline)
        current = _normalize(payload)
        return _restore(payload, rest + target * (current - rest))
    span = max(1, manifest.stop_index - manifest.start_index - 1)
    progress = (index - manifest.start_index) / float(span)
    gain = 1.0 - (1.0 - target) * progress
    return _restore(payload, _normalize(payload) * gain)


def _f2(payload: Array, baseline: Array, manifest: FaultManifest) -> Array:
    current = _normalize(payload)
    rest = _normalize(baseline)
    yy, xx = _grid(payload)
    height, width = payload.shape[:2]
    center_x, center_y = (float(value) for value in manifest.parameters["center_xy"])
    sigma = max(
        float(manifest.parameters["minimum_sigma_px"]),
        min(height, width) * float(manifest.parameters["sigma_fraction"]),
    )
    heat = np.exp(
        -((xx - width * center_x) ** 2 + (yy - height * center_y) ** 2)
        / (2.0 * sigma**2)
    )
    retained = float(manifest.parameters["retained_gain"])
    gain = 1.0 - (1.0 - retained) * heat[..., None]
    return _restore(payload, rest + gain * (current - rest))


def _f3(payload: Array, manifest: FaultManifest) -> Array:
    current = _normalize(payload)
    yy, xx = _grid(payload)
    height = payload.shape[0]
    distance = np.abs(
        yy
        - (
            float(manifest.parameters["scar_slope"]) * xx
            + height * float(manifest.parameters["scar_intercept_fraction"])
        )
    )
    mask = distance <= float(manifest.parameters["scar_half_width_px"])
    delta = np.asarray(manifest.parameters["rgb_delta"], dtype=np.float32)
    return _restore(payload, current + mask[..., None] * delta)


def _f4(payload: Array, baseline: Array, manifest: FaultManifest) -> Array:
    output = payload.copy()
    yy, xx = _grid(payload)
    height, width = payload.shape[:2]
    center_x, center_y = (float(value) for value in manifest.parameters["center_xy"])
    radius = max(
        int(manifest.parameters["minimum_radius_px"]),
        int(round(min(height, width) * float(manifest.parameters["radius_fraction"]))),
    )
    mask = (xx - width * center_x) ** 2 + (yy - height * center_y) ** 2 <= radius**2
    output[mask] = baseline[mask]
    return output


def _f5(
    payload: Array,
    phase: ContactPhase,
    manifest: FaultManifest,
) -> Array:
    active_phases = {
        ContactPhase(value) for value in manifest.parameters["activation_phases"]
    }
    if phase not in active_phases:
        return payload.copy()
    height, width = payload.shape[:2]
    center_x, center_y = (float(value) for value in manifest.parameters["center_xy"])
    center_x *= width - 1
    center_y *= height - 1
    radius = min(height, width) * float(manifest.parameters["support_radius_fraction"])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    radius_squared = ((xx - center_x) / radius) ** 2 + ((yy - center_y) / radius) ** 2
    support = radius_squared < 1.0
    weight = np.zeros((height, width), dtype=np.float64)
    weight[support] = (1.0 - radius_squared[support]) ** 2
    source_x = xx - float(manifest.parameters["displacement_px"]) * weight
    if np.any(source_x[support] < 0.0) or np.any(source_x[support] > width - 1):
        raise ValueError("F5 reference source map leaves the RGB frame")
    if np.diff(source_x, axis=1).size and float(np.diff(source_x, axis=1).min()) <= 0:
        raise ValueError("F5 reference source map folds")
    left = np.floor(source_x).astype(np.int64)
    right = np.minimum(left + 1, width - 1)
    fraction = (source_x - left)[..., None]
    rows = np.arange(height, dtype=np.int64)[:, None]
    current = payload.astype(np.float64)
    if np.issubdtype(payload.dtype, np.integer) or float(current.max(initial=0)) > 1.5:
        current /= 255.0
    sampled = (1.0 - fraction) * current[rows, left] + fraction * current[rows, right]
    restored = _restore(payload, sampled)
    output = payload.copy()
    output[support] = restored[support]
    return output


def _reflect_indices(length: int, radius: int) -> Tuple[Array, ...]:
    base = np.arange(length, dtype=np.int64)
    period = 2 * (length - 1)
    output = []
    for offset in range(-radius, radius + 1):
        raw = np.mod(base + offset, period)
        output.append(np.where(raw < length, raw, period - raw))
    return tuple(output)


def _f7(
    payload: Array,
    phase: ContactPhase,
    manifest: FaultManifest,
) -> Array:
    active_phases = {
        ContactPhase(value) for value in manifest.parameters["activation_phases"]
    }
    if phase not in active_phases:
        return payload.copy()
    current = payload.astype(np.float64)
    if np.issubdtype(payload.dtype, np.integer) or float(current.max(initial=0)) > 1.5:
        current /= 255.0
    height, width = payload.shape[:2]
    requested_radius = max(
        int(manifest.parameters["minimum_anchor_radius_px"]),
        int(
            np.floor(
                float(manifest.parameters["anchor_radius_fraction"])
                * min(height, width)
                + 0.5
            )
        ),
    )
    radius = min((min(height, width) - 1) // 2, requested_radius)
    vertical = np.zeros_like(current, dtype=np.float64)
    row_indices = _reflect_indices(height, radius)
    for indices in row_indices:
        vertical += np.take(current, indices, axis=0)
    vertical /= float(len(row_indices))
    anchor = np.zeros_like(current, dtype=np.float64)
    column_indices = _reflect_indices(width, radius)
    for indices in column_indices:
        anchor += np.take(vertical, indices, axis=1)
    anchor /= float(len(column_indices))
    detail = current - anchor
    magnitude = np.linalg.norm(detail, axis=-1, keepdims=True)
    knee = float(manifest.parameters["response_knee"])
    plateau = knee * float(manifest.parameters["plateau_width_ratio"])
    excess = np.maximum(magnitude - knee, 0.0)
    compressed_magnitude = np.where(
        magnitude <= knee,
        magnitude,
        knee + plateau * excess / (plateau + excess),
    )
    scale = np.divide(
        compressed_magnitude,
        magnitude,
        out=np.ones_like(compressed_magnitude),
        where=magnitude > 1e-12,
    )
    return _restore(payload, anchor + scale * detail)


def _c2(payload: Array, baseline: Array, manifest: FaultManifest) -> Array:
    shift_x, shift_y = (
        int(value) for value in manifest.parameters["translation_xy_px"]
    )
    output = baseline.copy()
    height, width = payload.shape[:2]
    if shift_x < width and shift_y < height:
        output[shift_y:, shift_x:, :] = payload[: height - shift_y, : width - shift_x]
    return output


def expected_pixel_payloads(
    clean_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    references: Optional[RestReferenceBundle],
) -> Dict[PayloadKey, Array]:
    """Materialize expected payloads using the independent reference path."""

    output: Dict[PayloadKey, Array] = {}
    histories: Dict[str, Array] = {}
    if manifest.operator_id == "F6_history_residual_imprint":
        if references is None:
            raise ValueError("F6 reference requires a rest bundle")
        for slot_id in manifest.sensor_slots:
            histories[slot_id] = np.zeros_like(
                _normalize(_baseline(slot_id, manifest, references))
            )
    for index, record in enumerate(clean_records):
        for slot_id in manifest.sensor_slots:
            sensor = record.observation.sensor(slot_id)
            if sensor.payload is None:
                raise ValueError("pixel-signature reference requires present payloads")
            needs_baseline = (
                manifest.operator_id
                in {
                    "F1_global_response_drift",
                    "F2_spatial_sensitivity_loss",
                    "F4_local_nonresponsive_patch",
                    "F6_history_residual_imprint",
                    "C2_frame_misregistration",
                }
                and manifest.parameters.get("response_domain") != "absolute_black_frame"
            )
            if needs_baseline and references is None:
                raise ValueError(
                    f"{manifest.operator_id} reference requires a rest bundle"
                )
            baseline = (
                _baseline(slot_id, manifest, references)
                if needs_baseline and references is not None
                else sensor.payload
            )
            if manifest.operator_id == "F6_history_residual_imprint":
                rest = _normalize(baseline)
                residual = _normalize(sensor.payload) - rest
                beta = float(manifest.parameters["memory_beta"])
                history = beta * histories[slot_id] + (1.0 - beta) * residual
                histories[slot_id] = history
                if manifest.active(index):
                    mix = float(manifest.parameters["history_mix"])
                    output[index, slot_id] = _restore(
                        sensor.payload,
                        rest + (1.0 - mix) * residual + mix * history,
                    )
                continue
            if not manifest.active(index):
                continue
            operator_id = manifest.operator_id
            if operator_id == "F1_global_response_drift":
                expected = _f1(sensor.payload, baseline, manifest, index)
            elif operator_id == "F2_spatial_sensitivity_loss":
                expected = _f2(sensor.payload, baseline, manifest)
            elif operator_id == "F3_persistent_surface_artifact":
                expected = _f3(sensor.payload, manifest)
            elif operator_id == "F4_local_nonresponsive_patch":
                expected = _f4(sensor.payload, baseline, manifest)
            elif operator_id == "F5_contact_shape_distortion":
                expected = _f5(
                    sensor.payload,
                    record.provenance_for(slot_id).phase,
                    manifest,
                )
            elif operator_id == "F7_high_load_saturation":
                expected = _f7(
                    sensor.payload,
                    record.provenance_for(slot_id).phase,
                    manifest,
                )
            elif operator_id == "C2_frame_misregistration":
                expected = _c2(sensor.payload, baseline, manifest)
            else:
                raise ValueError(f"operator has no pixel reference: {operator_id}")
            output[index, slot_id] = expected
    return output

"""Single-record fidelity transforms with explicit streaming state."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import replace
from typing import cast

import numpy as np

from robotactile_benchmark.contracts import Array, ContactPhase, EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators.base import (
    baseline_for,
    clip_like,
    fault_declared_validity,
    mark_provenance,
    normalize,
    replace_delivery,
)
from robotactile_benchmark.operators.fidelity_transforms import (
    compact_horizontal_warp,
    same_frame_response_compression,
)
from robotactile_benchmark.rest_references import RestReferenceBundle


def apply_fidelity(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
    histories: MutableMapping[str, Array],
) -> EvaluationRecord:
    """Apply one fidelity operator without revisiting prior records."""

    operator_id = manifest.operator_id
    if operator_id == "F6_history_residual_imprint":
        return _apply_f6(clean_record, manifest, rest_references, histories)
    index = clean_record.observation.step_index
    if not manifest.active(index):
        return clean_record
    output = clean_record
    for slot_id in manifest.sensor_slots:
        if operator_id in {
            "F5_contact_shape_distortion",
            "F7_high_load_saturation",
        } and not _contact_active(output, slot_id, manifest):
            continue
        sensor = output.observation.sensor(slot_id)
        if sensor.payload is None:
            raise ValueError("fidelity operator requires a present payload")
        transformed = _transform_current(
            index,
            slot_id,
            sensor.payload,
            manifest,
            rest_references,
        )
        replacement = replace(
            sensor,
            payload=transformed,
            payload_present=True,
            declared_validity=fault_declared_validity(sensor, manifest),
        )
        provenance = mark_provenance(
            output.provenance_for(slot_id), manifest, transformed
        )
        output = replace_delivery(output, replacement, provenance)
    return output


def _transform_current(
    index: int,
    slot_id: str,
    payload: Array,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
) -> Array:
    operator_id = manifest.operator_id
    if operator_id == "F1_global_response_drift":
        if manifest.parameters["temporal_path"] == "immediate_step":
            if manifest.parameters.get("response_domain") == "absolute_black_frame":
                return np.zeros_like(payload)
            baseline = normalize(baseline_for(slot_id, manifest, rest_references))
            current = normalize(payload)
            target_gain = float(manifest.parameters["target_gain"])
            return clip_like(payload, baseline + target_gain * (current - baseline))
        target_gain = float(manifest.parameters["target_gain"])
        span = max(1, manifest.stop_index - manifest.start_index - 1)
        progress = (index - manifest.start_index) / float(span)
        gain = 1.0 - (1.0 - target_gain) * progress
        return clip_like(payload, normalize(payload) * gain)
    if operator_id == "F2_spatial_sensitivity_loss":
        return _spatial_sensitivity_loss(slot_id, payload, manifest, rest_references)
    if operator_id == "F3_persistent_surface_artifact":
        return _surface_artifact(payload, manifest)
    if operator_id == "F4_local_nonresponsive_patch":
        return _nonresponsive_patch(slot_id, payload, manifest, rest_references)
    if operator_id == "F5_contact_shape_distortion":
        center_xy = tuple(float(value) for value in manifest.parameters["center_xy"])
        return compact_horizontal_warp(
            payload,
            center_xy=(center_xy[0], center_xy[1]),
            support_radius_fraction=float(
                manifest.parameters["support_radius_fraction"]
            ),
            displacement_px=float(manifest.parameters["displacement_px"]),
        )
    if operator_id == "F7_high_load_saturation":
        return same_frame_response_compression(
            payload,
            response_knee=float(manifest.parameters["response_knee"]),
            plateau_width_ratio=float(manifest.parameters["plateau_width_ratio"]),
            anchor_radius_fraction=float(manifest.parameters["anchor_radius_fraction"]),
            minimum_anchor_radius_px=int(
                manifest.parameters["minimum_anchor_radius_px"]
            ),
        )
    raise KeyError(f"unknown streaming fidelity operator: {operator_id}")


def _spatial_sensitivity_loss(
    slot_id: str,
    payload: Array,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
) -> Array:
    baseline = normalize(baseline_for(slot_id, manifest, rest_references))
    current = normalize(payload)
    height, width = payload.shape[:2]
    yy, xx = _grid(height, width)
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
    return clip_like(payload, baseline + gain * (current - baseline))


def _surface_artifact(payload: Array, manifest: FaultManifest) -> Array:
    current = normalize(payload)
    height, width = payload.shape[:2]
    yy, xx = _grid(height, width)
    slope = float(manifest.parameters["scar_slope"])
    intercept = float(manifest.parameters["scar_intercept_fraction"])
    half_width = float(manifest.parameters["scar_half_width_px"])
    rgb_delta = np.asarray(manifest.parameters["rgb_delta"], dtype=np.float32)
    distance = np.abs(yy - (slope * xx + height * intercept))
    mask = distance <= half_width
    return clip_like(payload, current + mask[..., None] * rgb_delta)


def _nonresponsive_patch(
    slot_id: str,
    payload: Array,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
) -> Array:
    baseline = baseline_for(slot_id, manifest, rest_references)
    output = payload.copy()
    height, width = payload.shape[:2]
    radius = max(
        int(manifest.parameters["minimum_radius_px"]),
        int(round(min(height, width) * float(manifest.parameters["radius_fraction"]))),
    )
    center_x, center_y = (float(value) for value in manifest.parameters["center_xy"])
    yy, xx = _grid(height, width)
    mask = (xx - width * center_x) ** 2 + (yy - height * center_y) ** 2 <= radius**2
    output[mask] = baseline[mask]
    return output


def _apply_f6(
    clean_record: EvaluationRecord,
    manifest: FaultManifest,
    rest_references: RestReferenceBundle | None,
    histories: MutableMapping[str, Array],
) -> EvaluationRecord:
    output = clean_record
    mix = float(manifest.parameters["history_mix"])
    beta = float(manifest.parameters["memory_beta"])
    for slot_id in manifest.sensor_slots:
        sensor = clean_record.observation.sensor(slot_id)
        if sensor.payload is None:
            raise ValueError("history operator requires present clean payload")
        baseline = normalize(baseline_for(slot_id, manifest, rest_references))
        previous = histories.get(slot_id)
        if previous is None:
            previous = np.zeros_like(baseline)
        residual = normalize(sensor.payload) - baseline
        history = beta * previous + (1.0 - beta) * residual
        histories[slot_id] = cast(Array, history)
        if not manifest.active(clean_record.observation.step_index):
            continue
        transformed = clip_like(
            sensor.payload, baseline + (1.0 - mix) * residual + mix * history
        )
        replacement = replace(
            sensor,
            payload=transformed,
            payload_present=True,
            declared_validity=fault_declared_validity(sensor, manifest),
        )
        provenance = mark_provenance(
            clean_record.provenance_for(slot_id), manifest, transformed
        )
        output = replace_delivery(output, replacement, provenance)
    return output


def _contact_active(
    record: EvaluationRecord, slot_id: str, manifest: FaultManifest
) -> bool:
    phases = {ContactPhase(value) for value in manifest.parameters["activation_phases"]}
    return record.provenance_for(slot_id).phase in phases


def _grid(height: int, width: int) -> tuple[Array, Array]:
    yy, xx = np.mgrid[0:height, 0:width]
    return cast(Array, yy.astype(np.float32)), cast(Array, xx.astype(np.float32))

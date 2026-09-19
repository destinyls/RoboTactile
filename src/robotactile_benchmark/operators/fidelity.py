"""Seven tactile-fidelity signatures synthesized from clean RGB episodes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators.base import (
    OperatorSpec,
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
from robotactile_benchmark.operators.registry import register_operator
from robotactile_benchmark.rest_references import RestReferenceBundle

Transform = Callable[[int, str, Array, EvaluationRecord], Array]
ActivePredicate = Callable[[str, EvaluationRecord], bool]


def _grid(height: int, width: int) -> Tuple[Array, Array]:
    yy, xx = np.mgrid[0:height, 0:width]
    return yy.astype(np.float32), xx.astype(np.float32)


def _active_records(
    clean_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    transform: Transform,
    active_predicate: Optional[ActivePredicate] = None,
) -> Tuple[EvaluationRecord, ...]:
    output = list(clean_records)
    for index, _record in enumerate(clean_records):
        if not manifest.active(index):
            continue
        for slot_id in manifest.sensor_slots:
            if active_predicate is not None and not active_predicate(
                slot_id, output[index]
            ):
                continue
            sensor = output[index].observation.sensor(slot_id)
            if sensor.payload is None:
                raise ValueError("fidelity operator requires a present payload")
            transformed = transform(index, slot_id, sensor.payload, output[index])
            replacement = replace(
                sensor,
                payload=transformed,
                payload_present=True,
                declared_validity=fault_declared_validity(sensor, manifest),
            )
            provenance = mark_provenance(
                output[index].provenance_for(slot_id), manifest, transformed
            )
            output[index] = replace_delivery(output[index], replacement, provenance)
    return tuple(output)


@register_operator("F1_global_response_drift")
class GlobalResponseDriftOperator:
    spec = OperatorSpec(
        "F1_global_response_drift",
        "fidelity",
        "retained_global_gain",
        False,
        "direct_observation",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        target_gain = float(manifest.parameters["target_gain"])
        immediate = manifest.parameters["temporal_path"] == "immediate_step"
        span = max(1, manifest.stop_index - manifest.start_index - 1)

        def transform(
            index: int,
            slot_id: str,
            payload: Array,
            record: EvaluationRecord,
        ) -> Array:
            if immediate:
                if manifest.parameters.get("response_domain") == "absolute_black_frame":
                    return np.zeros_like(payload)
                baseline = normalize(baseline_for(slot_id, manifest, rest_references))
                current = normalize(payload)
                return clip_like(payload, baseline + target_gain * (current - baseline))
            progress = (index - manifest.start_index) / float(span)
            gain = 1.0 - (1.0 - target_gain) * progress
            return clip_like(payload, normalize(payload) * gain)

        return _active_records(clean_records, manifest, transform)


@register_operator("F2_spatial_sensitivity_loss")
class SpatialSensitivityLossOperator:
    spec = OperatorSpec(
        "F2_spatial_sensitivity_loss",
        "fidelity",
        "retained_regional_gain",
        False,
        "direct_observation",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        retained = float(manifest.parameters["retained_gain"])
        center_x, center_y = (
            float(value) for value in manifest.parameters["center_xy"]
        )
        sigma_fraction = float(manifest.parameters["sigma_fraction"])
        minimum_sigma = float(manifest.parameters["minimum_sigma_px"])

        def transform(
            index: int,
            slot_id: str,
            payload: Array,
            record: EvaluationRecord,
        ) -> Array:
            baseline = normalize(baseline_for(slot_id, manifest, rest_references))
            current = normalize(payload)
            height, width = payload.shape[:2]
            yy, xx = _grid(height, width)
            sigma = max(minimum_sigma, min(height, width) * sigma_fraction)
            heat = np.exp(
                -((xx - width * center_x) ** 2 + (yy - height * center_y) ** 2)
                / (2.0 * sigma**2)
            )
            gain = 1.0 - (1.0 - retained) * heat[..., None]
            return clip_like(payload, baseline + gain * (current - baseline))

        return _active_records(clean_records, manifest, transform)


@register_operator("F3_persistent_surface_artifact")
class PersistentSurfaceArtifactOperator:
    spec = OperatorSpec(
        "F3_persistent_surface_artifact",
        "fidelity",
        "scar_dose",
        False,
        "mechanism_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        slope = float(manifest.parameters["scar_slope"])
        intercept = float(manifest.parameters["scar_intercept_fraction"])
        half_width = float(manifest.parameters["scar_half_width_px"])
        rgb_delta = np.asarray(manifest.parameters["rgb_delta"], dtype=np.float32)

        def transform(
            index: int,
            slot_id: str,
            payload: Array,
            record: EvaluationRecord,
        ) -> Array:
            current = normalize(payload)
            height, width = payload.shape[:2]
            yy, xx = _grid(height, width)
            distance = np.abs(yy - (slope * xx + height * intercept))
            mask = distance <= half_width
            return clip_like(payload, current + mask[..., None] * rgb_delta)

        return _active_records(clean_records, manifest, transform)


@register_operator("F4_local_nonresponsive_patch")
class LocalNonresponsivePatchOperator:
    spec = OperatorSpec(
        "F4_local_nonresponsive_patch",
        "fidelity",
        "failed_area_dose",
        False,
        "direct_observation",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        center_x, center_y = (
            float(value) for value in manifest.parameters["center_xy"]
        )
        radius_fraction = float(manifest.parameters["radius_fraction"])
        minimum_radius = int(manifest.parameters["minimum_radius_px"])

        def transform(
            index: int,
            slot_id: str,
            payload: Array,
            record: EvaluationRecord,
        ) -> Array:
            baseline = baseline_for(slot_id, manifest, rest_references)
            output = payload.copy()
            height, width = payload.shape[:2]
            radius = max(
                minimum_radius, int(round(min(height, width) * radius_fraction))
            )
            yy, xx = _grid(height, width)
            mask = (xx - width * center_x) ** 2 + (
                yy - height * center_y
            ) ** 2 <= radius**2
            output[mask] = baseline[mask]
            return output

        return _active_records(clean_records, manifest, transform)


@register_operator("F5_contact_shape_distortion")
class ContactShapeDistortionOperator:
    spec = OperatorSpec(
        "F5_contact_shape_distortion",
        "fidelity",
        "peak_warp_displacement_pixels",
        False,
        "mechanism_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        displacement = float(manifest.parameters["displacement_px"])
        center_xy = tuple(float(value) for value in manifest.parameters["center_xy"])
        radius_fraction = float(manifest.parameters["support_radius_fraction"])
        activation_phases = {
            ContactPhase(value) for value in manifest.parameters["activation_phases"]
        }

        def transform(
            index: int,
            slot_id: str,
            payload: Array,
            record: EvaluationRecord,
        ) -> Array:
            return compact_horizontal_warp(
                payload,
                center_xy=(center_xy[0], center_xy[1]),
                support_radius_fraction=radius_fraction,
                displacement_px=displacement,
            )

        def active_predicate(slot_id: str, record: EvaluationRecord) -> bool:
            return record.provenance_for(slot_id).phase in activation_phases

        return _active_records(
            clean_records, manifest, transform, active_predicate=active_predicate
        )


@register_operator("F6_history_residual_imprint")
class HistoryResidualImprintOperator:
    spec = OperatorSpec(
        "F6_history_residual_imprint",
        "fidelity",
        "history_mix",
        True,
        "direct_observation",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        mix = float(manifest.parameters["history_mix"])
        beta = float(manifest.parameters["memory_beta"])
        histories: Dict[str, Array] = {}
        output = list(clean_records)
        for slot_id in manifest.sensor_slots:
            baseline = normalize(baseline_for(slot_id, manifest, rest_references))
            histories[slot_id] = np.zeros_like(baseline)
        for index, record in enumerate(clean_records):
            for slot_id in manifest.sensor_slots:
                sensor = record.observation.sensor(slot_id)
                if sensor.payload is None:
                    raise ValueError("history operator requires present clean payload")
                baseline = normalize(baseline_for(slot_id, manifest, rest_references))
                residual = normalize(sensor.payload) - baseline
                history = beta * histories[slot_id] + (1.0 - beta) * residual
                histories[slot_id] = history
                if not manifest.active(index):
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
                    record.provenance_for(slot_id), manifest, transformed
                )
                output[index] = replace_delivery(output[index], replacement, provenance)
        return tuple(output)


@register_operator("F7_high_load_saturation")
class HighLoadSaturationOperator:
    spec = OperatorSpec(
        "F7_high_load_saturation",
        "fidelity",
        "response_knee_dose",
        False,
        "mechanism_proxy",
    )

    def apply(
        self,
        clean_records: Sequence[EvaluationRecord],
        manifest: FaultManifest,
        rest_references: Optional[RestReferenceBundle] = None,
    ) -> Tuple[EvaluationRecord, ...]:
        knee = float(manifest.parameters["response_knee"])
        plateau_width_ratio = float(manifest.parameters["plateau_width_ratio"])
        anchor_radius_fraction = float(manifest.parameters["anchor_radius_fraction"])
        minimum_anchor_radius = int(manifest.parameters["minimum_anchor_radius_px"])
        activation_phases = {
            ContactPhase(value) for value in manifest.parameters["activation_phases"]
        }

        def transform(
            index: int,
            slot_id: str,
            payload: Array,
            record: EvaluationRecord,
        ) -> Array:
            return same_frame_response_compression(
                payload,
                response_knee=knee,
                plateau_width_ratio=plateau_width_ratio,
                anchor_radius_fraction=anchor_radius_fraction,
                minimum_anchor_radius_px=minimum_anchor_radius,
            )

        def active_predicate(slot_id: str, record: EvaluationRecord) -> bool:
            return record.provenance_for(slot_id).phase in activation_phases

        return _active_records(
            clean_records, manifest, transform, active_predicate=active_predicate
        )

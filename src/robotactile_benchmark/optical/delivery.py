"""Production optical delivery shared by online and recorded evaluation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import cast

import numpy as np

from robotactile_benchmark.contracts import Array, ContactPhase, EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.operators.base import (
    baseline_for,
    fault_declared_validity,
    mark_provenance,
    replace_delivery,
)
from robotactile_benchmark.operators.fidelity_transforms import compact_horizontal_warp
from robotactile_benchmark.optical.fields import (
    optical_field,
    restore_current_markers,
    spatial_weight,
)
from robotactile_benchmark.optical.stress_templates import ranked_spatial_mask
from robotactile_benchmark.rest_references import RestReferenceBundle


class OpticalDelivery:
    """One causal state per episode; no borrowed contact images or future samples."""

    def __init__(
        self, manifest: FaultManifest, rest: RestReferenceBundle | None
    ) -> None:
        self.manifest = manifest
        self.rest = rest
        self._baselines: dict[str, Array] = {}
        self._memory: dict[str, tuple[float, Array]] = {}

    def _baseline(self, slot: str) -> Array:
        if slot not in self._baselines:
            payload = baseline_for(slot, self.manifest, self.rest)
            self._baselines[slot] = optical_field(
                payload, self.manifest.parameters["optical"]
            )[0]
        return self._baselines[slot]

    def deliver(self, record: EvaluationRecord) -> EvaluationRecord:
        manifest = self.manifest
        op = manifest.operator_id
        index = record.observation.step_index
        if op != "F6_history_residual_imprint" and not manifest.active(index):
            return record
        output = record
        for slot in manifest.sensor_slots:
            if op in {
                "F5_contact_shape_distortion",
                "F7_high_load_saturation",
            } and record.provenance_for(slot).phase not in {
                ContactPhase.ONSET,
                ContactPhase.SUSTAINED,
            }:
                continue
            sensor = record.observation.sensor(slot)
            if sensor.payload is None:
                raise ValueError("optical fidelity requires a present clean frame")
            time_s = record.provenance_for(slot).source_time_s
            if time_s is None:
                raise ValueError(
                    "optical fidelity requires an authentic source timestamp"
                )
            transformed = self._transform(sensor.payload, slot, index, float(time_s))
            if not manifest.active(index):
                continue
            replacement = replace(
                sensor,
                payload=transformed,
                payload_present=True,
                declared_validity=fault_declared_validity(sensor, manifest),
            )
            output = replace_delivery(
                output,
                replacement,
                mark_provenance(record.provenance_for(slot), manifest, transformed),
            )
        return output

    def _transform(self, payload: Array, slot: str, index: int, time_s: float) -> Array:
        p = self.manifest.parameters
        op = self.manifest.operator_id
        current = payload.astype(np.float64) / 255.0
        if p.get("marker_policy") == "occlude_including_markers_v1":
            mask = ranked_spatial_mask(
                payload.shape, tuple(p["center_xy"]), float(p["occluded_area_fraction"])
            )
            result = payload.copy()
            result[mask] = np.asarray(p["fill_rgb"], dtype=payload.dtype)
            return result
        if op == "F1_global_response_drift":
            progress = (index - self.manifest.start_index + 1) / (
                int(
                    p.get(
                        "rise_duration_frames",
                        self.manifest.stop_index - self.manifest.start_index,
                    )
                )
            )
            progress = min(1.0, progress)
            gain = 1.0 + progress * (np.asarray(p["target_gain_rgb"]) - 1.0)
            offset = progress * np.asarray(p["target_offset_rgb"])
            return cast(
                Array,
                np.rint(np.clip(current * gain + offset, 0, 1) * 255).astype(np.uint8),
            )
        if op == "F5_contact_shape_distortion":
            center = tuple(float(v) for v in p["center_xy"])
            return compact_horizontal_warp(
                payload,
                center_xy=(center[0], center[1]),
                support_radius_fraction=float(p["support_radius_fraction"]),
                displacement_px=float(p["displacement_fraction"])
                * min(payload.shape[:2]),
            )
        field, markers = optical_field(payload, p["optical"])
        if op == "F3_persistent_surface_artifact":
            yy, xx = np.mgrid[: payload.shape[0], : payload.shape[1]]
            distance = np.abs(
                yy
                - (
                    float(p["scar_slope"]) * xx
                    + payload.shape[0] * float(p["scar_intercept_fraction"])
                )
            )
            scar = np.exp(-0.5 * (distance / float(p["scar_half_width_px"])) ** 2)
            if p.get("scar_template_version") == "ranked_band_v1":
                template = p.get("spatial_calibration", {}).get("slots", {}).get(slot)
                center = (
                    (
                        float(template["center_xy"][0]),
                        float(template["center_xy"][1]),
                    )
                    if template
                    else (0.5, 0.5)
                )
                slope = float(template["slope"]) if template else 0.45
                fraction = (
                    float(
                        template["area_fractions"][
                            (self.manifest.severity_level - 1) // 2
                        ]
                    )
                    if template
                    else float(p["target_contact_coverage"])
                )
                scar = ranked_spatial_mask(payload.shape, center, fraction, slope=slope)
            result = field + scar[..., None] * np.asarray(p["rgb_delta"])
        else:
            baseline = self._baseline(slot)
            residual = field - baseline
            if op in {"F2_spatial_sensitivity_loss", "F4_local_nonresponsive_patch"}:
                dead = op == "F4_local_nonresponsive_patch"
                weight = spatial_weight(payload.shape, p, dead_patch=dead)
                lost = 1.0 if dead else 1.0 - float(p["retained_gain"])
                result = field - lost * weight[..., None] * residual
            elif op == "F6_history_residual_imprint":
                previous_time, previous = self._memory.get(
                    slot,
                    (time_s - float(p["sample_period_s"]), np.zeros_like(residual)),
                )
                dt = time_s - previous_time
                if dt < 0 or not math.isfinite(dt):
                    raise ValueError("F6 source time must be finite and non-decreasing")
                loading = np.linalg.norm(
                    residual, axis=-1, keepdims=True
                ) > np.linalg.norm(previous, axis=-1, keepdims=True)
                tau = np.where(
                    loading, float(p["buildup_tau_s"]), float(p["recovery_tau_s"])
                )
                beta = np.exp(-dt / tau)
                history = beta * previous + (1.0 - beta) * residual
                self._memory[slot] = (time_s, history)
                mix = float(p["history_mix"])
                result = baseline + (1.0 - mix) * residual + mix * history
            elif op == "F7_high_load_saturation":
                magnitude = np.linalg.norm(residual, axis=-1, keepdims=True)
                knee = float(p["response_knee"])
                plateau = knee * float(p["plateau_width_ratio"])
                excess = np.maximum(magnitude - knee, 0.0)
                compressed = np.where(
                    magnitude <= knee,
                    magnitude,
                    knee + plateau * excess / (plateau + excess),
                )
                scale = np.divide(
                    compressed,
                    magnitude,
                    out=np.ones_like(magnitude),
                    where=magnitude > 1e-12,
                )
                result = baseline + residual * scale
            else:
                raise KeyError(op)
        delivered = restore_current_markers(payload, result, markers)
        if op in {"F2_spatial_sensitivity_loss", "F4_local_nonresponsive_patch"}:
            delivered[weight == 0.0] = payload[weight == 0.0]
        return delivered

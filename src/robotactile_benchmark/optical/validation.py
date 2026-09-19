"""Optical-profile equations and observable witnesses, independent of delivery.

The reference evaluates fields directly; it never calls OpticalDelivery or a
streaming session. Shared marker extraction is an image primitive, not evidence
that the estimated optical field is physically calibrated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from robotactile_benchmark.contracts import Array, ContactPhase, EvaluationRecord
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.optical.fields import optical_field, spatial_weight
from robotactile_benchmark.optical.stress_templates import ranked_spatial_mask
from robotactile_benchmark.rest_references import RestReferenceBundle


def measure_optical(
    clean: Sequence[EvaluationRecord],
    delivered: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    rest: RestReferenceBundle | None,
) -> tuple[bool, float, str, dict[str, float], tuple[str, ...]]:
    p = manifest.parameters
    op = manifest.operator_id
    fullframe = p.get("marker_policy") == "occlude_including_markers_v1"
    matched = True
    marker_changes = outside_changes = changed = eligible = 0
    before = after = 0.0
    release_count = 0
    contact_samples = local_samples = core_samples = above_knee_samples = 0
    residual_witness_samples = 0
    core_energy = core_projection = 0.0
    failures: list[str] = []
    errors = []
    observed_marker_pixels = occluded_marker_pixels = clipped_elements = 0
    contact_response_pixels = covered_response_pixels = 0
    for slot in manifest.sensor_slots:
        baseline = (
            optical_field(rest.payload_for(slot), p["optical"])[0]
            if rest is not None
            and op not in {"F1_global_response_drift", "F5_contact_shape_distortion"}
            else None
        )
        history: Array | None = None
        last_time: float | None = None
        for i, (source, output) in enumerate(zip(clean, delivered)):
            payload = source.observation.sensor(slot).payload
            actual = output.observation.sensor(slot).payload
            if payload is None or actual is None:
                matched = False
                continue
            phase = source.provenance_for(slot).phase
            active = manifest.active(i)
            if op in {"F5_contact_shape_distortion", "F7_high_load_saturation"}:
                active = active and phase in {
                    ContactPhase.ONSET,
                    ContactPhase.SUSTAINED,
                }
            if active and phase in {ContactPhase.ONSET, ContactPhase.SUSTAINED}:
                contact_samples += 1
            if not active and op != "F6_history_residual_imprint":
                continue
            current = payload.astype(np.float64) / 255.0
            markers = np.zeros(payload.shape[:2], dtype=bool)
            if fullframe:
                mask = ranked_spatial_mask(
                    payload.shape,
                    tuple(p["center_xy"]),
                    float(p["occluded_area_fraction"]),
                )
                observed_markers = optical_field(payload, p["optical"])[1]
                observed_marker_pixels += int(np.count_nonzero(observed_markers))
                occluded_marker_pixels += int(np.count_nonzero(observed_markers & mask))
                expected = current.copy()
                expected[mask] = np.asarray(p["fill_rgb"], dtype=float) / 255.0
                outside_changes += int(
                    np.count_nonzero(np.any(actual != payload, axis=-1) & ~mask)
                )
            elif op == "F1_global_response_drift":
                q = (i + 1 - manifest.start_index) / (
                    int(
                        p.get(
                            "rise_duration_frames",
                            manifest.stop_index - manifest.start_index,
                        )
                    )
                )
                q = min(q, 1.0)
                expected = current * (
                    (1 - q) + q * np.asarray(p["target_gain_rgb"])
                ) + q * np.asarray(p["target_offset_rgb"])
            elif op == "F5_contact_shape_distortion":
                expected = _warp_reference(payload, manifest) / 255.0
            else:
                field, markers = optical_field(payload, p["optical"])
                if op == "F3_persistent_surface_artifact":
                    yy, xx = np.mgrid[: payload.shape[0], : payload.shape[1]]
                    line = float(p["scar_slope"]) * xx + payload.shape[0] * float(
                        p["scar_intercept_fraction"]
                    )
                    weight = np.exp(
                        -((yy - line) ** 2) / (2 * float(p["scar_half_width_px"]) ** 2)
                    )
                    if p.get("scar_template_version") == "ranked_band_v1":
                        template = (
                            p.get("spatial_calibration", {}).get("slots", {}).get(slot)
                        )
                        center = (
                            tuple(template["center_xy"]) if template else (0.5, 0.5)
                        )
                        slope = float(template["slope"]) if template else 0.45
                        fraction = (
                            float(
                                template["area_fractions"][
                                    (manifest.severity_level - 1) // 2
                                ]
                            )
                            if template
                            else float(p["target_contact_coverage"])
                        )
                        weight = ranked_spatial_mask(
                            payload.shape, center, fraction, slope=slope
                        )
                        if baseline is not None:
                            contact_response = (~markers) & (
                                np.linalg.norm(field - baseline, axis=-1) > 2 / 255
                            )
                            contact_response_pixels += int(
                                np.count_nonzero(contact_response)
                            )
                            covered_response_pixels += int(
                                np.count_nonzero(contact_response & weight)
                            )
                    expected = field + weight[..., None] * np.asarray(p["rgb_delta"])
                else:
                    if baseline is None:
                        raise ValueError("optical response measurement requires rest")
                    residual = field - baseline
                    if op in {
                        "F2_spatial_sensitivity_loss",
                        "F4_local_nonresponsive_patch",
                    }:
                        weight = spatial_weight(
                            payload.shape, p, dead_patch=op.startswith("F4")
                        )
                        gain = 1 - weight * (
                            1 if op.startswith("F4") else 1 - float(p["retained_gain"])
                        )
                        expected = baseline + gain[..., None] * residual
                        if phase in {ContactPhase.ONSET, ContactPhase.SUSTAINED}:
                            # Two normalized RGB quanta: a measurement floor, not force.
                            signal = (~markers) & (
                                np.linalg.norm(residual, axis=-1) > 2 / 255
                            )
                            local_samples += int(
                                np.count_nonzero(signal & (weight > 0))
                            )
                            core = signal & (weight >= 1 - 1e-12)
                            core_samples += int(np.count_nonzero(core))
                            actual_residual = actual.astype(np.float64) / 255 - baseline
                            core_energy += float(np.sum(residual[core] ** 2))
                            core_projection += float(
                                np.sum(residual[core] * actual_residual[core])
                            )
                        outside_changes += int(
                            np.count_nonzero(
                                np.any(actual != payload, axis=-1) & (weight == 0)
                            )
                        )
                    elif op == "F6_history_residual_imprint":
                        raw_time = source.provenance_for(slot).source_time_s
                        if raw_time is None:
                            raise ValueError(
                                "F6 measurement requires source timestamps"
                            )
                        now = float(raw_time)
                        dt = (
                            float(p["sample_period_s"])
                            if last_time is None
                            else now - last_time
                        )
                        if history is None:
                            history = np.zeros_like(residual)
                        loading = np.sum(residual**2, axis=-1) > np.sum(
                            history**2, axis=-1
                        )
                        decay = np.full(
                            loading.shape, math.exp(-dt / float(p["recovery_tau_s"]))
                        )
                        decay[loading] = math.exp(-dt / float(p["buildup_tau_s"]))
                        history = (
                            decay[..., None] * history
                            + (1 - decay[..., None]) * residual
                        )
                        last_time = now
                        expected = (
                            baseline
                            + residual * (1 - float(p["history_mix"]))
                            + history * float(p["history_mix"])
                        )
                    elif op == "F7_high_load_saturation":
                        norm = np.sqrt(np.sum(residual**2, axis=2))
                        knee = float(p["response_knee"])
                        width = knee * float(p["plateau_width_ratio"])
                        ratio = np.ones_like(norm)
                        above = norm > knee
                        above_knee_samples += int(np.count_nonzero(above & ~markers))
                        target = knee + width - width**2 / (width + norm[above] - knee)
                        ratio[above] = target / norm[above]
                        expected = baseline + residual * ratio[..., None]
                    else:
                        raise KeyError(op)
            if not active:
                continue
            expected_u8 = np.rint(np.clip(expected, 0, 1) * 255).astype(np.uint8)
            expected_u8[markers] = payload[markers]
            # Different arithmetic order may differ by one quantization unit.
            error = int(
                np.abs(actual.astype(np.int16) - expected_u8.astype(np.int16)).max()
            )
            errors.append(error)
            matched = matched and error <= (0 if fullframe else 1)
            marker_changes += int(np.count_nonzero(actual[markers] != payload[markers]))
            changed += int(np.count_nonzero(np.any(actual != payload, axis=-1)))
            eligible += payload.shape[0] * payload.shape[1]
            clipped_elements += int(np.count_nonzero((actual == 0) | (actual == 255)))
            if baseline is not None:
                valid = ~markers
                if op == "F6_history_residual_imprint":
                    if phase == ContactPhase.RELEASE:
                        release_count += 1
                        clean_norm = np.linalg.norm(current - baseline, axis=-1)
                        actual_norm = np.linalg.norm(
                            actual.astype(np.float64) / 255 - baseline, axis=-1
                        )
                        if history is None:
                            raise ValueError("missing causal response history")
                        past_norm = np.linalg.norm(history, axis=-1)
                        unloading = valid & (past_norm > clean_norm + 1 / 255)
                        residual_witness_samples += int(
                            np.count_nonzero(
                                unloading
                                & (actual_norm > clean_norm)
                                & np.any(actual != payload, axis=-1)
                            )
                        )
                        before += float(past_norm[unloading].sum())
                        after += float(
                            np.maximum(actual_norm - clean_norm, 0)[unloading].sum()
                        )
                elif phase in {ContactPhase.ONSET, ContactPhase.SUSTAINED}:
                    before += float(np.abs(current - baseline)[valid].sum())
                    after += float(
                        np.abs(actual.astype(np.float64) / 255 - baseline)[valid].sum()
                    )
    metrics = {
        "changed_pixel_fraction": changed / max(eligible, 1),
        "marker_changed_elements": float(marker_changes),
        "outside_support_changed_pixels": float(outside_changes),
        "max_reference_error_u8": float(max(errors, default=0)),
        "contact_observation_count": float(contact_samples),
    }
    dose, unit = changed / max(eligible, 1), "changed_pixel_fraction"
    if fullframe:
        metrics.update(
            marker_visible_fraction=1
            - occluded_marker_pixels / max(observed_marker_pixels, 1),
            observed_marker_pixel_samples=float(observed_marker_pixels),
            requested_occluded_area_fraction=float(p["occluded_area_fraction"]),
            boundary_value_fraction=clipped_elements / max(3 * eligible, 1),
        )
    if p.get("scar_template_version") == "ranked_band_v1":
        metrics.update(
            measured_response_support_coverage=covered_response_pixels
            / max(contact_response_pixels, 1),
            response_pixel_samples=float(contact_response_pixels),
        )
    if not fullframe and op in {
        "F2_spatial_sensitivity_loss",
        "F4_local_nonresponsive_patch",
        "F7_high_load_saturation",
    }:
        dose = max(0.0, 1 - after / max(before, 1e-12))
        unit = "marker_excluded_optical_response_attenuation"
        if before <= 1e-8:
            failures.append("NO_CONTACT_SAMPLE")
    if not fullframe and op in {
        "F2_spatial_sensitivity_loss",
        "F4_local_nonresponsive_patch",
    }:
        metrics.update(
            local_response_pixel_samples=float(local_samples),
            core_response_pixel_samples=float(core_samples),
            core_response_projection_gain=core_projection / max(core_energy, 1e-12),
            response_floor_u8=2.0,
        )
        if not local_samples:
            failures.append("NO_LOCAL_CONTACT_RESPONSE")
    if op == "F5_contact_shape_distortion" and not contact_samples:
        failures.append("NO_CONTACT_SAMPLE")
    if op == "F7_high_load_saturation":
        metrics["above_knee_pixel_samples"] = float(above_knee_samples)
        if not above_knee_samples:
            failures.append("NO_ABOVE_KNEE_SAMPLE")
    if op == "F6_history_residual_imprint":
        dose = after / max(before, 1 / 255)
        unit = "locally_unloaded_history_response_fraction"
        metrics.update(
            release_sample_count=float(release_count),
            recovery_tau_s=float(p["recovery_tau_s"]),
            residual_witness_pixel_samples=float(residual_witness_samples),
        )
        if not release_count:
            failures.append("NO_RELEASE_SAMPLE")
        elif not residual_witness_samples:
            failures.append("NO_RESIDUAL_WITNESS")
    if marker_changes:
        failures.append("MARKER_IDENTITY_CHANGED")
    if outside_changes:
        failures.append("OUTSIDE_SUPPORT_CHANGED")
    return matched and not failures, dose, unit, metrics, tuple(failures)


def _warp_reference(payload: Array, manifest: FaultManifest) -> Array:
    height, width = payload.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    cx, cy = (float(v) for v in manifest.parameters["center_xy"])
    radius = min(height, width) * float(manifest.parameters["support_radius_fraction"])
    r2 = ((xx - cx * (width - 1)) ** 2 + (yy - cy * (height - 1)) ** 2) / radius**2
    weight = np.maximum(1 - r2, 0) ** 2
    source_x = xx - weight * float(manifest.parameters["displacement_fraction"]) * min(
        height, width
    )
    output = payload.astype(np.float64).copy()
    for y in range(height):
        for channel in range(3):
            output[y, :, channel] = np.interp(
                source_x[y], np.arange(width), payload[y, :, channel]
            )
    return output

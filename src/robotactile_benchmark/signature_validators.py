"""Operator-specific model-visible signature measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from robotactile_benchmark.constants import (
    OPTICAL_MARKER_REGISTRY_IDS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.reference_signatures import expected_pixel_payloads
from robotactile_benchmark.rest_references import RestReferenceBundle


@dataclass(frozen=True)
class SignatureMeasurement:
    """Exact signature match plus an operator-native observable."""

    matched: bool
    achieved_dose: float
    unit: str
    diagnostics: Mapping[str, float]
    failure_codes: Tuple[str, ...] = ()


def _payload_pairs(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    rest_references: Optional[RestReferenceBundle],
    phases: Tuple[ContactPhase, ...] = (),
) -> Tuple[Tuple[Array, Array, Array], ...]:
    pairs: list[Tuple[Array, Array, Array]] = []
    for index in range(
        manifest.start_index, min(manifest.stop_index, len(delivered_records))
    ):
        for slot_id in manifest.sensor_slots:
            if (
                phases
                and clean_records[index].provenance_for(slot_id).phase not in phases
            ):
                continue
            clean_payload = clean_records[index].observation.sensor(slot_id).payload
            delivered_payload = (
                delivered_records[index].observation.sensor(slot_id).payload
            )
            baseline = (
                rest_references.payload_for(slot_id)
                if rest_references is not None
                else clean_payload
            )
            if clean_payload is None or delivered_payload is None or baseline is None:
                continue
            pairs.append(
                (
                    clean_payload.astype(np.float32),
                    delivered_payload.astype(np.float32),
                    baseline.astype(np.float32),
                )
            )
    return tuple(pairs)


def _mean_ratio(numerators: Sequence[float], denominators: Sequence[float]) -> float:
    numerator = float(np.sum(numerators))
    denominator = float(np.sum(denominators))
    return numerator / max(denominator, 1e-12)


def _f1(
    pairs: Sequence[Tuple[Array, Array, Array]], manifest: FaultManifest
) -> Tuple[float, str]:
    if manifest.parameters.get("response_domain") == "absolute_black_frame":
        pixel_count = sum(delivered.size for _, delivered, _ in pairs)
        black_count = sum(
            int(np.count_nonzero(delivered == 0)) for _, delivered, _ in pairs
        )
        return float(black_count) / max(pixel_count, 1), "black_element_fraction"
    if manifest.parameters["temporal_path"] == "immediate_step":
        retained = _mean_ratio(
            [
                float(np.abs(delivered - baseline).sum())
                for _, delivered, baseline in pairs
            ],
            [float(np.abs(clean - baseline).sum()) for clean, _, baseline in pairs],
        )
        return max(0.0, 1.0 - retained), "certified_rest_residual_attenuation"
    retained_ratios = [
        float(delivered.mean()) / max(float(clean.mean()), 1e-12)
        for clean, delivered, _ in pairs
    ]
    return (
        float(min(retained_ratios)) if retained_ratios else 0.0,
        "retained_global_gain",
    )


def _f2(pairs: Sequence[Tuple[Array, Array, Array]]) -> float:
    retained_numerators = []
    retained_denominators = []
    for clean, delivered, baseline in pairs:
        height, width = clean.shape[:2]
        y0, y1 = height // 4, height - height // 4
        x0, x1 = width // 4, width - width // 4
        retained_numerators.append(
            float(np.abs(delivered[y0:y1, x0:x1] - baseline[y0:y1, x0:x1]).sum())
        )
        retained_denominators.append(
            float(np.abs(clean[y0:y1, x0:x1] - baseline[y0:y1, x0:x1]).sum())
        )
    return max(0.0, 1.0 - _mean_ratio(retained_numerators, retained_denominators))


def _f3(pairs: Sequence[Tuple[Array, Array, Array]]) -> float:
    changed = sum(
        int(np.any(clean != delivered, axis=-1).sum()) for clean, delivered, _ in pairs
    )
    total = sum(int(clean.shape[0] * clean.shape[1]) for clean, _, _ in pairs)
    return float(changed) / max(total, 1)


def _f4(pairs: Sequence[Tuple[Array, Array, Array]]) -> float:
    replaced = 0
    eligible = 0
    for clean, delivered, baseline in pairs:
        clean_active = np.any(clean != baseline, axis=-1)
        delivered_is_baseline = np.all(delivered == baseline, axis=-1)
        replaced += int(np.logical_and(clean_active, delivered_is_baseline).sum())
        eligible += int(clean_active.sum())
    return float(replaced) / max(eligible, 1)


def _mean_rgb_delta(
    pairs: Sequence[Tuple[Array, Array, Array]],
) -> float:
    if not pairs:
        return 0.0
    return float(
        np.mean([np.abs(clean - delivered).mean() for clean, delivered, _ in pairs])
    )


def _f5_geometry(
    pairs: Sequence[Tuple[Array, Array, Array]], manifest: FaultManifest
) -> Tuple[float, float, float, bool]:
    peak_displacements = []
    minimum_steps = []
    changed = 0
    total = 0
    in_bounds = True
    center_x_fraction, center_y_fraction = (
        float(value) for value in manifest.parameters["center_xy"]
    )
    displacement = float(manifest.parameters["displacement_px"])
    radius_fraction = float(manifest.parameters["support_radius_fraction"])
    for clean, delivered, _ in pairs:
        height, width = clean.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        center_x = center_x_fraction * (width - 1)
        center_y = center_y_fraction * (height - 1)
        radius = min(height, width) * radius_fraction
        radius_squared = ((xx - center_x) / radius) ** 2 + (
            (yy - center_y) / radius
        ) ** 2
        support = radius_squared < 1.0
        weight = np.zeros((height, width), dtype=np.float64)
        weight[support] = (1.0 - radius_squared[support]) ** 2
        source_x = xx - displacement * weight
        in_bounds = in_bounds and bool(
            np.all(source_x[support] >= 0.0) and np.all(source_x[support] <= width - 1)
        )
        peak_displacements.append(float((displacement * weight).max(initial=0.0)))
        minimum_steps.append(float(np.diff(source_x, axis=1).min(initial=1.0)))
        changed += int(np.any(clean != delivered, axis=-1).sum())
        total += height * width
    return (
        max(peak_displacements, default=0.0),
        min(minimum_steps, default=0.0),
        float(changed) / max(total, 1),
        in_bounds,
    )


def _f6(pairs: Sequence[Tuple[Array, Array, Array]]) -> float:
    ratios = []
    for clean, delivered, baseline in pairs:
        clean_residual = float(np.abs(clean - baseline).sum())
        delivered_residual = float(np.abs(delivered - baseline).sum())
        ratios.append(
            max(0.0, delivered_residual - clean_residual) / max(clean_residual, 1.0)
        )
    return float(np.mean(ratios)) if ratios else 0.0


def _reflect_indices(length: int, radius: int) -> Tuple[Array, ...]:
    base = np.arange(length, dtype=np.int64)
    period = 2 * (length - 1)
    output = []
    for offset in range(-radius, radius + 1):
        raw = np.mod(base + offset, period)
        output.append(np.where(raw < length, raw, period - raw))
    return tuple(output)


def _box_anchor(payload: Array, manifest: FaultManifest) -> Array:
    current = payload.astype(np.float64) / 255.0
    height, width = payload.shape[:2]
    requested = max(
        int(manifest.parameters["minimum_anchor_radius_px"]),
        int(
            np.floor(
                float(manifest.parameters["anchor_radius_fraction"])
                * min(height, width)
                + 0.5
            )
        ),
    )
    radius = min((min(height, width) - 1) // 2, requested)
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
    return anchor


def _f7(
    pairs: Sequence[Tuple[Array, Array, Array]], manifest: FaultManifest
) -> Tuple[float, bool]:
    input_detail = []
    output_detail = []
    has_above_knee_sample = False
    knee = float(manifest.parameters["response_knee"])
    for clean, delivered, _ in pairs:
        anchor = _box_anchor(clean, manifest)
        clean_magnitude = np.linalg.norm(
            clean.astype(np.float64) / 255.0 - anchor, axis=-1
        )
        delivered_magnitude = np.linalg.norm(
            delivered.astype(np.float64) / 255.0 - anchor, axis=-1
        )
        has_above_knee_sample = has_above_knee_sample or bool(
            np.any(clean_magnitude > knee)
        )
        input_detail.append(float(clean_magnitude.sum()))
        output_detail.append(float(delivered_magnitude.sum()))
    if not input_detail or float(np.sum(input_detail)) <= 1e-12:
        return 0.0, False
    retained = float(np.sum(output_detail)) / max(float(np.sum(input_detail)), 1e-12)
    return max(0.0, 1.0 - retained), has_above_knee_sample


def measure_signature(
    clean_records: Sequence[EvaluationRecord],
    delivered_records: Sequence[EvaluationRecord],
    manifest: FaultManifest,
    rest_references: Optional[RestReferenceBundle] = None,
) -> SignatureMeasurement:
    """Validate an exact deterministic signature and measure its native effect."""

    if (
        manifest.severity_registry in OPTICAL_MARKER_REGISTRY_IDS
        and manifest.operator_id.startswith("F")
    ):
        from robotactile_benchmark.optical.validation import measure_optical

        return SignatureMeasurement(
            *measure_optical(
                clean_records, delivered_records, manifest, rest_references
            )
        )

    requires_rest = operator_requires_rest_reference(
        manifest.operator_id,
        severity_registry=manifest.severity_registry,
    )
    if requires_rest and rest_references is None:
        raise ValueError("this pixel signature requires rest references")
    expected_payloads = expected_pixel_payloads(
        clean_records, manifest, rest_references
    )
    matched = True
    for (index, slot_id), expected_payload in expected_payloads.items():
        actual_payload = delivered_records[index].observation.sensor(slot_id).payload
        if actual_payload is None or not np.array_equal(
            actual_payload, expected_payload
        ):
            matched = False
            break
    operator_id = manifest.operator_id
    contact_phases = (ContactPhase.ONSET, ContactPhase.SUSTAINED)
    all_pairs = _payload_pairs(
        clean_records, delivered_records, manifest, rest_references
    )
    contact_pairs = _payload_pairs(
        clean_records,
        delivered_records,
        manifest,
        rest_references,
        contact_phases,
    )
    failures = []
    if operator_id == "F1_global_response_drift":
        dose, unit = _f1(all_pairs, manifest)
        if manifest.parameters.get(
            "response_domain"
        ) == "absolute_black_frame" and not any(
            float(np.abs(clean).sum()) > 0.0 for clean, _, _ in all_pairs
        ):
            failures.append("NO_CLEAN_TACTILE_CONTENT")
        elif (
            manifest.parameters.get("response_domain") != "absolute_black_frame"
            and manifest.parameters["temporal_path"] == "immediate_step"
            and not any(
                float(np.abs(clean - baseline).sum()) > 0.0
                for clean, _, baseline in all_pairs
            )
        ):
            failures.append("NO_CLEAN_TACTILE_SIGNAL")
    elif operator_id == "F2_spatial_sensitivity_loss":
        dose, unit = _f2(contact_pairs), "central_residual_attenuation"
    elif operator_id == "F3_persistent_surface_artifact":
        dose, unit = _f3(all_pairs), "changed_pixel_fraction"
    elif operator_id == "F4_local_nonresponsive_patch":
        dose, unit = _f4(contact_pairs), "baseline_replaced_fraction"
    elif operator_id == "F5_contact_shape_distortion":
        dose, minimum_step, changed_fraction, in_bounds = _f5_geometry(
            contact_pairs, manifest
        )
        unit = "peak_warp_displacement_pixels"
        if not contact_pairs:
            failures.append("NO_CONTACT_SAMPLE")
        if not in_bounds:
            failures.append("F5_SOURCE_MAP_OUT_OF_BOUNDS")
        if minimum_step <= 0.0:
            failures.append("F5_SOURCE_MAP_FOLD")
        if changed_fraction <= 0.0:
            failures.append("F5_NO_VISIBLE_WARP_EFFECT")
    elif operator_id == "F6_history_residual_imprint":
        release_pairs = _payload_pairs(
            clean_records,
            delivered_records,
            manifest,
            rest_references,
            (ContactPhase.RELEASE,),
        )
        dose, unit = _f6(release_pairs), "post_release_residual_ratio"
        if not release_pairs:
            failures.append("NO_RELEASE_SAMPLE")
    elif operator_id == "F7_high_load_saturation":
        dose, has_detail = _f7(contact_pairs, manifest)
        unit = "same_frame_detail_compression"
        if not contact_pairs:
            failures.append("NO_CONTACT_SAMPLE")
        if not has_detail:
            failures.append("F7_NO_DETAIL_SAMPLE")
    elif operator_id == "C2_frame_misregistration":
        shift_x, shift_y = (
            float(value) for value in manifest.parameters["translation_xy_px"]
        )
        dose = float(np.hypot(shift_x, shift_y))
        unit = "reprojection_displacement_pixels"
    else:
        raise ValueError(f"operator has no pixel-signature validator: {operator_id}")
    if not matched and _mean_rgb_delta(all_pairs) <= 0.0:
        failures.append("SIGNATURE_NOT_DELIVERED")
    diagnostics = {"mean_absolute_rgb_delta": _mean_rgb_delta(all_pairs)}
    if (
        operator_id == "F1_global_response_drift"
        and manifest.parameters["temporal_path"] == "immediate_step"
    ):
        if manifest.parameters.get("response_domain") == "absolute_black_frame":
            diagnostics["black_frame_max_abs_value"] = max(
                (
                    float(np.abs(delivered).max(initial=0.0))
                    for _, delivered, _ in all_pairs
                ),
                default=0.0,
            )
            diagnostics["clean_tactile_l1"] = float(
                sum(np.abs(clean).sum() for clean, _, _ in all_pairs)
            )
        else:
            diagnostics["certified_rest_max_abs_error"] = max(
                (
                    float(np.abs(delivered - baseline).max(initial=0.0))
                    for _, delivered, baseline in all_pairs
                ),
                default=0.0,
            )
            diagnostics["clean_tactile_residual_l1"] = float(
                sum(np.abs(clean - baseline).sum() for clean, _, baseline in all_pairs)
            )
    if operator_id == "F5_contact_shape_distortion":
        diagnostics.update(
            {
                "minimum_source_x_step": minimum_step,
                "changed_pixel_fraction": changed_fraction,
            }
        )
    return SignatureMeasurement(
        matched=matched,
        achieved_dose=float(dose),
        unit=unit,
        diagnostics=diagnostics,
        failure_codes=tuple(failures),
    )

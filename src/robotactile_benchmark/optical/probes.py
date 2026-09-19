"""Analytical stimuli, NOT empirical/recorded samples or additional episodes.

These probes test G1 observation behavior through production delivery. Time
constants are engineering parameters, not literature-calibrated material values.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import numpy as np

from robotactile_benchmark.constants import OPTICAL_MARKER_REGISTRY_ID
from robotactile_benchmark.contracts import (
    Array,
    ContactPhase,
    EvaluationRecord,
    array_sha256,
    build_evaluation_record,
)
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.optical.delivery import OpticalDelivery
from robotactile_benchmark.rest_references import RestReferenceBundle


def _fixture(base: int = 100) -> tuple[EvaluationRecord, Array, RestReferenceBundle]:
    rgb = np.full((32, 32, 3), base, dtype=np.uint8)
    rgb[2, 2] = 10
    rest = replace(
        make_synthetic_rest_references(),
        reference_id="analytic-stimulus-not-empirical-rest",
        payloads={slot: rgb for slot in ("left", "right")},
    )
    return make_synthetic_episode()[0], rgb, rest


def _record(
    template: EvaluationRecord, rgb: Array, index: int, fps: int, phase: ContactPhase
) -> EvaluationRecord:
    sensor = replace(
        template.observation.sensor("right"),
        payload=rgb,
        delivery_index=index,
        delivery_time_s=index / fps,
    )
    source = replace(
        template.provenance_for("right"),
        source_index=index,
        source_time_s=index / fps,
        phase=phase,
        payload_sha256=array_sha256(rgb),
    )
    return build_evaluation_record(
        replace(
            template.observation,
            episode_id="analytic-stimulus-not-empirical",
            step_index=index,
            tactile=(template.observation.sensor("left"), sensor),
        ),
        (template.provenance_for("left"), source),
    )


def _manifest(
    op: str, level: int, fps: int, stop: int, rest: RestReferenceBundle
) -> FaultManifest:
    return FaultManifest(
        op,
        level,
        23,
        0,
        stop,
        ("right",),
        Observability.BLIND,
        {"sample_period_s": 1 / fps, "rest_reference_sha256": rest.sha256},
        severity_registry=OPTICAL_MARKER_REGISTRY_ID,
    )


def _payload(delivery: OpticalDelivery, record: EvaluationRecord) -> Array:
    result = delivery.deliver(record).observation.sensor("right").payload
    assert result is not None
    return result


def _recovery(
    level: int, fps: int, *, right: bool = False, tail_s: float = 12.0
) -> dict[str, Any]:
    template, base, rest = _fixture()
    release = int(1.5 * fps)
    count = release + int(tail_s * fps) + 1
    manifest = _manifest("F6_history_residual_imprint", level, fps, count, rest)
    delivery = OpticalDelivery(manifest, rest)
    tail, images = [], []
    x = 23 if right else 9
    for i in range(count):
        rgb = base.copy()
        phase = ContactPhase.RELEASE if i >= release else ContactPhase.SUSTAINED
        if i < release:
            amplitude = min(i / fps, 1.0)
            rgb[8:25, x - 4 : x + 5] = np.rint(
                100 + amplitude * np.array([100, 60, 20])
            ).astype(np.uint8)
        out = _payload(delivery, _record(template, rgb, i, fps, phase))
        if i >= release:
            tail.append(float(np.linalg.norm(out[16, x].astype(float) - 100) / 255))
            if i == release or i == count - 1:
                images.append(out.tolist())
    times = np.arange(len(tail), dtype=float) / fps
    values = np.asarray(tail)
    tau = float(manifest.parameters["recovery_tau_s"])
    expected = values[0] * np.exp(-times / tau)
    quantization = math.sqrt(3) / 255
    measured: dict[str, float | None] = {}
    for name, fraction in (("t50_s", 0.5), ("t90_s", 0.1)):
        crossed = np.flatnonzero(values <= values[0] * fraction)
        measured[name] = float(times[crossed[0]]) if crossed.size else None
    fresh = _payload(
        OpticalDelivery(manifest, rest),
        _record(template, base, 0, fps, ContactPhase.FREE),
    )
    eligible = bool(values.size > 1 and values[0] > quantization)
    checks = {
        "eligible": eligible,
        "positive_release_residual": bool(values[0] > quantization),
        "monotone_tail": bool(np.all(np.diff(values) <= 1e-12)),
        "tau_curve_within_quantization": bool(
            np.max(np.abs(values - expected)) <= quantization
        ),
        "returns_to_rest": bool(values[-1] <= quantization),
        "fresh_episode_has_no_memory": bool(np.array_equal(fresh, base)),
    }
    return {
        "fps": fps,
        "tau_s": tau,
        "times_s": times.tolist(),
        "response": tail,
        "expected_response": expected.tolist(),
        "quantization_bound": quantization,
        "tau_curve_max_error": float(np.max(np.abs(values - expected))),
        "expected_t50_s": tau * math.log(2),
        "expected_t90_s": tau * math.log(10),
        "measured": measured,
        "checks": checks,
        "passed": eligible and all(checks.values()),
        "release_and_final_rgb": images,
    }


def probe_f6_recovery(level: int = 5) -> dict[str, Any]:
    """Load for 1 s, hold 0.5 s, release for 12 s at two physical cadences."""
    runs = [_recovery(level, fps) for fps in (60, 120)]
    other = _recovery(level, 60, right=True, tail_s=0)
    first = np.asarray(runs[0]["release_and_final_rgb"][0])
    second = np.asarray(other["release_and_final_rgb"][0])
    cadence_error = float(
        np.max(
            np.abs(
                np.asarray(runs[0]["response"]) - np.asarray(runs[1]["response"])[::2]
            )
        )
    )
    history_local = bool(
        first[16, 9, 0] > 100
        and first[16, 23, 0] == 100
        and second[16, 23, 0] > 100
        and second[16, 9, 0] == 100
    )
    return {
        "stimulus": "analytic stimulus NOT empirical/recorded sample",
        "evidence_tier": "G1",
        "material_parameters": "engineering, not calibrated",
        "empirical_episode_count": 0,
        "runs": runs,
        "same_current_rest_different_history": history_local,
        "alternative_history_release_rgb": second.tolist(),
        "cadence_max_error": cadence_error,
        "passed": all(run["passed"] for run in runs)
        and history_local
        and cadence_error <= math.sqrt(3) / 255,
    }


def probe_f7_transfer(level: int = 5) -> dict[str, Any]:
    """Probe residual soft-knee; baseline invariance refutes camera clipping claims."""
    curves, outputs = [], []
    direction_errors = []
    amplitudes = range(0, 121, 2)
    for offset in (0, 30):
        template, base, rest = _fixture(100 + offset)
        manifest = _manifest("F7_high_load_saturation", level, 60, 61, rest)
        delivery = OpticalDelivery(manifest, rest)
        curve, vectors = [], []
        for i, amplitude in enumerate(amplitudes):
            rgb = base.copy()
            rgb[:] = np.array([100 + offset] * 3) + [amplitude, amplitude // 2, 0]
            rgb[2, 2] = 10
            out = _payload(
                delivery, _record(template, rgb, i, 60, ContactPhase.SUSTAINED)
            )
            current = (rgb[16, 16].astype(float) - (100 + offset)) / 255
            delivered = (out[16, 16].astype(float) - (100 + offset)) / 255
            curve.append(
                [float(np.linalg.norm(current)), float(np.linalg.norm(delivered))]
            )
            vectors.append(delivered.tolist())
            if np.linalg.norm(current) > 0:
                unit = current / np.linalg.norm(current)
                direction_errors.append(
                    float(np.linalg.norm(delivered - np.dot(delivered, unit) * unit))
                )
        curves.append(curve)
        outputs.append(vectors)
    xy = np.asarray(curves[0])
    gain = xy[1:, 1] / xy[1:, 0]
    knee = float(manifest.parameters["response_knee"])
    ceiling = knee * (1 + float(manifest.parameters["plateau_width_ratio"]))
    quantization = math.sqrt(3) / 255
    low = (xy[:, 0] > 0) & (xy[:, 0] <= knee)
    high = xy[:, 0] > knee
    checks = {
        "eligible": bool(np.any(low) and np.any(high)),
        "low_response_identity": bool(np.allclose(xy[low, 0], xy[low, 1], atol=1e-12)),
        "monotone_response": bool(np.all(np.diff(xy[:, 1]) >= -1e-12)),
        "high_response_gain_decreases": bool(
            gain[-1] < gain[0]
            and np.all(
                np.diff(gain) <= quantization * (1 / xy[2:, 0] + 1 / xy[1:-1, 0])
            )
        ),
        "bounded_by_plateau": bool(np.max(xy[:, 1]) <= ceiling + quantization),
        "high_end_increment_is_small": bool(xy[-1, 1] - xy[-6, 1] <= quantization),
        "direction_within_quantization": bool(max(direction_errors) <= quantization),
        "bright_baseline_shift_invariant": bool(
            np.allclose(outputs[0], outputs[1], atol=1e-12)
        ),
    }
    return {
        "stimulus": "analytic stimulus NOT empirical/recorded sample",
        "empirical_episode_count": 0,
        "evidence_tier": "G1",
        "claim": "residual contrast compression; not absolute camera clipping or force saturation",
        "baseline_levels": [100, 130],
        "curves": curves,
        "plateau_bound": ceiling,
        "eligible_low_samples": int(np.count_nonzero(low)),
        "eligible_high_samples": int(np.count_nonzero(high)),
        "quantization_bound": quantization,
        "checks": checks,
        "passed": all(checks.values()),
        "max_direction_error": max(direction_errors),
    }


def run_analytical_probes(level: int = 5) -> dict[str, Any]:
    """Return JSON-friendly measurements; never write figures or add dataset records."""
    return cast(
        dict[str, Any], {"F6": probe_f6_recovery(level), "F7": probe_f7_transfer(level)}
    )

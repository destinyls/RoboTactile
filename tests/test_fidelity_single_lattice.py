"""Regression tests for single-source F5/F7 image synthesis."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from robotactile_benchmark.contracts import (
    array_sha256,
    build_evaluation_record,
    canonical_hash,
)
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators.fidelity_transforms import (
    compact_horizontal_warp,
    same_frame_response_compression,
)
from robotactile_benchmark.rest_references import RestReferenceBundle
from robotactile_benchmark.runtime import apply_fault


def _alternate_rest(reference: RestReferenceBundle) -> RestReferenceBundle:
    shape = reference.payload_for("left").shape
    return replace(
        reference,
        reference_id="synthetic-dev-rest-contrast-v2",
        source_artifact_sha256=canonical_hash("alternate-rest-source-v2"),
        no_contact_validation_sha256=canonical_hash("alternate-rest-validation-v2"),
        qualified_record_ids={
            "left": "synthetic-alternate-free-left-0",
            "right": "synthetic-alternate-free-right-0",
        },
        payloads={
            "left": np.full(shape, (220, 15, 170), dtype=np.uint8),
            "right": np.full(shape, (12, 210, 45), dtype=np.uint8),
        },
    )


def _manifest(operator_id: str, references: RestReferenceBundle) -> FaultManifest:
    parameters = (
        {}
        if operator_id
        in {
            "F5_contact_shape_distortion",
            "F7_high_load_saturation",
        }
        else {"rest_reference_sha256": references.sha256}
    )
    return FaultManifest(
        operator_id=operator_id,
        severity_level=3,
        operator_seed=23,
        start_index=3,
        stop_index=10,
        sensor_slots=("left",),
        observability=Observability.BLIND,
        parameters=parameters,
    )


def _payload(records, index: int) -> np.ndarray:
    payload = records[index].observation.sensor("left").payload
    assert payload is not None
    return payload


def _flat_episode(records):
    output = []
    for record in records:
        flat = np.full_like(_payload(records, record.observation.step_index), 96)
        sensor = replace(record.observation.sensor("left"), payload=flat)
        observation = record.observation.replace_sensor(sensor)
        provenance = tuple(
            replace(item, payload_sha256=array_sha256(flat))
            if item.slot_id == "left"
            else item
            for item in record.provenance
        )
        output.append(build_evaluation_record(observation, provenance))
    return tuple(output)


def _box_reflect(image: np.ndarray, radius: int) -> np.ndarray:
    normalized = image.astype(np.float64) / 255.0
    padded = np.pad(
        normalized,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="reflect",
    )
    width = 2 * radius + 1
    output = np.empty_like(normalized)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            output[row, column] = padded[
                row : row + width, column : column + width
            ].mean(axis=(0, 1), dtype=np.float64)
    return output


class SingleLatticeFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = make_synthetic_episode(length=12)
        cls.rest = make_synthetic_rest_references()
        cls.alternate = _alternate_rest(cls.rest)

    def test_f5_and_f7_outputs_do_not_depend_on_rest_rgb(self) -> None:
        for operator_id in (
            "F5_contact_shape_distortion",
            "F7_high_load_saturation",
        ):
            first = apply_fault(
                self.clean,
                _manifest(operator_id, self.rest),
                rest_references=None,
            )
            second = apply_fault(
                self.clean,
                _manifest(operator_id, self.alternate),
                rest_references=self.alternate,
            )
            with self.subTest(operator_id=operator_id):
                self.assertTrue(first.validation.passed, first.validation.failures)
                self.assertTrue(second.validation.passed, second.validation.failures)
                for index in range(3, 10):
                    self.assertTrue(
                        np.array_equal(
                            _payload(first.records, index),
                            _payload(second.records, index),
                        )
                    )

    def test_f5_is_identity_outside_its_compact_warp_support(self) -> None:
        manifest = _manifest("F5_contact_shape_distortion", self.rest)
        result = apply_fault(self.clean, manifest, rest_references=self.rest)
        clean = _payload(self.clean, 6)
        delivered = _payload(result.records, 6)
        center_x, center_y = manifest.parameters["center_xy"]
        radius = min(clean.shape[:2]) * float(
            manifest.parameters["support_radius_fraction"]
        )
        yy, xx = np.mgrid[0 : clean.shape[0], 0 : clean.shape[1]]
        center = (
            float(center_x) * (clean.shape[1] - 1),
            float(center_y) * (clean.shape[0] - 1),
        )
        support = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 < radius**2
        self.assertTrue(np.array_equal(delivered[~support], clean[~support]))
        self.assertFalse(np.array_equal(delivered[support], clean[support]))

    def test_f5_literal_ramp_has_one_coherent_lattice(self) -> None:
        yy, xx = np.mgrid[0:96, 0:128]
        ramp = np.stack((xx, 2 * yy, xx + yy), axis=-1).astype(np.uint8)
        delivered = compact_horizontal_warp(
            ramp,
            center_xy=(0.52802393, 0.47252645),
            support_radius_fraction=0.30,
            displacement_px=3.0,
        )
        self.assertEqual(delivered[45, 68].tolist(), [65, 90, 110])
        self.assertEqual(delivered[45, 75].tolist(), [72, 90, 117])
        self.assertTrue(np.array_equal(delivered[0, 0], ramp[0, 0]))
        self.assertTrue(np.array_equal(delivered[95, 127], ramp[95, 127]))

    def test_f7_matches_same_frame_rational_soft_knee(self) -> None:
        manifest = _manifest("F7_high_load_saturation", self.rest)
        result = apply_fault(self.clean, manifest, rest_references=self.rest)
        clean = _payload(self.clean, 6)
        radius = min(
            (min(clean.shape[:2]) - 1) // 2,
            max(
                int(manifest.parameters["minimum_anchor_radius_px"]),
                int(
                    np.floor(
                        float(manifest.parameters["anchor_radius_fraction"])
                        * min(clean.shape[:2])
                        + 0.5
                    )
                ),
            ),
        )
        anchor = _box_reflect(clean, radius)
        current = clean.astype(np.float64) / 255.0
        detail = current - anchor
        magnitude = np.linalg.norm(detail, axis=-1, keepdims=True)
        knee = float(manifest.parameters["response_knee"])
        plateau = knee * float(manifest.parameters["plateau_width_ratio"])
        excess = np.maximum(magnitude - knee, 0.0)
        compressed = np.where(
            magnitude <= knee,
            magnitude,
            knee + plateau * excess / (plateau + excess),
        )
        scale = np.divide(
            compressed,
            magnitude,
            out=np.ones_like(compressed),
            where=magnitude > 1e-12,
        )
        expected = np.rint(np.clip(anchor + scale * detail, 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
        self.assertTrue(np.array_equal(_payload(result.records, 6), expected))

    def test_f7_is_byte_exact_outside_contact_phases(self) -> None:
        manifest = _manifest("F7_high_load_saturation", self.rest)
        result = apply_fault(self.clean, manifest, rest_references=self.rest)
        for index in (8, 9):
            with self.subTest(index=index):
                self.assertTrue(
                    np.array_equal(
                        _payload(result.records, index), _payload(self.clean, index)
                    )
                )

    def test_f5_and_f7_leave_noncontact_records_fully_untouched(self) -> None:
        for operator_id in (
            "F5_contact_shape_distortion",
            "F7_high_load_saturation",
        ):
            blind = _manifest(operator_id, self.rest)
            declared = FaultManifest.from_dict(
                {**blind.to_dict(), "observability": "declared"}
            )
            result = apply_fault(self.clean, declared, rest_references=None)
            for index in (8, 9):
                with self.subTest(operator_id=operator_id, index=index):
                    self.assertEqual(
                        canonical_hash(result.records[index]),
                        canonical_hash(self.clean[index]),
                    )
                    self.assertEqual(
                        result.records[index].provenance_for("left").active_fault_ids,
                        (),
                    )

    def test_f5_rejects_a_geometric_warp_with_no_visible_texture(self) -> None:
        flat = _flat_episode(self.clean)
        result = apply_fault(
            flat,
            _manifest("F5_contact_shape_distortion", self.rest),
            rest_references=None,
        )
        self.assertFalse(result.validation.passed)
        self.assertIn("F5_NO_VISIBLE_WARP_EFFECT", result.validation.failure_codes)

    def test_f7_rejects_contact_frames_without_response_detail(self) -> None:
        flat = _flat_episode(self.clean)
        result = apply_fault(
            flat,
            _manifest("F7_high_load_saturation", self.rest),
            rest_references=None,
        )
        self.assertFalse(result.validation.passed)
        self.assertIn("F7_NO_DETAIL_SAMPLE", result.validation.failure_codes)

    def test_f7_literal_soft_knee_is_exact_below_and_compressed_above(self) -> None:
        low = np.full((16, 16, 3), 100, dtype=np.uint8)
        low[8, 8] = (130, 100, 100)
        low_delivered = same_frame_response_compression(low, 0.1925, 0.35, 0.06, 2)
        self.assertTrue(np.array_equal(low_delivered, low))

        high = np.full((16, 16, 3), 100, dtype=np.uint8)
        high[8, 8] = (255, 100, 100)
        high_delivered = same_frame_response_compression(high, 0.1925, 0.35, 0.06, 2)
        self.assertEqual(high_delivered[8, 8].tolist(), [170, 100, 100])


if __name__ == "__main__":
    unittest.main()

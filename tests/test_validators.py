import unittest
from dataclasses import replace

import numpy as np

from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.contracts import array_sha256, build_evaluation_record
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.runtime import apply_fault as _runtime_apply_fault
from robotactile_benchmark.validators import validate_delivery as _validate_delivery

REST_REFERENCES = make_synthetic_rest_references()


def apply_fault(clean, manifest):
    return _runtime_apply_fault(clean, manifest, rest_references=REST_REFERENCES)


def validate_delivery(clean, delivered, manifest):
    return _validate_delivery(
        clean, delivered, manifest, rest_references=REST_REFERENCES
    )


def _manifest(operator_id: str) -> FaultManifest:
    start_index = 4 if operator_id == "T1_fixed_source_delay" else 3
    stop_index = 9 if operator_id == "F6_history_residual_imprint" else 8
    return FaultManifest(
        operator_id=operator_id,
        severity_level=3,
        operator_seed=5,
        start_index=start_index,
        stop_index=stop_index,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters=(
            {
                "realization": "registered_pixels",
                "rest_reference_sha256": REST_REFERENCES.sha256,
            }
            if operator_id == "C2_frame_misregistration"
            else (
                {"rest_reference_sha256": REST_REFERENCES.sha256}
                if operator_id in REST_REFERENCE_OPERATOR_IDS
                else {}
            )
        ),
    )


class DeliveryValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clean = make_synthetic_episode(length=11)

    def test_validator_rejects_black_image_substitution_for_absence(self) -> None:
        manifest = _manifest("A1_stream_absence")
        records = list(apply_fault(self.clean, manifest).records)
        bad = records[3]
        sensor = bad.observation.sensor("left")
        black = replace(
            sensor,
            payload=np.zeros_like(self.clean[3].observation.sensor("left").payload),
            payload_present=True,
        )
        observation = bad.observation.replace_sensor(black)
        records[3] = build_evaluation_record(
            observation=observation,
            provenance=bad.provenance,
            clean_record_sha256=bad.clean_record_sha256,
        )

        report = validate_delivery(self.clean, tuple(records), manifest)

        self.assertFalse(report.passed)
        self.assertIn("ABSENCE_NOT_STRUCTURAL", report.failure_codes)

    def test_validator_rejects_invalid_availability_metadata_and_clean_reference(
        self,
    ) -> None:
        manifest = _manifest("A1_stream_absence")
        records = list(apply_fault(self.clean, manifest).records)
        index = manifest.start_index
        record = records[index]
        provenance = replace(
            record.provenance_for("left"),
            source_index=index,
            source_time_s=index / 120.0,
            active_fault_ids=(),
        )
        sensor = replace(record.observation.sensor("left"), declared_validity=True)
        records[index] = build_evaluation_record(
            observation=record.observation.replace_sensor(sensor),
            provenance=tuple(
                provenance if item.slot_id == "left" else item
                for item in record.provenance
            ),
            clean_record_sha256="f" * 64,
        )

        report = validate_delivery(self.clean, tuple(records), manifest)

        self.assertFalse(report.passed)
        self.assertIn("ABSENCE_SOURCE_NOT_NULL", report.failure_codes)
        self.assertIn("OBSERVABILITY_MISMATCH", report.failure_codes)
        self.assertIn("MISSING_FAULT_PROVENANCE", report.failure_codes)
        self.assertIn("CLEAN_REFERENCE_MISMATCH", report.failure_codes)

    def test_validator_rejects_declared_and_blind_observability_tampering(
        self,
    ) -> None:
        for observability, invalid_validity in (
            (Observability.DECLARED, True),
            (Observability.BLIND, False),
        ):
            manifest = replace(
                _manifest("F1_global_response_drift"),
                observability=observability,
            )
            records = list(apply_fault(self.clean, manifest).records)
            index = manifest.start_index
            record = records[index]
            sensor = replace(
                record.observation.sensor("left"),
                declared_validity=invalid_validity,
            )
            records[index] = build_evaluation_record(
                observation=record.observation.replace_sensor(sensor),
                provenance=record.provenance,
                clean_record_sha256=record.clean_record_sha256,
            )

            report = validate_delivery(self.clean, tuple(records), manifest)

            with self.subTest(observability=observability):
                self.assertFalse(report.passed)
                self.assertIn("OBSERVABILITY_MISMATCH", report.failure_codes)

    def test_validator_rejects_temporal_payload_provenance_mismatch(self) -> None:
        manifest = _manifest("T1_fixed_source_delay")
        records = list(apply_fault(self.clean, manifest).records)
        bad = records[5]
        provenance = list(bad.provenance)
        provenance[0] = replace(
            provenance[0],
            source_index=5,
            source_time_s=5.0 / 120.0,
            payload_sha256=self.clean[5].provenance_for("left").payload_sha256,
        )
        records[5] = build_evaluation_record(
            observation=bad.observation,
            provenance=tuple(provenance),
            clean_record_sha256=bad.clean_record_sha256,
        )

        report = validate_delivery(self.clean, tuple(records), manifest)

        self.assertFalse(report.passed)
        self.assertIn("SOURCE_PAYLOAD_MISMATCH", report.failure_codes)

    def test_validator_rejects_changes_outside_registered_window(self) -> None:
        manifest = _manifest("F1_global_response_drift")
        records = list(apply_fault(self.clean, manifest).records)
        records[1] = records[4]

        report = validate_delivery(self.clean, tuple(records), manifest)

        self.assertFalse(report.passed)
        self.assertIn("OUTSIDE_WINDOW_CHANGED", report.failure_codes)

    def test_validator_recomputes_hashes_and_enforces_sensor_scope(self) -> None:
        manifest = _manifest("F1_global_response_drift")
        records = list(apply_fault(self.clean, manifest).records)

        outside = records[1]
        outside_sensor = outside.observation.sensor("left")
        tampered_payload = outside_sensor.payload.copy()
        tampered_payload[0, 0, 0] += 1
        tampered_sensor = replace(outside_sensor, payload=tampered_payload)
        records[1] = replace(
            outside,
            observation=outside.observation.replace_sensor(tampered_sensor),
        )
        stale_report = validate_delivery(self.clean, tuple(records), manifest)
        self.assertFalse(stale_report.passed)
        self.assertIn("STALE_DELIVERED_HASH", stale_report.failure_codes)

        records = list(apply_fault(self.clean, manifest).records)
        index = manifest.start_index
        active = records[index]
        right = active.observation.sensor("right")
        right_payload = right.payload.copy()
        right_payload[0, 0, 0] += 1
        right = replace(right, payload=right_payload)
        right_provenance = replace(
            active.provenance_for("right"),
            payload_sha256=array_sha256(right_payload),
            active_fault_ids=(manifest.operator_id,),
        )
        observation = active.observation.replace_sensor(right)
        provenance = tuple(
            right_provenance if item.slot_id == "right" else item
            for item in active.provenance
        )
        records[index] = build_evaluation_record(
            observation=observation,
            provenance=provenance,
            clean_record_sha256=active.clean_record_sha256,
        )
        scope_report = validate_delivery(self.clean, tuple(records), manifest)
        self.assertFalse(scope_report.passed)
        self.assertIn("WRITE_SET_VIOLATION", scope_report.failure_codes)

        records = list(apply_fault(self.clean, manifest).records)
        active = records[index]
        altered_vision = dict(active.observation.vision)
        top = altered_vision["top"].copy()
        top[0, 0, 0] += 1
        altered_vision["top"] = top
        records[index] = replace(
            active,
            observation=replace(active.observation, vision=altered_vision),
        )
        non_tactile_report = validate_delivery(self.clean, tuple(records), manifest)
        self.assertFalse(non_tactile_report.passed)
        self.assertIn("WRITE_SET_VIOLATION", non_tactile_report.failure_codes)

    def test_validator_rejects_metadata_only_fidelity_and_calibration_faults(
        self,
    ) -> None:
        for operator_id in (
            "F1_global_response_drift",
            "F2_spatial_sensitivity_loss",
            "F3_persistent_surface_artifact",
            "F4_local_nonresponsive_patch",
            "F5_contact_shape_distortion",
            "F6_history_residual_imprint",
            "F7_high_load_saturation",
            "C2_frame_misregistration",
        ):
            manifest = _manifest(operator_id)
            records = list(self.clean)
            for index in range(manifest.start_index, manifest.stop_index):
                record = records[index]
                provenance = list(record.provenance)
                provenance[0] = replace(provenance[0], active_fault_ids=(operator_id,))
                records[index] = build_evaluation_record(
                    observation=record.observation,
                    provenance=tuple(provenance),
                    clean_record_sha256=record.clean_record_sha256,
                )

            report = validate_delivery(self.clean, tuple(records), manifest)

            with self.subTest(operator_id=operator_id):
                self.assertFalse(report.passed)
                self.assertIn("SIGNATURE_NOT_DELIVERED", report.failure_codes)

    def test_each_fidelity_and_c2_report_an_achieved_dose(self) -> None:
        expected_units = {
            "F1_global_response_drift": "retained_global_gain",
            "F2_spatial_sensitivity_loss": "central_residual_attenuation",
            "F3_persistent_surface_artifact": "changed_pixel_fraction",
            "F4_local_nonresponsive_patch": "baseline_replaced_fraction",
            "F5_contact_shape_distortion": "peak_warp_displacement_pixels",
            "F6_history_residual_imprint": "post_release_residual_ratio",
            "F7_high_load_saturation": "same_frame_detail_compression",
            "C2_frame_misregistration": "reprojection_displacement_pixels",
        }
        for operator_id in (
            "F1_global_response_drift",
            "F2_spatial_sensitivity_loss",
            "F3_persistent_surface_artifact",
            "F4_local_nonresponsive_patch",
            "F5_contact_shape_distortion",
            "F6_history_residual_imprint",
            "F7_high_load_saturation",
            "C2_frame_misregistration",
        ):
            manifest = _manifest(operator_id)
            report = apply_fault(self.clean, manifest).validation

            with self.subTest(operator_id=operator_id):
                self.assertTrue(report.passed, report.failures)
                self.assertGreater(float(report.metrics["achieved_dose"]), 0.0)
                self.assertEqual(
                    report.metrics["achieved_dose_unit"],
                    expected_units[operator_id],
                )
                if operator_id == "F1_global_response_drift":
                    self.assertAlmostEqual(
                        float(report.metrics["achieved_dose"]),
                        float(manifest.parameters["target_gain"]),
                        places=2,
                    )
                if operator_id == "C2_frame_misregistration":
                    self.assertAlmostEqual(
                        float(report.metrics["achieved_dose"]),
                        float(manifest.severity_level),
                    )

    def test_validator_rejects_a_different_severity_signature(self) -> None:
        for operator_id in (
            "F1_global_response_drift",
            "F2_spatial_sensitivity_loss",
            "F3_persistent_surface_artifact",
            "F4_local_nonresponsive_patch",
            "F5_contact_shape_distortion",
            "F6_history_residual_imprint",
            "F7_high_load_saturation",
            "C2_frame_misregistration",
        ):
            expected_manifest = _manifest(operator_id).reparameterized(severity_level=1)
            wrong_manifest = _manifest(operator_id).reparameterized(severity_level=5)
            records = apply_fault(self.clean, wrong_manifest).records

            report = validate_delivery(self.clean, records, expected_manifest)

            with self.subTest(operator_id=operator_id):
                self.assertFalse(report.passed)
                self.assertIn("OPERATOR_SIGNATURE_MISMATCH", report.failure_codes)

    def test_validator_rejects_operator_effect_outside_native_footprint(self) -> None:
        for operator_id in (
            "A1_stream_absence",
            "A2_frame_erasure",
            "T2_held_last_freeze",
        ):
            manifest = _manifest(operator_id)
            if operator_id == "T2_held_last_freeze":
                manifest = manifest.reparameterized(severity_level=1)
            records = list(apply_fault(self.clean, manifest).records)
            extra_index = manifest.stop_index - 1
            if operator_id.startswith("A"):
                record = records[extra_index]
                sensor = record.observation.sensor("left").without_payload(
                    declared=True
                )
                provenance = replace(
                    record.provenance_for("left"),
                    source_index=None,
                    source_time_s=None,
                    payload_sha256=None,
                    active_fault_ids=(operator_id,),
                )
                observation = record.observation.replace_sensor(sensor)
                records[extra_index] = build_evaluation_record(
                    observation=observation,
                    provenance=tuple(
                        provenance if item.slot_id == "left" else item
                        for item in record.provenance
                    ),
                    clean_record_sha256=record.clean_record_sha256,
                )
            else:
                source = self.clean[manifest.start_index - 1]
                record = records[extra_index]
                sensor = replace(
                    source.observation.sensor("left"),
                    delivery_index=record.observation.sensor("left").delivery_index,
                    delivery_time_s=record.observation.sensor("left").delivery_time_s,
                )
                source_provenance = replace(
                    source.provenance_for("left"),
                    active_fault_ids=(operator_id,),
                )
                observation = record.observation.replace_sensor(sensor)
                records[extra_index] = build_evaluation_record(
                    observation=observation,
                    provenance=tuple(
                        source_provenance if item.slot_id == "left" else item
                        for item in record.provenance
                    ),
                    clean_record_sha256=record.clean_record_sha256,
                )

            report = validate_delivery(self.clean, tuple(records), manifest)

            with self.subTest(operator_id=operator_id):
                self.assertFalse(report.passed)
                self.assertIn("NATIVE_FOOTPRINT_VIOLATION", report.failure_codes)

    def test_validator_rejects_temporal_source_time_and_c1_payload_mismatch(
        self,
    ) -> None:
        temporal_manifest = _manifest("T1_fixed_source_delay")
        temporal_records = list(apply_fault(self.clean, temporal_manifest).records)
        index = temporal_manifest.start_index
        record = temporal_records[index]
        provenance = list(record.provenance)
        provenance[0] = replace(provenance[0], source_time_s=index / 120.0)
        temporal_records[index] = build_evaluation_record(
            observation=record.observation,
            provenance=tuple(provenance),
            clean_record_sha256=record.clean_record_sha256,
        )
        temporal_report = validate_delivery(
            self.clean, tuple(temporal_records), temporal_manifest
        )
        self.assertFalse(temporal_report.passed)
        self.assertIn("SOURCE_TIME_MISMATCH", temporal_report.failure_codes)

        c1_manifest = FaultManifest(
            operator_id="C1_sensor_identity_misrouting",
            severity_level=3,
            operator_seed=5,
            start_index=3,
            stop_index=8,
            sensor_slots=("left", "right"),
            observability=Observability.BLIND,
            parameters={},
        )
        c1_records = list(apply_fault(self.clean, c1_manifest).records)
        c1_record = c1_records[3]
        left_sensor = replace(
            c1_record.observation.sensor("left"),
            payload=self.clean[3].observation.sensor("left").payload,
        )
        observation = c1_record.observation.replace_sensor(left_sensor)
        c1_records[3] = build_evaluation_record(
            observation=observation,
            provenance=c1_record.provenance,
            clean_record_sha256=c1_record.clean_record_sha256,
        )
        c1_report = validate_delivery(self.clean, tuple(c1_records), c1_manifest)
        self.assertFalse(c1_report.passed)
        self.assertIn("SOURCE_PAYLOAD_MISMATCH", c1_report.failure_codes)


if __name__ == "__main__":
    unittest.main()

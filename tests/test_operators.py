import unittest
from dataclasses import replace

import numpy as np

from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators import EXPECTED_OPERATOR_IDS
from robotactile_benchmark.runtime import apply_fault as _runtime_apply_fault

REST_REFERENCES = make_synthetic_rest_references()


def apply_fault(clean, manifest):
    return _runtime_apply_fault(clean, manifest, rest_references=REST_REFERENCES)


def _manifest(operator_id: str, severity: int = 3) -> FaultManifest:
    slots = (
        ("left", "right")
        if operator_id
        in {
            "T3_inter_sensor_skew",
            "C1_sensor_identity_misrouting",
        }
        else ("left",)
    )
    start_index = 4 if operator_id == "T1_fixed_source_delay" else 3
    stop_index = 10 if operator_id == "T1_fixed_source_delay" else 9
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity,
        operator_seed=23,
        start_index=start_index,
        stop_index=stop_index,
        sensor_slots=slots,
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


def _sensor(record, slot: str):
    return record.observation.sensor(slot)


def _provenance(record, slot: str):
    return record.provenance_for(slot)


class AtomicOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = make_synthetic_episode(length=12)

    def test_all_fourteen_operators_are_deterministic_and_restore(self) -> None:
        for operator_id in sorted(EXPECTED_OPERATOR_IDS):
            with self.subTest(operator_id=operator_id):
                first = apply_fault(self.clean, _manifest(operator_id))
                second = apply_fault(self.clean, _manifest(operator_id))
                manifest = _manifest(operator_id)
                self.assertTrue(first.validation.passed, first.validation.failures)
                self.assertEqual(first.trace_sha256, second.trace_sha256)
                for index in range(len(self.clean)):
                    if manifest.start_index <= index < manifest.stop_index:
                        continue
                    self.assertEqual(
                        first.records[index].delivered_record_sha256,
                        self.clean[index].delivered_record_sha256,
                    )

    def test_only_availability_operators_remove_payloads(self) -> None:
        for operator_id in sorted(EXPECTED_OPERATOR_IDS):
            result = apply_fault(self.clean, _manifest(operator_id))
            present = [
                _sensor(result.records[index], "left").payload_present
                for index in range(3, 9)
            ]
            with self.subTest(operator_id=operator_id):
                if operator_id.startswith("A"):
                    self.assertIn(False, present)
                    for index in range(3, 9):
                        sensor = _sensor(result.records[index], "left")
                        if not sensor.payload_present:
                            self.assertIsNone(sensor.payload)
                else:
                    self.assertTrue(all(present))

    def test_observability_changes_only_the_current_health_alert(self) -> None:
        for operator_id in sorted(EXPECTED_OPERATOR_IDS):
            declared_manifest = _manifest(operator_id)
            blind_manifest = replace(
                declared_manifest, observability=Observability.BLIND
            )
            declared = apply_fault(self.clean, declared_manifest).records
            blind = apply_fault(self.clean, blind_manifest).records
            for index in range(len(self.clean)):
                for slot_id in declared_manifest.sensor_slots:
                    declared_faults = (
                        declared[index].provenance_for(slot_id).active_fault_ids
                    )
                    blind_faults = blind[index].provenance_for(slot_id).active_fault_ids
                    if operator_id in declared_faults:
                        with self.subTest(
                            operator_id=operator_id, index=index, mode="declared"
                        ):
                            self.assertIs(
                                declared[index]
                                .observation.sensor(slot_id)
                                .declared_validity,
                                False,
                            )
                    if operator_id in blind_faults:
                        blind_sensor = blind[index].observation.sensor(slot_id)
                        with self.subTest(
                            operator_id=operator_id, index=index, mode="blind"
                        ):
                            if blind_sensor.payload_present:
                                self.assertEqual(
                                    blind_sensor.declared_validity,
                                    self.clean[index]
                                    .observation.sensor(slot_id)
                                    .declared_validity,
                                )
                            else:
                                self.assertIsNone(blind_sensor.declared_validity)

    def test_fidelity_signatures_are_mechanically_distinct(self) -> None:
        clean_contact = _sensor(self.clean[6], "left").payload.astype(np.float32)
        clean_release = _sensor(self.clean[8], "left").payload.astype(np.float32)
        clean_free = _sensor(self.clean[1], "left").payload

        f1 = apply_fault(self.clean, _manifest("F1_global_response_drift")).records
        self.assertNotAlmostEqual(
            float(_sensor(f1[6], "left").payload.mean()), float(clean_contact.mean())
        )

        f2 = apply_fault(self.clean, _manifest("F2_spatial_sensitivity_loss")).records
        baseline = clean_free.astype(np.float32)
        clean_center = np.abs(
            clean_contact[14:18, 14:18] - baseline[14:18, 14:18]
        ).mean()
        f2_center = np.abs(
            _sensor(f2[6], "left").payload[14:18, 14:18].astype(np.float32)
            - baseline[14:18, 14:18]
        ).mean()
        self.assertLess(f2_center, clean_center)

        f3 = apply_fault(
            self.clean, _manifest("F3_persistent_surface_artifact")
        ).records
        self.assertFalse(
            np.array_equal(
                _sensor(f3[4], "left").payload, _sensor(self.clean[4], "left").payload
            )
        )
        self.assertFalse(
            np.array_equal(
                _sensor(f3[4], "left").payload, _sensor(f3[6], "left").payload
            )
        )

        f4 = apply_fault(self.clean, _manifest("F4_local_nonresponsive_patch")).records
        self.assertTrue(
            np.array_equal(
                _sensor(f4[6], "left").payload[14:18, 14:18],
                clean_free[14:18, 14:18],
            )
        )

        f5 = apply_fault(self.clean, _manifest("F5_contact_shape_distortion")).records
        self.assertTrue(np.array_equal(_sensor(f5[1], "left").payload, clean_free))
        self.assertFalse(np.array_equal(_sensor(f5[6], "left").payload, clean_contact))
        self.assertTrue(np.array_equal(_sensor(f5[8], "left").payload, clean_release))

        f6 = apply_fault(self.clean, _manifest("F6_history_residual_imprint")).records
        self.assertGreater(
            np.abs(_sensor(f6[8], "left").payload.astype(np.float32) - baseline).mean(),
            np.abs(clean_release - baseline).mean(),
        )

        f7 = apply_fault(self.clean, _manifest("F7_high_load_saturation")).records
        self.assertFalse(np.array_equal(_sensor(f7[6], "left").payload, clean_contact))
        self.assertTrue(np.array_equal(_sensor(f7[8], "left").payload, clean_release))

    def test_f6_matches_the_registered_equation(self) -> None:
        f6_manifest = _manifest("F6_history_residual_imprint")
        f6 = apply_fault(self.clean, f6_manifest).records
        baseline = _sensor(self.clean[0], "left").payload.astype(np.float32) / 255.0
        history = np.zeros_like(baseline)
        alpha = 0.35
        beta = 0.72
        for index in range(len(self.clean)):
            clean = _sensor(self.clean[index], "left").payload
            residual = clean.astype(np.float32) / 255.0 - baseline
            history = beta * history + (1.0 - beta) * residual
            if not f6_manifest.active(index):
                continue
            expected = np.rint(
                np.clip(
                    baseline + (1.0 - alpha) * residual + alpha * history,
                    0.0,
                    1.0,
                )
                * 255.0
            ).astype(np.uint8)
            self.assertTrue(
                np.array_equal(_sensor(f6[index], "left").payload, expected)
            )

    def test_temporal_and_context_operators_change_provenance_correctly(self) -> None:
        t1 = apply_fault(self.clean, _manifest("T1_fixed_source_delay")).records
        self.assertEqual(_provenance(t1[5], "left").source_index, 1)
        self.assertEqual(_provenance(t1[6], "left").source_index, 2)

        t2 = apply_fault(self.clean, _manifest("T2_held_last_freeze")).records
        held = [_provenance(t2[index], "left").source_index for index in range(3, 9)]
        self.assertEqual(len(set(held)), 1)

        t3 = apply_fault(self.clean, _manifest("T3_inter_sensor_skew")).records
        self.assertNotEqual(
            _provenance(t3[6], "left").source_index,
            _provenance(t3[6], "right").source_index,
        )

        c1 = apply_fault(self.clean, _manifest("C1_sensor_identity_misrouting")).records
        self.assertEqual(_provenance(c1[3], "left").physical_source_id, "right")
        self.assertEqual(_provenance(c1[3], "right").physical_source_id, "left")
        self.assertEqual(_provenance(c1[6], "left").physical_source_id, "left")

        c2 = apply_fault(self.clean, _manifest("C2_frame_misregistration")).records
        self.assertFalse(
            np.array_equal(
                _sensor(c2[6], "left").payload,
                _sensor(self.clean[6], "left").payload,
            )
        )
        self.assertNotEqual(
            _provenance(c2[6], "left").calibration_sha256,
            _provenance(self.clean[6], "left").calibration_sha256,
        )


if __name__ == "__main__":
    unittest.main()

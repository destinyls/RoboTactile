import unittest

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


def _manifest(operator_id: str, severity: int) -> FaultManifest:
    slots = (
        ("left", "right")
        if operator_id
        in {
            "T3_inter_sensor_skew",
            "C1_sensor_identity_misrouting",
        }
        else ("left",)
    )
    start_index = 17 if operator_id == "T1_fixed_source_delay" else 4
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity,
        operator_seed=31,
        start_index=start_index,
        stop_index=22,
        sensor_slots=slots,
        observability=Observability.BLIND,
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


def _image_delta(clean, delivered, start: int, stop: int) -> float:
    values = []
    for index in range(start, stop):
        clean_payload = clean[index].observation.sensor("left").payload
        delivered_payload = delivered[index].observation.sensor("left").payload
        if delivered_payload is not None:
            values.append(
                float(
                    np.abs(
                        delivered_payload.astype(np.float32)
                        - clean_payload.astype(np.float32)
                    ).mean()
                )
            )
    return float(np.mean(values)) if values else 0.0


class SeverityMatrixTests(unittest.TestCase):
    def test_all_fourteen_by_five_cells_validate_deterministically(self) -> None:
        clean = make_synthetic_episode(length=24)
        for operator_id in sorted(EXPECTED_OPERATOR_IDS):
            for severity in range(1, 6):
                with self.subTest(operator_id=operator_id, severity=severity):
                    manifest = _manifest(operator_id, severity)
                    first = apply_fault(clean, manifest)
                    second = apply_fault(clean, manifest)
                    self.assertTrue(first.validation.passed, first.validation.failures)
                    self.assertEqual(first.trace_sha256, second.trace_sha256)

    def test_registered_severity_paths_are_monotone_in_native_effect(self) -> None:
        clean = make_synthetic_episode(length=24)
        operators = {
            "A1_stream_absence",
            "A2_frame_erasure",
            "F1_global_response_drift",
            "F2_spatial_sensitivity_loss",
            "F3_persistent_surface_artifact",
            "F4_local_nonresponsive_patch",
            "F5_contact_shape_distortion",
            "F6_history_residual_imprint",
            "F7_high_load_saturation",
            "T1_fixed_source_delay",
            "T2_held_last_freeze",
            "T3_inter_sensor_skew",
            "C1_sensor_identity_misrouting",
            "C2_frame_misregistration",
        }
        for operator_id in sorted(operators):
            doses = []
            for severity in range(1, 6):
                manifest = _manifest(operator_id, severity)
                result = apply_fault(clean, manifest)
                if operator_id.startswith("A"):
                    dose = sum(
                        not result.records[index]
                        .observation.sensor("left")
                        .payload_present
                        for index in range(manifest.start_index, manifest.stop_index)
                    )
                elif operator_id in {
                    "T1_fixed_source_delay",
                    "T2_held_last_freeze",
                }:
                    if operator_id == "T2_held_last_freeze":
                        dose = sum(
                            result.records[index].provenance_for("left").source_index
                            != index
                            for index in range(
                                manifest.start_index, manifest.stop_index
                            )
                        )
                    else:
                        dose = max(
                            index
                            - result.records[index].provenance_for("left").source_index
                            for index in range(
                                manifest.start_index, manifest.stop_index
                            )
                        )
                elif operator_id == "T3_inter_sensor_skew":
                    dose = max(
                        abs(
                            result.records[index].provenance_for("left").source_index
                            - result.records[index].provenance_for("right").source_index
                        )
                        for index in range(manifest.start_index, manifest.stop_index)
                    )
                elif operator_id == "C1_sensor_identity_misrouting":
                    dose = sum(
                        result.records[index].provenance_for("left").physical_source_id
                        != "left"
                        for index in range(manifest.start_index, manifest.stop_index)
                    )
                elif operator_id in {
                    "F5_contact_shape_distortion",
                    "F7_high_load_saturation",
                    "C2_frame_misregistration",
                }:
                    dose = float(result.validation.metrics["achieved_dose"])
                else:
                    dose = _image_delta(
                        clean, result.records, manifest.start_index, manifest.stop_index
                    )
                doses.append(float(dose))
            with self.subTest(operator_id=operator_id, doses=doses):
                self.assertTrue(
                    all(right >= left for left, right in zip(doses, doses[1:])),
                    doses,
                )

    def test_window_fraction_and_duration_ladders_have_literal_effects(self) -> None:
        clean = make_synthetic_episode(length=24)
        expected_fraction_counts = [1, 2, 4, 7, 14]
        expected_freeze_counts = [2, 4, 8, 16, 18]
        for operator_id in (
            "A1_stream_absence",
            "A2_frame_erasure",
            "C1_sensor_identity_misrouting",
        ):
            observed = []
            for severity in range(1, 6):
                manifest = _manifest(operator_id, severity)
                result = apply_fault(clean, manifest)
                if operator_id.startswith("A"):
                    count = sum(
                        not result.records[index]
                        .observation.sensor("left")
                        .payload_present
                        for index in range(4, 22)
                    )
                else:
                    count = sum(
                        result.records[index].provenance_for("left").physical_source_id
                        != "left"
                        for index in range(4, 22)
                    )
                observed.append(count)
            with self.subTest(operator_id=operator_id):
                self.assertEqual(observed, expected_fraction_counts)

        observed_freeze = []
        for severity in range(1, 6):
            manifest = _manifest("T2_held_last_freeze", severity)
            result = apply_fault(clean, manifest)
            observed_freeze.append(
                sum(
                    result.records[index].provenance_for("left").source_index != index
                    for index in range(4, 22)
                )
            )
        self.assertEqual(observed_freeze, expected_freeze_counts)

    def test_a2_schedules_gaps_while_a1_is_one_contiguous_outage(self) -> None:
        clean = make_synthetic_episode(length=24)
        a1 = apply_fault(clean, _manifest("A1_stream_absence", 2)).records
        a2 = apply_fault(clean, _manifest("A2_frame_erasure", 2)).records
        a1_missing = [
            index
            for index in range(4, 22)
            if not a1[index].observation.sensor("left").payload_present
        ]
        a2_missing = [
            index
            for index in range(4, 22)
            if not a2[index].observation.sensor("left").payload_present
        ]

        self.assertEqual(a1_missing, [4, 5])
        self.assertEqual(len(a2_missing), 2)
        self.assertGreater(a2_missing[1] - a2_missing[0], 1)

        # Adjacent manifest offsets encode a contiguous transport burst without
        # changing the registered erased-frame-fraction severity path.
        later_burst = _manifest("A2_frame_erasure", 2)
        later_burst = FaultManifest(
            operator_id=later_burst.operator_id,
            severity_level=later_burst.severity_level,
            operator_seed=later_burst.operator_seed,
            start_index=later_burst.start_index,
            stop_index=later_burst.stop_index,
            sensor_slots=later_burst.sensor_slots,
            observability=later_burst.observability,
            parameters={"erased_offsets": [8, 9]},
        )
        later_records = apply_fault(clean, later_burst).records
        self.assertEqual(
            [
                index
                for index in range(4, 22)
                if not later_records[index].observation.sensor("left").payload_present
            ],
            [12, 13],
        )


if __name__ == "__main__":
    unittest.main()

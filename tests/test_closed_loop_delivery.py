"""Behavioral tests for causal online fault delivery."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Callable, Tuple
from unittest.mock import patch

from robotactile_benchmark.closed_loop.delivery import (
    DeliveryFinalization,
    IdentityDeliverySession,
    OnlineFaultSession,
)
from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.contracts import (
    EvaluationRecord,
    SensorProvenance,
    build_evaluation_record,
    canonical_hash,
    delivered_hash,
)
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators import EXPECTED_OPERATOR_IDS
from robotactile_benchmark.runtime import apply_fault, trace_hash
from robotactile_benchmark.validators import ValidationReport

REST_REFERENCES = make_synthetic_rest_references()


def _manifest(operator_id: str, severity: int = 3) -> FaultManifest:
    slots = (
        ("left", "right")
        if operator_id in {"C1_sensor_identity_misrouting", "T3_inter_sensor_skew"}
        else ("left",)
    )
    start_index = 16 if operator_id == "T1_fixed_source_delay" else 3
    parameters = (
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
    )
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity,
        operator_seed=23,
        start_index=start_index,
        stop_index=22 if operator_id == "T1_fixed_source_delay" else 9,
        sensor_slots=slots,
        observability=Observability.DECLARED,
        parameters=parameters,
    )


def _record_with(
    record: EvaluationRecord,
    *,
    observation=None,
    provenance: Tuple[SensorProvenance, ...] | None = None,
) -> EvaluationRecord:
    return build_evaluation_record(
        observation=record.observation if observation is None else observation,
        provenance=record.provenance if provenance is None else provenance,
        clean_record_sha256=record.clean_record_sha256,
    )


class OnlineFaultDeliveryTests(unittest.TestCase):
    def test_online_prefixes_match_canonical_batch_for_all_operators_and_severities(
        self,
    ) -> None:
        clean = make_synthetic_episode(length=24)
        for operator_id in sorted(EXPECTED_OPERATOR_IDS):
            for severity in range(1, 6):
                with self.subTest(operator_id=operator_id, severity=severity):
                    manifest = _manifest(operator_id, severity)
                    expected = apply_fault(clean, manifest, REST_REFERENCES)
                    session = OnlineFaultSession(manifest, REST_REFERENCES)
                    delivered = tuple(session.deliver(record) for record in clean)
                    finalization = session.finalize()

                    self.assertEqual(
                        tuple(record.delivered_record_sha256 for record in delivered),
                        tuple(
                            record.delivered_record_sha256
                            for record in expected.records
                        ),
                    )
                    self.assertEqual(
                        finalization.delivered_trace_sha256,
                        expected.trace_sha256,
                    )
                    self.assertTrue(
                        finalization.validation is not None
                        and finalization.validation.passed,
                        finalization.validation,
                    )

    def test_delivery_does_not_rewrite_already_delivered_history(self) -> None:
        clean = make_synthetic_episode(length=12)
        manifest = _manifest("F6_history_residual_imprint")
        expected = apply_fault(clean, manifest, REST_REFERENCES).records
        session = OnlineFaultSession(manifest, REST_REFERENCES)
        delivered = []
        for record in clean:
            delivered.append(session.deliver(record))
            self.assertEqual(
                tuple(item.delivered_record_sha256 for item in delivered),
                tuple(
                    item.delivered_record_sha256 for item in expected[: len(delivered)]
                ),
            )
        finalization = session.finalize()
        self.assertEqual(
            tuple(item.delivered_record_sha256 for item in delivered),
            tuple(
                item.delivered_record_sha256 for item in finalization.delivered_records
            ),
        )

        self.assertEqual(session.streaming_record_count, len(clean))

    def test_temporal_delivery_retains_only_registered_causal_history(self) -> None:
        clean = make_synthetic_episode(length=24)
        for operator_id in (
            "T1_fixed_source_delay",
            "T2_held_last_freeze",
            "T3_inter_sensor_skew",
        ):
            with self.subTest(operator_id=operator_id):
                manifest = _manifest(operator_id, severity=3)
                session = OnlineFaultSession(manifest, REST_REFERENCES)
                for record in clean:
                    session.deliver(record)
                    self.assertLessEqual(
                        session.retained_temporal_source_count,
                        session.maximum_temporal_source_age + 1,
                    )
                self.assertEqual(session.streaming_record_count, len(clean))

    def test_online_delivery_preserves_the_original_manifest(self) -> None:
        clean = make_synthetic_episode(length=12)
        manifest = _manifest("A2_frame_erasure", severity=4)
        original_dict = manifest.to_dict()
        original_hash = manifest.sha256
        session = OnlineFaultSession(manifest, REST_REFERENCES)
        for record in clean:
            session.deliver(record)
        finalization = session.finalize()

        self.assertEqual(manifest.to_dict(), original_dict)
        self.assertEqual(manifest.sha256, original_hash)
        self.assertEqual(finalization.manifest_sha256, original_hash)

    def test_fault_finalization_calls_authoritative_runtime_trace_hash(self) -> None:
        clean = make_synthetic_episode(length=12)
        manifest = _manifest("A1_stream_absence")
        session = OnlineFaultSession(manifest, REST_REFERENCES)
        for record in clean:
            session.deliver(record)

        with patch(
            "robotactile_benchmark.closed_loop.delivery.trace_hash",
            wraps=trace_hash,
            create=True,
        ) as mocked_trace_hash:
            finalization = session.finalize()

        mocked_trace_hash.assert_called_once_with(
            finalization.delivered_records, manifest
        )
        self.assertEqual(
            finalization.delivered_trace_sha256,
            trace_hash(finalization.delivered_records, manifest),
        )

    def test_identity_delivery_returns_original_objects_for_clean_and_no_touch(
        self,
    ) -> None:
        clean = make_synthetic_episode(length=12)
        session = IdentityDeliverySession()
        delivered = tuple(session.deliver(record) for record in clean)
        finalization = session.finalize()

        self.assertEqual(len(delivered), len(clean))
        for delivered_record, clean_record in zip(delivered, clean):
            self.assertIs(delivered_record, clean_record)
        self.assertIsNone(finalization.validation)
        self.assertIsNone(finalization.manifest_sha256)
        self.assertEqual(
            finalization.clean_trace_sha256, finalization.delivered_trace_sha256
        )

    def test_fault_window_is_identity_at_its_exclusive_stop_index(self) -> None:
        clean = make_synthetic_episode(length=12)
        manifest = _manifest("A2_frame_erasure")
        session = OnlineFaultSession(manifest, REST_REFERENCES)
        delivered = tuple(session.deliver(record) for record in clean)

        self.assertEqual(
            delivered[manifest.stop_index].delivered_record_sha256,
            clean[manifest.stop_index].delivered_record_sha256,
        )
        incomplete = OnlineFaultSession(manifest, REST_REFERENCES)
        for record in clean[:4]:
            incomplete.deliver(record)
        report = incomplete.finalize().validation
        self.assertIsNotNone(report)
        self.assertIn("A2_RESUME_MISSING", report.failure_codes)
        self.assertFalse(report.passed)

    def test_delivery_rejects_stream_contract_violations(self) -> None:
        clean = make_synthetic_episode(length=12)
        altered_observations: tuple[tuple[str, Callable[[], EvaluationRecord]], ...] = (
            (
                "non-dense step",
                lambda: _record_with(
                    clean[1], observation=replace(clean[1].observation, step_index=2)
                ),
            ),
            (
                "episode",
                lambda: _record_with(
                    clean[1],
                    observation=replace(
                        clean[1].observation, episode_id="other-episode"
                    ),
                ),
            ),
            (
                "task",
                lambda: _record_with(
                    clean[1],
                    observation=replace(clean[1].observation, task="other-task"),
                ),
            ),
            (
                "seed",
                lambda: _record_with(
                    clean[1], observation=replace(clean[1].observation, seed=2)
                ),
            ),
            (
                "frame identity",
                lambda: _record_with(
                    clean[1],
                    observation=clean[1].observation.replace_sensor(
                        replace(clean[1].observation.sensor("left"), frame_id="other")
                    ),
                ),
            ),
            (
                "calibration identity",
                lambda: _record_with(
                    clean[1],
                    observation=clean[1].observation.replace_sensor(
                        replace(
                            clean[1].observation.sensor("left"),
                            calibration_id="other-calibration",
                        )
                    ),
                ),
            ),
            (
                "source identity",
                lambda: _record_with(
                    clean[1],
                    provenance=(
                        replace(
                            clean[1].provenance_for("left"),
                            physical_source_id="other-source",
                        ),
                        clean[1].provenance_for("right"),
                    ),
                ),
            ),
            (
                "calibration hash",
                lambda: _record_with(
                    clean[1],
                    provenance=(
                        replace(
                            clean[1].provenance_for("left"),
                            calibration_sha256="d" * 64,
                        ),
                        clean[1].provenance_for("right"),
                    ),
                ),
            ),
        )
        for name, make_record in altered_observations:
            with self.subTest(name=name):
                session = IdentityDeliverySession()
                session.deliver(clean[0])
                with self.assertRaises(ValueError):
                    session.deliver(make_record())

    def test_delivery_rejects_faulted_and_stale_input_records(self) -> None:
        clean = make_synthetic_episode(length=12)
        faulted = _record_with(
            clean[0],
            provenance=(
                replace(
                    clean[0].provenance_for("left"),
                    active_fault_ids=("F1_global_response_drift",),
                ),
                clean[0].provenance_for("right"),
            ),
        )
        stale = replace(clean[0], delivered_record_sha256="0" * 64)
        forged_clean = replace(clean[0], clean_record_sha256="0" * 64)

        for name, record in (
            ("faulted", faulted),
            ("stale", stale),
            ("forged clean", forged_clean),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                IdentityDeliverySession().deliver(record)

    def test_rest_reference_preconditions_fail_closed(self) -> None:
        manifest = _manifest("F1_global_response_drift")
        with self.assertRaises(ValueError):
            OnlineFaultSession(manifest)
        with self.assertRaises(ValueError):
            OnlineFaultSession(manifest, replace(REST_REFERENCES, reference_id="other"))

    def test_rest_reference_calibration_must_match_on_first_record(self) -> None:
        clean = make_synthetic_episode(length=12)
        mismatched = _record_with(
            clean[0],
            provenance=(
                replace(clean[0].provenance_for("left"), calibration_sha256="d" * 64),
                clean[0].provenance_for("right"),
            ),
        )
        with self.assertRaises(ValueError):
            OnlineFaultSession(
                _manifest("F1_global_response_drift"), REST_REFERENCES
            ).deliver(mismatched)

    def test_finalization_lifecycle_rejects_empty_repeated_and_late_operations(
        self,
    ) -> None:
        clean = make_synthetic_episode(length=12)
        session = IdentityDeliverySession()
        with self.assertRaises(ValueError):
            session.finalize()
        session.deliver(clean[0])
        session.finalize()
        with self.assertRaises(ValueError):
            session.finalize()
        with self.assertRaises(ValueError):
            session.deliver(clean[1])

    def test_inactive_contact_gated_fault_retains_failed_validation(self) -> None:
        clean = make_synthetic_episode(length=12)
        manifest = FaultManifest(
            operator_id="F5_contact_shape_distortion",
            severity_level=3,
            operator_seed=23,
            start_index=0,
            stop_index=2,
            sensor_slots=("left",),
            observability=Observability.DECLARED,
            parameters={},
        )
        session = OnlineFaultSession(manifest)
        for record in clean:
            session.deliver(record)
        finalization = session.finalize()

        self.assertIsNotNone(finalization.validation)
        self.assertFalse(finalization.validation.passed)
        self.assertIn("INACTIVE_OPERATOR", finalization.validation.failure_codes)

    def test_finalization_validates_records_traces_and_frozen_metrics(self) -> None:
        clean = make_synthetic_episode(length=12)
        identity_records = clean[:1]
        identity_trace = canonical_hash(
            {
                "namespace": "robotactile_benchmark.closed_loop.clean_trace.v1",
                "records": [
                    delivered_hash(record.observation, record.provenance)
                    for record in identity_records
                ],
            }
        )
        with self.assertRaises(ValueError):
            DeliveryFinalization(
                clean_records=identity_records,
                delivered_records=identity_records,
                validation=None,
                clean_trace_sha256="a" * 64,
                delivered_trace_sha256="a" * 64,
                manifest_sha256=None,
            )
        with self.assertRaises(ValueError):
            DeliveryFinalization(
                clean_records=identity_records,
                delivered_records=(
                    replace(identity_records[0], clean_record_sha256="0" * 64),
                ),
                validation=None,
                clean_trace_sha256=identity_trace,
                delivered_trace_sha256=identity_trace,
                manifest_sha256=None,
            )
        manifest = _manifest("A1_stream_absence")
        delivered = apply_fault(clean, manifest, REST_REFERENCES).records
        clean_trace = canonical_hash(
            {
                "namespace": "robotactile_benchmark.closed_loop.clean_trace.v1",
                "records": [
                    delivered_hash(record.observation, record.provenance)
                    for record in clean
                ],
            }
        )
        report_metrics = {"nested": {"values": [1]}}
        finalization = DeliveryFinalization(
            clean_records=clean,
            delivered_records=delivered,
            validation=ValidationReport(True, (), (), report_metrics),
            clean_trace_sha256=clean_trace,
            delivered_trace_sha256=trace_hash(delivered, manifest),
            manifest_sha256=manifest.sha256,
        )
        report_metrics["nested"]["values"].append(2)
        self.assertEqual(finalization.validation.metrics["nested"]["values"], (1,))
        with self.assertRaises(TypeError):
            finalization.validation.metrics["nested"]["other"] = "mutate"
        for name, records in (
            (
                "stale delivered",
                (replace(delivered[0], delivered_record_sha256="0" * 64),)
                + delivered[1:],
            ),
            (
                "wrong clean reference",
                (replace(delivered[0], clean_record_sha256="0" * 64),) + delivered[1:],
            ),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                DeliveryFinalization(
                    clean_records=clean,
                    delivered_records=records,
                    validation=ValidationReport(True, (), (), {}),
                    clean_trace_sha256=clean_trace,
                    delivered_trace_sha256=trace_hash(records, manifest),
                    manifest_sha256=manifest.sha256,
                )


if __name__ == "__main__":
    unittest.main()

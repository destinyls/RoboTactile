"""Streaming delivery equivalence, causality, and bounded-work regressions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from robotactile_benchmark.closed_loop.delivery import OnlineFaultSession
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators.fidelity import GlobalResponseDriftOperator
from robotactile_benchmark.runtime import apply_fault

REST_REFERENCES = make_synthetic_rest_references()


def _manifest(
    operator_id: str,
    *,
    severity: int,
    start: int,
    stop: int,
    slots: tuple[str, ...],
) -> FaultManifest:
    parameters = (
        {"rest_reference_sha256": REST_REFERENCES.sha256}
        if operator_id.startswith("F1_")
        else {}
    )
    return FaultManifest(
        operator_id=operator_id,
        severity_level=severity,
        operator_seed=29,
        start_index=start,
        stop_index=stop,
        sensor_slots=slots,
        observability=Observability.BLIND,
        parameters=parameters,
    )


class StreamingFaultTests(unittest.TestCase):
    def test_online_delivery_does_not_call_batch_operator_apply(self) -> None:
        clean = make_synthetic_episode(length=12)
        manifest = _manifest(
            "F1_global_response_drift",
            severity=3,
            start=3,
            stop=9,
            slots=("left",),
        )
        with patch.object(
            GlobalResponseDriftOperator,
            "apply",
            side_effect=AssertionError("batch prefix replay is forbidden"),
        ):
            session = OnlineFaultSession(manifest, REST_REFERENCES)
            delivered = tuple(session.deliver(record) for record in clean)
            finalization = session.finalize()

        self.assertEqual(len(delivered), len(clean))
        self.assertEqual(session.streaming_record_count, len(clean))
        self.assertTrue(finalization.validation and finalization.validation.passed)

    def test_t1_routes_both_slots_to_one_fixed_historical_age(self) -> None:
        clean = make_synthetic_episode(length=24)
        manifest = _manifest(
            "T1_fixed_source_delay",
            severity=3,
            start=16,
            stop=22,
            slots=("left", "right"),
        )
        expected = apply_fault(clean, manifest).records
        session = OnlineFaultSession(manifest)
        delivered = tuple(session.deliver(record) for record in clean)

        self.assertEqual(session.maximum_temporal_source_age, 4)
        for index in range(manifest.start_index, manifest.stop_index):
            for slot_id in manifest.sensor_slots:
                self.assertEqual(
                    delivered[index].provenance_for(slot_id).source_index,
                    index - 4,
                )
                self.assertEqual(
                    delivered[index].observation.sensor(slot_id).delivery_index,
                    index,
                )
        self.assertEqual(
            tuple(record.delivered_record_sha256 for record in delivered),
            tuple(record.delivered_record_sha256 for record in expected),
        )

    def test_t3_keeps_left_current_and_delays_right_by_registered_gap(self) -> None:
        clean = make_synthetic_episode(length=16)
        manifest = _manifest(
            "T3_inter_sensor_skew",
            severity=3,
            start=5,
            stop=12,
            slots=("left", "right"),
        )
        session = OnlineFaultSession(manifest)
        delivered = tuple(session.deliver(record) for record in clean)

        self.assertEqual(session.maximum_temporal_source_age, 3)
        for index in range(manifest.start_index, manifest.stop_index):
            self.assertEqual(
                delivered[index].provenance_for("left").source_index, index
            )
            self.assertEqual(
                delivered[index].provenance_for("right").source_index, index - 3
            )
        self.assertEqual(
            delivered[manifest.stop_index].delivered_record_sha256,
            clean[manifest.stop_index].delivered_record_sha256,
        )

    def test_temporal_cache_and_work_remain_bounded_at_long_horizon(self) -> None:
        clean = make_synthetic_episode(length=300)
        manifest = _manifest(
            "T1_fixed_source_delay",
            severity=5,
            start=16,
            stop=290,
            slots=("left", "right"),
        )
        session = OnlineFaultSession(manifest)
        maximum_retained = 0
        for expected_count, record in enumerate(clean, start=1):
            session.deliver(record)
            self.assertEqual(session.streaming_record_count, expected_count)
            maximum_retained = max(
                maximum_retained, session.retained_temporal_source_count
            )
        self.assertLessEqual(maximum_retained, 17)


if __name__ == "__main__":
    unittest.main()

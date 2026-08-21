import unittest
from dataclasses import replace

import numpy as np

from robotactile_benchmark.contracts import (
    ObservationRecord,
    SensorObservation,
    build_evaluation_record,
    canonical_hash,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.rest_references import ReferenceSplit, RestReferenceBundle


class ObservationContractTests(unittest.TestCase):
    def test_payload_arrays_are_defensively_copied_and_read_only(self) -> None:
        source = np.full((8, 8, 3), 17, dtype=np.uint8)
        sensor = SensorObservation(
            slot_id="left",
            payload=source,
            payload_present=True,
            declared_validity=True,
            delivery_index=0,
            delivery_time_s=0.0,
            visible_source_time_s=None,
            frame_id="left_raw",
            calibration_id="calibration-v1",
        )
        source[0, 0, 0] = 99

        self.assertEqual(int(sensor.payload[0, 0, 0]), 17)
        with self.assertRaises(ValueError):
            sensor.payload[0, 0, 0] = 5

    def test_absence_is_structural_and_differs_from_a_black_payload(self) -> None:
        absent = SensorObservation(
            slot_id="left",
            payload=None,
            payload_present=False,
            declared_validity=False,
            delivery_index=3,
            delivery_time_s=0.025,
            visible_source_time_s=None,
            frame_id="left_raw",
            calibration_id="calibration-v1",
        )
        black = replace(
            absent,
            payload=np.zeros((8, 8, 3), dtype=np.uint8),
            payload_present=True,
        )

        self.assertIsNone(absent.payload)
        self.assertIsNotNone(black.payload)
        self.assertNotEqual(canonical_hash(absent), canonical_hash(black))
        with self.assertRaises(ValueError):
            replace(absent, payload_present=True)

        visible_time = replace(
            black,
            visible_source_time_s=0.02,
            delivery_time_s=0.03,
        )
        hidden = visible_time.without_payload(declared=False)
        self.assertIsNone(hidden.visible_source_time_s)
        self.assertIsNone(hidden.declared_validity)

    def test_model_visible_arrays_must_be_finite(self) -> None:
        invalid = np.zeros((8, 8, 3), dtype=np.float32)
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            SensorObservation(
                slot_id="left",
                payload=invalid,
                payload_present=True,
                declared_validity=True,
                delivery_index=0,
                delivery_time_s=0.0,
                visible_source_time_s=None,
                frame_id="left_raw",
                calibration_id="calibration-v1",
            )

    def test_tactile_payloads_use_one_canonical_uint8_encoding(self) -> None:
        for payload in (
            np.zeros((8, 8, 3), dtype=np.float32),
            np.full((8, 8, 3), 127.0, dtype=np.float32),
        ):
            with (
                self.subTest(maximum=float(payload.max())),
                self.assertRaisesRegex(TypeError, "uint8"),
            ):
                SensorObservation(
                    slot_id="left",
                    payload=payload,
                    payload_present=True,
                    declared_validity=True,
                    delivery_index=0,
                    delivery_time_s=0.0,
                    visible_source_time_s=None,
                    frame_id="left_raw",
                    calibration_id="calibration-v1",
                )

    def test_record_indices_times_and_seed_are_strict_finite_values(self) -> None:
        clean = make_synthetic_episode(length=10)[0]
        with self.assertRaises(TypeError):
            replace(clean.observation.sensor("left"), delivery_index=1.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(clean.observation.sensor("left"), delivery_time_s=float("nan"))
        with self.assertRaises(TypeError):
            replace(clean.observation, seed=True)
        with self.assertRaises(TypeError):
            replace(clean.observation, step_index=1.5)
        with self.assertRaises(TypeError):
            replace(clean.provenance[0], source_index=1.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(clean.provenance[0], source_time_s=float("inf"))

    def test_evaluation_record_rejects_a_future_source(self) -> None:
        clean = make_synthetic_episode(length=10)
        record = clean[4]
        bad_provenance = list(record.provenance)
        bad_provenance[0] = replace(
            bad_provenance[0], source_index=5, source_time_s=5.0 / 120.0
        )

        with self.assertRaisesRegex(ValueError, "future source"):
            build_evaluation_record(
                observation=record.observation,
                provenance=tuple(bad_provenance),
                clean_record_sha256=record.clean_record_sha256,
            )

    def test_records_require_exact_unique_slots_and_provenance(self) -> None:
        clean = make_synthetic_episode(length=10)[0]
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            ObservationRecord(
                episode_id="episode",
                task="task",
                seed=1,
                step_index=0,
                tactile=(clean.observation.sensor("left"),),
                vision=clean.observation.vision,
                proprio=clean.observation.proprio,
            )
        duplicate = (clean.provenance[0], clean.provenance[0])
        with self.assertRaisesRegex(ValueError, "slots must match"):
            build_evaluation_record(clean.observation, duplicate)

    def test_rest_reference_bundle_is_non_test_verified_and_content_addressed(
        self,
    ) -> None:
        clean = make_synthetic_episode(length=10)[0]
        payloads = {
            slot: clean.observation.sensor(slot).payload for slot in ("left", "right")
        }
        bundle = RestReferenceBundle(
            reference_id="dev-rest-v1",
            dataset_split=ReferenceSplit.DEVELOPMENT,
            split_manifest_sha256="d" * 64,
            source_artifact_sha256="a" * 64,
            no_contact_predicate_id="depth-below-threshold-v1",
            no_contact_validation_sha256="e" * 64,
            no_contact_verified=True,
            qualified_record_ids={"left": "dev-left-0", "right": "dev-right-0"},
            calibration_sha256={"left": "b" * 64, "right": "c" * 64},
            payloads=payloads,
        )
        self.assertEqual(len(bundle.sha256), 64)
        with self.assertRaises(ValueError):
            replace(bundle, dataset_split="held-out-test")
        with self.assertRaisesRegex(ValueError, "no-contact"):
            replace(bundle, no_contact_verified=False)
        with self.assertRaisesRegex(ValueError, "no-contact"):
            replace(bundle, no_contact_verified="true")
        exposed = bundle.payload_for("left")
        exposed.setflags(write=True)
        exposed[0, 0, 0] = 255
        self.assertNotEqual(int(bundle.payload_for("left")[0, 0, 0]), 255)

    def test_canonical_hash_is_stable_across_mapping_order(self) -> None:
        left = {"b": [2, 3], "a": {"x": 1}}
        right = {"a": {"x": 1}, "b": [2, 3]}

        self.assertEqual(canonical_hash(left), canonical_hash(right))

    def test_canonical_hash_rejects_non_finite_scalars(self) -> None:
        for invalid in (float("nan"), float("inf"), float("-inf"), np.float32("nan")):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                canonical_hash({"value": invalid})
        with self.assertRaisesRegex(TypeError, "mapping keys"):
            canonical_hash({1: "integer", "1": "string"})
        with self.assertRaisesRegex(TypeError, "JSON"):
            canonical_hash({"unordered": {1, 2}})


if __name__ == "__main__":
    unittest.main()

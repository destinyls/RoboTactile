import math
import unittest
from dataclasses import replace

from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.fixtures import (
    make_synthetic_episode,
    make_synthetic_rest_references,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators import (
    EXPECTED_OPERATOR_IDS,
    OperatorRegistry,
    get_operator,
    list_operator_ids,
)
from robotactile_benchmark.resources import load_operator_registry
from robotactile_benchmark.runtime import apply_fault as _runtime_apply_fault

REST_REFERENCES = make_synthetic_rest_references()


def apply_fault(clean, manifest):
    return _runtime_apply_fault(clean, manifest, rest_references=REST_REFERENCES)


EXPECTED_IDS = {
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


class RegistryAndManifestTests(unittest.TestCase):
    def test_core_registry_is_exactly_the_paper_2_7_3_2_set(self) -> None:
        self.assertEqual(EXPECTED_OPERATOR_IDS, EXPECTED_IDS)
        self.assertEqual(set(list_operator_ids()), EXPECTED_IDS)
        for operator_id in EXPECTED_IDS:
            self.assertEqual(get_operator(operator_id).spec.operator_id, operator_id)

    def test_machine_registry_metadata_matches_runtime_specs(self) -> None:
        entries = {
            entry["operator_id"]: entry
            for entry in load_operator_registry()["operators"]
        }
        self.assertEqual(set(entries), EXPECTED_IDS)
        for operator_id, entry in entries.items():
            spec = get_operator(operator_id).spec
            with self.subTest(operator_id=operator_id):
                self.assertEqual(entry["family"], spec.family)
                self.assertEqual(entry["native_unit"], spec.native_unit)
                self.assertEqual(entry["stateful"], spec.stateful)
                self.assertEqual(entry["evidence_tier"], spec.evidence_tier)

    def test_registry_rejects_duplicate_and_unknown_operator_names(self) -> None:
        registry = OperatorRegistry()

        class StubOperator:
            pass

        registry.register("test_operator", StubOperator)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("test_operator", StubOperator)
        with self.assertRaisesRegex(KeyError, "unknown operator"):
            registry.create("missing")

    def test_fault_manifest_round_trip_has_a_stable_hash(self) -> None:
        manifest = FaultManifest(
            operator_id="F2_spatial_sensitivity_loss",
            severity_level=3,
            operator_seed=17,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={
                "rest_reference_sha256": REST_REFERENCES.sha256,
            },
        )
        restored = FaultManifest.from_dict(manifest.to_dict())

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.sha256, manifest.sha256)
        with self.assertRaises(TypeError):
            manifest.parameters["new"] = 1

    def test_manifest_freezes_every_realized_operator_instance_parameter(self) -> None:
        clean_parameters = {"rest_reference_sha256": REST_REFERENCES.sha256}
        first = FaultManifest(
            operator_id="F4_local_nonresponsive_patch",
            severity_level=3,
            operator_seed=17,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters=clean_parameters,
        )
        second = FaultManifest(
            operator_id="F4_local_nonresponsive_patch",
            severity_level=3,
            operator_seed=18,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters=clean_parameters,
        )
        for field in (
            "parameterization_version",
            "template_seed",
            "center_xy",
            "radius_fraction",
            "fill_mode",
            "instance_descriptor_sha256",
        ):
            self.assertIn(field, first.parameters)
            self.assertIn(field, first.to_dict()["parameters"])
        self.assertNotEqual(
            first.parameters["center_xy"], second.parameters["center_xy"]
        )
        self.assertNotEqual(first.sha256, second.sha256)

    def test_direct_constructor_rejects_boolean_or_fractional_integer_fields(
        self,
    ) -> None:
        common = dict(
            operator_id="A1_stream_absence",
            severity_level=1,
            operator_seed=1,
            start_index=2,
            stop_index=4,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={},
        )
        for field_name, invalid in (
            ("severity_level", True),
            ("operator_seed", 1.5),
            ("start_index", False),
            ("stop_index", 4.2),
        ):
            values = dict(common)
            values[field_name] = invalid
            with self.subTest(field=field_name), self.assertRaises(TypeError):
                FaultManifest(**values)

    def test_fault_manifest_loader_rejects_coercion_extras_and_version_drift(
        self,
    ) -> None:
        manifest = FaultManifest(
            operator_id="A1_stream_absence",
            severity_level=1,
            operator_seed=1,
            start_index=2,
            stop_index=4,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={},
        )
        serialized = manifest.to_dict()
        for mutated in (
            {**serialized, "severity_level": "1"},
            {**serialized, "unknown": True},
            {**serialized, "implementation_version": "future"},
        ):
            with (
                self.subTest(mutated=mutated),
                self.assertRaises((TypeError, ValueError)),
            ):
                FaultManifest.from_dict(mutated)

    def test_fault_manifest_rejects_invalid_window_severity_and_scope(self) -> None:
        common = dict(
            operator_id="A1_stream_absence",
            operator_seed=1,
            observability=Observability.DECLARED,
            parameters={"rest_reference_sha256": REST_REFERENCES.sha256},
        )
        with self.assertRaisesRegex(ValueError, "severity"):
            FaultManifest(
                severity_level=0,
                start_index=2,
                stop_index=4,
                sensor_slots=("left",),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "window"):
            FaultManifest(
                severity_level=1,
                start_index=4,
                stop_index=4,
                sensor_slots=("left",),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "sensor slot"):
            FaultManifest(
                severity_level=1,
                start_index=2,
                stop_index=4,
                sensor_slots=("middle",),
                **common,
            )

    def test_fault_manifest_parameters_are_strict_deterministic_json(self) -> None:
        common = dict(
            operator_id="A1_stream_absence",
            severity_level=1,
            operator_seed=1,
            start_index=2,
            stop_index=4,
            sensor_slots=("left",),
            observability=Observability.BLIND,
        )
        invalid_parameters = (
            {"unordered": {1, 2}},
            {1: "numeric-key"},
            {"not_finite": math.nan},
            {"binary": b"bytes"},
        )
        for parameters in invalid_parameters:
            with (
                self.subTest(parameters=parameters),
                self.assertRaisesRegex((TypeError, ValueError), "parameters"),
            ):
                FaultManifest(parameters=parameters, **common)

        first = FaultManifest(parameters={}, **common)
        second = FaultManifest(parameters={}, **common)
        self.assertEqual(first.sha256, second.sha256)

        with self.assertRaisesRegex(TypeError, "parameters"):
            replace(first, parameters={"unordered": frozenset({1, 2})})

        with self.assertRaisesRegex(ValueError, "unregistered fields"):
            FaultManifest(parameters={"unknown": 1}, **common)

    def test_c2_requires_an_explicit_registered_spatial_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "registered_pixels"):
            FaultManifest(
                operator_id="C2_frame_misregistration",
                severity_level=1,
                operator_seed=1,
                start_index=1,
                stop_index=2,
                sensor_slots=("left",),
                observability=Observability.BLIND,
                parameters={"rest_reference_sha256": REST_REFERENCES.sha256},
            )

    def test_only_rest_bound_pixel_operators_require_rest_references(
        self,
    ) -> None:
        for operator_id in REST_REFERENCE_OPERATOR_IDS:
            parameters = {"rest_reference_sha256": REST_REFERENCES.sha256}
            if operator_id == "C2_frame_misregistration":
                parameters["realization"] = "registered_pixels"
            manifest = FaultManifest(
                operator_id=operator_id,
                severity_level=1,
                operator_seed=1,
                start_index=3,
                stop_index=9,
                sensor_slots=("left",),
                observability=Observability.BLIND,
                parameters=parameters,
            )
            with self.subTest(operator_id=operator_id):
                self.assertEqual(
                    manifest.parameters["rest_reference_sha256"],
                    REST_REFERENCES.sha256,
                )

        for operator_id in {
            "F5_contact_shape_distortion",
            "F7_high_load_saturation",
        }:
            manifest = FaultManifest(
                operator_id=operator_id,
                severity_level=1,
                operator_seed=1,
                start_index=3,
                stop_index=9,
                sensor_slots=("left",),
                observability=Observability.BLIND,
                parameters={},
            )
            with self.subTest(operator_id=operator_id):
                self.assertNotIn("rest_reference_sha256", manifest.parameters)

        with self.assertRaisesRegex(ValueError, "rest_reference_sha256"):
            FaultManifest(
                operator_id="F1_global_response_drift",
                severity_level=1,
                operator_seed=1,
                start_index=3,
                stop_index=9,
                sensor_slots=("left",),
                observability=Observability.BLIND,
                parameters={},
            )
        mismatched = FaultManifest(
            operator_id="F1_global_response_drift",
            severity_level=1,
            operator_seed=1,
            start_index=3,
            stop_index=9,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={"rest_reference_sha256": "a" * 64},
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_fault(make_synthetic_episode(length=10), mismatched)

    def test_a2_freezes_and_validates_an_explicit_erasure_schedule(self) -> None:
        generated = FaultManifest(
            operator_id="A2_frame_erasure",
            severity_level=2,
            operator_seed=7,
            start_index=4,
            stop_index=22,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={},
        )
        self.assertEqual(len(generated.parameters["erased_offsets"]), 2)
        self.assertEqual(tuple(generated.parameters["erased_offsets"]), (0, 17))
        self.assertIn("erased_offsets", generated.to_dict()["parameters"])

        burst = FaultManifest(
            operator_id="A2_frame_erasure",
            severity_level=2,
            operator_seed=7,
            start_index=4,
            stop_index=22,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={"erased_offsets": [2, 3]},
        )
        result = apply_fault(make_synthetic_episode(length=24), burst)
        missing = [
            index
            for index in range(4, 22)
            if not result.records[index].observation.sensor("left").payload_present
        ]
        self.assertEqual(missing, [6, 7])

        with self.assertRaisesRegex(ValueError, "erased_offsets"):
            FaultManifest(
                operator_id="A2_frame_erasure",
                severity_level=2,
                operator_seed=7,
                start_index=4,
                stop_index=22,
                sensor_slots=("left",),
                observability=Observability.BLIND,
                parameters={"erased_offsets": [2]},
            )

        no_resume = FaultManifest(
            operator_id="A2_frame_erasure",
            severity_level=1,
            operator_seed=7,
            start_index=22,
            stop_index=24,
            sensor_slots=("left",),
            observability=Observability.BLIND,
            parameters={"erased_offsets": [1]},
        )
        with self.assertRaisesRegex(ValueError, "resume"):
            apply_fault(make_synthetic_episode(length=24), no_resume)


if __name__ == "__main__":
    unittest.main()

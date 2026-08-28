import json
import unittest
from pathlib import Path

from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.fixtures import make_synthetic_rest_references
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.operators import EXPECTED_OPERATOR_IDS

ROOT = Path(__file__).resolve().parents[1]
VERSIONED_SCHEMA_BRANCHES = {
    "clean_campaign_manifest.schema.json": (
        "manifestBase",
        "manifestV1",
        "manifestV2",
    ),
    "clean_baseline_summary.schema.json": (
        "summaryBase",
        "summaryV1",
        "summaryV2",
    ),
    "clean_paper_result.schema.json": (
        "paperResultBase",
        "paperResultV1",
        "paperResultV2",
        "paperResultV3",
    ),
    "univtac_all_tasks_qualification.schema.json": (
        "qualificationBase",
        "qualificationV1",
        "qualificationV2",
        "qualificationV3",
    ),
}
OPTIONAL_SCHEMA_FIELDS = {
    "live_univtac_root_receipt.schema.json": {"capture_profile"},
    "live_univtac_request.schema.json": {
        "initial_state_policy",
        "wall_timeout_role",
    },
}


def _versioned_branch_fields(
    schema: dict[str, object], base_name: str, branch_name: str
) -> tuple[set[str], set[str], dict[str, object]]:
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    base = definitions[base_name]
    branch = definitions[branch_name]
    assert isinstance(base, dict)
    assert isinstance(branch, dict)
    all_of = branch["allOf"]
    assert isinstance(all_of, list)
    assert all_of[0] == {"$ref": f"#/$defs/{base_name}"}
    overlay = {} if len(all_of) == 1 else all_of[1]
    assert isinstance(overlay, dict)
    base_required = base.get("required", [])
    base_properties = base.get("properties", {})
    overlay_required = overlay.get("required", [])
    overlay_properties = overlay.get("properties", {})
    assert isinstance(base_required, list)
    assert isinstance(base_properties, dict)
    assert isinstance(overlay_required, list)
    assert isinstance(overlay_properties, dict)
    return (
        set(base_required) | set(overlay_required),
        set(base_properties) | set(overlay_properties),
        overlay,
    )


def _manifest(operator_id: str) -> FaultManifest:
    references = make_synthetic_rest_references()
    parameters = {}
    if operator_id in REST_REFERENCE_OPERATOR_IDS:
        parameters["rest_reference_sha256"] = references.sha256
    if operator_id == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    return FaultManifest(
        operator_id=operator_id,
        severity_level=3,
        operator_seed=29,
        start_index=8 if operator_id == "T1_fixed_source_delay" else 3,
        stop_index=12,
        sensor_slots=("left", "right")
        if operator_id in {"T3_inter_sensor_skew", "C1_sensor_identity_misrouting"}
        else ("left",),
        observability=Observability.BLIND,
        parameters=parameters,
    )


class SchemaSynchronizationTests(unittest.TestCase):
    def test_schema_json_is_well_formed_and_version_synchronized(self) -> None:
        schemas = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((ROOT / "schemas").glob("*.json"))
        }
        required_schemas = {
            "calibration_request_receipt.schema.json",
            "clean_baseline_summary.schema.json",
            "clean_campaign_run.schema.json",
            "clean_campaign_manifest.schema.json",
            "clean_paper_result.schema.json",
            "closed_loop_result.schema.json",
            "closed_loop_root_receipt.schema.json",
            "deployment_layout_receipt.schema.json",
            "fault_manifest.schema.json",
            "live_univtac_request.schema.json",
            "live_univtac_root_receipt.schema.json",
            "live_matrix_run_config.schema.json",
            "live_preflight_receipt.schema.json",
            "integrations_lock.schema.json",
            "model_integration_config.schema.json",
            "n0_twam_artifact_manifest.schema.json",
            "n0_observation_parity.schema.json",
            "n0_server_runtime_attestation.schema.json",
            "official_act_artifact_manifest.schema.json",
            "paired_execution_receipt.schema.json",
            "pull_out_key_matrix_receipt.schema.json",
            "matrix_manifest.schema.json",
            "matrix_cell_receipt.schema.json",
            "matrix_summary.schema.json",
            "observation_record.schema.json",
            "primary_matrix_generation_receipt.schema.json",
            "reporting_spec.schema.json",
            "reporting_outcome.schema.json",
            "benchmark_summary.schema.json",
            "report_receipt.schema.json",
            "rest_reference_bundle.schema.json",
            "rest_reference_root_receipt.schema.json",
            "rest_reference_validation.schema.json",
            "trial_manifest.schema.json",
            "univtac_all_tasks_qualification.schema.json",
            "univtac_task_registry.schema.json",
            "validation_report.schema.json",
        }
        self.assertTrue(required_schemas <= set(schemas))
        for name, schema in schemas.items():
            with self.subTest(schema=name):
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertEqual(schema["type"], "object")
                if name in VERSIONED_SCHEMA_BRANCHES:
                    base_name, *branch_names = VERSIONED_SCHEMA_BRANCHES[name]
                    self.assertEqual(
                        schema["oneOf"],
                        [
                            {"$ref": f"#/$defs/{branch_name}"}
                            for branch_name in branch_names
                        ],
                    )
                    self.assertIn(base_name, schema["$defs"])
                    for branch_name in branch_names:
                        self.assertIs(
                            schema["$defs"][branch_name]["unevaluatedProperties"],
                            False,
                        )
                    continue
                self.assertIs(schema["additionalProperties"], False)
                self.assertEqual(
                    set(schema["required"]),
                    set(schema["properties"]) - OPTIONAL_SCHEMA_FIELDS.get(name, set()),
                )
        fault = schemas["fault_manifest.schema.json"]
        self.assertEqual(
            set(fault["properties"]["operator_id"]["enum"]),
            set(EXPECTED_OPERATOR_IDS),
        )
        self.assertEqual(
            fault["properties"]["implementation_version"]["const"], "0.4.0"
        )
        self.assertIn(
            "base_system_manifest_sha256",
            schemas["trial_manifest.schema.json"]["required"],
        )

    def test_live_request_schema_matches_runtime_policy_contract(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/live_univtac_request.schema.json").read_text(
                encoding="utf-8"
            )
        )
        launcher = schema["properties"]["launcher_args"]
        self.assertIs(launcher["additionalProperties"], False)
        self.assertEqual(
            set(launcher["required"]), {"enable_cameras", "headless", "kit_args"}
        )
        self.assertEqual(
            launcher["properties"],
            {
                "enable_cameras": {"const": True},
                "headless": {"const": True},
                "kit_args": {"const": "--/app/hangDetector/enabled=false"},
            },
        )

        policy_branch = schema["allOf"][0]
        act_properties = policy_branch["then"]["properties"]
        n0_properties = policy_branch["else"]["properties"]
        self.assertEqual(act_properties["execute_action_steps"], {"const": 1})
        self.assertIs(act_properties["initial_state_policy"], False)
        self.assertIs(act_properties["wall_timeout_role"], False)
        self.assertEqual(n0_properties["execute_action_steps"], {"const": 24})
        self.assertEqual(
            schema["properties"]["initial_state_policy"]["enum"],
            ["replace_initial_terminal_v1", "diagnostic_allow_invalid_v1"],
        )
        self.assertEqual(
            schema["properties"]["wall_timeout_role"],
            {"const": "infrastructure_watchdog_v1"},
        )

        required = set(schema["required"])
        for relative in (
            "examples/act/request.json",
            "examples/n0_twam/request.json",
        ):
            with self.subTest(example=relative):
                document = json.loads((ROOT / relative).read_text(encoding="utf-8"))
                self.assertEqual(set(document), required)
                self.assertEqual(
                    set(document["launcher_args"]), set(launcher["required"])
                )

    def test_live_matrix_and_reporting_schemas_follow_typed_contracts(self) -> None:
        from robotactile_benchmark.calibration.contracts import (
            CALIBRATION_SEMANTIC_VERSION,
            NoContactValidationReceipt,
            RestReferenceArtifactRootReceipt,
        )
        from robotactile_benchmark.calibration.request_contracts import (
            CALIBRATION_REQUEST_SEMANTIC_VERSION,
            CalibrationRequestReceipt,
        )
        from robotactile_benchmark.clean_baseline.contracts import (
            CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION,
            CLEAN_CAMPAIGN_SEMANTIC_VERSION,
            CleanCampaignManifest,
        )
        from robotactile_benchmark.clean_baseline.summary import CleanBaselineSummary
        from robotactile_benchmark.deployment.contracts import (
            DEPLOYMENT_LAYOUT_SEMANTIC_VERSION,
            DeploymentLayoutReceipt,
        )
        from robotactile_benchmark.execution.contracts import (
            LIVE_REQUEST_SEMANTIC_VERSION,
            LiveUniVTACRunRequest,
        )
        from robotactile_benchmark.execution.live_artifacts_contracts import (
            LIVE_ARTIFACT_SEMANTIC_VERSION,
            LiveArtifactRootReceipt,
        )
        from robotactile_benchmark.execution.paired_live_univtac import (
            PAIRED_EXECUTION_SEMANTIC_VERSION,
            PairedLiveUniVTACExecutionResult,
        )
        from robotactile_benchmark.execution.preflight_contracts import (
            LIVE_PREFLIGHT_SEMANTIC_VERSION,
            LivePreflightReceipt,
        )
        from robotactile_benchmark.matrix.contracts import MATRIX_SEMANTIC_VERSION
        from robotactile_benchmark.matrix.live_run_config import (
            LIVE_MATRIX_RUN_CONFIG_SEMANTIC_VERSION,
            LiveMatrixRunConfig,
        )
        from robotactile_benchmark.matrix.manifest import MatrixManifest
        from robotactile_benchmark.matrix.primary_generation_contracts import (
            PRIMARY_GENERATION_SEMANTIC_VERSION,
            PrimaryMatrixGenerationReceipt,
        )
        from robotactile_benchmark.matrix.results import MatrixCellReceipt
        from robotactile_benchmark.matrix.summary import MatrixSummary
        from robotactile_benchmark.reporting.bundle import ReportReceipt
        from robotactile_benchmark.reporting.contracts import (
            REPORTING_SEMANTIC_VERSION,
            OutcomeRecord,
            ReportingSpec,
        )
        from robotactile_benchmark.reporting.summary_contracts import BenchmarkSummary
        from robotactile_benchmark.rest_references import RestReferenceBundle

        schemas = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((ROOT / "schemas").glob("*.json"))
        }
        exact_fields = {
            "deployment_layout_receipt.schema.json": set(
                DeploymentLayoutReceipt.__dataclass_fields__
            ),
            "calibration_request_receipt.schema.json": set(
                CalibrationRequestReceipt.__dataclass_fields__
            ),
            "live_univtac_request.schema.json": set(
                LiveUniVTACRunRequest.__dataclass_fields__
            ),
            "live_univtac_root_receipt.schema.json": set(
                LiveArtifactRootReceipt.__dataclass_fields__
            ),
            "live_matrix_run_config.schema.json": set(
                LiveMatrixRunConfig.__dataclass_fields__
            ),
            "live_preflight_receipt.schema.json": set(
                LivePreflightReceipt.__dataclass_fields__
            ),
            "paired_execution_receipt.schema.json": (
                set(PairedLiveUniVTACExecutionResult.__dataclass_fields__)
                - {"witness_indices"}
                | {"reset_receipt_sha256"}
            ),
            "matrix_manifest.schema.json": set(MatrixManifest.__dataclass_fields__),
            "primary_matrix_generation_receipt.schema.json": set(
                PrimaryMatrixGenerationReceipt.__dataclass_fields__
            ),
            "matrix_cell_receipt.schema.json": set(
                MatrixCellReceipt.__dataclass_fields__
            ),
            "matrix_summary.schema.json": set(MatrixSummary.__dataclass_fields__)
            | {"status_counts"},
            "reporting_spec.schema.json": set(ReportingSpec.__dataclass_fields__),
            "reporting_outcome.schema.json": set(OutcomeRecord.__dataclass_fields__),
            "benchmark_summary.schema.json": set(BenchmarkSummary.__dataclass_fields__),
            "report_receipt.schema.json": set(ReportReceipt.__dataclass_fields__),
            "rest_reference_bundle.schema.json": set(
                RestReferenceBundle.__dataclass_fields__
            ),
            "rest_reference_root_receipt.schema.json": set(
                RestReferenceArtifactRootReceipt.__dataclass_fields__
            ),
            "rest_reference_validation.schema.json": set(
                NoContactValidationReceipt.__dataclass_fields__
            ),
        }
        versions = {
            "deployment_layout_receipt.schema.json": (
                DEPLOYMENT_LAYOUT_SEMANTIC_VERSION
            ),
            "calibration_request_receipt.schema.json": (
                CALIBRATION_REQUEST_SEMANTIC_VERSION
            ),
            "live_univtac_request.schema.json": LIVE_REQUEST_SEMANTIC_VERSION,
            "live_univtac_root_receipt.schema.json": LIVE_ARTIFACT_SEMANTIC_VERSION,
            "live_matrix_run_config.schema.json": (
                LIVE_MATRIX_RUN_CONFIG_SEMANTIC_VERSION
            ),
            "live_preflight_receipt.schema.json": LIVE_PREFLIGHT_SEMANTIC_VERSION,
            "paired_execution_receipt.schema.json": (PAIRED_EXECUTION_SEMANTIC_VERSION),
            "matrix_manifest.schema.json": MATRIX_SEMANTIC_VERSION,
            "primary_matrix_generation_receipt.schema.json": (
                PRIMARY_GENERATION_SEMANTIC_VERSION
            ),
            "matrix_cell_receipt.schema.json": MATRIX_SEMANTIC_VERSION,
            "matrix_summary.schema.json": MATRIX_SEMANTIC_VERSION,
            "reporting_spec.schema.json": REPORTING_SEMANTIC_VERSION,
            "reporting_outcome.schema.json": REPORTING_SEMANTIC_VERSION,
            "benchmark_summary.schema.json": REPORTING_SEMANTIC_VERSION,
            "report_receipt.schema.json": REPORTING_SEMANTIC_VERSION,
            "rest_reference_root_receipt.schema.json": CALIBRATION_SEMANTIC_VERSION,
            "rest_reference_validation.schema.json": CALIBRATION_SEMANTIC_VERSION,
        }
        for name, fields in exact_fields.items():
            with self.subTest(schema=name):
                schema = schemas[name]
                self.assertEqual(
                    set(schema["required"]),
                    fields - OPTIONAL_SCHEMA_FIELDS.get(name, set()),
                )
                self.assertEqual(set(schema["properties"]), fields)
                if name in versions:
                    version_schema = schema["properties"]["semantic_version"]
                    if name == "live_univtac_root_receipt.schema.json":
                        self.assertIn(versions[name], version_schema["enum"])
                    else:
                        self.assertEqual(version_schema["const"], versions[name])
        manifest_schema = schemas["clean_campaign_manifest.schema.json"]
        manifest_fields = set(CleanCampaignManifest.__dataclass_fields__)
        manifest_v1_required, manifest_v1_properties, manifest_v1_overlay = (
            _versioned_branch_fields(manifest_schema, "manifestBase", "manifestV1")
        )
        manifest_v2_required, manifest_v2_properties, manifest_v2_overlay = (
            _versioned_branch_fields(manifest_schema, "manifestBase", "manifestV2")
        )
        self.assertEqual(manifest_v1_required, manifest_fields - {"sampling"})
        self.assertEqual(manifest_v1_properties, manifest_fields - {"sampling"})
        self.assertEqual(manifest_v2_required, manifest_fields)
        self.assertEqual(manifest_v2_properties, manifest_fields)
        self.assertEqual(
            manifest_v1_overlay["properties"]["semantic_version"]["const"],
            CLEAN_CAMPAIGN_SEMANTIC_VERSION,
        )
        self.assertEqual(
            manifest_v2_overlay["properties"]["semantic_version"]["const"],
            "2.0",
        )

        from robotactile_benchmark.clean_baseline.summary import CleanTaskSummary

        summary_schema = schemas["clean_baseline_summary.schema.json"]
        v2_summary_fields = {
            "target_valid_trial_count",
            "candidate_trial_count",
            "attempted_candidate_count",
            "exception_replacement_count",
            "unused_reserve_count",
            "valid_outcome_count",
            "missing_required_candidate_count",
            "candidate_provenance",
            "exception_attempt_receipt_sha256s",
        }
        summary_fields = set(CleanBaselineSummary.__dataclass_fields__)
        summary_v1_required, summary_v1_properties, summary_v1_overlay = (
            _versioned_branch_fields(summary_schema, "summaryBase", "summaryV1")
        )
        summary_v2_required, summary_v2_properties, summary_v2_overlay = (
            _versioned_branch_fields(summary_schema, "summaryBase", "summaryV2")
        )
        self.assertEqual(summary_v1_required, summary_fields - v2_summary_fields)
        self.assertEqual(summary_v1_properties, summary_fields - v2_summary_fields)
        self.assertEqual(summary_v2_required, summary_fields)
        self.assertEqual(summary_v2_properties, summary_fields)
        self.assertEqual(
            summary_v1_overlay["properties"]["semantic_version"]["const"],
            CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION,
        )
        self.assertEqual(
            summary_v2_overlay["properties"]["semantic_version"]["const"],
            "2.0",
        )

        v2_task_fields = v2_summary_fields - {
            "candidate_provenance",
            "exception_attempt_receipt_sha256s",
        }
        task_fields = set(CleanTaskSummary.__dataclass_fields__)
        task_v1_required, task_v1_properties, _ = _versioned_branch_fields(
            summary_schema, "taskBase", "taskV1"
        )
        task_v2_required, task_v2_properties, _ = _versioned_branch_fields(
            summary_schema, "taskBase", "taskV2"
        )
        self.assertEqual(task_v1_required, task_fields - v2_task_fields)
        self.assertEqual(task_v1_properties, task_fields - v2_task_fields)
        self.assertEqual(task_v2_required, task_fields)
        self.assertEqual(task_v2_properties, task_fields)
        for name in (
            "official_act_artifact_manifest.schema.json",
            "pull_out_key_matrix_receipt.schema.json",
        ):
            self.assertEqual(
                schemas[name]["properties"]["semantic_version"]["const"], "1.0"
            )

        known_ids = {schema["$id"] for schema in schemas.values()}

        def external_references(value: object) -> set[str]:
            if isinstance(value, dict):
                own = {
                    item
                    for key, item in value.items()
                    if key == "$ref"
                    and isinstance(item, str)
                    and item.startswith("https://robotactile.invalid/")
                }
                return own | set().union(
                    *(external_references(item) for item in value.values())
                )
            if isinstance(value, list):
                return set().union(*(external_references(item) for item in value))
            return set()

        for name, schema in schemas.items():
            with self.subTest(external_references=name):
                self.assertTrue(external_references(schema) <= known_ids)

    def test_univtac_schemas_are_strict_and_synchronized(self) -> None:
        registry_document = json.loads(
            (ROOT / "configs" / "univtac" / "tasks_v1.json").read_text(encoding="utf-8")
        )
        registry_schema = json.loads(
            (ROOT / "schemas" / "univtac_task_registry.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(registry_schema["additionalProperties"])
        self.assertEqual(set(registry_schema["required"]), set(registry_document))
        task_schema = registry_schema["$defs"]["task"]
        self.assertFalse(task_schema["additionalProperties"])
        for document, contract in zip(
            registry_document["tasks"],
            registry_schema["properties"]["tasks"]["prefixItems"],
        ):
            with self.subTest(task_id=document["task_id"]):
                self.assertEqual(set(task_schema["required"]), set(document))
                constants = contract["allOf"][1]["properties"]
                for name in (
                    "task_id",
                    "prompt_id",
                    "prompt",
                    "module_name",
                    "class_name",
                    "success_predicate_id",
                    "task_source_sha256",
                    "action_horizon",
                    "early_stop_capable",
                    "predicate_note",
                ):
                    self.assertEqual(constants[name]["const"], document[name])

    def test_every_operator_manifest_materializes_schema_registered_keys(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "fault_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        known = set(schema["properties"]["parameters"]["properties"])
        for operator_id in sorted(EXPECTED_OPERATOR_IDS):
            manifest = _manifest(operator_id)
            with self.subTest(operator_id=operator_id):
                self.assertTrue(set(manifest.to_dict()["parameters"]) <= known)
                self.assertEqual(
                    manifest.parameters["parameterization_version"],
                    "canonical_operator_instance_v2",
                )

    def test_single_frame_operators_forbid_rest_reference_fields(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "fault_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        conditions = {
            entry["if"]["properties"]["operator_id"]["const"]: entry["then"]
            for entry in schema["allOf"]
        }
        for operator_id in (
            "F5_contact_shape_distortion",
            "F7_high_load_saturation",
        ):
            with self.subTest(operator_id=operator_id):
                self.assertEqual(
                    conditions[operator_id]["properties"]["parameters"]["not"],
                    {"required": ["rest_reference_sha256"]},
                )


if __name__ == "__main__":
    unittest.main()

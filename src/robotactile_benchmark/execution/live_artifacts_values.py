"""Path-free run identity, validation, and rest-reference artifact values."""

from __future__ import annotations

from collections.abc import Mapping

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.closed_loop.contracts import (
    ClosedLoopRunSpec,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import canonical_hash, freeze_value, thaw_value
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_ARTIFACT_SEMANTIC_VERSION,
    LiveArtifactValidationError,
    require_live_sha256,
    require_optional_live_sha256,
)
from robotactile_benchmark.execution.live_artifacts_fs import LiveBundleSnapshot
from robotactile_benchmark.execution.live_artifacts_io import (
    LiveArrayWriter,
    load_live_array,
)
from robotactile_benchmark.execution.loading import LoadedLiveUniVTACRun
from robotactile_benchmark.rest_references import (
    ReferenceSplit,
    RestReferenceBundle,
)
from robotactile_benchmark.trials import Condition, TrialManifest
from robotactile_benchmark.validators import ValidationReport

_REQUEST_FIELDS = frozenset(
    {
        "task_id",
        "condition",
        "policy_kind",
        "base_system_id",
        "dataset_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "requested_base_system_manifest_sha256",
        "initial_seed",
        "exogenous_seed",
        "max_control_cycles",
        "max_observation_steps",
        "execute_action_steps",
        "wall_timeout_s",
        "restoration_index",
        "restoration_mode",
        "matched_no_touch_system_id",
        "act_device_name",
        "simulator_device",
        "launcher_args",
        "n0_source_commit",
        "n0_normalizer_sha256",
        "n0_serve_bundle_sha256",
        "n0_prompt_manifest_sha256",
        "backend_config_sha256",
        "policy_identity_sha256",
        "trial_manifest_sha256",
        "run_spec_sha256",
        "fault_manifest_sha256",
        "rest_references_sha256",
        "source_binding",
        "semantic_version",
    }
)
_SOURCE_FIELDS = frozenset(
    {"upstream_commit", "registry_resource_sha256", "task_source_sha256"}
)
_VALIDATION_FIELDS = frozenset(
    {
        "report",
        "manifest_sha256",
        "clean_trace_sha256",
        "delivered_trace_sha256",
        "semantic_version",
    }
)
_REPORT_FIELDS = frozenset({"passed", "failure_codes", "failures", "metrics"})
_REST_FIELDS = frozenset(
    {
        "reference_id",
        "dataset_split",
        "split_manifest_sha256",
        "source_artifact_sha256",
        "no_contact_predicate_id",
        "no_contact_validation_sha256",
        "no_contact_verified",
        "qualified_record_ids",
        "calibration_sha256",
        "payloads",
    }
)


def live_request_identity(loaded: LoadedLiveUniVTACRun) -> dict[str, object]:
    """Build the path-free request/source identity stored by the live exporter."""

    request = loaded.request
    config = loaded.backend_config
    return {
        "task_id": request.task_id,
        "condition": request.condition.value,
        "policy_kind": request.policy_kind.value,
        "base_system_id": request.base_system_id,
        "dataset_sha256": request.dataset_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "config_sha256": request.config_sha256,
        "requested_base_system_manifest_sha256": request.base_system_manifest_sha256,
        "initial_seed": request.initial_seed,
        "exogenous_seed": request.exogenous_seed,
        "max_control_cycles": request.max_control_cycles,
        "max_observation_steps": request.max_observation_steps,
        "execute_action_steps": request.execute_action_steps,
        "wall_timeout_s": request.wall_timeout_s,
        "restoration_index": request.restoration_index,
        "restoration_mode": (
            None if request.restoration_mode is None else request.restoration_mode.value
        ),
        "matched_no_touch_system_id": request.matched_no_touch_system_id,
        "act_device_name": request.act_device_name,
        "simulator_device": request.simulator_device,
        "launcher_args": thaw_value(request.launcher_args),
        "n0_source_commit": request.n0_source_commit,
        "n0_normalizer_sha256": request.n0_normalizer_sha256,
        "n0_serve_bundle_sha256": request.n0_serve_bundle_sha256,
        "n0_prompt_manifest_sha256": request.n0_prompt_manifest_sha256,
        "backend_config_sha256": config.sha256,
        "policy_identity_sha256": loaded.policy_identity.sha256,
        "trial_manifest_sha256": loaded.trial.sha256,
        "run_spec_sha256": loaded.run_spec.sha256,
        "fault_manifest_sha256": (
            None if loaded.fault_manifest is None else loaded.fault_manifest.sha256
        ),
        "rest_references_sha256": (
            None if loaded.rest_references is None else loaded.rest_references.sha256
        ),
        "source_binding": {
            "upstream_commit": config.upstream_commit,
            "registry_resource_sha256": config.registry_resource_sha256,
            "task_source_sha256": config.task.task_source_sha256,
        },
        "semantic_version": LIVE_ARTIFACT_SEMANTIC_VERSION,
    }


def validate_live_request_identity(
    value: object,
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault_sha256: str | None,
    rest_sha256: str | None,
) -> dict[str, object]:
    """Strictly bind a stored path-free request back to packaged UniVTAC."""

    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        raise LiveArtifactValidationError("live request identity fields mismatch")
    if value["semantic_version"] != LIVE_ARTIFACT_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live request identity version mismatch")
    source = value["source_binding"]
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise LiveArtifactValidationError("live source binding fields mismatch")
    config = build_univtac_backend_config(_string(value["task_id"], "task id"))
    expected_source = {
        "upstream_commit": config.upstream_commit,
        "registry_resource_sha256": config.registry_resource_sha256,
        "task_source_sha256": config.task.task_source_sha256,
    }
    if source != expected_source or value["backend_config_sha256"] != config.sha256:
        raise LiveArtifactValidationError("live source/backend binding drift")
    expected_policy = PolicyIdentity(
        system_id=trial.executed_system_id,
        checkpoint_sha256=trial.checkpoint_sha256,
        config_sha256=trial.config_sha256,
        action_spec=trial.action_spec,
        consumes_tactile=trial.condition is not Condition.NO_TOUCH,
        supports_structural_absence=trial.condition is Condition.NO_TOUCH,
    )
    checks = (
        value["task_id"] == trial.task,
        value["condition"] == trial.condition.value,
        value["base_system_id"] == trial.base_system_id,
        value["dataset_sha256"] == trial.dataset_sha256,
        value["checkpoint_sha256"] == trial.checkpoint_sha256,
        value["config_sha256"] == trial.config_sha256,
        value["initial_seed"] == trial.initial_seed,
        value["exogenous_seed"] == trial.exogenous_seed,
        value["max_control_cycles"] == run_spec.max_control_cycles,
        value["max_observation_steps"] == run_spec.max_observation_steps,
        value["execute_action_steps"] == run_spec.execute_action_steps,
        value["wall_timeout_s"] == run_spec.wall_timeout_s,
        value["restoration_index"] == trial.restoration_index,
        value["restoration_mode"]
        == (None if trial.restoration_mode is None else trial.restoration_mode.value),
        value["matched_no_touch_system_id"] == trial.matched_no_touch_system_id,
        value["trial_manifest_sha256"] == trial.sha256,
        value["run_spec_sha256"] == run_spec.sha256,
        value["fault_manifest_sha256"] == fault_sha256,
        value["rest_references_sha256"] == rest_sha256,
        value["policy_identity_sha256"] == expected_policy.sha256,
    )
    if not all(checks):
        raise LiveArtifactValidationError("live request identity cross-link mismatch")
    for name in (
        "dataset_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "backend_config_sha256",
        "policy_identity_sha256",
        "trial_manifest_sha256",
        "run_spec_sha256",
    ):
        require_live_sha256(value[name], name)
    require_optional_live_sha256(value["fault_manifest_sha256"], "fault manifest")
    require_optional_live_sha256(value["rest_references_sha256"], "rest references")
    freeze_value(value["launcher_args"])
    return value


def run_content_sha256_from_identity(value: Mapping[str, object]) -> str:
    return canonical_hash(
        {
            "backend_config_sha256": value["backend_config_sha256"],
            "policy_identity_sha256": value["policy_identity_sha256"],
            "trial_manifest_sha256": value["trial_manifest_sha256"],
            "run_spec_sha256": value["run_spec_sha256"],
            "fault_manifest_sha256": value["fault_manifest_sha256"],
            "rest_references_sha256": value["rest_references_sha256"],
            "policy_kind": value["policy_kind"],
        }
    )


def source_binding_sha256(value: Mapping[str, object]) -> str:
    return canonical_hash(value["source_binding"])


def validation_to_live_dict(
    finalization: DeliveryFinalization | None,
) -> dict[str, object]:
    report = None if finalization is None else finalization.validation
    return {
        "report": (
            None
            if report is None
            else {
                "passed": report.passed,
                "failure_codes": list(report.failure_codes),
                "failures": list(report.failures),
                "metrics": thaw_value(report.metrics),
            }
        ),
        "manifest_sha256": (
            None if finalization is None else finalization.manifest_sha256
        ),
        "clean_trace_sha256": (
            None if finalization is None else finalization.clean_trace_sha256
        ),
        "delivered_trace_sha256": (
            None if finalization is None else finalization.delivered_trace_sha256
        ),
        "semantic_version": LIVE_ARTIFACT_SEMANTIC_VERSION,
    }


def validation_from_live_dict(
    value: object,
) -> tuple[ValidationReport | None, dict[str, str | None]]:
    if not isinstance(value, dict) or set(value) != _VALIDATION_FIELDS:
        raise LiveArtifactValidationError("live validation fields mismatch")
    if value["semantic_version"] != LIVE_ARTIFACT_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live validation version mismatch")
    report_value = value["report"]
    report = None
    if report_value is not None:
        if not isinstance(report_value, dict) or set(report_value) != _REPORT_FIELDS:
            raise LiveArtifactValidationError("live validation report fields mismatch")
        codes = report_value["failure_codes"]
        failures = report_value["failures"]
        metrics = report_value["metrics"]
        if (
            type(report_value["passed"]) is not bool
            or not isinstance(codes, list)
            or not all(isinstance(item, str) and item for item in codes)
            or not isinstance(failures, list)
            or not all(isinstance(item, str) and item for item in failures)
            or not isinstance(metrics, dict)
        ):
            raise LiveArtifactValidationError("live validation report is malformed")
        report = ValidationReport(
            report_value["passed"], tuple(codes), tuple(failures), freeze_value(metrics)
        )
    links = {
        "manifest_sha256": require_optional_live_sha256(
            value["manifest_sha256"], "validation manifest"
        ),
        "clean_trace_sha256": require_optional_live_sha256(
            value["clean_trace_sha256"], "validation clean trace"
        ),
        "delivered_trace_sha256": require_optional_live_sha256(
            value["delivered_trace_sha256"], "validation delivered trace"
        ),
    }
    if (report is None) != (links["manifest_sha256"] is None):
        raise LiveArtifactValidationError("live validation report/link mismatch")
    return report, links


def rest_references_to_live_dict(
    references: RestReferenceBundle | None, arrays: LiveArrayWriter
) -> object:
    if references is None:
        return None
    return {
        "reference_id": references.reference_id,
        "dataset_split": references.dataset_split.value,
        "split_manifest_sha256": references.split_manifest_sha256,
        "source_artifact_sha256": references.source_artifact_sha256,
        "no_contact_predicate_id": references.no_contact_predicate_id,
        "no_contact_validation_sha256": references.no_contact_validation_sha256,
        "no_contact_verified": references.no_contact_verified,
        "qualified_record_ids": dict(references.qualified_record_ids),
        "calibration_sha256": dict(references.calibration_sha256),
        "payloads": {
            slot: arrays.add(references.payload_for(slot)) for slot in SENSOR_SLOTS
        },
    }


def rest_references_from_live_dict(
    value: object,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> RestReferenceBundle | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _REST_FIELDS:
        raise LiveArtifactValidationError("live rest-reference fields mismatch")
    payloads = value["payloads"]
    qualified = value["qualified_record_ids"]
    calibration = value["calibration_sha256"]
    if (
        not isinstance(payloads, dict)
        or set(payloads) != set(SENSOR_SLOTS)
        or not isinstance(qualified, dict)
        or not isinstance(calibration, dict)
    ):
        raise LiveArtifactValidationError("live rest-reference mappings are malformed")
    return RestReferenceBundle(
        reference_id=_string(value["reference_id"], "reference id"),
        dataset_split=ReferenceSplit(_string(value["dataset_split"], "dataset split")),
        split_manifest_sha256=require_live_sha256(
            value["split_manifest_sha256"], "split manifest"
        ),
        source_artifact_sha256=require_live_sha256(
            value["source_artifact_sha256"], "source artifact"
        ),
        no_contact_predicate_id=_string(
            value["no_contact_predicate_id"], "no-contact predicate"
        ),
        no_contact_validation_sha256=require_live_sha256(
            value["no_contact_validation_sha256"], "no-contact validation"
        ),
        no_contact_verified=value["no_contact_verified"],
        qualified_record_ids=qualified,
        calibration_sha256=calibration,
        payloads={
            slot: load_live_array(payloads[slot], snapshot, referenced_paths)
            for slot in SENSOR_SLOTS
        },
    )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LiveArtifactValidationError(f"{name} must be a non-empty string")
    return value

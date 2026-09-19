"""Path-free run identity, validation, and rest-reference artifact values."""

from __future__ import annotations

from collections.abc import Mapping

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    build_univtac_backend_config,
)
from robotactile_benchmark.closed_loop.contracts import (
    ClosedLoopRunSpec,
    InitialStatePolicy,
    PolicyIdentity,
    WallTimeoutRole,
)
from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.constants import SENSOR_SLOTS
from robotactile_benchmark.contracts import canonical_hash, freeze_value, thaw_value
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    N0ObservedTactileMode,
    effective_univtac_control_hz,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_ARTIFACT_SEMANTIC_VERSION,
    LIVE_REQUEST_IDENTITY_SEMANTIC_VERSION,
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
from robotactile_benchmark.policies.n0_vtla_execution import n0_vtla_execution_steps
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode,
    normalize_zero_shape,
    validate_availability_config,
)
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
_ROBUST_REQUEST_FIELDS = _REQUEST_FIELDS | frozenset({"initial_state_policy"})
_WATCHDOG_REQUEST_FIELDS = _REQUEST_FIELDS | frozenset({"wall_timeout_role"})
_ROBUST_WATCHDOG_REQUEST_FIELDS = _ROBUST_REQUEST_FIELDS | frozenset(
    {"wall_timeout_role"}
)
_TACTILE_MODE_REQUEST_FIELDS = tuple(
    fields | frozenset({"n0_observed_tactile_mode"})
    for fields in (
        _REQUEST_FIELDS,
        _ROBUST_REQUEST_FIELDS,
        _WATCHDOG_REQUEST_FIELDS,
        _ROBUST_WATCHDOG_REQUEST_FIELDS,
    )
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
    identity: dict[str, object] = {
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
        "semantic_version": LIVE_REQUEST_IDENTITY_SEMANTIC_VERSION,
    }
    if request.initial_state_policy is not InitialStatePolicy.OFFICIAL_REPRODUCTION:
        identity["initial_state_policy"] = request.initial_state_policy.value
    if request.wall_timeout_role is not WallTimeoutRole.SCORING_BOUNDARY_V1:
        identity["wall_timeout_role"] = request.wall_timeout_role.value
    if request.n0_observed_tactile_mode is not N0ObservedTactileMode.REQUIRED:
        identity["n0_observed_tactile_mode"] = request.n0_observed_tactile_mode.value
    if request.n0_action_per_frame != 12 or request.n0_prompt_override is not None:
        identity["n0_action_per_frame"] = request.n0_action_per_frame
        identity["n0_prompt_override"] = request.n0_prompt_override
    if request.retrained_prompt is not None:
        identity.update(
            retrained_prompt=request.retrained_prompt,
            retrained_control_hz=request.retrained_control_hz,
            retrained_tactile_payload=request.retrained_tactile_payload,
        )
    if request.tactile_availability_mode is not TactileAvailabilityMode.REQUIRED:
        identity["tactile_availability_mode"] = request.tactile_availability_mode.value
    if request.tactile_zero_shape is not None:
        identity["tactile_zero_shape"] = list(request.tactile_zero_shape)
    if request.n0_vtla_execution_profile is not None:
        identity["n0_vtla_execution_profile"] = request.n0_vtla_execution_profile
    return identity


def validate_live_request_identity(
    value: object,
    trial: TrialManifest,
    run_spec: ClosedLoopRunSpec,
    fault_sha256: str | None,
    rest_sha256: str | None,
) -> dict[str, object]:
    """Strictly bind a stored path-free request back to packaged UniVTAC."""

    retrained_fields = {"n0_action_per_frame", "n0_prompt_override"}
    external_fields = {
        "retrained_prompt",
        "retrained_control_hz",
        "retrained_tactile_payload",
    }
    availability_fields = {"tactile_availability_mode", "tactile_zero_shape"}
    if not isinstance(value, dict) or set(
        value
    ) - retrained_fields - external_fields - availability_fields - {
        "n0_vtla_execution_profile"
    } not in {
        _REQUEST_FIELDS,
        _ROBUST_REQUEST_FIELDS,
        _WATCHDOG_REQUEST_FIELDS,
        _ROBUST_WATCHDOG_REQUEST_FIELDS,
        *_TACTILE_MODE_REQUEST_FIELDS,
    }:
        raise LiveArtifactValidationError("live request identity fields mismatch")
    execution_profile = value.get("n0_vtla_execution_profile")
    if "n0_vtla_execution_profile" in value:
        try:
            if (
                not isinstance(execution_profile, str)
                or value["policy_kind"] != "n0_vtla"
                or value.get("retrained_control_hz") != 10
                or not external_fields <= set(value)
            ):
                raise ValueError("invalid N0-VTLA execution profile scope")
            n0_vtla_execution_steps(execution_profile, value["task_id"])
        except ValueError as error:
            raise LiveArtifactValidationError(
                "live N0-VTLA execution profile is invalid"
            ) from error
    try:
        availability = TactileAvailabilityMode(
            value.get("tactile_availability_mode", "required")
        )
        zero_shape = normalize_zero_shape(value.get("tactile_zero_shape"))
        validate_availability_config(availability, zero_shape, value["policy_kind"])
        if availability is not TactileAvailabilityMode.REQUIRED and (
            trial.condition is Condition.NO_TOUCH or "n0_observed_tactile_mode" in value
        ):
            raise ValueError("availability protocols cannot mix with no-touch modes")
    except (ValueError, TypeError) as error:
        raise LiveArtifactValidationError(
            "live tactile availability config is invalid"
        ) from error
    if set(value) & external_fields:
        if not external_fields <= set(value) or value["policy_kind"] not in {
            "n0_vtla",
            "ftp1_policy",
            "dream_tac",
        }:
            raise LiveArtifactValidationError("live retrained policy fields mismatch")
        if (
            value["retrained_prompt"] != run_spec.prompt
            or not isinstance(value["retrained_prompt"], str)
            or not value["retrained_prompt"].strip()
        ):
            raise LiveArtifactValidationError("live retrained policy prompt mismatch")
        if type(value["retrained_control_hz"]) is not int or value[
            "retrained_control_hz"
        ] not in (10, 60):
            raise LiveArtifactValidationError("live retrained policy Hz mismatch")
        expected_steps = (
            n0_vtla_execution_steps(execution_profile, value["task_id"])
            if value["policy_kind"] == "n0_vtla"
            else {"ftp1_policy": 1, "dream_tac": 20}[value["policy_kind"]]
        )
        if value["execute_action_steps"] != expected_steps:
            raise LiveArtifactValidationError("live retrained action horizon mismatch")
    action_per_frame = value.get("n0_action_per_frame", 12)
    if type(action_per_frame) is not int or action_per_frame not in (4, 12):
        raise LiveArtifactValidationError("live N0 action_per_frame is invalid")
    if set(value) & retrained_fields:
        if not retrained_fields <= set(value) or value["policy_kind"] != "n0":
            raise LiveArtifactValidationError("live retrained N0 fields mismatch")
        prompt = value["n0_prompt_override"]
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or prompt != run_spec.prompt
        ):
            raise LiveArtifactValidationError("live retrained N0 prompt mismatch")
        if value["execute_action_steps"] != 2 * action_per_frame:
            raise LiveArtifactValidationError("live retrained N0 action chunk mismatch")
    stored_initial_state_policy = value.get(
        "initial_state_policy", InitialStatePolicy.OFFICIAL_REPRODUCTION.value
    )
    explicit_initial_state_policies = {
        InitialStatePolicy.REPLACE_INITIAL_TERMINAL_V1.value,
        InitialStatePolicy.DIAGNOSTIC_ALLOW_INVALID_V1.value,
    }
    if (
        "initial_state_policy" in value
        and stored_initial_state_policy not in explicit_initial_state_policies
    ):
        raise LiveArtifactValidationError("live initial-state policy is invalid")
    stored_wall_timeout_role = value.get(
        "wall_timeout_role", WallTimeoutRole.SCORING_BOUNDARY_V1.value
    )
    if "wall_timeout_role" in value and stored_wall_timeout_role != (
        WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1.value
    ):
        raise LiveArtifactValidationError("live wall-timeout role is invalid")
    try:
        tactile_mode = N0ObservedTactileMode(
            value.get(
                "n0_observed_tactile_mode",
                N0ObservedTactileMode.REQUIRED.value,
            )
        )
    except (TypeError, ValueError) as error:
        raise LiveArtifactValidationError(
            "live N0 observed-tactile mode is invalid"
        ) from error
    if (
        tactile_mode is N0ObservedTactileMode.ABSENT
        and trial.condition is not Condition.FAULTED
    ):
        raise LiveArtifactValidationError(
            "observed tactile absence requires a faulted trial"
        )
    if value["semantic_version"] != LIVE_REQUEST_IDENTITY_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live request identity version mismatch")
    source = value["source_binding"]
    if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
        raise LiveArtifactValidationError("live source binding fields mismatch")
    config = build_univtac_backend_config(
        _string(value["task_id"], "task id"),
        action_spec=trial.action_spec,
        n0_action_execution_contract=(
            N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT
            if action_per_frame == 4 or value["policy_kind"] == "dream_tac"
            else None
        ),
        control_hz=effective_univtac_control_hz(
            LivePolicyKind(value["policy_kind"]),
            value.get("retrained_control_hz"),
        ),
        tactile_payload=value.get("retrained_tactile_payload"),
    )
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
        supports_structural_absence=(
            trial.condition is Condition.NO_TOUCH
            or tactile_mode is N0ObservedTactileMode.ABSENT
            or availability is not TactileAvailabilityMode.REQUIRED
        ),
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
        stored_wall_timeout_role == run_spec.wall_timeout_role.value,
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
    content_identity = {
        "backend_config_sha256": value["backend_config_sha256"],
        "policy_identity_sha256": value["policy_identity_sha256"],
        "trial_manifest_sha256": value["trial_manifest_sha256"],
        "run_spec_sha256": value["run_spec_sha256"],
        "fault_manifest_sha256": value["fault_manifest_sha256"],
        "rest_references_sha256": value["rest_references_sha256"],
        "policy_kind": value["policy_kind"],
    }
    if "initial_state_policy" in value:
        content_identity["initial_state_policy"] = value["initial_state_policy"]
    if "wall_timeout_role" in value:
        content_identity["wall_timeout_role"] = value["wall_timeout_role"]
    if "n0_observed_tactile_mode" in value:
        content_identity["n0_observed_tactile_mode"] = value["n0_observed_tactile_mode"]
    for field in (
        "tactile_availability_mode",
        "tactile_zero_shape",
        "n0_vtla_execution_profile",
    ):
        if field in value:
            content_identity[field] = value[field]
    return canonical_hash(content_identity)


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

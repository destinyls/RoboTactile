"""Strict request and referenced-contract loading for live UniVTAC runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, cast

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC, QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACBackendConfig,
    build_univtac_backend_config,
)
from robotactile_benchmark.closed_loop.contracts import (
    ClosedLoopRunSpec,
    InitialStatePolicy,
    PolicyIdentity,
    WallTimeoutRole,
)
from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS, SENSOR_SLOTS
from robotactile_benchmark.contracts import Array, canonical_hash
from robotactile_benchmark.execution.contracts import (
    LiveUniVTACRunRequest,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.policies.act_loading import ArtifactUnavailableError
from robotactile_benchmark.rest_references import (
    FrozenPayload,
    ReferenceSplit,
    RestReferenceBundle,
)
from robotactile_benchmark.trials import (
    Condition,
    TrialManifest,
    system_manifest_hash,
)

_REQUEST_FIELDS = frozenset(
    {
        "task_id",
        "condition",
        "policy_kind",
        "base_system_id",
        "dataset_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "base_system_manifest_sha256",
        "initial_seed",
        "exogenous_seed",
        "max_control_cycles",
        "max_observation_steps",
        "execute_action_steps",
        "wall_timeout_s",
        "upstream_root",
        "runtime_dir",
        "output_dir",
        "fault_manifest_path",
        "rest_references_path",
        "restoration_index",
        "restoration_mode",
        "matched_no_touch_system_id",
        "matched_no_touch_artifact_path",
        "act_device_name",
        "simulator_device",
        "launcher_args",
        "n0_source_commit",
        "n0_normalizer_sha256",
        "n0_serve_bundle_sha256",
        "n0_prompt_manifest_sha256",
        "semantic_version",
    }
)
_ROBUST_REQUEST_FIELDS = _REQUEST_FIELDS | frozenset({"initial_state_policy"})
_WATCHDOG_REQUEST_FIELDS = _REQUEST_FIELDS | frozenset({"wall_timeout_role"})
_ROBUST_WATCHDOG_REQUEST_FIELDS = _ROBUST_REQUEST_FIELDS | frozenset(
    {"wall_timeout_role"}
)
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


def _pairs(pairs: Sequence[Tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_object(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return cast(Mapping[str, Any], value)


def _request_path(value: Any, name: str, root: Path) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty path string or null")
    path = Path(value)
    return (path if path.is_absolute() else root / path).absolute()


def load_live_univtac_request(path: Path) -> LiveUniVTACRunRequest:
    """Load one exact-field JSON request with paths relative to its own file."""

    if not isinstance(path, Path):
        raise TypeError("live request path must be a pathlib.Path")
    request_path = path.absolute()
    document = _read_object(request_path, "live UniVTAC request")
    fields = set(document)
    if fields not in {
        _REQUEST_FIELDS,
        _ROBUST_REQUEST_FIELDS,
        _WATCHDOG_REQUEST_FIELDS,
        _ROBUST_WATCHDOG_REQUEST_FIELDS,
    }:
        missing = sorted(_REQUEST_FIELDS - fields)
        extra = sorted(fields - _ROBUST_WATCHDOG_REQUEST_FIELDS)
        raise ValueError(
            f"live request fields mismatch: missing={missing}, extra={extra}"
        )
    initial_state_policy = document.get(
        "initial_state_policy", InitialStatePolicy.OFFICIAL_REPRODUCTION.value
    )
    explicit_initial_state_policies = {
        InitialStatePolicy.REPLACE_INITIAL_TERMINAL_V1.value,
        InitialStatePolicy.DIAGNOSTIC_ALLOW_INVALID_V1.value,
    }
    if (
        "initial_state_policy" in document
        and initial_state_policy not in explicit_initial_state_policies
    ):
        raise ValueError("explicit initial_state_policy is invalid")
    wall_timeout_role = document.get(
        "wall_timeout_role", WallTimeoutRole.SCORING_BOUNDARY_V1.value
    )
    if "wall_timeout_role" in document and wall_timeout_role != (
        WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1.value
    ):
        raise ValueError(
            "explicit wall_timeout_role must select infrastructure watchdog"
        )
    base = request_path.parent
    launcher_args = document["launcher_args"]
    if not isinstance(launcher_args, Mapping):
        raise TypeError("launcher_args must be a JSON object")
    return LiveUniVTACRunRequest(
        task_id=document["task_id"],
        condition=document["condition"],
        policy_kind=document["policy_kind"],
        base_system_id=document["base_system_id"],
        dataset_sha256=document["dataset_sha256"],
        checkpoint_sha256=document["checkpoint_sha256"],
        config_sha256=document["config_sha256"],
        base_system_manifest_sha256=document["base_system_manifest_sha256"],
        initial_seed=document["initial_seed"],
        exogenous_seed=document["exogenous_seed"],
        max_control_cycles=document["max_control_cycles"],
        max_observation_steps=document["max_observation_steps"],
        execute_action_steps=document["execute_action_steps"],
        wall_timeout_s=document["wall_timeout_s"],
        upstream_root=_required_request_path(
            document["upstream_root"], "upstream_root", base
        ),
        runtime_dir=_required_request_path(
            document["runtime_dir"], "runtime_dir", base
        ),
        output_dir=_request_path(document["output_dir"], "output_dir", base),
        fault_manifest_path=_request_path(
            document["fault_manifest_path"], "fault_manifest_path", base
        ),
        rest_references_path=_request_path(
            document["rest_references_path"], "rest_references_path", base
        ),
        restoration_index=document["restoration_index"],
        restoration_mode=document["restoration_mode"],
        matched_no_touch_system_id=document["matched_no_touch_system_id"],
        matched_no_touch_artifact_path=_request_path(
            document["matched_no_touch_artifact_path"],
            "matched_no_touch_artifact_path",
            base,
        ),
        act_device_name=document["act_device_name"],
        simulator_device=document["simulator_device"],
        launcher_args=dict(launcher_args),
        n0_source_commit=document["n0_source_commit"],
        n0_normalizer_sha256=document["n0_normalizer_sha256"],
        n0_serve_bundle_sha256=document["n0_serve_bundle_sha256"],
        n0_prompt_manifest_sha256=document["n0_prompt_manifest_sha256"],
        initial_state_policy=initial_state_policy,
        wall_timeout_role=wall_timeout_role,
        semantic_version=document["semantic_version"],
    )


def _required_request_path(value: Any, name: str, root: Path) -> Path:
    path = _request_path(value, name, root)
    if path is None:
        raise TypeError(f"{name} cannot be null")
    return path


def _load_fault(path: Optional[Path]) -> Optional[FaultManifest]:
    if path is None:
        return None
    return FaultManifest.from_dict(_read_object(path, "fault manifest"))


def _string_mapping(value: Any, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise TypeError(f"{name} must be a string-to-string mapping")
    return cast(Mapping[str, str], value)


def _payload(value: Any, name: str) -> Array:
    if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data_hex"}:
        raise ValueError(f"{name} frozen payload fields mismatch")
    shape = value["shape"]
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
    ):
        raise TypeError(f"{name} shape must be an integer sequence")
    frozen = FrozenPayload(
        dtype=value["dtype"],
        shape=tuple(shape),
        data_hex=value["data_hex"],
    )
    return cast(Array, frozen.to_array())


def _load_rest_references(path: Optional[Path]) -> Optional[RestReferenceBundle]:
    if path is None:
        return None
    document = _read_object(path, "rest reference bundle")
    if set(document) != _REST_FIELDS:
        raise ValueError("rest reference bundle fields mismatch")
    payloads = document["payloads"]
    if not isinstance(payloads, Mapping) or set(payloads) != set(SENSOR_SLOTS):
        raise ValueError("rest reference payloads must cover left and right")
    return RestReferenceBundle(
        reference_id=document["reference_id"],
        dataset_split=ReferenceSplit(document["dataset_split"]),
        split_manifest_sha256=document["split_manifest_sha256"],
        source_artifact_sha256=document["source_artifact_sha256"],
        no_contact_predicate_id=document["no_contact_predicate_id"],
        no_contact_validation_sha256=document["no_contact_validation_sha256"],
        no_contact_verified=document["no_contact_verified"],
        qualified_record_ids=_string_mapping(
            document["qualified_record_ids"], "qualified_record_ids"
        ),
        calibration_sha256=_string_mapping(
            document["calibration_sha256"], "calibration_sha256"
        ),
        payloads={slot: _payload(payloads[slot], slot) for slot in SENSOR_SLOTS},
    )


@dataclass(frozen=True)
class LoadedLiveUniVTACRun:
    """Path-free content contracts consumed by runtime construction."""

    request: LiveUniVTACRunRequest
    backend_config: UniVTACBackendConfig
    policy_identity: PolicyIdentity
    trial: TrialManifest
    run_spec: ClosedLoopRunSpec
    fault_manifest: Optional[FaultManifest]
    rest_references: Optional[RestReferenceBundle]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        content_identity: dict[str, object] = {
            "backend_config_sha256": self.backend_config.sha256,
            "policy_identity_sha256": self.policy_identity.sha256,
            "trial_manifest_sha256": self.trial.sha256,
            "run_spec_sha256": self.run_spec.sha256,
            "fault_manifest_sha256": (
                None if self.fault_manifest is None else self.fault_manifest.sha256
            ),
            "rest_references_sha256": (
                None if self.rest_references is None else self.rest_references.sha256
            ),
            "policy_kind": self.request.policy_kind.value,
        }
        if (
            self.request.initial_state_policy
            is not InitialStatePolicy.OFFICIAL_REPRODUCTION
        ):
            content_identity["initial_state_policy"] = (
                self.request.initial_state_policy.value
            )
        if self.request.wall_timeout_role is not WallTimeoutRole.SCORING_BOUNDARY_V1:
            content_identity["wall_timeout_role"] = self.request.wall_timeout_role.value
        object.__setattr__(
            self,
            "content_sha256",
            canonical_hash(content_identity),
        )


def load_live_univtac_run(request: LiveUniVTACRunRequest) -> LoadedLiveUniVTACRun:
    """Materialize frozen config/manifests without starting model or simulator."""

    if not isinstance(request, LiveUniVTACRunRequest):
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    if request.condition is Condition.NO_TOUCH and (
        request.matched_no_touch_system_id is None
        or request.matched_no_touch_artifact_path is None
    ):
        raise ArtifactUnavailableError(
            "artifact_unavailable: matched no-touch identity/artifact is missing"
        )
    action_spec = (
        EE8_ACTION_SPEC if request.policy_kind.value == "n0" else QPOS8_ACTION_SPEC
    )
    config = build_univtac_backend_config(request.task_id, action_spec=action_spec)
    if request.max_observation_steps > config.task.action_horizon + 1:
        raise ValueError("max_observation_steps exceeds the frozen task horizon")
    fault = _load_fault(request.fault_manifest_path)
    references = _load_rest_references(request.rest_references_path)
    _validate_fault_links(request, fault, references)
    executed_system_id = (
        request.matched_no_touch_system_id
        if request.condition is Condition.NO_TOUCH
        else request.base_system_id
    )
    if executed_system_id is None:
        raise ArtifactUnavailableError(
            "artifact_unavailable: matched no-touch identity missing"
        )
    identity = PolicyIdentity(
        system_id=executed_system_id,
        checkpoint_sha256=request.checkpoint_sha256,
        config_sha256=request.config_sha256,
        action_spec=action_spec,
        consumes_tactile=request.condition is not Condition.NO_TOUCH,
        supports_structural_absence=request.condition is Condition.NO_TOUCH,
    )
    expected_base = system_manifest_hash(
        request.base_system_id,
        request.checkpoint_sha256,
        request.config_sha256,
        action_spec,
    )
    base_manifest = request.base_system_manifest_sha256 or expected_base
    if request.condition is not Condition.NO_TOUCH and base_manifest != expected_base:
        raise ValueError("base system manifest content identity mismatch")
    trial = TrialManifest(
        task=config.task.task_id,
        initial_seed=request.initial_seed,
        exogenous_seed=request.exogenous_seed,
        condition=request.condition,
        base_system_id=request.base_system_id,
        executed_system_id=executed_system_id,
        dataset_sha256=request.dataset_sha256,
        base_system_manifest_sha256=base_manifest,
        checkpoint_sha256=request.checkpoint_sha256,
        config_sha256=request.config_sha256,
        action_spec=action_spec,
        fault_manifest_sha256=None if fault is None else fault.sha256,
        matched_no_touch_system_id=(
            request.matched_no_touch_system_id
            if request.condition is Condition.NO_TOUCH
            else None
        ),
        restoration_index=request.restoration_index,
        restoration_mode=request.restoration_mode,
    )
    prompt = config.task.prompt
    if request.policy_kind.value == "n0":
        from robotactile_benchmark.policies.n0_official import n0_training_prompt

        prompt = n0_training_prompt(request.task_id)
    run_spec = ClosedLoopRunSpec(
        prompt=prompt,
        success_predicate_id=config.task.success_predicate_id,
        max_control_cycles=request.max_control_cycles,
        max_observation_steps=request.max_observation_steps,
        execute_action_steps=request.execute_action_steps,
        wall_timeout_s=request.wall_timeout_s,
        wall_timeout_role=request.wall_timeout_role,
    )
    return LoadedLiveUniVTACRun(
        request=request,
        backend_config=config,
        policy_identity=identity,
        trial=trial,
        run_spec=run_spec,
        fault_manifest=fault,
        rest_references=references,
    )


def _validate_fault_links(
    request: LiveUniVTACRunRequest,
    fault: Optional[FaultManifest],
    references: Optional[RestReferenceBundle],
) -> None:
    faulted = request.condition in {Condition.FAULTED, Condition.RESTORED}
    if faulted != (fault is not None):
        raise ValueError("condition and loaded fault manifest disagree")
    if fault is None:
        if references is not None:
            raise ValueError("rest references require a loaded fault manifest")
        return
    requires_reference = fault.operator_id in REST_REFERENCE_OPERATOR_IDS
    if requires_reference != (references is not None):
        raise ValueError("fault rest-reference requirement mismatch")
    if (
        references is not None
        and fault.parameters.get("rest_reference_sha256") != references.sha256
    ):
        raise ValueError("fault and rest-reference content identities disagree")
    if fault.start_index >= request.max_observation_steps:
        raise ValueError("fault window is outside the requested observation budget")
    if request.condition is Condition.RESTORED and (
        request.restoration_index != fault.stop_index
    ):
        raise ValueError("restoration index must equal the fault stop index")

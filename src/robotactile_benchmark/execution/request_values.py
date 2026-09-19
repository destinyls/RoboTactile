"""Canonical value representation for live UniVTAC requests."""

from __future__ import annotations

from robotactile_benchmark.backends.univtac_success_profiles import (
    UniVTACSuccessProfile,
)
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.execution.contracts import (
    LiveUniVTACRunRequest,
    N0ObservedTactileMode,
)
from robotactile_benchmark.policies.tactile_availability import TactileAvailabilityMode


def live_univtac_request_to_dict(
    request: LiveUniVTACRunRequest,
) -> dict[str, object]:
    """Serialize one typed request using the strict loader's exact field set."""

    if type(request) is not LiveUniVTACRunRequest:
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    document: dict[str, object] = {
        "task_id": request.task_id,
        "condition": request.condition.value,
        "policy_kind": request.policy_kind.value,
        "base_system_id": request.base_system_id,
        "dataset_sha256": request.dataset_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "config_sha256": request.config_sha256,
        "base_system_manifest_sha256": request.base_system_manifest_sha256,
        "initial_seed": request.initial_seed,
        "exogenous_seed": request.exogenous_seed,
        "max_control_cycles": request.max_control_cycles,
        "max_observation_steps": request.max_observation_steps,
        "execute_action_steps": request.execute_action_steps,
        "wall_timeout_s": request.wall_timeout_s,
        "upstream_root": str(request.upstream_root),
        "runtime_dir": str(request.runtime_dir),
        "output_dir": None if request.output_dir is None else str(request.output_dir),
        "fault_manifest_path": (
            None
            if request.fault_manifest_path is None
            else str(request.fault_manifest_path)
        ),
        "rest_references_path": (
            None
            if request.rest_references_path is None
            else str(request.rest_references_path)
        ),
        "matched_no_touch_system_id": request.matched_no_touch_system_id,
        "matched_no_touch_artifact_path": (
            None
            if request.matched_no_touch_artifact_path is None
            else str(request.matched_no_touch_artifact_path)
        ),
        "act_device_name": request.act_device_name,
        "simulator_device": request.simulator_device,
        "launcher_args": dict(request.launcher_args),
        "n0_source_commit": request.n0_source_commit,
        "n0_normalizer_sha256": request.n0_normalizer_sha256,
        "n0_serve_bundle_sha256": request.n0_serve_bundle_sha256,
        "n0_prompt_manifest_sha256": request.n0_prompt_manifest_sha256,
        "semantic_version": request.semantic_version,
    }
    if request.initial_state_policy is not InitialStatePolicy.OFFICIAL_REPRODUCTION:
        document["initial_state_policy"] = request.initial_state_policy.value
    if request.wall_timeout_role is not WallTimeoutRole.SCORING_BOUNDARY_V1:
        document["wall_timeout_role"] = request.wall_timeout_role.value
    if request.n0_observed_tactile_mode is not N0ObservedTactileMode.REQUIRED:
        document["n0_observed_tactile_mode"] = request.n0_observed_tactile_mode.value
    if request.success_profile_id is not UniVTACSuccessProfile.OFFICIAL_V1:
        document["success_profile_id"] = request.success_profile_id.value
    if request.n0_action_per_frame != 12 or request.n0_prompt_override is not None:
        document["n0_action_per_frame"] = request.n0_action_per_frame
        document["n0_prompt_override"] = request.n0_prompt_override
    if request.retrained_prompt is not None:
        document.update(
            retrained_prompt=request.retrained_prompt,
            retrained_control_hz=request.retrained_control_hz,
            retrained_tactile_payload=request.retrained_tactile_payload,
        )
    if request.tactile_availability_mode is not TactileAvailabilityMode.REQUIRED:
        document["tactile_availability_mode"] = request.tactile_availability_mode.value
    if request.tactile_zero_shape is not None:
        document["tactile_zero_shape"] = list(request.tactile_zero_shape)
    if request.n0_vtla_execution_profile is not None:
        document["n0_vtla_execution_profile"] = request.n0_vtla_execution_profile
    return document

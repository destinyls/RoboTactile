"""Canonical value representation for live UniVTAC requests."""

from __future__ import annotations

from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest


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
        "restoration_index": request.restoration_index,
        "restoration_mode": (
            None if request.restoration_mode is None else request.restoration_mode.value
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
    return document

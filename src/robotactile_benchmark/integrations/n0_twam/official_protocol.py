"""Fail-closed public N0-TWAM/UniVTAC evaluation contracts."""

from __future__ import annotations

from robotactile_benchmark.backends.univtac_success_profiles import (
    UniVTACSuccessProfile,
)
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.execution.contracts import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    N0ObservedTactileMode,
)
from robotactile_benchmark.policies.tactile_availability import (
    TactileAvailabilityMode,
)
from robotactile_benchmark.trials import Condition

OFFICIAL_N0_ACTION_PER_FRAME = 12
OFFICIAL_N0_EXECUTE_ACTION_STEPS = 2 * OFFICIAL_N0_ACTION_PER_FRAME


def validate_official_n0_release_request(
    request: LiveUniVTACRunRequest,
) -> None:
    """Reject retrained-client options on the released N0-TWAM integration."""

    if type(request) is not LiveUniVTACRunRequest:
        raise TypeError("request must be an exact LiveUniVTACRunRequest")
    if request.policy_kind is not LivePolicyKind.N0:
        raise ValueError("official N0 execution requires policy_kind=n0")
    if request.n0_action_per_frame != OFFICIAL_N0_ACTION_PER_FRAME:
        raise ValueError(
            "official N0 release requires n0_action_per_frame="
            f"{OFFICIAL_N0_ACTION_PER_FRAME}"
        )
    if request.execute_action_steps != OFFICIAL_N0_EXECUTE_ACTION_STEPS:
        raise ValueError(
            "official N0 release requires execute_action_steps="
            f"{OFFICIAL_N0_EXECUTE_ACTION_STEPS}"
        )
    if request.n0_prompt_override is not None:
        raise ValueError("official N0 release forbids n0_prompt_override")
    if any(
        value is not None
        for value in (
            request.retrained_prompt,
            request.retrained_control_hz,
            request.retrained_tactile_payload,
            request.n0_vtla_execution_profile,
        )
    ):
        raise ValueError("official N0 release forbids retrained policy options")


def validate_official_n0_clean_claim_request(
    request: LiveUniVTACRunRequest,
    *,
    action_horizon: int,
) -> None:
    """Require the public-source-aligned Clean profile before a formal run."""

    validate_official_n0_release_request(request)
    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int):
        raise TypeError("action_horizon must be an integer")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    if request.condition is not Condition.CLEAN:
        raise ValueError("official Clean claim requires condition=clean")
    if request.success_profile_id is not UniVTACSuccessProfile.OFFICIAL_V1:
        raise ValueError("official Clean claim requires official_v1 success")
    if request.initial_state_policy is not InitialStatePolicy.OFFICIAL_REPRODUCTION:
        raise ValueError("official Clean claim requires official reset reproduction")
    if request.wall_timeout_role is not WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1:
        raise ValueError(
            "official Clean claim requires a non-scoring infrastructure watchdog"
        )
    if request.initial_seed != request.exogenous_seed:
        raise ValueError("official Clean claim requires one shared task/policy seed")
    if request.max_control_cycles != action_horizon:
        raise ValueError("official Clean claim requires the task action horizon")
    if request.max_observation_steps != action_horizon + 1:
        raise ValueError(
            "official Clean claim requires action_horizon + 1 observations"
        )
    if request.n0_observed_tactile_mode is not N0ObservedTactileMode.REQUIRED:
        raise ValueError("official Clean claim requires both observed tactile streams")
    if request.tactile_availability_mode is not TactileAvailabilityMode.REQUIRED:
        raise ValueError("official Clean claim requires tactile availability")


__all__ = [
    "OFFICIAL_N0_ACTION_PER_FRAME",
    "OFFICIAL_N0_EXECUTE_ACTION_STEPS",
    "validate_official_n0_clean_claim_request",
    "validate_official_n0_release_request",
]

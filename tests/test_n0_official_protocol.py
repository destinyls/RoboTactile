"""Fail-closed checks for the public N0-TWAM/UniVTAC profile."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from robotactile_benchmark.backends.univtac_contracts import (
    N0_PHYSICS_STEPS_PER_ACTION,
    SIM_HZ,
)
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.integrations.n0_twam.official_protocol import (
    OFFICIAL_N0_ACTION_PER_FRAME,
    OFFICIAL_N0_EXECUTE_ACTION_STEPS,
    validate_official_n0_clean_claim_request,
    validate_official_n0_release_request,
)

ROOT = Path(__file__).resolve().parents[1]


def _released_request():
    return load_live_univtac_request(ROOT / "examples/n0_twam/request.json")


def test_released_request_freezes_public_chunk_and_training_cadence() -> None:
    request = _released_request()

    validate_official_n0_release_request(request)

    assert OFFICIAL_N0_ACTION_PER_FRAME == 12
    assert OFFICIAL_N0_EXECUTE_ACTION_STEPS == 24
    assert SIM_HZ == 120
    assert SIM_HZ / N0_PHYSICS_STEPS_PER_ACTION == 60


def test_released_request_rejects_retrained_chunk_or_prompt() -> None:
    request = _released_request()
    retrained = replace(
        request,
        n0_action_per_frame=4,
        n0_prompt_override="retrained task prompt",
        execute_action_steps=8,
    )

    with pytest.raises(ValueError, match="n0_action_per_frame=12"):
        validate_official_n0_release_request(retrained)
    with pytest.raises(ValueError, match="forbids n0_prompt_override"):
        validate_official_n0_release_request(
            replace(request, n0_prompt_override="another prompt")
        )


def test_formal_clean_profile_rejects_nonofficial_reset_seed_or_timeout() -> None:
    base = _released_request()
    request = replace(
        base,
        exogenous_seed=base.initial_seed,
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
    )
    validate_official_n0_clean_claim_request(request, action_horizon=300)

    with pytest.raises(ValueError, match="official reset"):
        validate_official_n0_clean_claim_request(
            replace(
                request,
                initial_state_policy=InitialStatePolicy.REPLACE_INITIAL_TERMINAL_V1,
            ),
            action_horizon=300,
        )
    with pytest.raises(ValueError, match="shared task/policy seed"):
        validate_official_n0_clean_claim_request(
            replace(request, exogenous_seed=request.initial_seed + 1),
            action_horizon=300,
        )
    with pytest.raises(ValueError, match="infrastructure watchdog"):
        validate_official_n0_clean_claim_request(
            replace(request, wall_timeout_role=WallTimeoutRole.SCORING_BOUNDARY_V1),
            action_horizon=300,
        )


def test_formal_clean_profile_rejects_truncated_horizon() -> None:
    base = _released_request()
    request = replace(
        base,
        exogenous_seed=base.initial_seed,
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
        max_control_cycles=299,
        max_observation_steps=300,
    )

    with pytest.raises(ValueError, match="task action horizon"):
        validate_official_n0_clean_claim_request(request, action_horizon=300)

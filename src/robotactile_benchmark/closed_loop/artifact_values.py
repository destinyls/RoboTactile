"""Strict JSON views for run specifications and terminal result receipts."""

from __future__ import annotations

from typing import Any, Optional

from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
    require_sha256,
)
from robotactile_benchmark.closed_loop.contracts import (
    ClosedLoopRunSpec,
    WallTimeoutRole,
)
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.trials import TerminalStatus

_LEGACY_RUN_SPEC_FIELDS = frozenset(
    {
        "prompt",
        "success_predicate_id",
        "max_control_cycles",
        "max_observation_steps",
        "execute_action_steps",
        "wall_timeout_s",
        "semantic_version",
    }
)
_WATCHDOG_RUN_SPEC_FIELDS = _LEGACY_RUN_SPEC_FIELDS | frozenset({"wall_timeout_role"})
_RESULT_FIELDS = frozenset(
    {
        "trial_manifest_sha256",
        "pair_key",
        "run_spec_sha256",
        "initial_state_sha256",
        "terminal_status",
        "execution_status",
        "score_eligible",
        "score_success",
        "validation_passed",
        "validation_failure_codes",
        "clean_trace_sha256",
        "delivered_trace_sha256",
        "action_trace_sha256",
        "terminal_trace_sha256",
        "observation_count",
        "control_cycle_count",
        "failure_stage",
        "failure_code",
        "semantic_version",
    }
)


def _mapping(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ArtifactValidationError(f"{name} fields mismatch")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> Optional[str]:
    return None if value is None else _string(value, name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactValidationError(f"{name} must be an integer")
    return value


def _optional_bool(value: object, name: str) -> Optional[bool]:
    if value is not None and type(value) is not bool:
        raise ArtifactValidationError(f"{name} must be boolean or null")
    return value


def run_spec_to_dict(value: ClosedLoopRunSpec) -> dict[str, object]:
    document: dict[str, object] = {
        "prompt": value.prompt,
        "success_predicate_id": value.success_predicate_id,
        "max_control_cycles": value.max_control_cycles,
        "max_observation_steps": value.max_observation_steps,
        "execute_action_steps": value.execute_action_steps,
        "wall_timeout_s": value.wall_timeout_s,
        "semantic_version": value.semantic_version,
    }
    if value.wall_timeout_role is not WallTimeoutRole.SCORING_BOUNDARY_V1:
        document["wall_timeout_role"] = value.wall_timeout_role.value
    return document


def run_spec_from_dict(value: object) -> ClosedLoopRunSpec:
    if not isinstance(value, dict) or set(value) not in {
        _LEGACY_RUN_SPEC_FIELDS,
        _WATCHDOG_RUN_SPEC_FIELDS,
    }:
        raise ArtifactValidationError("run spec fields mismatch")
    document = value
    timeout_role = document.get(
        "wall_timeout_role", WallTimeoutRole.SCORING_BOUNDARY_V1.value
    )
    if "wall_timeout_role" in document and timeout_role != (
        WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1.value
    ):
        raise ArtifactValidationError(
            "explicit wall timeout role must select infrastructure watchdog"
        )
    timeout = document["wall_timeout_s"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ArtifactValidationError("wall timeout must be a JSON number")
    return ClosedLoopRunSpec(
        prompt=_string(document["prompt"], "prompt"),
        success_predicate_id=_string(
            document["success_predicate_id"], "success predicate id"
        ),
        max_control_cycles=_integer(
            document["max_control_cycles"], "max control cycles"
        ),
        max_observation_steps=_integer(
            document["max_observation_steps"], "max observation steps"
        ),
        execute_action_steps=_integer(
            document["execute_action_steps"], "execute action steps"
        ),
        wall_timeout_s=float(timeout),
        wall_timeout_role=WallTimeoutRole(timeout_role),
        semantic_version=_string(document["semantic_version"], "semantic version"),
    )


def result_to_dict(value: ClosedLoopTrialResult) -> dict[str, object]:
    return {
        "trial_manifest_sha256": value.trial_manifest_sha256,
        "pair_key": value.pair_key,
        "run_spec_sha256": value.run_spec_sha256,
        "initial_state_sha256": value.initial_state_sha256,
        "terminal_status": value.terminal_status.value,
        "execution_status": (
            None if value.execution_status is None else value.execution_status.value
        ),
        "score_eligible": value.score_eligible,
        "score_success": value.score_success,
        "validation_passed": value.validation_passed,
        "validation_failure_codes": list(value.validation_failure_codes),
        "clean_trace_sha256": value.clean_trace_sha256,
        "delivered_trace_sha256": value.delivered_trace_sha256,
        "action_trace_sha256": value.action_trace_sha256,
        "terminal_trace_sha256": value.terminal_trace_sha256,
        "observation_count": value.observation_count,
        "control_cycle_count": value.control_cycle_count,
        "failure_stage": value.failure_stage,
        "failure_code": value.failure_code,
        "semantic_version": value.semantic_version,
    }


def result_from_dict(value: object) -> ClosedLoopTrialResult:
    document = _mapping(value, _RESULT_FIELDS, "terminal result")
    codes = document["validation_failure_codes"]
    if not isinstance(codes, list) or not all(
        isinstance(code, str) and code for code in codes
    ):
        raise ArtifactValidationError("validation failure codes must be a string list")
    if type(document["score_eligible"]) is not bool:
        raise ArtifactValidationError("score eligible must be boolean")
    execution = document["execution_status"]
    initial_hash = document["initial_state_sha256"]
    if initial_hash is not None:
        initial_hash = require_sha256(initial_hash, "initial state sha256")
    return ClosedLoopTrialResult(
        trial_manifest_sha256=require_sha256(
            document["trial_manifest_sha256"], "trial manifest sha256"
        ),
        pair_key=require_sha256(document["pair_key"], "pair key"),
        run_spec_sha256=require_sha256(document["run_spec_sha256"], "run spec sha256"),
        initial_state_sha256=initial_hash,
        terminal_status=TerminalStatus(
            _string(document["terminal_status"], "terminal status")
        ),
        execution_status=(
            None
            if execution is None
            else TerminalStatus(_string(execution, "execution status"))
        ),
        score_eligible=document["score_eligible"],
        score_success=_optional_bool(document["score_success"], "score success"),
        validation_passed=_optional_bool(
            document["validation_passed"], "validation passed"
        ),
        validation_failure_codes=tuple(codes),
        clean_trace_sha256=require_sha256(
            document["clean_trace_sha256"], "clean trace sha256"
        ),
        delivered_trace_sha256=require_sha256(
            document["delivered_trace_sha256"], "delivered trace sha256"
        ),
        action_trace_sha256=require_sha256(
            document["action_trace_sha256"], "action trace sha256"
        ),
        terminal_trace_sha256=require_sha256(
            document["terminal_trace_sha256"], "terminal trace sha256"
        ),
        observation_count=_integer(document["observation_count"], "observation count"),
        control_cycle_count=_integer(
            document["control_cycle_count"], "control cycle count"
        ),
        failure_stage=_optional_string(document["failure_stage"], "failure stage"),
        failure_code=_optional_string(document["failure_code"], "failure code"),
        semantic_version=_string(document["semantic_version"], "semantic version"),
    )

"""Versioned success profiles layered over the pinned UniVTAC evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError

INSERT_HOLE_TASK_ID: Final[str] = "insert_hole"
INSERT_HOLE_STRICT_PREDICATE_ID: Final[str] = "insert_hole_strict_v1"


class UniVTACSuccessProfile(str, Enum):
    """Selectable terminal predicates for live UniVTAC execution."""

    OFFICIAL_V1 = "official_v1"
    INSERT_HOLE_STRICT_V1 = INSERT_HOLE_STRICT_PREDICATE_ID


@dataclass(frozen=True)
class InsertHoleStrictThresholds:
    """Frozen geometry and hold thresholds for ``insert_hole_strict_v1``."""

    xy_error_m: float = 0.005
    insertion_depth_m: float = 0.050
    alignment_dot: float = 0.999
    inhand_z_drift_m: float = 0.025
    stable_hold_s: float = 0.25
    simulator_hz: int = 120
    stable_hold_steps: int = 30

    def __post_init__(self) -> None:
        expected_steps = self.stable_hold_s * self.simulator_hz
        if not math.isclose(expected_steps, self.stable_hold_steps):
            raise ValueError("strict hold duration and step count disagree")

    def to_dict(self) -> dict[str, object]:
        """Return the exact threshold contract persisted in diagnostics."""

        return {
            "xy_error_m_lt": self.xy_error_m,
            "insertion_depth_m_gt": self.insertion_depth_m,
            "alignment_dot_gt": self.alignment_dot,
            "inhand_z_drift_m_lt": self.inhand_z_drift_m,
            "stable_hold_s_gte": self.stable_hold_s,
            "simulator_hz": self.simulator_hz,
            "stable_hold_steps_gte": self.stable_hold_steps,
        }


INSERT_HOLE_STRICT_THRESHOLDS: Final[InsertHoleStrictThresholds] = (
    InsertHoleStrictThresholds()
)


@dataclass(frozen=True)
class UniVTACSuccessEvaluation:
    """One dual-protocol evaluation at a reset or transition boundary."""

    success_profile_id: UniVTACSuccessProfile
    selected_success_predicate_id: str
    selected_success: bool
    official_success_predicate_id: str
    official_success: bool
    strict_instantaneous_success: Optional[bool]
    strict_consecutive_steps: int
    strict_success: Optional[bool]

    def to_dict(self) -> dict[str, object]:
        """Return path-free evidence for the transition diagnostics."""

        return {
            "success_profile_id": self.success_profile_id.value,
            "selected_success_predicate_id": self.selected_success_predicate_id,
            "selected_success": self.selected_success,
            "official_success_predicate_id": self.official_success_predicate_id,
            "official_success": self.official_success,
            "strict_success_predicate_id": INSERT_HOLE_STRICT_PREDICATE_ID,
            "strict_instantaneous_success": self.strict_instantaneous_success,
            "strict_consecutive_steps": self.strict_consecutive_steps,
            "strict_required_steps": (INSERT_HOLE_STRICT_THRESHOLDS.stable_hold_steps),
            "strict_success": self.strict_success,
            "strict_thresholds": INSERT_HOLE_STRICT_THRESHOLDS.to_dict(),
        }


def selected_success_predicate_id(
    *,
    task_id: str,
    official_predicate_id: str,
    profile: UniVTACSuccessProfile,
) -> str:
    """Resolve one public profile to its terminal-predicate identity."""

    selected = (
        profile
        if isinstance(profile, UniVTACSuccessProfile)
        else UniVTACSuccessProfile(profile)
    )
    if selected is UniVTACSuccessProfile.OFFICIAL_V1:
        return official_predicate_id
    if task_id != INSERT_HOLE_TASK_ID:
        raise ValueError("insert_hole_strict_v1 is only valid for insert_hole")
    return INSERT_HOLE_STRICT_PREDICATE_ID


def success_profile_from_predicate_id(
    *, task_id: str, official_predicate_id: str, selected_predicate_id: str
) -> UniVTACSuccessProfile:
    """Recover the profile from the source-bound run-spec predicate identity."""

    if selected_predicate_id == official_predicate_id:
        return UniVTACSuccessProfile.OFFICIAL_V1
    if (
        task_id == INSERT_HOLE_TASK_ID
        and selected_predicate_id == INSERT_HOLE_STRICT_PREDICATE_ID
    ):
        return UniVTACSuccessProfile.INSERT_HOLE_STRICT_V1
    raise UniVTACContractError("selected UniVTAC success predicate is unsupported")


def strict_instantaneous_success(diagnostics: Mapping[str, object]) -> bool:
    """Read the independently recomputed strict geometry predicate."""

    available = diagnostics.get("success_metrics_available")
    observed = diagnostics.get("strict_instantaneous_success")
    if available is not True or type(observed) is not bool:
        raise UniVTACContractError("insert_hole strict success metrics are unavailable")
    return observed


def strict_hold_update(
    *,
    previous_steps: int,
    instantaneous_success: bool,
    physics_step_delta: int,
) -> tuple[int, bool]:
    """Advance the consecutive 120 Hz hold counter without crossing failures."""

    if isinstance(previous_steps, bool) or previous_steps < 0:
        raise ValueError("previous strict hold steps must be non-negative")
    if isinstance(physics_step_delta, bool) or physics_step_delta < 1:
        raise ValueError("strict hold physics-step delta must be positive")
    steps = previous_steps + physics_step_delta if instantaneous_success else 0
    return steps, steps >= INSERT_HOLE_STRICT_THRESHOLDS.stable_hold_steps


def evaluate_success_profile(
    *,
    task_id: str,
    profile: UniVTACSuccessProfile,
    official_predicate_id: str,
    official_success: bool,
    task_diagnostics: Mapping[str, object],
    previous_strict_steps: int,
    physics_step_delta: Optional[int],
) -> UniVTACSuccessEvaluation:
    """Evaluate both predicates and select exactly one terminal result."""

    predicate_id = selected_success_predicate_id(
        task_id=task_id,
        official_predicate_id=official_predicate_id,
        profile=profile,
    )
    strict_instantaneous: Optional[bool] = None
    strict_steps = 0
    strict_success: Optional[bool] = None
    if task_id == INSERT_HOLE_TASK_ID:
        metrics_available = task_diagnostics.get("success_metrics_available") is True
        if metrics_available:
            strict_instantaneous = strict_instantaneous_success(task_diagnostics)
            if physics_step_delta is None:
                strict_steps = previous_strict_steps
                strict_success = (
                    strict_steps >= INSERT_HOLE_STRICT_THRESHOLDS.stable_hold_steps
                )
            else:
                strict_steps, strict_success = strict_hold_update(
                    previous_steps=previous_strict_steps,
                    instantaneous_success=strict_instantaneous,
                    physics_step_delta=physics_step_delta,
                )
        elif profile is UniVTACSuccessProfile.INSERT_HOLE_STRICT_V1:
            raise UniVTACContractError(
                "insert_hole_strict_v1 requires live target-relative geometry"
            )
    selected_success = (
        official_success
        if profile is UniVTACSuccessProfile.OFFICIAL_V1
        else strict_success is True
    )
    return UniVTACSuccessEvaluation(
        success_profile_id=profile,
        selected_success_predicate_id=predicate_id,
        selected_success=selected_success,
        official_success_predicate_id=official_predicate_id,
        official_success=official_success,
        strict_instantaneous_success=strict_instantaneous,
        strict_consecutive_steps=strict_steps,
        strict_success=strict_success,
    )


__all__ = [
    "INSERT_HOLE_STRICT_PREDICATE_ID",
    "INSERT_HOLE_STRICT_THRESHOLDS",
    "InsertHoleStrictThresholds",
    "UniVTACSuccessEvaluation",
    "UniVTACSuccessProfile",
    "evaluate_success_profile",
    "selected_success_predicate_id",
    "success_profile_from_predicate_id",
    "strict_hold_update",
    "strict_instantaneous_success",
]

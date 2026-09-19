"""Opt-in execution cadence; the N0-VTLA prediction horizon stays at 50."""

from __future__ import annotations

RECEDING_HORIZON_50X8 = "receding_horizon_50x8_v1"
RECEDING_HORIZON_TASK = "put_bottle_in_shelf"


def n0_vtla_execution_steps(profile: str | None, task_id: str) -> int:
    """Keep the official default and narrowly scope the experimental prefix."""
    if profile is None:
        return 50
    if type(profile) is not str or profile != RECEDING_HORIZON_50X8:
        raise ValueError("unsupported N0-VTLA execution profile")
    if task_id != RECEDING_HORIZON_TASK:
        raise ValueError("N0-VTLA 50x8 execution is restricted to put_bottle_in_shelf")
    return 8

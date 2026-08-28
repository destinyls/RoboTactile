"""Strict UniVTAC terminal-signal normalization."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import (
    EARLY_STOP_NONE_IS_FALSE_TASK_IDS,
    UniVTACContractError,
)
from robotactile_benchmark.closed_loop.contracts import BackendSignal


def strict_bool(value: Any, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise UniVTACContractError(f"{name} must be bool")


def action_result(value: Any) -> Tuple[bool, bool]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise UniVTACContractError(
            "take_action must return (execution_success, eval_success)"
        )
    return (
        strict_bool(value[0], "take_action execution_success"),
        strict_bool(value[1], "take_action eval_success"),
    )


def early_stop_result(value: Any, task_id: str) -> bool:
    if value is None and task_id in EARLY_STOP_NONE_IS_FALSE_TASK_IDS:
        return False
    return strict_bool(value, "check_early_stop")


def resolve_backend_signal(
    *,
    success: bool,
    execution_success: bool,
    plan_success: bool,
    early_stop: bool,
    horizon_reached: bool,
) -> BackendSignal:
    if success:
        return BackendSignal.SUCCESS
    if not execution_success or not plan_success:
        return BackendSignal.TASK_FAILURE
    if early_stop:
        return BackendSignal.EARLY_STOP
    if horizon_reached:
        return BackendSignal.TIMEOUT
    return BackendSignal.RUNNING

"""Canonical action specifications shared by policies and simulator backends."""

from __future__ import annotations

from typing import Final

QPOS8_ACTION_SPEC: Final = "qpos8_next_step"
EE8_ACTION_SPEC: Final = "ee8_absolute"
SUPPORTED_ACTION_SPECS: Final = frozenset({QPOS8_ACTION_SPEC, EE8_ACTION_SPEC})

ACTION_MODE_BY_SPEC: Final = {
    QPOS8_ACTION_SPEC: "qpos",
    EE8_ACTION_SPEC: "ee",
}

# UniVTAC's ``ee`` action is [xyz, quaternion(wxyz), gripper_qpos].  These
# bounds are a fail-closed simulator envelope, not model normalization stats.
EE8_LOWER_BOUNDS: Final = (-2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, 0.0)
EE8_UPPER_BOUNDS: Final = (2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.04)


def validate_action_spec(value: object) -> str:
    """Return one exact registered action specification."""

    if not isinstance(value, str) or not value:
        raise TypeError("action_spec must be a non-empty string")
    if value not in SUPPORTED_ACTION_SPECS:
        raise ValueError(f"action_spec must be one of {sorted(SUPPORTED_ACTION_SPECS)}")
    return value


__all__ = [
    "ACTION_MODE_BY_SPEC",
    "EE8_ACTION_SPEC",
    "EE8_LOWER_BOUNDS",
    "EE8_UPPER_BOUNDS",
    "QPOS8_ACTION_SPEC",
    "SUPPORTED_ACTION_SPECS",
    "validate_action_spec",
]

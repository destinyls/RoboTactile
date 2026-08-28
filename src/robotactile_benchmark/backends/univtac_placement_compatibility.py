"""Narrow placement compatibility for the pinned UniVTAC task contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.backends.univtac_planner_compatibility import (
    LOCAL_IK_FALLBACK_MARKER,
)
from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)

_DISTANCE_ATOL_M = 1e-6
_POSE_ATOL = 1e-8


@dataclass(frozen=True)
class _PlacementContinuationSpec:
    approach_pre_distance_m: float
    approach_final_distance_m: float
    final_pre_distance_m: float
    final_distance_m: float
    use_local_ik_fallback: bool

    @property
    def continuation_m(self) -> float:
        return self.final_pre_distance_m - self.final_distance_m


_PLACEMENT_CONTINUATION_SPECS: Mapping[str, _PlacementContinuationSpec] = (
    MappingProxyType(
        {
            "insert_hole": _PlacementContinuationSpec(
                approach_pre_distance_m=0.05,
                approach_final_distance_m=0.01,
                final_pre_distance_m=0.01,
                final_distance_m=0.002,
                use_local_ik_fallback=True,
            ),
            "insert_tube": _PlacementContinuationSpec(
                approach_pre_distance_m=0.1,
                approach_final_distance_m=0.05,
                final_pre_distance_m=0.05,
                final_distance_m=0.002,
                use_local_ik_fallback=False,
            ),
        }
    )
)


@dataclass(frozen=True)
class _PlacementAnchor:
    actor: Any
    requested_target: NDArray[np.float64]
    approach_target: Any
    zero_distance_target: Any


def _pose_values(pose: Any, label: str) -> NDArray[np.float64]:
    to_list = getattr(pose, "tolist", None)
    if not callable(to_list):
        raise UniVTACContractError(f"{label} does not expose tolist")
    values = np.asarray(to_list(), dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise UniVTACContractError(f"{label} must be a finite 7D pose")
    return values


def _single_move_action(actions: Any, label: str) -> Any:
    if not isinstance(actions, list):
        raise UniVTACContractError(f"upstream {label} placement is not a list")
    move_actions = [
        action for action in actions if getattr(action, "action", None) == "move"
    ]
    if len(move_actions) != 1:
        raise UniVTACContractError(
            f"upstream {label} placement must contain one move action"
        )
    return move_actions[0]


def _is_distance_pair(
    kwargs: dict[str, Any], *, pre_distance_m: float, final_distance_m: float
) -> bool:
    return bool(
        kwargs.get("pre_dis", 0.1) == pre_distance_m
        and kwargs.get("dis", 0.02) == final_distance_m
    )


def _placement_state(task: Any, task_id: str, phase: str) -> Mapping[str, object]:
    diagnostics = capture_task_diagnostics(
        task,
        task_id,
        include_placement_assessment=False,
    )
    required = {
        "available",
        "current_inhand_pose",
        "early_stop_predicate",
        "gripper_center_pose",
        "inhand_z_bias_m",
        "prism_pose",
    }
    if diagnostics.get("available") is not True or not required.issubset(diagnostics):
        raise UniVTACContractError(
            f"{task_id} placement witness is unavailable at {phase}"
        )
    return diagnostics


def _native_step(task: Any) -> int | None:
    value = getattr(task, "step_count", None)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    if int(value) < 0:
        raise UniVTACContractError("placement native step must be non-negative")
    return int(value)


def install_constrained_placement_compatibility(task: Any, task_id: str) -> bool:
    """Replace a stale-state recomputation with its exact task continuation."""

    spec = _PLACEMENT_CONTINUATION_SPECS.get(task_id)
    if spec is None:
        return False
    atom = getattr(task, "atom", None)
    upstream_place = getattr(atom, "place_actor", None)
    get_place_pose = getattr(atom, "get_place_pose", None)
    upstream_move = getattr(task, "move", None)
    logger = getattr(task, "logger", None)
    log_info = getattr(logger, "info", None)
    if (
        atom is None
        or not callable(upstream_place)
        or not callable(get_place_pose)
        or not callable(upstream_move)
    ):
        raise UniVTACContractError("upstream placement contract is incomplete")
    existing_witnesses = getattr(task, "_robotactile_placement_witnesses", None)
    if existing_witnesses is not None:
        raise UniVTACContractError("placement compatibility is already installed")
    task._robotactile_placement_witnesses = ()

    anchor: _PlacementAnchor | None = None
    action_phases: dict[int, str] = {}

    def compatible_move(actions: Any, *args: Any, **kwargs: Any) -> Any:
        phase = None
        if isinstance(actions, list):
            phases = {
                action_phases.pop(id(action))
                for action in actions
                if id(action) in action_phases
            }
            if len(phases) > 1:
                raise UniVTACContractError("placement action phases are ambiguous")
            phase = next(iter(phases), None)
        before = (
            None
            if phase is None
            else _placement_state(task, task_id, f"{phase}:before")
        )
        native_step_before = _native_step(task)
        result = upstream_move(actions, *args, **kwargs)
        if phase is None:
            return result
        if before is None:
            raise UniVTACContractError("placement witness before-state is unavailable")
        after = _placement_state(task, task_id, f"{phase}:after")
        existing = getattr(task, "_robotactile_placement_witnesses", ())
        if not isinstance(existing, tuple):
            raise UniVTACContractError("placement witnesses changed type")
        before_bias_value = before["inhand_z_bias_m"]
        after_bias_value = after["inhand_z_bias_m"]
        if (
            not isinstance(before_bias_value, (int, float))
            or isinstance(before_bias_value, bool)
            or not isinstance(after_bias_value, (int, float))
            or isinstance(after_bias_value, bool)
        ):
            raise UniVTACContractError("placement in-hand bias must be numeric")
        before_bias = float(before_bias_value)
        after_bias = float(after_bias_value)
        before_early_stop = before["early_stop_predicate"]
        after_early_stop = after["early_stop_predicate"]
        if type(before_early_stop) is not bool or type(after_early_stop) is not bool:
            raise UniVTACContractError("placement early-stop witness must be bool")
        plan_success = getattr(task, "plan_success", None)
        if type(plan_success) is not bool:
            raise UniVTACContractError("placement plan_success witness must be bool")
        witness = {
            "phase": phase,
            "native_step_before": native_step_before,
            "native_step_after": _native_step(task),
            "move_returned": result if type(result) is bool else None,
            "plan_success_after": plan_success,
            "before_inhand_z_bias_m": before_bias,
            "inhand_z_bias_m": after_bias,
            "inhand_z_bias_delta_m": after_bias - before_bias,
            "early_stop_predicate_before": before_early_stop,
            "early_stop_predicate": after_early_stop,
            "threshold_crossed": not before_early_stop and after_early_stop,
            "prism_pose_before": before["prism_pose"],
            "prism_pose": after["prism_pose"],
            "gripper_center_pose_before": before["gripper_center_pose"],
            "gripper_center_pose": after["gripper_center_pose"],
            "current_inhand_pose_before": before["current_inhand_pose"],
            "current_inhand_pose": after["current_inhand_pose"],
        }
        task._robotactile_placement_witnesses = (*existing, witness)
        return result

    def compatible_place_actor(*args: Any, **kwargs: Any) -> Any:
        nonlocal anchor
        actions = upstream_place(*args, **kwargs)
        actor = args[0] if args else kwargs.get("actor")
        requested_target = kwargs.get("target_pose")

        if _is_distance_pair(
            kwargs,
            pre_distance_m=spec.approach_pre_distance_m,
            final_distance_m=spec.approach_final_distance_m,
        ):
            move_action = _single_move_action(actions, "approach")
            approach_target = getattr(move_action, "target_pose", None)
            zero_distance_target = get_place_pose(
                actor,
                requested_target,
                functional_point_id=kwargs.get("functional_point_id"),
                pre_dis=0.0,
            )
            anchor = _PlacementAnchor(
                actor=actor,
                requested_target=_pose_values(requested_target, "requested target"),
                approach_target=approach_target,
                zero_distance_target=zero_distance_target,
            )
            action_phases[id(move_action)] = "approach_complete"
            return actions

        if not _is_distance_pair(
            kwargs,
            pre_distance_m=spec.final_pre_distance_m,
            final_distance_m=spec.final_distance_m,
        ):
            return actions
        if anchor is None:
            raise UniVTACContractError(f"{task_id} placement anchor is unavailable")
        if actor is not anchor.actor:
            raise UniVTACContractError(f"{task_id} placement actor changed")
        if not np.allclose(
            _pose_values(requested_target, "requested target"),
            anchor.requested_target,
            rtol=0.0,
            atol=_POSE_ATOL,
        ):
            raise UniVTACContractError(f"{task_id} placement target changed")

        move_action = _single_move_action(actions, "final")
        action_args = getattr(move_action, "args", None)
        expected_offset = spec.continuation_m
        if (
            not isinstance(action_args, dict)
            or action_args.get("pre_dis") != expected_offset
        ):
            raise UniVTACContractError(
                "upstream constrained placement offset contract changed"
            )

        approach_values = _pose_values(anchor.approach_target, "approach target")
        zero_values = _pose_values(anchor.zero_distance_target, "zero-distance target")
        approach_distance = float(np.linalg.norm(zero_values[:3] - approach_values[:3]))
        if not np.isclose(
            approach_distance,
            spec.approach_final_distance_m,
            rtol=0.0,
            atol=_DISTANCE_ATOL_M,
        ):
            raise UniVTACContractError(
                f"{task_id} approach geometry is not the expected "
                f"{spec.approach_final_distance_m * 1000:g} mm"
            )
        quaternion_alignment = float(
            abs(np.dot(approach_values[3:7], zero_values[3:7]))
        )
        if quaternion_alignment < 1.0 - _POSE_ATOL:
            raise UniVTACContractError(f"{task_id} approach orientation changed")

        continuation = (zero_values[:3] - approach_values[:3]) * (
            spec.continuation_m / approach_distance
        )
        add_bias = getattr(anchor.approach_target, "add_bias", None)
        if not callable(add_bias):
            raise UniVTACContractError("approach target cannot apply a continuation")
        continuation_target = add_bias(continuation, coord="world")
        move_action.target_pose = continuation_target
        compatible_args = dict(action_args)
        compatible_args["pre_dis"] = None
        compatible_args["constraint_pose"] = None
        if spec.use_local_ik_fallback:
            compatible_args[LOCAL_IK_FALLBACK_MARKER] = True
        move_action.args = compatible_args
        action_phases[id(move_action)] = "final_continuation_complete"
        anchor = None

        if callable(log_info):
            log_info(
                "ROBOTACTILE_PLACEMENT_CONTINUATION "
                + json.dumps(
                    {
                        "continuation_m": float(np.linalg.norm(continuation)),
                        "target_pose": _pose_values(
                            continuation_target, "continuation target"
                        ).tolist(),
                        "task_id": task_id,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return actions

    atom.place_actor = compatible_place_actor
    task.move = compatible_move
    return True

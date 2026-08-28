"""Diagnostic alignment of one failed live trace to expert demonstrations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

from robotactile_benchmark.contracts import Array, ContactPhase, canonical_hash
from robotactile_benchmark.recorded.alignment_contracts import AlignmentTrajectory
from robotactile_benchmark.recorded.metrics import action_metrics

_CLOSE_DELTA = 1e-4
_LIFT_DELTA_M = 0.01
_REOPEN_TOLERANCE = 1e-3
_ROTATION_THRESHOLDS_DEG = (10.0, 45.0, 60.0)


def _rotation_deg(quaternions: Array, anchor: Array) -> Array:
    dots = np.abs(np.sum(quaternions * anchor[None, :], axis=1))
    return np.asarray(
        np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))), dtype=np.float64
    )


def _adjacent_rotation_deg(quaternions: Array) -> Array:
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    return np.asarray(
        np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))), dtype=np.float64
    )


def _rgb_mae(left: Array, right: Array) -> float:
    if left.shape != right.shape:
        raise ValueError("alignment RGB shapes do not match")
    return float(
        np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0
    )


def _first_true(values: Array) -> Optional[int]:
    indices = np.flatnonzero(values)
    return None if indices.size == 0 else int(indices[0])


def _contact_state(phases: Sequence[ContactPhase]) -> Array:
    active = False
    values = []
    for phase in phases:
        if phase is ContactPhase.ONSET:
            active = True
        elif phase in (ContactPhase.FREE, ContactPhase.RELEASE):
            active = False
        values.append(active)
    return np.asarray(values, dtype=np.bool_)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _phase_landmarks(trajectory: AlignmentTrajectory) -> dict[str, object]:
    states = trajectory.states
    rotation = _rotation_deg(states[:, 3:7], states[0, 3:7])
    left = _contact_state(trajectory.left_phases)
    right = _contact_state(trajectory.right_phases)
    gripper = states[:, 7]
    minimum = int(np.argmin(gripper))
    reopen = _first_true(gripper[minimum:] >= gripper[0] - _REOPEN_TOLERANCE)
    return {
        "first_left_contact_index": _first_true(left),
        "first_right_contact_index": _first_true(right),
        "first_bilateral_contact_index": _first_true(left & right),
        "first_lift_10mm_index": _first_true(
            states[:, 2] - states[0, 2] >= _LIFT_DELTA_M
        ),
        "first_close_index": _first_true(gripper <= gripper[0] - _CLOSE_DELTA),
        "minimum_gripper_index": minimum,
        "minimum_gripper_qpos": float(gripper[minimum]),
        "first_reopen_index": None if reopen is None else minimum + reopen,
        "rotation_threshold_indices": {
            str(int(threshold)): _first_true(rotation >= threshold)
            for threshold in _ROTATION_THRESHOLDS_DEG
        },
    }


def _motion_summary(trajectory: AlignmentTrajectory) -> dict[str, object]:
    states = trajectory.states
    xyz = states[:, :3]
    rotation = _rotation_deg(states[:, 3:7], states[0, 3:7])
    left = _contact_state(trajectory.left_phases)
    right = _contact_state(trajectory.right_phases)
    return {
        "state_count": trajectory.count,
        "native_step_first": trajectory.native_steps[0],
        "native_step_last": trajectory.native_steps[-1],
        "native_step_median_delta": float(
            np.median(np.diff(np.asarray(trajectory.native_steps, dtype=np.int64)))
        ),
        "initial_xyz_m": xyz[0].tolist(),
        "final_xyz_m": xyz[-1].tolist(),
        "net_xyz_m": (xyz[-1] - xyz[0]).tolist(),
        "translation_path_length_m": float(
            np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
        ),
        "max_z_rise_m": float(np.max(xyz[:, 2] - xyz[0, 2])),
        "final_rotation_from_initial_deg": float(rotation[-1]),
        "max_rotation_from_initial_deg": float(np.max(rotation)),
        "rotation_path_length_deg": float(
            np.sum(_adjacent_rotation_deg(states[:, 3:7]))
        ),
        "initial_gripper_qpos": float(states[0, 7]),
        "final_gripper_qpos": float(states[-1, 7]),
        "minimum_gripper_qpos": float(np.min(states[:, 7])),
        "left_contact_fraction": float(np.mean(left)),
        "right_contact_fraction": float(np.mean(right)),
        "bilateral_contact_fraction": float(np.mean(left & right)),
        "landmarks": _phase_landmarks(trajectory),
    }


def _initial_alignment(
    live: AlignmentTrajectory, expert: AlignmentTrajectory
) -> dict[str, object]:
    dot = abs(float(np.dot(live.states[0, 3:7], expert.states[0, 3:7])))
    camera_mae = {
        "top": _rgb_mae(live.initial_top_rgb, expert.initial_top_rgb),
        "wrist": _rgb_mae(live.initial_wrist_rgb, expert.initial_wrist_rgb),
    }
    tactile_mae = {
        "left": _rgb_mae(live.initial_left_rgb, expert.initial_left_rgb),
        "right": _rgb_mae(live.initial_right_rgb, expert.initial_right_rgb),
    }
    return {
        "translation_l2_m": float(
            np.linalg.norm(live.states[0, :3] - expert.states[0, :3])
        ),
        "rotation_geodesic_deg": math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot)))),
        "gripper_abs": abs(float(live.states[0, 7] - expert.states[0, 7])),
        "camera_rgb_mae_normalized": camera_mae,
        "camera_rgb_mae_mean": float(np.mean(tuple(camera_mae.values()))),
        "tactile_rgb_mae_normalized": tactile_mae,
        "tactile_rgb_mae_mean": float(np.mean(tuple(tactile_mae.values()))),
    }


def _normalized_comparison(
    live: AlignmentTrajectory, expert: AlignmentTrajectory, sample_count: int
) -> dict[str, object]:
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2")
    live_indices = np.rint(np.linspace(0, live.count - 1, sample_count)).astype(int)
    expert_indices = np.rint(np.linspace(0, expert.count - 1, sample_count)).astype(int)
    live_states = live.states[live_indices]
    expert_states = expert.states[expert_indices]
    absolute = action_metrics(live_states, expert_states)
    live_relative_xyz = live_states[:, :3] - live_states[0, :3]
    expert_relative_xyz = expert_states[:, :3] - expert_states[0, :3]
    relative_translation = np.linalg.norm(
        live_relative_xyz - expert_relative_xyz, axis=1
    )
    live_rotation = _rotation_deg(live_states[:, 3:7], live_states[0, 3:7])
    expert_rotation = _rotation_deg(expert_states[:, 3:7], expert_states[0, 3:7])
    live_gripper = live_states[:, 7] - live_states[0, 7]
    expert_gripper = expert_states[:, 7] - expert_states[0, 7]
    return {
        "method": "nearest_index_normalized_progress_v1",
        "sample_count": sample_count,
        "absolute_ee8": absolute.to_dict(),
        "relative_translation_mean_l2_m": float(np.mean(relative_translation)),
        "relative_translation_final_l2_m": float(relative_translation[-1]),
        "rotation_progress_mean_abs_deg": float(
            np.mean(np.abs(live_rotation - expert_rotation))
        ),
        "rotation_progress_final_abs_deg": float(
            abs(live_rotation[-1] - expert_rotation[-1])
        ),
        "gripper_progress_mean_abs": float(
            np.mean(np.abs(live_gripper - expert_gripper))
        ),
        "gripper_progress_final_abs": float(abs(live_gripper[-1] - expert_gripper[-1])),
    }


def build_expert_alignment_report(
    *,
    task_id: str,
    live: AlignmentTrajectory,
    experts: Sequence[AlignmentTrajectory],
    executed_action_count: int,
    sample_count: int = 101,
) -> Mapping[str, object]:
    """Build a source-bound diagnostic report without claiming model accuracy."""

    candidates = tuple(experts)
    if not task_id or not candidates:
        raise ValueError("task_id and at least one expert trajectory are required")
    initial = {
        expert.trajectory_id: _initial_alignment(live, expert) for expert in candidates
    }
    selected = min(
        candidates,
        key=lambda item: (
            _number(initial[item.trajectory_id]["camera_rgb_mae_mean"], "camera MAE"),
            _number(initial[item.trajectory_id]["translation_l2_m"], "translation"),
            item.source_sha256,
        ),
    )
    live_motion = _motion_summary(live)
    selected_motion = _motion_summary(selected)
    live_landmarks = live_motion["landmarks"]
    expert_landmarks = selected_motion["landmarks"]
    assert isinstance(live_landmarks, dict) and isinstance(expert_landmarks, dict)
    codes = []
    if live_landmarks["first_bilateral_contact_index"] is None:
        codes.append("live_no_bilateral_contact")
    if expert_landmarks["first_bilateral_contact_index"] is not None:
        codes.append("selected_expert_has_bilateral_contact")
    if (
        _number(live_motion["max_rotation_from_initial_deg"], "live rotation")
        < 10.0
        <= _number(selected_motion["max_rotation_from_initial_deg"], "expert rotation")
    ):
        codes.append("live_rotation_progress_below_expert")
    report: dict[str, object] = {
        "schema_version": "robotactile-expert-live-alignment-v1",
        "evidence_level": "recorded_expert_vs_live_closed_loop_diagnostic_v1",
        "task_id": task_id,
        "ranking_method": "mean_normalized_top_wrist_rgb_mae_then_ee_translation_v1",
        "live": {
            "trajectory_id": live.trajectory_id,
            "artifact_root_sha256": live.source_sha256,
            "executed_action_count": executed_action_count,
            "motion": live_motion,
        },
        "expert_candidates": [
            {
                "trajectory_id": expert.trajectory_id,
                "hdf5_sha256": expert.source_sha256,
                "initial_alignment": initial[expert.trajectory_id],
                "motion": _motion_summary(expert),
            }
            for expert in candidates
        ],
        "selected_expert_id": selected.trajectory_id,
        "selected_expert_sha256": selected.source_sha256,
        "selected_alignment": _normalized_comparison(live, selected, sample_count),
        "diagnostic_codes": codes,
        "claim_boundaries": [
            "expert HDF5 files are demonstrations without a live success receipt",
            "normalized-progress alignment is diagnostic and not a benchmark score",
            "current live artifacts do not contain object pose or contact actor identity",
        ],
    }
    report["report_content_sha256"] = canonical_hash(report)
    return report


__all__ = ["build_expert_alignment_report"]

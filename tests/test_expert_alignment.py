from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robotactile_benchmark.contracts import ContactPhase
from robotactile_benchmark.recorded.alignment import build_expert_alignment_report
from robotactile_benchmark.recorded.alignment_contracts import AlignmentTrajectory
from robotactile_benchmark.recorded.alignment_io import write_alignment_report


def _trajectory(
    trajectory_id: str,
    *,
    image_value: int,
    bilateral: bool,
    final_rotation_deg: float,
) -> AlignmentTrajectory:
    count = 20
    states = np.zeros((count, 8), dtype=np.float32)
    states[:, 0] = np.linspace(0.5, 0.55, count)
    states[:, 2] = np.linspace(0.14, 0.20, count)
    angle = np.radians(np.linspace(0.0, final_rotation_deg, count)) / 2.0
    states[:, 3] = np.cos(angle)
    states[:, 6] = np.sin(angle)
    states[:, 7] = np.concatenate(
        (np.linspace(0.02, 0.009, 10), np.linspace(0.009, 0.02, 10))
    )
    left = [ContactPhase.FREE] * count
    right = [ContactPhase.FREE] * count
    left[5] = ContactPhase.ONSET
    left[6:] = [ContactPhase.SUSTAINED] * (count - 6)
    if bilateral:
        right[6] = ContactPhase.ONSET
        right[7:] = [ContactPhase.SUSTAINED] * (count - 7)
    image = np.full((8, 10, 3), image_value, dtype=np.uint8)
    return AlignmentTrajectory(
        trajectory_id=trajectory_id,
        source_sha256=("a" if trajectory_id == "live" else "b") * 64,
        states=states,
        native_steps=tuple(range(count)),
        left_phases=tuple(left),
        right_phases=tuple(right),
        initial_top_rgb=image,
        initial_wrist_rgb=image,
        initial_left_rgb=image,
        initial_right_rgb=image,
    )


def test_alignment_selects_visual_nearest_and_reports_behavior_gap(
    tmp_path: Path,
) -> None:
    live = _trajectory("live", image_value=10, bilateral=False, final_rotation_deg=5)
    near = _trajectory(
        "expert:near", image_value=11, bilateral=True, final_rotation_deg=70
    )
    far = _trajectory(
        "expert:far", image_value=200, bilateral=True, final_rotation_deg=70
    )

    report = build_expert_alignment_report(
        task_id="lift_bottle",
        live=live,
        experts=(far, near),
        executed_action_count=19,
        sample_count=11,
    )

    assert report["selected_expert_id"] == "expert:near"
    assert report["diagnostic_codes"] == [
        "live_no_bilateral_contact",
        "selected_expert_has_bilateral_contact",
        "live_rotation_progress_below_expert",
    ]
    output = tmp_path / "alignment.json"
    first = write_alignment_report(output, report)
    second = write_alignment_report(output, report)
    assert first == second
    assert json.loads(output.read_text())["selected_expert_id"] == "expert:near"

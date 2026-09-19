"""Official q01/q99 normalization artifacts for single-arm absEE20 data."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from .contracts import TASKS, USED_ACTION_CHANNEL_IDS
from .lerobot import read_action_rows
from .manifest_io import write_or_verify_json

FloatArray: TypeAlias = npt.NDArray[np.float32]
DEAD_SPAN = 1e-4
FLOORED_HALF_SPAN = 5e-4


def action_normalization(
    rows: npt.ArrayLike,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute N0's 20D q01/q99 policy with neutral unused dimensions."""

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
        raise ValueError("normalization rows must be finite [N,20]")
    if values.shape[0] == 0:
        raise ValueError("normalization rows must be non-empty")
    q01_raw = np.quantile(values, 0.01, axis=0)
    q99_raw = np.quantile(values, 0.99, axis=0)
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    q01 = q01_raw.copy()
    q99 = q99_raw.copy()

    # Match the official post-training pool builder: preserve the full observed
    # gripper lower range and floor dead active channels to a non-zero span.
    q01[9] = minimum[9]
    floored: list[int] = []
    for channel in USED_ACTION_CHANNEL_IDS:
        if float(q99[channel] - q01[channel]) < DEAD_SPAN:
            center = 0.5 * float(q01[channel] + q99[channel])
            q01[channel] = center - FLOORED_HALF_SPAN
            q99[channel] = center + FLOORED_HALF_SPAN
            floored.append(channel)
    q01[10:] = -1.0
    q99[10:] = 1.0
    if np.any(q99 <= q01):
        raise ValueError("normalization produced a non-positive channel span")
    norm: dict[str, object] = {
        "q01": [float(value) for value in q01],
        "q99": [float(value) for value in q99],
    }
    report: dict[str, object] = {
        "action_schema": "ee20_absolute_next_step",
        "sample_count": int(values.shape[0]),
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "q01_raw": [float(value) for value in q01_raw],
        "q99_raw": [float(value) for value in q99_raw],
        "minimum_raw": [float(value) for value in minimum],
        "maximum_raw": [float(value) for value in maximum],
        "gripper_lower_policy": "observed_minimum",
        "floored_channels": floored,
        "unused_channel_policy": "neutral_minus1_plus1",
    }
    return norm, report


def write_normalization_artifacts(
    task_repos: Mapping[str, Path], *, output_root: Path
) -> dict[str, object]:
    """Write per-repo and global artifacts from all eight completed repos."""

    if tuple(task for task in TASKS if task in task_repos) != TASKS or set(
        task_repos
    ) != set(TASKS):
        raise ValueError("normalization requires exactly the eight UniVTAC task repos")
    all_rows: list[FloatArray] = []
    per_repo: dict[str, object] = {}
    per_task_reports: dict[str, object] = {}
    for task in TASKS:
        rows = read_action_rows(task_repos[task])
        norm, report = action_normalization(rows)
        all_rows.append(rows)
        per_repo[task] = norm
        per_task_reports[task] = report
        write_or_verify_json(
            task_repos[task] / "_robotactile_action_stats.json",
            {
                "schema_version": 1,
                "task": task,
                "normalization": norm,
                "report": report,
            },
        )

    global_norm, global_report = action_normalization(np.concatenate(all_rows, axis=0))
    norm_path = output_root / "norm_stat_absee.json"
    per_repo_path = output_root / "norm_stat_absee_per_repo.json"
    report_path = output_root / "norm_stat_absee_raw_report.json"
    write_or_verify_json(norm_path, global_norm)
    write_or_verify_json(per_repo_path, per_repo)
    write_or_verify_json(
        report_path,
        {
            "schema_version": 1,
            "global": global_report,
            "per_repo": per_task_reports,
        },
    )
    return {
        "norm_stat_path": str(norm_path),
        "per_repo_norm_stat_path": str(per_repo_path),
        "raw_report_path": str(report_path),
        "global_sample_count": global_report["sample_count"],
    }


__all__ = ["action_normalization", "write_normalization_artifacts"]

"""Exact task/global receipts consumed by latent and training launchers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import (
    ACTION_PER_FRAME,
    SOURCE_FPS,
    SOURCE_MANIFEST_NAME,
    TASKS,
    USED_ACTION_CHANNEL_IDS,
    SourceEpisode,
    sha256_file,
)
from .episode import IMAGE_PATHS
from .lerobot import verify_lerobot_payload
from .manifest_io import (
    load_json_object,
    receipt_sha256,
    signed_payload,
)

TASK_RECEIPT_NAME = "_robotactile_task_receipt.json"
GLOBAL_RECEIPT_NAME = "train759_receipt.json"


def expected_episode_map(
    records: Sequence[SourceEpisode],
) -> list[dict[str, object]]:
    """Map dense task-local LeRobot IDs back to immutable source IDs."""

    return [
        {
            "lerobot_episode_index": index,
            "source_episode_id": record.episode_id,
            "source_relative_path": record.relative_path,
            "source_sha256": record.sha256,
            "usable_source_range": (
                None
                if record.usable_source_range is None
                else list(record.usable_source_range)
            ),
        }
        for index, record in enumerate(records)
    ]


def build_task_receipt(
    *,
    task: str,
    repo_relative_path: str,
    manifest_sha256: str,
    records: Sequence[SourceEpisode],
    writer_result: Mapping[str, object],
) -> dict[str, object]:
    """Bind one completed physical repo to train sources and its norm key."""

    raw_map = writer_result.get("episode_map")
    if not isinstance(raw_map, list):
        raise ValueError("task writer did not return an episode_map")
    expected_map = expected_episode_map(records)
    if len(raw_map) != len(expected_map):
        raise ValueError("task writer episode map length is invalid")
    episode_map: list[dict[str, object]] = []
    total_frames = 0
    for expected, actual in zip(expected_map, raw_map):
        if not isinstance(actual, dict):
            raise ValueError("task writer episode map entries must be objects")
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"task writer source mapping mismatch: {key}")
        frame_count = actual.get("converted_frame_count")
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count <= 0
        ):
            raise ValueError("task writer converted frame count is invalid")
        episode_map.append(dict(actual))
        total_frames += frame_count
    if writer_result.get("frame_count") != total_frames:
        raise ValueError("task writer total frame count is invalid")
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "univtac_train759_absee20_task_v1",
        "task": task,
        "repo_relative_path": repo_relative_path,
        "physical_repo_basename": task,
        "per_repo_norm_key": task,
        "source_manifest_sha256": manifest_sha256,
        "source_split": "train",
        "episode_count": len(records),
        "frame_count": total_frames,
        "source_fps": SOURCE_FPS,
        "action_per_frame": ACTION_PER_FRAME,
        "action_schema": "ee20_absolute_next_step",
        "quaternion_order": "wxyz",
        "gripper_source": "embodiment/joint[:,7]",
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "observation_keys": [name for name, _, _ in IMAGE_PATHS],
        "episode_map": episode_map,
    }
    return signed_payload(unsigned, field="task_receipt_sha256")


def validate_task_repo(
    *, destination: Path, records: Sequence[SourceEpisode], manifest_sha256: str
) -> dict[str, object]:
    """Validate a completed task repo before treating it as resumable output."""

    receipt_path = destination / TASK_RECEIPT_NAME
    if not destination.is_dir() or not receipt_path.is_file():
        raise FileExistsError(
            f"existing task destination is not receipted: {destination}"
        )
    receipt = load_json_object(receipt_path)
    receipt_sha256(receipt, field="task_receipt_sha256")
    task = records[0].task
    expected_scalars = {
        "status": "complete",
        "task": task,
        "source_manifest_sha256": manifest_sha256,
        "source_split": "train",
        "episode_count": len(records),
        "source_fps": SOURCE_FPS,
        "action_per_frame": ACTION_PER_FRAME,
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "physical_repo_basename": task,
        "per_repo_norm_key": task,
    }
    if any(receipt.get(key) != value for key, value in expected_scalars.items()):
        raise ValueError(f"task receipt contract mismatch: {destination}")
    raw_map = receipt.get("episode_map")
    if not isinstance(raw_map, list) or len(raw_map) != len(records):
        raise ValueError(f"task receipt episode map is invalid: {destination}")
    expected_map = expected_episode_map(records)
    for actual, expected in zip(raw_map, expected_map):
        if not isinstance(actual, dict) or any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"task receipt source mapping mismatch: {destination}")
    verify_lerobot_payload(destination, expected_episodes=len(records))
    return receipt


def build_global_receipt(
    *,
    output_root: Path,
    manifest: Mapping[str, object],
    task_receipts: Mapping[str, Mapping[str, object]],
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact complete-only receipt consumed by downstream launchers."""

    repos: list[dict[str, object]] = []
    total_frames = 0
    for task in TASKS:
        receipt = task_receipts[task]
        frame_count = receipt.get("frame_count")
        digest = receipt.get("task_receipt_sha256")
        if not isinstance(frame_count, int) or not isinstance(digest, str):
            raise ValueError(f"task receipt is incomplete: {task}")
        total_frames += frame_count
        repos.append(
            {
                "task": task,
                "repo_relative_path": f"train759/{task}",
                "physical_repo_basename": task,
                "per_repo_norm_key": task,
                "task_receipt_relative_path": (f"train759/{task}/{TASK_RECEIPT_NAME}"),
                "task_receipt_sha256": digest,
                "episode_count": receipt["episode_count"],
                "frame_count": frame_count,
            }
        )
    norm_fields: dict[str, object] = {}
    for key in ("norm_stat_path", "per_repo_norm_stat_path", "raw_report_path"):
        raw_path = normalization.get(key)
        if not isinstance(raw_path, str):
            raise ValueError(f"normalization result is missing {key}")
        path = Path(raw_path)
        norm_fields[key] = path.relative_to(output_root).as_posix()
        norm_fields[f"{key}_sha256"] = sha256_file(path)
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "univtac_train759_absee20_v1",
        "dataset_path": str((output_root / "train759").resolve()),
        "source_manifest_relative_path": SOURCE_MANIFEST_NAME,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_split_counts": manifest["split_counts"],
        "materialized_split": "train",
        "materialized_episode_count": 759,
        "excluded_episode_count": 41,
        "completed_task_count": len(TASKS),
        "frame_count": total_frames,
        "source_fps": SOURCE_FPS,
        "action_per_frame": ACTION_PER_FRAME,
        "action_schema": "ee20_absolute_next_step",
        "quaternion_order": "wxyz",
        "gripper_source": "embodiment/joint[:,7]",
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "validation_dataset_path": None,
        "materialized_splits": ["train"],
        "task_repos": repos,
        "normalization": norm_fields,
    }
    return signed_payload(unsigned, field="receipt_sha256")


__all__ = [
    "GLOBAL_RECEIPT_NAME",
    "TASK_RECEIPT_NAME",
    "build_global_receipt",
    "build_task_receipt",
    "expected_episode_map",
    "validate_task_repo",
]

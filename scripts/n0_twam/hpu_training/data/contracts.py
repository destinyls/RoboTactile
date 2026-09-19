"""Immutable UniVTAC split and source-identity contracts.

This module deliberately lives outside the official N0-TWAM checkout.  It
prepares the only data tree mounted by formal training: ``train759``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence

Split = Literal["train", "frozen", "quarantine"]

TASKS: Final[tuple[str, ...]] = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
FROZEN_EPISODE_IDS: Final[frozenset[int]] = frozenset({0, 1, 2, 3, 5})
QUARANTINE_EPISODES: Final[frozenset[tuple[str, int]]] = frozenset(
    {("grasp_classify", 90)}
)
SOURCE_FPS: Final[int] = 10
ACTION_PER_FRAME: Final[int] = 4
USED_ACTION_CHANNEL_IDS: Final[tuple[int, ...]] = tuple(range(10))
EXPECTED_SOURCE_EPISODES_PER_TASK: Final[int] = 100
EXPECTED_SPLIT_COUNTS: Final[dict[Split, int]] = {
    "train": 759,
    "frozen": 40,
    "quarantine": 1,
}
SOURCE_MANIFEST_NAME: Final[str] = "source_split_manifest.json"
SOURCE_MANIFEST_PROTOCOL: Final[str] = "univtac_train759_absee20_v1"
LIFT_CAN_62_SHA256: Final[str] = (
    "335f6c32947b1d37acb638e67a756b7c37204df4c849121822234c7931fd63f2"
)

TASK_PROMPTS: Final[dict[str, str]] = {
    "grasp_classify": "Grasp an object and classify it by tactile texture",
    "insert_HDMI": "Insert an HDMI connector into a port",
    "insert_hole": "Precision peg-in-hole insertion",
    "insert_tube": "Insert a tube into a tilted fixture",
    "lift_bottle": "Grasp and lift a bottle off a surface near a wall",
    "lift_can": "Rotate a lying can so it stands upright",
    "pull_out_key": "Untwist and extract a key from a lock",
    "put_bottle_in_shelf": "Reorient a bottle upright and place it on a shelf",
}


@dataclass(frozen=True)
class SourceEpisode:
    """One content-addressed UniVTAC HDF5 source and its frozen split."""

    root: Path
    relative_path: str
    task: str
    episode_id: int
    split: Split
    size_bytes: int
    mtime_ns: int
    sha256: str

    @property
    def path(self) -> Path:
        return self.root / self.relative_path

    @property
    def usable_source_range(self) -> tuple[int, int] | None:
        if self.task == "lift_can" and self.episode_id == 62:
            return (0, 228)
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "task": self.task,
            "episode_id": self.episode_id,
            "split": self.split,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "usable_source_range": (
                None
                if self.usable_source_range is None
                else list(self.usable_source_range)
            ),
        }


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(task: str, episode_id: int) -> Split:
    """Apply the frozen40/quarantine1 split before any materialization."""

    if task not in TASKS:
        raise ValueError(f"unsupported UniVTAC task: {task}")
    if episode_id < 0 or episode_id >= EXPECTED_SOURCE_EPISODES_PER_TASK:
        raise ValueError(f"episode ID is outside [0, 100): {task}/{episode_id}")
    if episode_id in FROZEN_EPISODE_IDS:
        return "frozen"
    if (task, episode_id) in QUARANTINE_EPISODES:
        return "quarantine"
    return "train"


def _regular_source(path: Path, root: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"source is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"source must be a non-symlink regular file: {path}")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"source escapes the UniVTAC root: {path}")
    return metadata


def _source_path(root: Path, task: str, episode_id: int) -> Path:
    candidates = (
        root / task / "clean" / f"{episode_id}.hdf5",
        root / task / "clean" / f"{episode_id}.h5",
    )
    present = tuple(path for path in candidates if path.exists())
    if len(present) != 1:
        raise ValueError(
            f"expected exactly one HDF5 source for {task}/{episode_id}, "
            f"found {len(present)}"
        )
    return present[0]


def discover_source_episodes(
    raw_root: Path,
    *,
    hash_sources: bool = True,
) -> tuple[SourceEpisode, ...]:
    """Discover the exact 8 x 100 source grid and freeze every split identity."""

    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    expected_paths: set[Path] = set()
    records: list[SourceEpisode] = []
    for task in TASKS:
        task_clean = root / task / "clean"
        if not task_clean.is_dir():
            raise ValueError(f"missing UniVTAC task directory: {task_clean}")
        for episode_id in range(EXPECTED_SOURCE_EPISODES_PER_TASK):
            path = _source_path(root, task, episode_id)
            metadata = _regular_source(path, root)
            expected_paths.add(path.resolve(strict=True))
            records.append(
                SourceEpisode(
                    root=root,
                    relative_path=path.relative_to(root).as_posix(),
                    task=task,
                    episode_id=episode_id,
                    split=split_for(task, episode_id),
                    size_bytes=metadata.st_size,
                    mtime_ns=metadata.st_mtime_ns,
                    sha256=sha256_file(path) if hash_sources else "",
                )
            )
    actual_paths = {
        path.resolve(strict=True)
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".h5", ".hdf5"}
    }
    unexpected = sorted(str(path) for path in actual_paths - expected_paths)
    if unexpected:
        raise ValueError(f"unexpected UniVTAC HDF5 sources: {unexpected[:5]}")
    _validate_split_counts(records)
    return tuple(records)


def _validate_split_counts(records: Sequence[SourceEpisode]) -> None:
    if len(records) != len(TASKS) * EXPECTED_SOURCE_EPISODES_PER_TASK:
        raise ValueError("UniVTAC source grid must contain exactly 800 episodes")
    counts = {
        split: sum(record.split == split for record in records)
        for split in EXPECTED_SPLIT_COUNTS
    }
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"UniVTAC split counts are invalid: {counts}")
    per_task_train = {
        task: sum(record.task == task and record.split == "train" for record in records)
        for task in TASKS
    }
    expected = {task: 95 for task in TASKS}
    expected["grasp_classify"] = 94
    if per_task_train != expected:
        raise ValueError(f"per-task train759 counts are invalid: {per_task_train}")


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_source_manifest(
    records: Sequence[SourceEpisode], *, raw_root: Path
) -> dict[str, object]:
    """Build the immutable split manifest consumed by every task transaction."""

    _validate_split_counts(records)
    if any(not record.sha256 for record in records):
        raise ValueError("source manifest requires SHA256 for every HDF5 episode")
    episodes = [record.to_json() for record in records]
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in EXPECTED_SPLIT_COUNTS
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": SOURCE_MANIFEST_PROTOCOL,
        "raw_root": str(raw_root.resolve(strict=True)),
        "source_fps": SOURCE_FPS,
        "action_per_frame": ACTION_PER_FRAME,
        "action_schema": "ee20_absolute_next_step",
        "quaternion_order": "wxyz",
        "gripper_source": "embodiment/joint[:,7]",
        "used_action_channel_ids": list(USED_ACTION_CHANNEL_IDS),
        "split_counts": split_counts,
        "materialized_splits": ["train"],
        "episodes": episodes,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


__all__ = [
    "ACTION_PER_FRAME",
    "EXPECTED_SPLIT_COUNTS",
    "FROZEN_EPISODE_IDS",
    "LIFT_CAN_62_SHA256",
    "QUARANTINE_EPISODES",
    "SOURCE_FPS",
    "SOURCE_MANIFEST_NAME",
    "TASK_PROMPTS",
    "TASKS",
    "USED_ACTION_CHANNEL_IDS",
    "SourceEpisode",
    "build_source_manifest",
    "canonical_json_sha256",
    "discover_source_episodes",
    "sha256_file",
    "split_for",
]

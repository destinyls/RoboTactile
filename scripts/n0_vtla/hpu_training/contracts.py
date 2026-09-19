"""Fail-closed UniVTAC train-only contracts for N0-VTLA.

The split was originally frozen by the N0-TWAM data preparation pipeline.  It
is action-schema independent, so N0-VTLA reuses the identities and split labels
while materializing a separate qpos8 view.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

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
FROZEN_IDS: Final[frozenset[int]] = frozenset({0, 1, 2, 3, 5})
SOURCE_PROTOCOL: Final[str] = "univtac_train759_absee20_v1"
EXPECTED_SPLITS: Final[dict[str, int]] = {
    "train": 759,
    "frozen": 40,
    "quarantine": 1,
}


@dataclass(frozen=True)
class SourceRecord:
    """One source episode accepted from the certified split manifest."""

    raw_root: Path
    relative_path: str
    task: str
    episode_id: int
    split: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    usable_source_range: tuple[int, int] | None

    @property
    def path(self) -> Path:
        return self.raw_root / self.relative_path


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_split(task: str, episode_id: int) -> str:
    if episode_id in FROZEN_IDS:
        return "frozen"
    if task == "grasp_classify" and episode_id == 90:
        return "quarantine"
    return "train"


def _regular_source(record: SourceRecord) -> None:
    path = record.path
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if record.raw_root not in resolved.parents:
        raise ValueError(f"source escapes raw root: {path}")
    if (metadata.st_size, metadata.st_mtime_ns) != (
        record.size_bytes,
        record.mtime_ns,
    ):
        raise ValueError(f"source identity changed: {record.relative_path}")


def _parse_record(item: Mapping[str, object], raw_root: Path) -> SourceRecord:
    task = item.get("task")
    episode_id = item.get("episode_id")
    split = item.get("split")
    relative_path = item.get("relative_path")
    size_bytes = item.get("size_bytes")
    mtime_ns = item.get("mtime_ns")
    digest = item.get("sha256")
    source_range = item.get("usable_source_range")
    if (
        task not in TASKS
        or isinstance(episode_id, bool)
        or not isinstance(episode_id, int)
        or not 0 <= episode_id < 100
        or split != _expected_split(str(task), episode_id)
        or relative_path != f"{task}/clean/{episode_id}.hdf5"
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or isinstance(mtime_ns, bool)
        or not isinstance(mtime_ns, int)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"invalid source manifest record: {relative_path!r}")
    expected_range: tuple[int, int] | None = (
        (0, 228) if task == "lift_can" and episode_id == 62 else None
    )
    observed_range = (
        tuple(source_range)
        if isinstance(source_range, list)
        and len(source_range) == 2
        and all(isinstance(value, int) for value in source_range)
        else None
    )
    if observed_range != expected_range:
        raise ValueError(f"unexpected usable range: {relative_path}")
    record = SourceRecord(
        raw_root=raw_root,
        relative_path=str(relative_path),
        task=str(task),
        episode_id=episode_id,
        split=str(split),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        sha256=digest,
        usable_source_range=expected_range,
    )
    _regular_source(record)
    return record


def load_certified_manifest(
    manifest_path: Path, *, raw_root: Path
) -> tuple[str, tuple[SourceRecord, ...]]:
    """Validate all 800 identities and return the pinned manifest digest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest must be a JSON object")
    unsigned = dict(payload)
    claimed_sha = unsigned.pop("manifest_sha256", None)
    actual_sha = canonical_json_sha256(unsigned)
    root = raw_root.resolve(strict=True)
    if (
        claimed_sha != actual_sha
        or payload.get("protocol_id") != SOURCE_PROTOCOL
        or payload.get("raw_root") != str(root)
        or payload.get("split_counts") != EXPECTED_SPLITS
        or payload.get("materialized_splits") != ["train"]
    ):
        raise ValueError("source manifest header is not the certified train759 split")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 800:
        raise ValueError("source manifest must contain exactly 800 episodes")
    records = tuple(
        _parse_record(item, root) for item in episodes if isinstance(item, dict)
    )
    if len(records) != 800:
        raise ValueError("source manifest contains non-object episode records")
    identities = {(record.task, record.episode_id) for record in records}
    if len(identities) != 800:
        raise ValueError("source manifest contains duplicate task/episode identities")
    return actual_sha, records


def select_train_records(
    records: Sequence[SourceRecord], *, task: str
) -> tuple[SourceRecord, ...]:
    if task not in TASKS:
        raise ValueError(f"unsupported UniVTAC task: {task}")
    selected = tuple(
        sorted(
            (
                record
                for record in records
                if record.task == task and record.split == "train"
            ),
            key=lambda record: record.episode_id,
        )
    )
    expected = 94 if task == "grasp_classify" else 95
    if len(selected) != expected:
        raise ValueError(
            f"{task} must have {expected} train episodes, got {len(selected)}"
        )
    return selected


def atomic_json(path: Path, payload: object) -> None:
    """Create one JSON artifact without overwriting an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    data = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

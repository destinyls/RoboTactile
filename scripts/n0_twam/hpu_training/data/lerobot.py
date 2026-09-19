"""LeRobot v2.1 writer boundary for the train759 task repositories."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from .contracts import SOURCE_FPS, TASK_PROMPTS, SourceEpisode
from .episode import IMAGE_PATHS, iter_episode_frames

FloatArray: TypeAlias = npt.NDArray[np.float32]


class LeRobotSink(Protocol):
    """Narrow surface shared by LeRobot 0.3.3 and unit-test sinks."""

    def add_frame(
        self, frame: dict[str, object], task: str, timestamp: float
    ) -> None: ...

    def save_episode(self) -> None: ...

    def stop_image_writer(self) -> None: ...


def official_feature_schema() -> dict[str, dict[str, object]]:
    """Return the 20D absEE + four-video feature schema used by official N0."""

    channel_names = [
        "x",
        "y",
        "z",
        "rot6d_0",
        "rot6d_1",
        "rot6d_2",
        "rot6d_3",
        "rot6d_4",
        "rot6d_5",
        "gripper",
    ] + [f"padding_{index}" for index in range(10)]
    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": channel_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (20,),
            "names": channel_names,
        },
    }
    for feature_name, _, shape in IMAGE_PATHS:
        features[feature_name] = {
            "dtype": "video",
            "shape": shape,
            "names": ("height", "width", "channels"),
        }
    return features


def convert_episode_to_sink(record: SourceEpisode, sink: LeRobotSink) -> int:
    """Write one source episode with dense local frames and a strict next target."""

    if record.split != "train":
        raise ValueError(
            f"refusing to materialize non-train source: {record.relative_path}"
        )
    count = 0
    prompt = TASK_PROMPTS[record.task]
    for frame, timestamp in iter_episode_frames(record):
        sink.add_frame(frame, task=prompt, timestamp=timestamp)
        count += 1
    if count <= 0:
        raise ValueError(f"source emitted no next-step rows: {record.relative_path}")
    sink.save_episode()
    return count


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def freeze_episode_action_config(root: Path) -> None:
    """Add whole-episode action segments required by the official latent loader."""

    path = root / "meta" / "episodes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"LeRobot episodes metadata is empty: {path}")
    output: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"episodes line {line_number} must be a JSON object")
        episode_index = payload.get("episode_index")
        length = payload.get("length")
        tasks = payload.get("tasks")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or not isinstance(tasks, list)
            or not tasks
            or not isinstance(tasks[0], str)
        ):
            raise ValueError(f"episodes line {line_number} is incomplete")
        expected = [{"start_frame": 0, "end_frame": length, "action_text": tasks[0]}]
        if payload.get("action_config", expected) != expected:
            raise ValueError(f"episode {episode_index} has incompatible action_config")
        payload["action_config"] = expected
        output.append(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    _atomic_text(path, "\n".join(output) + "\n")


def _lerobot_dataset_class() -> Any:
    try:
        module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    except ImportError as exc:
        raise RuntimeError(
            "materialization requires the official N0-TWAM LeRobot 0.3.3 runtime"
        ) from exc
    return module.LeRobotDataset


def write_task_repo(
    records: Sequence[SourceEpisode],
    *,
    output_root: Path,
    image_writer_threads: int = 8,
) -> dict[str, object]:
    """Create one complete task-local LeRobot v2.1 repository."""

    ordered = tuple(sorted(records, key=lambda item: item.episode_id))
    if not ordered or len({record.task for record in ordered}) != 1:
        raise ValueError("one task writer call requires non-empty same-task records")
    if any(record.split != "train" for record in ordered):
        raise ValueError("task repo may contain train split records only")
    if output_root.exists():
        raise FileExistsError(f"LeRobot staging root already exists: {output_root}")
    task = ordered[0].task
    dataset_class = _lerobot_dataset_class()
    dataset: LeRobotSink = dataset_class.create(
        repo_id=f"robotactile/univtac_{task}_train759",
        root=str(output_root),
        fps=SOURCE_FPS,
        robot_type="franka_panda",
        features=official_feature_schema(),
        use_videos=True,
        image_writer_threads=image_writer_threads,
    )
    episode_map: list[dict[str, object]] = []
    total_frames = 0
    try:
        for lerobot_episode_index, record in enumerate(ordered):
            frame_count = convert_episode_to_sink(record, dataset)
            total_frames += frame_count
            episode_map.append(
                {
                    "lerobot_episode_index": lerobot_episode_index,
                    "source_episode_id": record.episode_id,
                    "source_relative_path": record.relative_path,
                    "source_sha256": record.sha256,
                    "usable_source_range": (
                        None
                        if record.usable_source_range is None
                        else list(record.usable_source_range)
                    ),
                    "converted_frame_count": frame_count,
                }
            )
    finally:
        dataset.stop_image_writer()
    freeze_episode_action_config(output_root)
    verify_lerobot_payload(output_root, expected_episodes=len(ordered))
    return {
        "task": task,
        "episode_count": len(ordered),
        "frame_count": total_frames,
        "episode_map": episode_map,
    }


def verify_lerobot_payload(root: Path, *, expected_episodes: int) -> None:
    """Fail closed unless the complete local v2.1 surface is present."""

    required_meta = (
        "info.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "episodes_stats.jsonl",
    )
    missing = [name for name in required_meta if not (root / "meta" / name).is_file()]
    if missing:
        raise ValueError(f"LeRobot v2.1 metadata is incomplete: {missing}")
    parquet = tuple(sorted((root / "data").rglob("*.parquet")))
    if len(parquet) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episode parquet files, found {len(parquet)}"
        )
    for feature_name, _, _ in IMAGE_PATHS:
        videos = tuple(sorted((root / "videos").rglob(f"{feature_name}/*.mp4")))
        if len(videos) != expected_episodes:
            raise ValueError(
                f"expected {expected_episodes} videos for {feature_name}, "
                f"found {len(videos)}"
            )


def read_action_rows(root: Path) -> FloatArray:
    """Read all physical 20D action rows from one completed task repository."""

    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeError("normalization stats require pyarrow") from exc
    parts: list[FloatArray] = []
    for path in sorted((root / "data").rglob("*.parquet")):
        values = parquet.read_table(path, columns=["action"])["action"].to_pylist()
        rows = np.asarray(values, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 20 or not np.isfinite(rows).all():
            raise ValueError(f"invalid 20D action rows in {path}")
        parts.append(rows)
    if not parts:
        raise ValueError(f"no action parquet rows found under {root}")
    return cast(
        FloatArray,
        np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32),
    )


__all__ = [
    "LeRobotSink",
    "convert_episode_to_sink",
    "freeze_episode_action_config",
    "official_feature_schema",
    "read_action_rows",
    "verify_lerobot_payload",
    "write_task_repo",
]

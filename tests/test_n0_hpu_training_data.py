from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "n0_twam"
sys.path.insert(0, str(SCRIPT_ROOT))

Array: TypeAlias = npt.NDArray[Any]

from hpu_training.data.contracts import (  # noqa: E402
    ACTION_PER_FRAME,
    EXPECTED_SPLIT_COUNTS,
    LIFT_CAN_62_SHA256,
    SOURCE_FPS,
    TASKS,
    SourceEpisode,
    build_source_manifest,
    discover_source_episodes,
    sha256_file,
)
from hpu_training.data.episode import (  # noqa: E402
    IMAGE_PATHS,
    ee7_gripper_to_state20,
    inspect_hdf5_root,
    pack_next_step_trajectories,
)
from hpu_training.data.lerobot import (  # noqa: E402
    freeze_episode_action_config,
    official_feature_schema,
)
from hpu_training.data.materialize_train759 import (  # noqa: E402
    freeze_source_manifest,
    materialize_task,
)
from hpu_training.data.stats import action_normalization  # noqa: E402


def _source_grid(root: Path) -> None:
    for task in TASKS:
        clean = root / task / "clean"
        clean.mkdir(parents=True)
        for episode_id in range(100):
            (clean / f"{episode_id}.hdf5").write_bytes(
                f"{task}:{episode_id}".encode("utf-8")
            )


def test_source_manifest_freezes_exact_train759_split(tmp_path: Path) -> None:
    raw = tmp_path / "UniVTAC"
    _source_grid(raw)

    records = discover_source_episodes(raw)
    manifest = build_source_manifest(records, raw_root=raw)

    assert len(records) == 800
    assert manifest["split_counts"] == EXPECTED_SPLIT_COUNTS
    assert manifest["materialized_splits"] == ["train"]
    assert manifest["source_fps"] == 10
    assert manifest["action_per_frame"] == 4
    assert manifest["used_action_channel_ids"] == list(range(10))
    train = [record for record in records if record.split == "train"]
    assert len(train) == 759
    assert sum(record.task == "grasp_classify" for record in train) == 94
    assert all(
        sum(record.task == task for record in train) == 95
        for task in TASKS
        if task != "grasp_classify"
    )
    frozen_ids = {
        record.episode_id
        for record in records
        if record.task == "insert_tube" and record.split == "frozen"
    }
    assert frozen_ids == {0, 1, 2, 3, 5}
    quarantined = [
        record.relative_path for record in records if record.split == "quarantine"
    ]
    assert quarantined == ["grasp_classify/clean/90.hdf5"]

    output = tmp_path / "converted"
    first = freeze_source_manifest(raw_root=raw, output_root=output)
    second = freeze_source_manifest(raw_root=raw, output_root=output)
    assert first == second
    assert first["source_episode_count"] == 800
    assert first["materialized_task_count"] == 0
    assert not (output / "train759").exists()


def test_wxyz_rot6d_and_strict_next_step_action() -> None:
    half = np.float32(np.sqrt(0.5))
    ee = np.asarray(
        (
            (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0),
            (0.4, 0.5, 0.6, half, 0.0, 0.0, half),
            (0.7, 0.8, 0.9, half, half, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    joint = np.zeros((3, 9), dtype=np.float32)
    joint[:, 7] = np.asarray((0.01, 0.02, 0.03), dtype=np.float32)

    state, action = pack_next_step_trajectories(ee, joint)

    assert state.shape == action.shape == (2, 20)
    np.testing.assert_allclose(state[0, 3:9], (1, 0, 0, 0, 1, 0), atol=1e-6)
    np.testing.assert_allclose(action[0, 3:9], (0, 1, 0, -1, 0, 0), atol=1e-6)
    np.testing.assert_array_equal(action[0], ee7_gripper_to_state20(ee[1], joint[1, 7]))
    assert state[0, 9] == np.float32(0.01)
    assert action[0, 9] == np.float32(0.02)
    np.testing.assert_array_equal(state[:, 10:], np.zeros((2, 10), np.float32))
    np.testing.assert_array_equal(action[:, 10:], np.zeros((2, 10), np.float32))


class _FakeDataset:
    def __init__(self, values: Array, shape: tuple[int, ...] | None = None) -> None:
        self._values = values
        self.dtype = values.dtype
        self.shape = values.shape if shape is None else shape

    def __getitem__(self, index: object) -> Array:
        if isinstance(index, slice):
            return cast(Array, self._values[index])
        if self.shape != self._values.shape:
            return self._values
        if self._values.ndim == 1:
            return cast(Array, np.asarray(self._values[index]))
        return cast(Array, self._values[0])


def test_lift_can_62_uses_only_pinned_0_228_prefix(tmp_path: Path) -> None:
    steps = np.arange(256, dtype=np.int64) + 367
    steps[228:] = np.arange(586, 614, dtype=np.int64)
    ee = np.zeros((256, 7), dtype=np.float32)
    ee[:, 3] = 1.0
    joint = np.zeros((256, 9), dtype=np.float32)
    root: dict[str, _FakeDataset] = {
        "embodiment/ee": _FakeDataset(ee),
        "embodiment/joint": _FakeDataset(joint),
        "step": _FakeDataset(steps),
    }
    for _, path, shape in IMAGE_PATHS:
        root[path] = _FakeDataset(np.zeros(shape, dtype=np.uint8), (256, *shape))
    record = SourceEpisode(
        root=tmp_path,
        relative_path="lift_can/clean/62.hdf5",
        task="lift_can",
        episode_id=62,
        split="train",
        size_bytes=1,
        mtime_ns=1,
        sha256=LIFT_CAN_62_SHA256,
    )

    layout = inspect_hdf5_root(root, record)

    assert (layout.usable_start, layout.usable_stop) == (0, 228)
    assert layout.converted_length == 227


def test_official_feature_schema_is_four_video_abs_ee20() -> None:
    features = official_feature_schema()
    assert features["observation.state"]["shape"] == (20,)
    assert features["action"]["shape"] == (20,)
    assert [name for name, _, _ in IMAGE_PATHS] == [
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.tactile_a",
        "observation.images.tactile_b",
    ]
    assert all(features[name]["dtype"] == "video" for name, _, _ in IMAGE_PATHS)


def test_action_stats_mask_padding_and_floor_dead_active_channels() -> None:
    rows = np.zeros((4, 20), dtype=np.float32)
    rows[:, 0] = (0.1, 0.2, 0.3, 0.4)
    rows[:, 9] = (0.01, 0.02, 0.03, 0.04)

    norm, report = action_normalization(rows)

    q01 = np.asarray(norm["q01"])
    q99 = np.asarray(norm["q99"])
    assert np.all(q99 > q01)
    np.testing.assert_array_equal(q01[10:], np.full(10, -1.0))
    np.testing.assert_array_equal(q99[10:], np.full(10, 1.0))
    assert np.isclose(q01[9], 0.01)
    assert report["used_action_channel_ids"] == list(range(10))


def _fake_task_writer(
    records: Sequence[SourceEpisode], root: Path, _threads: int
) -> dict[str, object]:
    meta = root / "meta"
    meta.mkdir(parents=True)
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        (meta / name).write_text("{}\n", encoding="utf-8")
    episode_map: list[dict[str, object]] = []
    for index, record in enumerate(records):
        data = root / "data" / "chunk-000"
        data.mkdir(parents=True, exist_ok=True)
        (data / f"episode_{index:06d}.parquet").write_bytes(b"fixture")
        for feature_name, _, _ in IMAGE_PATHS:
            video = root / "videos" / "chunk-000" / feature_name
            video.mkdir(parents=True, exist_ok=True)
            (video / f"episode_{index:06d}.mp4").write_bytes(b"fixture")
        episode_map.append(
            {
                "lerobot_episode_index": index,
                "source_episode_id": record.episode_id,
                "source_relative_path": record.relative_path,
                "source_sha256": record.sha256,
                "usable_source_range": None,
                "converted_frame_count": 2,
            }
        )
    return {
        "task": records[0].task,
        "episode_count": len(records),
        "frame_count": 2 * len(records),
        "episode_map": episode_map,
    }


def test_task_materialization_is_no_clobber_and_resumable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source = raw / "insert_tube" / "clean" / "4.hdf5"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    metadata = source.stat()
    record = SourceEpisode(
        root=raw,
        relative_path="insert_tube/clean/4.hdf5",
        task="insert_tube",
        episode_id=4,
        split="train",
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=sha256_file(source),
    )
    output = tmp_path / "converted"
    manifest_sha = "a" * 64

    first = materialize_task(
        task="insert_tube",
        records=(record,),
        output_root=output,
        manifest_sha256=manifest_sha,
        image_writer_threads=1,
        writer=_fake_task_writer,
    )

    def _must_not_run(
        _records: Sequence[SourceEpisode], _root: Path, _threads: int
    ) -> dict[str, object]:
        raise AssertionError("completed task must be validated and skipped")

    second = materialize_task(
        task="insert_tube",
        records=(record,),
        output_root=output,
        manifest_sha256=manifest_sha,
        image_writer_threads=1,
        writer=_must_not_run,
    )
    assert first == second
    assert first["physical_repo_basename"] == "insert_tube"
    assert first["per_repo_norm_key"] == "insert_tube"
    relative_path = first["repo_relative_path"]
    assert isinstance(relative_path, str)
    assert Path(str(output / relative_path)).name == first["per_repo_norm_key"]


def test_action_config_covers_each_whole_episode(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    episode = {
        "episode_index": 0,
        "length": 17,
        "tasks": ["Precision peg-in-hole insertion"],
    }
    (meta / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")

    freeze_episode_action_config(tmp_path)

    result = json.loads((meta / "episodes.jsonl").read_text(encoding="utf-8"))
    assert result["action_config"] == [
        {
            "start_frame": 0,
            "end_frame": 17,
            "action_text": "Precision peg-in-hole insertion",
        }
    ]
    assert SOURCE_FPS == 10
    assert ACTION_PER_FRAME == 4

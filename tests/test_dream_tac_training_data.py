"""Synthetic tests for train-only Dream-Tac UniVTAC materialization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from scripts.dream_tac.training.contracts import (
    DATASET_DIR_NAME,
    VIDEO_SOURCES,
    build_prompt_manifest,
    build_t5_cache_request,
)
from scripts.dream_tac.training.episode_io import export_episode
from scripts.dream_tac.training.kinematics import (
    continuous_quaternions_wxyz,
    pack_dream_tac_trajectories,
    quaternion_sequence_to_unwrapped_euler_xyz,
)
from scripts.dream_tac.training.materialize_train759 import (
    dry_run_plan,
    episode_destination,
    materialize_episode,
)
from scripts.dream_tac.training.statistics import dataset_statistics
from scripts.n0_twam.hpu_training.data.contracts import (
    TASK_PROMPTS,
    TASKS,
    SourceEpisode,
    sha256_file,
)


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    radians = np.deg2rad(degrees)
    return float(np.cos(radians / 2)), 0.0, 0.0, float(np.sin(radians / 2))


def test_wxyz_continuity_euler_unwrap_and_strict_next_row() -> None:
    quaternions = np.asarray(
        (
            _yaw_quaternion(179.0),
            tuple(-value for value in _yaw_quaternion(-179.0)),
            _yaw_quaternion(-177.0),
        ),
        dtype=np.float32,
    )
    continuous = continuous_quaternions_wxyz(quaternions)
    assert np.all(np.sum(continuous[:-1] * continuous[1:], axis=1) >= 0.0)
    euler = quaternion_sequence_to_unwrapped_euler_xyz(quaternions)
    np.testing.assert_allclose(np.rad2deg(euler[:, 2]), (179, 181, 183), atol=1e-3)

    ee = np.zeros((3, 7), dtype=np.float32)
    ee[:, :3] = np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8)), np.float32)
    ee[:, 3:7] = quaternions
    joint = np.zeros((3, 8), dtype=np.float32)
    joint[:, 7] = (0.1, 0.2, 0.3)

    qpos, action = pack_dream_tac_trajectories(ee, joint)

    assert qpos.shape == (2, 6)
    assert action.shape == (2, 7)
    np.testing.assert_array_equal(qpos[0, :3], ee[0, :3])
    np.testing.assert_array_equal(action[0, :3], ee[1, :3])
    assert action[0, 6] == np.float32(0.2)
    assert np.rad2deg(action[0, 5]) == pytest.approx(181.0, abs=1e-3)


def _source_record(tmp_path: Path, *, split: str = "train") -> SourceEpisode:
    raw = tmp_path / "raw"
    source = raw / "insert_tube" / "clean" / "4.hdf5"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic-source")
    metadata = source.stat()
    return SourceEpisode(
        root=raw,
        relative_path="insert_tube/clean/4.hdf5",
        task="insert_tube",
        episode_id=4,
        split=split,  # type: ignore[arg-type]
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=sha256_file(source),
    )


def _synthetic_root() -> dict[str, npt.NDArray[Any]]:
    ee = np.zeros((3, 7), dtype=np.float32)
    ee[:, :3] = ((0, 0, 0), (1, 2, 3), (4, 5, 6))
    ee[:, 3] = 1.0
    joint = np.zeros((3, 8), dtype=np.float32)
    joint[:, 7] = (0.1, 0.2, 0.3)
    root: dict[str, npt.NDArray[Any]] = {
        "embodiment/ee": ee,
        "embodiment/joint": joint,
    }
    shapes = {
        "observation/head/rgb": (270, 480, 3),
        "observation/wrist/rgb": (270, 480, 3),
        "tactile/left_gsmini/rgb_marker": (240, 320, 3),
        "tactile/right_gsmini/rgb_marker": (240, 320, 3),
    }
    for index, (_, source_path) in enumerate(VIDEO_SOURCES, start=1):
        frames = np.empty((3, *shapes[source_path]), dtype=np.uint8)
        frames[0].fill(index * 10)
        frames[1].fill(index * 10 + 1)
        frames[2].fill(index * 10 + 2)
        root[source_path] = frames
    return root


def _root_opener(
    root: Mapping[str, npt.NDArray[Any]],
) -> Any:
    @contextmanager
    def _open(_path: Path) -> Iterator[Mapping[str, Any]]:
        yield root

    return _open


def test_episode_materialization_writes_four_aligned_streams_and_prompt(
    tmp_path: Path,
) -> None:
    record = _source_record(tmp_path)
    root = _synthetic_root()
    video_values: dict[str, list[int]] = {}
    hdf5_values: dict[str, object] = {}

    def _video_writer(frames: Any, path: Path, fps: int) -> int:
        values = list(frames)
        assert fps == 10
        video_values[path.name] = [int(frame[0, 0, 0]) for frame in values]
        path.write_bytes(path.name.encode())
        return len(values)

    def _hdf5_writer(
        path: Path,
        qpos: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32],
        video_paths: Mapping[str, str],
        prompt: str,
        task_id: str,
    ) -> None:
        hdf5_values.update(
            qpos=qpos.copy(),
            action=action.copy(),
            video_paths=dict(video_paths),
            prompt=prompt,
            task_id=task_id,
        )
        path.write_bytes(b"synthetic-hdf5")

    def _artifact_verifier(
        _path: Path, *, prompt: str, task_id: str, converted_length: int
    ) -> None:
        assert prompt == TASK_PROMPTS["insert_tube"]
        assert task_id == "insert_tube"
        assert converted_length == 2

    receipt = materialize_episode(
        record=record,
        output_root=tmp_path / "output",
        source_manifest_sha256="a" * 64,
        video_writer=_video_writer,
        hdf5_writer=_hdf5_writer,
        root_opener=_root_opener(root),
        artifact_verifier=_artifact_verifier,
    )

    destination = episode_destination(tmp_path / "output", record)
    assert destination.is_dir()
    assert destination.parts[-4:-1] == (DATASET_DIR_NAME, "train", "insert_tube")
    assert not (tmp_path / "output" / DATASET_DIR_NAME / "val").exists()
    assert receipt["converted_frame_count"] == 2
    assert receipt["tactile_required"] is True
    assert hdf5_values["prompt"] == TASK_PROMPTS["insert_tube"]
    paths = cast(Mapping[str, str], hdf5_values["video_paths"])
    assert set(paths) == {name for name, _ in VIDEO_SOURCES}
    action = np.asarray(hdf5_values["action"])
    np.testing.assert_array_equal(action[0, :3], (1, 2, 3))
    assert len(video_values) == 4
    assert sorted(video_values.values()) == [[10, 11], [20, 21], [30, 31], [40, 41]]

    def _must_not_write(_frames: Any, _path: Path, _fps: int) -> int:
        raise AssertionError("completed episode must resume without rewriting")

    resumed = materialize_episode(
        record=record,
        output_root=tmp_path / "output",
        source_manifest_sha256="a" * 64,
        video_writer=_must_not_write,
        hdf5_writer=_hdf5_writer,
        root_opener=_root_opener(root),
        artifact_verifier=_artifact_verifier,
    )
    assert resumed == receipt


def test_missing_tactile_is_rejected_and_frozen_source_cannot_publish(
    tmp_path: Path,
) -> None:
    record = _source_record(tmp_path)
    root = _synthetic_root()
    del root["tactile/right_gsmini/rgb_marker"]
    with pytest.raises(ValueError, match="missing UniVTAC HDF5 datasets"):
        export_episode(
            record,
            staging_root=tmp_path / "staging",
            prompt=TASK_PROMPTS[record.task],
            video_writer=lambda _frames, _path, _fps: 2,
            hdf5_writer=lambda *_args: None,
            root_opener=_root_opener(root),
        )

    frozen = _source_record(tmp_path / "frozen", split="frozen")
    with pytest.raises(ValueError, match="non-training"):
        materialize_episode(
            record=frozen,
            output_root=tmp_path / "output",
            source_manifest_sha256="a" * 64,
            artifact_verifier=lambda *_args, **_kwargs: None,
        )


def test_dry_run_selects_train759_without_writing_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw-grid"
    for task in TASKS:
        clean = raw / task / "clean"
        clean.mkdir(parents=True)
        for episode_id in range(100):
            (clean / f"{episode_id}.hdf5").write_bytes(b"source")
    output = tmp_path / "must-not-exist"

    result = dry_run_plan(raw_root=raw, output_root=output)

    assert result["writes_performed"] is False
    assert result["planned_episode_count"] == 759
    assert result["split_counts"] == {"train": 759, "frozen": 40, "quarantine": 1}
    assert result["creates_validation_tree"] is False
    assert not output.exists()


def test_prompt_and_t5_request_bind_all_hdf5_task_names(tmp_path: Path) -> None:
    prompts = build_prompt_manifest("b" * 64)
    request = build_t5_cache_request(prompt_manifest=prompts, output_root=tmp_path)

    assert prompts["t5_cache_keys"] == [TASK_PROMPTS[task] for task in TASKS]
    prompt_items = cast(list[dict[str, object]], prompts["prompts"])
    assert [item["prompt"] for item in prompt_items] == [
        TASK_PROMPTS[task] for task in TASKS
    ]
    assert request["status"] == "external_generation_required"
    assert request["expected_cache_keys"] == prompts["t5_cache_keys"]
    assert request["encoder_runtime"] == {
        "protocol_id": "cosmos_t5_cpu_fp32_active_tokens_v1",
        "device": "cpu",
        "model_compute_dtype": "torch.float32",
        "active_token_batching": True,
        "max_output_tokens": 512,
        "output_dtype": "torch.bfloat16",
        "padding_value": 0.0,
    }
    command = cast(list[str], request["command"])
    assert command[0:3] == [
        "python",
        "-m",
        "scripts.dream_tac.training.t5_cache",
    ]
    assert not (tmp_path / DATASET_DIR_NAME / "t5_embeddings.pkl").exists()


def test_statistics_match_official_action_and_proprio_fields() -> None:
    action = np.arange(21, dtype=np.float32).reshape(3, 7)
    proprio = np.arange(18, dtype=np.float32).reshape(3, 6)

    stats = dataset_statistics(action_rows=action, proprio_rows=proprio)

    assert set(stats) == {
        "actions_min",
        "actions_max",
        "actions_mean",
        "actions_std",
        "actions_median",
        "proprio_min",
        "proprio_max",
        "proprio_mean",
        "proprio_std",
        "proprio_median",
    }
    actions_mean = cast(list[float], stats["actions_mean"])
    proprio_median = cast(list[float], stats["proprio_median"])
    np.testing.assert_allclose(actions_mean, np.mean(action, axis=0))
    np.testing.assert_allclose(proprio_median, np.median(proprio, axis=0))

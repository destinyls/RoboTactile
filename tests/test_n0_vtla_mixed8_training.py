import json
from pathlib import Path

import numpy as np
import pytest

from scripts.n0_vtla.hpu_training.materialize_mixed8 import (
    MIXED_VIEW_PROTOCOL,
    _replace_index_columns,
    _validate_episode_metadata,
)
from scripts.n0_vtla.hpu_training.multitask_sampler import (
    TASKS,
    UniformTaskDistributedSampler,
    _canonical_sha256,
    _load_ranges,
)


class _Dataset:
    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


def test_mixed_parquet_rewrites_only_global_indices(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    state = pa.FixedSizeListArray.from_arrays(
        pa.array(np.arange(24, dtype=np.float32)), 8
    )
    table = pa.table(
        {
            "observation.state": state,
            "episode_index": pa.array([2, 2, 2], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "index": pa.array([7, 8, 9], type=pa.int64()),
            "task_index": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, source)

    count = _replace_index_columns(
        source,
        destination,
        local_episode=2,
        local_frame_start=7,
        global_episode=94,
        global_frame_start=4956,
        task_index=1,
    )

    result = pq.read_table(destination)
    assert count == 3
    assert result["episode_index"].to_pylist() == [94, 94, 94]
    assert result["index"].to_pylist() == [4956, 4957, 4958]
    assert result["task_index"].to_pylist() == [1, 1, 1]
    assert result["frame_index"].to_pylist() == [0, 1, 2]
    assert (
        result["observation.state"].to_pylist()
        == table["observation.state"].to_pylist()
    )


def test_episode_metadata_requires_the_task_prompt() -> None:
    prompt = "Precision peg-in-hole insertion"
    row = {
        "episode_index": 0,
        "tasks": [prompt],
        "length": 3,
        "action_config": [{"action_text": prompt}],
    }

    assert _validate_episode_metadata(row, episode=0, prompt=prompt) == 3
    row["tasks"] = ["wrong task"]
    try:
        _validate_episode_metadata(row, episode=0, prompt=prompt)
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("a mismatched task prompt must be rejected")


def test_mixed_receipt_ranges_are_contiguous_and_train_only(tmp_path: Path) -> None:
    ranges = {task: [index * 10, (index + 1) * 10] for index, task in enumerate(TASKS)}
    payload = {
        "protocol_id": MIXED_VIEW_PROTOCOL,
        "status": "complete",
        "tasks": list(TASKS),
        "source_split": "train",
        "validation_data_used": False,
        "frame_count": 80,
        "task_frame_ranges": ranges,
    }
    payload["view_receipt_sha256"] = _canonical_sha256(payload)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_ranges(path) == tuple(
        (index * 10, (index + 1) * 10) for index in range(8)
    )


def test_balanced_sampler_has_one_task_per_rank_per_microstep() -> None:
    ranges = tuple((index * 10, (index + 1) * 10) for index in range(8))
    samples = [
        list(
            UniformTaskDistributedSampler(
                _Dataset(80),
                task_ranges=ranges,
                num_replicas=8,
                rank=rank,
                seed=7,
                drop_last=True,
            )
        )
        for rank in range(8)
    ]

    assert all(len(rank_samples) == 10 for rank_samples in samples)
    for microstep in range(10):
        task_indices = sorted(samples[rank][microstep] // 10 for rank in range(8))
        assert task_indices == list(range(8))
    for rank in range(8):
        assert [value // 10 for value in samples[rank][:8]] == [
            (rank + microstep) % 8 for microstep in range(8)
        ]


def test_eight_rank_sampler_preserves_legacy_sample_order() -> None:
    ranges = tuple((index * 10, (index + 1) * 10) for index in range(8))
    streams = [
        [
            task_index * 10 + int(value)
            for value in np.random.default_rng(7 + task_index).permutation(10)
        ]
        for task_index in range(8)
    ]

    for rank in range(8):
        actual = list(
            UniformTaskDistributedSampler(
                _Dataset(80),
                task_ranges=ranges,
                num_replicas=8,
                rank=rank,
                seed=7,
                drop_last=True,
            )
        )
        expected = [
            streams[(rank + microstep) % 8][microstep] for microstep in range(10)
        ]
        assert actual == expected


def test_32_rank_sampler_has_four_distinct_samples_per_task() -> None:
    widths = (5, 37, 41, 43, 47, 53, 59, 61)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for width in widths:
        ranges.append((cursor, cursor + width))
        cursor += width
    task_ranges = tuple(ranges)
    num_replicas = 32
    samples = [
        list(
            UniformTaskDistributedSampler(
                _Dataset(cursor),
                task_ranges=task_ranges,
                num_replicas=num_replicas,
                rank=rank,
                seed=13,
                drop_last=True,
            )
        )
        for rank in range(num_replicas)
    ]

    samples_per_rank = cursor // num_replicas
    assert all(len(rank_samples) == samples_per_rank for rank_samples in samples)
    for microstep in range(samples_per_rank):
        cohorts: list[list[int]] = [[] for _ in TASKS]
        for rank in range(num_replicas):
            task_index = (rank + microstep) % len(TASKS)
            sample = samples[rank][microstep]
            start, stop = task_ranges[task_index]
            assert start <= sample < stop
            cohorts[task_index].append(sample)
        assert all(len(cohort) == 4 for cohort in cohorts)
        assert all(len(set(cohort)) == 4 for cohort in cohorts)


def test_8_and_32_rank_updates_use_the_same_task_sample_sets() -> None:
    ranges = tuple((index * 80, (index + 1) * 80) for index in range(8))
    dataset = _Dataset(640)

    samples_8 = [
        list(
            UniformTaskDistributedSampler(
                dataset,
                task_ranges=ranges,
                num_replicas=8,
                rank=rank,
                seed=17,
                drop_last=True,
            )
        )
        for rank in range(8)
    ]
    samples_32 = [
        list(
            UniformTaskDistributedSampler(
                dataset,
                task_ranges=ranges,
                num_replicas=32,
                rank=rank,
                seed=17,
                drop_last=True,
            )
        )
        for rank in range(32)
    ]

    update_8 = {
        samples_8[rank][microstep] for microstep in range(8) for rank in range(8)
    }
    update_32 = {
        samples_32[rank][microstep] for microstep in range(2) for rank in range(32)
    }
    assert len(update_8) == 64
    assert update_32 == update_8


def test_balanced_sampler_rejects_unsupported_replica_count() -> None:
    ranges = tuple((index * 10, (index + 1) * 10) for index in range(8))

    with pytest.raises(ValueError, match="requires 8 or 32 DDP ranks"):
        UniformTaskDistributedSampler(
            _Dataset(80),
            task_ranges=ranges,
            num_replicas=16,
            rank=0,
            seed=7,
            drop_last=True,
        )

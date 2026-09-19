"""Exact task-balanced DDP sampling for the certified mixed8 dataset."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections import deque
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any, Final

import numpy as np

DATA_LOADER_MODULE: Final[str] = "n0vtla.training.data_loader"
DATA_LOADER_SHA256: Final[str] = (
    "5b26b7c6393e306bab1a4e557fecc45381675fee3a43bc5fc9744c5831793d54"
)
MIXED_VIEW_PROTOCOL: Final[str] = (
    "robotactile.n0_vtla.univtac_qpos8_mixed8_train_only.v1"
)
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
SUPPORTED_REPLICA_COUNTS: Final[tuple[int, ...]] = (8, 32)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_ranges(path: Path) -> tuple[tuple[int, int], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mixed8 receipt must be a JSON object")
    unsigned = dict(payload)
    claimed = unsigned.pop("view_receipt_sha256", None)
    ranges = payload.get("task_frame_ranges")
    if (
        claimed != _canonical_sha256(unsigned)
        or payload.get("protocol_id") != MIXED_VIEW_PROTOCOL
        or payload.get("status") != "complete"
        or payload.get("tasks") != list(TASKS)
        or payload.get("source_split") != "train"
        or payload.get("validation_data_used") is not False
        or not isinstance(ranges, dict)
    ):
        raise ValueError("mixed8 receipt does not certify task-balanced training")
    parsed: list[tuple[int, int]] = []
    cursor = 0
    for task in TASKS:
        pair = ranges.get(task)
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in pair
            )
            or pair[0] != cursor
            or pair[1] <= pair[0]
        ):
            raise ValueError(f"invalid mixed8 task frame range: {task}")
        parsed.append((pair[0], pair[1]))
        cursor = pair[1]
    if cursor != payload.get("frame_count"):
        raise ValueError("mixed8 frame ranges do not cover the dataset")
    return tuple(parsed)


class UniformTaskDistributedSampler:
    """Give each physical microbatch equal, distinct samples from every task."""

    def __init__(
        self,
        dataset: Sized,
        *,
        task_ranges: tuple[tuple[int, int], ...],
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool,
    ) -> None:
        if num_replicas not in SUPPORTED_REPLICA_COUNTS:
            raise ValueError("mixed8 balanced sampling requires 8 or 32 DDP ranks")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"invalid DDP rank: {rank}")
        if not drop_last:
            raise ValueError("mixed8 balanced sampling requires drop_last=True")
        if len(task_ranges) != len(TASKS):
            raise ValueError(
                f"mixed8 sampling requires {len(TASKS)} task ranges, "
                f"got {len(task_ranges)}"
            )
        if len(dataset) != task_ranges[-1][1]:
            raise ValueError(
                f"mixed8 dataset/receipt length mismatch: {len(dataset)} != {task_ranges[-1][1]}"
            )
        self._task_ranges = task_ranges
        self._num_replicas = num_replicas
        self._rank = rank
        self._seed = seed
        self._epoch = 0
        self._num_samples = len(dataset) // num_replicas
        self._cohort_size = num_replicas // len(TASKS)
        for task_index, (start, stop) in enumerate(task_ranges):
            if stop - start < self._cohort_size:
                raise ValueError(
                    f"mixed8 task range {task_index} has fewer samples than "
                    f"its per-microbatch cohort: {stop - start} < {self._cohort_size}"
                )

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self._epoch = epoch

    def _stream(self, task_index: int, generator: np.random.Generator) -> list[int]:
        start, stop = self._task_ranges[task_index]
        width = stop - start
        target_length = self._num_samples * self._cohort_size
        result: list[int] = []
        pending: deque[int] = deque()
        while len(result) < target_length:
            cohort: list[int] = []
            deferred: list[int] = []
            while len(cohort) < self._cohort_size:
                if not pending:
                    pending.extend(int(value) for value in generator.permutation(width))
                candidate = pending.popleft()
                if candidate in cohort:
                    deferred.append(candidate)
                else:
                    cohort.append(candidate)
            pending.extendleft(reversed(deferred))
            result.extend(start + value for value in cohort)
        return result

    def __iter__(self) -> Iterator[int]:
        epoch = self._epoch
        self._epoch += 1
        streams: list[list[int]] = []
        for task_index in range(len(TASKS)):
            generator = np.random.default_rng(
                self._seed + epoch * len(TASKS) + task_index
            )
            streams.append(self._stream(task_index, generator))
        return iter(
            streams[(self._rank + microstep) % len(TASKS)][
                microstep * self._cohort_size + self._rank // len(TASKS)
            ]
            for microstep in range(self._num_samples)
        )


def install_uniform_task_sampler() -> Path:
    """Install a source-bound factory before the upstream trainer is imported."""

    receipt_value = os.environ.get("ROBOTACTILE_N0_VTLA_MIXED_RECEIPT")
    if not receipt_value:
        raise RuntimeError("mixed8 receipt path is not configured")
    receipt_path = Path(receipt_value).resolve(strict=True)
    task_ranges = _load_ranges(receipt_path)
    spec = importlib.util.find_spec(DATA_LOADER_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {DATA_LOADER_MODULE}")
    source_path = Path(spec.origin).resolve(strict=True)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != DATA_LOADER_SHA256:
        raise RuntimeError(f"N0-VTLA data loader SHA256 mismatch: {digest}")
    import torch  # type: ignore[import-not-found]

    original_sampler = torch.utils.data.distributed.DistributedSampler

    def sampler_factory(
        dataset: Sized,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> Any:
        replicas = (
            torch.distributed.get_world_size() if num_replicas is None else num_replicas
        )
        current_rank = torch.distributed.get_rank() if rank is None else rank
        if not shuffle:
            return original_sampler(
                dataset,
                num_replicas=replicas,
                rank=current_rank,
                shuffle=False,
                seed=seed,
                drop_last=drop_last,
            )
        return UniformTaskDistributedSampler(
            dataset,
            task_ranges=task_ranges,
            num_replicas=replicas,
            rank=current_rank,
            seed=seed,
            drop_last=drop_last,
        )

    torch.utils.data.distributed.DistributedSampler = sampler_factory
    return source_path

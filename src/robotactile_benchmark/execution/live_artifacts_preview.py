"""Bounded, deterministic observation subsets for live artifact previews."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Tuple

import numpy as np

from robotactile_benchmark.closed_loop.delivery import DeliveryFinalization
from robotactile_benchmark.contracts import Array, EvaluationRecord
from robotactile_benchmark.execution.capture_profiles import (
    DEFAULT_PREVIEW_RECORD_LIMIT,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LiveArtifactValidationError,
)
from robotactile_benchmark.execution.live_artifacts_fs import LiveBundleSnapshot
from robotactile_benchmark.execution.live_artifacts_io import LiveArrayWriter
from robotactile_benchmark.execution.live_artifacts_records import (
    evaluation_record_from_live_dict,
    evaluation_record_to_live_dict,
)

LIVE_PREVIEW_TRACE_SEMANTIC_VERSION = "1.0"
_PREVIEW_FIELDS = frozenset(
    {
        "selected_indices",
        "clean_records",
        "delivered_records",
        "semantic_version",
    }
)


@dataclass(frozen=True)
class LivePreviewTrace:
    """A bounded subset of paired records, never a full-trace substitute."""

    selected_indices: Tuple[int, ...]
    clean_records: Tuple[EvaluationRecord, ...]
    delivered_records: Tuple[EvaluationRecord, ...]
    semantic_version: str = LIVE_PREVIEW_TRACE_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        indices = tuple(self.selected_indices)
        clean = tuple(self.clean_records)
        delivered = tuple(self.delivered_records)
        if self.semantic_version != LIVE_PREVIEW_TRACE_SEMANTIC_VERSION:
            raise ValueError("live preview trace semantic version mismatch")
        if not (
            len(indices) == len(clean) == len(delivered)
            and len(indices) <= DEFAULT_PREVIEW_RECORD_LIMIT
        ):
            raise ValueError("live preview trace lengths are invalid")
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, Integral)
            or int(index) < 0
            for index in indices
        ):
            raise TypeError("live preview indices must be non-negative integers")
        normalized_indices = tuple(int(index) for index in indices)
        if any(
            current <= previous
            for previous, current in zip(normalized_indices, normalized_indices[1:])
        ):
            raise ValueError("live preview indices must be strictly increasing")
        if not all(isinstance(record, EvaluationRecord) for record in clean):
            raise TypeError(
                "live preview clean records must be EvaluationRecord values"
            )
        if not all(isinstance(record, EvaluationRecord) for record in delivered):
            raise TypeError(
                "live preview delivered records must be EvaluationRecord values"
            )
        for index, clean_record, delivered_record in zip(
            normalized_indices, clean, delivered
        ):
            clean_observation = clean_record.observation
            delivered_observation = delivered_record.observation
            if (
                clean_observation.step_index != index
                or delivered_observation.step_index != index
                or clean_observation.episode_id != delivered_observation.episode_id
                or clean_observation.task != delivered_observation.task
                or clean_observation.seed != delivered_observation.seed
                or clean_record.clean_record_sha256
                != delivered_record.clean_record_sha256
            ):
                raise ValueError("live preview record pairing is inconsistent")
        object.__setattr__(self, "selected_indices", normalized_indices)
        object.__setattr__(self, "clean_records", clean)
        object.__setattr__(self, "delivered_records", delivered)


def select_preview_indices(
    finalization: DeliveryFinalization | None,
    limit: int = DEFAULT_PREVIEW_RECORD_LIMIT,
) -> Tuple[int, ...]:
    """Select endpoints, state transitions, then evenly distributed fill frames."""

    _validate_limit(limit)
    if finalization is None:
        return ()
    if not isinstance(finalization, DeliveryFinalization):
        raise TypeError("preview selection requires DeliveryFinalization or None")
    record_count = len(finalization.clean_records)
    target = min(record_count, limit)
    if record_count > 1 and target < 2:
        raise ValueError("preview limit must preserve both endpoints")
    if target == record_count:
        return tuple(range(record_count))

    selected = {0, record_count - 1}
    transition_candidates: set[int] = set()
    previous = _pair_signature(
        finalization.clean_records[0], finalization.delivered_records[0]
    )
    for index in range(1, record_count):
        current = _pair_signature(
            finalization.clean_records[index],
            finalization.delivered_records[index],
        )
        if current != previous:
            transition_candidates.update((index - 1, index))
        previous = current

    available_transition_slots = target - len(selected)
    transition_pool = tuple(sorted(transition_candidates - selected))
    selected.update(
        _evenly_pick(
            transition_pool,
            min(available_transition_slots, len(transition_pool)),
        )
    )
    while len(selected) < target:
        selected.add(_farthest_unselected(record_count, selected))
    return tuple(sorted(selected))


def build_live_preview_trace(
    finalization: DeliveryFinalization | None,
    limit: int = DEFAULT_PREVIEW_RECORD_LIMIT,
) -> LivePreviewTrace:
    """Build one bounded preview without changing full-trace hash semantics."""

    indices = select_preview_indices(finalization, limit)
    if finalization is None:
        return LivePreviewTrace((), (), ())
    return LivePreviewTrace(
        selected_indices=indices,
        clean_records=tuple(finalization.clean_records[index] for index in indices),
        delivered_records=tuple(
            finalization.delivered_records[index] for index in indices
        ),
    )


def preview_trace_to_live_dict(
    trace: LivePreviewTrace, arrays: LiveArrayWriter
) -> dict[str, object]:
    """Encode one bounded preview using the live content-addressed array store."""

    if not isinstance(trace, LivePreviewTrace):
        raise TypeError("preview encoder requires LivePreviewTrace")
    if not isinstance(arrays, LiveArrayWriter):
        raise TypeError("preview encoder requires LiveArrayWriter")
    return {
        "selected_indices": list(trace.selected_indices),
        "clean_records": [
            evaluation_record_to_live_dict(record, arrays)
            for record in trace.clean_records
        ],
        "delivered_records": [
            evaluation_record_to_live_dict(record, arrays)
            for record in trace.delivered_records
        ],
        "semantic_version": trace.semantic_version,
    }


def preview_trace_from_live_dict(
    value: object,
    snapshot: LiveBundleSnapshot,
    referenced_paths: set[str],
) -> LivePreviewTrace:
    """Strictly reconstruct one bounded preview and all referenced arrays."""

    if not isinstance(value, dict) or set(value) != _PREVIEW_FIELDS:
        raise LiveArtifactValidationError("live preview trace fields mismatch")
    if value["semantic_version"] != LIVE_PREVIEW_TRACE_SEMANTIC_VERSION:
        raise LiveArtifactValidationError("live preview trace version mismatch")
    raw_indices = value["selected_indices"]
    raw_clean = value["clean_records"]
    raw_delivered = value["delivered_records"]
    if (
        not isinstance(raw_indices, list)
        or not isinstance(raw_clean, list)
        or not isinstance(raw_delivered, list)
        or len(raw_indices) > DEFAULT_PREVIEW_RECORD_LIMIT
        or len(raw_indices) != len(raw_clean)
        or len(raw_indices) != len(raw_delivered)
    ):
        raise LiveArtifactValidationError("live preview trace lengths are invalid")
    indices = tuple(_preview_index(item) for item in raw_indices)
    return LivePreviewTrace(
        selected_indices=indices,
        clean_records=tuple(
            evaluation_record_from_live_dict(item, snapshot, referenced_paths)
            for item in raw_clean
        ),
        delivered_records=tuple(
            evaluation_record_from_live_dict(item, snapshot, referenced_paths)
            for item in raw_delivered
        ),
        semantic_version=LIVE_PREVIEW_TRACE_SEMANTIC_VERSION,
    )


def iter_preview_arrays(trace: LivePreviewTrace) -> Iterable[Array]:
    """Yield exactly the arrays referenced by one preview trace."""

    if not isinstance(trace, LivePreviewTrace):
        raise TypeError("preview array iteration requires LivePreviewTrace")
    for record in (*trace.clean_records, *trace.delivered_records):
        observation = record.observation
        for sensor in observation.tactile:
            if sensor.payload is not None:
                yield sensor.payload
        for name in sorted(observation.vision):
            yield observation.vision[name]
        yield observation.proprio


def _validate_limit(limit: int) -> None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= DEFAULT_PREVIEW_RECORD_LIMIT
    ):
        raise ValueError(
            f"preview limit must be an integer in [1,{DEFAULT_PREVIEW_RECORD_LIMIT}]"
        )


def _pair_signature(
    clean_record: EvaluationRecord, delivered_record: EvaluationRecord
) -> tuple[object, ...]:
    return (
        _provenance_signature(clean_record),
        _provenance_signature(delivered_record),
    )


def _provenance_signature(record: EvaluationRecord) -> tuple[object, ...]:
    return tuple(
        sorted(
            (
                item.slot_id,
                item.phase.value,
                tuple(sorted(item.active_fault_ids)),
            )
            for item in record.provenance
        )
    )


def _evenly_pick(values: Sequence[int], count: int) -> Tuple[int, ...]:
    if count <= 0:
        return ()
    if count >= len(values):
        return tuple(values)
    if count == 1:
        return (values[len(values) // 2],)
    denominator = count - 1
    maximum = len(values) - 1
    return tuple(
        values[(index * maximum + denominator // 2) // denominator]
        for index in range(count)
    )


def _farthest_unselected(record_count: int, selected: set[int]) -> int:
    candidates = (index for index in range(record_count) if index not in selected)
    return max(
        candidates,
        key=lambda index: (
            min(abs(index - chosen) for chosen in selected),
            -index,
        ),
    )


def _preview_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LiveArtifactValidationError(
            "live preview index must be a non-negative integer"
        )
    return value


__all__ = [
    "DEFAULT_PREVIEW_RECORD_LIMIT",
    "LIVE_PREVIEW_TRACE_SEMANTIC_VERSION",
    "LivePreviewTrace",
    "build_live_preview_trace",
    "iter_preview_arrays",
    "preview_trace_from_live_dict",
    "preview_trace_to_live_dict",
    "select_preview_indices",
]

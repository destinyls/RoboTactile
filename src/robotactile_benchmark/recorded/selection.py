"""Deterministic contact-anchor selection for recorded UniVTAC HDF5."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_registry import build_config
from robotactile_benchmark.recorded.source import ACTION_HORIZON, RecordedSourceError

_LEFT_DEPTH = "tactile/left_gsmini/depth"
_RIGHT_DEPTH = "tactile/right_gsmini/depth"


@dataclass(frozen=True)
class RecordedAnchorSelection:
    """Strongest-contact anchor plus a loader-compatible reference frame."""

    anchor_index: int
    rest_index: int
    total_record_count: int
    contact_mode: str
    left_indentation_mm: float
    right_indentation_mm: float
    rest_mode: str = "bilateral_free"

    def __post_init__(self) -> None:
        if self.contact_mode not in {"bilateral", "unilateral_fallback"}:
            raise RecordedSourceError("unknown recorded contact-selection mode")
        if self.rest_mode not in {
            "bilateral_free",
            "earliest_available_fallback",
        }:
            raise RecordedSourceError("unknown recorded rest-selection mode")
        if not 0 <= self.rest_index <= self.anchor_index < self.total_record_count:
            raise RecordedSourceError("recorded anchor/rest ordering is invalid")
        if self.anchor_index + ACTION_HORIZON > self.total_record_count:
            raise RecordedSourceError("recorded anchor lacks a complete N0 horizon")
        if min(self.left_indentation_mm, self.right_indentation_mm) < 0.0:
            raise RecordedSourceError("recorded indentation must be non-negative")


def _h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as error:
        raise RecordedSourceError(
            "recorded UniVTAC selection requires h5py in the selected runtime"
        ) from error


def _depth_arrays(root: Any) -> tuple[Any, Any, int]:
    try:
        left = root[_LEFT_DEPTH]
        right = root[_RIGHT_DEPTH]
    except KeyError as error:
        raise RecordedSourceError(
            "required tactile depth dataset is missing"
        ) from error
    if left.dtype != np.dtype(np.float32) or right.dtype != np.dtype(np.float32):
        raise RecordedSourceError("tactile depth datasets must use float32")
    if len(left.shape) != 3 or tuple(left.shape[1:]) != (240, 320):
        raise RecordedSourceError("left tactile depth must have shape [T,240,320]")
    if tuple(right.shape) != tuple(left.shape):
        raise RecordedSourceError("left/right tactile depth shapes must match")
    count = int(left.shape[0])
    if count < ACTION_HORIZON:
        raise RecordedSourceError("HDF5 episode is shorter than the N0 horizon")
    return left, right, count


def select_univtac_contact_anchor(
    path: Path,
    *,
    task_id: str,
) -> RecordedAnchorSelection:
    """Select strongest bilateral contact, with a unilateral fallback."""

    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        raise RecordedSourceError("HDF5 source must be a non-symlink regular file")
    config = build_config(task_id, action_spec=EE8_ACTION_SPEC)
    far_plane_mm = config.aliases.far_plane_mm
    threshold_mm = config.phase_tracker.on_threshold_mm
    with _h5py().File(source, "r") as root:
        left_depth, right_depth, total_count = _depth_arrays(root)
        valid_count = total_count - ACTION_HORIZON + 1
        left = np.empty(valid_count, dtype=np.float32)
        right = np.empty(valid_count, dtype=np.float32)
        for index in range(valid_count):
            left[index] = max(
                0.0, far_plane_mm - float(np.asarray(left_depth[index]).min())
            )
            right[index] = max(
                0.0, far_plane_mm - float(np.asarray(right_depth[index]).min())
            )
    bilateral = np.flatnonzero((left >= threshold_mm) & (right >= threshold_mm))
    if bilateral.size:
        strength = np.minimum(left[bilateral], right[bilateral])
        anchor_index = int(bilateral[int(np.argmax(strength))])
        contact_mode = "bilateral"
    else:
        strength = np.maximum(left, right)
        anchor_index = int(np.argmax(strength))
        if float(strength[anchor_index]) < threshold_mm:
            raise RecordedSourceError(
                "episode has no tactile contact before the final N0 horizon"
            )
        contact_mode = "unilateral_fallback"
    free = np.flatnonzero(
        (left[:anchor_index] < threshold_mm) & (right[:anchor_index] < threshold_mm)
    )
    if free.size:
        rest_index = int(free[0])
        rest_mode = "bilateral_free"
    else:
        # The tactile-causal diagnostic does not consume a rest reference. Some
        # official episodes begin in contact, so retain the earliest frame only
        # to satisfy the generic recorded-episode loader and disclose the fallback.
        rest_index = 0
        rest_mode = "earliest_available_fallback"
    return RecordedAnchorSelection(
        anchor_index=anchor_index,
        rest_index=rest_index,
        total_record_count=total_count,
        contact_mode=contact_mode,
        left_indentation_mm=float(left[anchor_index]),
        right_indentation_mm=float(right[anchor_index]),
        rest_mode=rest_mode,
    )


__all__ = ["RecordedAnchorSelection", "select_univtac_contact_anchor"]

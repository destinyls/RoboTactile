"""Official-format train-only action/proprio statistics for Dream-Tac."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from scripts.n0_twam.hpu_training.data.contracts import SourceEpisode

from .episode_io import RootOpener, open_hdf5_root, read_dream_tac_trajectories

_STATISTIC_FIELDS: Final[tuple[str, ...]] = (
    "min",
    "max",
    "mean",
    "std",
    "median",
)
_FEATURE_DIMENSIONS: Final[tuple[tuple[str, int], ...]] = (
    ("actions", 7),
    ("proprio", 6),
)


def _scale_locations(
    values: Sequence[float], minimum: Sequence[float], span: Sequence[float]
) -> list[float]:
    return [
        2.0 * ((values[index] - minimum[index]) / span[index]) - 1.0
        for index in range(len(values))
    ]


def validate_dataset_statistics(
    payload: Mapping[str, object],
) -> dict[str, list[float]]:
    """Validate the exact Franka action/proprio statistics contract."""

    expected = {
        f"{feature}_{statistic}"
        for feature, _ in _FEATURE_DIMENSIONS
        for statistic in _STATISTIC_FIELDS
    }
    if set(payload) != expected:
        raise ValueError("Dream-Tac statistics fields do not match Franka contract")
    validated: dict[str, list[float]] = {}
    for feature, dimension in _FEATURE_DIMENSIONS:
        for statistic in _STATISTIC_FIELDS:
            name = f"{feature}_{statistic}"
            raw_values = payload[name]
            if not isinstance(raw_values, list) or len(raw_values) != dimension:
                raise ValueError(
                    f"Dream-Tac statistics {name} must have dimension {dimension}"
                )
            values: list[float] = []
            for value in raw_values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Dream-Tac statistics {name} must contain numbers"
                    )
                converted = float(value)
                if not math.isfinite(converted):
                    raise ValueError(f"Dream-Tac statistics {name} must be finite")
                values.append(converted)
            validated[name] = values
        minimum = validated[f"{feature}_min"]
        maximum = validated[f"{feature}_max"]
        if any(maximum[index] <= minimum[index] for index in range(dimension)):
            raise ValueError(
                f"Dream-Tac statistics {feature} require max greater than min"
            )
    return validated


def derive_post_normalization_statistics(
    source_statistics: Mapping[str, object],
) -> dict[str, list[float]]:
    """Derive statistics after upstream's exact ``[-1,+1]`` min-max map."""

    source = validate_dataset_statistics(source_statistics)
    derived: dict[str, list[float]] = {}
    for feature, dimension in _FEATURE_DIMENSIONS:
        minimum = source[f"{feature}_min"]
        maximum = source[f"{feature}_max"]
        span = [maximum[index] - minimum[index] for index in range(dimension)]

        derived[f"{feature}_min"] = [-1.0] * dimension
        derived[f"{feature}_max"] = [1.0] * dimension
        derived[f"{feature}_mean"] = _scale_locations(
            source[f"{feature}_mean"], minimum, span
        )
        derived[f"{feature}_std"] = [
            2.0 * (source[f"{feature}_std"][index] / span[index])
            for index in range(dimension)
        ]
        derived[f"{feature}_median"] = _scale_locations(
            source[f"{feature}_median"], minimum, span
        )
    return validate_dataset_statistics(derived)


def validate_post_normalization_statistics(
    source_statistics: Mapping[str, object],
    post_statistics: Mapping[str, object],
) -> dict[str, list[float]]:
    """Require a post-normalization artifact derived only from source stats."""

    expected = derive_post_normalization_statistics(source_statistics)
    actual = validate_dataset_statistics(post_statistics)
    if actual != expected:
        raise ValueError(
            "Dream-Tac post-normalization statistics do not match source statistics"
        )
    return actual


def dataset_statistics(
    *, action_rows: npt.ArrayLike, proprio_rows: npt.ArrayLike
) -> dict[str, list[float]]:
    """Match Dream-Tac's ``calculate_dataset_statistics`` JSON fields."""

    actions = np.asarray(action_rows)
    proprio = np.asarray(proprio_rows)
    if (
        actions.dtype != np.float32
        or actions.ndim != 2
        or actions.shape[1] != 7
        or actions.shape[0] == 0
        or not np.isfinite(actions).all()
    ):
        raise ValueError("Dream-Tac statistics actions must be finite float32 [N,7]")
    if (
        proprio.dtype != np.float32
        or proprio.shape != (actions.shape[0], 6)
        or not np.isfinite(proprio).all()
    ):
        raise ValueError("Dream-Tac statistics proprio must be finite float32 [N,6]")

    def _values(array: npt.NDArray[np.float32], operation: str) -> list[float]:
        function = getattr(np, operation)
        result = function(array, axis=0)
        return [float(value) for value in result]

    result: dict[str, object] = {
        "actions_min": _values(actions, "min"),
        "actions_max": _values(actions, "max"),
        "actions_mean": _values(actions, "mean"),
        "actions_std": _values(actions, "std"),
        "actions_median": _values(actions, "median"),
        "proprio_min": _values(proprio, "min"),
        "proprio_max": _values(proprio, "max"),
        "proprio_mean": _values(proprio, "mean"),
        "proprio_std": _values(proprio, "std"),
        "proprio_median": _values(proprio, "median"),
    }
    return validate_dataset_statistics(result)


def statistics_from_records(
    records: Sequence[SourceEpisode],
    *,
    root_opener: RootOpener = open_hdf5_root,
) -> tuple[dict[str, list[float]], int]:
    """Compute statistics exclusively from already-selected train records."""

    if len(records) != 759 or any(record.split != "train" for record in records):
        raise ValueError("Dream-Tac statistics require exact train759 records")
    proprio_parts: list[npt.NDArray[np.float32]] = []
    action_parts: list[npt.NDArray[np.float32]] = []
    for record in records:
        proprio, action = read_dream_tac_trajectories(record, root_opener=root_opener)
        proprio_parts.append(proprio)
        action_parts.append(action)
    proprio = np.concatenate(proprio_parts, axis=0)
    actions = np.concatenate(action_parts, axis=0)
    return (
        dataset_statistics(action_rows=actions, proprio_rows=proprio),
        int(actions.shape[0]),
    )


__all__ = [
    "dataset_statistics",
    "derive_post_normalization_statistics",
    "statistics_from_records",
    "validate_dataset_statistics",
    "validate_post_normalization_statistics",
]

"""Numerical contracts for N0-TWAM train-versus-serve latent parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from robotactile_benchmark.contracts import Array


@dataclass(frozen=True)
class ArrayParityMetrics:
    """Scale-aware errors between two tensors with identical semantics."""

    shape: Tuple[int, ...]
    mean_abs: float
    root_mean_square: float
    max_abs: float
    relative_l2: float
    cosine_similarity: float
    reference_mean: float
    reference_std: float
    candidate_mean: float
    candidate_std: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe deterministic representation."""

        return {
            "candidate_mean": self.candidate_mean,
            "candidate_std": self.candidate_std,
            "cosine_similarity": self.cosine_similarity,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "reference_mean": self.reference_mean,
            "reference_std": self.reference_std,
            "relative_l2": self.relative_l2,
            "root_mean_square": self.root_mean_square,
            "shape": list(self.shape),
        }


def temporal_chunk_slices(frame_count: int) -> Tuple[Tuple[int, int], ...]:
    """Partition a Wan-compatible clip like the live cold/warm VAE stream.

    The server encodes one cold frame, followed by four RGB keyframes per latent
    frame. Consequently, a comparable clip contains ``4k + 1`` keyframes.
    """

    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if (frame_count - 1) % 4 != 0:
        raise ValueError("frame_count must satisfy the Wan 4k+1 temporal contract")
    chunks: list[tuple[int, int]] = [(0, 1)]
    for start in range(1, frame_count, 4):
        chunks.append((start, start + 4))
    return tuple(chunks)


def sampled_source_indices(
    *, anchor_index: int, keyframe_count: int, source_stride: int
) -> Tuple[int, ...]:
    """Return the causal source rows ending at one expert anchor."""

    if type(anchor_index) is not int or anchor_index < 0:
        raise ValueError("anchor_index must be a non-negative integer")
    if type(source_stride) is not int or source_stride < 1:
        raise ValueError("source_stride must be a positive integer")
    temporal_chunk_slices(keyframe_count)
    start = anchor_index - (keyframe_count - 1) * source_stride
    if start < 0:
        raise ValueError("selected history starts before the episode")
    return tuple(start + index * source_stride for index in range(keyframe_count))


def array_parity_metrics(reference: object, candidate: object) -> ArrayParityMetrics:
    """Measure parity without hiding scale or near-zero reference tensors."""

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "parity tensors must have identical shapes: "
            f"{reference_array.shape} vs {candidate_array.shape}"
        )
    if reference_array.size == 0:
        raise ValueError("parity tensors cannot be empty")
    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise ValueError("parity tensors must be finite")
    difference = candidate_array - reference_array
    reference_flat = reference_array.reshape(-1)
    candidate_flat = candidate_array.reshape(-1)
    reference_l2 = float(np.linalg.norm(reference_flat))
    difference_l2 = float(np.linalg.norm(difference.reshape(-1)))
    epsilon = float(np.finfo(np.float64).eps)
    denominator = max(reference_l2, epsilon)
    cosine_denominator = max(
        reference_l2 * float(np.linalg.norm(candidate_flat)),
        epsilon,
    )
    return ArrayParityMetrics(
        shape=tuple(int(value) for value in reference_array.shape),
        mean_abs=float(np.mean(np.abs(difference))),
        root_mean_square=float(np.sqrt(np.mean(np.square(difference)))),
        max_abs=float(np.max(np.abs(difference))),
        relative_l2=difference_l2 / denominator,
        cosine_similarity=float(reference_flat @ candidate_flat / cosine_denominator),
        reference_mean=float(np.mean(reference_array)),
        reference_std=float(np.std(reference_array)),
        candidate_mean=float(np.mean(candidate_array)),
        candidate_std=float(np.std(candidate_array)),
    )


def freeze_float32(value: object) -> Array:
    """Canonicalize a runtime tensor before hashing or persistence."""

    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("runtime tensor must be non-empty and finite")
    result.setflags(write=False)
    return result


__all__ = [
    "ArrayParityMetrics",
    "array_parity_metrics",
    "freeze_float32",
    "sampled_source_indices",
    "temporal_chunk_slices",
]

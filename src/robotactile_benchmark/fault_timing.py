"""Reproducible fault-onset timing for paired closed-loop episodes."""

from __future__ import annotations

from robotactile_benchmark.contracts import canonical_hash

FIXED_FAULT_ONSET_MODE = "fixed_v1"
EARLY_RANDOM_ONSET_MODE = "early_random_onset_v1"
EARLY_RANDOM_ONSET_DERIVATION = "robotactile.early-random-onset.v1"
DEFAULT_EARLY_ONSET_MAX_INDEX = 8


def derive_early_random_onset(
    *, task: str, seed: int, stop: int, cap: int = DEFAULT_EARLY_ONSET_MAX_INDEX
) -> int:
    """Pick a nonzero onset in the first third, capped for short episodes."""

    if not isinstance(task, str) or not task:
        raise ValueError("task must be a nonempty string")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(stop) is not int or stop < 4:
        raise ValueError("episode horizon is too short for an early fault onset")
    if type(cap) is not int or cap < 1:
        raise ValueError("fault_onset_max_index must be a positive integer")
    upper = min(cap, (stop - 1) // 3)
    digest = canonical_hash(
        {"namespace": EARLY_RANDOM_ONSET_DERIVATION, "task": task, "seed": seed}
    )
    return 1 + int(digest[:16], 16) % upper

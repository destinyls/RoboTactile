"""Stateless preregistered seed derivation for clean campaigns."""

from __future__ import annotations

import hashlib

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes

UNIVTAC_TASK_SEED_STRIDE = 1_000_000


def derive_clean_campaign_seed(
    *,
    protocol_id: str,
    master_seed: int,
    task_id: str,
    task_ordinal: int,
    role: str,
) -> int:
    """Derive a non-zero signed-31-bit seed from one immutable descriptor."""

    if not protocol_id or not task_id or role not in {"initial", "exogenous"}:
        raise ValueError("clean campaign seed descriptor is invalid")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if type(task_ordinal) is not int or task_ordinal < 0:
        raise ValueError("task_ordinal must be a non-negative integer")
    for counter in range(16):
        descriptor = {
            "counter": counter,
            "master_seed": master_seed,
            "namespace": protocol_id,
            "ordinal": task_ordinal,
            "role": role,
            "task_id": task_id,
        }
        digest = hashlib.sha256(canonical_json_bytes(descriptor)).digest()
        value = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if value != 0:
            return value
    raise RuntimeError("failed to derive a non-zero signed-31-bit seed")


def expected_clean_seed_pairs(
    *,
    protocol_id: str,
    master_seed: int,
    task_id: str,
    trial_count: int,
) -> frozenset[tuple[int, int]]:
    """Return the unordered realized seed-pair identity for one task stratum."""

    if type(trial_count) is not int or trial_count < 1:
        raise ValueError("trial_count must be a positive integer")
    return frozenset(
        (
            derive_clean_campaign_seed(
                protocol_id=protocol_id,
                master_seed=master_seed,
                task_id=task_id,
                task_ordinal=index,
                role="initial",
            ),
            derive_clean_campaign_seed(
                protocol_id=protocol_id,
                master_seed=master_seed,
                task_id=task_id,
                task_ordinal=index,
                role="exogenous",
            ),
        )
        for index in range(trial_count)
    )


def univtac_task_seed_start(eval_seed: int = 0) -> int:
    """Return the released evaluator's first task-local episode seed."""

    if type(eval_seed) is not int or eval_seed < 0:
        raise ValueError("eval_seed must be a non-negative integer")
    return UNIVTAC_TASK_SEED_STRIDE * (1 + eval_seed)


def derive_univtac_task_seed(*, eval_seed: int, candidate_ordinal: int) -> int:
    """Derive one consecutive seed; each task reuses this same sequence."""

    if type(candidate_ordinal) is not int or candidate_ordinal < 0:
        raise ValueError("candidate_ordinal must be a non-negative integer")
    return univtac_task_seed_start(eval_seed) + candidate_ordinal


def expected_univtac_task_seed_pairs(
    *, eval_seed: int, candidate_count: int
) -> frozenset[tuple[int, int]]:
    """Return candidate seed pairs for one released-evaluator task stratum."""

    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    return frozenset(
        (seed, seed)
        for seed in (
            derive_univtac_task_seed(
                eval_seed=eval_seed,
                candidate_ordinal=index,
            )
            for index in range(candidate_count)
        )
    )


__all__ = [
    "UNIVTAC_TASK_SEED_STRIDE",
    "derive_clean_campaign_seed",
    "derive_univtac_task_seed",
    "expected_clean_seed_pairs",
    "expected_univtac_task_seed_pairs",
    "univtac_task_seed_start",
]

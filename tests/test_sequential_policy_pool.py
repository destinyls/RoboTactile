"""Lifecycle tests for sequential policy reuse in paired ACT execution."""

from __future__ import annotations

from dataclasses import replace

import pytest

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    PolicyEpisodeContext,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.fakes import DeterministicFakePolicy
from robotactile_benchmark.execution.sequential_policy_pool import (
    SequentialPolicyPool,
)


def _identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="official-act-test",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _context(index: int) -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id=f"episode-{index}",
        task="grasp_classify",
        initial_seed=10,
        exogenous_seed=20,
        instruction="grasp and classify",
        action_spec=ACTION_SPEC,
    )


def test_pool_loads_once_resets_each_episode_and_closes_owner_once() -> None:
    identity = _identity()
    owners: list[DeterministicFakePolicy] = []

    def factory() -> DeterministicFakePolicy:
        owner = DeterministicFakePolicy(identity)
        owners.append(owner)
        return owner

    with SequentialPolicyPool() as pool:
        first = pool.acquire("univtac", identity, factory)
        first.reset(_context(0))
        first.close()
        second = pool.acquire("univtac", identity, factory)
        second.reset(_context(1))
        second.close()

    assert len(owners) == 1
    assert owners[0].reset_count == 2
    assert owners[0].close_count == 1


def test_pool_rejects_overlap_and_wrong_cached_identity() -> None:
    identity = _identity()
    owner = DeterministicFakePolicy(identity)
    pool = SequentialPolicyPool()
    lease = pool.acquire("univtac", identity, lambda: owner)
    with pytest.raises(RuntimeError, match="lease overlap"):
        pool.acquire("univtac", identity, lambda: owner)
    lease.close()
    with pytest.raises(ValueError, match="changed policy identity"):
        pool.acquire(
            "univtac",
            replace(identity, system_id="different-act"),
            lambda: owner,
        )
    pool.close()
    assert owner.close_count == 1


def test_pool_preserves_primary_error_and_records_close_failure() -> None:
    identity = _identity()
    owner = DeterministicFakePolicy(identity, raise_on_close=True)

    with (
        pytest.raises(RuntimeError, match="primary failure") as captured,
        SequentialPolicyPool() as pool,
    ):
        lease = pool.acquire("univtac", identity, lambda: owner)
        lease.close()
        raise RuntimeError("primary failure")

    assert any(
        "pooled policy close failure" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert owner.close_count == 1

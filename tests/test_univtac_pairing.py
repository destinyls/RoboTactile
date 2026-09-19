from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robotactile_benchmark.backends.qualification_fakes import make_fake_runtime
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_pairing import (
    PAIRING_EVIDENCE_LEVEL,
    UniVTACPairedBackendSession,
    UniVTACPairingError,
)
from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    PolicyEpisodeContext,
)


def _context(
    episode_id: str,
    *,
    exogenous_seed: int = 29,
    instruction: str | None = None,
) -> PolicyEpisodeContext:
    config = build_univtac_backend_config("pull_out_key")
    return PolicyEpisodeContext(
        episode_id=episode_id,
        task=config.task.task_id,
        initial_seed=11,
        exogenous_seed=exogenous_seed,
        instruction=instruction or config.task.prompt,
        action_spec=ACTION_SPEC,
    )


def _action() -> np.ndarray:
    action = np.zeros((1, 8), dtype=np.float32)
    action[0, 3] = -1.0
    action[0, 7] = 0.02
    return action


def test_paired_session_resets_once_and_restores_exact_state() -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtime, task = make_fake_runtime(config)
    session = UniVTACPairedBackendSession(config, runtime)

    clean = session.new_backend()
    clean_receipt = clean.reset(_context("clean-episode"))
    clean_record = clean.observe()
    clean.execute(_action())
    clean.close()

    faulted = session.new_backend()
    replay_receipt = faulted.reset(_context("faulted-episode"))
    replay_record = faulted.observe()
    faulted.close()

    receipt = session.reset_receipt
    assert clean.reset_mode == "canonical_reset"
    assert faulted.reset_mode == "snapshot_replay"
    assert clean_receipt.simulator_state_sha256 == replay_receipt.simulator_state_sha256
    assert clean_record.observation.episode_id != replay_record.observation.episode_id
    np.testing.assert_array_equal(
        clean_record.observation.proprio, replay_record.observation.proprio
    )
    assert task.reset_count == 1
    assert task.capture_count == 1
    assert task.restore_count == 1
    assert task.close_count == 0
    assert receipt.evidence_level == PAIRING_EVIDENCE_LEVEL
    assert receipt.all_exact
    assert (
        receipt.witnesses[0].snapshot_state_sha256
        == receipt.witnesses[1].snapshot_state_sha256
    )
    assert [item.reset_mode for item in receipt.witnesses] == [
        "canonical_reset",
        "snapshot_replay",
    ]
    assert receipt.witnesses[0].equivalence_key == receipt.witnesses[1].equivalence_key

    session.close()
    session.close()
    assert task.close_count == 1


def test_paired_reset_uses_native_prompt_for_n0_policy_context() -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtime, task = make_fake_runtime(config)
    session = UniVTACPairedBackendSession(config, runtime)
    policy_prompt = "Untwist and extract a key from a lock"
    backend = session.new_backend()

    backend.reset(_context("n0-clean", instruction=policy_prompt))

    assert policy_prompt != config.task.prompt
    assert task.reset_calls == [(11, (config.task.prompt,))]
    backend.close()
    session.close()


def test_paired_session_rejects_restore_drift_before_observation_delivery() -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtime, task = make_fake_runtime(config)
    original_restore = runtime.restore_state
    assert original_restore is not None

    def drifting_restore(snapshot: object) -> None:
        original_restore(snapshot)
        task._state += 0.25

    session = UniVTACPairedBackendSession(
        config, replace(runtime, restore_state=drifting_restore)
    )
    canonical = session.new_backend()
    canonical.reset(_context("clean-episode"))
    canonical.observe()
    canonical.close()
    replay = session.new_backend()

    with pytest.raises(UniVTACPairingError) as captured:
        replay.reset(_context("faulted-episode"))

    assert captured.value.code == "reset_equivalence_mismatch"
    receipt = session.reset_receipt
    assert not receipt.all_exact
    assert receipt.witnesses[-1].exact_match is False
    replay.close()
    session.close()


def test_paired_session_rejects_context_drift_and_missing_callbacks() -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtime, _ = make_fake_runtime(config)
    session = UniVTACPairedBackendSession(config, runtime)
    first = session.new_backend()
    first.reset(_context("clean-episode"))
    first.observe()
    first.close()
    second = session.new_backend()

    with pytest.raises(UniVTACPairingError) as captured:
        second.reset(_context("faulted-episode", exogenous_seed=30))
    assert captured.value.code == "paired_context_mismatch"
    second.close()
    session.close()

    no_snapshot, task = make_fake_runtime(config)
    no_snapshot = replace(
        no_snapshot,
        capture_state=None,
        restore_state=None,
        snapshot_state_sha256=None,
    )
    with pytest.raises(UniVTACPairingError) as unavailable:
        UniVTACPairedBackendSession(config, no_snapshot)
    assert unavailable.value.code == "snapshot_callbacks_unavailable"
    no_snapshot.close_runtime()
    assert task.close_count == 1


def test_trajectory_profile_requires_exact_trajectory_type() -> None:
    config = build_univtac_backend_config("pull_out_key")
    runtime, task = make_fake_runtime(config)

    with pytest.raises(TypeError, match="exact UniVTACPreMoveTrajectory"):
        UniVTACPairedBackendSession(
            config,
            runtime,
            reset_trajectory=object(),  # type: ignore[arg-type]
        )

    runtime.close_runtime()
    assert task.close_count == 1

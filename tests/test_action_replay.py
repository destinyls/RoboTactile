"""Tests for simulator-only replay of a captured Clean action trace."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.closed_loop.capture import ActionTraceEntry
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    ClosedLoopRunSpec,
)
from robotactile_benchmark.closed_loop.fakes import DeterministicFakeBackend
from robotactile_benchmark.closed_loop.result_hashes import action_trace_sha256
from robotactile_benchmark.closed_loop.runner_checks import build_episode_context
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.action_replay import (
    ActionReplayError,
    ActionReplaySource,
    execute_action_trace_replay,
    write_action_trace_replay_receipt,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.trials import (
    Condition,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


def _trial() -> TrialManifest:
    system_id = "action-replay-test"
    checkpoint = "b" * 64
    config = "c" * 64
    return TrialManifest(
        task="insert_hole",
        initial_seed=10,
        exogenous_seed=20,
        condition=Condition.CLEAN,
        base_system_id=system_id,
        executed_system_id=system_id,
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            system_id, checkpoint, config, "qpos8_next_step"
        ),
        checkpoint_sha256=checkpoint,
        config_sha256=config,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=None,
        matched_no_touch_system_id=None,
    )


def _run_spec() -> ClosedLoopRunSpec:
    return ClosedLoopRunSpec(
        prompt="insert the peg",
        success_predicate_id="fake-success-v1",
        max_control_cycles=4,
        max_observation_steps=6,
        execute_action_steps=1,
        wall_timeout_s=5.0,
    )


def _entries(*source_steps: int) -> tuple[ActionTraceEntry, ...]:
    return tuple(
        ActionTraceEntry(
            action_plan_sha256=canonical_hash({"plan": index}),
            source_step_index=step,
            executed_actions=np.full((1, 8), index + 1, dtype=np.float32),
        )
        for index, step in enumerate(source_steps)
    )


def _source(
    *,
    entries: tuple[ActionTraceEntry, ...] | None = None,
    initial_state_sha256: str | None = None,
    terminal_status: TerminalStatus = TerminalStatus.SUCCESS,
) -> ActionReplaySource:
    trial = _trial()
    run_spec = _run_spec()
    selected = _entries(0) if entries is None else entries
    context = build_episode_context(trial, run_spec.prompt)
    initial = initial_state_sha256 or canonical_hash(
        {"episode_id": context.episode_id, "state": 0.0}
    )
    return ActionReplaySource(
        trial=trial,
        run_spec=run_spec,
        artifact_root_sha256="d" * 64,
        initial_state_sha256=initial,
        action_trace_sha256=action_trace_sha256(
            tuple(entry.to_hash_entry() for entry in selected)
        ),
        source_terminal_status=terminal_status,
        source_score_success=terminal_status is TerminalStatus.SUCCESS,
        source_control_cycle_count=len(selected),
        entries=selected,
    )


def test_action_replay_executes_a_fresh_backend_without_policy_inference() -> None:
    source = _source()
    backend = DeterministicFakeBackend(
        make_synthetic_episode(length=10), terminal_signal=BackendSignal.SUCCESS
    )

    receipt = execute_action_trace_replay(source, backend)

    assert receipt.terminal_status is TerminalStatus.SUCCESS
    assert receipt.task_success is True
    assert receipt.policy_inference_executed is False
    assert receipt.initial_state_exact_match is True
    assert receipt.replayed_action_trace_sha256 == source.action_trace_sha256
    assert backend.reset_count == 1
    assert backend.observe_count == 1
    assert backend.execute_count == 1
    assert backend.close_count == 1


def test_action_replay_marks_a_running_trace_end_as_exhausted() -> None:
    source = _source(entries=_entries(0, 1), terminal_status=TerminalStatus.TIMEOUT)
    backend = DeterministicFakeBackend(make_synthetic_episode(length=10))

    receipt = execute_action_trace_replay(source, backend)

    assert receipt.terminal_status is TerminalStatus.TIMEOUT
    assert receipt.failure_code == "action_trace_exhausted"
    assert receipt.task_success is False
    assert receipt.replayed_action_count == source.planned_action_count
    assert backend.execute_count == 2
    assert backend.close_count == 1


def test_action_replay_rejects_initial_state_drift_before_actions() -> None:
    source = _source(initial_state_sha256="e" * 64)
    backend = DeterministicFakeBackend(make_synthetic_episode(length=10))

    with pytest.raises(ActionReplayError, match="initial_state_mismatch"):
        execute_action_trace_replay(source, backend)

    assert backend.execute_count == 0
    assert backend.close_count == 1


def test_action_replay_source_rejects_non_dense_plan_anchors() -> None:
    entries = _entries(0, 2)

    with pytest.raises(ValueError, match="densely anchored"):
        _source(entries=entries, terminal_status=TerminalStatus.TIMEOUT)


def test_action_replay_receipt_writer_is_no_clobber(tmp_path: Path) -> None:
    source = _source()
    first = execute_action_trace_replay(
        source,
        DeterministicFakeBackend(
            make_synthetic_episode(length=10), terminal_signal=BackendSignal.SUCCESS
        ),
    )
    path = tmp_path / "action_replay_receipt.json"

    first_sha = write_action_trace_replay_receipt(path, first)
    repeated_sha = write_action_trace_replay_receipt(path, first)

    assert first_sha == repeated_sha
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_action_trace_replay_receipt(
            path,
            replace(first, final_transition_diagnostics={"changed": True}),
        )

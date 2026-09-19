"""Diagnostic live replay of actions captured from a Clean policy episode."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.capture import (
    ActionTraceEntry,
    TransitionTraceEntry,
)
from robotactile_benchmark.closed_loop.contracts import BackendSignal
from robotactile_benchmark.closed_loop.interfaces import SimulationBackend
from robotactile_benchmark.closed_loop.result_hashes import action_trace_sha256
from robotactile_benchmark.closed_loop.runner_checks import (
    build_episode_context,
    validate_batch,
    validate_clean_record,
    validate_reset,
)
from robotactile_benchmark.contracts import canonical_hash, thaw_value
from robotactile_benchmark.execution.action_replay_contracts import (
    ACTION_REPLAY_EVIDENCE_LEVEL,
    ACTION_REPLAY_SEMANTIC_VERSION,
    ActionReplayError,
    ActionReplaySource,
    ActionTraceReplayReceipt,
)
from robotactile_benchmark.trials import TerminalStatus

_TRANSITION_TRACE_NAMESPACE = (
    "robotactile_benchmark.diagnostic_action_replay.transition_trace.v1"
)


def execute_action_trace_replay(
    source: ActionReplaySource,
    backend: SimulationBackend,
) -> ActionTraceReplayReceipt:
    """Replay exact Clean actions in a fresh live backend without policy inference."""

    if type(source) is not ActionReplaySource:
        raise TypeError("source must be an exact ActionReplaySource")
    if backend.action_spec != source.trial.action_spec:
        raise ValueError("replay backend action spec differs from source trial")
    if backend.success_predicate_id != source.run_spec.success_predicate_id:
        raise ValueError("replay backend success predicate differs from source run")
    context = build_episode_context(source.trial, source.run_spec.prompt)
    replay_entries: list[ActionTraceEntry] = []
    transitions: list[TransitionTraceEntry] = []
    receipt_diagnostics: Mapping[str, object] = {}
    initial_state_sha256: Optional[str] = None
    replayed_action_count = 0
    replayed_cycle_count = 0
    signal = BackendSignal.RUNNING
    previous_native_step_id: Optional[int] = None
    try:
        reset_receipt = backend.reset(context)
        validate_reset(reset_receipt, context)
        receipt_diagnostics = reset_receipt.diagnostics
        initial_state_sha256 = reset_receipt.simulator_state_sha256
        if initial_state_sha256 != source.initial_state_sha256:
            raise ActionReplayError("initial_state_mismatch")
        initial = backend.observe()
        validate_clean_record(initial, context, 0)
        expected_step = 1
        for source_entry in source.entries:
            batch = backend.execute(source_entry.executed_actions)
            previous_native_step_id = validate_batch(
                batch,
                int(source_entry.executed_actions.shape[0]),
                context,
                expected_step,
                previous_native_step_id,
            )
            executed_count = batch.executed_action_count
            replay_entries.append(
                ActionTraceEntry(
                    action_plan_sha256=source_entry.action_plan_sha256,
                    source_step_index=source_entry.source_step_index,
                    executed_actions=source_entry.executed_actions[:executed_count],
                )
            )
            transitions.extend(
                TransitionTraceEntry.from_backend_transition(item)
                for item in batch.transitions
            )
            replayed_action_count += executed_count
            replayed_cycle_count += 1
            expected_step += executed_count
            signal = batch.transitions[-1].signal
            if signal is not BackendSignal.RUNNING:
                break
        if signal is BackendSignal.RUNNING:
            terminal_status = TerminalStatus.TIMEOUT
            failure_code = "action_trace_exhausted"
        else:
            terminal_status = TerminalStatus(signal.value)
            failure_code = None
        replay_hash = action_trace_sha256(
            tuple(entry.to_hash_entry() for entry in replay_entries)
        )
        final_diagnostics = {} if not transitions else transitions[-1].diagnostics
        return ActionTraceReplayReceipt(
            task=source.trial.task,
            initial_seed=source.trial.initial_seed,
            exogenous_seed=source.trial.exogenous_seed,
            source_artifact_root_sha256=source.artifact_root_sha256,
            source_trial_manifest_sha256=source.trial.sha256,
            source_action_trace_sha256=source.action_trace_sha256,
            source_initial_state_sha256=source.initial_state_sha256,
            source_terminal_status=source.source_terminal_status,
            source_score_success=source.source_score_success,
            replay_initial_state_sha256=initial_state_sha256,
            initial_state_exact_match=True,
            planned_control_cycle_count=source.source_control_cycle_count,
            replayed_control_cycle_count=replayed_cycle_count,
            planned_action_count=source.planned_action_count,
            replayed_action_count=replayed_action_count,
            replayed_action_trace_sha256=replay_hash,
            transition_trace_sha256=_transition_trace_sha256(transitions),
            terminal_status=terminal_status,
            task_success=terminal_status is TerminalStatus.SUCCESS,
            failure_code=failure_code,
            initial_diagnostics=receipt_diagnostics,
            final_transition_diagnostics=final_diagnostics,
        )
    finally:
        try:
            backend.close()
        except SystemExit as error:
            if error.code not in (None, 0):
                raise


def _transition_trace_sha256(entries: list[TransitionTraceEntry]) -> str:
    return canonical_hash(
        {
            "namespace": _TRANSITION_TRACE_NAMESPACE,
            "entries": [
                {
                    "benchmark_step_index": entry.benchmark_step_index,
                    "native_step_id": entry.native_step_id,
                    "signal": entry.signal.value,
                    "diagnostics": thaw_value(entry.diagnostics),
                }
                for entry in entries
            ],
        }
    )


def write_action_trace_replay_receipt(
    path: Path, receipt: ActionTraceReplayReceipt
) -> str:
    """Atomically publish canonical replay evidence without replacing bytes."""

    if type(receipt) is not ActionTraceReplayReceipt:
        raise TypeError("receipt must be an exact ActionTraceReplayReceipt")
    target = Path(path).expanduser().absolute()
    raw = canonical_json_bytes(receipt.to_dict())
    digest = hashlib.sha256(raw).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
            raise FileExistsError(f"refusing to replace replay receipt: {target}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise FileExistsError(
                    f"refusing to replace replay receipt: {target}"
                ) from None
        return digest
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ACTION_REPLAY_EVIDENCE_LEVEL",
    "ACTION_REPLAY_SEMANTIC_VERSION",
    "ActionReplayError",
    "ActionReplaySource",
    "ActionTraceReplayReceipt",
    "execute_action_trace_replay",
    "write_action_trace_replay_receipt",
]

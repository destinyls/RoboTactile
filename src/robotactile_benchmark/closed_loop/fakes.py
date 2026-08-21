"""Deterministic in-memory doubles for closed-loop runner qualification."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    ActionPlan,
    BackendSignal,
    BackendTransition,
    ClosedLoopRunSpec,
    ExecutionBatch,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
    ResetReceipt,
)
from robotactile_benchmark.contracts import (
    Array,
    EvaluationRecord,
    ObservationRecord,
    build_evaluation_record,
    canonical_hash,
)
from robotactile_benchmark.trials import TrialManifest


class DeterministicFakeBackend:
    """A stateful backend whose clean observations depend on executed actions."""

    backend_id = "deterministic-fake-backend-v1"
    action_spec = ACTION_SPEC

    def __init__(
        self,
        records: Sequence[EvaluationRecord],
        terminal_signal: BackendSignal = BackendSignal.RUNNING,
        append_after_terminal: bool = False,
        partial_batch: bool = False,
        reset_receipt_mismatch: bool = False,
        raise_on_close: bool = False,
        success_predicate_id: str = "fake-success-v1",
    ) -> None:
        if len(records) < 2:
            raise ValueError("fake backend requires at least two clean records")
        self._records = tuple(records)
        self._terminal_signal = terminal_signal
        self._append_after_terminal = append_after_terminal
        self._partial_batch = partial_batch
        self._reset_receipt_mismatch = reset_receipt_mismatch
        self._raise_on_close = raise_on_close
        if not success_predicate_id:
            raise ValueError("success_predicate_id must be non-empty")
        self.success_predicate_id = success_predicate_id
        self._context: Optional[PolicyEpisodeContext] = None
        self._next_index = 0
        self._state = 0.0
        self.reset_count = 0
        self.observe_count = 0
        self.execute_count = 0
        self.close_count = 0
        self.state_trace: list[float] = []
        self.observation_trace: list[ObservationRecord] = []

    def reset(self, context: PolicyEpisodeContext) -> ResetReceipt:
        self.reset_count += 1
        self._context = context
        self._next_index = 0
        self._state = 0.0
        self.state_trace = [self._state]
        self.observation_trace = []
        return ResetReceipt(
            episode_id=(
                "mismatched-episode"
                if self._reset_receipt_mismatch
                else context.episode_id
            ),
            initial_seed=context.initial_seed,
            exogenous_seed=context.exogenous_seed,
            simulator_state_sha256=canonical_hash(
                {"episode_id": context.episode_id, "state": self._state}
            ),
            native_reset_id=f"fake-reset-{self.reset_count}",
        )

    def observe(self) -> EvaluationRecord:
        if self._context is None:
            raise RuntimeError("fake backend has not been reset")
        if self.observe_count:
            raise RuntimeError("fake backend permits exactly one initial observe")
        self.observe_count += 1
        record = self._stateful_record(self._next_index)
        self.observation_trace.append(record.observation)
        return record

    def execute(self, actions: Array) -> ExecutionBatch:
        if self._context is None or not self.observe_count:
            raise RuntimeError("fake backend must observe before execute")
        if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions):
            raise ValueError("fake backend requires a non-empty [H, 8] action batch")
        self.execute_count += 1
        transitions: list[BackendTransition] = []
        action_batch = actions[:1] if self._partial_batch else actions
        for action_index, action in enumerate(action_batch):
            self._next_index += 1
            if self._next_index >= len(self._records):
                raise RuntimeError("fake backend exhausted its clean reference records")
            self._state += float(action[0])
            self.state_trace.append(self._state)
            signal = (
                self._terminal_signal
                if self._terminal_signal is not BackendSignal.RUNNING
                and not transitions
                else BackendSignal.RUNNING
            )
            record = self._stateful_record(self._next_index)
            self.observation_trace.append(record.observation)
            transitions.append(
                BackendTransition(
                    clean_record=record,
                    signal=signal,
                    native_step_id=self._next_index,
                    diagnostics={
                        "fake_state": self._state,
                        "action_index": action_index,
                    },
                )
            )
            if signal is not BackendSignal.RUNNING and not self._append_after_terminal:
                break
        return ExecutionBatch(
            transitions=tuple(transitions),
            executed_action_count=len(transitions),
        )

    def close(self) -> None:
        self.close_count += 1
        if self._raise_on_close:
            raise RuntimeError("fake backend close failure")

    def _stateful_record(self, index: int) -> EvaluationRecord:
        if self._context is None:
            raise RuntimeError("fake backend has not been reset")
        template = self._records[index]
        vision = {
            name: np.clip(
                values.astype(np.float32) + self._state * 8.0,
                0,
                255,
            ).astype(np.uint8)
            for name, values in template.observation.vision.items()
        }
        observation = replace(
            template.observation,
            episode_id=self._context.episode_id,
            task=self._context.task,
            seed=self._context.initial_seed,
            vision=vision,
            proprio=template.observation.proprio + np.float32(self._state),
        )
        return build_evaluation_record(observation, template.provenance)


class DeterministicFakePolicy:
    """A policy double that records only its public boundary values."""

    def __init__(
        self,
        identity: PolicyIdentity,
        source_step_offset: int = 0,
        action_horizon: int = 1,
        initial_action_bias: float = 0.0,
        fail_on_infer: bool = False,
        fail_on_commit: bool = False,
        fail_on_reset: bool = False,
        raise_on_close: bool = False,
    ) -> None:
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.identity = identity
        self._source_step_offset = source_step_offset
        self._action_horizon = action_horizon
        self._initial_action_bias = initial_action_bias
        self._fail_on_infer = fail_on_infer
        self._fail_on_commit = fail_on_commit
        self._fail_on_reset = fail_on_reset
        self._raise_on_close = raise_on_close
        self.reset_count = 0
        self.abort_count = 0
        self.close_count = 0
        self.inferred: list[ObservationRecord] = []
        self.committed: list[PolicyExecution] = []
        self.abort_reasons: list[str] = []
        self.last_context: Optional[PolicyEpisodeContext] = None

    @classmethod
    def for_trial(
        cls,
        trial: TrialManifest,
        consumes_tactile: Optional[bool] = None,
        supports_structural_absence: bool = True,
        source_step_offset: int = 0,
        action_horizon: int = 1,
        action_bias: float = 0.0,
        fail_on_infer: bool = False,
        fail_on_commit: bool = False,
        fail_on_reset: bool = False,
        raise_on_close: bool = False,
    ) -> DeterministicFakePolicy:
        tactile = (
            trial.condition.value != "no_touch"
            if consumes_tactile is None
            else consumes_tactile
        )
        return cls(
            PolicyIdentity(
                system_id=trial.executed_system_id,
                checkpoint_sha256=trial.checkpoint_sha256,
                config_sha256=trial.config_sha256,
                action_spec=trial.action_spec,
                consumes_tactile=tactile,
                supports_structural_absence=supports_structural_absence,
            ),
            source_step_offset=source_step_offset,
            action_horizon=action_horizon,
            initial_action_bias=action_bias,
            fail_on_infer=fail_on_infer,
            fail_on_commit=fail_on_commit,
            fail_on_reset=fail_on_reset,
            raise_on_close=raise_on_close,
        )

    @staticmethod
    def context_for_trial(
        trial: TrialManifest, run_spec: ClosedLoopRunSpec
    ) -> PolicyEpisodeContext:
        return PolicyEpisodeContext(
            episode_id=canonical_hash(
                {
                    "namespace": "robotactile_benchmark.closed_loop.episode.v1",
                    "trial_manifest_sha256": trial.sha256,
                }
            ),
            task=trial.task,
            initial_seed=trial.initial_seed,
            exogenous_seed=trial.exogenous_seed,
            instruction=run_spec.prompt,
            action_spec=trial.action_spec,
        )

    def reset(self, context: PolicyEpisodeContext) -> None:
        self.reset_count += 1
        self.last_context = context
        if self._fail_on_reset:
            raise RuntimeError("fake policy reset failure")

    def infer(self, observation: ObservationRecord) -> ActionPlan:
        if self._fail_on_infer:
            raise RuntimeError("fake policy infer failure")
        self.inferred.append(observation)
        left_payload = observation.sensor("left").payload
        tactile_signal = (
            -1.0 if left_payload is None else float(left_payload.mean()) / 255.0
        )
        action_signal = tactile_signal + float(observation.proprio.mean()) * 0.01
        if observation.step_index == 0:
            action_signal += self._initial_action_bias
        actions = np.full((self._action_horizon, 8), action_signal, dtype=np.float32)
        return ActionPlan(
            action_spec=self.identity.action_spec,
            source_step_index=observation.step_index + self._source_step_offset,
            actions=actions,
        )

    def commit(self, execution: PolicyExecution) -> None:
        self.committed.append(execution)
        if self._fail_on_commit:
            raise RuntimeError("fake policy commit failure")

    def abort(self, reason_code: str) -> None:
        self.abort_count += 1
        self.abort_reasons.append(reason_code)

    def close(self) -> None:
        self.close_count += 1
        if self._raise_on_close:
            raise RuntimeError("fake policy close failure")

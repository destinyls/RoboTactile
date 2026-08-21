"""Behavioral tests for the in-memory single-trial closed-loop runner."""

from __future__ import annotations

import unittest
from dataclasses import fields, replace

import numpy as np

from robotactile_benchmark.closed_loop.contracts import BackendSignal, ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.result_hashes import terminal_trace_sha256
from robotactile_benchmark.closed_loop.results import ClosedLoopTrialResult
from robotactile_benchmark.closed_loop.runner import run_closed_loop_trial
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


def _trial(
    condition: Condition = Condition.CLEAN,
    **overrides: object,
) -> TrialManifest:
    base_system_id = "fake-tactile-policy"
    values: dict[str, object] = {
        "task": "insert_HDMI",
        "initial_seed": 10,
        "exogenous_seed": 20,
        "condition": condition,
        "base_system_id": base_system_id,
        "executed_system_id": base_system_id,
        "dataset_sha256": "a" * 64,
        "base_system_manifest_sha256": system_manifest_hash(
            base_system_id, "b" * 64, "c" * 64, "qpos8_next_step"
        ),
        "checkpoint_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "action_spec": "qpos8_next_step",
        "fault_manifest_sha256": None,
        "matched_no_touch_system_id": None,
        "restoration_index": None,
        "restoration_mode": None,
    }
    if condition is Condition.NO_TOUCH:
        values.update(
            {
                "executed_system_id": "fake-no-touch-policy",
                "checkpoint_sha256": "d" * 64,
                "config_sha256": "e" * 64,
                "matched_no_touch_system_id": "fake-no-touch-policy",
            }
        )
    values.update(overrides)
    return TrialManifest(**values)


def _fault(operator_id: str = "A1_stream_absence") -> FaultManifest:
    return FaultManifest(
        operator_id=operator_id,
        severity_level=3,
        operator_seed=23,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )


def _spec(**overrides: object) -> ClosedLoopRunSpec:
    values: dict[str, object] = {
        "prompt": "insert the cable",
        "success_predicate_id": "fake-success-v1",
        "max_control_cycles": 4,
        "max_observation_steps": 6,
        "execute_action_steps": 1,
        "wall_timeout_s": 5.0,
    }
    values.update(overrides)
    return ClosedLoopRunSpec(**values)


def _rehashed_result(
    result: ClosedLoopTrialResult,
    **overrides: object,
) -> ClosedLoopTrialResult:
    values = {field.name: getattr(result, field.name) for field in fields(result)}
    values.update(overrides)
    values["terminal_trace_sha256"] = terminal_trace_sha256(
        trial_manifest_sha256=values["trial_manifest_sha256"],
        pair_key=values["pair_key"],
        run_spec_sha256=values["run_spec_sha256"],
        initial_state_sha256=values["initial_state_sha256"],
        terminal_status=values["terminal_status"],
        execution_status=values["execution_status"],
        score_eligible=values["score_eligible"],
        score_success=values["score_success"],
        validation_passed=values["validation_passed"],
        validation_failure_codes=values["validation_failure_codes"],
        clean_trace_sha256=values["clean_trace_sha256"],
        delivered_trace_sha256=values["delivered_trace_sha256"],
        action_trace_sha256_value=values["action_trace_sha256"],
        observation_count=values["observation_count"],
        control_cycle_count=values["control_cycle_count"],
        failure_stage=values["failure_stage"],
        failure_code=values["failure_code"],
        semantic_version=values["semantic_version"],
    )
    return ClosedLoopTrialResult(**values)


class ClosedLoopRunnerTests(unittest.TestCase):
    def _run(
        self,
        *,
        trial: TrialManifest | None = None,
        fault: FaultManifest | None = None,
        backend: DeterministicFakeBackend | None = None,
        policy: DeterministicFakePolicy | None = None,
        spec: ClosedLoopRunSpec | None = None,
    ):
        active_trial = _trial() if trial is None else trial
        active_backend = (
            DeterministicFakeBackend(make_synthetic_episode(length=10))
            if backend is None
            else backend
        )
        active_policy = (
            DeterministicFakePolicy.for_trial(active_trial)
            if policy is None
            else policy
        )
        result = run_closed_loop_trial(
            active_trial,
            _spec() if spec is None else spec,
            active_backend,
            active_policy,
            fault_manifest=fault,
        )
        return result, active_backend, active_policy

    def test_policy_only_receives_delivered_observations_and_commit_uses_faulted_next(
        self,
    ) -> None:
        fault = _fault()
        trial = _trial(Condition.FAULTED, fault_manifest_sha256=fault.sha256)
        result, backend, policy = self._run(trial=trial, fault=fault)

        self.assertEqual(result.terminal_status, TerminalStatus.TIMEOUT)
        self.assertTrue(
            all(type(item).__name__ == "ObservationRecord" for item in policy.inferred)
        )
        self.assertFalse(any(hasattr(item, "provenance") for item in policy.inferred))
        self.assertTrue(policy.committed)
        self.assertIsNone(
            policy.committed[0].delivered_observations[0].sensor("left").payload
        )
        self.assertIsNotNone(backend.observation_trace[1].sensor("left").payload)

    def test_action_plan_must_be_anchored_to_the_delivered_step(self) -> None:
        policy = DeterministicFakePolicy.for_trial(_trial(), source_step_offset=1)

        result, backend, observed_policy = self._run(policy=policy)

        self.assertEqual(result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(result.failure_code, "action_plan_source_step_mismatch")
        self.assertEqual(backend.execute_count, 0)
        self.assertEqual(result.control_cycle_count, 0)
        self.assertEqual(observed_policy.abort_count, 1)

    def test_clean_and_faulted_observations_causally_change_fake_backend_state(
        self,
    ) -> None:
        clean_result, clean_backend, _ = self._run()
        fault = _fault()
        fault_trial = _trial(Condition.FAULTED, fault_manifest_sha256=fault.sha256)
        fault_result, fault_backend, _ = self._run(trial=fault_trial, fault=fault)

        self.assertNotEqual(clean_backend.state_trace, fault_backend.state_trace)
        self.assertNotEqual(
            clean_result.action_trace_sha256,
            fault_result.action_trace_sha256,
        )

    def test_terminal_statuses_have_explicit_score_semantics(self) -> None:
        cases = (
            (BackendSignal.SUCCESS, TerminalStatus.SUCCESS, True),
            (BackendSignal.TASK_FAILURE, TerminalStatus.TASK_FAILURE, False),
            (BackendSignal.EARLY_STOP, TerminalStatus.EARLY_STOP, False),
        )
        for signal, expected_status, expected_score in cases:
            with self.subTest(signal=signal):
                backend = DeterministicFakeBackend(
                    make_synthetic_episode(length=10), terminal_signal=signal
                )
                result, _, _ = self._run(backend=backend)
                self.assertEqual(result.terminal_status, expected_status)
                self.assertTrue(result.score_eligible)
                self.assertEqual(result.score_success, expected_score)

    def test_timeout_validator_reject_and_unsupported_are_explicit(self) -> None:
        timeout_result, _, _ = self._run(
            spec=_spec(max_control_cycles=1, max_observation_steps=4)
        )
        self.assertEqual(timeout_result.terminal_status, TerminalStatus.TIMEOUT)
        self.assertFalse(timeout_result.score_success)

        fault = _fault("A2_frame_erasure")
        fault_trial = _trial(Condition.FAULTED, fault_manifest_sha256=fault.sha256)
        rejected_result, _, _ = self._run(
            trial=fault_trial,
            fault=fault,
            backend=DeterministicFakeBackend(
                make_synthetic_episode(length=10), terminal_signal=BackendSignal.SUCCESS
            ),
            policy=DeterministicFakePolicy.for_trial(fault_trial),
        )
        self.assertEqual(rejected_result.execution_status, TerminalStatus.SUCCESS)
        self.assertEqual(
            rejected_result.terminal_status,
            TerminalStatus.VALIDATOR_REJECTED,
        )
        self.assertFalse(rejected_result.score_eligible)
        self.assertIsNone(rejected_result.score_success)

        absence = _fault("A1_stream_absence")
        absence_trial = _trial(Condition.FAULTED, fault_manifest_sha256=absence.sha256)
        unsupported_result, backend, _ = self._run(
            trial=absence_trial,
            fault=absence,
            policy=DeterministicFakePolicy.for_trial(
                absence_trial, supports_structural_absence=False
            ),
        )
        self.assertEqual(
            unsupported_result.terminal_status,
            TerminalStatus.UNSUPPORTED_CONTRACT,
        )
        self.assertFalse(unsupported_result.score_eligible)
        self.assertIsNone(unsupported_result.score_success)
        self.assertEqual(backend.reset_count, 0)

    def test_no_touch_requires_non_tactile_policy(self) -> None:
        trial = _trial(Condition.NO_TOUCH)
        tactile_policy = DeterministicFakePolicy.for_trial(trial, consumes_tactile=True)
        backend = DeterministicFakeBackend(make_synthetic_episode(length=10))

        with self.assertRaisesRegex(ValueError, "no-touch"):
            self._run(trial=trial, policy=tactile_policy, backend=backend)

        self.assertEqual(backend.reset_count, 0)
        self.assertEqual(backend.close_count, 1)
        self.assertEqual(tactile_policy.close_count, 1)

    def test_condition_fault_mismatch_raises_before_reset(self) -> None:
        backend = DeterministicFakeBackend(make_synthetic_episode(length=10))
        policy = DeterministicFakePolicy.for_trial(_trial())

        with self.assertRaisesRegex(ValueError, "fault manifest"):
            self._run(backend=backend, policy=policy, fault=_fault())

        self.assertEqual(backend.reset_count, 0)
        self.assertEqual(backend.close_count, 1)
        self.assertEqual(policy.close_count, 1)

    def test_success_predicate_mismatch_is_rejected_before_reset(self) -> None:
        backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            success_predicate_id="different-success-v1",
        )
        policy = DeterministicFakePolicy.for_trial(_trial())

        with self.assertRaisesRegex(ValueError, "success predicate"):
            self._run(backend=backend, policy=policy)

        self.assertEqual(backend.reset_count, 0)
        self.assertEqual(backend.close_count, 1)
        self.assertEqual(policy.reset_count, 0)
        self.assertEqual(policy.close_count, 1)

    def test_same_frozen_inputs_produce_identical_audit_hashes(self) -> None:
        first, _, _ = self._run()
        second, _, _ = self._run()

        self.assertEqual(first, second)
        self.assertEqual(first.clean_trace_sha256, second.clean_trace_sha256)
        self.assertEqual(first.delivered_trace_sha256, second.delivered_trace_sha256)

    def test_resources_close_once_after_success_and_crash(self) -> None:
        success, success_backend, success_policy = self._run(
            backend=DeterministicFakeBackend(
                make_synthetic_episode(length=10), terminal_signal=BackendSignal.SUCCESS
            )
        )
        self.assertEqual(success.terminal_status, TerminalStatus.SUCCESS)
        self.assertEqual(
            (success_backend.close_count, success_policy.close_count),
            (1, 1),
        )

        crash_policy = DeterministicFakePolicy.for_trial(_trial(), fail_on_infer=True)
        crash, crash_backend, observed_policy = self._run(policy=crash_policy)
        self.assertEqual(crash.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(
            (crash_backend.close_count, observed_policy.close_count),
            (1, 1),
        )
        self.assertEqual(observed_policy.abort_count, 1)

    def test_terminal_transition_cannot_be_followed_by_another_transition(self) -> None:
        backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            terminal_signal=BackendSignal.SUCCESS,
            append_after_terminal=True,
        )

        result, _, policy = self._run(
            backend=backend,
            policy=DeterministicFakePolicy.for_trial(_trial(), action_horizon=2),
            spec=_spec(execute_action_steps=2),
        )

        self.assertEqual(result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(result.failure_code, "terminal_transition_followed")
        self.assertEqual(policy.abort_count, 1)

    def test_fake_policy_uses_action_conditioned_proprio_beyond_identical_tactile(
        self,
    ) -> None:
        trial = _trial()
        first_backend = DeterministicFakeBackend(make_synthetic_episode(length=10))
        second_backend = DeterministicFakeBackend(make_synthetic_episode(length=10))
        first_policy = DeterministicFakePolicy.for_trial(trial, action_bias=0.0)
        second_policy = DeterministicFakePolicy.for_trial(trial, action_bias=0.5)
        context = first_backend.reset(first_policy.context_for_trial(trial, _spec()))
        self.assertIsNotNone(context)
        second_backend.reset(second_policy.context_for_trial(trial, _spec()))
        first_policy.reset(first_policy.last_context)
        second_policy.reset(second_policy.last_context)
        first_step0 = first_backend.observe().observation
        second_step0 = second_backend.observe().observation
        first_backend.execute(first_policy.infer(first_step0).actions)
        second_backend.execute(second_policy.infer(second_step0).actions)
        first_step1 = first_backend.observation_trace[1]
        second_step1 = second_backend.observation_trace[1]

        self.assertTrue(
            np.array_equal(
                first_step1.sensor("left").payload,
                second_step1.sensor("left").payload,
            )
        )
        self.assertFalse(np.array_equal(first_step1.proprio, second_step1.proprio))
        self.assertFalse(
            np.array_equal(
                first_policy.infer(first_step1).actions,
                second_policy.infer(second_step1).actions,
            )
        )

    def test_commit_failure_receipt_counts_already_delivered_transition(self) -> None:
        policy = DeterministicFakePolicy.for_trial(_trial(), fail_on_commit=True)

        result, _, observed_policy = self._run(policy=policy)

        self.assertEqual(result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(result.failure_stage, "commit")
        self.assertEqual(result.observation_count, 2)
        self.assertEqual(observed_policy.abort_count, 1)

    def test_budget_counts_the_initial_observation(self) -> None:
        result, _, _ = self._run(
            spec=_spec(max_control_cycles=4, max_observation_steps=4)
        )

        self.assertEqual(result.terminal_status, TerminalStatus.TIMEOUT)
        self.assertEqual(result.observation_count, 4)
        self.assertEqual(result.control_cycle_count, 3)

    def test_restoration_failure_keeps_operator_validation_codes(self) -> None:
        fault = _fault("A2_frame_erasure")
        trial = _trial(
            Condition.RESTORED,
            fault_manifest_sha256=fault.sha256,
            restoration_index=fault.stop_index,
            restoration_mode=RestorationMode.VALID_STREAM_RESUME,
        )

        result, _, _ = self._run(
            trial=trial,
            fault=fault,
            spec=_spec(max_control_cycles=1, max_observation_steps=2),
        )

        self.assertEqual(result.terminal_status, TerminalStatus.VALIDATOR_REJECTED)
        self.assertEqual(result.failure_stage, "validation")
        self.assertEqual(result.failure_code, "validator_rejected")
        self.assertIn("A2_RESUME_MISSING", result.validation_failure_codes)
        self.assertIn("RESTORATION_NOT_OBSERVED", result.validation_failure_codes)

    def test_trial_result_rejects_forged_trace_and_mutable_semantics(self) -> None:
        result, _, _ = self._run()
        self.assertIsInstance(result, ClosedLoopTrialResult)

        with self.assertRaises(ValueError):
            replace(result, terminal_trace_sha256="0" * 64)
        with self.assertRaises(ValueError):
            replace(result, validation_failure_codes=("DUPLICATE", "DUPLICATE"))
        with self.assertRaises(ValueError):
            replace(result, score_success=True)
        with self.assertRaises(TypeError):
            replace(result, validation_failure_codes=["NOT_A_TUPLE"])

    def test_reset_receipt_mismatch_is_a_reset_crash_with_the_returned_state_hash(
        self,
    ) -> None:
        backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10), reset_receipt_mismatch=True
        )

        result, _, policy = self._run(backend=backend)

        self.assertEqual(result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(result.execution_status, TerminalStatus.CRASH)
        self.assertEqual(result.failure_stage, "reset")
        self.assertEqual(result.failure_code, "reset_receipt_mismatch")
        self.assertIsNotNone(result.initial_state_sha256)
        self.assertEqual(policy.abort_count, 0)

    def test_partial_batches_require_a_terminal_signal(self) -> None:
        terminal_backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10), terminal_signal=BackendSignal.SUCCESS
        )
        terminal_result, _, _ = self._run(
            backend=terminal_backend,
            policy=DeterministicFakePolicy.for_trial(_trial(), action_horizon=2),
            spec=_spec(execute_action_steps=2),
        )
        self.assertEqual(terminal_result.terminal_status, TerminalStatus.SUCCESS)

        running_backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10), partial_batch=True
        )
        running_result, _, _ = self._run(
            backend=running_backend,
            policy=DeterministicFakePolicy.for_trial(_trial(), action_horizon=2),
            spec=_spec(execute_action_steps=2),
        )
        self.assertEqual(running_result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(running_result.failure_code, "running_partial_batch")
        self.assertEqual(running_result.control_cycle_count, 1)

    def test_close_failures_crash_a_normal_run_without_masking_a_primary_crash(
        self,
    ) -> None:
        backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            terminal_signal=BackendSignal.SUCCESS,
            raise_on_close=True,
        )
        policy = DeterministicFakePolicy.for_trial(_trial(), raise_on_close=True)
        close_result, closed_backend, closed_policy = self._run(
            backend=backend, policy=policy
        )
        self.assertEqual(close_result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(close_result.execution_status, TerminalStatus.SUCCESS)
        self.assertEqual(close_result.failure_stage, "close")
        self.assertEqual(close_result.failure_code, "multiple_close_failures")
        self.assertEqual(
            (closed_backend.close_count, closed_policy.close_count),
            (1, 1),
        )

        backend_only = DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            terminal_signal=BackendSignal.SUCCESS,
            raise_on_close=True,
        )
        single_close, _, _ = self._run(backend=backend_only)
        self.assertEqual(single_close.failure_code, "backend_close_failed")

        crash_backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10), raise_on_close=True
        )
        crash_policy = DeterministicFakePolicy.for_trial(
            _trial(), fail_on_infer=True, raise_on_close=True
        )
        primary_result, _, _ = self._run(backend=crash_backend, policy=crash_policy)
        self.assertEqual(primary_result.failure_stage, "infer")
        self.assertEqual(primary_result.failure_code, "infer_failed")

    def test_close_crash_preserves_restoration_and_operator_validation_codes(
        self,
    ) -> None:
        for operator_id, expected_codes in (
            ("A1_stream_absence", ("RESTORATION_NOT_OBSERVED",)),
            ("A2_frame_erasure", ("A2_RESUME_MISSING", "RESTORATION_NOT_OBSERVED")),
        ):
            with self.subTest(operator_id=operator_id):
                fault = _fault(operator_id)
                trial = _trial(
                    Condition.RESTORED,
                    fault_manifest_sha256=fault.sha256,
                    restoration_index=fault.stop_index,
                    restoration_mode=RestorationMode.VALID_STREAM_RESUME,
                )
                result, _, _ = self._run(
                    trial=trial,
                    fault=fault,
                    backend=DeterministicFakeBackend(
                        make_synthetic_episode(length=10),
                        terminal_signal=BackendSignal.SUCCESS,
                        raise_on_close=True,
                    ),
                    spec=_spec(max_control_cycles=1, max_observation_steps=2),
                )

                self.assertEqual(result.terminal_status, TerminalStatus.CRASH)
                self.assertEqual(result.failure_stage, "close")
                self.assertFalse(result.validation_passed)
                for expected_code in expected_codes:
                    self.assertIn(expected_code, result.validation_failure_codes)

    def test_rehashed_receipt_rejects_unknown_and_impossible_failure_stages(
        self,
    ) -> None:
        crash_result, _, _ = self._run(
            policy=DeterministicFakePolicy.for_trial(_trial(), fail_on_infer=True)
        )

        with self.assertRaises(ValueError):
            _rehashed_result(crash_result, failure_stage="banana")
        with self.assertRaises(ValueError):
            _rehashed_result(
                crash_result,
                failure_stage="commit",
                initial_state_sha256=None,
            )


if __name__ == "__main__":
    unittest.main()

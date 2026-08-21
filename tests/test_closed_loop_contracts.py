import unittest
from dataclasses import dataclass, fields, replace
from typing import Tuple

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
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
from robotactile_benchmark.closed_loop.interfaces import (
    ClosedLoopPolicy,
    SimulationBackend,
)
from robotactile_benchmark.contracts import EvaluationRecord, ObservationRecord
from robotactile_benchmark.fixtures import make_synthetic_episode

CHECKPOINT_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64
STATE_SHA256 = "c" * 64


def make_context() -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="episode-001",
        task="lift_bottle",
        initial_seed=7,
        exogenous_seed=11,
        instruction="Lift the bottle.",
        action_spec="qpos8_next_step",
    )


def make_identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="policy-v1",
        checkpoint_sha256=CHECKPOINT_SHA256,
        config_sha256=CONFIG_SHA256,
        action_spec="qpos8_next_step",
        consumes_tactile=True,
        supports_structural_absence=True,
    )


def make_plan() -> ActionPlan:
    return ActionPlan(
        action_spec="qpos8_next_step",
        source_step_index=0,
        actions=np.arange(16, dtype=np.float64).reshape(2, 8),
    )


def make_receipt() -> ResetReceipt:
    return ResetReceipt(
        episode_id="episode-001",
        initial_seed=7,
        exogenous_seed=11,
        simulator_state_sha256=STATE_SHA256,
        native_reset_id="native-reset-1",
    )


class ClosedLoopContractTests(unittest.TestCase):
    def test_context_is_model_only_and_content_addressed(self) -> None:
        context = make_context()

        forbidden = {
            "fault_manifest",
            "evaluator_provenance",
            "phase",
            "outcome",
            "schedule",
        }
        self.assertFalse({field.name for field in fields(context)} & forbidden)
        self.assertEqual(context.sha256, make_context().sha256)
        with self.assertRaises((TypeError, ValueError)):
            replace(context, initial_seed=True)
        with self.assertRaises((TypeError, ValueError)):
            replace(context, action_spec="ee_pose")

    def test_action_plan_freezes_contiguous_float32_actions(self) -> None:
        source = np.arange(16, dtype=np.float64).reshape(8, 2).T
        plan = ActionPlan("qpos8_next_step", 3, source)
        source[0, 0] = -1.0

        self.assertEqual(plan.actions.dtype, np.dtype(np.float32))
        self.assertTrue(plan.actions.flags.c_contiguous)
        self.assertFalse(plan.actions.flags.writeable)
        self.assertEqual(float(plan.actions[0, 0]), 0.0)
        with self.assertRaises(ValueError):
            plan.actions[0, 0] = 3.0

    def test_action_plan_rejects_malformed_arrays_and_indices(self) -> None:
        for actions in (
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((2, 7), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.zeros((2, 8), dtype=np.int64),
            np.full((2, 8), np.nan, dtype=np.float32),
        ):
            with (
                self.subTest(shape=actions.shape, dtype=actions.dtype),
                self.assertRaises((TypeError, ValueError)),
            ):
                ActionPlan("qpos8_next_step", 0, actions)
        with self.assertRaises(TypeError):
            ActionPlan("qpos8_next_step", True, np.zeros((1, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            ActionPlan("qpos8_next_step", -1, np.zeros((1, 8), dtype=np.float32))

    def test_run_spec_enforces_control_budget_and_stable_hash(self) -> None:
        spec = ClosedLoopRunSpec(
            prompt="Lift the bottle.",
            success_predicate_id="bottle-lifted-v1",
            max_control_cycles=4,
            max_observation_steps=8,
            execute_action_steps=2,
            wall_timeout_s=30.0,
        )

        self.assertEqual(spec.sha256, spec.sha256)
        for invalid in (
            {"prompt": ""},
            {"max_control_cycles": 9},
            {"execute_action_steps": 9},
            {"max_control_cycles": True},
            {"wall_timeout_s": float("inf")},
            {"semantic_version": "1.1"},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                replace(spec, **invalid)

    def test_identity_validates_digests_action_and_capabilities(self) -> None:
        identity = make_identity()

        self.assertEqual(len(identity.sha256), 64)
        for invalid in (
            {"system_id": ""},
            {"checkpoint_sha256": "A" * 64},
            {"config_sha256": "short"},
            {"action_spec": "cartesian"},
            {"consumes_tactile": np.bool_(True)},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                replace(identity, **invalid)

    def test_reset_receipt_validates_identity_seeds_and_digest(self) -> None:
        receipt = make_receipt()

        self.assertEqual(len(receipt.sha256), 64)
        for invalid in (
            {"episode_id": ""},
            {"initial_seed": True},
            {"exogenous_seed": -1},
            {"simulator_state_sha256": "C" * 64},
            {"native_reset_id": ""},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                replace(receipt, **invalid)

    def test_transition_rejects_faulted_provenance_and_non_json_diagnostics(
        self,
    ) -> None:
        clean = make_synthetic_episode()[0]
        faulted = clean.with_observation(clean.observation)
        faulted_provenance = list(faulted.provenance)
        faulted_provenance[0] = replace(
            faulted_provenance[0], active_fault_ids=("A1_stream_absence",)
        )
        faulted = EvaluationRecord(
            observation=faulted.observation,
            provenance=tuple(faulted_provenance),
            clean_record_sha256=faulted.clean_record_sha256,
            delivered_record_sha256=faulted.delivered_record_sha256,
        )

        with self.assertRaises(ValueError):
            BackendTransition(faulted, "running", 0, {})
        with self.assertRaises(ValueError):
            BackendTransition(clean, "running", 0, {"invalid": float("nan")})
        transition = BackendTransition(clean, "success", 0, {"score": [1, 2]})
        self.assertIs(transition.signal, BackendSignal.SUCCESS)
        self.assertEqual(transition.diagnostics["score"], (1, 2))
        with self.assertRaises(TypeError):
            transition.diagnostics["score"] = ()

    def test_execution_batch_requires_dense_benchmark_steps_and_native_order(
        self,
    ) -> None:
        records = make_synthetic_episode()
        first = BackendTransition(records[0], BackendSignal.RUNNING, 10, {})
        second = BackendTransition(records[1], BackendSignal.SUCCESS, 11, {})
        batch = ExecutionBatch((first, second), 2)

        self.assertEqual(batch.executed_action_count, 2)
        with self.assertRaises(ValueError):
            ExecutionBatch((first, second), 3)
        skipped = BackendTransition(records[2], BackendSignal.SUCCESS, 12, {})
        with self.assertRaises(ValueError):
            ExecutionBatch((first, skipped), 2)
        with self.assertRaises(ValueError):
            ExecutionBatch((first, replace(second, native_step_id=10)), 2)

    def test_policy_execution_exposes_only_observations_delivered_to_policy(
        self,
    ) -> None:
        records = make_synthetic_episode()
        source = np.ones((2, 8), dtype=np.float64)
        execution = PolicyExecution(
            action_plan_sha256=make_plan().sha256,
            executed_actions=source,
            delivered_observations=(records[0].observation, records[1].observation),
            terminal_signal="early_stop",
        )
        source[0, 0] = 7.0

        self.assertEqual(execution.executed_actions.dtype, np.dtype(np.float32))
        self.assertFalse(execution.executed_actions.flags.writeable)
        self.assertIs(execution.terminal_signal, BackendSignal.EARLY_STOP)
        with self.assertRaises(TypeError):
            PolicyExecution(
                action_plan_sha256=make_plan().sha256,
                executed_actions=np.ones((1, 8), dtype=np.float32),
                delivered_observations=(records[0],),
                terminal_signal=BackendSignal.SUCCESS,
            )
        with self.assertRaises(ValueError):
            PolicyExecution(
                action_plan_sha256=make_plan().sha256,
                executed_actions=np.ones((1, 8), dtype=np.float32),
                delivered_observations=(records[0].observation, records[1].observation),
                terminal_signal=BackendSignal.SUCCESS,
            )

    def test_policy_execution_rejects_observation_subclasses_with_private_fields(
        self,
    ) -> None:
        @dataclass(frozen=True)
        class LeakyObservation(ObservationRecord):
            evaluator_provenance: str = "private-receipt"
            active_fault_ids: Tuple[str, ...] = ("A1_stream_absence",)

        observation = make_synthetic_episode()[0].observation
        leaky = LeakyObservation(
            episode_id=observation.episode_id,
            task=observation.task,
            seed=observation.seed,
            step_index=observation.step_index,
            tactile=observation.tactile,
            vision=observation.vision,
            proprio=observation.proprio,
        )

        with self.assertRaisesRegex(TypeError, "exact ObservationRecord"):
            PolicyExecution(
                action_plan_sha256=make_plan().sha256,
                executed_actions=np.ones((1, 8), dtype=np.float32),
                delivered_observations=(leaky,),
                terminal_signal=BackendSignal.SUCCESS,
            )

    def test_concrete_fakes_satisfy_runtime_protocols(self) -> None:
        record = make_synthetic_episode()[0]

        class FakeBackend:
            backend_id = "fake-backend"
            action_spec = "qpos8_next_step"
            success_predicate_id = "fake-success-v1"

            def reset(self, context: PolicyEpisodeContext) -> ResetReceipt:
                return make_receipt()

            def observe(self) -> EvaluationRecord:
                return record

            def execute(self, actions: np.ndarray) -> ExecutionBatch:
                transition = BackendTransition(record, BackendSignal.SUCCESS, 0, {})
                return ExecutionBatch((transition,), 1)

            def close(self) -> None:
                return None

        class FakePolicy:
            identity = make_identity()

            def reset(self, context: PolicyEpisodeContext) -> None:
                return None

            def infer(self, observation: ObservationRecord) -> ActionPlan:
                return make_plan()

            def commit(self, execution: PolicyExecution) -> None:
                return None

            def abort(self, reason_code: str) -> None:
                return None

            def close(self) -> None:
                return None

        self.assertIsInstance(FakeBackend(), SimulationBackend)
        self.assertIsInstance(FakePolicy(), ClosedLoopPolicy)


if __name__ == "__main__":
    unittest.main()

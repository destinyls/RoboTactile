from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Optional

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.backends.qualification_checks import (
    qualification_action,
    qualification_context,
)
from robotactile_benchmark.backends.qualification_fakes import (
    FakeUpstreamScenario,
    make_fake_runtime,
)
from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_conversion import (
    UniVTACConversionError,
)
from robotactile_benchmark.backends.univtac_isaac import (
    UniVTACIsaacBackend,
    resolve_backend_signal,
)
from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    BackendSignal,
    PolicyEpisodeContext,
)


def _actions(count: int = 1) -> np.ndarray:
    actions = np.zeros((count, 8), dtype=np.float32)
    actions[:, 0] = np.arange(1, count + 1, dtype=np.float32) / 10.0
    actions[:, 3] = -1.0
    actions[:, 7] = 0.02
    return actions


class UniVTACBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_univtac_backend_config("pull_out_key")

    def _backend(
        self,
        scenario: Optional[FakeUpstreamScenario] = None,
        *,
        config=None,
    ):
        active_config = self.config if config is None else config
        runtime, task = make_fake_runtime(active_config, scenario=scenario)
        backend = UniVTACIsaacBackend(active_config, runtime)
        return backend, task

    def _context(self, **overrides: object) -> PolicyEpisodeContext:
        values: dict[str, object] = {
            "episode_id": "episode-1",
            "task": self.config.task.task_id,
            "initial_seed": 11,
            "exogenous_seed": 999,
            "instruction": self.config.task.prompt,
            "action_spec": ACTION_SPEC,
        }
        values.update(overrides)
        return PolicyEpisodeContext(**values)

    def test_reset_ignores_stale_return_and_observes_fresh_step_zero(self) -> None:
        backend, task = self._backend()

        receipt = backend.reset(self._context())
        record = backend.observe()

        self.assertEqual(task.reset_calls, [(11, (self.config.task.prompt,))])
        self.assertEqual(task.observation_count, 1)
        self.assertEqual(record.observation.step_index, 0)
        self.assertEqual(record.observation.seed, 11)
        self.assertEqual(int(record.observation.vision["top"][0, 0, 0]), 20)
        self.assertNotEqual(receipt.simulator_state_sha256, task.stale_return_sha256)
        self.assertEqual(receipt.exogenous_seed, 999)
        self.assertNotIn(999, task.reset_seed_arguments)
        self.assertIs(receipt.diagnostics["success_check"], False)
        self.assertIs(receipt.diagnostics["plan_success"], True)
        self.assertIs(receipt.diagnostics["early_stop"], False)
        self.assertIsNone(receipt.diagnostics["n0_reset"])

    def test_reset_seed_must_match_task_construction_seed(self) -> None:
        backend, task = self._backend()
        with self.assertRaisesRegex(UniVTACContractError, "construction seed"):
            backend.reset(self._context(initial_seed=12))
        self.assertEqual(task.reset_calls, [])

    def test_dense_steps_keep_native_steps_and_each_qpos8_row_executes_once(
        self,
    ) -> None:
        backend, task = self._backend()
        backend.reset(self._context())
        backend.observe()

        batch = backend.execute(_actions(3))

        self.assertEqual(batch.executed_action_count, 3)
        self.assertEqual(
            [item.clean_record.observation.step_index for item in batch.transitions],
            [1, 2, 3],
        )
        self.assertEqual(
            [item.native_step_id for item in batch.transitions],
            [418, 419, 420],
        )
        self.assertEqual(len(task.take_action_calls), 3)
        self.assertEqual(backend.initial_native_step_id, 417)
        self.assertTrue(
            all(
                call.action_type == "qpos" and call.force
                for call in task.take_action_calls
            )
        )
        for expected, call in zip(_actions(3), task.take_action_calls):
            np.testing.assert_array_equal(call.action, expected)

    def test_complete_action_batch_is_validated_before_first_side_effect(self) -> None:
        backend, task = self._backend()
        backend.reset(self._context())
        backend.observe()
        actions = _actions(3)
        actions[-1, 0] = np.nan

        with self.assertRaisesRegex(UniVTACConversionError, "finite"):
            backend.execute(actions)

        self.assertEqual(task.take_action_calls, [])
        self.assertEqual(task.observation_count, 1)

    def test_terminal_priority_and_partial_action_prefix_are_exact(self) -> None:
        cases = (
            (
                FakeUpstreamScenario(
                    success_steps=(2,),
                    execution_failure_steps=(2,),
                    plan_failure_steps=(2,),
                    early_stop_steps=(2,),
                ),
                BackendSignal.SUCCESS,
                2,
            ),
            (
                FakeUpstreamScenario(
                    execution_failure_steps=(1,), early_stop_steps=(1,)
                ),
                BackendSignal.TASK_FAILURE,
                1,
            ),
            (
                FakeUpstreamScenario(early_stop_steps=(1,)),
                BackendSignal.EARLY_STOP,
                1,
            ),
        )
        for scenario, expected, count in cases:
            with self.subTest(expected=expected):
                backend, task = self._backend(scenario)
                backend.reset(self._context())
                backend.observe()
                batch = backend.execute(_actions(3))
                self.assertEqual(batch.transitions[-1].signal, expected)
                self.assertEqual(batch.executed_action_count, count)
                self.assertEqual(len(task.take_action_calls), count)

        self.assertEqual(
            resolve_backend_signal(
                success=False,
                execution_success=True,
                plan_success=True,
                early_stop=False,
                horizon_reached=True,
            ),
            BackendSignal.TIMEOUT,
        )

    def test_plan_success_attribute_is_checked_independently_of_return_tuple(
        self,
    ) -> None:
        backend, _ = self._backend(FakeUpstreamScenario(plan_failure_steps=(1,)))
        backend.reset(self._context())
        backend.observe()

        batch = backend.execute(_actions())

        self.assertEqual(batch.transitions[-1].signal, BackendSignal.TASK_FAILURE)
        self.assertTrue(batch.transitions[-1].diagnostics["execution_success"])
        self.assertFalse(batch.transitions[-1].diagnostics["plan_success"])

    def test_upstream_none_early_stop_is_false_only_for_two_frozen_tasks(
        self,
    ) -> None:
        for task_id in ("insert_hole", "insert_tube"):
            with self.subTest(task_id=task_id):
                config = build_univtac_backend_config(task_id)
                runtime, _ = make_fake_runtime(
                    config,
                    scenario=FakeUpstreamScenario(none_when_early_stop_false=True),
                )
                backend = UniVTACIsaacBackend(config, runtime)
                backend.reset(qualification_context(config))
                backend.observe()

                transition = backend.execute(qualification_action()).transitions[0]

                self.assertEqual(transition.signal, BackendSignal.RUNNING)
                self.assertFalse(transition.diagnostics["early_stop"])
                backend.close()

        runtime, task = make_fake_runtime(
            self.config,
            scenario=FakeUpstreamScenario(none_when_early_stop_false=True),
        )
        backend = UniVTACIsaacBackend(self.config, runtime)
        with self.assertRaisesRegex(UniVTACContractError, "check_early_stop"):
            backend.reset(self._context())
        backend.close()
        self.assertEqual(task.close_count, 1)

    def test_no_native_progress_fails_closed(self) -> None:
        backend, task = self._backend(FakeUpstreamScenario(native_step_increment=0))
        backend.reset(self._context())
        backend.observe()
        with self.assertRaisesRegex(UniVTACConversionError, "native step"):
            backend.execute(_actions())
        self.assertEqual(len(task.take_action_calls), 1)

        skipped_backend, _ = self._backend(
            FakeUpstreamScenario(native_step_increment=2)
        )
        skipped_backend.reset(self._context())
        skipped_backend.observe()
        with self.assertRaisesRegex(UniVTACConversionError, "native step"):
            skipped_backend.execute(_actions())

    def test_ee8_accepts_source_bound_stock_variable_native_steps(self) -> None:
        config = build_univtac_backend_config(
            "pull_out_key", action_spec=EE8_ACTION_SPEC
        )
        for native_step_increment in (1, 2, 47):
            with self.subTest(native_step_increment=native_step_increment):
                backend, task = self._backend(
                    FakeUpstreamScenario(native_step_increment=native_step_increment),
                    config=config,
                )
                task._robotactile_n0_fixed_cadence_enabled = False
                task._robotactile_n0_action_execution_contract = "univtac_stock_ee_v1"
                receipt = backend.reset(self._context(action_spec=EE8_ACTION_SPEC))
                backend.observe()

                transition = backend.execute(_actions()).transitions[0]

                self.assertEqual(
                    transition.native_step_id,
                    417 + native_step_increment,
                )
                self.assertEqual(
                    transition.diagnostics["native_step_delta"],
                    native_step_increment,
                )
                self.assertEqual(
                    transition.diagnostics["physics_step_delta"],
                    native_step_increment,
                )
                self.assertEqual(
                    transition.diagnostics["action_execution_contract"],
                    "univtac_stock_ee_v1",
                )
                self.assertEqual(
                    transition.diagnostics["native_step_contract"],
                    "univtac_stock_ee_variable_native_steps_v1",
                )
                self.assertIsNone(transition.diagnostics["physics_steps_per_action"])
                self.assertIsNone(receipt.diagnostics["camera_delivery_hz"])
                self.assertIsNone(receipt.diagnostics["physics_steps_per_action"])
                self.assertEqual(
                    receipt.diagnostics["action_execution_source"],
                    {
                        "action_type": "ee",
                        "method": "BaseTask.take_action",
                        "task_source_sha256": config.task.task_source_sha256,
                        "upstream_commit": config.upstream_commit,
                    },
                )
                self.assertEqual(task.take_action_calls[0].action_type, "ee")
                self.assertIsNotNone(backend.latest_canonical_joint9)
                self.assertIsNotNone(backend.latest_model_visible_qpos8)

        stalled, stalled_task = self._backend(
            FakeUpstreamScenario(native_step_increment=0),
            config=config,
        )
        stalled_task._robotactile_n0_fixed_cadence_enabled = False
        stalled_task._robotactile_n0_action_execution_contract = "univtac_stock_ee_v1"
        stalled.reset(self._context(action_spec=EE8_ACTION_SPEC))
        stalled.observe()
        with self.assertRaisesRegex(UniVTACConversionError, "native step"):
            stalled.execute(_actions())

    def test_ee8_explicit_training_cadence_enforces_two_native_steps(self) -> None:
        config = build_univtac_backend_config(
            "pull_out_key", action_spec=EE8_ACTION_SPEC
        )
        backend, task = self._backend(
            FakeUpstreamScenario(native_step_increment=2),
            config=config,
        )
        task._robotactile_n0_fixed_cadence_enabled = True
        task._robotactile_n0_action_execution_contract = (
            "robotactile_n0_training_60hz_ee_v1"
        )

        receipt = backend.reset(self._context(action_spec=EE8_ACTION_SPEC))
        backend.observe()
        transition = backend.execute(_actions()).transitions[0]

        self.assertEqual(receipt.diagnostics["camera_delivery_hz"], 60.0)
        self.assertEqual(receipt.diagnostics["physics_steps_per_action"], 2)
        self.assertEqual(
            receipt.diagnostics["action_execution_contract"],
            "robotactile_n0_training_60hz_ee_v1",
        )
        self.assertEqual(
            receipt.diagnostics["native_step_contract"],
            "fixed_physics_steps_per_action_v1",
        )
        self.assertEqual(transition.diagnostics["native_step_delta"], 2)
        self.assertEqual(transition.diagnostics["physics_step_delta"], 2)
        self.assertEqual(
            transition.diagnostics["action_execution_source"]["method"],
            "robotactile_benchmark.backends.univtac_n0_cadence.fixed_take_action",
        )

        invalid, invalid_task = self._backend(
            FakeUpstreamScenario(native_step_increment=31),
            config=config,
        )
        invalid_task._robotactile_n0_fixed_cadence_enabled = True
        invalid_task._robotactile_n0_action_execution_contract = (
            "robotactile_n0_training_60hz_ee_v1"
        )
        invalid.reset(self._context(action_spec=EE8_ACTION_SPEC))
        invalid.observe()
        with self.assertRaisesRegex(UniVTACConversionError, "native step"):
            invalid.execute(_actions())

    def test_ee8_executor_marker_mismatch_fails_closed(self) -> None:
        config = build_univtac_backend_config(
            "pull_out_key", action_spec=EE8_ACTION_SPEC
        )
        backend, task = self._backend(config=config)
        task._robotactile_n0_fixed_cadence_enabled = False
        task._robotactile_n0_action_execution_contract = (
            "robotactile_n0_training_60hz_ee_v1"
        )

        with self.assertRaisesRegex(UniVTACContractError, "marker disagrees"):
            backend.reset(self._context(action_spec=EE8_ACTION_SPEC))

    def test_policy_prompt_is_distinct_from_native_simulator_prompt(self) -> None:
        backend, task = self._backend()
        policy_prompt = "Untwist and extract a key from a lock"

        backend.reset(self._context(instruction=policy_prompt))

        self.assertNotEqual(policy_prompt, self.config.task.prompt)
        self.assertEqual(task.reset_calls, [(11, (self.config.task.prompt,))])

    def test_phase_state_is_per_slot_and_fresh_backend_reset_is_reproducible(
        self,
    ) -> None:
        first, first_task = self._backend()
        second, second_task = self._backend()
        first_receipt = first.reset(self._context())
        second_receipt = second.reset(self._context())
        first_record = first.observe()
        second_record = second.observe()

        self.assertEqual(
            first_receipt.simulator_state_sha256, second_receipt.simulator_state_sha256
        )
        self.assertEqual(
            first_record.clean_record_sha256, second_record.clean_record_sha256
        )
        self.assertEqual(
            first_record.provenance_for("left").phase.value,
            "free",
        )
        self.assertEqual(
            first_record.provenance_for("right").phase.value,
            "contact_onset",
        )

        next_record = first.execute(_actions()).transitions[0].clean_record
        self.assertEqual(next_record.provenance_for("left").phase.value, "free")
        self.assertEqual(
            next_record.provenance_for("right").phase.value,
            "sustained_contact",
        )
        self.assertEqual((first_task.reset_count, second_task.reset_count), (1, 1))

    def test_handshake_tampering_single_use_and_close_are_fail_closed(self) -> None:
        runtime, _ = make_fake_runtime(self.config)
        tampered = replace(runtime.handshake, config_sha256="0" * 64)
        with self.assertRaisesRegex(UniVTACContractError, "config"):
            UniVTACIsaacBackend(self.config, replace(runtime, handshake=tampered))

        backend, task = self._backend()
        backend.reset(self._context())
        with self.assertRaisesRegex(UniVTACContractError, "single-use"):
            backend.reset(self._context())
        backend.close()
        backend.close()
        self.assertEqual(task.close_count, 1)

    def test_coordinated_config_and_handshake_tampering_is_rejected(self) -> None:
        upper = (100.0,) + self.config.action_upper_bounds[1:]
        tampered_configs = (
            replace(self.config, action_upper_bounds=upper),
            replace(self.config, head_shape=(1, 1, 3)),
            replace(
                self.config,
                phase_tracker=replace(
                    self.config.phase_tracker,
                    on_threshold_mm=0.9,
                ),
            ),
            replace(self.config, registry_resource_sha256="0" * 64),
        )
        for config in tampered_configs:
            with self.subTest(config_sha256=config.sha256):
                runtime, task = make_fake_runtime(config)
                with self.assertRaisesRegex(UniVTACContractError, "packaged registry"):
                    UniVTACIsaacBackend(config, runtime)
                runtime.close_runtime()
                self.assertEqual(task.close_count, 1)


if __name__ == "__main__":
    unittest.main()

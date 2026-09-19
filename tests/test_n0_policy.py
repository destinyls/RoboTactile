from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Callable, Mapping

import numpy as np

from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    BackendSignal,
    ClosedLoopRunSpec,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.fakes import DeterministicFakeBackend
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.closed_loop.runner import run_closed_loop_trial
from robotactile_benchmark.contracts import ObservationRecord, canonical_hash
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.n0 import N0Policy
from robotactile_benchmark.transport.n0_client import N0ClientState
from robotactile_benchmark.transport.n0_contracts import (
    N0GroundingFrame,
    N0Handshake,
)
from robotactile_benchmark.trials import (
    Condition,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


def make_record(step_index: int) -> ObservationRecord:
    return replace(
        make_synthetic_episode()[step_index].observation,
        proprio=np.arange(8, dtype=np.float32),
    )


def make_context() -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="synthetic-episode-v1",
        task="insert_HDMI",
        initial_seed=1001,
        exogenous_seed=2002,
        instruction="insert HDMI",
        action_spec=ACTION_SPEC,
    )


class FakeClient:
    def __init__(
        self,
        *,
        checkpoint_sha256: str | None = None,
        config_sha256: str | None = None,
    ) -> None:
        self.state = N0ClientState.CONNECTED
        self.handshake_count = 0
        self.reset_count = 0
        self.infer_count = 0
        self.commit_count = 0
        self.abort_count = 0
        self.close_count = 0
        identity = make_identity()
        self._handshake_identity = N0Handshake(
            schema_version="robotactile-n0-v1",
            source_commit="9" * 40,
            checkpoint_sha256=(
                identity.checkpoint_sha256
                if checkpoint_sha256 is None
                else checkpoint_sha256
            ),
            config_sha256=(
                identity.config_sha256 if config_sha256 is None else config_sha256
            ),
            normalizer_sha256="c" * 64,
            serve_bundle_sha256="d" * 64,
            action_spec="qpos8_next_step",
            action_semantics="absolute",
            action_dim=8,
            native_action_shape=(8, 2, 4),
            simulator_action_shape=(8, 8),
            camera_keys=("top", "wrist_l"),
            tactile_keys=("tactile_a", "tactile_b"),
            camera_shapes=((12, 16, 3), (12, 16, 3)),
            tactile_shapes=((32, 32, 3), (32, 32, 3)),
            tactile_optional=False,
            action_lower_bounds=(-100.0,) * 8,
            action_upper_bounds=(100.0,) * 8,
            prompt_manifest_sha256="e" * 64,
            requires_commit=True,
            cold_seed_mode="free",
            no_cold_frame_skip=True,
            server_epoch="epoch-001",
        )
        self.last_grounding_frames: tuple[N0GroundingFrame, ...] = ()
        self.last_infer_observation: Mapping[str, object] | None = None

    @property
    def handshake_identity(self) -> N0Handshake:
        return self._handshake_identity

    def handshake(self) -> None:
        self.handshake_count += 1
        self.state = N0ClientState.RESET_REQUIRED

    def reset(self, episode_id: str, prompt: str, task: str, seed: int) -> None:
        self.reset_count += 1
        self.state = N0ClientState.READY

    def infer(self, step_index: int, observation: Mapping[str, object]) -> np.ndarray:
        self.infer_count += 1
        self.last_infer_observation = observation
        self.state = N0ClientState.AWAITING_EXECUTION
        return np.arange(64, dtype=np.float32).reshape(8, 2, 4)

    def commit(
        self,
        *,
        step_index: int,
        native_action: np.ndarray,
        executed_actions: np.ndarray,
        grounding_frames: tuple[N0GroundingFrame, ...],
    ) -> None:
        self.commit_count += 1
        self.last_grounding_frames = grounding_frames
        self.state = N0ClientState.READY

    def terminal(self) -> None:
        self.state = N0ClientState.RESET_REQUIRED

    def abort(self, reason_code: str) -> None:
        self.abort_count += 1
        self.state = N0ClientState.RESET_REQUIRED

    def close(self) -> None:
        self.close_count += 1
        self.state = N0ClientState.CLOSED


def make_identity() -> PolicyIdentity:
    return PolicyIdentity(
        system_id="n0-twam-touch-v1",
        checkpoint_sha256=canonical_hash("n0-checkpoint"),
        config_sha256=canonical_hash("n0-config"),
        action_spec=ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def recording_client_factory(
    created: list[FakeClient],
) -> Callable[[], FakeClient]:
    def factory() -> FakeClient:
        client = FakeClient()
        created.append(client)
        return client

    return factory


class N0PolicyTests(unittest.TestCase):
    def test_structural_absence_preflight_makes_zero_transport_calls(self) -> None:
        identity = make_identity()
        for operator_id in ("A1_stream_absence", "A2_frame_erasure"):
            fault = FaultManifest(
                operator_id=operator_id,
                severity_level=3,
                operator_seed=23,
                start_index=1,
                stop_index=4,
                sensor_slots=("left",),
                observability=Observability.DECLARED,
                parameters={},
            )
            trial = TrialManifest(
                task="insert_HDMI",
                initial_seed=1001,
                exogenous_seed=2002,
                condition=Condition.FAULTED,
                base_system_id=identity.system_id,
                executed_system_id=identity.system_id,
                dataset_sha256="f" * 64,
                base_system_manifest_sha256=system_manifest_hash(
                    identity.system_id,
                    identity.checkpoint_sha256,
                    identity.config_sha256,
                    identity.action_spec,
                ),
                checkpoint_sha256=identity.checkpoint_sha256,
                config_sha256=identity.config_sha256,
                action_spec=identity.action_spec,
                fault_manifest_sha256=fault.sha256,
                matched_no_touch_system_id=None,
            )
            created: list[FakeClient] = []

            backend = DeterministicFakeBackend(make_synthetic_episode())
            result = run_closed_loop_trial(
                trial,
                ClosedLoopRunSpec(
                    prompt="insert HDMI",
                    success_predicate_id="fake-success-v1",
                    max_control_cycles=2,
                    max_observation_steps=8,
                    execute_action_steps=8,
                    wall_timeout_s=5.0,
                ),
                backend,
                N0Policy(identity, recording_client_factory(created)),
                fault_manifest=fault,
            )
            with self.subTest(operator_id=operator_id):
                self.assertEqual(
                    result.terminal_status, TerminalStatus.UNSUPPORTED_CONTRACT
                )
                self.assertEqual(backend.reset_count, 0)
                self.assertEqual(created, [])

    def test_lazy_client_and_exact_model_visible_routing(self) -> None:
        created: list[FakeClient] = []

        def factory() -> FakeClient:
            client = FakeClient()
            created.append(client)
            return client

        policy = N0Policy(make_identity(), factory)
        self.assertIsInstance(policy, ClosedLoopPolicy)
        self.assertEqual(created, [])

        policy.reset(make_context())
        plan = policy.infer(make_record(0))

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].handshake_count, 1)
        self.assertEqual(
            plan.actions[0].tolist(), [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0]
        )
        self.assertEqual(plan.actions.shape, (8, 8))
        payload = created[0].last_infer_observation
        assert payload is not None
        encoded = {
            "vision": payload["vision"],
            "tactile": payload["tactile"],
            "proprio": payload["proprio"],
        }
        self.assertEqual(payload["observation_digest"], canonical_hash(encoded))

    def test_reset_rejects_client_artifact_identity_before_server_reset(self) -> None:
        for field, client in (
            ("checkpoint", FakeClient(checkpoint_sha256="0" * 64)),
            ("config", FakeClient(config_sha256="0" * 64)),
        ):
            policy = N0Policy(make_identity(), lambda client=client: client)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                policy.reset(make_context())
            self.assertEqual(client.handshake_count, 1)
            self.assertEqual(client.reset_count, 0)

    def test_complete_running_execution_commits_exactly_once(self) -> None:
        client = FakeClient()
        policy = N0Policy(make_identity(), lambda: client)
        policy.reset(make_context())
        plan = policy.infer(make_record(0))
        delivered = tuple(make_record(index) for index in range(1, 9))
        policy.commit(
            PolicyExecution(
                action_plan_sha256=plan.sha256,
                executed_actions=plan.actions,
                delivered_observations=delivered,
                terminal_signal=BackendSignal.RUNNING,
            )
        )
        self.assertEqual(client.commit_count, 1)
        self.assertEqual(client.state, N0ClientState.READY)
        self.assertEqual(len(client.last_grounding_frames), 8)
        for expected_step, (frame, observation) in enumerate(
            zip(client.last_grounding_frames, delivered), start=1
        ):
            self.assertIsInstance(frame, N0GroundingFrame)
            self.assertEqual(frame.step_index, expected_step)
            self.assertFalse(frame.top.flags.writeable)
            self.assertFalse(frame.proprio.flags.writeable)
            self.assertEqual(frame.top.dtype, np.uint8)
            self.assertEqual(frame.proprio.dtype, np.float32)
            np.testing.assert_array_equal(frame.top, observation.vision["top"])
            np.testing.assert_array_equal(frame.wrist_l, observation.vision["wrist_l"])
            left = observation.sensor("left").payload
            right = observation.sensor("right").payload
            assert left is not None and right is not None
            np.testing.assert_array_equal(frame.tactile_a, left)
            np.testing.assert_array_equal(frame.tactile_b, right)
            np.testing.assert_array_equal(frame.proprio, observation.proprio)

    def test_running_requires_complete_exact_chunk_but_terminal_never_commits(
        self,
    ) -> None:
        client = FakeClient()
        policy = N0Policy(make_identity(), lambda: client)
        policy.reset(make_context())
        plan = policy.infer(make_record(0))

        partial = PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions[:7],
            delivered_observations=tuple(make_record(index) for index in range(1, 8)),
            terminal_signal=BackendSignal.RUNNING,
        )
        with self.assertRaisesRegex(RuntimeError, "complete"):
            policy.commit(partial)
        self.assertEqual(client.commit_count, 0)

        policy.abort("partial_running")
        policy.reset(make_context())
        terminal_plan = policy.infer(make_record(0))
        terminal = PolicyExecution(
            action_plan_sha256=terminal_plan.sha256,
            executed_actions=terminal_plan.actions[:3],
            delivered_observations=tuple(make_record(index) for index in range(1, 4)),
            terminal_signal=BackendSignal.SUCCESS,
        )
        policy.commit(terminal)
        self.assertEqual(client.commit_count, 0)
        self.assertEqual(client.state, N0ClientState.RESET_REQUIRED)

        policy.reset(make_context())
        full_terminal_plan = policy.infer(make_record(0))
        full_terminal = PolicyExecution(
            action_plan_sha256=full_terminal_plan.sha256,
            executed_actions=full_terminal_plan.actions,
            delivered_observations=tuple(make_record(index) for index in range(1, 9)),
            terminal_signal=BackendSignal.SUCCESS,
        )
        policy.commit(full_terminal)
        self.assertEqual(client.commit_count, 0)
        self.assertEqual(client.state, N0ClientState.RESET_REQUIRED)

    def test_running_delivery_steps_must_be_dense_and_end_at_next_decision(
        self,
    ) -> None:
        client = FakeClient()
        policy = N0Policy(make_identity(), lambda: client)
        policy.reset(make_context())
        plan = policy.infer(make_record(0))
        delivered = tuple(make_record(index) for index in (1, 2, 3, 4, 5, 6, 7, 9))
        execution = PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=delivered,
            terminal_signal=BackendSignal.RUNNING,
        )
        with self.assertRaisesRegex(ValueError, "dense"):
            policy.commit(execution)
        self.assertEqual(client.commit_count, 0)

    def test_running_delivery_identity_must_match_reset_context(self) -> None:
        for field, value in (
            ("episode_id", "wrong-episode"),
            ("task", "wrong-task"),
            ("seed", 999),
        ):
            client = FakeClient()
            policy = N0Policy(make_identity(), lambda client=client: client)
            policy.reset(make_context())
            plan = policy.infer(make_record(0))
            delivered = list(make_record(index) for index in range(1, 9))
            delivered[3] = replace(delivered[3], **{field: value})
            execution = PolicyExecution(
                action_plan_sha256=plan.sha256,
                executed_actions=plan.actions,
                delivered_observations=tuple(delivered),
                terminal_signal=BackendSignal.RUNNING,
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "identity"),
            ):
                policy.commit(execution)
            self.assertEqual(client.commit_count, 0)

    def test_exact_proprio_and_tactile_absence_fail_before_client_infer(self) -> None:
        client = FakeClient()
        policy = N0Policy(make_identity(), lambda: client)
        policy.reset(make_context())
        invalid_proprio = replace(
            make_record(0), proprio=np.arange(9, dtype=np.float32)
        )
        with self.assertRaises(ValueError):
            policy.infer(invalid_proprio)
        self.assertEqual(client.infer_count, 0)

        nonfinite_proprio = make_record(0)
        nonfinite_proprio.proprio.setflags(write=True)
        nonfinite_proprio.proprio.fill(np.nan)
        nonfinite_proprio.proprio.setflags(write=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            policy.infer(nonfinite_proprio)
        self.assertEqual(client.infer_count, 0)

        missing = replace(
            make_record(0),
            tactile=tuple(
                sensor.without_payload(True) for sensor in make_record(0).tactile
            ),
        )
        with self.assertRaises(ValueError):
            policy.infer(missing)
        self.assertEqual(client.infer_count, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Mapping
from unittest.mock import patch

import numpy as np

import robotactile_benchmark.policies as policy_exports
from robotactile_benchmark.closed_loop.contracts import (
    ACTION_SPEC,
    BackendSignal,
    PolicyEpisodeContext,
    PolicyExecution,
    PolicyIdentity,
)
from robotactile_benchmark.closed_loop.interfaces import ClosedLoopPolicy
from robotactile_benchmark.contracts import ObservationRecord
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.policies.univtac_official_act import (
    OfficialACTProfile,
    OfficialUniVTACACTPolicy,
)


class RecordingOfficialRuntime:
    def __init__(self, action: np.ndarray | None = None) -> None:
        self.action = (
            np.arange(8, dtype=np.float32)[None, :] if action is None else action
        )
        self.reset_count = 0
        self.close_count = 0
        self.inputs: list[Mapping[str, object]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def get_action(self, observation: Mapping[str, object]) -> np.ndarray:
        self.inputs.append(observation)
        return self.action

    def close(self) -> None:
        self.close_count += 1


def _identity(profile: OfficialACTProfile) -> PolicyIdentity:
    tactile = profile is OfficialACTProfile.UNIVTAC
    return PolicyIdentity(
        system_id=f"official-act-{profile.value}",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        action_spec=ACTION_SPEC,
        consumes_tactile=tactile,
        supports_structural_absence=not tactile,
    )


def _context() -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="synthetic-episode-v1",
        task="pull_out_key",
        initial_seed=1001,
        exogenous_seed=2002,
        instruction="pull out key",
        action_spec=ACTION_SPEC,
    )


def _record(step_index: int = 0) -> ObservationRecord:
    source = make_synthetic_episode()[step_index].observation
    return replace(
        source,
        task="pull_out_key",
        proprio=np.arange(8, dtype=np.float32),
        vision={"top": np.full((7, 11, 3), 127, dtype=np.uint8)},
    )


class OfficialUniVTACACTPolicyTests(unittest.TestCase):
    def test_dependency_light_package_exports_official_adapter_contract(self) -> None:
        self.assertIs(policy_exports.OfficialUniVTACACTPolicy, OfficialUniVTACACTPolicy)
        self.assertIs(policy_exports.OfficialACTProfile, OfficialACTProfile)
        self.assertTrue(hasattr(policy_exports, "OfficialUniVTACACTArtifactManifest"))
        self.assertTrue(hasattr(policy_exports, "load_official_univtac_act_policy"))

    def test_profile_capabilities_must_match_policy_identity(self) -> None:
        runtime = RecordingOfficialRuntime()
        for profile in OfficialACTProfile:
            with (
                self.subTest(profile=profile),
                self.assertRaisesRegex(ValueError, "capabil"),
            ):
                OfficialUniVTACACTPolicy(
                    _identity(
                        OfficialACTProfile.VISION_ONLY
                        if profile is OfficialACTProfile.UNIVTAC
                        else OfficialACTProfile.UNIVTAC
                    ),
                    runtime,
                    artifact_task="pull_out_key",
                    profile=profile,
                )

    def test_vision_only_does_not_read_tactile_even_when_payloads_exist(self) -> None:
        runtime = RecordingOfficialRuntime()
        policy = OfficialUniVTACACTPolicy(
            _identity(OfficialACTProfile.VISION_ONLY),
            runtime,
            artifact_task="pull_out_key",
            profile=OfficialACTProfile.VISION_ONLY,
        )
        policy.reset(_context())
        observation = _record()

        with patch.object(
            ObservationRecord,
            "sensor",
            side_effect=AssertionError("vision-only touched tactile"),
        ):
            plan = policy.infer(observation)

        self.assertIsInstance(policy, ClosedLoopPolicy)
        self.assertEqual(plan.actions.shape, (1, 8))
        self.assertEqual(set(runtime.inputs[0]), {"cam_high", "qpos"})

    def test_univtac_profile_exposes_delivered_tactile_fault_to_runtime(self) -> None:
        runtime = RecordingOfficialRuntime()
        policy = OfficialUniVTACACTPolicy(
            _identity(OfficialACTProfile.UNIVTAC),
            runtime,
            artifact_task="pull_out_key",
            profile=OfficialACTProfile.UNIVTAC,
        )
        clean = _record()
        altered_sensors = tuple(
            replace(sensor, payload=np.zeros_like(sensor.payload))
            if sensor.slot_id == "left" and sensor.payload is not None
            else sensor
            for sensor in clean.tactile
        )
        faulted = replace(clean, tactile=altered_sensors)

        policy.reset(_context())
        policy.infer(clean)
        policy.reset(_context())
        policy.infer(faulted)

        clean_left = np.asarray(runtime.inputs[0]["tac_left"])
        faulted_left = np.asarray(runtime.inputs[1]["tac_left"])
        self.assertFalse(np.array_equal(clean_left, faulted_left))
        self.assertEqual(
            set(runtime.inputs[0]),
            {"cam_high", "tac_left", "tac_right", "qpos"},
        )

    def test_runtime_output_is_exact_finite_float32_one_by_eight(self) -> None:
        invalid = (
            np.zeros((8,), dtype=np.float32),
            np.zeros((1, 8), dtype=np.float64),
            np.full((1, 8), np.nan, dtype=np.float32),
        )
        for output in invalid:
            runtime = RecordingOfficialRuntime(output)
            policy = OfficialUniVTACACTPolicy(
                _identity(OfficialACTProfile.VISION_ONLY),
                runtime,
                artifact_task="pull_out_key",
                profile=OfficialACTProfile.VISION_ONLY,
            )
            policy.reset(_context())
            with (
                self.subTest(output=output),
                self.assertRaises((TypeError, ValueError)),
            ):
                policy.infer(_record())

    def test_commit_and_close_follow_closed_loop_lifecycle(self) -> None:
        runtime = RecordingOfficialRuntime()
        policy = OfficialUniVTACACTPolicy(
            _identity(OfficialACTProfile.VISION_ONLY),
            runtime,
            artifact_task="pull_out_key",
            profile=OfficialACTProfile.VISION_ONLY,
        )
        policy.reset(_context())
        plan = policy.infer(_record())
        policy.commit(
            PolicyExecution(
                action_plan_sha256=plan.sha256,
                executed_actions=plan.actions,
                delivered_observations=(_record(1),),
                terminal_signal=BackendSignal.RUNNING,
            )
        )
        policy.infer(_record(1))
        policy.close()
        policy.close()

        self.assertEqual(runtime.reset_count, 1)
        self.assertEqual(runtime.close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            policy.reset(_context())


if __name__ == "__main__":
    unittest.main()

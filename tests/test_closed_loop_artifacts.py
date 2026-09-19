"""Artifact round-trip and tamper tests for one captured closed-loop trial."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from robotactile_benchmark.closed_loop import runner as runner_module
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.closed_loop.contracts import ClosedLoopRunSpec
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import Condition, TrialManifest, system_manifest_hash


def _fault() -> FaultManifest:
    return FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=41,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )


def _trial(fault: FaultManifest) -> TrialManifest:
    system_id = "deterministic-fake-policy-v1"
    checkpoint_sha256 = "b" * 64
    config_sha256 = "c" * 64
    return TrialManifest(
        task="insert_HDMI",
        initial_seed=10,
        exogenous_seed=20,
        condition=Condition.FAULTED,
        base_system_id=system_id,
        executed_system_id=system_id,
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            system_id,
            checkpoint_sha256,
            config_sha256,
            "qpos8_next_step",
        ),
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=fault.sha256,
        matched_no_touch_system_id=None,
    )


def _spec() -> ClosedLoopRunSpec:
    return ClosedLoopRunSpec(
        prompt="insert the cable",
        success_predicate_id="fake-success-v1",
        max_control_cycles=5,
        max_observation_steps=6,
        execute_action_steps=1,
        wall_timeout_s=5.0,
    )


def _capture() -> tuple[
    TrialManifest,
    ClosedLoopRunSpec,
    FaultManifest,
    ClosedLoopExecutionEvidence,
    DeterministicFakePolicy,
]:
    fault = _fault()
    trial = _trial(fault)
    spec = _spec()
    policy = DeterministicFakePolicy.for_trial(trial)
    evidence = runner_module.run_closed_loop_trial_with_evidence(
        trial,
        spec,
        DeterministicFakeBackend(make_synthetic_episode(length=10)),
        policy,
        fault_manifest=fault,
    )
    return trial, spec, fault, evidence, policy


def _artifact_api() -> Any:
    try:
        from robotactile_benchmark.closed_loop import artifacts
    except ImportError as error:
        raise AssertionError("Task 4 artifact API is missing") from error
    return artifacts


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ClosedLoopCaptureTests(unittest.TestCase):
    def test_capture_api_exposes_exact_result_hash_inputs(self) -> None:
        """Removing the capture bridge must break this evaluator evidence test."""

        self.assertTrue(
            hasattr(runner_module, "run_closed_loop_trial_with_evidence"),
            "Task 4 capture API is missing",
        )
        fault = _fault()
        trial = _trial(fault)
        policy = DeterministicFakePolicy.for_trial(trial)
        backend = DeterministicFakeBackend(make_synthetic_episode(length=10))

        captured = runner_module.run_closed_loop_trial_with_evidence(
            trial,
            _spec(),
            backend,
            policy,
            fault_manifest=fault,
        )

        self.assertEqual(
            captured.result.action_trace_sha256,
            captured.action_trace_sha256,
        )
        self.assertEqual(
            captured.result.delivered_trace_sha256,
            captured.finalization.delivered_trace_sha256,
        )
        self.assertEqual(
            captured.result.clean_trace_sha256,
            captured.finalization.clean_trace_sha256,
        )
        self.assertTrue(captured.finalization.validation.passed)

    def test_capture_keeps_evaluator_fields_outside_policy_boundary(self) -> None:
        """Passing a capture sink must not leak provenance into policy inputs."""

        _, _, _, captured, policy = _capture()

        self.assertTrue(policy.inferred)
        self.assertTrue(
            all(
                type(observation).__name__ == "ObservationRecord"
                for observation in policy.inferred
            )
        )
        self.assertFalse(
            any(hasattr(observation, "provenance") for observation in policy.inferred)
        )
        self.assertTrue(captured.finalization.clean_records[0].provenance)

    def test_capture_is_frozen_after_close_stage_result_is_settled(self) -> None:
        """Capturing before close-stage status replacement must break this test."""

        fault = _fault()
        trial = _trial(fault)
        policy = DeterministicFakePolicy.for_trial(trial, raise_on_close=True)
        captured = runner_module.run_closed_loop_trial_with_evidence(
            trial,
            _spec(),
            DeterministicFakeBackend(make_synthetic_episode(length=10)),
            policy,
            fault_manifest=fault,
        )

        self.assertEqual(captured.result.failure_stage, "close")
        self.assertEqual(captured.result.failure_code, "policy_close_failed")
        self.assertEqual(policy.close_count, 1)
        with self.assertRaises(ValueError):
            captured.action_entries[0].executed_actions[0, 0] = np.float32(3.0)


class ClosedLoopArtifactRoundTripTests(unittest.TestCase):
    def test_bundle_round_trips_into_typed_contracts(self) -> None:
        """Skipping typed reconstruction or cross-links must break this test."""

        artifacts = _artifact_api()
        trial, spec, fault, evidence, _ = _capture()
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path = Path(temporary) / "bundle"
            written = artifacts.write_closed_loop_bundle(
                bundle_path,
                trial=trial,
                run_spec=spec,
                fault_manifest=fault,
                evidence=evidence,
            )
            loaded = artifacts.load_closed_loop_bundle(bundle_path)

        self.assertEqual(loaded.root_receipt_sha256, written.root_receipt_sha256)
        self.assertEqual(loaded.trial, trial)
        self.assertEqual(loaded.run_spec, spec)
        self.assertEqual(loaded.fault_manifest, fault)
        self.assertEqual(loaded.result, evidence.result)
        self.assertEqual(
            loaded.finalization.delivered_trace_sha256,
            evidence.finalization.delivered_trace_sha256,
        )
        self.assertEqual(
            len(loaded.action_entries), evidence.result.control_cycle_count
        )
        for actual, expected in zip(loaded.action_entries, evidence.action_entries):
            self.assertEqual(actual.action_plan_sha256, expected.action_plan_sha256)
            self.assertTrue(
                np.array_equal(actual.executed_actions, expected.executed_actions)
            )

    def test_independent_bundle_directories_are_byte_identical(self) -> None:
        """Nondeterministic serialization or NPY headers must break this test."""

        artifacts = _artifact_api()
        first = _capture()
        second = _capture()
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "first"
            second_path = Path(temporary) / "second"
            artifacts.write_closed_loop_bundle(
                first_path,
                trial=first[0],
                run_spec=first[1],
                fault_manifest=first[2],
                evidence=first[3],
            )
            artifacts.write_closed_loop_bundle(
                second_path,
                trial=second[0],
                run_spec=second[1],
                fault_manifest=second[2],
                evidence=second[3],
            )
            self.assertEqual(_inventory(first_path), _inventory(second_path))

    def test_identical_nonempty_bundle_is_a_verified_noop(self) -> None:
        """A valid identical target should be reusable without rewriting files."""

        artifacts = _artifact_api()
        trial, spec, fault, evidence, _ = _capture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            first = artifacts.write_closed_loop_bundle(
                output,
                trial=trial,
                run_spec=spec,
                fault_manifest=fault,
                evidence=evidence,
            )
            before = _inventory(output)
            second = artifacts.write_closed_loop_bundle(
                output,
                trial=trial,
                run_spec=spec,
                fault_manifest=fault,
                evidence=evidence,
            )
            after = _inventory(output)

        self.assertEqual(first.root_receipt_sha256, second.root_receipt_sha256)
        self.assertEqual(before, after)

    def test_cli_writes_reloads_and_emits_stable_bounded_summary(self) -> None:
        """The CLI must use its strict loader and never overclaim evidence."""

        artifacts = _artifact_api()
        from robotactile_benchmark.cli import main

        summaries = []
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("first", "second"):
                output = Path(temporary) / name
                stream = StringIO()
                with redirect_stdout(stream):
                    self.assertEqual(
                        main(["closed-loop-smoke", "--output", str(output)]),
                        0,
                    )
                summary = json.loads(stream.getvalue())
                loaded = artifacts.load_closed_loop_bundle(output)
                self.assertEqual(
                    summary["root_receipt_sha256"],
                    loaded.root_receipt_sha256,
                )
                summaries.append(summary)

        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(
            summaries[0]["evidence_level"],
            "deterministic_cpu_fake_closed_loop",
        )
        self.assertEqual(summaries[0]["condition"], "faulted")
        self.assertTrue(summaries[0]["validation_passed"])
        forbidden = {"UniVTAC", "Isaac", "policy benchmark", "task result"}
        self.assertFalse(any(term in json.dumps(summaries[0]) for term in forbidden))


if __name__ == "__main__":
    unittest.main()

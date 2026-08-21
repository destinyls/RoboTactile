from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Mapping
from unittest.mock import patch

import numpy as np

import robotactile_benchmark.policies as policy_exports
import robotactile_benchmark.policies.act as act_module
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
from robotactile_benchmark.policies.act import (
    StrictACTPolicy,
    preprocess_camera,
    preprocess_tactile,
)
from robotactile_benchmark.policies.act_loading import (
    ArtifactUnavailableError,
    load_matched_no_touch_policy,
    load_strict_act_policy,
)
from robotactile_benchmark.trials import (
    Condition,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


class RecordingRuntime:
    def __init__(self, action: np.ndarray | None = None) -> None:
        self.action = (
            np.arange(8, dtype=np.float32)[None, :] if action is None else action
        )
        self.reset_count = 0
        self.infer_count = 0
        self.inputs: list[Mapping[str, object]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def get_action(self, observation: Mapping[str, object]) -> np.ndarray:
        self.infer_count += 1
        self.inputs.append(observation)
        return self.action


def make_context() -> PolicyEpisodeContext:
    return PolicyEpisodeContext(
        episode_id="synthetic-episode-v1",
        task="insert_HDMI",
        initial_seed=1001,
        exogenous_seed=2002,
        instruction="insert HDMI",
        action_spec=ACTION_SPEC,
    )


def make_record(step_index: int = 0) -> ObservationRecord:
    observation = make_synthetic_episode()[step_index].observation
    return replace(
        observation,
        proprio=np.arange(8, dtype=np.float32),
        vision={"top": np.full((7, 11, 3), 255, dtype=np.uint8)},
    )


def make_identity(*, tactile: bool = True) -> PolicyIdentity:
    return PolicyIdentity(
        system_id="act-touch-v1" if tactile else "act-no-touch-v1",
        checkpoint_sha256=canonical_hash("act-checkpoint"),
        config_sha256=canonical_hash("act-config"),
        action_spec=ACTION_SPEC,
        consumes_tactile=tactile,
        supports_structural_absence=not tactile,
    )


class StrictACTPolicyTests(unittest.TestCase):
    def test_dependency_light_import_loads_neither_torch_nor_upstream_act(self) -> None:
        code = (
            "import sys; import robotactile_benchmark.policies; "
            "assert 'torch' not in sys.modules; "
            "assert 'deployment.ACTStrict.deploy_policy' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_numpy_preprocessing_matches_qualified_torchvision_downsampling(
        self,
    ) -> None:
        try:
            import torch
            from torchvision import transforms
        except ImportError:
            self.skipTest("optional ACT qualification dependencies are unavailable")
        image = np.random.default_rng(7).integers(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        )
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        expected_tactile = transforms.Resize((256, 256))(tensor).numpy()
        expected_camera = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(torch.from_numpy(expected_tactile.copy())).numpy()

        np.testing.assert_allclose(
            preprocess_tactile(image, "left"), expected_tactile, atol=5e-7, rtol=0.0
        )
        np.testing.assert_allclose(
            preprocess_camera(image), expected_camera, atol=3e-6, rtol=0.0
        )

    def test_routes_only_observation_with_exact_preprocessing(self) -> None:
        runtime = RecordingRuntime()
        policy = StrictACTPolicy(make_identity(), runtime, artifact_task="insert_HDMI")
        policy.reset(make_context())

        plan = policy.infer(make_record())

        self.assertIsInstance(policy, ClosedLoopPolicy)
        self.assertEqual(
            plan.actions.tolist(), [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
        )
        encoded = runtime.inputs[0]
        self.assertEqual(set(encoded), {"cam_high", "tac_left", "tac_right", "qpos"})
        self.assertEqual(np.asarray(encoded["cam_high"]).shape, (3, 256, 256))
        expected_camera = np.asarray(
            [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            np.asarray(encoded["cam_high"])[:, 100, 100], expected_camera, rtol=1e-6
        )
        left_source = make_record().sensor("left").payload
        assert left_source is not None
        self.assertAlmostEqual(
            float(np.asarray(encoded["tac_left"])[0].mean()),
            float(left_source[..., 0].mean() / 255.0),
            places=2,
        )
        np.testing.assert_array_equal(encoded["qpos"], np.arange(8, dtype=np.float32))

    def test_malformed_inputs_fail_before_runtime_side_effects(self) -> None:
        for invalid in (
            replace(make_record(), proprio=np.arange(9, dtype=np.float32)),
            replace(
                make_record(), vision={"wrist_l": np.zeros((2, 2, 3), dtype=np.uint8)}
            ),
            replace(
                make_record(),
                tactile=tuple(
                    sensor.without_payload(declared=True)
                    if sensor.slot_id == "left"
                    else sensor
                    for sensor in make_record().tactile
                ),
            ),
        ):
            runtime = RecordingRuntime()
            policy = StrictACTPolicy(
                make_identity(), runtime, artifact_task="insert_HDMI"
            )
            policy.reset(make_context())
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                policy.infer(invalid)
            self.assertEqual(runtime.infer_count, 0)

        runtime = RecordingRuntime()
        policy = StrictACTPolicy(make_identity(), runtime, artifact_task="insert_HDMI")
        policy.reset(make_context())
        with self.assertRaises(TypeError):
            policy.infer(make_synthetic_episode()[0])  # type: ignore[arg-type]
        self.assertEqual(runtime.infer_count, 0)

    def test_mutated_nonfinite_proprio_fails_before_runtime_side_effects(self) -> None:
        runtime = RecordingRuntime()
        policy = StrictACTPolicy(make_identity(), runtime, artifact_task="insert_HDMI")
        policy.reset(make_context())
        observation = make_record()
        observation.proprio.setflags(write=True)
        observation.proprio.fill(np.nan)
        observation.proprio.setflags(write=False)

        with self.assertRaisesRegex(ValueError, "finite"):
            policy.infer(observation)
        self.assertEqual(runtime.infer_count, 0)

    def test_artifact_task_mismatch_fails_before_runtime_reset(self) -> None:
        runtime = RecordingRuntime()
        policy = StrictACTPolicy(make_identity(), runtime, artifact_task="insert_HDMI")
        wrong_context = replace(make_context(), task="wrong-task")

        with self.assertRaisesRegex(ValueError, "task"):
            policy.reset(wrong_context)
        self.assertEqual(runtime.reset_count, 0)

    def test_output_must_be_exact_finite_float32_one_by_eight(self) -> None:
        invalid_outputs = (
            np.zeros((8,), dtype=np.float32),
            np.zeros((1, 8), dtype=np.float64),
            np.full((1, 8), np.nan, dtype=np.float32),
        )
        for output in invalid_outputs:
            policy = StrictACTPolicy(
                make_identity(),
                RecordingRuntime(output),
                artifact_task="insert_HDMI",
            )
            policy.reset(make_context())
            with (
                self.subTest(output=output),
                self.assertRaises((TypeError, ValueError)),
            ):
                policy.infer(make_record())

    def test_invalid_runtime_result_aborts_until_an_explicit_reset(self) -> None:
        runtime = RecordingRuntime(np.zeros((8,), dtype=np.float32))
        policy = StrictACTPolicy(make_identity(), runtime, artifact_task="insert_HDMI")
        policy.reset(make_context())
        with self.assertRaises(ValueError):
            policy.infer(make_record())

        runtime.action = np.zeros((1, 8), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            policy.infer(make_record())
        self.assertEqual(runtime.infer_count, 1)

        policy.reset(make_context())
        policy.infer(make_record())
        self.assertEqual(runtime.infer_count, 2)

    def test_lifecycle_advances_runtime_once_and_commit_never_reruns_it(self) -> None:
        runtime = RecordingRuntime()
        policy = StrictACTPolicy(make_identity(), runtime, artifact_task="insert_HDMI")
        policy.reset(make_context())
        plan = policy.infer(make_record())
        execution = PolicyExecution(
            action_plan_sha256=plan.sha256,
            executed_actions=plan.actions,
            delivered_observations=(make_record(1),),
            terminal_signal=BackendSignal.RUNNING,
        )
        policy.commit(execution)

        self.assertEqual(runtime.reset_count, 1)
        self.assertEqual(runtime.infer_count, 1)
        with self.assertRaises(RuntimeError):
            policy.commit(execution)
        policy.abort("test_abort")
        with self.assertRaises(RuntimeError):
            policy.infer(make_record(1))
        policy.reset(make_context())
        self.assertEqual(runtime.reset_count, 2)

    def test_no_touch_has_no_public_or_direct_constructor_bypass(self) -> None:
        self.assertFalse(hasattr(policy_exports, "NoTouchACTPolicy"))
        self.assertFalse(hasattr(act_module, "NoTouchACTPolicy"))
        with self.assertRaisesRegex(ArtifactUnavailableError, "artifact_unavailable"):
            load_matched_no_touch_policy(None)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "paired.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactUnavailableError, "artifact_unavailable"
            ):
                load_matched_no_touch_policy(manifest)

    def test_live_loader_is_fixed_and_closes_artifact_identity(self) -> None:
        checkpoint = canonical_hash("act-checkpoint")
        config = canonical_hash("act-config")
        identity = make_identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trusted"
            source = root / "deployment/ACTStrict/deploy_policy.py"
            source.parent.mkdir(parents=True)
            (source.parents[1] / "__init__.py").write_text("", encoding="utf-8")
            (source.parent / "__init__.py").write_text("", encoding="utf-8")
            source.write_text(
                """from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CONSTRUCTED = []


class StrictACTRuntime:
    def __init__(self, task: str, device: str) -> None:
        CONSTRUCTED.append((task, device))
        evidence = {
            "hashes": {
                "policy_best_sha256": os.environ["ACT_EXPECTED_CHECKPOINT_SHA256"],
                "run_manifest_sha256": os.environ[
                    "ACT_EXPECTED_RUN_MANIFEST_SHA256"
                ],
            },
            "status": "strict_policy_loaded",
            "task": task,
        }
        evidence_dir = Path(os.environ["ACT_RUNTIME_EVIDENCE_DIR"])
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = evidence_dir / f"{task}_strict_policy_load.json"
        path.write_text(json.dumps(evidence) + "\\n", encoding="utf-8")

    def reset(self) -> None:
        return None

    def get_action(self, observation: object) -> np.ndarray:
        return np.zeros((1, 8), dtype=np.float32)
""",
                encoding="utf-8",
            )
            active_environment = {
                "ACT_RUNTIME_EVIDENCE_DIR": str(root / "evidence"),
                "ACT_EXPECTED_CHECKPOINT_SHA256": checkpoint,
                "ACT_EXPECTED_RUN_MANIFEST_SHA256": config,
                "ACT_RUNTIME_SOURCE_ROOT": str(root),
                "ACT_EXPECTED_RUNTIME_FILE": str(source),
                "ACT_EXPECTED_RUNTIME_SHA256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
            }
            module_names = (
                "deployment",
                "deployment.ACTStrict",
                "deployment.ACTStrict.deploy_policy",
            )
            saved_modules = {
                name: sys.modules[name] for name in module_names if name in sys.modules
            }
            for name in module_names:
                sys.modules.pop(name, None)
            try:
                with (
                    patch.object(sys, "path", [str(root), *sys.path]),
                    patch.dict("os.environ", active_environment, clear=False),
                ):
                    importlib.invalidate_caches()
                    loaded = load_strict_act_policy(
                        identity,
                        task="insert_HDMI",
                        device_name="cuda:0",
                    )
                    self.assertIsInstance(loaded, StrictACTPolicy)
                    runtime_module = sys.modules[module_names[-1]]
                    self.assertEqual(
                        runtime_module.CONSTRUCTED,  # type: ignore[attr-defined]
                        [("insert_HDMI", "cuda:0")],
                    )
                    with self.assertRaises(TypeError):
                        load_strict_act_policy(
                            identity,
                            task="insert_HDMI",
                            device_name="cuda:0",
                            loader=object(),  # type: ignore[call-arg]
                        )
                    mismatch = replace(identity, checkpoint_sha256="0" * 64)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        load_strict_act_policy(
                            mismatch,
                            task="insert_HDMI",
                            device_name="cuda:0",
                        )
            finally:
                for name in module_names:
                    sys.modules.pop(name, None)
                sys.modules.update(saved_modules)
                importlib.invalidate_caches()

    def test_live_loader_rejects_shadow_despite_self_signed_evidence(self) -> None:
        checkpoint = canonical_hash("act-checkpoint")
        config = canonical_hash("act-config")
        constructed: list[tuple[str, str]] = []
        active_environment: dict[str, str] = {}

        class ShadowRuntime(RecordingRuntime):
            def __init__(self, task: str, device: str) -> None:
                super().__init__()
                constructed.append((task, device))
                evidence_dir = Path(active_environment["ACT_RUNTIME_EVIDENCE_DIR"])
                evidence = (
                    '{"hashes":{"policy_best_sha256":"'
                    + checkpoint
                    + '","run_manifest_sha256":"'
                    + config
                    + '"},"status":"strict_policy_loaded",'
                    '"task":"insert_HDMI"}\n'
                )
                (evidence_dir / f"{task}_strict_policy_load.json").write_text(
                    evidence, encoding="utf-8"
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trusted"
            expected_file = root / "deployment/ACTStrict/deploy_policy.py"
            expected_file.parent.mkdir(parents=True)
            expected_file.write_text("trusted runtime source\n", encoding="utf-8")
            shadow_file = Path(directory) / "shadow/deploy_policy.py"
            shadow_file.parent.mkdir(parents=True)
            shadow_file.write_text("shadow runtime source\n", encoding="utf-8")
            source_sha256 = hashlib.sha256(expected_file.read_bytes()).hexdigest()
            active_environment.update(
                {
                    "ACT_RUNTIME_EVIDENCE_DIR": directory,
                    "ACT_EXPECTED_CHECKPOINT_SHA256": checkpoint,
                    "ACT_EXPECTED_RUN_MANIFEST_SHA256": config,
                    "ACT_RUNTIME_SOURCE_ROOT": str(root),
                    "ACT_EXPECTED_RUNTIME_FILE": str(expected_file),
                    "ACT_EXPECTED_RUNTIME_SHA256": source_sha256,
                }
            )
            shadow_module = ModuleType("deployment.ACTStrict.deploy_policy")
            shadow_module.__file__ = str(shadow_file)
            shadow_module.__spec__ = ModuleSpec(
                shadow_module.__name__, loader=None, origin=str(shadow_file)
            )
            shadow_module.StrictACTRuntime = ShadowRuntime  # type: ignore[attr-defined]
            with (
                patch.dict(sys.modules, {shadow_module.__name__: shadow_module}),
                patch.dict("os.environ", active_environment, clear=False),
                self.assertRaisesRegex(ValueError, "origin|source|runtime"),
            ):
                load_strict_act_policy(
                    make_identity(), task="insert_HDMI", device_name="cuda:0"
                )
        self.assertEqual(constructed, [])

    def test_structural_absence_preflight_has_zero_backend_and_model_effects(
        self,
    ) -> None:
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
                restoration_index=None,
                restoration_mode=None,
            )
            backend = DeterministicFakeBackend(make_synthetic_episode())
            runtime = RecordingRuntime()
            policy = StrictACTPolicy(identity, runtime, artifact_task="insert_HDMI")
            result = run_closed_loop_trial(
                trial,
                ClosedLoopRunSpec(
                    prompt="insert HDMI",
                    success_predicate_id="fake-success-v1",
                    max_control_cycles=2,
                    max_observation_steps=8,
                    execute_action_steps=1,
                    wall_timeout_s=5.0,
                ),
                backend,
                policy,
                fault_manifest=fault,
            )
            with self.subTest(operator_id=operator_id):
                self.assertEqual(
                    result.terminal_status, TerminalStatus.UNSUPPORTED_CONTRACT
                )
                self.assertEqual(backend.reset_count, 0)
                self.assertEqual(runtime.reset_count, 0)
                self.assertEqual(runtime.infer_count, 0)


if __name__ == "__main__":
    unittest.main()

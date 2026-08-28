from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC, QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    _build_action_encoder,
    _close_runtime_component,
    _install_constrained_placement_compatibility,
    _install_grasp_approach_compatibility,
    _install_task_seed_hook,
    _n0_training_cadence_diagnostic_requested,
    _prepare_process_determinism,
    _resolve_n0_action_execution_contract,
    _resolved_antialiasing_mode,
    _seed_torch_process,
    _validated_antialiasing_mode,
    _validated_initial_seed,
    _verify_checkout,
    launch_univtac_runtime,
)
from robotactile_benchmark.backends.univtac_lifecycle import (
    HANG_DETECTOR_SETTING_PATH,
    require_hang_detector_disabled,
)


class UniVTACFactoryTests(unittest.TestCase):
    def test_n0_training_cadence_requires_explicit_boolean_opt_in(self) -> None:
        variable = "ROBOTACTILE_N0_TRAINING_CADENCE_DIAGNOSTIC"
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_n0_training_cadence_diagnostic_requested())
        with patch.dict(os.environ, {variable: "1"}, clear=True):
            self.assertTrue(_n0_training_cadence_diagnostic_requested())
        with (
            patch.dict(os.environ, {variable: "true"}, clear=True),
            self.assertRaisesRegex(UniVTACContractError, "must be either 0 or 1"),
        ):
            _n0_training_cadence_diagnostic_requested()

    def test_explicit_stock_execution_rejects_diagnostic_env_conflict(self) -> None:
        variable = "ROBOTACTILE_N0_TRAINING_CADENCE_DIAGNOSTIC"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _resolve_n0_action_execution_contract(None),
                (N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT, False),
            )
            self.assertEqual(
                _resolve_n0_action_execution_contract("univtac_stock_ee_v1"),
                ("univtac_stock_ee_v1", False),
            )
            self.assertEqual(
                _resolve_n0_action_execution_contract("robotactile_fixed_endpoint_v1"),
                ("robotactile_fixed_endpoint_v1", True),
            )
        with (
            patch.dict(os.environ, {variable: "1"}, clear=True),
            self.assertRaisesRegex(UniVTACContractError, "conflicts"),
        ):
            _resolve_n0_action_execution_contract("univtac_stock_ee_v1")

    def test_insert_tasks_enable_approach_metric_without_changing_other_calls(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        def grasp_actor(actor: object, **kwargs: object) -> object:
            calls.append({"actor": actor, **kwargs})
            return "actions"

        task = types.SimpleNamespace(
            atom=types.SimpleNamespace(grasp_actor=grasp_actor)
        )
        self.assertTrue(_install_grasp_approach_compatibility(task, "insert_hole"))
        self.assertEqual(
            task.atom.grasp_actor("prism", pre_dis=0.0, dis=0.0, contact_point_id=7),
            "actions",
        )
        self.assertEqual(calls[-1]["pre_dis"], 0.05)
        self.assertEqual(calls[-1]["dis"], 0.0)
        self.assertEqual(calls[-1]["contact_point_id"], 7)

        task.atom.grasp_actor("prism", pre_dis=0.02, dis=0.01)
        self.assertEqual(calls[-1]["pre_dis"], 0.02)
        untouched = types.SimpleNamespace(atom=types.SimpleNamespace())
        self.assertFalse(
            _install_grasp_approach_compatibility(untouched, "pull_out_key")
        )

    def test_insert_hole_removes_only_unstable_constrained_placement_offset(
        self,
    ) -> None:
        class FakePose:
            def __init__(self, values: list[float]) -> None:
                self.values = np.asarray(values, dtype=np.float64)

            @property
            def p(self) -> np.ndarray:
                return self.values[:3]

            @property
            def q(self) -> np.ndarray:
                return self.values[3:]

            def tolist(self) -> list[float]:
                return self.values.tolist()

            def add_bias(self, delta: object, coord: str) -> FakePose:
                self.assert_world(coord)
                values = self.values.copy()
                values[:3] += np.asarray(delta, dtype=np.float64)
                return FakePose(values.tolist())

            def rebase(self, target: FakePose) -> FakePose:
                values = self.values.copy()
                values[:3] -= target.values[:3]
                return FakePose(values.tolist())

            @staticmethod
            def assert_world(coord: str) -> None:
                if coord != "world":
                    raise AssertionError(coord)

        target_pose = FakePose([0.6, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        approach_pose = FakePose([0.6, 0.0, 0.59, 1.0, 0.0, 0.0, 0.0])
        stale_final_pose = FakePose([0.8, 0.0, 0.59, 1.0, 0.0, 0.0, 0.0])
        zero_distance_pose = FakePose([0.6, 0.0, 0.58, 1.0, 0.0, 0.0, 0.0])
        calls: list[dict[str, object]] = []

        def place_actor(*args: object, **kwargs: object) -> list[object]:
            calls.append({"args": args, **kwargs})
            offset = float(kwargs.get("pre_dis", 0.1)) - float(kwargs.get("dis", 0.02))
            return [
                types.SimpleNamespace(
                    action="move",
                    target_pose=(
                        approach_pose if kwargs.get("dis") == 0.01 else stale_final_pose
                    ),
                    args={"pre_dis": offset},
                )
            ]

        def get_place_pose(
            actor: object,
            requested_target: object,
            *,
            functional_point_id: object,
            pre_dis: float,
        ) -> object:
            self.assertEqual(actor, "prism")
            self.assertIs(requested_target, target_pose)
            self.assertIsNone(functional_point_id)
            self.assertEqual(pre_dis, 0.0)
            return zero_distance_pose

        messages: list[str] = []
        prism_pose = FakePose([0.35, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
        gripper_pose = FakePose([0.35, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0])
        task = types.SimpleNamespace(
            atom=types.SimpleNamespace(
                get_place_pose=get_place_pose,
                place_actor=place_actor,
            ),
            logger=types.SimpleNamespace(info=messages.append),
            move=lambda actions, *_args, **_kwargs: actions,
            plan_success=True,
            origin_inhand_pose=prism_pose.rebase(gripper_pose),
            prism=types.SimpleNamespace(get_pose=lambda: prism_pose),
            _robot_manager=types.SimpleNamespace(
                get_gripper_center_pose=lambda: gripper_pose
            ),
        )
        self.assertTrue(
            _install_constrained_placement_compatibility(task, "insert_hole")
        )
        approach_actions = task.atom.place_actor(
            "prism", target_pose=target_pose, pre_dis=0.05, dis=0.01
        )
        self.assertIs(approach_actions[0].target_pose, approach_pose)
        actions = task.atom.place_actor(
            "prism", target_pose=target_pose, pre_dis=0.01, dis=0.002
        )
        np.testing.assert_allclose(
            actions[0].target_pose.tolist(),
            [0.6, 0.0, 0.582, 1.0, 0.0, 0.0, 0.0],
        )
        self.assertIsNone(actions[0].args["pre_dis"])
        self.assertIsNone(actions[0].args["constraint_pose"])
        self.assertTrue(actions[0].args["robotactile_partial_ik_fallback"])
        self.assertEqual(calls[-1]["pre_dis"], 0.01)
        self.assertEqual(calls[-1]["dis"], 0.002)
        self.assertIn('"continuation_m":0.008', messages[0])
        untouched = types.SimpleNamespace(atom=types.SimpleNamespace())
        self.assertFalse(
            _install_constrained_placement_compatibility(untouched, "pull_out_key")
        )

    def test_insert_tube_uses_48_mm_continuation_without_local_ik(self) -> None:
        class FakePose:
            def __init__(self, values: list[float]) -> None:
                self.values = np.asarray(values, dtype=np.float64)

            @property
            def p(self) -> np.ndarray:
                return self.values[:3]

            @property
            def q(self) -> np.ndarray:
                return self.values[3:]

            def tolist(self) -> list[float]:
                return self.values.tolist()

            def add_bias(self, delta: object, coord: str) -> FakePose:
                self.assert_world(coord)
                values = self.values.copy()
                values[:3] += np.asarray(delta, dtype=np.float64)
                return FakePose(values.tolist())

            def rebase(self, target: FakePose) -> FakePose:
                values = self.values.copy()
                values[:3] -= target.values[:3]
                return FakePose(values.tolist())

            @staticmethod
            def assert_world(coord: str) -> None:
                if coord != "world":
                    raise AssertionError(coord)

        target_pose = FakePose([0.35, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
        approach_pose = FakePose([0.35, 0.0, 0.63, 1.0, 0.0, 0.0, 0.0])
        stale_final_pose = FakePose([0.8, 0.0, 0.63, 1.0, 0.0, 0.0, 0.0])
        zero_distance_pose = FakePose([0.35, 0.0, 0.58, 1.0, 0.0, 0.0, 0.0])

        def place_actor(*_args: object, **kwargs: object) -> list[object]:
            offset = float(kwargs.get("pre_dis", 0.1)) - float(kwargs.get("dis", 0.02))
            return [
                types.SimpleNamespace(
                    action="move",
                    target_pose=(
                        approach_pose if kwargs.get("dis") == 0.05 else stale_final_pose
                    ),
                    args={"pre_dis": offset},
                )
            ]

        def get_place_pose(
            _actor: object,
            _requested_target: object,
            *,
            functional_point_id: object,
            pre_dis: float,
        ) -> object:
            self.assertIsNone(functional_point_id)
            self.assertEqual(pre_dis, 0.0)
            return zero_distance_pose

        messages: list[str] = []
        prism_pose = FakePose([0.35, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0])
        gripper_pose = FakePose([0.35, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0])
        task = types.SimpleNamespace(
            atom=types.SimpleNamespace(
                get_place_pose=get_place_pose,
                place_actor=place_actor,
            ),
            logger=types.SimpleNamespace(info=messages.append),
            move=lambda actions, *_args, **_kwargs: actions,
            plan_success=True,
            origin_inhand_pose=prism_pose.rebase(gripper_pose),
            prism=types.SimpleNamespace(get_pose=lambda: prism_pose),
            _robot_manager=types.SimpleNamespace(
                get_gripper_center_pose=lambda: gripper_pose
            ),
        )
        self.assertTrue(
            _install_constrained_placement_compatibility(task, "insert_tube")
        )
        approach_actions = task.atom.place_actor(
            "prism", target_pose=target_pose, pre_dis=0.1, dis=0.05
        )
        task.move(approach_actions)
        actions = task.atom.place_actor(
            "prism", target_pose=target_pose, pre_dis=0.05, dis=0.002
        )
        task.move(actions)
        np.testing.assert_allclose(
            actions[0].target_pose.tolist(),
            [0.35, 0.0, 0.582, 1.0, 0.0, 0.0, 0.0],
        )
        self.assertIsNone(actions[0].args["pre_dis"])
        self.assertIsNone(actions[0].args["constraint_pose"])
        self.assertNotIn("robotactile_partial_ik_fallback", actions[0].args)
        self.assertIn('"continuation_m":0.048', messages[0])
        self.assertEqual(
            [item["phase"] for item in task._robotactile_placement_witnesses],
            ["approach_complete", "final_continuation_complete"],
        )
        self.assertTrue(
            all(
                item["early_stop_predicate"] is False
                for item in task._robotactile_placement_witnesses
            )
        )
        self.assertTrue(
            all(
                item["early_stop_predicate_before"] is False
                for item in task._robotactile_placement_witnesses
            )
        )
        self.assertTrue(
            all(
                item["threshold_crossed"] is False
                for item in task._robotactile_placement_witnesses
            )
        )
        self.assertTrue(
            all(
                item["native_step_before"] is None and item["native_step_after"] is None
                for item in task._robotactile_placement_witnesses
            )
        )

    def test_ee8_action_encoder_matches_pose_and_scalar_gripper_contract(self) -> None:
        source = np.arange(16, dtype=np.float64)[::2]
        original = source.copy()

        def unexpected_as_tensor(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("EE8 actions must not be converted to torch")

        torch = types.SimpleNamespace(
            as_tensor=unexpected_as_tensor,
            float32="float32",
        )
        task = types.SimpleNamespace(device="cuda:7")
        encoded = _build_action_encoder(EE8_ACTION_SPEC, task, torch)(source)

        position = encoded[:3]
        quaternion = encoded[3:7]
        gripper = encoded[7:]
        for values in (position, quaternion):
            self.assertIsInstance(values, np.ndarray)
            self.assertEqual(values.dtype, np.dtype(np.float32))
            self.assertTrue(values.flags.c_contiguous)
            self.assertTrue(values.flags.owndata)
            self.assertFalse(np.shares_memory(values, source))
        self.assertIs(type(gripper), float)
        np.testing.assert_array_equal(position, original[:3].astype(np.float32))
        np.testing.assert_array_equal(quaternion, original[3:7].astype(np.float32))
        self.assertEqual(gripper, float(original[7]))
        position[0] = -1.0
        np.testing.assert_array_equal(encoded[:3], original[:3].astype(np.float32))
        np.testing.assert_array_equal(source, original)

    def test_qpos8_action_encoder_returns_private_device_float32_tensor(self) -> None:
        source = np.arange(16, dtype=np.float64)[::2]
        original = source.copy()
        captured: list[np.ndarray] = []

        def as_tensor(row: np.ndarray, *, dtype: object, device: str) -> object:
            captured.append(row)
            return types.SimpleNamespace(values=row, dtype=dtype, device=device)

        torch = types.SimpleNamespace(as_tensor=as_tensor, float32="float32")
        task = types.SimpleNamespace(device="cuda:7")
        encoded = _build_action_encoder(QPOS8_ACTION_SPEC, task, torch)(source)

        self.assertEqual(encoded.dtype, "float32")
        self.assertEqual(encoded.device, "cuda:7")
        self.assertEqual(len(captured), 1)
        private_row = captured[0]
        self.assertEqual(private_row.dtype, np.dtype(np.float32))
        self.assertTrue(private_row.flags.c_contiguous)
        self.assertTrue(private_row.flags.owndata)
        self.assertFalse(np.shares_memory(private_row, source))
        np.testing.assert_array_equal(private_row, original.astype(np.float32))
        private_row[0] = -1.0
        np.testing.assert_array_equal(source, original)

    def test_action_encoder_rejects_unregistered_action_spec(self) -> None:
        with self.assertRaisesRegex(UniVTACContractError, "unsupported action spec"):
            _build_action_encoder(
                "unregistered",
                types.SimpleNamespace(device="cuda:7"),
                types.SimpleNamespace(),
            )

    def test_clean_system_exit_does_not_terminate_runtime_close(self) -> None:
        events: list[str] = []

        def clean_exit() -> None:
            events.append("close")
            raise SystemExit(0)

        _close_runtime_component(clean_exit, "test component")
        self.assertEqual(events, ["close"])

        def failed_exit() -> None:
            raise SystemExit(2)

        with self.assertRaisesRegex(UniVTACContractError, "non-zero SystemExit"):
            _close_runtime_component(failed_exit, "test component")

    def test_initial_seed_validation_is_strict(self) -> None:
        self.assertEqual(_validated_initial_seed(0), 0)
        for value in (True, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(UniVTACContractError):
                _validated_initial_seed(value)  # type: ignore[arg-type]

    def test_antialiasing_mode_validation_is_strict(self) -> None:
        self.assertIsNone(_validated_antialiasing_mode(None))
        self.assertEqual(_validated_antialiasing_mode("TAA"), "TAA")
        for value in ("taa", "invalid", True):
            with self.subTest(value=value), self.assertRaises(UniVTACContractError):
                _validated_antialiasing_mode(value)  # type: ignore[arg-type]

    def test_n0_defaults_to_empirically_matched_taa_renderer(self) -> None:
        n0_config = build_univtac_backend_config(
            "lift_bottle", action_spec=EE8_ACTION_SPEC
        )
        act_config = build_univtac_backend_config(
            "pull_out_key", action_spec=QPOS8_ACTION_SPEC
        )

        self.assertEqual(_resolved_antialiasing_mode(n0_config, None), "TAA")
        self.assertEqual(_resolved_antialiasing_mode(n0_config, "DLSS"), "DLSS")
        self.assertIsNone(_resolved_antialiasing_mode(act_config, None))

    def test_process_and_torch_determinism_are_seeded(self) -> None:
        events: list[object] = []
        cudnn = types.SimpleNamespace(benchmark=True, deterministic=False)
        torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(cudnn=cudnn),
            cuda=types.SimpleNamespace(
                manual_seed_all=lambda seed: events.append(("cuda", seed))
            ),
            manual_seed=lambda seed: events.append(("torch", seed)),
            use_deterministic_algorithms=lambda enabled, warn_only: events.append(
                ("algorithms", enabled, warn_only)
            ),
        )
        with patch.dict(os.environ, {}, clear=True):
            _prepare_process_determinism(17)
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(os.environ["PYTHONHASHSEED"], "17")
        _seed_torch_process(torch, 17)
        self.assertEqual(
            events,
            [("torch", 17), ("cuda", 17), ("algorithms", True, True)],
        )
        self.assertFalse(cudnn.benchmark)
        self.assertTrue(cudnn.deterministic)

    def test_task_seed_hook_reasserts_determinism_after_upstream_seed(self) -> None:
        events: list[object] = []
        cudnn = types.SimpleNamespace(benchmark=True, deterministic=False)
        torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(cudnn=cudnn),
            cuda=types.SimpleNamespace(manual_seed_all=lambda seed: None),
            manual_seed=lambda seed: None,
            use_deterministic_algorithms=lambda enabled, warn_only: None,
        )
        task = types.SimpleNamespace(cfg=types.SimpleNamespace(seed=None))

        def upstream_seed(seed: int) -> None:
            events.append(("upstream", seed))
            task.cfg.seed = seed

        task.seed = upstream_seed
        _install_task_seed_hook(task, torch, 17)
        self.assertEqual(task.seed(17), 17)
        self.assertEqual(events, [("upstream", 17)])
        self.assertFalse(cudnn.benchmark)
        self.assertTrue(cudnn.deterministic)
        with self.assertRaises(UniVTACContractError):
            task.seed(18)

    def test_checkout_verification_rejects_untracked_module_shadowing(self) -> None:
        config = build_univtac_backend_config("pull_out_key")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_git_output(_root: Path, arguments: tuple[str, ...]) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return config.upstream_commit
                if "--untracked-files=no" in arguments:
                    return ""
                return "?? envs/shadow.py"

            with (
                patch(
                    "robotactile_benchmark.backends.univtac_factory._git_output",
                    side_effect=fake_git_output,
                ),
                self.assertRaisesRegex(UniVTACContractError, "dirty"),
            ):
                _verify_checkout(root, config)

    def test_checkout_rejects_a_self_signed_tampered_config(self) -> None:
        config = build_univtac_backend_config("pull_out_key")
        tampered = replace(
            config,
            action_upper_bounds=(100.0,) + config.action_upper_bounds[1:],
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(UniVTACContractError, "packaged registry"),
        ):
            _verify_checkout(Path(directory), tampered)

    def test_checkout_rejects_task_source_hash_drift(self) -> None:
        config = build_univtac_backend_config("pull_out_key")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "envs" / "pull_out_key.py"
            source.parent.mkdir(parents=True)
            source.write_text("# modified task source\n", encoding="utf-8")

            def fake_git_output(_root: Path, arguments: tuple[str, ...]) -> str:
                if arguments == ("rev-parse", "HEAD"):
                    return config.upstream_commit
                return ""

            with (
                patch(
                    "robotactile_benchmark.backends.univtac_factory._git_output",
                    side_effect=fake_git_output,
                ),
                self.assertRaisesRegex(UniVTACContractError, "source hash"),
            ):
                _verify_checkout(root, config)

    def test_importing_factory_does_not_import_simulator_or_torch(self) -> None:
        script = """
import json
import sys
from robotactile_benchmark.backends import univtac_factory
blocked = sorted(
    name for name in sys.modules
    if name == 'torch' or name.startswith(('torch.', 'omni.', 'isaaclab', 'envs.'))
)
print(json.dumps(blocked))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(completed.stdout), [])

    def test_task_and_torch_imports_happen_only_after_app_launcher_starts(self) -> None:
        config = build_univtac_backend_config("pull_out_key")
        events: list[str] = []
        lifecycle_stages: list[str] = []
        state: dict[str, object] = {}
        teardown_patcher = patch(
            "robotactile_benchmark.backends.univtac_factory."
            "prepare_univtac_task_teardown"
        )
        teardown_patcher.start()
        self.addCleanup(teardown_patcher.stop)

        class FakeApp:
            def close(self) -> None:
                events.append("app:close")

        class FakeAppLauncher:
            def __init__(self, args: argparse.Namespace) -> None:
                self.app = FakeApp()
                events.append(f"launcher:start:{args.headless}")

        class FakeExtensionManager:
            def __init__(self) -> None:
                self.enabled: set[str] = set()

            def set_extension_enabled_immediate(
                self, extension_id: str, enabled: bool
            ) -> None:
                self.assert_extension_id(extension_id)
                self.assert_enabled(enabled)
                self.enabled.add(extension_id)
                events.append(f"extension:enable:{extension_id}")

            def is_extension_enabled(self, extension_id: str) -> bool:
                return extension_id in self.enabled

            @staticmethod
            def assert_extension_id(extension_id: str) -> None:
                if extension_id != "omni.ui":
                    raise AssertionError(extension_id)

            @staticmethod
            def assert_enabled(enabled: bool) -> None:
                if enabled is not True:
                    raise AssertionError(enabled)

        extension_manager = FakeExtensionManager()
        fake_kit_app = types.SimpleNamespace(
            get_app=lambda: types.SimpleNamespace(
                get_extension_manager=lambda: extension_manager
            )
        )

        class FakeTaskCfg:
            def __init__(self) -> None:
                self.step_lim = config.task.action_horizon
                self.decimation = 99
                self.obs_data_type = {}
                self.save_frequency = 1
                self.video_frequency = 1
                self.render_frequency = 1
                self.random_texture = True
                self.tactile_sensor_type = "wrong"
                self.save_dir = "wrong"
                self.scene = types.SimpleNamespace(num_envs=99)
                self.sim = types.SimpleNamespace(
                    dt=1.0,
                    device="cpu",
                    render=types.SimpleNamespace(antialiasing_mode=None),
                )

        class FakeTask:
            def __init__(self, cfg: FakeTaskCfg, mode: str) -> None:
                events.append(f"task:init:{mode}")
                self.cfg = cfg
                self.device = "cuda:7"
                self._robot_manager = types.SimpleNamespace(
                    robot=types.SimpleNamespace(
                        joint_names=list(reversed(config.canonical_joint_names))
                    )
                )
                self.close_count = 0
                state["task"] = self

            def seed(self, seed: int) -> int:
                events.append(f"upstream:seed:{seed}")
                return seed

            def close(self) -> None:
                self.close_count += 1
                events.append("task:close")

        fake_isaac = types.SimpleNamespace(AppLauncher=FakeAppLauncher)
        fake_carb = types.SimpleNamespace(
            settings=types.SimpleNamespace(
                get_settings=lambda: types.SimpleNamespace(
                    get=lambda path: events.append(f"setting:get:{path}") or False
                )
            )
        )
        fake_task_module = types.SimpleNamespace(TaskCfg=FakeTaskCfg, Task=FakeTask)
        fake_cudnn = types.SimpleNamespace(benchmark=True, deterministic=False)
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(cudnn=fake_cudnn),
            cuda=types.SimpleNamespace(
                manual_seed_all=lambda seed: events.append(f"cuda:seed:{seed}")
            ),
            float32="float32",
            manual_seed=lambda seed: events.append(f"torch:seed:{seed}"),
            as_tensor=lambda row, dtype, device: (
                np.asarray(row, dtype=np.float32),
                dtype,
                device,
            ),
            use_deterministic_algorithms=lambda enabled, warn_only: events.append(
                f"torch:deterministic:{enabled}:{warn_only}"
            ),
        )

        def importer(name: str):
            events.append(f"import:{name}")
            if name == "isaaclab.app":
                return fake_isaac
            self.assertIn("launcher:start:True", events)
            if name == "carb":
                return fake_carb
            if name == "omni.kit.app":
                return fake_kit_app
            if name == config.task.module_name:
                return fake_task_module
            if name == "torch":
                return fake_torch
            raise AssertionError(name)

        with tempfile.TemporaryDirectory() as directory:
            fake_task_module.__file__ = str(
                Path(directory) / "envs" / "pull_out_key.py"
            )
            with (
                patch(
                    "robotactile_benchmark.backends.univtac_factory._verify_checkout"
                ),
                patch(
                    "robotactile_benchmark.backends.univtac_factory."
                    "require_hang_detector_disabled",
                    side_effect=lambda: require_hang_detector_disabled(importer),
                ),
                patch(
                    "robotactile_benchmark.backends.univtac_factory."
                    "install_runtime_signal_tracing",
                    side_effect=lambda: events.append("signals:installed"),
                ),
                patch(
                    "robotactile_benchmark.backends.univtac_factory."
                    "install_gsmini_attachment_constructor_compatibility",
                    return_value=True,
                ),
                patch(
                    "robotactile_benchmark.backends.univtac_factory."
                    "repair_gsmini_tactile_attachments",
                    return_value=True,
                ),
                patch(
                    "robotactile_benchmark.backends.univtac_factory.importlib.import_module",
                    side_effect=importer,
                ),
            ):
                runtime = launch_univtac_runtime(
                    config,
                    upstream_root=Path(directory),
                    runtime_dir=Path(directory) / "run",
                    initial_seed=17,
                    launcher_args={"headless": True},
                    antialiasing_mode="TAA",
                    stage_observer=lifecycle_stages.append,
                )

        launcher_index = events.index("launcher:start:True")
        setting_index = events.index(f"setting:get:{HANG_DETECTOR_SETTING_PATH}")
        signal_index = events.index("signals:installed")
        extension_index = events.index("extension:enable:omni.ui")
        self.assertLess(launcher_index, setting_index)
        self.assertLess(setting_index, signal_index)
        self.assertLess(signal_index, extension_index)
        self.assertGreater(extension_index, launcher_index)
        self.assertGreater(
            events.index(f"import:{config.task.module_name}"), launcher_index
        )
        self.assertLess(
            extension_index, events.index(f"import:{config.task.module_name}")
        )
        self.assertLess(events.index("import:torch"), events.index("task:init:eval"))
        self.assertIn("torch:seed:17", events)
        self.assertIn("cuda:seed:17", events)
        self.assertIn("torch:deterministic:True:True", events)
        self.assertFalse(fake_cudnn.benchmark)
        self.assertTrue(fake_cudnn.deterministic)
        self.assertGreater(events.index("import:torch"), launcher_index)
        self.assertEqual(
            lifecycle_stages,
            [
                "app_launcher_import",
                "app_launcher",
                "runtime_preparation",
                "tactile_constructor_hook",
                "univtac_task_construction",
                "tactile_attachment_validation",
                "runtime_compatibility_installation",
                "runtime_ready",
            ],
        )
        task = state["task"]
        self.assertEqual(
            runtime.handshake.live_joint_names,
            tuple(reversed(config.canonical_joint_names)),
        )
        self.assertEqual(task.cfg.decimation, 1)
        self.assertEqual(task.cfg.sim.dt, 1.0 / 120.0)
        self.assertEqual(task.cfg.sim.render.antialiasing_mode, "TAA")
        self.assertEqual(task.cfg.scene.num_envs, 1)
        self.assertEqual(task.cfg.seed, 17)
        self.assertEqual(runtime.construction_seed, 17)
        self.assertEqual(
            task.cfg.obs_data_type,
            {
                "camera": ["rgb"],
                "tactile": ["rgb_marker", "depth"],
                "embodiment": ["joint"],
            },
        )
        encoded, dtype, device = runtime.encode_action(np.zeros(8, dtype=np.float32))
        self.assertEqual((dtype, device), ("float32", "cuda:7"))
        np.testing.assert_array_equal(encoded, np.zeros(8, dtype=np.float32))
        runtime.prepare_reset()
        self.assertEqual(events.count("torch:seed:17"), 2)
        self.assertEqual(events.count("cuda:seed:17"), 2)
        self.assertEqual(task.seed(17), 17)
        self.assertEqual(events.count("torch:seed:17"), 3)
        self.assertEqual(events.count("cuda:seed:17"), 3)
        runtime.close_runtime()
        self.assertEqual(task.close_count, 1)
        self.assertEqual(events[-2:], ["task:close", "app:close"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
    build_univtac_backend_config,
)
from robotactile_benchmark.backends.univtac_factory import (
    _verify_checkout,
    launch_univtac_runtime,
)


class UniVTACFactoryTests(unittest.TestCase):
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
        state: dict[str, object] = {}

        class FakeApp:
            def close(self) -> None:
                events.append("app:close")

        class FakeAppLauncher:
            def __init__(self, args: argparse.Namespace) -> None:
                self.app = FakeApp()
                events.append(f"launcher:start:{args.headless}")

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
                self.sim = types.SimpleNamespace(dt=1.0, device="cpu")

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

            def close(self) -> None:
                self.close_count += 1
                events.append("task:close")

        fake_isaac = types.SimpleNamespace(AppLauncher=FakeAppLauncher)
        fake_task_module = types.SimpleNamespace(TaskCfg=FakeTaskCfg, Task=FakeTask)
        fake_torch = types.SimpleNamespace(
            float32="float32",
            as_tensor=lambda row, dtype, device: (
                np.asarray(row, dtype=np.float32),
                dtype,
                device,
            ),
        )

        def importer(name: str):
            events.append(f"import:{name}")
            if name == "isaaclab.app":
                return fake_isaac
            self.assertIn("launcher:start:True", events)
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
                    "robotactile_benchmark.backends.univtac_factory.importlib.import_module",
                    side_effect=importer,
                ),
            ):
                runtime = launch_univtac_runtime(
                    config,
                    upstream_root=Path(directory),
                    runtime_dir=Path(directory) / "run",
                    launcher_args={"headless": True},
                )

        launcher_index = events.index("launcher:start:True")
        self.assertGreater(
            events.index(f"import:{config.task.module_name}"), launcher_index
        )
        self.assertGreater(events.index("import:torch"), launcher_index)
        task = state["task"]
        self.assertEqual(
            runtime.handshake.live_joint_names,
            tuple(reversed(config.canonical_joint_names)),
        )
        self.assertEqual(task.cfg.decimation, 1)
        self.assertEqual(task.cfg.sim.dt, 1.0 / 120.0)
        self.assertEqual(task.cfg.scene.num_envs, 1)
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
        runtime.close_runtime()
        self.assertEqual(task.close_count, 1)
        self.assertEqual(events[-2:], ["task:close", "app:close"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.backends.univtac_contracts import (
    ACTION_MODE,
    DECIMATION,
    EARLY_STOP_NONE_IS_FALSE_TASK_IDS,
    PHYSICS_STEPS_PER_ACTION,
    SIM_HZ,
    UPSTREAM_COMMIT,
    UniVTACContractError,
    build_univtac_backend_config,
    load_univtac_task_registry,
)

ROOT = Path(__file__).resolve().parents[1]


class UniVTACBackendContractTests(unittest.TestCase):
    def test_core_imports_do_not_load_simulator_or_torch_modules(self) -> None:
        script = """
import json
import sys
import robotactile_benchmark
from robotactile_benchmark.backends import univtac_contracts
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

    def test_registry_binds_all_eight_upstream_tasks_and_exact_horizons(self) -> None:
        registry = load_univtac_task_registry()
        expected_horizons = {
            "grasp_classify": 300,
            "insert_HDMI": 600,
            "insert_hole": 300,
            "insert_tube": 300,
            "lift_bottle": 500,
            "lift_can": 300,
            "pull_out_key": 300,
            "put_bottle_in_shelf": 300,
        }

        self.assertEqual(registry.upstream_commit, UPSTREAM_COMMIT)
        self.assertEqual(
            {task.task_id: task.action_horizon for task in registry.tasks},
            expected_horizons,
        )
        for task in registry.tasks:
            with self.subTest(task=task.task_id):
                self.assertEqual(task.module_name, f"envs.{task.task_id}")
                self.assertEqual(task.class_name, "Task")
                self.assertTrue(task.prompt.strip())
                self.assertTrue(task.prompt_id.endswith(".v1"))
                self.assertEqual(len(task.task_source_sha256), 64)
                self.assertEqual(
                    task.success_predicate_id,
                    f"univtac-upstream-05bcd3ed.{task.task_id}.check_success.v1",
                )

    def test_backend_config_freezes_runtime_and_live_joint_contract(self) -> None:
        config = build_univtac_backend_config("pull_out_key")

        self.assertEqual(config.action_mode, ACTION_MODE)
        self.assertTrue(config.force)
        self.assertEqual(config.sim_hz, SIM_HZ)
        self.assertEqual(config.decimation, DECIMATION)
        self.assertEqual(
            config.physics_steps_per_action,
            PHYSICS_STEPS_PER_ACTION,
        )
        self.assertEqual(config.action_lower_bounds[-1], 0.0)
        self.assertEqual(config.action_upper_bounds[-1], 0.039)
        self.assertEqual(
            config.canonical_joint_names,
            tuple(f"panda_joint{index}" for index in range(1, 8))
            + ("panda_finger_joint1", "panda_finger_joint2"),
        )

    def test_registry_and_handshake_tampering_fail_closed(self) -> None:
        registry = load_univtac_task_registry()
        config = build_univtac_backend_config("pull_out_key")
        handshake = config.expected_handshake(
            live_joint_names=tuple(reversed(config.canonical_joint_names))
        )

        config.validate_handshake(handshake)
        with self.assertRaisesRegex(UniVTACContractError, "handshake"):
            config.validate_handshake(replace(handshake, upstream_commit="0" * 40))
        with self.assertRaisesRegex(UniVTACContractError, "config"):
            config.validate_handshake(replace(handshake, config_sha256="0" * 64))

        document = json.loads(
            (ROOT / "configs" / "univtac" / "tasks_v1.json").read_text(encoding="utf-8")
        )
        document["tasks"][0]["prompt"] = ""
        with self.assertRaisesRegex(UniVTACContractError, "prompt"):
            load_univtac_task_registry(document=document)

        self.assertEqual(len(registry.resource_sha256), 64)

    def test_unknown_task_and_known_predicate_quirks_are_explicit(self) -> None:
        registry = load_univtac_task_registry()
        by_id = {task.task_id: task for task in registry.tasks}

        with self.assertRaisesRegex(KeyError, "unknown UniVTAC task"):
            build_univtac_backend_config("not_a_task")
        self.assertIn("x-coordinate omission", by_id["insert_HDMI"].predicate_note)
        self.assertIn("signed offsets", by_id["insert_tube"].predicate_note)
        self.assertIn(
            "not benchmark-corrected",
            by_id["insert_HDMI"].predicate_note.lower(),
        )
        self.assertEqual(
            EARLY_STOP_NONE_IS_FALSE_TASK_IDS,
            frozenset({"insert_hole", "insert_tube"}),
        )

    def test_registry_nested_configuration_is_immutable(self) -> None:
        registry = load_univtac_task_registry()

        for mapping, key in (
            (registry.runtime, "sim_hz"),
            (registry.aliases, "head_camera"),
            (registry.phase_tracker, "on_threshold_mm"),
        ):
            with self.subTest(key=key), self.assertRaises(TypeError):
                mapping[key] = "tampered"  # type: ignore[index, assignment]


if __name__ == "__main__":
    unittest.main()

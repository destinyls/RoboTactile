from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.backends.qualification import (
    CPU_FAKE_EVIDENCE_LEVEL,
    QUALIFICATION_CHECKS,
    UniVTACQualificationError,
    run_cpu_fake_qualification,
)
from robotactile_benchmark.backends.univtac_contracts import (
    load_univtac_task_registry,
)
from robotactile_benchmark.closed_loop.smoke import (
    smoke_summary,
    write_cpu_smoke_bundle,
)
from robotactile_benchmark.contracts import canonical_hash


class UniVTACQualificationTests(unittest.TestCase):
    def test_task4_canonical_smoke_external_root_pin_tracks_v2_trial_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = smoke_summary(
                write_cpu_smoke_bundle(Path(directory) / "task4-bundle")
            )

        self.assertEqual(
            summary["root_receipt_sha256"],
            "645c6953bc0fc19ca3e2de9180a65b2640d6c9c512ef09da110b421e158f2734",
        )

    def test_cpu_qualification_binds_deterministic_resets_and_action_progress(
        self,
    ) -> None:
        first = run_cpu_fake_qualification("pull_out_key")
        second = run_cpu_fake_qualification("pull_out_key")

        self.assertTrue(first.passed)
        self.assertEqual(first.failure_codes, ())
        self.assertEqual(first.evidence_level, CPU_FAKE_EVIDENCE_LEVEL)
        self.assertEqual(
            first.success_predicate_id,
            "univtac-upstream-05bcd3ed.pull_out_key.check_success.v1",
        )
        self.assertEqual(first.checks, QUALIFICATION_CHECKS)
        self.assertEqual(first, second)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(
            first.first_reset_state_sha256,
            first.second_reset_state_sha256,
        )
        self.assertEqual(
            first.first_clean_record_sha256,
            first.second_clean_record_sha256,
        )
        self.assertEqual(first.first_initial_native_step_id, 417)
        self.assertEqual(first.second_initial_native_step_id, 417)
        self.assertEqual(first.action_native_step_id, 418)
        self.assertEqual(first.action_benchmark_step, 1)
        self.assertNotEqual(
            first.first_reset_state_sha256,
            first.action_state_sha256,
        )
        self.assertEqual(
            dict(first.terminal_priority_results),
            {
                "early_over_timeout": "early_stop",
                "failure_over_early": "task_failure",
                "success_over_failure": "success",
                "timeout_fallback": "timeout",
            },
        )
        self.assertFalse(first.environment["isaac_sim_executed"])
        self.assertEqual(first.environment["accelerator"], "none")
        self.assertEqual(
            first.first_joint_reorder_witness_sha256,
            first.second_joint_reorder_witness_sha256,
        )
        self.assertAlmostEqual(first.initial_canonical_joint9[-2], 0.02)
        self.assertAlmostEqual(first.initial_canonical_joint9[-1], 0.02)
        self.assertEqual(len(first.initial_model_visible_qpos8), 8)
        self.assertEqual(canonical_hash(first.config_descriptor), first.config_sha256)
        self.assertEqual(
            canonical_hash(first.handshake_descriptor), first.handshake_sha256
        )

    def test_cpu_qualification_is_honest_and_available_for_all_eight_tasks(
        self,
    ) -> None:
        for task in load_univtac_task_registry().tasks:
            with self.subTest(task_id=task.task_id):
                receipt = run_cpu_fake_qualification(task.task_id)
                self.assertTrue(receipt.passed)
                self.assertEqual(receipt.task_id, task.task_id)
                self.assertEqual(
                    receipt.terminal_priority_results["early_over_timeout"],
                    "early_stop" if task.early_stop_capable else "not_applicable",
                )

    def test_receipt_is_deeply_immutable_and_rejects_tampered_witnesses(self) -> None:
        receipt = run_cpu_fake_qualification("pull_out_key")

        with self.assertRaises(TypeError):
            receipt.environment["accelerator"] = "cuda"  # type: ignore[index]
        with self.assertRaises(TypeError):
            receipt.config_descriptor["task"]["task_id"] = "tampered"
        with self.assertRaisesRegex(UniVTACQualificationError, "environment"):
            replace(
                receipt,
                environment={**receipt.environment, "accelerator": "cuda"},
            )
        with self.assertRaisesRegex(UniVTACQualificationError, "environment"):
            replace(
                receipt,
                environment={**receipt.environment, "python_version": 313},
            )
        with self.assertRaisesRegex(UniVTACQualificationError, "action witness"):
            replace(receipt, action_progress_witness_sha256="0" * 64)
        with self.assertRaisesRegex(UniVTACQualificationError, "terminal witness"):
            replace(receipt, terminal_priority_witness_sha256="0" * 64)
        tampered_terminal = dict(receipt.terminal_priority_results)
        tampered_terminal["success_over_failure"] = "task_failure"
        with self.assertRaisesRegex(UniVTACQualificationError, "terminal priority"):
            replace(
                receipt,
                terminal_priority_results=tampered_terminal,
                terminal_priority_witness_sha256=canonical_hash(tampered_terminal),
            )
        with self.assertRaisesRegex(UniVTACQualificationError, "native step"):
            replace(
                receipt,
                first_initial_native_step_id=-1,
                action_progress_witness_sha256=canonical_hash(
                    {
                        "first_state_sha256": receipt.first_reset_state_sha256,
                        "action_state_sha256": receipt.action_state_sha256,
                        "first_native_step_id": -1,
                        "action_native_step_id": receipt.action_native_step_id,
                        "action_benchmark_step": receipt.action_benchmark_step,
                        "action_sha256": receipt.action_sha256,
                    }
                ),
            )
        with self.assertRaisesRegex(UniVTACQualificationError, "joint witness"):
            replace(
                receipt,
                initial_canonical_joint9=(0.0,) * 9,
            )
        tampered_config = dict(receipt.config_descriptor)
        tampered_config["task"] = {
            **tampered_config["task"],
            "task_id": "coordinated_tamper",
        }
        with self.assertRaisesRegex(UniVTACQualificationError, "identity"):
            replace(
                receipt,
                config_descriptor=tampered_config,
                config_sha256=canonical_hash(tampered_config),
            )
        with self.assertRaisesRegex(UniVTACQualificationError, "failure codes"):
            replace(receipt, passed=True, failure_codes=("reset_hash_mismatch",))
        with self.assertRaisesRegex(UniVTACQualificationError, "unknown failure"):
            replace(receipt, passed=False, failure_codes=("invented_code",))


if __name__ == "__main__":
    unittest.main()

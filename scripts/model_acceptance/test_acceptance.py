"""CPU contract tests; synthetic fixtures do not constitute model acceptance."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.model_acceptance.probe import action_summary, audit, policy_kind_matches
from scripts.model_acceptance.run import MODELS, load_manifest, main, worker_environment


class AcceptanceTests(unittest.TestCase):
    def test_registered_training_config_proof_rebuilds_ast_and_rejects_bad_contract(
        self,
    ) -> None:
        import hashlib

        from scripts.model_acceptance.report import (
            GateFailure,
            verify_source_attestation,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = "n0_twam/configs/twam_posttrain_cfg.py"
            snapshot = root / relative
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("_VALUES = {'x': 1}\n")
            renderer = root / "renderer.py"
            renderer.write_text('source = f"_VALUES = {literal}\\n"\n')
            contract = root / "contract.py"
            contract.write_text(
                f"OFFICIAL_COMMIT = {'c' * 40!r}\nALLOWED_OFFICIAL_CHANGE = {relative!r}\n"
            )

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            original = {relative: digest(snapshot)}
            artifact = {
                "artifact_sha256": "a" * 64,
                "source_tree_sha256": "b" * 64,
                "source": {
                    "root": str(root),
                    "source_commit": "c" * 40,
                    "source_tree_sha256": "b" * 64,
                    "python_files": original,
                },
            }
            patch = {
                "upstream_sha256": "e" * 64,
                "snapshot_sha256": original[relative],
                "renderer_path": str(renderer),
                "renderer_sha256": digest(renderer),
                "renderer_reconstructed_sha256": original[relative],
                "policy_contract_path": str(contract),
                "policy_contract_sha256": digest(contract),
            }
            proof = {
                "status": "passed",
                "verification_scope": "independent_git_object_comparison_with_registered_training_config",
                "normal_python_file_set_equal": True,
                "source_commit": "c" * 40,
                "snapshot_root": str(root),
                "prepared_artifact_sha256": "a" * 64,
                "prepared_source_tree_sha256": "b" * 64,
                "python_files": {
                    relative: {
                        "git_blob": "d" * 40,
                        "expected_sha256": "e" * 64,
                        "actual_sha256": original[relative],
                    }
                },
                "appledouble_files": {},
                "documented_config_patches": {relative: patch},
            }
            path = root / "proof.json"
            path.write_text(json.dumps(proof))
            result = verify_source_attestation(artifact, path)
            self.assertFalse(result["pure_upstream_python_tree"])
            self.assertTrue(result["registered_patch_ast_reconstruction_verified"])
            contract.write_text(
                f"OFFICIAL_COMMIT = {'c' * 40!r}\nALLOWED_OFFICIAL_CHANGE = 'wrong.py'\n"
            )
            patch["policy_contract_sha256"] = digest(contract)
            path.write_text(json.dumps(proof))
            with self.assertRaises(GateFailure):
                verify_source_attestation(artifact, path)

    def test_independent_source_proof_requires_exact_partition_and_hashes(self) -> None:
        import hashlib

        from scripts.model_acceptance.report import (
            GateFailure,
            verify_source_attestation,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "normal.py").write_bytes(b"pass\n")
            (root / "._normal.py").write_bytes(bytes.fromhex("00051607") + b"metadata")
            members = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("normal.py", "._normal.py")
            }
            artifact = {
                "artifact_sha256": "a" * 64,
                "source_tree_sha256": "b" * 64,
                "source": {
                    "root": str(root),
                    "source_commit": "c" * 40,
                    "source_tree_sha256": "b" * 64,
                    "python_files": members,
                },
            }
            proof = {
                "status": "passed",
                "verification_scope": "independent_git_object_comparison_at_acceptance_time",
                "normal_python_file_set_equal": True,
                "source_commit": "c" * 40,
                "snapshot_root": str(root),
                "prepared_artifact_sha256": "a" * 64,
                "prepared_source_tree_sha256": "b" * 64,
                "python_files": {
                    "normal.py": {
                        "git_blob": "d" * 40,
                        "expected_sha256": members["normal.py"],
                        "actual_sha256": members["normal.py"],
                    }
                },
                "appledouble_files": {
                    "._normal.py": {
                        "sha256": members["._normal.py"],
                        "magic_hex": "00051607",
                    }
                },
            }
            path = root / "attestation.json"
            with self.assertRaises(GateFailure):
                verify_source_attestation(artifact, path)
            path.write_text(json.dumps(proof))
            self.assertEqual(
                verify_source_attestation(artifact, path)["normal_python_files"], 1
            )
            proof["python_files"]["normal.py"]["actual_sha256"] = "e" * 64
            path.write_text(json.dumps(proof))
            with self.assertRaises(GateFailure):
                verify_source_attestation(artifact, path)
            proof["python_files"]["normal.py"]["actual_sha256"] = members["normal.py"]
            proof["appledouble_files"] = {}
            path.write_text(json.dumps(proof))
            with self.assertRaises(GateFailure):
                verify_source_attestation(artifact, path)

    def test_report_revalidates_jsonl_without_postclose_receipt(self) -> None:
        from robotactile_benchmark.closed_loop.contracts import ActionPlan
        from scripts.model_acceptance.report import GateFailure, verify_plans

        plan = ActionPlan("qpos8_next_step", 0, np.ones((20, 8), dtype=np.float32))
        bundle = SimpleNamespace(
            trial=SimpleNamespace(action_spec=plan.action_spec),
            evidence=SimpleNamespace(
                action_entries=[
                    SimpleNamespace(
                        action_plan_sha256=plan.sha256,
                        source_step_index=0,
                        executed_actions=plan.actions[:1],
                    )
                ]
            ),
        )
        record = {
            "action_spec": plan.action_spec,
            "plan_source_step_index": 0,
            "source_step_index": 0,
            "action_plan_sha256": plan.sha256,
            "planned_action_values": plan.actions.tolist(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            path = episode / "inference_trace.jsonl"
            path.write_text(json.dumps(record) + "\n")
            proof = verify_plans(episode, bundle)
            self.assertTrue(proof["prefixes_verified"])
            self.assertEqual(proof["path"], str(path))
            self.assertFalse((episode / "acceptance.json").exists())
            record["planned_action_values"][0][0] = 0
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaises(GateFailure):
                verify_plans(episode, bundle)

    def test_report_distinguishes_smoke_full_horizon_and_success(self) -> None:
        from scripts.model_acceptance.report import completed

        result = SimpleNamespace(
            validation_passed=True, score_eligible=True, score_success=False
        )
        evidence = SimpleNamespace(
            result=result,
            action_entries=[SimpleNamespace(executed_actions=np.zeros((3, 8)))],
        )
        bundle = SimpleNamespace(evidence=evidence)
        self.assertFalse(completed(bundle, 300))
        evidence.action_entries = [SimpleNamespace(executed_actions=np.zeros((300, 8)))]
        self.assertTrue(completed(bundle, 300))
        evidence.action_entries = [SimpleNamespace(executed_actions=np.zeros((35, 8)))]
        result.score_success = True
        self.assertTrue(completed(bundle, 300))
        result.validation_passed = False
        self.assertFalse(completed(bundle, 300))

    def test_report_source_commit_mismatch_is_not_suppressed(self) -> None:
        from scripts.model_acceptance.report import GateFailure, source_gate

        inventory = {
            "binding": {"source_commit": "expected"},
            "source": {
                "expected_commit": "expected",
                "git_head": {"returncode": 0, "stdout": "wrong"},
            },
        }
        with self.assertRaises(GateFailure) as raised:
            source_gate(inventory, "dream_tac", {}, Path("/inventory/dream.json"))
        self.assertEqual(raised.exception.gate, "source")

    def test_episode_wrapper_keeps_full_plan_and_forwards_protocol(self) -> None:
        from robotactile_benchmark.closed_loop.contracts import ActionPlan
        from scripts.model_acceptance.episode import TracedPolicy

        plan = ActionPlan("qpos8_next_step", 0, np.ones((20, 8), dtype=np.float32))
        observation = SimpleNamespace(
            step_index=0,
            proprio=np.zeros(8, dtype=np.float32),
            vision={"top": np.zeros((2, 2, 3), dtype=np.uint8)},
            tactile=[],
        )
        policy = Mock(identity="identity")
        policy.infer.return_value = plan
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            traced = TracedPolicy(policy, path)
            traced.reset("context")
            self.assertIs(traced.infer(observation), plan)
            traced.commit("execution")
            traced.abort("reason")
            traced.close()
            row = json.loads(path.read_text())
            self.assertEqual(row["planned_actions"]["shape"], [20, 8])
            self.assertEqual(row["action_plan_sha256"], plan.sha256)
            self.assertEqual(len(row["planned_action_values"]), 20)
            policy.reset.assert_called_once_with("context")
            policy.commit.assert_called_once_with("execution")
            policy.abort.assert_called_once_with("reason")
            policy.close.assert_called_once_with()

    def test_historical_n0_policy_alias_is_narrow(self) -> None:
        self.assertTrue(policy_kind_matches("n0_twam", "n0"))
        self.assertTrue(policy_kind_matches("n0_twam", "n0_twam"))
        self.assertFalse(policy_kind_matches("n0_vtla", "n0"))
        self.assertFalse(policy_kind_matches("n0_twam", "act"))

    def test_dream_shared_environment_and_act_config_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binding = Path(temporary) / "binding.json"
            binding.write_text(
                json.dumps(
                    {
                        "model": "dream_tac",
                        "deployment_root": "/deploy",
                        "shared_pythonpath": "/dream/site:/ftp/site",
                    }
                )
            )
            row = {
                "model": "dream_tac",
                "binding_path": str(binding),
                "package": "/package",
                "code": "/code",
            }
            env = worker_environment(row)
            self.assertEqual(env["PYTHONPATH"], "/package:/code:/dream/site:/ftp/site")
            self.assertEqual(env["CUDA_HOME"], "/deploy/runtime/cuda-toolkit-12.8")
            self.assertIn("cudnn/lib", env["LD_LIBRARY_PATH"])
            binding.write_text("ACT config is deliberately not a retrained binding")
            row["model"] = "act"
            self.assertEqual(worker_environment(row)["PYTHONPATH"], "/package:/code")

    def test_actions_reject_empty_wrong_dimension_and_nonfinite(self) -> None:
        self.assertTrue(action_summary(np.ones((20, 8), dtype=np.float32))["valid"])
        for value in (np.ones((0, 8)), np.ones((1, 7)), np.full((1, 8), np.nan)):
            self.assertFalse(action_summary(value)["valid"])

    def test_manifest_requires_five_distinct_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            rows = [
                {
                    "model": model,
                    **{
                        key: "/absolute/path"
                        for key in (
                            "binding_path",
                            "code",
                            "package",
                            "live_artifact",
                            "source_root",
                            "runtime_python",
                        )
                    },
                }
                for model in sorted(MODELS)
            ]
            path.write_text(json.dumps({"models": rows}))
            self.assertEqual(len(load_manifest(path)), 5)
            rows[0]["model"] = rows[1]["model"]
            path.write_text(json.dumps({"models": rows}))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_strict_artifact_roundtrip_and_tampering(self) -> None:
        from robotactile_benchmark.execution import write_live_univtac_artifact
        from robotactile_benchmark.trials import Condition
        from tests.test_live_univtac_artifacts import _capture

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded, evidence = _capture(root, Condition.CLEAN)
            artifact = root / "artifact"
            write_live_univtac_artifact(artifact, loaded, evidence)
            binding = root / "binding.json"
            binding.write_text("{}")
            row = {
                "model": "act",
                "live_artifact": str(artifact),
                "binding_path": str(binding),
            }
            resolved = SimpleNamespace(
                manifest=SimpleNamespace(checkpoint_sha256="b" * 64)
            )
            with patch(
                "robotactile_benchmark.integrations.runtime_config.resolve_act_runtime_artifacts",
                return_value=resolved,
            ):
                report, _ = audit(row)
                self.assertGreater(report["retained_observations"], 0)
                self.assertTrue(report["record_links_valid"])
                self.assertTrue(report["observation_arrays_finite"])
                self.assertGreater(report["executed_action_count"], 0)
                (artifact / "untrusted-extra.json").write_text("{}")
                with self.assertRaises(ValueError):
                    audit(row)

    def test_missing_runtimes_are_isolated_for_all_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "output"
            rows = [
                {
                    "model": model,
                    **{
                        key: str(root / "missing")
                        for key in (
                            "binding_path",
                            "package",
                            "live_artifact",
                            "source_root",
                            "runtime_python",
                        )
                    },
                    "code": str(root),
                }
                for model in sorted(MODELS)
            ]
            manifest.write_text(json.dumps({"models": rows}))
            with (
                patch(
                    "sys.argv",
                    ["run", "--manifest", str(manifest), "--output", str(output)],
                ),
                self.assertRaises(SystemExit) as raised,
            ):
                main()
            self.assertEqual(raised.exception.code, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["models"]), 5)
            self.assertTrue(all(not row["passed"] for row in summary["models"]))


if __name__ == "__main__":
    unittest.main()

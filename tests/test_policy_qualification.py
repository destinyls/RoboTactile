from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC, PolicyIdentity
from robotactile_benchmark.qualification import (
    ACT_EVIDENCE_TYPE,
    N0_EVIDENCE_TYPE,
    PolicyQualificationError,
    PolicyQualificationReceipt,
    policy_qualification_receipt_bytes,
    run_cpu_policy_qualification,
    write_policy_qualification_receipt,
)

PAPER_ROOT = Path(__file__).resolve().parents[2]
ACT_RUNTIME_PATH = (
    "../visual-tactile_world_model_pipeline/deployment/ACTStrict/deploy_policy.py"
)


def _identity(kind: str = "act", suffix: str = "") -> PolicyIdentity:
    return PolicyIdentity(
        system_id=f"{kind}-touch-cpu-qualification{suffix}",
        checkpoint_sha256=("a" if kind == "act" else "c") * 64,
        config_sha256=("b" if kind == "act" else "d") * 64,
        action_spec=ACTION_SPEC,
        consumes_tactile=True,
        supports_structural_absence=False,
    )


def _require_external_source(kind: str) -> None:
    source = PAPER_ROOT / ACT_RUNTIME_PATH if kind == "act" else PAPER_ROOT / "N0-TWAM"
    if not source.exists():
        raise unittest.SkipTest(
            f"requires the reviewed external {kind.upper()} source checkout"
        )


def _run(kind: str, *, task_id: str = "insert_HDMI") -> PolicyQualificationReceipt:
    _require_external_source(kind)
    return run_cpu_policy_qualification(
        kind,
        identity=_identity(kind),
        workspace_root=PAPER_ROOT,
        task_id=task_id,
    )


class PolicyQualificationTests(unittest.TestCase):
    def test_act_protocol_binds_pull_out_key_observation_task(self) -> None:
        receipt = _run("act", task_id="pull_out_key")

        self.assertTrue(receipt.passed)
        self.assertEqual(receipt.task_id, "pull_out_key")

    def test_act_receipt_is_byte_identical_source_bound_and_honest(self) -> None:
        first = _run("act")
        second = _run("act")

        self.assertEqual(first.evidence_type, ACT_EVIDENCE_TYPE)
        self.assertEqual(
            policy_qualification_receipt_bytes(first),
            policy_qualification_receipt_bytes(second),
        )
        self.assertEqual(first.act_runtime_source_path, ACT_RUNTIME_PATH)
        runtime = (PAPER_ROOT / ACT_RUNTIME_PATH).resolve(strict=True)
        self.assertEqual(
            first.act_runtime_source_sha256,
            hashlib.sha256(runtime.read_bytes()).hexdigest(),
        )
        self.assertEqual(first.artifact_status, "artifact_not_loaded")
        self.assertEqual(first.no_touch_status, "artifact_unavailable")
        self.assertEqual(first.protocol_result["runtime_reset_count"], 1)
        self.assertEqual(first.protocol_result["runtime_infer_count"], 1)
        self.assertEqual(first.protocol_result["runtime_commit_model_calls"], 0)
        self._assert_honest_and_zero_absence(first)
        for path, digest in first.task6a_source_hashes.items():
            self.assertEqual(
                digest,
                hashlib.sha256((PAPER_ROOT / path).read_bytes()).hexdigest(),
            )
        self.assertEqual(first.task6b_source_hashes, {})
        self.assertIsNone(first.task6b_source_manifest_sha256)

    def test_n0_receipt_runs_actual_cross_package_two_phase_protocol(self) -> None:
        first = _run("n0")
        second = _run("n0")

        self.assertEqual(first.evidence_type, N0_EVIDENCE_TYPE)
        self.assertEqual(
            policy_qualification_receipt_bytes(first),
            policy_qualification_receipt_bytes(second),
        )
        self.assertEqual(
            first.protocol_result["request_operations"],
            ("reset", "infer", "prepare_commit", "finalize_commit"),
        )
        self.assertEqual(first.protocol_result["prepare_engine_commit_count"], 0)
        self.assertEqual(first.protocol_result["final_engine_commit_count"], 1)
        self.assertEqual(first.protocol_result["final_cache_position"], 1)
        self.assertEqual(first.protocol_result["client_state"], "ready")
        self.assertEqual(first.protocol_result["grounding_frame_count"], 8)
        self.assertTrue(first.protocol_result["fake_gateway_executed"])
        self.assertIsNone(first.act_runtime_source_path)
        self.assertIsNone(first.act_runtime_source_sha256)
        self._assert_honest_and_zero_absence(first)
        self.assertGreater(len(first.task6b_source_hashes), 5)
        for path, digest in first.task6b_source_hashes.items():
            self.assertEqual(
                digest,
                hashlib.sha256((PAPER_ROOT / path).read_bytes()).hexdigest(),
            )

    def _assert_honest_and_zero_absence(
        self, receipt: PolicyQualificationReceipt
    ) -> None:
        self.assertTrue(receipt.passed)
        self.assertFalse(receipt.live_model_executed)
        self.assertFalse(receipt.twam_server_executed)
        self.assertFalse(receipt.isaac_sim_executed)
        self.assertFalse(receipt.task_success_measured)
        self.assertFalse(receipt.tls_authenticated)
        for operator_id in ("A1_stream_absence", "A2_frame_erasure"):
            result = receipt.structural_absence_results[operator_id]
            self.assertEqual(result["terminal_status"], "unsupported_contract")
            self.assertEqual(result["backend_reset_count"], 0)
            self.assertEqual(result["policy_effect_count"], 0)
            self.assertEqual(result["transport_effect_count"], 0)

    def test_receipt_is_deep_immutable_self_hashed_and_fail_closed(self) -> None:
        receipt = _run("act")
        document = receipt.to_dict()
        restored = PolicyQualificationReceipt.from_dict(document)
        self.assertEqual(restored, receipt)
        self.assertEqual(document["receipt_sha256"], receipt.sha256)

        with self.assertRaises(TypeError):
            receipt.policy_identity["system_id"] = "tampered"  # type: ignore[index]
        with self.assertRaises(TypeError):
            receipt.protocol_result["runtime_reset_count"] = 9  # type: ignore[index]
        with self.assertRaisesRegex(PolicyQualificationError, "evidence"):
            replace(receipt, evidence_type="live_act_policy")
        for field in (
            "live_model_executed",
            "twam_server_executed",
            "isaac_sim_executed",
            "task_success_measured",
            "tls_authenticated",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(PolicyQualificationError, "evidence"),
            ):
                replace(receipt, **{field: True})
        coordinated = dict(receipt.task6a_source_hashes)
        coordinated[next(iter(coordinated))] = "0" * 64
        with self.assertRaisesRegex(PolicyQualificationError, "source"):
            replace(
                receipt,
                task6a_source_hashes=coordinated,
                task6a_source_manifest_sha256="0" * 64,
            )
        tampered = dict(document)
        tampered["passed"] = False
        with self.assertRaisesRegex(PolicyQualificationError, "hash"):
            PolicyQualificationReceipt.from_dict(tampered)

    def test_atomic_no_clobber_and_idempotent_identical_write(self) -> None:
        receipt = _run("act")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "policy-qualification.json"
            self.assertTrue(write_policy_qualification_receipt(output, receipt))
            expected = output.read_bytes()
            self.assertFalse(write_policy_qualification_receipt(output, receipt))
            self.assertEqual(output.read_bytes(), expected)
            with self.assertRaisesRegex(PolicyQualificationError, "already exists"):
                write_policy_qualification_receipt(
                    output,
                    run_cpu_policy_qualification(
                        "act",
                        identity=_identity("act", "-different"),
                        workspace_root=PAPER_ROOT,
                        task_id="insert_HDMI",
                    ),
                )
            self.assertEqual(output.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()

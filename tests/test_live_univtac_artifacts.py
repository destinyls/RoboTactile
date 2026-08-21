"""Live UniVTAC artifact round-trip and fail-closed security tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution import (
    LIVE_ARTIFACT_EVIDENCE_LEVEL,
    LiveArtifactValidationError,
    LivePolicyKind,
    LiveUniVTACRunRequest,
    load_live_univtac_artifact,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import Condition, RestorationMode


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fault(root: Path) -> Path:
    manifest = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=23,
        start_index=1,
        stop_index=3,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )
    path = root / "fault.json"
    path.write_bytes(_canonical(manifest.to_dict()))
    return path


def _request(root: Path, condition: Condition) -> LiveUniVTACRunRequest:
    fault_path = _fault(root) if condition is not Condition.CLEAN else None
    return LiveUniVTACRunRequest(
        task_id="insert_HDMI",
        condition=condition,
        policy_kind=LivePolicyKind.ACT,
        base_system_id="live-artifact-test-policy-v1",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        base_system_manifest_sha256=None,
        initial_seed=10,
        exogenous_seed=20,
        max_control_cycles=4,
        max_observation_steps=5,
        execute_action_steps=1,
        wall_timeout_s=5.0,
        upstream_root=root / "upstream",
        runtime_dir=root / "runtime",
        output_dir=None,
        fault_manifest_path=fault_path,
        rest_references_path=None,
        restoration_index=3 if condition is Condition.RESTORED else None,
        restoration_mode=(
            RestorationMode.VALID_STREAM_RESUME
            if condition is Condition.RESTORED
            else None
        ),
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu",
        simulator_device=None,
        launcher_args={"headless": True},
    )


def _capture(root: Path, condition: Condition):
    loaded = load_live_univtac_run(_request(root, condition))
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            success_predicate_id=loaded.run_spec.success_predicate_id,
        ),
        DeterministicFakePolicy.for_trial(loaded.trial),
        fault_manifest=loaded.fault_manifest,
    )
    return loaded, evidence


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _repin_member(bundle: Path, relative: str, raw: bytes) -> None:
    (bundle / relative).write_bytes(raw)
    receipt_path = bundle / "root_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for member in receipt["members"]:
        if member["path"] == relative:
            member["sha256"] = hashlib.sha256(raw).hexdigest()
            member["size_bytes"] = len(raw)
    receipt_path.write_bytes(_canonical(receipt))


class LiveUniVTACArtifactRoundTripTests(unittest.TestCase):
    def test_clean_faulted_restored_round_trip_is_typed_and_byte_identical(
        self,
    ) -> None:
        for condition in (
            Condition.CLEAN,
            Condition.FAULTED,
            Condition.RESTORED,
        ):
            with (
                self.subTest(condition=condition),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                loaded, evidence = _capture(root, condition)
                first = root / "first"
                second = root / "second"

                receipt = write_live_univtac_artifact(first, loaded, evidence)
                write_live_univtac_artifact(second, loaded, evidence)
                reopened = load_live_univtac_artifact(first)

                self.assertEqual(receipt.evidence_level, LIVE_ARTIFACT_EVIDENCE_LEVEL)
                self.assertNotEqual(
                    receipt.evidence_level, "deterministic_cpu_fake_closed_loop"
                )
                self.assertFalse(reopened.root_receipt.simulator_qualification_claimed)
                self.assertEqual(reopened.run_content_sha256, loaded.content_sha256)
                self.assertEqual(reopened.trial, loaded.trial)
                self.assertEqual(reopened.run_spec, loaded.run_spec)
                self.assertEqual(reopened.fault_manifest, loaded.fault_manifest)
                self.assertEqual(reopened.evidence.result, evidence.result)
                self.assertEqual(
                    canonical_hash(reopened.evidence.finalization),
                    canonical_hash(evidence.finalization),
                )
                self.assertEqual(_inventory(first), _inventory(second))

    def test_identical_nonempty_target_is_verified_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded, evidence = _capture(root, Condition.FAULTED)
            output = root / "bundle"
            first = write_live_univtac_artifact(output, loaded, evidence)
            before = _inventory(output)
            second = write_live_univtac_artifact(output, loaded, evidence)

            self.assertEqual(first, second)
            self.assertEqual(before, _inventory(output))

    def test_profile_accepts_600_observation_budget_without_cpu_fake_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = replace(
                _request(root, Condition.CLEAN),
                max_control_cycles=599,
                max_observation_steps=600,
            )
            loaded = load_live_univtac_run(request)

            self.assertEqual(loaded.run_spec.max_observation_steps, 600)


class LiveUniVTACArtifactTamperTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        loaded, evidence = _capture(root, Condition.FAULTED)
        output = root / "bundle"
        write_live_univtac_artifact(output, loaded, evidence)
        return output

    def test_unknown_member_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp))
            (bundle / "unknown.bin").write_bytes(b"unexpected")
            with self.assertRaises(LiveArtifactValidationError):
                load_live_univtac_artifact(bundle)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp))
            (bundle / "escape").symlink_to(bundle / "root_receipt.json")
            with self.assertRaises(LiveArtifactValidationError):
                load_live_univtac_artifact(bundle)

    def test_descriptor_and_payload_provenance_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp))
            trace_path = bundle / "delivery_trace.json"
            document = json.loads(trace_path.read_text(encoding="utf-8"))
            descriptor = document["finalization"]["clean_records"][0]["observation"][
                "tactile"
            ][0]["payload"]
            descriptor["array_sha256"] = "0" * 64
            _repin_member(bundle, "delivery_trace.json", _canonical(document))
            with self.assertRaises(LiveArtifactValidationError):
                load_live_univtac_artifact(bundle)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp))
            trace_path = bundle / "delivery_trace.json"
            document = json.loads(trace_path.read_text(encoding="utf-8"))
            document["finalization"]["delivered_records"][0]["provenance"][0][
                "payload_sha256"
            ] = "0" * 64
            _repin_member(bundle, "delivery_trace.json", _canonical(document))
            with self.assertRaises(LiveArtifactValidationError):
                load_live_univtac_artifact(bundle)


if __name__ == "__main__":
    unittest.main()

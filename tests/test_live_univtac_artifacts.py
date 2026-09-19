"""Live UniVTAC artifact round-trip and fail-closed security tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.closed_loop import ClosedLoopExecutionEvidence
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
    LoadedLiveUniVTACRun,
    load_live_univtac_artifact,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LIVE_REQUEST_IDENTITY_SEMANTIC_VERSION,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.trials import Condition


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
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu",
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
    )


def _capture(
    root: Path, condition: Condition
) -> tuple[LoadedLiveUniVTACRun, ClosedLoopExecutionEvidence]:
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


def _capture_n0(
    root: Path,
) -> tuple[LoadedLiveUniVTACRun, ClosedLoopExecutionEvidence]:
    request = replace(
        _request(root, Condition.CLEAN),
        task_id="pull_out_key",
        policy_kind=LivePolicyKind.N0,
        max_control_cycles=1,
        max_observation_steps=9,
        execute_action_steps=24,
        act_device_name=None,
        n0_source_commit="9" * 40,
        n0_normalizer_sha256="d" * 64,
        n0_serve_bundle_sha256="e" * 64,
        n0_prompt_manifest_sha256="f" * 64,
    )
    loaded = load_live_univtac_run(request)
    backend = DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )
    backend.action_spec = loaded.trial.action_spec
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        DeterministicFakePolicy.for_trial(loaded.trial),
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
    def test_terminal_a2_erasure_exports_and_reopens_strict_invalid(self) -> None:
        self._terminal_a2_round_trip(censored=False)

    def test_terminal_a2_censoring_exports_and_reopens_explicit_metrics(self) -> None:
        self._terminal_a2_round_trip(censored=True)

    def _terminal_a2_round_trip(self, *, censored: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _request(root, Condition.FAULTED)
            manifest = FaultManifest(
                operator_id="A2_frame_erasure",
                severity_level=3,
                operator_seed=23,
                start_index=4,
                stop_index=5,
                sensor_slots=("left",),
                observability=Observability.DECLARED,
                parameters={"a2_end_policy": "episode_censored_v1"} if censored else {},
            )
            assert request.fault_manifest_path is not None
            request.fault_manifest_path.write_bytes(_canonical(manifest.to_dict()))
            loaded = load_live_univtac_run(request)
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
            assert evidence.finalization is not None
            report = evidence.finalization.validation
            assert report is not None
            if censored:
                self.assertTrue(report.passed)
                self.assertEqual(report.metrics["a2_resume_status"], "right_censored")
                self.assertTrue(evidence.result.score_eligible)
            else:
                self.assertIn("A2_RESUME_MISSING", report.failure_codes)
                self.assertFalse(report.passed)
                self.assertFalse(evidence.result.score_eligible)

            output = root / "terminal-a2-bundle"
            write_live_univtac_artifact(output, loaded, evidence)
            reopened = load_live_univtac_artifact(output)
            assert reopened.evidence.finalization is not None
            self.assertEqual(reopened.evidence.finalization.validation, report)
            self.assertEqual(reopened.evidence.result, evidence.result)

    def test_n0_ee_artifact_round_trip_preserves_backend_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded, evidence = _capture_n0(root)
            output = root / "n0-ee-bundle"

            write_live_univtac_artifact(output, loaded, evidence)
            reopened = load_live_univtac_artifact(output)

            self.assertEqual(loaded.trial.action_spec, EE8_ACTION_SPEC)
            self.assertEqual(loaded.backend_config.action_spec, EE8_ACTION_SPEC)
            self.assertEqual(reopened.trial.action_spec, EE8_ACTION_SPEC)
            self.assertEqual(reopened.run_content_sha256, loaded.content_sha256)
            self.assertEqual(reopened.evidence.result, evidence.result)
            self.assertEqual(
                reopened.evidence.transition_entries,
                evidence.transition_entries,
            )
            self.assertEqual(
                reopened.evidence.initial_diagnostics,
                evidence.initial_diagnostics,
            )
            self.assertEqual(reopened.root_receipt.semantic_version, "1.1")
            self.assertTrue((output / "transition_trace.json").is_file())

    def test_clean_and_faulted_round_trip_is_typed_and_byte_identical(
        self,
    ) -> None:
        for condition in (
            Condition.CLEAN,
            Condition.FAULTED,
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
                self.assertEqual(
                    reopened.request_identity["semantic_version"],
                    LIVE_REQUEST_IDENTITY_SEMANTIC_VERSION,
                )
                self.assertEqual(reopened.trial, loaded.trial)
                self.assertEqual(reopened.run_spec, loaded.run_spec)
                self.assertEqual(reopened.fault_manifest, loaded.fault_manifest)
                self.assertEqual(reopened.evidence.result, evidence.result)
                self.assertEqual(
                    reopened.evidence.transition_entries,
                    evidence.transition_entries,
                )
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

    def test_capture_profiles_preserve_outcome_with_explicit_storage_semantics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded, evidence = _capture(root, Condition.CLEAN)
            reopened = {}
            for profile in LiveCaptureProfile:
                output = root / profile.value
                write_live_univtac_artifact(
                    output,
                    loaded,
                    evidence,
                    capture_profile=profile,
                )
                reopened[profile] = load_live_univtac_artifact(output)

            full = reopened[LiveCaptureProfile.PAPER_FULL]
            metrics = reopened[LiveCaptureProfile.METRICS_ONLY]
            preview = reopened[LiveCaptureProfile.PREVIEW]
            assert evidence.finalization is not None
            self.assertEqual(full.evidence.result, evidence.result)
            self.assertEqual(metrics.evidence.result, evidence.result)
            self.assertEqual(preview.evidence.result, evidence.result)
            self.assertIsNotNone(full.evidence.finalization)
            self.assertIsNone(metrics.evidence.finalization)
            self.assertIsNone(preview.evidence.finalization)
            self.assertIsNone(metrics.preview_trace)
            self.assertIsNotNone(preview.preview_trace)
            assert preview.preview_trace is not None
            self.assertLessEqual(len(preview.preview_trace.selected_indices), 64)
            self.assertEqual(preview.preview_trace.selected_indices[0], 0)
            self.assertEqual(
                preview.preview_trace.selected_indices[-1],
                len(evidence.finalization.clean_records) - 1,
            )
            self.assertEqual(full.root_receipt.semantic_version, "1.1")
            self.assertEqual(metrics.root_receipt.semantic_version, "1.2")
            self.assertEqual(preview.root_receipt.semantic_version, "1.2")
            self.assertFalse(
                (
                    root / LiveCaptureProfile.METRICS_ONLY.value / "preview_trace.json"
                ).exists()
            )
            metrics_delivery = json.loads(
                (
                    root / LiveCaptureProfile.METRICS_ONLY.value / "delivery_trace.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIsNone(metrics_delivery["finalization"])

    def test_capture_profile_is_part_of_no_clobber_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded, evidence = _capture(root, Condition.CLEAN)
            output = root / "bundle"
            write_live_univtac_artifact(
                output,
                loaded,
                evidence,
                capture_profile=LiveCaptureProfile.METRICS_ONLY,
            )
            with self.assertRaises(FileExistsError):
                write_live_univtac_artifact(
                    output,
                    loaded,
                    evidence,
                    capture_profile=LiveCaptureProfile.PREVIEW,
                )


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

    def test_v1_restored_era_request_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._bundle(Path(tmp))
            identity_path = bundle / "request_identity.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity["semantic_version"] = "1.0"
            _repin_member(bundle, "request_identity.json", _canonical(identity))

            with self.assertRaisesRegex(
                LiveArtifactValidationError, "request identity version"
            ):
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

    def test_compact_capture_summary_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded, evidence = _capture(root, Condition.CLEAN)
            bundle = root / "compact"
            write_live_univtac_artifact(
                bundle,
                loaded,
                evidence,
                capture_profile=LiveCaptureProfile.METRICS_ONLY,
            )
            summary_path = bundle / "capture_summary.json"
            document = json.loads(summary_path.read_text(encoding="utf-8"))
            document["capture_profile"] = LiveCaptureProfile.PREVIEW.value
            _repin_member(bundle, "capture_summary.json", _canonical(document))
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

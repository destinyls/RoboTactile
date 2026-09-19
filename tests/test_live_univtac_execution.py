"""Module-level execution tests for live UniVTAC request orchestration."""

from __future__ import annotations

import json
import os
import platform
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.lifecycle import (
    ClosedLoopResourceCloseError,
)
from robotactile_benchmark.execution import (
    ArtifactExportStatus,
    LiveArtifactExportReceipt,
    LiveExecutionUnavailableError,
    LivePolicyKind,
    LiveUniVTACRunRequest,
    default_live_backend_factory,
    default_live_policy_factory,
    execute_live_univtac_run,
    live_univtac_request_to_dict,
    load_live_univtac_artifact,
    load_live_univtac_request,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.contracts import (
    ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG,
    LIVE_REQUEST_SEMANTIC_VERSION,
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.lifecycle_watchdog import (
    LifecycleIdentity,
    LifecycleStageJournal,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.act_loading import ArtifactUnavailableError
from robotactile_benchmark.trials import Condition, TerminalStatus


def _fault(root: Path, *, stop_index: int = 3) -> Path:
    manifest = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=23,
        start_index=1,
        stop_index=stop_index,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={},
    )
    path = root / "fault.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def _request(
    root: Path,
    *,
    condition: Condition = Condition.CLEAN,
    fault_path: Path | None = None,
    output: bool = False,
    policy_kind: LivePolicyKind = LivePolicyKind.ACT,
) -> LiveUniVTACRunRequest:
    return LiveUniVTACRunRequest(
        task_id="pull_out_key",
        condition=condition,
        policy_kind=policy_kind,
        base_system_id="module-e2e-policy-v1",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        base_system_manifest_sha256=None,
        initial_seed=10,
        exogenous_seed=20,
        max_control_cycles=1 if policy_kind is LivePolicyKind.N0 else 3,
        max_observation_steps=9 if policy_kind is LivePolicyKind.N0 else 5,
        execute_action_steps=24 if policy_kind is LivePolicyKind.N0 else 1,
        wall_timeout_s=5.0,
        upstream_root=root / "upstream",
        runtime_dir=root / "runtime",
        output_dir=root / "output" if output else None,
        fault_manifest_path=fault_path,
        rest_references_path=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu" if policy_kind is LivePolicyKind.ACT else None,
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit="9" * 40 if policy_kind is LivePolicyKind.N0 else None,
        n0_normalizer_sha256="d" * 64 if policy_kind is LivePolicyKind.N0 else None,
        n0_serve_bundle_sha256="e" * 64 if policy_kind is LivePolicyKind.N0 else None,
        n0_prompt_manifest_sha256="f" * 64
        if policy_kind is LivePolicyKind.N0
        else None,
    )


def _document(request: LiveUniVTACRunRequest, base: Path) -> dict[str, object]:
    document = live_univtac_request_to_dict(request)
    for name in (
        "upstream_root",
        "runtime_dir",
        "output_dir",
        "fault_manifest_path",
        "rest_references_path",
        "matched_no_touch_artifact_path",
    ):
        value = document[name]
        if value is not None:
            document[name] = Path(value).relative_to(base).as_posix()
    return document


class _Harness:
    def __init__(self, *, fail_policy: bool = False, fail_infer: bool = False) -> None:
        self.fail_policy = fail_policy
        self.fail_infer = fail_infer
        self.backends: list[DeterministicFakeBackend] = []
        self.policies: list[DeterministicFakePolicy] = []

    def backend(self, loaded: Any) -> DeterministicFakeBackend:
        backend = DeterministicFakeBackend(
            make_synthetic_episode(length=10),
            success_predicate_id=loaded.run_spec.success_predicate_id,
        )
        self.backends.append(backend)
        return backend

    def policy(self, loaded: Any) -> DeterministicFakePolicy:
        if self.fail_policy:
            raise RuntimeError("policy construction failed")
        policy = DeterministicFakePolicy.for_trial(
            loaded.trial,
            supports_structural_absence=True,
            fail_on_infer=self.fail_infer,
        )
        self.policies.append(policy)
        return policy


class LiveRequestLoadingTests(unittest.TestCase):
    def test_launcher_contract_is_exact_and_rejects_missing_or_conflicting_kit_args(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            request = _request(Path(temporary))
            invalid = (
                (
                    {"enable_cameras": True, "headless": True},
                    "fields mismatch",
                ),
                (
                    {
                        **production_univtac_launcher_args(),
                        "kit_args": (
                            f"{ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG} "
                            f"{ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG}"
                        ),
                    },
                    "contain exactly",
                ),
                (
                    {
                        **production_univtac_launcher_args(),
                        "kit_args": "--/app/hangDetector/enabled=true",
                    },
                    "contain exactly",
                ),
                (
                    {**production_univtac_launcher_args(), "headless": False},
                    "headless must be true",
                ),
                (
                    {
                        **production_univtac_launcher_args(),
                        "enable_cameras": False,
                    },
                    "enable_cameras must be true",
                ),
            )

            self.assertEqual(
                dict(request.launcher_args), production_univtac_launcher_args()
            )
            for launcher_args, message in invalid:
                with (
                    self.subTest(launcher_args=launcher_args),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    replace(request, launcher_args=launcher_args)

    def test_content_hash_excludes_paths_but_binds_loaded_fault(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_root = Path(temporary) / "first"
            second_root = Path(temporary) / "second"
            first_root.mkdir()
            second_root.mkdir()
            first = load_live_univtac_run(
                _request(
                    first_root,
                    condition=Condition.FAULTED,
                    fault_path=_fault(first_root),
                )
            )
            second = load_live_univtac_run(
                _request(
                    second_root,
                    condition=Condition.FAULTED,
                    fault_path=_fault(second_root),
                )
            )

        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.trial, second.trial)
        self.assertNotEqual(first.request.upstream_root, second.request.upstream_root)
        with self.assertRaises(TypeError):
            first.request.launcher_args["headless"] = False

    def test_request_file_loader_is_strict_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            document = _document(request, root)
            request_path = root / "request.json"
            request_path.write_text(json.dumps(document), encoding="utf-8")

            loaded = load_live_univtac_request(request_path)
            document["unknown"] = True
            request_path.write_text(json.dumps(document), encoding="utf-8")

            self.assertEqual(loaded.upstream_root, (root / "upstream").absolute())
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                load_live_univtac_request(request_path)

    def test_v1_restored_era_request_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            self.assertEqual(request.semantic_version, LIVE_REQUEST_SEMANTIC_VERSION)
            document = _document(request, root)
            document["semantic_version"] = "1.0"
            request_path = root / "legacy-request.json"
            request_path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "semantic version"):
                load_live_univtac_request(request_path)

    def test_condition_mismatches_fail_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fault_path = _fault(root, stop_index=3)
            with self.assertRaisesRegex(ValueError, "clean"):
                _request(root, fault_path=fault_path)
            with self.assertRaisesRegex(ValueError, "requires a fault"):
                _request(root, condition=Condition.FAULTED)
            with self.assertRaisesRegex(ValueError, "only valid for no-touch"):
                replace(_request(root), matched_no_touch_system_id="unexpected-control")


class LiveExecutionE2ETests(unittest.TestCase):
    def test_execution_records_identity_bound_lifecycle_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root)
            loaded = load_live_univtac_run(request)
            journal = LifecycleStageJournal.create(
                root / "lifecycle.journal",
                LifecycleIdentity(
                    campaign_manifest_sha256="1" * 64,
                    request_file_sha256="2" * 64,
                    trial_manifest_sha256=loaded.trial.sha256,
                    task_id=request.task_id,
                    ordinal=0,
                    attempt_id="test-attempt",
                ),
            )
            journal.record("process_spawn")
            harness = _Harness()

            execute_live_univtac_run(
                request,
                backend_factory=harness.backend,
                policy_factory=harness.policy,
                lifecycle_journal=journal,
            )
            stages = [receipt.stage for receipt in journal.records()]

        self.assertEqual(
            stages[:6],
            [
                "process_spawn",
                "capability_preflight",
                "run_loading",
                "backend_construction",
                "backend_ready",
                "policy_construction",
            ],
        )
        self.assertIn("reset", stages)
        self.assertIn("infer", stages)
        self.assertIn("execute", stages)
        self.assertEqual(stages[-2:], ["close", "execution_completed"])

    def test_clean_module_e2e_returns_typed_capture_without_fake_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _Harness()
            result = execute_live_univtac_run(
                _request(Path(temporary)),
                backend_factory=harness.backend,
                policy_factory=harness.policy,
            )

        self.assertEqual(result.evidence.result.terminal_status, TerminalStatus.TIMEOUT)
        self.assertEqual(result.artifact_export, ArtifactExportStatus.NOT_REQUESTED)
        self.assertIsNone(result.artifact_receipt)
        self.assertFalse(result.simulator_qualification_claimed)
        self.assertEqual(harness.backends[0].close_count, 1)
        self.assertEqual(harness.policies[0].close_count, 1)

    def test_faulted_module_e2e_preserves_delivery_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _request(
                root,
                condition=Condition.FAULTED,
                fault_path=_fault(root, stop_index=3),
                output=True,
            )
            harness = _Harness()
            result = execute_live_univtac_run(
                request,
                backend_factory=harness.backend,
                policy_factory=harness.policy,
            )

            self.assertTrue(result.evidence.finalization.validation.passed)
            self.assertEqual(
                result.artifact_export,
                ArtifactExportStatus.UNSUPPORTED_CONTRACT,
            )
            self.assertFalse(request.output_dir.exists())

    def test_live_export_hook_is_typed_and_does_not_reclose_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness()
            calls: list[Path] = []
            lifecycle: list[str] = []

            def backend_factory(loaded: Any) -> DeterministicFakeBackend:
                backend = harness.backend(loaded)
                original_close = backend.close

                def close() -> None:
                    lifecycle.append("backend_close")
                    original_close()

                backend.close = close  # type: ignore[method-assign]
                return backend

            def exporter(output: Path, loaded: Any, evidence: Any):
                calls.append(output)
                lifecycle.append("export")
                self.assertEqual(harness.backends[0].close_count, 0)
                output.mkdir()
                return LiveArtifactExportReceipt.for_execution(loaded, evidence)

            result = execute_live_univtac_run(
                _request(
                    root,
                    condition=Condition.FAULTED,
                    fault_path=_fault(root),
                    output=True,
                ),
                backend_factory=backend_factory,
                policy_factory=harness.policy,
                artifact_exporter=exporter,
            )

        self.assertEqual(result.artifact_export, ArtifactExportStatus.EXPORTED)
        self.assertEqual(calls, [root / "output"])
        self.assertEqual(lifecycle, ["export", "backend_close"])
        self.assertEqual(harness.backends[0].close_count, 1)
        self.assertEqual(harness.policies[0].close_count, 1)

    def test_execute_crash_is_published_and_strictly_reloadable_before_close(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness()
            lifecycle: list[str] = []

            def backend_factory(loaded: Any) -> DeterministicFakeBackend:
                backend = harness.backend(loaded)
                original_close = backend.close

                def fail_execute(actions: Any) -> Any:
                    del actions
                    backend.execute_count += 1
                    raise TypeError("fake execute failure")

                def close() -> None:
                    lifecycle.append("backend_close")
                    original_close()

                backend.execute = fail_execute  # type: ignore[method-assign]
                backend.close = close  # type: ignore[method-assign]
                return backend

            def exporter(output: Path, loaded: Any, evidence: Any):
                lifecycle.append("export")
                return write_live_univtac_artifact(output, loaded, evidence)

            request = _request(root, output=True)
            result = execute_live_univtac_run(
                request,
                backend_factory=backend_factory,
                policy_factory=harness.policy,
                artifact_exporter=exporter,
            )
            reopened = load_live_univtac_artifact(request.output_dir)

        self.assertEqual(result.artifact_export, ArtifactExportStatus.EXPORTED)
        self.assertEqual(result.evidence.result.terminal_status, TerminalStatus.CRASH)
        self.assertEqual(result.evidence.result.failure_code, "execute_failed")
        self.assertEqual(result.evidence.result.control_cycle_count, 0)
        self.assertEqual(result.evidence.result.observation_count, 1)
        self.assertEqual(result.evidence.action_entries, ())
        self.assertEqual(reopened.evidence.result, result.evidence.result)
        self.assertEqual(
            reopened.evidence.finalization.clean_trace_sha256,
            result.evidence.finalization.clean_trace_sha256,
        )
        self.assertEqual(reopened.evidence.action_entries, ())
        self.assertEqual(lifecycle, ["export", "backend_close"])
        self.assertEqual(harness.backends[0].close_count, 1)
        self.assertEqual(harness.policies[0].close_count, 1)

    def test_published_artifact_is_not_success_when_runtime_close_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness()

            def backend_factory(loaded: Any) -> DeterministicFakeBackend:
                backend = DeterministicFakeBackend(
                    make_synthetic_episode(length=10),
                    success_predicate_id=loaded.run_spec.success_predicate_id,
                    raise_on_close=True,
                )
                harness.backends.append(backend)
                return backend

            def exporter(output: Path, loaded: Any, evidence: Any):
                output.mkdir()
                return LiveArtifactExportReceipt.for_execution(loaded, evidence)

            with self.assertRaises(ClosedLoopResourceCloseError) as captured:
                execute_live_univtac_run(
                    _request(root, output=True),
                    backend_factory=backend_factory,
                    policy_factory=harness.policy,
                    artifact_exporter=exporter,
                )

            self.assertEqual(
                captured.exception.failure_codes,
                ("backend_close_failed",),
            )
            self.assertTrue((root / "output").is_dir())
            self.assertEqual(harness.backends[0].close_count, 1)
            self.assertEqual(harness.policies[0].close_count, 1)

    def test_export_receipt_without_published_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            harness = _Harness()

            def empty_exporter(output: Path, loaded: Any, evidence: Any):
                return LiveArtifactExportReceipt.for_execution(loaded, evidence)

            with self.assertRaisesRegex(RuntimeError, "did not publish"):
                execute_live_univtac_run(
                    _request(root, output=True),
                    backend_factory=harness.backend,
                    policy_factory=harness.policy,
                    artifact_exporter=empty_exporter,
                )

    def test_no_touch_unavailable_fails_before_backend_or_policy_factory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = replace(
                _request(root, condition=Condition.NO_TOUCH),
                base_system_manifest_sha256="d" * 64,
            )
            harness = _Harness()

            with self.assertRaises(ArtifactUnavailableError) as captured:
                execute_live_univtac_run(
                    request,
                    backend_factory=harness.backend,
                    policy_factory=harness.policy,
                )

        self.assertEqual(captured.exception.code, "artifact_unavailable")
        self.assertFalse(harness.backends)
        self.assertFalse(harness.policies)

    def test_policy_construction_failure_closes_started_backend_exactly_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            harness = _Harness(fail_policy=True)
            with self.assertRaisesRegex(RuntimeError, "policy construction"):
                execute_live_univtac_run(
                    _request(Path(temporary)),
                    backend_factory=harness.backend,
                    policy_factory=harness.policy,
                )

        self.assertEqual(harness.backends[0].close_count, 1)

    def test_runner_crash_and_export_failure_do_not_double_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            crash_harness = _Harness(fail_infer=True)
            crash = execute_live_univtac_run(
                _request(Path(temporary)),
                backend_factory=crash_harness.backend,
                policy_factory=crash_harness.policy,
            )
            self.assertEqual(
                crash.evidence.result.terminal_status, TerminalStatus.CRASH
            )
            self.assertEqual(crash_harness.backends[0].close_count, 1)
            self.assertEqual(crash_harness.policies[0].close_count, 1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_harness = _Harness()

            def failing_exporter(output: Path, loaded: Any, evidence: Any):
                raise RuntimeError("export failed")

            with self.assertRaisesRegex(RuntimeError, "export failed"):
                execute_live_univtac_run(
                    _request(
                        root,
                        condition=Condition.FAULTED,
                        fault_path=_fault(root),
                        output=True,
                    ),
                    backend_factory=export_harness.backend,
                    policy_factory=export_harness.policy,
                    artifact_exporter=failing_exporter,
                )
            self.assertEqual(export_harness.backends[0].close_count, 1)
            self.assertEqual(export_harness.policies[0].close_count, 1)


class LiveDefaultFactoryTests(unittest.TestCase):
    def test_default_backend_rejects_non_linux_before_isaac_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            loaded = load_live_univtac_run(_request(Path(temporary)))
            with (
                patch.object(platform, "system", return_value="Darwin"),
                patch(
                    "robotactile_benchmark.execution.live_univtac.launch_univtac_runtime"
                ) as launcher,
                self.assertRaises(LiveExecutionUnavailableError) as captured,
            ):
                default_live_backend_factory(loaded)

        self.assertEqual(captured.exception.code, "live_univtac_requires_linux")
        launcher.assert_not_called()

    def test_models_require_manifest_bound_official_factory_before_backend(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for policy_kind in (LivePolicyKind.ACT, LivePolicyKind.N0):
                with self.subTest(policy_kind=policy_kind):
                    request = _request(
                        root / policy_kind.value, policy_kind=policy_kind
                    )
                    harness = _Harness()
                    with self.assertRaisesRegex(
                        LiveExecutionUnavailableError, "official"
                    ):
                        execute_live_univtac_run(
                            request, backend_factory=harness.backend
                        )
                    self.assertFalse(harness.backends)

                    loaded = load_live_univtac_run(request)
                    with self.assertRaisesRegex(
                        LiveExecutionUnavailableError, "official"
                    ):
                        default_live_policy_factory(loaded)


def test_artifact_export_failure_emits_redacted_marker_before_close(
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret = "SECRET_EXPORT_PAYLOAD"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        harness = _Harness()

        def backend_factory(loaded: Any) -> DeterministicFakeBackend:
            backend = harness.backend(loaded)
            original_close = backend.close

            def close() -> None:
                os.write(2, b"TEST_BACKEND_CLOSE\n")
                original_close()

            backend.close = close  # type: ignore[method-assign]
            return backend

        def failing_exporter(output: Path, loaded: Any, evidence: Any) -> Any:
            del output, loaded, evidence
            raise RuntimeError(secret)

        with pytest.raises(RuntimeError, match=secret):
            execute_live_univtac_run(
                _request(root, output=True),
                backend_factory=backend_factory,
                policy_factory=harness.policy,
                artifact_exporter=failing_exporter,
            )

    stderr = capfd.readouterr().err
    marker = "ROBOTACTILE_CLOSED_LOOP_CRASH stage=artifact_export"
    assert marker in stderr
    assert "exception_type=builtins.RuntimeError" in stderr
    assert "failure_code=artifact_export_failed" in stderr
    assert secret not in stderr
    assert stderr.index(marker) < stderr.index("TEST_BACKEND_CLOSE")
    assert harness.backends[0].close_count == 1
    assert harness.policies[0].close_count == 1

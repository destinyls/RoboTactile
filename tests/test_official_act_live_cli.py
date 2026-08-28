"""Injected E2E tests for the official UniVTAC ACT live CLI wiring."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from robotactile_benchmark.cli import main
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.execution import (
    LiveExecutionUnavailableError,
    LivePolicyKind,
    LiveUniVTACRunRequest,
    live_univtac_request_to_dict,
    load_live_univtac_artifact,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.official_act import (
    build_official_act_live_binding,
    execute_official_act_live_run,
    official_act_profile,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.trials import Condition

_TACTILE_CONFIG_SHA256 = (
    "acdab30e50fa7280918804c4a196f75a6db6e854a3c7a86a0ddd84791c533397"
)
_VISION_CONFIG_SHA256 = (
    "427c54337b56b0958606de4a235b6e13527877f75c21c2939045b5a6b1375cc2"
)


def _request(root: Path, condition: Condition) -> LiveUniVTACRunRequest:
    no_touch = condition is Condition.NO_TOUCH
    profile = "vision_only" if no_touch else "univtac"
    return LiveUniVTACRunRequest(
        task_id="pull_out_key",
        condition=condition,
        policy_kind=LivePolicyKind.ACT,
        base_system_id="official-univtac-act",
        dataset_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256=(_VISION_CONFIG_SHA256 if no_touch else _TACTILE_CONFIG_SHA256),
        base_system_manifest_sha256="d" * 64 if no_touch else None,
        initial_seed=10,
        exogenous_seed=20,
        max_control_cycles=2,
        max_observation_steps=3,
        execute_action_steps=1,
        wall_timeout_s=5.0,
        upstream_root=root / "UniVTAC",
        runtime_dir=root / "runtime",
        output_dir=root / "artifact",
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=("official-vision-only-act" if no_touch else None),
        matched_no_touch_artifact_path=(
            root / "checkpoints" / "pull_out_key" / profile / "policy_last.ckpt"
            if no_touch
            else None
        ),
        act_device_name="cuda:0",
        simulator_device="cuda:0",
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit=None,
        n0_normalizer_sha256=None,
        n0_serve_bundle_sha256=None,
        n0_prompt_manifest_sha256=None,
    )


def _write_request(path: Path, request: LiveUniVTACRunRequest) -> None:
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
            document[name] = Path(value).relative_to(path.parent).as_posix()
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _backend(loaded: Any) -> DeterministicFakeBackend:
    return DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )


class _PolicyLoader:
    def __init__(self) -> None:
        self.profiles: list[OfficialACTProfile] = []

    def __call__(self, identity: Any, request: Any) -> DeterministicFakePolicy:
        self.profiles.append(request.profile)
        return DeterministicFakePolicy(identity)


class OfficialACTLiveBindingTests(unittest.TestCase):
    def test_condition_selects_tactile_or_matched_vision_only_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition, expected in (
                (Condition.CLEAN, OfficialACTProfile.UNIVTAC),
                (Condition.FAULTED, OfficialACTProfile.UNIVTAC),
                (Condition.RESTORED, OfficialACTProfile.UNIVTAC),
                (Condition.NO_TOUCH, OfficialACTProfile.VISION_ONLY),
            ):
                self.assertIs(official_act_profile(condition), expected)
            for condition, expected in (
                (Condition.CLEAN, OfficialACTProfile.UNIVTAC),
                (Condition.NO_TOUCH, OfficialACTProfile.VISION_ONLY),
            ):
                request = _request(root, condition)
                binding = build_official_act_live_binding(
                    request,
                    artifact_root=root / "checkpoints",
                    stats_sha256="e" * 64,
                    encoder_sha256="f" * 64,
                )
                self.assertIs(binding.manifest.profile, expected)
                self.assertEqual(binding.load_request.device_name, "cuda:0")

    def test_missing_output_and_config_mismatch_fail_before_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root, Condition.CLEAN)
            with self.assertRaisesRegex(ValueError, "output_dir"):
                build_official_act_live_binding(
                    replace(request, output_dir=None),
                    artifact_root=root / "checkpoints",
                    stats_sha256="e" * 64,
                    encoder_sha256="f" * 64,
                )
            with self.assertRaisesRegex(ValueError, "config SHA256"):
                build_official_act_live_binding(
                    replace(request, config_sha256="0" * 64),
                    artifact_root=root / "checkpoints",
                    stats_sha256="e" * 64,
                    encoder_sha256="f" * 64,
                )


class OfficialACTLiveCLITests(unittest.TestCase):
    def test_cli_executes_exports_reloads_and_prints_one_canonical_line(self) -> None:
        for condition, expected_profile in (
            (Condition.CLEAN, OfficialACTProfile.UNIVTAC),
            (Condition.NO_TOUCH, OfficialACTProfile.VISION_ONLY),
        ):
            with (
                self.subTest(condition=condition),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                request = _request(root, condition)
                request_path = root / "request.json"
                _write_request(request_path, request)
                policy_loader = _PolicyLoader()
                stream = io.StringIO()
                with (
                    patch(
                        "robotactile_benchmark.execution.official_act.default_live_backend_factory",
                        _backend,
                    ),
                    patch(
                        "robotactile_benchmark.execution.official_act.load_official_univtac_act_policy",
                        policy_loader,
                    ),
                    contextlib.redirect_stdout(stream),
                ):
                    exit_code = main(
                        [
                            "live-univtac-run",
                            "--request",
                            str(request_path),
                            "--official-act-artifact-root",
                            str(root / "checkpoints"),
                            "--stats-sha256",
                            "e" * 64,
                            "--encoder-sha256",
                            "f" * 64,
                        ]
                    )
                payload = json.loads(stream.getvalue())
                reopened = load_live_univtac_artifact(request.output_dir)

                self.assertEqual(exit_code, 0)
                self.assertEqual(policy_loader.profiles, [expected_profile])
                self.assertEqual(
                    stream.getvalue(),
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n",
                )
                self.assertEqual(
                    payload["evidence_level"],
                    "unqualified_live_univtac_execution_v1",
                )
                self.assertIs(payload["simulator_qualification_claimed"], False)
                self.assertEqual(
                    payload["artifact_root_sha256"], reopened.external_root_sha256
                )
                self.assertEqual(payload["terminal_status"], "timeout")
                self.assertIn("validation_passed", payload)

    def test_custom_no_touch_factory_skips_strict_act_artifact_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root, Condition.NO_TOUCH)
            loader = _PolicyLoader()
            artifact = execute_official_act_live_run(
                request,
                artifact_root=root / "checkpoints",
                stats_sha256="e" * 64,
                encoder_sha256="f" * 64,
                backend_factory=_backend,
                policy_loader=loader,
            )

            self.assertEqual(loader.profiles, [OfficialACTProfile.VISION_ONLY])
            self.assertIs(artifact.trial.condition, Condition.NO_TOUCH)

    def test_runner_still_rejects_tactile_identity_for_no_touch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root, Condition.NO_TOUCH)

            def tactile_loader(identity: Any, load_request: Any):
                tactile_identity = identity.__class__(
                    system_id=identity.system_id,
                    checkpoint_sha256=identity.checkpoint_sha256,
                    config_sha256=identity.config_sha256,
                    action_spec=identity.action_spec,
                    consumes_tactile=True,
                    supports_structural_absence=False,
                )
                return DeterministicFakePolicy(tactile_identity)

            with self.assertRaisesRegex(ValueError, "non-tactile"):
                execute_official_act_live_run(
                    request,
                    artifact_root=root / "checkpoints",
                    stats_sha256="e" * 64,
                    encoder_sha256="f" * 64,
                    backend_factory=_backend,
                    policy_loader=tactile_loader,
                )

    def test_live_dependency_error_propagates_for_nonzero_console_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(root, Condition.CLEAN)
            request_path = root / "request.json"
            _write_request(request_path, request)

            def unavailable_backend(loaded: Any) -> DeterministicFakeBackend:
                raise LiveExecutionUnavailableError(
                    "isaaclab_app_unavailable", "isaaclab.app is unavailable"
                )

            with (
                patch(
                    "robotactile_benchmark.execution.official_act.default_live_backend_factory",
                    unavailable_backend,
                ),
                self.assertRaises(LiveExecutionUnavailableError),
            ):
                main(
                    [
                        "live-univtac-run",
                        "--request",
                        str(request_path),
                        "--official-act-artifact-root",
                        str(root / "checkpoints"),
                        "--stats-sha256",
                        "e" * 64,
                        "--encoder-sha256",
                        "f" * 64,
                    ]
                )


if __name__ == "__main__":
    unittest.main()

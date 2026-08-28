"""Shared builders for clean campaign contract tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from robotactile_benchmark.clean_baseline.seeds import derive_clean_campaign_seed
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.contracts import BackendSignal
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import (
    run_closed_loop_trial_with_evidence,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
)
from robotactile_benchmark.execution import (
    LivePolicyKind,
    LiveUniVTACRunRequest,
    live_univtac_request_to_dict,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.trials import Condition


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def make_layout(tmp_path: Path) -> DeploymentLayout:
    root = tmp_path / "deployment"
    layout = DeploymentLayout(root)
    initialize_deployment_layout(layout)
    return layout


def write_clean_request(
    layout: DeploymentLayout,
    *,
    task: str = "pull_out_key",
    ordinal: int = 0,
    initial_seed: Optional[int] = None,
    exogenous_seed: Optional[int] = None,
    protocol_id: str = "diagnostic_v1",
    master_seed: int = 20260823,
    policy_kind: LivePolicyKind = LivePolicyKind.ACT,
) -> tuple[Path, LiveUniVTACRunRequest]:
    if initial_seed is None:
        initial_seed = derive_clean_campaign_seed(
            protocol_id=protocol_id,
            master_seed=master_seed,
            task_id=task,
            task_ordinal=ordinal,
            role="initial",
        )
    if exogenous_seed is None:
        exogenous_seed = derive_clean_campaign_seed(
            protocol_id=protocol_id,
            master_seed=master_seed,
            task_id=task,
            task_ordinal=ordinal,
            role="exogenous",
        )
    request_path = (
        layout.requests
        / "clean-campaigns"
        / "campaign-test"
        / "trials"
        / task
        / f"{ordinal:04d}"
        / "request.json"
    )
    artifact_path = (
        layout.artifacts
        / "live-univtac"
        / "clean-campaigns"
        / "campaign-test"
        / task
        / f"{ordinal:04d}"
    )
    request = LiveUniVTACRunRequest(
        task_id=task,
        condition=Condition.CLEAN,
        policy_kind=policy_kind,
        base_system_id="clean-campaign-test-policy-v1",
        dataset_sha256=digest(f"dataset-{ordinal}"),
        checkpoint_sha256=digest("checkpoint"),
        config_sha256=digest("config"),
        base_system_manifest_sha256=None,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        max_control_cycles=3,
        max_observation_steps=5,
        execute_action_steps=1 if policy_kind is LivePolicyKind.ACT else 24,
        wall_timeout_s=5.0,
        upstream_root=layout.sources / "UniVTAC",
        runtime_dir=layout.runtime / "clean-test",
        output_dir=artifact_path,
        fault_manifest_path=None,
        rest_references_path=None,
        restoration_index=None,
        restoration_mode=None,
        matched_no_touch_system_id=None,
        matched_no_touch_artifact_path=None,
        act_device_name="cpu" if policy_kind is LivePolicyKind.ACT else None,
        simulator_device=None,
        launcher_args=production_univtac_launcher_args(),
        n0_source_commit=(
            None
            if policy_kind is LivePolicyKind.ACT
            else "c43a2160dd31c449d92b28eab52c0e2f09e4738a"
        ),
        n0_normalizer_sha256=(
            None if policy_kind is LivePolicyKind.ACT else digest("normalizer")
        ),
        n0_serve_bundle_sha256=(
            None if policy_kind is LivePolicyKind.ACT else digest("serve-bundle")
        ),
        n0_prompt_manifest_sha256=(
            None if policy_kind is LivePolicyKind.ACT else digest("prompt-manifest")
        ),
    )
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_bytes(
        canonical_json_bytes(live_univtac_request_to_dict(request))
    )
    return request_path, request


def write_artifact_and_attempt(
    layout: DeploymentLayout,
    manifest_campaign_id: str,
    campaign_manifest_sha256: str,
    request: LiveUniVTACRunRequest,
    *,
    manifest_ordinal: int,
    request_file_sha256: str,
    include_attempt: bool = True,
    return_code: int = 0,
    semantic_version: str = "1.0",
    candidate_disposition: Optional[str] = None,
    exception_code: Optional[str] = None,
    crash: bool = False,
    terminal_signal: BackendSignal = BackendSignal.RUNNING,
    capture_profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
) -> None:
    loaded = load_live_univtac_run(request)
    backend = DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        terminal_signal=terminal_signal,
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )
    backend.action_spec = loaded.trial.action_spec
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        DeterministicFakePolicy.for_trial(loaded.trial, fail_on_infer=crash),
    )
    assert request.output_dir is not None
    write_live_univtac_artifact(
        request.output_dir,
        loaded,
        evidence,
        capture_profile=capture_profile,
    )
    if not include_attempt:
        return
    log_path = (
        layout.logs
        / "clean-campaigns"
        / manifest_campaign_id
        / request.task_id
        / f"{manifest_ordinal:04d}-attempt.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"test live command\n")
    log_sha256 = hashlib.sha256(log_path.read_bytes()).hexdigest()
    result = evidence.result
    started = datetime(2026, 8, 23, tzinfo=timezone.utc)
    receipt = {
        "artifact": {
            "artifact_root_sha256": hashlib.sha256(
                (request.output_dir / "root_receipt.json").read_bytes()
            ).hexdigest(),
            "execution_status": (
                None
                if result.execution_status is None
                else result.execution_status.value
            ),
            "score_eligible": result.score_eligible,
            "score_success": result.score_success,
            "terminal_status": result.terminal_status.value,
        },
        "artifact_validation_error_type": None,
        "campaign_id": manifest_campaign_id,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "command_kind": "robotactile_live_univtac_run_v1",
        "duration_s": 1.0,
        "finished_at_utc": (started + timedelta(seconds=1)).isoformat(),
        "log_relpath": log_path.relative_to(layout.root).as_posix(),
        "log_sha256": log_sha256,
        "ordinal": manifest_ordinal,
        "request_file_sha256": request_file_sha256,
        "return_code": return_code,
        "semantic_version": semantic_version,
        "started_at_utc": started.isoformat(),
        "task_id": request.task_id,
        "trial_manifest_sha256": loaded.trial.sha256,
    }
    if semantic_version == "2.0":
        receipt.update(
            {
                "candidate_disposition": candidate_disposition,
                "exception_code": exception_code,
            }
        )
    receipt_path = (
        layout.outputs
        / "clean-campaigns"
        / manifest_campaign_id
        / "attempts"
        / request.task_id
        / f"{manifest_ordinal:04d}-attempt.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(canonical_json_bytes(receipt))


def write_exception_attempt_without_artifact(
    layout: DeploymentLayout,
    manifest_campaign_id: str,
    campaign_manifest_sha256: str,
    request: LiveUniVTACRunRequest,
    *,
    manifest_ordinal: int,
    request_file_sha256: str,
    return_code: int = 7,
    exception_code: str = "live_command_return_code_7",
) -> None:
    """Write one v2 execution-exception receipt with no live artifact."""

    loaded = load_live_univtac_run(request)
    log_path = (
        layout.logs
        / "clean-campaigns"
        / manifest_campaign_id
        / request.task_id
        / f"{manifest_ordinal:04d}-attempt.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"test live command failed before artifact publication\n")
    started = datetime(2026, 8, 23, tzinfo=timezone.utc)
    receipt = {
        "artifact": None,
        "artifact_validation_error_type": None,
        "campaign_id": manifest_campaign_id,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "candidate_disposition": "exception_replaced",
        "command_kind": "robotactile_live_univtac_run_v1",
        "duration_s": 1.0,
        "exception_code": exception_code,
        "finished_at_utc": (started + timedelta(seconds=1)).isoformat(),
        "log_relpath": log_path.relative_to(layout.root).as_posix(),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "ordinal": manifest_ordinal,
        "request_file_sha256": request_file_sha256,
        "return_code": return_code,
        "semantic_version": "2.0",
        "started_at_utc": started.isoformat(),
        "task_id": request.task_id,
        "trial_manifest_sha256": loaded.trial.sha256,
    }
    receipt_path = (
        layout.outputs
        / "clean-campaigns"
        / manifest_campaign_id
        / "attempts"
        / request.task_id
        / f"{manifest_ordinal:04d}-attempt.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(canonical_json_bytes(receipt))


__all__ = [
    "digest",
    "make_layout",
    "write_artifact_and_attempt",
    "write_clean_request",
    "write_exception_attempt_without_artifact",
]

"""Contract tests for resumable Clean campaign execution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robotactile_benchmark.clean_baseline import (
    CleanCampaignProtocol,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.source_bound_attempts import (
    source_bound_arguments_requested,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.deployment.layout import DeploymentLayout

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/live_univtac/run_clean_campaign.py"


def _module() -> ModuleType:
    name = "clean_campaign_runner"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_attempt_receipt_is_no_clobber_and_idempotent(tmp_path: Path) -> None:
    write_receipt = _module()._write_receipt
    receipt = tmp_path / "attempt.json"
    document = {"return_code": 0, "semantic_version": "1.0"}

    write_receipt(receipt, document)
    write_receipt(receipt, document)

    assert receipt.read_bytes() == canonical_json_bytes(document)
    with pytest.raises(FileExistsError):
        write_receipt(receipt, {"return_code": 1, "semantic_version": "1.0"})


def test_resume_requires_a_matching_successful_attempt_receipt(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    layout = DeploymentLayout(root)
    digest = "a" * 64
    entry = CleanCampaignTrialSpec(
        ordinal=0,
        task="pull_out_key",
        initial_seed=17,
        exogenous_seed=29,
        base_system_id="official-n0-test",
        dataset_sha256=digest,
        base_system_manifest_sha256=digest,
        checkpoint_sha256=digest,
        config_sha256=digest,
        trial_manifest_sha256=digest,
        pair_key=digest,
        run_spec_sha256=digest,
        run_content_sha256=digest,
        request_file_sha256=digest,
        request_relpath="requests/clean/0000/request.json",
        artifact_relpath="artifacts/live/0000",
        max_control_cycles=1,
        max_observation_steps=2,
        execute_action_steps=24,
    )
    artifact = {
        "artifact_root_sha256": "b" * 64,
        "execution_status": "completed",
        "score_eligible": True,
        "score_success": False,
        "terminal_status": "task_failure",
    }
    receipt = {
        "artifact": artifact,
        "artifact_validation_error_type": None,
        "campaign_id": "pilot-n0-v1",
        "campaign_manifest_sha256": "d" * 64,
        "command_kind": "robotactile_live_univtac_run_v1",
        "duration_s": 1.0,
        "finished_at_utc": "2026-08-23T00:00:01+00:00",
        "log_relpath": "logs/campaign/attempt.log",
        "log_sha256": "c" * 64,
        "ordinal": 0,
        "request_file_sha256": digest,
        "return_code": 0,
        "semantic_version": "1.0",
        "started_at_utc": "2026-08-23T00:00:00+00:00",
        "task_id": "pull_out_key",
        "trial_manifest_sha256": digest,
    }
    path = (
        root
        / "outputs/clean-campaigns/pilot-n0-v1/attempts/pull_out_key"
        / "0000-attempt.json"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(receipt))

    attempts = module._prior_attempts(
        layout=layout,
        campaign_id="pilot-n0-v1",
        campaign_manifest_sha256="d" * 64,
        entry=entry,
    )

    assert len(attempts) == 1
    assert module._has_successful_attempt(attempts, artifact) is True
    with pytest.raises(ValueError, match="fields mismatch"):
        module._prior_attempts(
            layout=layout,
            campaign_id="pilot-n0-v1",
            campaign_manifest_sha256="d" * 64,
            entry=entry,
            required_semantic_version="3.0",
        )
    changed = dict(artifact)
    changed["score_success"] = True
    assert module._has_successful_attempt(attempts, changed) is False


def test_versioned_config_is_validated_and_passed_to_live_cli(
    tmp_path: Path,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    config = root / "artifacts/models/n0_twam/configs/lift_bottle/v11.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    selected = module._optional_config(root, config)
    command = module._live_command(
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        deployment_root=root,
        request_path=root / "requests/clean/request.json",
        n0_source_root=root / "src/N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29601,
        integration_config=selected,
        action_execution_contract="univtac_stock_ee_v1",
        lifecycle_journal=root / "outputs/attempt.journal",
    )

    index = command.index("--config")
    assert command[index + 1] == str(config.absolute())
    journal_index = command.index("--lifecycle-journal")
    assert command[journal_index + 1] == str(root / "outputs/attempt.journal")
    execution_index = command.index("--action-execution-contract")
    assert command[execution_index + 1] == "univtac_stock_ee_v1"
    assert command[command.index("--capture-profile") + 1] == "paper_full_v1"
    assert "--config" not in module._live_command(
        isaac_python=root / "runtime/isaac-sim-4.5.0/python.sh",
        deployment_root=root,
        request_path=root / "requests/clean/request.json",
        n0_source_root=root / "src/N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29601,
        integration_config=None,
    )
    defaults = module._parser().parse_args(
        ["--manifest", str(root / "manifest.json"), "--task", "lift_bottle"]
    )
    assert defaults.hard_lifecycle_timeout_s is None
    assert defaults.capture_profile == "paper_full_v1"


def test_formal_protocols_require_training_60hz_and_diagnostics_are_explicit() -> None:
    resolve = _module()._required_action_execution_contract

    production = "robotactile_n0_training_60hz_ee_v1"
    assert resolve(CleanCampaignProtocol.PILOT) == production
    assert resolve(CleanCampaignProtocol.PAPER) == production
    assert resolve(CleanCampaignProtocol.DIAGNOSTIC) == production
    assert resolve(CleanCampaignProtocol.PILOT, production) == production
    assert (
        resolve(
            CleanCampaignProtocol.DIAGNOSTIC,
            "robotactile_fixed_endpoint_v1",
        )
        == "robotactile_fixed_endpoint_v1"
    )
    assert (
        resolve(
            CleanCampaignProtocol.DIAGNOSTIC,
            "univtac_stock_ee_v1",
        )
        == "univtac_stock_ee_v1"
    )
    with pytest.raises(ValueError, match="pilot_v1 requires"):
        resolve(
            CleanCampaignProtocol.PILOT,
            "robotactile_fixed_endpoint_v1",
        )
    with pytest.raises(ValueError, match="paper_v1 requires"):
        resolve(
            CleanCampaignProtocol.PAPER,
            "univtac_stock_ee_v1",
        )
    with pytest.raises(ValueError, match="unsupported N0 EE"):
        resolve(CleanCampaignProtocol.DIAGNOSTIC, "unknown")


def test_source_bound_campaign_rejects_executor_binding_mismatch() -> None:
    require = _module()._require_source_bound_execution_contract
    production = "robotactile_n0_training_60hz_ee_v1"
    context = SimpleNamespace(
        runtime_source_binding=SimpleNamespace(
            action_execution_contract=production,
            native_step_contract="fixed_physics_steps_per_action_v1",
        )
    )

    require(context, production)
    with pytest.raises(ValueError, match="action execution contract mismatch"):
        require(context, "univtac_stock_ee_v1")


@pytest.mark.parametrize("missing", ("qualification", "server", "sha256"))
def test_source_bound_campaign_flags_are_all_or_none(missing: str) -> None:
    qualification = None if missing == "qualification" else Path("qualification.json")
    server = None if missing == "server" else Path("server.json")
    digest = None if missing == "sha256" else "a" * 64

    with pytest.raises(ValueError, match="must be complete"):
        source_bound_arguments_requested(qualification, server, digest)

    assert source_bound_arguments_requested(None, None, None) is False
    assert (
        source_bound_arguments_requested(
            Path("qualification.json"), Path("server.json"), "a" * 64
        )
        is True
    )


def test_versioned_config_rejects_outside_missing_and_symlink_files(
    tmp_path: Path,
) -> None:
    optional_config = _module()._optional_config
    root = tmp_path / "deployment"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="below"):
        optional_config(root, outside)
    with pytest.raises(ValueError, match="below"):
        optional_config(root, root / "missing.json")

    config = root / "config.json"
    config.write_text("{}", encoding="utf-8")
    symlink = root / "config-link.json"
    symlink.symlink_to(config.name)
    with pytest.raises(ValueError, match="non-symlink"):
        optional_config(root, symlink)


def test_official_candidate_classifies_only_episode_outcomes_as_consumed() -> None:
    module = _module()
    task_failure = {
        "artifact_root_sha256": "a" * 64,
        "execution_status": "task_failure",
        "score_eligible": True,
        "score_success": False,
        "terminal_status": "task_failure",
    }
    crash = {
        **task_failure,
        "execution_status": "crash",
        "score_eligible": False,
        "score_success": None,
        "terminal_status": "crash",
    }

    assert module._candidate_disposition(
        return_code=0, artifact=task_failure, artifact_error=None
    ) == ("valid_outcome", None)
    assert module._candidate_disposition(
        return_code=0, artifact=crash, artifact_error=None
    ) == ("exception_replaced", "live_artifact_crash")
    initial = module._RunnerStageEvidence(
        stage="reset",
        failure_code="invalid_initial_state",
        exception_type="RunnerViolation",
    )
    assert module._candidate_disposition(
        return_code=0,
        artifact=crash,
        artifact_error=None,
        stage_evidence=initial,
    ) == ("exception_replaced", "initial_state_rejected")
    assert module._candidate_disposition(
        return_code=-15,
        artifact=task_failure,
        artifact_error=None,
        close_after_export_timeout=True,
    ) == ("exception_replaced", "close_after_export_hard_timeout")
    assert (
        module._replacement_requires_fresh_n0_server("initial_state_rejected") is False
    )
    assert module._replacement_requires_fresh_n0_server("live_artifact_crash") is True
    assert module._candidate_disposition(
        return_code=17, artifact=None, artifact_error=None
    ) == ("fatal_unattempted", "live_command_return_code_17")
    stage = module._RunnerStageEvidence(
        stage="policy_reset",
        failure_code="policy_reset_failed",
        exception_type="ConnectionRefusedError",
    )
    assert module._candidate_disposition(
        return_code=17,
        artifact=None,
        artifact_error=None,
        stage_evidence=stage,
    ) == (
        "exception_replaced",
        "runner_stage_policy_reset_policy_reset_failed",
    )
    assert module._candidate_disposition(
        return_code=0,
        artifact=None,
        artifact_error="LiveArtifactValidationError",
    ) == ("fatal_unattempted", "invalid_live_artifact")


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("app_launcher", "fatal_unattempted"),
        ("runtime_preparation", "fatal_unattempted"),
        ("reset", "exception_replaced"),
        ("infer", "exception_replaced"),
    ],
)
def test_hard_watchdog_consumes_only_reset_or_later_stage(
    stage: str, expected: str
) -> None:
    module = _module()
    evidence = module._RunnerStageEvidence(
        stage=stage,
        failure_code="hard_lifecycle_timeout",
        exception_type="TimeoutExpired",
    )

    disposition, _ = module._candidate_disposition(
        return_code=-9,
        artifact=None,
        artifact_error=None,
        stage_evidence=evidence,
    )

    assert disposition == expected


@pytest.mark.parametrize(
    ("child_stage", "expected"),
    [(None, "fatal_unattempted"), ("reset", "exception_replaced")],
)
def test_execute_entry_uses_journal_stage_for_hard_timeout_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_stage: str | None,
    expected: str,
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    layout = DeploymentLayout(root)
    digest = "a" * 64
    entry = CleanCampaignTrialSpec(
        ordinal=0,
        task="lift_bottle",
        initial_seed=1,
        exogenous_seed=1,
        base_system_id="official-n0-test",
        dataset_sha256=digest,
        base_system_manifest_sha256=digest,
        checkpoint_sha256=digest,
        config_sha256=digest,
        trial_manifest_sha256=digest,
        pair_key=digest,
        run_spec_sha256=digest,
        run_content_sha256=digest,
        request_file_sha256=digest,
        request_relpath="requests/clean/0000/request.json",
        artifact_relpath="artifacts/live/0000",
        max_control_cycles=1,
        max_observation_steps=2,
        execute_action_steps=24,
    )
    request = root / entry.request_relpath
    request.parent.mkdir(parents=True)
    request.write_text("{}\n", encoding="utf-8")
    captured: dict[str, Path] = {}

    def fake_popen(command: tuple[str, ...], **_kwargs: object) -> object:
        index = command.index("--lifecycle-journal")
        captured["journal"] = Path(command[index + 1])
        return SimpleNamespace(pid=9876)

    def fake_wait(_process: object, **_kwargs: object) -> object:
        if child_stage is not None:
            module.LifecycleStageJournal.open(captured["journal"]).record(child_stage)
        return module.HardLifecycleOutcome(-9, True, True, True)

    monkeypatch.setattr(
        module,
        "load_live_univtac_request",
        lambda _path: SimpleNamespace(wall_timeout_s=1.0),
    )
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module, "wait_with_lifecycle_watchdog", fake_wait)
    monkeypatch.setattr(
        module,
        "_inspect_artifact",
        lambda *_args: module._ArtifactInspection(None, None, None),
    )

    disposition, receipt = module._execute_entry(
        layout=layout,
        campaign_id="campaign",
        campaign_manifest_sha256="b" * 64,
        entry=entry,
        request_path=request,
        artifact_path=root / entry.artifact_relpath,
        isaac_python=root / "python",
        n0_source_root=root / "N0-TWAM",
        n0_host="127.0.0.1",
        n0_port=29601,
        integration_config=None,
        hard_lifecycle_timeout_s=2.0,
        watchdog_term_grace_s=0.1,
        semantic_version="2.0",
    )

    assert disposition == expected
    assert receipt["candidate_disposition"] == expected


def test_runner_stage_marker_is_required_for_artifactless_replacement(
    tmp_path: Path,
) -> None:
    module = _module()
    log = tmp_path / "attempt.log"
    log.write_bytes(
        b"setup noise\n"
        b"ROBOTACTILE_CLOSED_LOOP_CRASH stage=infer "
        b"exception_type=TimeoutError failure_code=infer_failed "
        b"system_exit_code=None\n"
    )

    evidence = module._runner_stage_evidence(log)

    assert evidence is not None
    assert evidence.stage == "infer"
    assert evidence.exception_type == "TimeoutError"
    assert (
        module._candidate_disposition(
            return_code=1,
            artifact=None,
            artifact_error=None,
            stage_evidence=evidence,
        )[0]
        == "exception_replaced"
    )


def test_official_preflight_refuses_to_rerun_an_orphan_command_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root = tmp_path / "deployment"
    root.mkdir()
    layout = DeploymentLayout(root)
    digest = "a" * 64
    entry = CleanCampaignTrialSpec(
        ordinal=0,
        task="lift_bottle",
        initial_seed=1_000_000,
        exogenous_seed=1_000_000,
        base_system_id="official-n0-test",
        dataset_sha256=digest,
        base_system_manifest_sha256=digest,
        checkpoint_sha256=digest,
        config_sha256=digest,
        trial_manifest_sha256=digest,
        pair_key=digest,
        run_spec_sha256=digest,
        run_content_sha256=digest,
        request_file_sha256=digest,
        request_relpath="requests/clean/0000/request.json",
        artifact_relpath="artifacts/live/0000",
        max_control_cycles=1,
        max_observation_steps=2,
        execute_action_steps=24,
    )
    request = root / entry.request_relpath
    artifact = root / entry.artifact_relpath
    orphan_log = root / "logs/clean-campaigns/test/lift_bottle/0000-orphan.log"
    monkeypatch.setattr(module, "_validate_entry", lambda *_: (request, artifact))
    monkeypatch.setattr(
        module,
        "_inspect_artifact",
        lambda *_: module._ArtifactInspection(None, None, None),
    )
    monkeypatch.setattr(module, "_prior_attempts", lambda **_: ())
    monkeypatch.setattr(module, "_candidate_logs", lambda *_: (orphan_log,))
    monkeypatch.setattr(module, "_trusted_log_evidence", lambda *_: None)

    with pytest.raises(ValueError, match="refusing rerun"):
        module._preflight_official_candidates(
            layout=layout,
            campaign_id="test",
            campaign_manifest_sha256="b" * 64,
            selected=(entry,),
            target=1,
        )


def test_official_candidate_rejects_non_crash_ineligible_artifact() -> None:
    module = _module()
    artifact = {
        "artifact_root_sha256": "a" * 64,
        "execution_status": "validator_rejected",
        "score_eligible": False,
        "score_success": None,
        "terminal_status": "validator_rejected",
    }

    with pytest.raises(ValueError, match="not a replaceable exception"):
        module._candidate_disposition(
            return_code=0, artifact=artifact, artifact_error=None
        )


def test_official_attempt_requires_one_consistent_disposition() -> None:
    module = _module()
    artifact = {
        "artifact_root_sha256": "a" * 64,
        "execution_status": "completed",
        "score_eligible": True,
        "score_success": False,
        "terminal_status": "task_failure",
    }
    valid = {
        "artifact": artifact,
        "artifact_validation_error_type": None,
        "candidate_disposition": "valid_outcome",
        "exception_code": None,
        "return_code": 0,
    }

    assert module._official_attempt_disposition((valid,), artifact) == "valid_outcome"
    invalid = dict(valid)
    invalid["candidate_disposition"] = "exception_replaced"
    invalid["exception_code"] = "not_a_crash"
    with pytest.raises(ValueError, match="non-crash"):
        module._official_attempt_disposition((invalid,), artifact)

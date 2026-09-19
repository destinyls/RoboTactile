"""ACT reset-reference extraction and persistence tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotactile_benchmark.act_fault_campaign.reset_reference import (
    ACTResetReferenceArtifactError,
    act_reset_reference_relpath,
    build_act_reset_reference_from_artifact,
    load_act_reset_reference,
    write_act_reset_reference,
)
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.contracts import EvaluationRecord, build_evaluation_record
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts_contracts import (
    LoadedLiveUniVTACArtifact,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.trials import (
    Condition,
    TerminalStatus,
    TrialManifest,
    system_manifest_hash,
)


def _trial(condition: Condition = Condition.CLEAN) -> TrialManifest:
    checkpoint = "3" * 64
    config = "4" * 64
    system_id = "official-act-reset-test"
    return TrialManifest(
        task="insert_HDMI",
        initial_seed=1001,
        exogenous_seed=2002,
        condition=condition,
        base_system_id=system_id,
        executed_system_id=system_id,
        dataset_sha256="2" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            system_id, checkpoint, config, "qpos8_next_step"
        ),
        checkpoint_sha256=checkpoint,
        config_sha256=config,
        action_spec="qpos8_next_step",
        fault_manifest_sha256=(None if condition is Condition.CLEAN else "5" * 64),
        matched_no_touch_system_id=None,
    )


def _initial_record(trial: TrialManifest) -> EvaluationRecord:
    source = make_synthetic_episode(length=10)[0]
    qpos8 = np.linspace(-0.35, 0.35, 8, dtype=np.float32)
    observation = replace(
        source.observation,
        task=trial.task,
        seed=trial.initial_seed,
        step_index=0,
        proprio=qpos8,
    )
    return build_evaluation_record(observation, source.provenance)


def _artifact(
    *,
    profile: LiveCaptureProfile = LiveCaptureProfile.PAPER_FULL,
    condition: Condition = Condition.CLEAN,
    terminal_status: TerminalStatus = TerminalStatus.SUCCESS,
    score_success: bool = True,
    validation_passed: bool = True,
    native_step: object = 240,
    include_first: bool = True,
    preview_first_index: int = 0,
    receipt_pair_key: str | None = None,
    include_reset_witness: bool = False,
) -> LoadedLiveUniVTACArtifact:
    """Construct the minimum exact outer type with typed contract-facing fakes."""

    trial = _trial(condition)
    first = _initial_record(trial)
    records = (first,) if include_first else ()
    result = SimpleNamespace(
        terminal_status=terminal_status,
        score_eligible=True,
        score_success=score_success,
        validation_passed=validation_passed,
        trial_manifest_sha256=trial.sha256,
        pair_key=trial.pair_key,
        sha256="7" * 64,
        initial_state_sha256="9" * 64,
    )
    finalization: object | None = None
    preview: object | None = None
    if profile is LiveCaptureProfile.PAPER_FULL:
        finalization = SimpleNamespace(validation=None, delivered_records=records)
    elif profile is LiveCaptureProfile.PREVIEW:
        preview = SimpleNamespace(
            selected_indices=(preview_first_index,) if include_first else (),
            delivered_records=records,
        )
    initial_diagnostics: dict[str, object] = {"native_step_id": native_step}
    if include_reset_witness:
        initial_diagnostics["task"] = {
            "reset_witness": {
                "schema": "univtac-reset-witness-v1",
                "task_id": trial.task,
                "initial_seed": trial.initial_seed,
                "exogenous_seed": trial.exogenous_seed,
                "simulator_state_sha256": "9" * 64,
                "native_step": native_step,
                "qpos8": [float(item) for item in first.observation.proprio],
            }
        }
    evidence = SimpleNamespace(
        result=result,
        finalization=finalization,
        initial_diagnostics=initial_diagnostics,
    )
    pair_key = trial.pair_key if receipt_pair_key is None else receipt_pair_key
    root_receipt = SimpleNamespace(
        capture_profile=profile,
        trial_manifest_sha256=trial.sha256,
        pair_key=pair_key,
        result_sha256=result.sha256,
    )
    artifact = object.__new__(LoadedLiveUniVTACArtifact)
    for name, value in {
        "request_identity": {},
        "run_content_sha256": "6" * 64,
        "trial": trial,
        "run_spec": None,
        "fault_manifest": None if condition is Condition.CLEAN else object(),
        "rest_references": None,
        "evidence": evidence,
        "root_receipt": root_receipt,
        "root_receipt_sha256": "8" * 64,
        "preview_trace": preview,
    }.items():
        object.__setattr__(artifact, name, value)
    return artifact


def test_build_reference_binds_clean_trial_and_source_hashes() -> None:
    artifact = _artifact()

    reference = build_act_reset_reference_from_artifact(artifact)

    assert reference.task_id == artifact.trial.task
    assert reference.initial_seed == artifact.trial.initial_seed
    assert reference.exogenous_seed == artifact.trial.exogenous_seed
    assert reference.pair_key == artifact.trial.pair_key
    assert reference.dataset_sha256 == artifact.trial.dataset_sha256
    assert reference.checkpoint_sha256 == artifact.trial.checkpoint_sha256
    assert reference.config_sha256 == artifact.trial.config_sha256
    assert reference.source_artifact_root_sha256 == "8" * 64
    assert reference.source_result_sha256 == "7" * 64
    assert reference.source_run_content_sha256 == "6" * 64
    assert reference.expected_simulator_state_sha256 == "9" * 64
    assert reference.expected_native_step == 240
    assert reference.expected_qpos8 == pytest.approx(
        np.linspace(-0.35, 0.35, 8, dtype=np.float32)
    )
    assert reference.qpos_atol == 1e-5


def test_preview_index_zero_is_a_valid_initial_frame_source() -> None:
    reference = build_act_reset_reference_from_artifact(
        _artifact(profile=LiveCaptureProfile.PREVIEW),
        qpos_atol=2e-5,
    )

    assert reference.expected_native_step == 240
    assert reference.qpos_atol == 2e-5


def test_metrics_only_is_valid_when_strict_reset_witness_is_retained() -> None:
    reference = build_act_reset_reference_from_artifact(
        _artifact(
            profile=LiveCaptureProfile.METRICS_ONLY,
            include_reset_witness=True,
        )
    )

    assert reference.expected_native_step == 240
    assert reference.expected_simulator_state_sha256 == "9" * 64


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (_artifact(condition=Condition.FAULTED), "Clean artifact"),
        (
            _artifact(terminal_status=TerminalStatus.TIMEOUT, score_success=False),
            "validated successful Clean result",
        ),
        (_artifact(score_success=False), "validated successful Clean result"),
        (_artifact(validation_passed=False), "validated successful Clean result"),
        (
            _artifact(profile=LiveCaptureProfile.METRICS_ONLY),
            "metrics_only source requires",
        ),
        (_artifact(include_first=False), "initial Clean frame is missing"),
        (
            _artifact(
                profile=LiveCaptureProfile.PREVIEW,
                preview_first_index=1,
            ),
            "selected index 0",
        ),
        (_artifact(native_step=None), "initial native step"),
        (_artifact(receipt_pair_key="9" * 64), "provenance links"),
    ],
)
def test_invalid_clean_sources_fail_closed(
    artifact: LoadedLiveUniVTACArtifact,
    message: str,
) -> None:
    with pytest.raises(ACTResetReferenceArtifactError, match=message):
        build_act_reset_reference_from_artifact(artifact)


def test_reference_json_is_atomic_canonical_and_no_clobber(tmp_path: Path) -> None:
    reference = build_act_reset_reference_from_artifact(_artifact())
    target = tmp_path / act_reset_reference_relpath(
        reference.task_id, reference.pair_key
    )

    assert write_act_reset_reference(target, reference) is True
    assert target.read_bytes() == canonical_json_bytes(reference.to_dict())
    assert load_act_reset_reference(target) == reference
    assert write_act_reset_reference(target, reference) is False

    with pytest.raises(FileExistsError, match="different ACT reset reference"):
        write_act_reset_reference(target, replace(reference, qpos_atol=2e-5))


def test_formal_qpos_tolerance_cannot_disable_reset_gate() -> None:
    with pytest.raises(ACTResetReferenceArtifactError, match="formal maximum"):
        build_act_reset_reference_from_artifact(_artifact(), qpos_atol=0.01)


def test_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    reference = build_act_reset_reference_from_artifact(_artifact())
    target = tmp_path / "noncanonical.json"
    target.write_text(json.dumps(reference.to_dict()), encoding="utf-8")

    with pytest.raises(ACTResetReferenceArtifactError, match="canonical JSON"):
        load_act_reset_reference(target)


@pytest.mark.parametrize(
    ("task", "pair_key"),
    [
        ("../insert_HDMI", "a" * 64),
        ("insert_HDMI/escape", "a" * 64),
        ("insert_HDMI", "A" * 64),
        ("insert_HDMI", "short"),
    ],
)
def test_reset_reference_relpath_rejects_unsafe_components(
    task: str,
    pair_key: str,
) -> None:
    with pytest.raises(ACTResetReferenceArtifactError):
        act_reset_reference_relpath(task, pair_key)


def test_reset_reference_relpath_is_conventional() -> None:
    pair_key = "a" * 64
    assert act_reset_reference_relpath("insert_HDMI", pair_key) == (
        f"reset_references/insert_HDMI/{pair_key}.json"
    )

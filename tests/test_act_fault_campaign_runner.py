"""Focused execution/resume tests for official ACT fault campaigns."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotactile_benchmark.act_fault_campaign import runner
from robotactile_benchmark.act_fault_campaign.contracts import (
    ACT_UNSUPPORTED_OPERATOR_IDS,
    ACTFaultCampaignCellSpec,
    ACTFaultCampaignError,
    ACTFaultCellDisposition,
)
from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.trials import Condition, TerminalStatus

PAIR = "a" * 64


def _digest(index: int) -> str:
    return f"{index + 1:064x}"


def _cells() -> tuple[ACTFaultCampaignCellSpec, ...]:
    clean = ACTFaultCampaignCellSpec(
        ordinal=0,
        task="insert_tube",
        initial_seed=7,
        exogenous_seed=7,
        pair_key=PAIR,
        condition=Condition.CLEAN,
        disposition=ACTFaultCellDisposition.LIVE_REQUEST,
        operator_id=None,
        severity_level=None,
        operator_template_seed=None,
        request_relpath=f"requests/insert_tube/{PAIR}/clean.json",
        artifact_relpath=f"artifacts/insert_tube/{PAIR}/clean",
        request_file_sha256=_digest(20),
        trial_manifest_sha256=_digest(40),
        fault_manifest_sha256=None,
        fault_manifest_relpath=None,
        unsupported_receipt_relpath=None,
        unsupported_receipt_sha256=None,
    )
    result = [clean]
    for ordinal, operator_id in enumerate(sorted(CORE_OPERATOR_IDS), start=1):
        label = f"{operator_id}/s5"
        unsupported = operator_id in ACT_UNSUPPORTED_OPERATOR_IDS
        result.append(
            ACTFaultCampaignCellSpec(
                ordinal=ordinal,
                task="insert_tube",
                initial_seed=7,
                exogenous_seed=7,
                pair_key=PAIR,
                condition=Condition.FAULTED,
                disposition=(
                    ACTFaultCellDisposition.UNSUPPORTED_CONTRACT
                    if unsupported
                    else ACTFaultCellDisposition.LIVE_REQUEST
                ),
                operator_id=operator_id,
                severity_level=5,
                operator_template_seed=ordinal,
                request_relpath=(
                    None if unsupported else f"requests/insert_tube/{PAIR}/{label}.json"
                ),
                artifact_relpath=(
                    None if unsupported else f"artifacts/insert_tube/{PAIR}/{label}"
                ),
                request_file_sha256=None if unsupported else _digest(20 + ordinal),
                trial_manifest_sha256=_digest(40 + ordinal),
                fault_manifest_sha256=_digest(60 + ordinal),
                fault_manifest_relpath=(
                    f"fault_manifests/insert_tube/{PAIR}/{label}.json"
                ),
                unsupported_receipt_relpath=(
                    f"unsupported_contracts/insert_tube/{PAIR}/{label}.json"
                    if unsupported
                    else None
                ),
                unsupported_receipt_sha256=(
                    _digest(80 + ordinal) if unsupported else None
                ),
            )
        )
    return tuple(result)


def _artifact(index: int, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        external_root_sha256=_digest(120 + index),
        root_receipt=SimpleNamespace(result_sha256=_digest(160 + index)),
        evidence=SimpleNamespace(
            result=SimpleNamespace(
                score_success=success,
                validation_passed=True,
                terminal_status=(
                    TerminalStatus.SUCCESS if success else TerminalStatus.TIMEOUT
                ),
                failure_code=None,
                sha256=_digest(160 + index),
            )
        ),
    )


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[tuple[ACTFaultCampaignCellSpec, ...], dict[int, Any]]:
    cells = _cells()
    bundle = SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(
            campaign_id="act-fault", sha256=_digest(200), cells=cells
        ),
        receipt_file_sha256=_digest(201),
    )
    artifacts: dict[int, Any] = {}
    monkeypatch.setattr(runner, "load_act_fault_campaign_bundle", lambda _root: bundle)
    paths = {
        str(cell.request_relpath): cell.ordinal
        for cell in cells
        if cell.request_relpath
    }
    monkeypatch.setattr(
        runner,
        "load_live_univtac_request",
        lambda path: SimpleNamespace(
            ordinal=paths[str(Path(path).relative_to(tmp_path))]
        ),
    )
    runtime = SimpleNamespace(
        manifest=SimpleNamespace(task_id="insert_tube"),
        artifact_root=tmp_path / "models",
        stats_sha256=_digest(202),
        encoder_sha256=_digest(203),
    )
    monkeypatch.setattr(runner, "resolve_act_runtime_artifacts", lambda _path: runtime)
    monkeypatch.setattr(
        runner,
        "_load_task_reset_reference",
        lambda *_args: SimpleNamespace(reference="test"),
    )
    monkeypatch.setattr(
        runner,
        "_load_task_reset_trajectory",
        lambda *_args: SimpleNamespace(trajectory="test"),
    )
    monkeypatch.setattr(
        runner,
        "build_act_trajectory_replay_reference",
        lambda source, trajectory: SimpleNamespace(
            reference="replay",
            source=source,
            trajectory=trajectory,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_existing",
        lambda _bundle, cell, _request, _capture: artifacts.get(cell.ordinal),
    )
    return cells, artifacts


def test_selection_is_clean_then_twelve_faults_and_never_a1_a2() -> None:
    selected = runner.select_live_task_cells(_cells(), task=None, pair_key=None)

    assert len(selected) == 13
    assert selected[0].condition is Condition.CLEAN
    assert [cell.operator_id for cell in selected[1:]] == sorted(
        CORE_OPERATOR_IDS - ACT_UNSUPPORTED_OPERATOR_IDS
    )


def test_existing_accepts_portable_parent_relative_output_path(
    tmp_path: Path,
) -> None:
    cell = _cells()[0]
    request_parent = tmp_path / "requests" / cell.task / cell.pair_key
    portable_output = request_parent / ".." / ".." / ".." / str(cell.artifact_relpath)
    bundle = SimpleNamespace(root=tmp_path)
    request = SimpleNamespace(output_dir=portable_output)

    assert (
        runner._existing(
            bundle,
            cell,
            request,
            LiveCaptureProfile.PAPER_FULL,
        )
        is None
    )


def test_fresh_task_uses_one_official_paired_call_and_keeps_false_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cells_value, artifacts = _setup(tmp_path, monkeypatch)
    calls: list[tuple[int, ...]] = []
    session_arguments: list[dict[str, object]] = []
    pre_close_receipt_exists: list[bool] = []

    def execute(requests: tuple[Any, ...], **kwargs: Any) -> SimpleNamespace:
        calls.append(tuple(request.ordinal for request in requests))
        session_arguments.append(kwargs["session_factory"].keywords)
        for request in requests:
            artifacts[request.ordinal] = _artifact(
                request.ordinal, request.ordinal == 0
            )
        result = SimpleNamespace(
            paired=SimpleNamespace(
                group_content_sha256=_digest(210),
                reset_receipt=SimpleNamespace(sha256=_digest(211)),
            ),
            artifacts=tuple(artifacts[request.ordinal] for request in requests),
        )
        kwargs["pre_close_publisher"](result)
        receipt_path = (
            tmp_path / "executions" / "insert_tube" / PAIR / "task_run_receipt.json"
        )
        pre_close_receipt_exists.append(receipt_path.is_file())
        return result

    monkeypatch.setattr(runner, "execute_official_act_paired_live_runs", execute)
    monkeypatch.setattr(
        runner,
        "_execute_missing",
        lambda *_args: pytest.fail("fresh execution used partial resume"),
    )

    receipt = runner.run_act_fault_task(
        tmp_path, integration_config=tmp_path / "act.json"
    )

    assert len(calls) == 1 and len(calls[0]) == 13
    assert pre_close_receipt_exists == [True]
    assert calls[0][0] == 0
    assert session_arguments == [
        {
            "reset_reference": SimpleNamespace(
                reference="replay",
                source=SimpleNamespace(reference="test"),
                trajectory=SimpleNamespace(trajectory="test"),
            ),
            "reset_trajectory": SimpleNamespace(trajectory="test"),
        }
    ]
    assert receipt.resume_mode == runner.FRESH_RESUME_MODE
    assert receipt.newly_executed_count == 13
    assert receipt.score_successes == (True,) + (False,) * 12


def test_partial_resume_executes_only_missing_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells, artifacts = _setup(tmp_path, monkeypatch)
    live_ordinals = [
        cell.ordinal
        for cell in runner.select_live_task_cells(cells, task=None, pair_key=None)
    ]
    for index in live_ordinals[:5]:
        artifacts[index] = _artifact(index, index == 0)
    executed: list[int] = []

    replay_references: list[Any] = []
    pre_close_receipt_exists: list[bool] = []

    def resume(requests: tuple[Any, ...], *args: Any, **kwargs: Any) -> Any:
        executed.extend(request.ordinal for request in requests)
        replay_references.append(args[2])
        for request in requests:
            artifacts[request.ordinal] = _artifact(request.ordinal, False)
        group = runner._ExecutionGroup(_digest(212), _digest(213))
        kwargs["pre_close_publisher"](group)
        receipt_path = (
            tmp_path / "executions" / "insert_tube" / PAIR / "task_run_receipt.json"
        )
        pre_close_receipt_exists.append(receipt_path.is_file())
        return group

    monkeypatch.setattr(runner, "_execute_missing", resume)
    monkeypatch.setattr(
        runner,
        "execute_official_act_paired_live_runs",
        lambda *_args, **_kwargs: pytest.fail("resume reran Clean"),
    )

    receipt = runner.run_act_fault_task(
        tmp_path, integration_config=tmp_path / "act.json"
    )

    assert executed == live_ordinals[5:]
    assert pre_close_receipt_exists == [True]
    assert 0 not in executed
    assert len(replay_references) == 1
    assert replay_references[0].reference == "replay"
    assert replay_references[0].source == SimpleNamespace(reference="test")
    assert receipt.resume_mode == runner.PARTIAL_RESUME_MODE
    assert receipt.newly_executed_count == 8
    assert receipt.reused_artifact_count == 5


def test_partial_executor_reuses_policy_and_publishes_before_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_identity = "act-identity"
    backend_config = SimpleNamespace(name="univtac")
    runtime_dir = tmp_path / "runtime"
    outputs = (tmp_path / "fault-1", tmp_path / "fault-2")
    requests = tuple(SimpleNamespace(index=index) for index in range(2))
    loaded = tuple(
        SimpleNamespace(
            request=SimpleNamespace(
                output_dir=outputs[index],
                runtime_dir=runtime_dir,
                initial_state_policy="upstream_reset",
            ),
            trial=SimpleNamespace(
                task="insert_tube",
                pair_key=PAIR,
                condition=Condition.FAULTED,
            ),
            backend_config=backend_config,
            run_spec=SimpleNamespace(name="run"),
            fault_manifest=SimpleNamespace(name=f"fault-{index}"),
            rest_references=None,
            policy_identity=policy_identity,
            content_sha256=_digest(230 + index),
        )
        for index in range(2)
    )
    by_request = {id(request): item for request, item in zip(requests, loaded)}
    by_output = {item.request.output_dir: item for item in loaded}
    monkeypatch.setattr(
        runner,
        "load_live_univtac_run",
        lambda request: by_request[id(request)],
    )

    events: list[str] = []

    class FakeSession:
        def __init__(self) -> None:
            self.witnesses: list[Any] = []
            self.closed = False

        @property
        def reset_receipt(self) -> Any:
            if not self.witnesses:
                raise runner.UniVTACPairingError(
                    "canonical_reset_missing", "not captured"
                )
            return SimpleNamespace(
                witnesses=tuple(self.witnesses),
                all_exact=True,
                sha256=_digest(240),
            )

        def new_backend(self) -> Any:
            return SimpleNamespace(close=lambda: None)

        def close(self) -> None:
            self.closed = True
            events.append("session_closed")

    session = FakeSession()
    monkeypatch.setattr(
        runner,
        "default_paired_backend_session_factory",
        lambda *_args, **_kwargs: session,
    )
    monkeypatch.setattr(
        runner,
        "build_official_act_live_binding",
        lambda *_args, **_kwargs: SimpleNamespace(load_request="shared-load"),
    )

    loader_calls: list[int] = []
    policy_close_calls: list[int] = []

    class FakePolicy:
        identity = policy_identity

        def close(self) -> None:
            policy_close_calls.append(1)

    def load_policy(*_args: Any) -> FakePolicy:
        loader_calls.append(1)
        return FakePolicy()

    monkeypatch.setattr(runner, "load_official_univtac_act_policy", load_policy)

    results: dict[Path, Any] = {}

    def run_trial(
        trial: Any,
        _run_spec: Any,
        backend: Any,
        policy: Any,
        **_kwargs: Any,
    ) -> Any:
        index = len(session.witnesses)
        state_sha256 = _digest(250 + index)
        result = SimpleNamespace(
            initial_state_sha256=state_sha256,
            sha256=_digest(260 + index),
            failure_code=None,
            score_success=True,
        )
        session.witnesses.append(
            SimpleNamespace(
                exact_match=True,
                simulator_state_sha256=state_sha256,
            )
        )
        policy.close()
        backend.close()
        results[outputs[index]] = result
        return SimpleNamespace(result=result, trial=trial)

    monkeypatch.setattr(runner, "run_closed_loop_trial_with_evidence", run_trial)
    monkeypatch.setattr(runner, "write_live_univtac_artifact", lambda *_a, **_k: None)

    def load_artifact(path: Path) -> Any:
        item = by_output[path]
        result = results[path]
        return SimpleNamespace(
            run_content_sha256=item.content_sha256,
            root_receipt=SimpleNamespace(result_sha256=result.sha256),
            capture_profile=LiveCaptureProfile.METRICS_ONLY,
        )

    monkeypatch.setattr(runner, "load_live_univtac_artifact", load_artifact)
    published: list[runner._ExecutionGroup] = []

    def publish(group: runner._ExecutionGroup) -> None:
        assert not session.closed
        events.append("published")
        published.append(group)

    group = runner._execute_missing(
        requests,
        SimpleNamespace(
            artifact_root=tmp_path / "models",
            stats_sha256=_digest(270),
            encoder_sha256=_digest(271),
        ),
        LiveCaptureProfile.METRICS_ONLY,
        SimpleNamespace(reference="reset"),
        SimpleNamespace(trajectory="reset"),
        pre_close_publisher=publish,
    )

    assert len(loader_calls) == 1
    assert len(policy_close_calls) == 1
    assert published == [group]
    assert events == ["published", "session_closed"]


def test_complete_artifacts_are_adopted_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells, artifacts = _setup(tmp_path, monkeypatch)
    live_ordinals = [
        cell.ordinal
        for cell in runner.select_live_task_cells(cells, task=None, pair_key=None)
    ]
    for index in live_ordinals:
        artifacts[index] = _artifact(index, index == 0)
    monkeypatch.setattr(
        runner,
        "execute_official_act_paired_live_runs",
        lambda *_args, **_kwargs: pytest.fail("complete task was rerun"),
    )
    monkeypatch.setattr(
        runner,
        "_execute_missing",
        lambda *_args: pytest.fail("complete task entered resume"),
    )

    receipt = runner.run_act_fault_task(
        tmp_path, integration_config=tmp_path / "act.json"
    )

    assert receipt.resume_mode == runner.COMPLETE_RESUME_MODE
    assert receipt.newly_executed_count == 0
    assert receipt.reused_artifact_count == 13


def test_existing_failed_clean_blocks_fault_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells, artifacts = _setup(tmp_path, monkeypatch)
    artifacts[0] = _artifact(0, False)
    monkeypatch.setattr(
        runner,
        "execute_official_act_paired_live_runs",
        lambda *_args, **_kwargs: pytest.fail("failed Clean started fault execution"),
    )
    monkeypatch.setattr(
        runner,
        "_execute_missing",
        lambda *_args: pytest.fail("failed Clean entered partial resume"),
    )

    with pytest.raises(
        runner.ACTCleanBaselineUnqualifiedError,
        match="refusing to execute Faulted cells",
    ) as captured:
        runner.run_act_fault_task(
            tmp_path,
            integration_config=tmp_path / "act.json",
        )

    assert captured.value.code == "clean_baseline_unqualified"
    assert captured.value.task == cells[0].task


def test_fresh_failed_clean_gate_stops_before_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells, artifacts = _setup(tmp_path, monkeypatch)

    def execute(requests: tuple[Any, ...], **kwargs: Any) -> None:
        clean = requests[0]
        artifacts[clean.ordinal] = _artifact(clean.ordinal, False)
        gate = kwargs["post_execution_gate"]
        gate(
            0,
            SimpleNamespace(
                loaded=SimpleNamespace(
                    trial=SimpleNamespace(
                        condition=Condition.CLEAN,
                        task=cells[0].task,
                        pair_key=cells[0].pair_key,
                    )
                ),
                evidence=artifacts[0].evidence,
            ),
        )
        pytest.fail("failed Clean gate returned")

    monkeypatch.setattr(runner, "execute_official_act_paired_live_runs", execute)

    with pytest.raises(runner.ACTCleanBaselineUnqualifiedError):
        runner.run_act_fault_task(
            tmp_path,
            integration_config=tmp_path / "act.json",
        )

    assert set(artifacts) == {0}


def test_campaign_gate_stops_reset_mismatch_before_next_fault() -> None:
    execution = SimpleNamespace(
        loaded=SimpleNamespace(
            trial=SimpleNamespace(
                condition=Condition.FAULTED,
                task="insert_tube",
                pair_key=PAIR,
            )
        ),
        evidence=SimpleNamespace(
            result=SimpleNamespace(failure_code="qualification_reset_not_viable")
        ),
    )

    with pytest.raises(ACTFaultCampaignError, match="before policy inference"):
        runner._campaign_execution_gate(3, execution)


def test_existing_artifact_must_match_exact_loaded_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cells()[0]
    output = tmp_path / str(cell.artifact_relpath)
    output.mkdir(parents=True)
    (output / "member.json").write_text("{}")
    request = SimpleNamespace(output_dir=output)
    artifact = SimpleNamespace(
        run_content_sha256=_digest(220),
        trial=SimpleNamespace(pair_key=PAIR, task="insert_tube"),
        capture_profile=LiveCaptureProfile.PAPER_FULL,
    )
    monkeypatch.setattr(runner, "load_live_univtac_artifact", lambda _path: artifact)
    monkeypatch.setattr(
        runner,
        "load_live_univtac_run",
        lambda _request: SimpleNamespace(
            content_sha256=_digest(220),
            trial=SimpleNamespace(pair_key=PAIR, task="other"),
        ),
    )

    with pytest.raises(ACTFaultCampaignError, match="identity mismatch"):
        runner._existing(
            SimpleNamespace(root=tmp_path),
            cell,
            request,
            LiveCaptureProfile.PAPER_FULL,
        )

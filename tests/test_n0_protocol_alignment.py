"""Executable public-result gates over already verified Clean artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from test_clean_campaign_support import (
    make_layout,
    write_artifact_and_attempt,
    write_clean_request,
)

from robotactile_benchmark.backends.univtac_contracts import (
    FIXED_NATIVE_STEP_CONTRACT,
    N0_DECIMATION,
    N0_PHYSICS_STEPS_PER_ACTION,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    SIM_HZ,
)
from robotactile_benchmark.clean_baseline import (
    CleanCampaignProtocol,
    build_clean_campaign_manifest,
    load_clean_artifact_inventory,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.cli import main
from robotactile_benchmark.closed_loop.capture import TransitionTraceEntry
from robotactile_benchmark.closed_loop.contracts import BackendSignal
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.protocol_alignment import (
    EvaluationSemantics,
    GateStatus,
    TrialProtocolDiagnostic,
    build_n0_protocol_alignment_report,
    load_n0_paper_reference,
)
from robotactile_benchmark.protocol_alignment.report import (
    _action_execution_mode,
    _cadence_status,
    _classification,
    _initial_predicate_observability,
)
from robotactile_benchmark.trials import TerminalStatus

TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
ROOT = Path(__file__).resolve().parents[1]


def _protocol_diagnostic(
    *,
    initial_success_check: bool | None,
    initial_early_stop: bool | None,
    classification: str = "timeout",
) -> TrialProtocolDiagnostic:
    return TrialProtocolDiagnostic(
        ordinal=0,
        task_id="lift_bottle",
        artifact_root_sha256="0" * 64,
        terminal_status="timeout",
        score_success=False,
        observation_count=2,
        control_cycle_count=1,
        transition_count=1,
        executed_action_count=1,
        first_signal="running",
        last_signal="running",
        classification=classification,
        immediate_terminal=False,
        initial_success_check=initial_success_check,
        initial_early_stop=initial_early_stop,
        detailed_predicate_witness=True,
        transition_contract_status=GateStatus.PASS,
        execution_contract_status=GateStatus.PASS,
        cadence_status=GateStatus.PASS,
        reset_mode="eval",
        action_execution_mode=N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        failure_stage=None,
        failure_code=None,
    )


def _complete_diagnostic(tmp_path: Path):
    layout = make_layout(tmp_path)
    requests = [
        write_clean_request(layout, task=task, policy_kind=LivePolicyKind.N0)
        for task in TASKS
    ]
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=tuple(item[0] for item in requests),
        campaign_id="campaign-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=20260823,
        bootstrap_resamples=20,
    )
    specs = {item.task: item for item in manifest.trials}
    for request_path, request in requests:
        spec = specs[request.task_id]
        write_artifact_and_attempt(
            layout,
            manifest.campaign_id,
            manifest.sha256,
            request,
            manifest_ordinal=spec.ordinal,
            request_file_sha256=spec.request_file_sha256,
        )
        assert request_path.is_file()
    return layout, manifest


def test_complete_diagnostic_is_integration_go_but_paper_no_go(
    tmp_path: Path,
) -> None:
    layout, manifest = _complete_diagnostic(tmp_path)
    inventory = load_clean_artifact_inventory(layout.root, manifest)
    reference = load_n0_paper_reference()

    report = build_n0_protocol_alignment_report(manifest, inventory, reference)

    assert report["completed_task_count"] == 8
    assert report["loaded_artifact_count"] == 8
    assert report["evaluation_semantics"] == "official_reproduction"
    assert report["go_for_integration_diagnostic"] is True
    assert report["go_for_paper_comparison"] is False
    assert report["reported_rate_difference"] is None
    blockers = set(report["blocker_codes"])
    assert "not_100_per_task" in blockers
    assert "seed_contract_mismatch" in blockers
    assert "crash_counted_as_failure" in blockers
    planning = next(
        gate for gate in report["gates"] if gate["gate_id"] == "planning_execution"
    )
    assert planning["paper_blocking"] is False
    assert planning["code"] not in blockers


def test_cli_writes_json_and_markdown_without_starting_an_episode(
    tmp_path: Path, capsys
) -> None:
    layout, manifest = _complete_diagnostic(tmp_path)
    manifest_path = layout.requests / "clean-campaigns/campaign-test/manifest.json"
    write_clean_campaign_manifest(manifest_path, manifest)

    assert (
        main(
            (
                "n0-protocol-alignment",
                "--root",
                str(layout.root),
                "--manifest",
                str(manifest_path),
            )
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    output = Path(payload["output"])
    markdown = Path(payload["markdown"])
    assert output.is_file()
    assert markdown.is_file()
    assert payload["completed_task_count"] == 8
    assert payload["evaluation_semantics"] == "official_reproduction"
    assert payload["go_for_paper_comparison"] is False
    assert "NO-GO" in markdown.read_text(encoding="utf-8")
    assert "Evaluation semantics: `official_reproduction`" in markdown.read_text(
        encoding="utf-8"
    )


def test_official_reference_is_explicit_and_legacy_loader_defaults_safely() -> None:
    reference = load_n0_paper_reference()

    assert reference.evaluation_semantics is EvaluationSemantics.OFFICIAL_REPRODUCTION
    assert (
        reference.required_action_execution
        == N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )
    assert "unpublished paper-exact implementation" in reference.claim_boundary
    legacy = reference.to_dict()
    del legacy["evaluation_semantics"]
    loaded = type(reference).from_dict(legacy)
    assert loaded.evaluation_semantics is EvaluationSemantics.OFFICIAL_REPRODUCTION


def test_initial_predicates_are_observability_only_even_when_true() -> None:
    diagnostic = _protocol_diagnostic(
        initial_success_check=True,
        initial_early_stop=True,
    )

    status, missing = _initial_predicate_observability((diagnostic,))

    assert status is GateStatus.PASS
    assert missing == ()


def test_first_action_classifications_replace_immediate_aliases() -> None:
    assert _classification(TerminalStatus.SUCCESS, (), True) == ("first_action_success")
    assert _classification(TerminalStatus.EARLY_STOP, (), True) == (
        "first_action_early_stop"
    )
    assert (
        _protocol_diagnostic(
            initial_success_check=False,
            initial_early_stop=False,
            classification="immediate_success",
        ).classification
        == "first_action_success"
    )


def test_protocol_schemas_bind_official_reproduction_fields() -> None:
    alignment_schema = json.loads(
        (ROOT / "schemas/n0_protocol_alignment.schema.json").read_text(encoding="utf-8")
    )
    reference_schema = json.loads(
        (ROOT / "schemas/n0_paper_reference.schema.json").read_text(encoding="utf-8")
    )

    for schema in (alignment_schema, reference_schema):
        assert set(schema["required"]) == set(schema["properties"])
        assert schema["properties"]["evaluation_semantics"] == {
            "const": "official_reproduction"
        }
    gate_schema = alignment_schema["$defs"]["gate"]
    assert set(gate_schema["required"]) == set(gate_schema["properties"])
    assert gate_schema["properties"]["paper_blocking"] == {"type": "boolean"}


def test_stock_ee_variable_native_steps_are_source_bound_not_unknown() -> None:
    contract = "univtac_stock_ee_variable_native_steps_v1"
    execution = "univtac_stock_ee_v1"
    transitions = tuple(
        TransitionTraceEntry(
            benchmark_step_index=index + 1,
            native_step_id=sum((1, 2, 47)[: index + 1]),
            signal=BackendSignal.RUNNING,
            diagnostics={
                "action_execution_contract": execution,
                "native_step_contract": contract,
                "physics_step_delta": delta,
            },
        )
        for index, delta in enumerate((1, 2, 47))
    )
    initial = {
        "action_execution_contract": execution,
        "native_step_contract": contract,
        "physics_steps_per_action": None,
    }

    assert _action_execution_mode(initial, transitions) == execution
    assert _cadence_status(initial, transitions).value == "pass"


def test_training_60hz_execution_requires_exact_source_bound_cadence() -> None:
    transitions = tuple(
        TransitionTraceEntry(
            benchmark_step_index=index + 1,
            native_step_id=(index + 1) * N0_PHYSICS_STEPS_PER_ACTION,
            signal=BackendSignal.RUNNING,
            diagnostics={
                "action_execution_contract": (
                    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
                ),
                "native_step_contract": FIXED_NATIVE_STEP_CONTRACT,
                "physics_step_delta": N0_PHYSICS_STEPS_PER_ACTION,
            },
        )
        for index in range(3)
    )
    initial = {
        "action_execution_contract": (N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT),
        "camera_delivery_hz": 60.0,
        "decimation": N0_DECIMATION,
        "native_step_contract": FIXED_NATIVE_STEP_CONTRACT,
        "physics_steps_per_action": N0_PHYSICS_STEPS_PER_ACTION,
        "sim_hz": SIM_HZ,
    }

    assert (
        _action_execution_mode(initial, transitions)
        == N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )
    assert _cadence_status(initial, transitions) is GateStatus.PASS

    wrong_camera = dict(initial, camera_delivery_hz=30.0)
    assert _cadence_status(wrong_camera, transitions) is GateStatus.FAIL

    missing_camera = dict(initial)
    del missing_camera["camera_delivery_hz"]
    assert _cadence_status(missing_camera, transitions) is GateStatus.UNKNOWN

    wrong_decimation = dict(initial, decimation=2)
    assert _cadence_status(wrong_decimation, transitions) is GateStatus.FAIL

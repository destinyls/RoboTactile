"""FTP-1 robustness request-group generation contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration import write_rest_reference_artifact
from robotactile_benchmark.calibration.contracts import (
    NO_CONTACT_PREDICATE_ID,
    NoContactValidationReceipt,
)
from robotactile_benchmark.constants import (
    REST_REFERENCE_OPERATOR_IDS,
    SENSOR_SLOTS,
    SEVERITY_REGISTRY_ID,
)
from robotactile_benchmark.contracts import array_sha256, canonical_hash
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.integrations.configuration import (
    configure_ftp1_policy_integration,
)
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.rest_references import (
    ReferenceSplit,
    RestReferenceBundle,
)
from robotactile_benchmark.trials import Condition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ftp1_policy/prepare_robustness_group.py"
PLAN_SCHEMA = ROOT / "schemas/ftp1_policy_robustness_plan.schema.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ftp1_robustness_group", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLANNER = _load_script()


def _write_checkpoint(root: Path, task_id: str = "insert_hole") -> Path:
    name = "FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1"
    checkpoint = root / name / "19999"
    tokenizer = checkpoint / "hpt_tokenizer"
    normalization = checkpoint / "normalization"
    domain = normalization / "UniVTAC_insert_hole"
    tokenizer.mkdir(parents=True)
    domain.mkdir(parents=True)
    for name, payload in (
        ("model.safetensors", b"model"),
        ("model_config.json", b'{"action_dim":120}\n'),
        ("train_config.json", b'{"action_joint_rep":"mix"}\n'),
        ("tactile_input_config_file.json", b'{"use_tactile":true}\n'),
    ):
        (checkpoint / name).write_bytes(payload)
    (tokenizer / "GelSightMini_image_224_224_3.safetensors").write_bytes(b"gel")
    (tokenizer / "shared_image_chunk_encoder.safetensors").write_bytes(b"shared")
    for name in (
        "action_group_frequency_stats_train.json",
        "dataset_stats.json",
        "norm_params_snapshot.json",
        "share_norm_stats_all_t0_zscore.json",
    ):
        (normalization / name).write_text(f'{{"file":"{name}"}}\n', encoding="utf-8")
    for name in (
        "contact_detection_thresholds.json",
        "independent_norm_stats_all_t0_zscore.json",
        "train_val_split.json",
    ):
        (domain / name).write_text(f'{{"file":"{name}"}}\n', encoding="utf-8")
    assert task_id == "insert_hole"
    return checkpoint


def _integration_config(deployment: Path) -> Path:
    bundle = deployment / "artifacts/models/ftp1_policy"
    checkpoint = _write_checkpoint(bundle)
    config = deployment / "artifacts/models/ftp1_policy/configs/insert_hole.json"
    configure_ftp1_policy_integration(
        bundle_root=bundle,
        checkpoint_root=checkpoint,
        task_id="insert_hole",
        manifest_path=config.with_name("insert_hole.manifest.json"),
        config_path=config,
        device="cuda:0",
    )
    return config


def _rest_reference(root: Path, task_id: str = "insert_hole") -> Path:
    config = build_univtac_backend_config(task_id, action_spec=QPOS8_ACTION_SPEC)
    payloads = {
        slot: np.full(config.tactile_rgb_shape, 11 + index, dtype=np.uint8)
        for index, slot in enumerate(SENSOR_SLOTS)
    }
    payload_hashes = {slot: array_sha256(payloads[slot]) for slot in SENSOR_SLOTS}
    source_root = "2" * 64
    clean_hashes = tuple(canonical_hash({"clean": index}) for index in range(3))
    selected = 1
    record_ids = {
        slot: canonical_hash(
            {
                "clean_record_sha256": clean_hashes[selected],
                "namespace": "robotactile.rest-reference-record.v1",
                "payload_sha256": payload_hashes[slot],
                "slot_id": slot,
                "source_index": selected,
                "source_live_artifact_root_sha256": source_root,
            }
        )
        for slot in SENSOR_SLOTS
    }
    calibration = {
        slot: config.aliases.calibration_config_sha256 for slot in SENSOR_SLOTS
    }
    validation = NoContactValidationReceipt(
        source_live_artifact_root_sha256=source_root,
        source_trial_manifest_sha256="3" * 64,
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="4" * 64,
        task=task_id,
        episode_id=f"calibration-{task_id}",
        initial_seed=31,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        phase_tracker_sha256=canonical_hash(config.phase_tracker),
        minimum_consecutive_free_records=3,
        qualified_start_index=0,
        qualified_stop_index=3,
        selected_step_index=selected,
        qualified_clean_record_sha256s=clean_hashes,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        selected_payload_sha256=payload_hashes,
    )
    references = RestReferenceBundle(
        reference_id=f"rest-{task_id}-test",
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="4" * 64,
        source_artifact_sha256=source_root,
        no_contact_predicate_id=NO_CONTACT_PREDICATE_ID,
        no_contact_validation_sha256=validation.sha256,
        no_contact_verified=True,
        qualified_record_ids=record_ids,
        calibration_sha256=calibration,
        payloads=payloads,
    )
    output = root / f"rest-{task_id}"
    write_rest_reference_artifact(output, references, validation)
    return output


def _spec(tmp_path: Path) -> object:
    deployment = (tmp_path / "deployment").absolute()
    return PLANNER.PreparationSpec(
        deployment_root=deployment,
        config_path=_integration_config(deployment),
        task_id="insert_hole",
        dataset_sha256="d" * 64,
        initial_seed=101,
        exogenous_seed=202,
        severity_profile="registered_s5",
        rest_reference_root=_rest_reference(tmp_path),
        operator_seed_master=303,
        fault_start_index=0,
        fault_stop_index=40,
        max_control_cycles=20,
        max_observation_steps=41,
        wall_timeout_s=321.5,
        device="cuda:7",
        output_root=(tmp_path / "prepared-group").absolute(),
        live_output_root=(tmp_path / "live-results").absolute(),
        paired_receipt=(tmp_path / "live-results/paired.json").absolute(),
    )


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_prepares_clean_plus_twelve_directly_runnable_requests(
    tmp_path: Path,
) -> None:
    summary = PLANNER.prepare_robustness_group(_spec(tmp_path))
    output = Path(summary["output_root"])
    receipt = _document(output / PLANNER.PLAN_RECEIPT)
    ordered = receipt["ordered_requests"]
    assert isinstance(ordered, list)

    assert summary["status"] == "created"
    assert len(ordered) == 13
    assert ordered[0]["condition"] == "clean"
    assert ordered[0]["fault_start_index"] is None
    assert [item["operator_id"] for item in ordered[1:]] == list(
        PLANNER.EXECUTABLE_OPERATORS
    )
    assert len(list((output / "fault_manifests").glob("*.json"))) == 12
    assert set(receipt["member_sha256"]) == {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != PLANNER.PLAN_RECEIPT
    }

    loaded = []
    for item in ordered:
        request = load_live_univtac_request(output / item["request_path"])
        run = load_live_univtac_run(request)
        loaded.append(run)
        assert request.max_control_cycles == 20
        assert request.max_observation_steps == 41
        assert request.wall_timeout_s == 321.5
        assert request.simulator_device == "cuda:7"
        assert request.execute_action_steps == 1
    assert loaded[0].trial.condition is Condition.CLEAN
    assert all(item.trial.pair_key == loaded[0].trial.pair_key for item in loaded)
    assert all(item.trial.condition is Condition.FAULTED for item in loaded[1:])

    for item in loaded[1:]:
        fault = item.fault_manifest
        assert isinstance(fault, FaultManifest)
        assert fault.severity_level == 5
        assert fault.severity_registry == SEVERITY_REGISTRY_ID
        expected_start = 16 if fault.operator_id == "T1_fixed_source_delay" else 0
        assert fault.start_index == expected_start
        assert fault.stop_index == 40
        if fault.operator_id in REST_REFERENCE_OPERATOR_IDS:
            assert item.rest_references is not None
        else:
            assert item.rest_references is None

    fault_window = receipt["fault_window"]
    assert fault_window["start_index"] == 0
    assert fault_window["stop_index"] == 40
    assert fault_window["operator_start_indices"] == {
        operator_id: (16 if operator_id == "T1_fixed_source_delay" else 0)
        for operator_id in PLANNER.EXECUTABLE_OPERATORS
    }
    assert [item["fault_start_index"] for item in ordered[1:]] == [
        fault_window["operator_start_indices"][operator_id]
        for operator_id in PLANNER.EXECUTABLE_OPERATORS
    ]


def test_a1_a2_are_explicit_na_without_requests_or_substitution(
    tmp_path: Path,
) -> None:
    summary = PLANNER.prepare_robustness_group(_spec(tmp_path))
    output = Path(summary["output_root"])
    receipt = _document(output / PLANNER.PLAN_RECEIPT)
    unsupported = receipt["unsupported_contracts"]
    assert isinstance(unsupported, list)

    assert {item["operator_id"] for item in unsupported} == {
        "A1_stream_absence",
        "A2_frame_erasure",
    }
    for item in unsupported:
        assert item["status"] == "unsupported_contract"
        assert item["applicability"] == "N/A"
        assert item["request_generated"] is False
        assert item["fault_manifest_generated"] is False
        assert item["black_frame_used"] is False
        assert item["substituted_payload"] is False
        standalone = _document(output / item["path"])
        assert "path" not in standalone and "file_sha256" not in standalone
    generated_names = {path.name for path in (output / "requests").glob("*.json")}
    generated_names |= {
        path.name for path in (output / "fault_manifests").glob("*.json")
    }
    assert not any("A1_stream_absence" in name for name in generated_names)
    assert not any("A2_frame_erasure" in name for name in generated_names)


def test_plan_is_schema_synchronized_idempotent_and_no_clobber(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    first = PLANNER.prepare_robustness_group(spec)
    repeated = PLANNER.prepare_robustness_group(spec)
    output = Path(first["output_root"])
    receipt = _document(output / PLANNER.PLAN_RECEIPT)
    schema = _document(PLAN_SCHEMA)

    assert repeated["status"] == "already_present"
    assert first["receipt_file_sha256"] == repeated["receipt_file_sha256"]
    assert set(receipt) == set(schema["required"]) == set(schema["properties"])
    fault_window_schema = schema["$defs"]["faultWindow"]
    assert set(fault_window_schema["required"]) == set(
        fault_window_schema["properties"]
    )
    operator_start_schema = fault_window_schema["properties"]["operator_start_indices"]
    assert (
        set(operator_start_schema["required"])
        == set(PLANNER.EXECUTABLE_OPERATORS)
        == set(operator_start_schema["properties"])
    )
    paired = receipt["paired_run"]
    assert paired["subcommand"] == "live-univtac-paired-run"
    assert paired["request_paths"] == [
        item["request_path"] for item in receipt["ordered_requests"]
    ]
    assert first["ordered_request_paths"] == [
        str(output / path) for path in paired["request_paths"]
    ]

    target = output / paired["request_paths"][0]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        PLANNER.prepare_robustness_group(spec)


def test_cli_accepts_severity_alias_and_emits_direct_request_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deployment = (tmp_path / "deployment").absolute()
    output = (tmp_path / "cli-plan").absolute()
    live = (tmp_path / "cli-live").absolute()
    status = PLANNER.main(
        [
            "--root",
            str(deployment),
            "--config",
            str(_integration_config(deployment)),
            "--task",
            "insert_hole",
            "--dataset-sha256",
            "e" * 64,
            "--initial-seed",
            "7",
            "--exogenous-seed",
            "9",
            "--severity-profile",
            "s3",
            "--rest-reference",
            str(_rest_reference(tmp_path)),
            "--fault-start-index",
            "4",
            "--fault-stop-index",
            "20",
            "--max-control-cycles",
            "20",
            "--max-observation-steps",
            "41",
            "--output-root",
            str(output),
            "--live-output-root",
            str(live),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    receipt = _document(output / PLANNER.PLAN_RECEIPT)

    assert status == 0
    assert len(summary["ordered_request_paths"]) == 13
    assert receipt["severity_profile"]["profile_id"] == "registered_s3"
    assert receipt["severity_profile"]["severity_level"] == 3


def test_rejects_wrong_task_rest_reference_and_empty_operator_window(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    wrong_rest = _rest_reference(tmp_path / "other", "lift_can")
    with pytest.raises(ValueError, match="rest-reference task"):
        PLANNER.prepare_robustness_group(replace(spec, rest_reference_root=wrong_rest))
    with pytest.raises(
        ValueError,
        match=(
            "operator fault start must be below fault_stop_index.*"
            "T1_fixed_source_delay=16"
        ),
    ):
        PLANNER.prepare_robustness_group(
            replace(
                spec,
                fault_stop_index=16,
                output_root=(tmp_path / "empty-t1-window").absolute(),
            )
        )

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import robotactile_benchmark.integrations.act.requests as act_requests
from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.fixtures import make_synthetic_rest_references
from robotactile_benchmark.integrations.act import (
    ACTArtifactManifest,
    build_official_act_request,
    write_official_act_request,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.rest_references import FrozenPayload
from robotactile_benchmark.trials import Condition, system_manifest_hash


def _manifest(
    root: Path,
    profile: OfficialACTProfile,
    *,
    task_id: str = "pull_out_key",
) -> ACTArtifactManifest:
    marker = "1" if profile is OfficialACTProfile.UNIVTAC else "2"
    return ACTArtifactManifest.for_shared_root(
        task_id=task_id,
        profile=profile,
        artifact_root=root / "artifacts/models/act",
        upstream_root=root / "sources/UniVTAC",
        checkpoint_sha256=marker * 64,
        stats_sha256=("3" if marker == "1" else "4") * 64,
        encoder_sha256="5" * 64,
    )


def _arguments(root: Path) -> dict[str, object]:
    return {
        "layout": DeploymentLayout(root / "deployment"),
        "dataset_sha256": "a" * 64,
        "initial_seed": 17,
        "exogenous_seed": 29,
        "max_control_cycles": 2,
        "max_observation_steps": 3,
        "wall_timeout_s": 15.0,
        "act_device_name": "cuda:2",
        "simulator_device": "cuda:3",
        "live_output_dir": root / "outputs/live-act",
    }


def _write_rest_references(path: Path) -> str:
    references = make_synthetic_rest_references()
    payloads: dict[str, object] = {}
    for slot, value in references.payloads.items():
        assert isinstance(value, FrozenPayload)
        payloads[slot] = {
            "dtype": value.dtype,
            "shape": list(value.shape),
            "data_hex": value.data_hex,
        }
    path.write_bytes(
        canonical_json_bytes(
            {
                "reference_id": references.reference_id,
                "dataset_split": references.dataset_split.value,
                "split_manifest_sha256": references.split_manifest_sha256,
                "source_artifact_sha256": references.source_artifact_sha256,
                "no_contact_predicate_id": references.no_contact_predicate_id,
                "no_contact_validation_sha256": (
                    references.no_contact_validation_sha256
                ),
                "no_contact_verified": references.no_contact_verified,
                "qualified_record_ids": dict(references.qualified_record_ids),
                "calibration_sha256": dict(references.calibration_sha256),
                "payloads": payloads,
            }
        )
    )
    return references.sha256


@pytest.fixture(autouse=True)
def _accept_typed_test_manifests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        act_requests,
        "validate_official_univtac_act_artifact",
        lambda manifest: {"profile": manifest.profile.value},
    )


def test_clean_and_no_touch_requests_bind_a_matched_manifest_pair(
    tmp_path: Path,
) -> None:
    tactile = _manifest(tmp_path, OfficialACTProfile.UNIVTAC)
    vision = _manifest(tmp_path, OfficialACTProfile.VISION_ONLY)
    arguments = _arguments(tmp_path)
    clean = build_official_act_request(
        base_manifest=tactile,
        condition=Condition.CLEAN,
        **arguments,
    )
    no_touch = build_official_act_request(
        base_manifest=tactile,
        execution_manifest=vision,
        condition=Condition.NO_TOUCH,
        **arguments,
    )

    expected_base_hash = system_manifest_hash(
        clean.base_system_id,
        tactile.checkpoint_sha256,
        tactile.config_sha256,
        QPOS8_ACTION_SPEC,
    )
    assert clean.checkpoint_sha256 == tactile.checkpoint_sha256
    assert clean.config_sha256 == tactile.config_sha256
    assert clean.base_system_manifest_sha256 == expected_base_hash
    assert clean.execute_action_steps == 1
    assert clean.act_device_name == "cuda:2"
    assert clean.simulator_device == "cuda:3"
    assert no_touch.checkpoint_sha256 == vision.checkpoint_sha256
    assert no_touch.config_sha256 == vision.config_sha256
    assert no_touch.base_system_id == clean.base_system_id
    assert no_touch.base_system_manifest_sha256 == expected_base_hash
    assert no_touch.matched_no_touch_system_id == (
        "official-univtac-act.pull_out_key.vision_only.policy_last.v1"
    )
    assert no_touch.matched_no_touch_artifact_path == vision.checkpoint_path
    assert load_live_univtac_run(clean).trial.pair_key == (
        load_live_univtac_run(no_touch).trial.pair_key
    )


def test_faulted_request_loads_fault_and_rest_references(tmp_path: Path) -> None:
    tactile = _manifest(tmp_path, OfficialACTProfile.UNIVTAC)
    rest_path = tmp_path / "rest_reference.json"
    rest_sha256 = _write_rest_references(rest_path)
    fault = FaultManifest(
        operator_id="F4_local_nonresponsive_patch",
        severity_level=3,
        operator_seed=101,
        start_index=0,
        stop_index=2,
        sensor_slots=("left",),
        observability=Observability.DECLARED,
        parameters={"rest_reference_sha256": rest_sha256},
    )
    fault_path = tmp_path / "fault.json"
    fault_path.write_bytes(canonical_json_bytes(fault.to_dict()))

    request = build_official_act_request(
        base_manifest=tactile,
        condition=Condition.FAULTED,
        fault_manifest_path=fault_path,
        rest_references_path=rest_path,
        **_arguments(tmp_path),
    )
    loaded = load_live_univtac_run(request)

    assert request.checkpoint_sha256 == tactile.checkpoint_sha256
    assert request.fault_manifest_path == fault_path.absolute()
    assert request.rest_references_path == rest_path.absolute()
    assert loaded.fault_manifest == fault
    assert loaded.rest_references is not None
    assert loaded.rest_references.sha256 == rest_sha256


def test_no_touch_requires_a_matching_vision_only_manifest(tmp_path: Path) -> None:
    tactile = _manifest(tmp_path, OfficialACTProfile.UNIVTAC)
    with pytest.raises(ValueError, match="vision_only execution"):
        build_official_act_request(
            base_manifest=tactile,
            condition=Condition.NO_TOUCH,
            **_arguments(tmp_path),
        )
    with pytest.raises(ValueError, match="vision_only"):
        build_official_act_request(
            base_manifest=tactile,
            execution_manifest=tactile,
            condition=Condition.NO_TOUCH,
            **_arguments(tmp_path),
        )
    other_task = _manifest(
        tmp_path,
        OfficialACTProfile.VISION_ONLY,
        task_id="lift_can",
    )
    with pytest.raises(ValueError, match="task mismatch"):
        build_official_act_request(
            base_manifest=tactile,
            execution_manifest=other_task,
            condition=Condition.NO_TOUCH,
            **_arguments(tmp_path),
        )


def test_canonical_writer_is_idempotent_and_no_clobber(tmp_path: Path) -> None:
    request = build_official_act_request(
        base_manifest=_manifest(tmp_path, OfficialACTProfile.UNIVTAC),
        condition=Condition.CLEAN,
        **_arguments(tmp_path),
    )
    target = tmp_path / "requests/act/pull_out_key/clean.json"
    first = write_official_act_request(target, request)
    second = write_official_act_request(target, request)

    assert first == second
    assert load_live_univtac_request(target) == request
    assert first.trial_manifest_sha256 == load_live_univtac_run(request).trial.sha256
    with pytest.raises(FileExistsError, match="different"):
        write_official_act_request(target, replace(request, initial_seed=18))

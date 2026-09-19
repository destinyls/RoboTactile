from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

import robotactile_benchmark.integrations.act.requests as act_requests
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.fixtures import make_synthetic_rest_references
from robotactile_benchmark.integrations.act import ACTArtifactManifest
from robotactile_benchmark.integrations.runtime_config import ACTRuntimeArtifacts
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.rest_references import FrozenPayload
from robotactile_benchmark.trials import Condition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/act/prepare_requests.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("act_prepare_requests", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLANNER = _load_script()


def _manifest(root: Path, profile: OfficialACTProfile) -> ACTArtifactManifest:
    marker = "1" if profile is OfficialACTProfile.UNIVTAC else "2"
    return ACTArtifactManifest.for_shared_root(
        task_id="pull_out_key",
        profile=profile,
        artifact_root=root / "artifacts/models/act",
        upstream_root=root / "sources/UniVTAC",
        checkpoint_sha256=marker * 64,
        stats_sha256=("3" if marker == "1" else "4") * 64,
        encoder_sha256="5" * 64,
    )


def _runtime(
    root: Path,
    profile: OfficialACTProfile,
    *,
    device: str,
) -> ACTRuntimeArtifacts:
    return ACTRuntimeArtifacts(
        manifest=_manifest(root, profile),
        artifact_root=root / "artifacts/models/act",
        stats_sha256=("3" if profile is OfficialACTProfile.UNIVTAC else "4") * 64,
        encoder_sha256="5" * 64,
        manifest_path=(
            root
            / "artifacts/models/act/configs/pull_out_key"
            / profile.value
            / "artifact_manifest.json"
        ),
        device=device,
    )


def _patch_runtimes(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> list[Path]:
    calls: list[Path] = []
    tactile = _runtime(root, OfficialACTProfile.UNIVTAC, device="cuda:2")
    vision = _runtime(root, OfficialACTProfile.VISION_ONLY, device="cuda:4")

    def resolve(path: Path) -> ACTRuntimeArtifacts:
        calls.append(path)
        return vision if "vision_only" in path.parts else tactile

    monkeypatch.setattr(PLANNER, "resolve_act_runtime_artifacts", resolve)
    monkeypatch.setattr(
        act_requests,
        "validate_official_univtac_act_artifact",
        lambda manifest: {"profile": manifest.profile.value},
    )
    return calls


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


def _spec(root: Path, fault_path: Path, rest_path: Path) -> object:
    return PLANNER.PreparationSpec(
        deployment_root=root,
        task_id="pull_out_key",
        dataset_sha256="a" * 64,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=2,
        max_observation_steps=3,
        wall_timeout_s=15.0,
        simulator_device="cuda:7",
        include_no_touch=True,
        fault_manifest_path=fault_path,
        rest_references_path=rest_path,
    )


def test_prepares_three_loadable_requests_from_profile_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "deployment").absolute()
    calls = _patch_runtimes(monkeypatch, root)
    rest_path = tmp_path / "rest.json"
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
    spec = _spec(root, fault_path, rest_path)

    first = PLANNER.prepare_requests(spec)
    second = PLANNER.prepare_requests(spec)
    request_root = root / "requests/act/pull_out_key"
    requests = {
        condition: load_live_univtac_request(request_root / f"{condition.value}.json")
        for condition in Condition
    }
    loaded = {
        condition: load_live_univtac_run(request)
        for condition, request in requests.items()
    }

    assert first == second
    assert first["request_count"] == 3
    assert first["live_execution_claimed"] is False
    assert {item.trial.pair_key for item in loaded.values()} == {
        loaded[Condition.CLEAN].trial.pair_key
    }
    assert requests[Condition.CLEAN].act_device_name == "cuda:2"
    assert requests[Condition.FAULTED].act_device_name == "cuda:2"
    assert requests[Condition.NO_TOUCH].act_device_name == "cuda:4"
    assert {item.simulator_device for item in requests.values()} == {"cuda:7"}
    assert loaded[Condition.FAULTED].fault_manifest == fault
    assert loaded[Condition.FAULTED].rest_references is not None
    assert loaded[Condition.FAULTED].rest_references.sha256 == rest_sha256
    assert {
        item.backend_config.physics_steps_per_action for item in loaded.values()
    } == {1}
    assert {
        item.backend_config.action_execution_contract for item in loaded.values()
    } == {"univtac_stock_qpos_v1"}
    assert requests[Condition.NO_TOUCH].matched_no_touch_artifact_path == (
        _manifest(root, OfficialACTProfile.VISION_ONLY).checkpoint_path
    )
    assert (
        calls
        == [
            root
            / "artifacts/models/act/configs/pull_out_key/univtac/integration_config.json",
            root
            / "artifacts/models/act/configs/pull_out_key/vision_only/integration_config.json",
        ]
        * 2
    )
    summary_path = request_root / "request_generation_summary.json"
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path.read_bytes() == canonical_json_bytes(document)
    assert document == first


def test_cli_prints_one_machine_readable_clean_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = (tmp_path / "deployment").absolute()
    _patch_runtimes(monkeypatch, root)

    exit_code = PLANNER.main(
        [
            "--root",
            str(root),
            "--task",
            "pull_out_key",
            "--dataset-sha256",
            "a" * 64,
            "--initial-seed",
            "17",
            "--exogenous-seed",
            "29",
            "--max-control-cycles",
            "2",
            "--max-observation-steps",
            "3",
        ]
    )
    output = capsys.readouterr().out
    summary = json.loads(output)

    assert exit_code == 0
    assert output == json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    assert summary["request_count"] == 1
    assert [item["condition"] for item in summary["requests"]] == ["clean"]


def test_rejects_or_refuses_ambiguous_request_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "deployment").absolute()
    _patch_runtimes(monkeypatch, root)
    base = PLANNER.PreparationSpec(
        deployment_root=root,
        task_id="pull_out_key",
        dataset_sha256="a" * 64,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=2,
        max_observation_steps=3,
        wall_timeout_s=15.0,
        simulator_device="cuda:0",
    )

    with pytest.raises(ValueError, match="requires --fault-manifest"):
        PLANNER.prepare_requests(
            replace(base, rest_references_path=tmp_path / "rest.json")
        )
    PLANNER.prepare_requests(base)
    before = {
        path: path.read_bytes()
        for path in (root / "requests/act/pull_out_key").glob("*.json")
    }
    with pytest.raises(FileExistsError, match="different ACT request"):
        PLANNER.prepare_requests(replace(base, exogenous_seed=30))
    assert before == {path: path.read_bytes() for path in before}

"""Recovery planning tests mock only the strict artifact-reader boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_retrained_group_options import binding

from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from scripts.retrained_evaluation import availability_recovery as recovery
from scripts.retrained_evaluation.group import prepare_group, write_json


def _source(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path, dict[Path, SimpleNamespace]]:
    value = binding(root)
    value["evaluation"] = {
        "tactile_availability_mode": "native_missing_v1",
        "operators": ["A1_stream_absence", "A2_frame_erasure"],
        "severity_registries": ["optical_marker_extreme_v1"],
        "fault_start_index": 0,
        "fault_window_mode": "full_episode_v1",
    }
    source = root / "source"
    plan = json.loads(prepare_group(value, "lift_can", source, 0).read_text())
    artifacts: dict[Path, SimpleNamespace] = {}
    for index, name in enumerate(plan["ordered_requests"][:2]):
        loaded = load_live_univtac_run(load_live_univtac_request(source / name))
        artifact_path = loaded.request.output_dir
        assert artifact_path is not None
        result = SimpleNamespace(score_eligible=True, validation_passed=True)
        artifacts[artifact_path] = SimpleNamespace(
            run_content_sha256=loaded.content_sha256,
            root_receipt_sha256=str(index + 1) * 64,
            evidence=SimpleNamespace(result=result),
        )
        write_json(
            source / "results" / f"{index:02d}.json",
            {
                "artifact": str(artifact_path),
                "score_eligible": True,
                "validation_passed": True,
                "root_receipt_sha256": str(index + 1) * 64,
            },
        )
    clean = load_live_univtac_run(
        load_live_univtac_request(source / plan["ordered_requests"][0])
    )
    reference = SimpleNamespace(
        pair_key=clean.trial.pair_key,
        to_dict=lambda: {"pair_key": clean.trial.pair_key},
    )
    monkeypatch.setattr(recovery, "load_live_univtac_artifact", artifacts.__getitem__)
    monkeypatch.setattr(recovery, "result_to_dict", vars)
    monkeypatch.setattr(
        recovery, "build_act_reset_reference_from_artifact", lambda _: reference
    )
    return value, source, artifacts


def test_recovery_adopts_controls_and_executes_only_a2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, source, _ = _source(tmp_path, monkeypatch)
    original = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "recovery"
    plan = json.loads(
        recovery.prepare_recovery_group(
            value, "lift_can", output, 0, source
        ).read_text()
    )
    assert plan["execute_indices"] == [2]
    assert plan["recovery_protocol"] == "historical_clean_reference_v1"
    assert (
        plan["evidence_scope"]
        == "cross_process_reference_recovery_not_same_process_pairing"
    )
    old_request = load_live_univtac_request(source / plan["ordered_requests"][2])
    new_request = load_live_univtac_request(output / plan["ordered_requests"][2])
    old_run, new_run = (
        load_live_univtac_run(old_request),
        load_live_univtac_run(new_request),
    )
    assert new_run.trial.pair_key == old_run.trial.pair_key
    assert new_run.fault_manifest is not None and old_run.fault_manifest is not None
    assert new_run.fault_manifest.parameters["a2_end_policy"] == "episode_censored_v1"
    assert (
        new_run.fault_manifest.parameters["erased_offsets"]
        == old_run.fault_manifest.parameters["erased_offsets"]
    )
    assert new_request.output_dir == output / "artifacts/A2_frame_erasure"
    recovery.adopt_control_rows(output, plan)
    assert sorted(p.name for p in (output / "results").iterdir()) == [
        "00.json",
        "01.json",
    ]
    assert (
        json.loads((output / "results/00.json").read_text())["adoption"][
            "executed_again"
        ]
        is False
    )
    assert all(path.read_bytes() == content for path, content in original.items())
    with pytest.raises(FileExistsError):
        recovery.adopt_control_rows(output, plan)
    with pytest.raises(FileExistsError):
        recovery.prepare_recovery_group(value, "lift_can", output, 0, source)


@pytest.mark.parametrize("existing", ["result", "artifact"])
def test_existing_a2_prevents_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    value, source, _ = _source(tmp_path, monkeypatch)
    if existing == "result":
        write_json(source / "results/02.json", {})
    else:
        plan = json.loads((source / "group.json").read_text())
        request = load_live_univtac_request(source / plan["ordered_requests"][2])
        assert request.output_dir is not None
        request.output_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="A2 already"):
        recovery.prepare_recovery_group(
            value, "lift_can", tmp_path / "recovery", 0, source
        )
    assert not (tmp_path / "recovery").exists()


def test_preparation_rejects_artifact_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, source, artifacts = _source(tmp_path, monkeypatch)
    next(iter(artifacts.values())).root_receipt_sha256 = "f" * 64
    with pytest.raises(ValueError, match="mismatch"):
        recovery.prepare_recovery_group(
            value, "lift_can", tmp_path / "recovery", 0, source
        )
    assert not (tmp_path / "recovery").exists()


@pytest.mark.parametrize("changed", ["result", "artifact"])
def test_adoption_rechecks_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    value, source, artifacts = _source(tmp_path, monkeypatch)
    output = tmp_path / "recovery"
    plan = json.loads(
        recovery.prepare_recovery_group(
            value, "lift_can", output, 0, source
        ).read_text()
    )
    if changed == "result":
        path = source / "results/00.json"
        path.write_text(path.read_text() + "\n")
    else:
        next(iter(artifacts.values())).root_receipt_sha256 = "f" * 64
    with pytest.raises(ValueError, match="changed since"):
        recovery.adopt_control_rows(output, plan)
    assert not (output / "results").exists()

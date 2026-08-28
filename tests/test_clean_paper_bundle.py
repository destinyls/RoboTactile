from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_clean_campaign_support import (
    make_layout,
    write_artifact_and_attempt,
    write_clean_request,
)
from test_n0_observation_parity_artifact import source_binding
from test_source_bound_qualification import (
    _qualification as _source_bound_qualification,
)

from robotactile_benchmark.clean_baseline import (
    CleanCampaignProtocol,
    CleanCampaignSamplingSpec,
    build_clean_baseline_summary,
    build_clean_campaign_manifest,
    build_paper_result_bundle,
    load_clean_artifact_inventory,
    verify_all_task_qualification,
    write_clean_baseline_summary,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.clean_baseline.aggregation import CleanArtifactInventory
from robotactile_benchmark.clean_baseline.io import write_canonical_no_clobber
from robotactile_benchmark.clean_baseline.qualification import (
    VerifiedAllTaskQualification,
)
from robotactile_benchmark.clean_baseline.seeds import univtac_task_seed_start
from robotactile_benchmark.cli import main
from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.execution.contracts import LivePolicyKind

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification(root: Path, action_spec: str = "qpos8_next_step") -> Path:
    inventory = []
    for task_index, task_id in enumerate(TASKS):
        task_receipts: dict[str, list[object]] = {
            "reset_paths": [],
            "reset_hashes": [],
            "pairing_paths": [],
            "pairing_hashes": [],
        }
        import_path = root / f"artifacts/deployment/import-{task_id}.json"
        write_canonical_no_clobber(import_path, {"status": "qualified"})
        for kind in ("reset", "pairing"):
            for repeat in range(1):
                path = root / f"artifacts/deployment/{kind}-{task_id}-{repeat}.json"
                write_canonical_no_clobber(
                    path, {"repeat": repeat, "status": "qualified"}
                )
                cast_paths = task_receipts[f"{kind}_paths"]
                cast_hashes = task_receipts[f"{kind}_hashes"]
                assert isinstance(cast_paths, list) and isinstance(cast_hashes, list)
                cast_paths.append(path.relative_to(root).as_posix())
                cast_hashes.append(_sha256(path))
        inventory.append(
            {
                "exogenous_seed": 200 + task_index,
                "import_receipt_relpath": import_path.relative_to(root).as_posix(),
                "import_receipt_sha256": _sha256(import_path),
                "initial_seed": 100 + task_index,
                "pairing_receipt_relpaths": task_receipts["pairing_paths"],
                "pairing_receipt_sha256s": task_receipts["pairing_hashes"],
                "reset_receipt_relpaths": task_receipts["reset_paths"],
                "reset_receipt_sha256s": task_receipts["reset_hashes"],
                "task_id": task_id,
            }
        )
    path = root / "artifacts/deployment/all-task-qualification.json"
    document = {
        "action_spec": action_spec,
        "campaign_id": "qualification-test",
        "closed_loop_episode_executed": False,
        "evidence_level": "all_tasks_reset_action_contract_qualification_v1",
        "master_seed": 20260823,
        "policy_loaded": False,
        "repetitions": 1,
        "recorded_at_utc": "2026-08-23T00:00:00+00:00",
        "seed_derivation": "sha256_signed31_v1",
        "semantic_version": "1.0",
        "success_rate_claimed": False,
        "tasks": inventory,
    }
    path.write_bytes(canonical_json_bytes(document))
    return path


def test_qualification_verifies_every_bound_receipt(tmp_path: Path) -> None:
    layout = make_layout(tmp_path)
    path = _qualification(layout.root)
    verified = verify_all_task_qualification(layout.root, path)
    assert verified.tasks == TASKS
    assert verified.repetitions == 1
    assert verified.semantic_version == "1.0"
    assert verified.source_bound is False
    assert verified.source_binding is None
    bound = layout.root / "artifacts/deployment/reset-grasp_classify-0.json"
    bound.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_all_task_qualification(layout.root, path)


def test_publish_cli_writes_explicit_nonclaimable_development_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_layout(tmp_path)
    request_path, request = write_clean_request(layout)
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=(request_path,),
        campaign_id="campaign-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=20260823,
        confidence_level=0.95,
        bootstrap_resamples=20,
        bootstrap_seed=3,
    )
    manifest_path = layout.requests / "clean-campaigns/campaign-test/manifest.json"
    write_clean_campaign_manifest(manifest_path, manifest)
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=0,
        request_file_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
    )
    inventory = load_clean_artifact_inventory(layout.root, manifest)
    summary = build_clean_baseline_summary(manifest, inventory)
    summary_path = layout.outputs / "clean-campaigns/campaign-test/summary.json"
    write_clean_baseline_summary(summary_path, summary)
    qualification_path = _qualification(layout.root)
    assert (
        main(
            (
                "clean-campaign-publish",
                "--root",
                str(layout.root),
                "--summary",
                str(summary_path),
                "--qualification",
                str(qualification_path),
            )
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["paper_claim_eligible"] is False
    assert "protocol_not_paper" in result["paper_claim_blockers"]
    assert "simulator_not_qualified" in result["paper_claim_blockers"]
    assert "qualification_not_source_bound" in result["paper_claim_blockers"]
    assert Path(result["paper_result"]).is_file()
    assert Path(result["csv"]).read_text(encoding="utf-8").startswith("task,")
    assert "\\begin{tabular}" in Path(result["latex"]).read_text(encoding="utf-8")


def test_source_bound_qualification_remains_blocked_without_clean_runtime_binding(
    tmp_path: Path,
) -> None:
    layout = make_layout(tmp_path)
    request_path, request = write_clean_request(layout)
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=(request_path,),
        campaign_id="campaign-source-bound-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=20260823,
        confidence_level=0.95,
        bootstrap_resamples=20,
        bootstrap_seed=3,
    )
    manifest_path = layout.requests / "clean-campaigns/source-bound/manifest.json"
    write_clean_campaign_manifest(manifest_path, manifest)
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=0,
        request_file_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
    )
    summary = build_clean_baseline_summary(
        manifest, load_clean_artifact_inventory(layout.root, manifest)
    )
    summary_path = layout.outputs / "clean-campaigns/source-bound/summary.json"
    write_clean_baseline_summary(summary_path, summary)
    qualification_path, _ = _source_bound_qualification(layout.root)
    bundle = build_paper_result_bundle(
        deployment_root=layout.root,
        summary_path=summary_path,
        qualification_path=qualification_path,
        output_directory=layout.outputs / "clean-campaigns/source-bound/paper",
    )
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))
    assert bundle.paper_claim_eligible is False
    assert "qualification_runtime_binding_unbound" in bundle.blockers
    assert receipt["qualification_source_bound"] is True
    assert receipt["qualification_semantic_version"] == "2.0"
    assert len(receipt["qualification_observation_parity_sha256s"]) == 8


def test_v3_dual_attested_attempt_removes_only_simulator_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = make_layout(tmp_path)
    official_seed = univtac_task_seed_start(0)
    request_path, request = write_clean_request(
        layout,
        task="lift_bottle",
        initial_seed=official_seed,
        exogenous_seed=official_seed,
        policy_kind=LivePolicyKind.N0,
    )
    sampling = CleanCampaignSamplingSpec(
        seed_protocol="univtac_consecutive_task_seed_v1",
        exception_handling="replace_exception_until_target_valid_v1",
        target_valid_trials_per_task=1,
        candidate_trials_per_task=1,
        official_eval_seed=0,
        task_seed_start=official_seed,
        policy_seed_mode="same_as_task_seed_v1",
    )
    manifest = build_clean_campaign_manifest(
        deployment_root=layout.root,
        request_paths=(request_path,),
        campaign_id="campaign-v3-paper-test",
        protocol_id=CleanCampaignProtocol.DIAGNOSTIC,
        master_seed=0,
        confidence_level=0.95,
        bootstrap_resamples=20,
        bootstrap_seed=3,
        sampling=sampling,
    )
    manifest_path = layout.requests / "clean-campaigns/v3/manifest.json"
    write_clean_campaign_manifest(manifest_path, manifest)
    write_artifact_and_attempt(
        layout,
        manifest.campaign_id,
        manifest.sha256,
        request,
        manifest_ordinal=0,
        request_file_sha256=_sha256(request_path),
        semantic_version="2.0",
        candidate_disposition="valid_outcome",
    )
    inventory = load_clean_artifact_inventory(layout.root, manifest)
    summary = build_clean_baseline_summary(manifest, inventory)
    summary_path = layout.outputs / "clean-campaigns/v3/summary.json"
    write_clean_baseline_summary(summary_path, summary)
    source = source_binding()
    qualification_path = layout.root / "artifacts/deployment/qualification-v3.json"
    qualification_path.parent.mkdir(parents=True, exist_ok=True)
    qualification_path.write_text("{}\n", encoding="utf-8")
    qualification = VerifiedAllTaskQualification(
        path=qualification_path,
        sha256="a" * 64,
        campaign_id="qualification-v3-test",
        action_spec="ee8_absolute",
        repetitions=1,
        tasks=TASKS,
        semantic_version="3.0",
        source_binding=None,
        task_source_bindings=(source,) * len(TASKS),
        campaign_manifest_sha256=manifest.sha256,
        observation_parity_sha256s=tuple(f"{index + 1:x}" * 64 for index in range(8)),
    )
    valid = replace(
        inventory.artifacts[0],
        attempt_semantic_version="3.0",
        qualification_relpath=qualification_path.relative_to(layout.root).as_posix(),
        qualification_sha256=qualification.sha256,
        runtime_source_binding=source,
        isaac_attestation_relpath="outputs/isaac-v3.json",
        isaac_attestation_sha256="b" * 64,
        n0_server_attestation_relpath="outputs/server-v3.json",
        n0_server_attestation_sha256="c" * 64,
    )
    attested_inventory = CleanArtifactInventory(
        campaign_manifest_sha256=inventory.campaign_manifest_sha256,
        artifacts=(valid,),
        missing_trials=inventory.missing_trials,
        protocol_invalid_trials=inventory.protocol_invalid_trials,
        candidate_provenance=inventory.candidate_provenance,
        semantic_version=inventory.semantic_version,
    )
    selected_inventory = [attested_inventory]
    monkeypatch.setattr(
        "robotactile_benchmark.clean_baseline.paper_bundle.verify_all_task_qualification",
        lambda *_args: qualification,
    )
    monkeypatch.setattr(
        "robotactile_benchmark.clean_baseline.paper_bundle._qualification_campaign_manifest",
        lambda *_args: manifest,
    )
    inventory_calls: list[dict[str, object]] = []

    def _full_inventory_only(
        *_args: object, **kwargs: object
    ) -> CleanArtifactInventory:
        inventory_calls.append(dict(kwargs))
        return selected_inventory[0]

    monkeypatch.setattr(
        "robotactile_benchmark.clean_baseline.paper_bundle.load_clean_artifact_inventory",
        _full_inventory_only,
    )
    bundle = build_paper_result_bundle(
        deployment_root=layout.root,
        summary_path=summary_path,
        qualification_path=qualification_path,
        output_directory=layout.outputs / "clean-campaigns/v3/paper",
    )
    assert inventory_calls == [{"allow_compact": False}]
    receipt = json.loads(bundle.receipt_path.read_text(encoding="utf-8"))

    assert "simulator_not_qualified" not in bundle.blockers
    assert "qualification_runtime_binding_unbound" not in bundle.blockers
    assert receipt["semantic_version"] == "3.0"
    assert (
        receipt["valid_attempt_attestations"][0]["isaac_attestation_sha256"] == "b" * 64
    )
    assert receipt["qualification_task_source_bindings"]["lift_bottle"] == (
        source.to_dict()
    )

    mixed = replace(
        valid,
        runtime_source_binding=replace(source, normalizer_sha256="d" * 64),
    )
    selected_inventory[0] = replace(attested_inventory, artifacts=(mixed,))
    mismatch = build_paper_result_bundle(
        deployment_root=layout.root,
        summary_path=summary_path,
        qualification_path=qualification_path,
        output_directory=layout.outputs / "clean-campaigns/v3/paper-mismatch",
    )
    assert "simulator_not_qualified" in mismatch.blockers
    assert "qualification_runtime_binding_mismatch" in mismatch.blockers

    selected_inventory[0] = attested_inventory
    monkeypatch.setattr(
        type(valid.artifact),
        "has_full_trace",
        property(lambda _: False),
    )
    compact = build_paper_result_bundle(
        deployment_root=layout.root,
        summary_path=summary_path,
        qualification_path=qualification_path,
        output_directory=layout.outputs / "clean-campaigns/v3/paper-compact",
    )
    assert "simulator_not_qualified" in compact.blockers
    assert "qualification_runtime_binding_mismatch" in compact.blockers

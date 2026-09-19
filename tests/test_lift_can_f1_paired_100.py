"""Focused evidence accounting for the 100-seed lift_can comparison."""

from pathlib import Path

import pytest

from scripts.n0_twam import lift_can_f1_paired_100 as study
from scripts.retrained_evaluation.group import write_json


def test_random_seed_set_is_frozen_unique_and_held_out_range() -> None:
    first = study.select_seeds()
    assert first == study.select_seeds()
    assert first != study.select_seeds(study.MASTER_SEED + 1)
    assert len(first) == len(set(first)) == 100
    assert min(first) >= 1_000_000


def test_preparation_and_summary_preserve_pair_denominator(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    isaac = tmp_path / "isaac-python.sh"
    isaac.touch()
    write_json(
        source,
        {
            "model": "n0_twam",
            "tasks": {"lift_can": {"prompt": "Rotate can"}},
            "checkpoint_sha256": study.EXPECTED_CHECKPOINT,
            "dataset_sha256": study.EXPECTED_DATASET,
            "training_step": 10000,
        },
    )
    campaign = tmp_path / "study"
    study.prepare(source, campaign, isaac)
    with pytest.raises(FileExistsError):
        study.prepare(source, campaign, isaac)
    seed = study.select_seeds()[0]
    group = study.seed_group(campaign, seed)
    write_json(
        group / "group.json",
        {
            "ordered_requests": list(study.EXPECTED_REQUESTS),
        },
    )
    write_json(
        group / "paired_receipt.json",
        {
            "reset_receipt": {
                "initial_seed": seed,
                "all_exact": True,
                "witnesses": [{"ordinal": 0}, {"ordinal": 1}],
            },
            "executions": [{"condition": "clean"}, {"condition": "faulted"}],
        },
    )
    for index, success in enumerate((True, False)):
        write_json(
            group / "results" / f"{index:02d}.json",
            {
                "score_eligible": True,
                "score_success": success,
                "validation_passed": True,
                "failure_stage": None,
                "terminal_status": "success" if success else "timeout",
            },
        )
    report = study.summarize(campaign)
    assert report["completed_valid_pairs"] == 1
    assert report["missing_or_invalid_pairs"] == 99
    assert report["conditions"]["clean"]["success_rate"] == 1.0
    assert report["conditions"][study.FAULT]["success_rate"] == 0.0
    assert report["paired_f1_minus_clean"] == -1.0
    assert report["paired_difference_bootstrap95"] == [-1.0, -1.0]
    assert report["evidence_scope"] == "incomplete_paired_simulator_diagnostic"


def test_replacement_artifact_is_bound_without_changing_seed_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.json"
    isaac = tmp_path / "isaac-python.sh"
    artifact = tmp_path / "replacement.json"
    isaac.touch()
    artifact.touch()
    write_json(
        source,
        {
            "model": "n0_twam",
            "tasks": {"lift_can": {}},
            "checkpoint_sha256": study.EXPECTED_CHECKPOINT,
            "dataset_sha256": study.EXPECTED_DATASET,
            "training_step": 10000,
        },
    )
    monkeypatch.setattr(
        study,
        "load_retrained",
        lambda _: {
            "checkpoint_sha256": study.EXPECTED_CHECKPOINT,
            "training_step": 10000,
            "artifact_sha256": "new-artifact-digest",
        },
    )
    campaign = tmp_path / "successor"
    study.prepare(source, campaign, isaac, artifact)
    binding = study.read_object(campaign / "binding.json")
    plan = study.read_object(campaign / "plan.json")
    assert binding["artifact"] == str(artifact)
    assert plan["model_artifact_sha256"] == "new-artifact-digest"
    assert plan["seeds"] == study.select_seeds()

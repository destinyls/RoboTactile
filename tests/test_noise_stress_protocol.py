"""Frozen cohorts, absolute-drop inference, and mixed-registry generation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from test_n0_fault_campaign_generation import _clean_request, _rest_reference

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.n0_fault_campaign.stress_group import (
    load_stress_group,
    prepare_stress_group,
)
from robotactile_benchmark.n0_fault_campaign.stress_metrics import (
    eligible_terminal,
    summarize_stress,
)
from robotactile_benchmark.n0_fault_campaign.stress_protocol import (
    StressVariant,
    freeze_protocol,
    validate_protocol,
)
from robotactile_benchmark.optical.stress_templates import calibrate_contact_scars


def protocol(
    stage="confirmation",
    seeds=tuple(range(100_000, 100_100)),
    variants=None,
    spatial=None,
    group_paths=None,
):
    variants = variants or (StressVariant("fast_f1", 5), StressVariant("fullframe", 5))
    if stage == "calibration":
        variants = tuple(
            StressVariant(family, level)
            for family in dict.fromkeys(v.family for v in variants)
            for level in (1, 3, 5)
        )
    excluded = tuple(range(20))
    evidence = None
    if stage == "confirmation":
        development = tuple(range(200_000, 200_020))
        parent = protocol(
            stage="calibration",
            seeds=development,
            variants=variants if all(v.family != "null" for v in variants) else None,
            spatial=spatial,
            group_paths={
                str(s): f"/frozen-unit-fixture/calibration/{s}" for s in development
            },
        )
        rows = results(parent, loss_count=5)
        for row in rows:
            if row["condition"] not in {v.label for v in variants}:
                row.update(score_success=True, terminal_status="success")
        evidence = {
            "schema": "n0_stress_selection_evidence_v1",
            "protocol": parent,
            "report": summarize_stress(parent, rows),
        }
        excluded += development
        group_paths = group_paths or {
            str(s): f"/frozen-unit-fixture/confirmation/{s}" for s in seeds
        }
    return freeze_protocol(
        stage=stage,
        seeds=seeds,
        variants=variants,
        excluded_seeds=excluded,
        binding={
            "dataset_sha256": "c" * 64,
            "model_sha256": "a" * 64,
            "integration_config_sha256": "b" * 64,
            "code_sha256": "d" * 64,
        },
        selection_evidence_sha256=canonical_hash(evidence) if evidence else None,
        spatial_calibration_sha256=canonical_hash(spatial) if spatial else None,
        group_paths=group_paths,
        calibration_evidence=evidence,
    )


def results(plan, loss_count=20):
    rows = []
    for index, seed in enumerate(plan["seeds"]):
        for label in ["clean", *(v["label"] for v in plan["variants"])]:
            success = label == "clean" or index >= loss_count
            rows.append(
                {
                    "seed": seed,
                    "condition": label,
                    "protocol_sha256": plan["protocol_sha256"],
                    "attempt_id": f"seed-{seed}-attempt1",
                    "group_accepted": True,
                    "exact_snapshot_reset_verified": True,
                    "score_eligible": True,
                    "validation_passed": True,
                    "terminal_status": "success" if success else "task_failure",
                    "observation_count": 400,
                    "score_success": success,
                    **(
                        {"group_root": plan["group_paths"][str(seed)]}
                        if "group_paths" in plan
                        else {}
                    ),
                }
            )
    return rows


def test_exact_twenty_percentage_points_passes_only_full_confirmation():
    plan = protocol()
    report = summarize_stress(plan, results(plan))
    assert report["accepted_live_rollouts"] == 300
    assert report["goal_met"]
    for condition in report["conditions"]:
        assert condition["drop_pp"] == pytest.approx(20)
        assert condition["paired_bootstrap_95ci"][0] > 0
        assert condition["holm_adjusted_p"] < 0.05
    plan = protocol(stage="calibration")
    assert not summarize_stress(plan, results(plan))["goal_met"]


def test_nineteen_points_and_partial_cohorts_cannot_pass():
    plan = protocol()
    assert not summarize_stress(plan, results(plan, 19))["goal_met"]
    assert not summarize_stress(plan, results(plan)[:-3])["goal_met"]


def test_infrastructure_provisional_and_unsupported_are_not_model_failures():
    plan = protocol(stage="screening")
    rows = results(plan)
    provisional = [dict(r, group_accepted=False, attempt_id="old") for r in rows[:3]]
    crash = dict(provisional[0], terminal_status="crash", score_success=False)
    unsupported = dict(
        provisional[0], terminal_status="unsupported_contract", score_success=None
    )
    report = summarize_stress(plan, rows + provisional + [crash, unsupported])
    assert not report["goal_met"]
    assert report["provisional_live_attempt_rollouts"] == 3
    assert report["invalid_attempt_rollouts"] == 1
    assert report["unsupported_contract"] == 1


@pytest.mark.parametrize(
    "mutation", ["duplicate", "attempt", "partial", "hash", "reset"]
)
def test_rejects_mixed_or_corrupt_accepted_evidence(mutation):
    plan = protocol()
    rows = results(plan)
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "attempt":
        rows[0]["attempt_id"] = "different"
    elif mutation == "partial":
        rows.pop(0)
    elif mutation == "hash":
        rows[0]["protocol_sha256"] = "bad"
    else:
        rows[0]["exact_snapshot_reset_verified"] = False
    with pytest.raises(ValueError):
        summarize_stress(plan, rows)


def test_protocol_freeze_rejects_overlap_tamper_null_confirmation():
    with pytest.raises(ValueError, match="overlap"):
        protocol(seeds=tuple(range(100)))
    with pytest.raises(ValueError, match="two different"):
        protocol(variants=(StressVariant("null", 5), StressVariant("fast_f1", 5)))
    changed = copy.deepcopy(protocol())
    changed["target_drop_absolute"] = 0.01
    with pytest.raises(ValueError, match="frozen"):
        validate_protocol(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("score_eligible", False),
        ("validation_passed", False),
        ("observation_count", 0),
        ("observation_count", True),
        ("terminal_status", "crash"),
        ("score_success", None),
    ],
)
def test_terminal_eligibility_is_strict(field, value):
    row = results(protocol())[0]
    row[field] = value
    assert not eligible_terminal(row)


def test_generates_one_shared_clean_and_five_diagnostic_variants(tmp_path: Path):
    spatial = calibrate_contact_scars(
        {s: np.ones((16, 16)) for s in ("left", "right")}, "1" * 64
    )
    variants = tuple(
        StressVariant(v, 5)
        for v in ("null", "fast_f1", "contact_f3", "fullframe", "skew_t3")
    )
    plan = protocol(
        stage="screening", seeds=(100_000,), variants=variants, spatial=spatial
    )
    root = tmp_path / "group"
    prepared = prepare_stress_group(
        root,
        protocol=plan,
        clean_request=_clean_request(tmp_path, "lift_bottle", 100_000),
        rest_reference=_rest_reference(tmp_path, "lift_bottle"),
        spatial_calibration=spatial,
    )
    loaded_plan, group, requests = load_stress_group(root)
    assert loaded_plan == plan and group == prepared
    assert len(requests) == 6
    assert len({r.output_dir for r in requests}) == 6
    assert [s["condition"] for s in group["selected"]].count("clean") == 1
    with pytest.raises(FileExistsError):
        prepare_stress_group(
            root,
            protocol=plan,
            clean_request=tmp_path / "base-clean/request.json",
            rest_reference=tmp_path / "rest-reference",
            spatial_calibration=spatial,
        )


def test_contact_template_is_required_and_hash_bound(tmp_path: Path):
    with pytest.raises(ValueError, match="template"):
        protocol(
            stage="screening",
            seeds=(100_000,),
            variants=(StressVariant("contact_f3", 5),),
        )


@pytest.mark.parametrize("family", ["delay_t1", "skew_t3"])
def test_temporal_profiles_have_causal_startup(tmp_path: Path, family: str):
    plan = protocol(
        stage="screening", seeds=(100_000,), variants=(StressVariant(family, 5),)
    )
    root = tmp_path / "group"
    prepare_stress_group(
        root,
        protocol=plan,
        clean_request=_clean_request(tmp_path, "lift_bottle", 100_000),
        rest_reference=None,
    )
    _, _, requests = load_stress_group(root)
    import json

    fault = json.loads(requests[1].fault_manifest_path.read_text())
    assert fault["parameters"]["temporal_schedule"] == "window_to_end_v1"

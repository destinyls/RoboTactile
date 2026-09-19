"""Confirmation must consume the complete, separate calibration cohort."""

import copy
from pathlib import Path

import pytest
from test_noise_stress_protocol import protocol, results

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.n0_fault_campaign import stress_group, stress_selection
from robotactile_benchmark.n0_fault_campaign.stress_metrics import summarize_stress
from robotactile_benchmark.n0_fault_campaign.stress_protocol import (
    StressVariant,
    freeze_protocol,
    select_calibration_doses,
    validate_protocol,
)


def freeze_again(plan, **changes):
    kwargs = {
        k: plan[k]
        for k in (
            "stage",
            "binding",
            "selection_evidence_sha256",
            "spatial_calibration_sha256",
            "group_paths",
            "calibration_evidence",
        )
    }
    kwargs.update(
        seeds=tuple(plan["seeds"]),
        excluded_seeds=tuple(plan["excluded_seeds"]),
        variants=tuple(
            StressVariant(v["family"], v["level"]) for v in plan["variants"]
        ),
    )
    kwargs.update(changes)
    return freeze_protocol(**kwargs)


def test_bare_hash_or_unbound_path_cannot_freeze_confirmation():
    plan = protocol()
    for changes in ({"calibration_evidence": None}, {"group_paths": None}):
        with pytest.raises(ValueError, match="verified calibration"):
            freeze_again(plan, **changes)
    with pytest.raises(ValueError, match="content/hash"):
        freeze_again(plan, selection_evidence_sha256="0" * 64)


def test_calibration_seed_exclusion_cannot_be_bypassed_by_empty_list():
    plan = protocol()
    with pytest.raises(ValueError, match="exclusions"):
        freeze_again(plan, excluded_seeds=())
    seeds = tuple(range(200_000, 200_100))
    with pytest.raises(ValueError, match="overlaps"):
        freeze_again(
            plan,
            seeds=seeds,
            excluded_seeds=(),
            group_paths={str(s): f"/confirmation/{s}" for s in seeds},
        )


def test_binding_and_selected_dose_cannot_change_after_calibration():
    plan = protocol()
    with pytest.raises(ValueError, match="remain frozen"):
        freeze_again(plan, binding={**plan["binding"], "code_sha256": "9" * 64})
    with pytest.raises(ValueError, match="minimum qualifying"):
        freeze_again(
            plan, variants=(StressVariant("fast_f1", 1), StressVariant("fullframe", 5))
        )


def test_calibration_coverage_is_required_even_with_recomputed_hash():
    plan = protocol()
    evidence = copy.deepcopy(plan["calibration_evidence"])
    evidence["report"]["accepted_live_rollouts"] -= 1
    with pytest.raises(ValueError, match="complete validated"):
        freeze_again(
            plan,
            calibration_evidence=evidence,
            selection_evidence_sha256=canonical_hash(evidence),
        )


@pytest.mark.parametrize("mutation", ["root", "retry", "provisional_retry"])
def test_fixed_path_statistics_reject_replacement_outcomes(mutation):
    plan = protocol()
    rows = results(plan)
    if mutation == "root":
        rows[0]["group_root"] += "-retry"
    else:
        rows.append(
            {**rows[0], "attempt_id": "another", "group_accepted": mutation == "retry"}
        )
    with pytest.raises(ValueError, match="unplanned|duplicate"):
        summarize_stress(plan, rows)


def test_report_derives_full_scope_and_refuses_user_selected_subset(
    tmp_path, monkeypatch
):
    seeds = (101, 102)
    paths = {str(s): str(tmp_path / str(s)) for s in seeds}
    plan = protocol(stage="calibration", seeds=seeds, group_paths=paths)
    root = Path(paths["101"])
    root.mkdir()
    (root / "group.json").write_text("{}")
    verified_rows = [r for r in results(plan) if r["seed"] == 101]
    monkeypatch.setattr(
        stress_selection, "read_stress_rows", lambda root: verified_rows
    )
    report = stress_selection.report_frozen_groups(plan)
    assert report["accepted_live_rollouts"] == 7
    assert report["missing_accepted_cells"] == 7
    with pytest.raises(ValueError, match="all and only"):
        stress_selection.report_frozen_groups(plan, [root])
    with pytest.raises(ValueError, match="incomplete"):
        stress_selection.calibration_selection_evidence(plan)


def test_selection_reads_every_frozen_group_including_negative_results(
    tmp_path, monkeypatch
):
    seeds = tuple(range(20, 40))
    paths = {str(s): str(tmp_path / str(s)) for s in seeds}
    plan = protocol(stage="calibration", seeds=seeds, group_paths=paths)
    rows = results(plan, loss_count=0)
    read = []
    for path in paths.values():
        Path(path).mkdir()
        (Path(path) / "group.json").write_text("{}")

    def read_rows(root):
        read.append(str(root))
        return [r for r in rows if r["group_root"] == str(root)]

    monkeypatch.setattr(stress_selection, "read_stress_rows", read_rows)
    evidence = stress_selection.calibration_selection_evidence(plan)
    assert read == list(paths.values())
    assert evidence["report"]["accepted_live_rollouts"] == 140
    assert all(c["drop_pp"] == 0 for c in evidence["report"]["conditions"])
    assert evidence["selection"]["decision"] == "No-Go"
    assert evidence["selection"]["selected_variants"] == []


def test_prepare_path_is_frozen_and_legacy_hash_stays_valid(tmp_path):
    plan = protocol()
    with pytest.raises(ValueError, match="unique frozen"):
        stress_group._verify_group_path(plan, plan["seeds"][0], tmp_path / "retry")
    old = protocol(stage="screening")
    assert "group_paths" not in old and "calibration_evidence" not in old
    assert validate_protocol(old) == old


def calibration_with_drops(drops):
    plan = protocol(
        stage="calibration",
        seeds=tuple(range(20, 40)),
        group_paths={str(s): f"/calibration/{s}" for s in range(20, 40)},
    )
    rows = results(plan, loss_count=0)
    for row in rows:
        success = (
            row["condition"] == "clean" or row["seed"] - 20 >= drops[row["condition"]]
        )
        row.update(
            score_success=success,
            terminal_status="success" if success else "task_failure",
        )
    return plan, summarize_stress(plan, rows)


def test_minimum_dose_preserves_nonmonotonic_and_zero_effects():
    plan, report = calibration_with_drops(
        {
            "fast_f1-s1": 4,
            "fast_f1-s3": 0,
            "fast_f1-s5": 8,
            "fullframe-s1": 0,
            "fullframe-s3": 4,
            "fullframe-s5": 0,
        }
    )
    assert not report["goal_met"]
    selected = select_calibration_doses(plan, report)
    assert selected["decision"] == "Go"
    assert [v["label"] for v in selected["selected_variants"]] == [
        "fast_f1-s1",
        "fullframe-s3",
    ]
    assert [d["observed_drop_pp"] for d in selected["families"][0]["doses"]] == [
        20,
        0,
        40,
    ]


def test_no_go_when_one_family_never_reaches_threshold():
    plan, report = calibration_with_drops(
        {
            "fast_f1-s1": 4,
            "fast_f1-s3": 8,
            "fast_f1-s5": 10,
            "fullframe-s1": 0,
            "fullframe-s3": 3,
            "fullframe-s5": 0,
        }
    )
    assert select_calibration_doses(plan, report)["decision"] == "No-Go"


def test_confirmation_recomputes_selection_and_rejects_hand_chosen_higher_dose():
    confirmation = protocol()
    evidence = copy.deepcopy(confirmation["calibration_evidence"])
    parent = evidence["protocol"]
    evidence["report"] = summarize_stress(parent, results(parent, loss_count=4))
    evidence["selection"] = {
        "decision": "Go",
        "selected_variants": confirmation["variants"],
    }
    with pytest.raises(ValueError, match="minimum qualifying"):
        freeze_again(
            confirmation,
            calibration_evidence=evidence,
            selection_evidence_sha256=canonical_hash(evidence),
        )
    selected = freeze_again(
        confirmation,
        calibration_evidence=evidence,
        selection_evidence_sha256=canonical_hash(evidence),
        variants=(StressVariant("fast_f1", 1), StressVariant("fullframe", 1)),
    )
    assert all(v["level"] == 1 for v in selected["variants"])
    evidence["report"] = summarize_stress(parent, results(parent, loss_count=0))
    with pytest.raises(ValueError, match="No-Go"):
        freeze_again(
            confirmation,
            calibration_evidence=evidence,
            selection_evidence_sha256=canonical_hash(evidence),
        )


def test_calibration_rule_is_hash_bound_and_complete_ladder_required():
    plan = protocol(stage="calibration")
    changed = copy.deepcopy(plan)
    changed["selection_rule"]["minimum_drop_pp"] = 1
    changed["protocol_sha256"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "protocol_sha256"}
    )
    with pytest.raises(ValueError, match="frozen"):
        validate_protocol(changed)
    with pytest.raises(ValueError, match="all s1/s3/s5"):
        freeze_protocol(
            stage="calibration",
            seeds=(30,),
            excluded_seeds=(),
            variants=(StressVariant("fast_f1", 5), StressVariant("fullframe", 5)),
            binding=plan["binding"],
        )


def test_selection_uses_net_paired_drop_and_retains_negative_dose():
    plan = protocol(
        stage="calibration",
        seeds=tuple(range(20, 40)),
        group_paths={str(s): f"/calibration/{s}" for s in range(20, 40)},
    )
    rows = results(plan, loss_count=0)
    for row in rows:
        index = row["seed"] - 20
        if row["condition"] == "clean":
            success = index >= 2
        elif row["condition"].endswith("s1"):
            success = index < 2 or index >= 6  # Four losses minus two gains = 10pp.
        elif row["condition"].endswith("s3"):
            success = index >= 6  # Four losses = exactly 20pp.
        else:
            success = True  # Two gains = -10pp, retained unchanged.
        row.update(
            score_success=success,
            terminal_status="success" if success else "task_failure",
        )
    report = summarize_stress(plan, rows)
    selection = select_calibration_doses(plan, report)
    assert [v["level"] for v in selection["selected_variants"]] == [3, 3]
    assert [d["observed_drop_pp"] for d in selection["families"][0]["doses"]] == [
        10,
        20,
        -10,
    ]
    report["conditions"].pop()
    with pytest.raises(ValueError, match="coverage"):
        select_calibration_doses(plan, report)


def test_direct_run_checks_imported_code_before_model_or_simulator(
    tmp_path, monkeypatch
):
    plan = protocol(stage="screening")
    monkeypatch.setattr(stress_group, "load_stress_group", lambda root: (plan, {}, []))
    monkeypatch.delenv("ROBOTACTILE_REPOSITORY_ROOT", raising=False)
    with pytest.raises(ValueError, match="REPOSITORY_ROOT"):
        stress_group.run_stress_group(
            tmp_path,
            integration_config=tmp_path / "missing",
            n0_source_root=tmp_path,
            host="localhost",
            port=1,
        )
    monkeypatch.setenv("ROBOTACTILE_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.delenv("ROBOTACTILE_PACKAGE_PATH", raising=False)
    checked = []

    def reject(code, package, digest):
        checked.append((code, package, digest))
        raise ValueError("runtime code differs from frozen protocol")

    monkeypatch.setattr(stress_group, "verify_runtime_code", reject)
    with pytest.raises(ValueError, match="code differs"):
        stress_group.run_stress_group(
            tmp_path,
            integration_config=tmp_path / "missing",
            n0_source_root=tmp_path,
            host="localhost",
            port=1,
        )
    assert checked[0][2] == plan["binding"]["code_sha256"]


def test_failed_direct_attempt_cannot_restart_at_frozen_path(tmp_path, monkeypatch):
    plan = protocol(stage="screening")
    monkeypatch.setattr(
        stress_group,
        "load_stress_group",
        lambda root: (plan, {"group_sha256": "f" * 64}, []),
    )
    monkeypatch.setattr(stress_group, "_verify_imported_runtime", lambda plan: None)
    monkeypatch.setattr(
        stress_group,
        "file_sha256",
        lambda path: plan["binding"]["integration_config_sha256"],
    )

    def failed_start(config):
        raise RuntimeError("infrastructure startup failure")

    monkeypatch.setattr(stress_group, "resolve_n0_runtime_artifacts", failed_start)
    kwargs = {
        "integration_config": tmp_path / "config",
        "n0_source_root": tmp_path,
        "host": "localhost",
        "port": 1,
    }
    with pytest.raises(RuntimeError, match="startup failure"):
        stress_group.run_stress_group(tmp_path, **kwargs)
    assert (tmp_path / "execution_started.json").is_file()
    with pytest.raises(FileExistsError):
        stress_group.run_stress_group(tmp_path, **kwargs)

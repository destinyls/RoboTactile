"""Contract tests for frozen multi-task N0 Clean request generation."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest

from robotactile_benchmark.deployment.layout import DeploymentLayout

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/n0_twam/generate_clean_campaign_requests.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("n0_clean_campaign_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_derivation_is_deterministic_separated_and_signed31() -> None:
    module = _module()
    derive = module.derive_campaign_seed
    arguments = {
        "namespace": "pilot_v1",
        "master_seed": 20260823,
        "task_id": "pull_out_key",
        "ordinal": 0,
    }
    first = derive(**arguments, role="initial")
    repeated = derive(**arguments, role="initial")
    exogenous = derive(**arguments, role="exogenous")
    next_trial = derive(**{**arguments, "ordinal": 1}, role="initial")

    assert first == repeated
    assert len({first, exogenous, next_trial}) == 3
    assert all(0 < value <= 0x7FFFFFFF for value in (first, exogenous, next_trial))


def test_seed_derivation_rejects_ambiguous_inputs() -> None:
    derive = _module().derive_campaign_seed
    with pytest.raises(ValueError, match="descriptor"):
        derive(
            namespace="pilot_v1",
            master_seed=1,
            task_id="pull_out_key",
            ordinal=0,
            role="operator",
        )
    with pytest.raises(ValueError, match="master_seed"):
        derive(
            namespace="pilot_v1",
            master_seed=-1,
            task_id="pull_out_key",
            ordinal=0,
            role="initial",
        )


def test_watchdog_budget_is_frozen_from_action_horizon() -> None:
    derive = _module().derive_watchdog_timeout_s

    assert (
        derive(
            requested_floor_s=1800.0,
            action_horizon=100,
            seconds_per_action=15.0,
        )
        == 1800.0
    )
    assert (
        derive(
            requested_floor_s=1800.0,
            action_horizon=500,
            seconds_per_action=15.0,
        )
        == 7500.0
    )

    with pytest.raises(ValueError, match="action_horizon"):
        derive(
            requested_floor_s=1800.0,
            action_horizon=0,
            seconds_per_action=15.0,
        )


def test_protocol_scope_enforces_pilot_and_paper_inventory() -> None:
    validate = _module()._validate_scope
    tasks = (
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    )
    validate("pilot_v1", tasks, tasks, 10)
    validate("paper_v1", tasks, tasks, 100)
    validate("diagnostic_v1", ("pull_out_key",), tasks, 1)

    with pytest.raises(ValueError, match="exactly 10"):
        validate("pilot_v1", tasks, tasks, 9)
    with pytest.raises(ValueError, match="exactly 100"):
        validate("paper_v1", tasks, tasks, 99)
    with pytest.raises(ValueError, match="all eight"):
        validate("paper_v1", tasks[:-1], tasks, 100)


def test_paper_requires_explicit_official_sampling_and_reserve() -> None:
    sampling_plan = _module()._sampling_plan
    base = {
        "protocol": "paper_v1",
        "sampling_contract": None,
        "master_seed": None,
        "trials_per_task": 100,
        "official_eval_seed": 0,
        "replacement_reserve_per_task": None,
    }
    with pytest.raises(ValueError, match="explicit"):
        sampling_plan(Namespace(**base))
    with pytest.raises(ValueError, match="non-negative"):
        sampling_plan(Namespace(**{**base, "sampling_contract": "univtac_official_v1"}))


def test_official_sampling_separates_valid_target_and_candidate_reserve() -> None:
    module = _module()
    campaign_seed, candidate_count, sampling = module._sampling_plan(
        Namespace(
            protocol="paper_v1",
            sampling_contract="univtac_official_v1",
            master_seed=None,
            trials_per_task=100,
            official_eval_seed=0,
            replacement_reserve_per_task=7,
        )
    )

    assert campaign_seed == 0
    assert candidate_count == 107
    assert sampling is not None
    assert sampling.target_valid_trials_per_task == 100
    assert sampling.candidate_trials_per_task == 107
    assert sampling.task_seed_start == 1_000_000
    first = module.derive_univtac_task_seed(eval_seed=0, candidate_ordinal=0)
    last = module.derive_univtac_task_seed(eval_seed=0, candidate_ordinal=106)
    assert (first, last) == (1_000_000, 1_000_106)


def test_legacy_sampling_keeps_existing_master_seed_contract() -> None:
    campaign_seed, candidate_count, sampling = _module()._sampling_plan(
        Namespace(
            protocol="diagnostic_v1",
            sampling_contract=None,
            master_seed=20260823,
            trials_per_task=1,
            official_eval_seed=0,
            replacement_reserve_per_task=None,
        )
    )

    assert (campaign_seed, candidate_count, sampling) == (20260823, 1, None)


def test_relative_campaign_paths_stay_below_deployment_root(
    tmp_path: Path,
) -> None:
    relative = _module()._relative
    root = tmp_path / "deployment"
    root.mkdir()

    assert relative(root, root / "requests/campaign/request.json", "request") == (
        "requests/campaign/request.json"
    )
    with pytest.raises(ValueError, match="below"):
        relative(root, tmp_path / "outside/request.json", "request")


def test_versioned_integration_config_is_single_task_and_root_bound(
    tmp_path: Path,
) -> None:
    select = _module()._select_integration_config
    root = tmp_path / "deployment"
    root.mkdir()
    layout = DeploymentLayout(root)
    config = root / "artifacts/models/n0_twam/configs/lift_bottle/v11.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    assert (
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=["lift_bottle"],
            requested=config,
            label=None,
        )
        == config.absolute()
    )
    with pytest.raises(ValueError, match="exactly one explicit --task"):
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=None,
            requested=config,
            label=None,
        )
    with pytest.raises(ValueError, match="exactly one explicit --task"):
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=["lift_bottle", "lift_can"],
            requested=config,
            label=None,
        )

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="below"):
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=["lift_bottle"],
            requested=outside,
            label=None,
        )

    symlink = config.with_name("symlink.json")
    symlink.symlink_to(config.name)
    with pytest.raises(ValueError, match="non-symlink"):
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=["lift_bottle"],
            requested=symlink,
            label=None,
        )


def test_default_campaign_config_path_is_unchanged(tmp_path: Path) -> None:
    select = _module()._select_integration_config
    layout = DeploymentLayout(tmp_path / "deployment")

    selected = select(
        layout=layout,
        task_id="lift_bottle",
        explicit_tasks=None,
        requested=None,
        label=None,
    )

    assert selected == (
        layout.model_artifacts / "n0_twam/configs/lift_bottle/integration_config.json"
    )


def test_labeled_campaign_config_selects_a_versioned_config_per_task(
    tmp_path: Path,
) -> None:
    select = _module()._select_integration_config
    layout = DeploymentLayout(tmp_path / "deployment")
    config = (
        layout.model_artifacts
        / "n0_twam/configs/lift_bottle-source-c43a216/integration_config.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")

    assert (
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=None,
            requested=None,
            label="source-c43a216",
        )
        == config
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        select(
            layout=layout,
            task_id="lift_bottle",
            explicit_tasks=["lift_bottle"],
            requested=config,
            label="source-c43a216",
        )

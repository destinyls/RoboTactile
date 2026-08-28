from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.integrations.configuration import (
    configure_n0_twam_integration,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    N0TWAMArtifactManifest,
    load_n0_twam_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_twam.preparation import (
    prepare_official_n0_artifacts,
)
from robotactile_benchmark.integrations.n0_twam.requests import (
    build_official_n0_clean_request,
    derive_n0_infrastructure_watchdog_timeout,
    write_official_n0_clean_request,
    write_official_n0_trial_set,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
)
from robotactile_benchmark.policies.n0_official import n0_training_prompt


def _manifest(root: Path) -> N0TWAMArtifactManifest:
    model_root = root / "artifacts/models/n0_twam"
    model_root.mkdir(parents=True, exist_ok=True)
    base = model_root / "base"
    checkpoint = model_root / "univtac-delta"
    for component in ("vae", "tokenizer", "text_encoder"):
        (base / component).mkdir(parents=True)
        (base / component / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "transformer/config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "transformer/diffusion_pytorch_model.safetensors").write_bytes(
        b"weights"
    )
    (checkpoint / "train_meta.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "norm").mkdir()
    task_key = "univtac_pull_out_key_rot6d_current"
    (checkpoint / "norm/pull_out_key.norm_stat_per_robot.json").write_text(
        json.dumps({task_key: {"q01": [0.0] * 20, "q99": [1.0] * 20}}),
        encoding="utf-8",
    )
    (checkpoint / "norm/PROMPTS.json").write_text(
        json.dumps({task_key: n0_training_prompt("pull_out_key")}), encoding="utf-8"
    )
    prepared = prepare_official_n0_artifacts(
        bundle_root=model_root, task_id="pull_out_key"
    )
    config_root = model_root / "configs/pull_out_key"
    generated = configure_n0_twam_integration(
        bundle_root=prepared.bundle_root,
        task_id="pull_out_key",
        base_root=prepared.base_root,
        checkpoint_root=prepared.checkpoint_root,
        serve_bundle_root=prepared.serve_bundle_root,
        serve_pool_root=prepared.serve_pool_root,
        checkpoint_path=prepared.checkpoint_path,
        model_config_path=prepared.config_path,
        train_meta_path=prepared.train_meta_path,
        normalizer_path=prepared.normalizer_path,
        prompt_manifest_path=prepared.prompt_manifest_path,
        serve_bundle_manifest_path=prepared.serve_bundle_manifest_path,
        serve_info_path=prepared.serve_info_path,
        serve_tasks_path=prepared.serve_tasks_path,
        manifest_path=config_root / "artifact_manifest.json",
        config_path=config_root / "integration_config.json",
        device="cuda",
    )
    return load_n0_twam_artifact_manifest(generated.artifact_manifest_path)


def test_clean_request_generation_is_content_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    initialize_deployment_layout(layout)
    manifest = _manifest(layout.root)
    request = build_official_n0_clean_request(
        manifest=manifest,
        layout=layout,
        dataset_sha256="a" * 64,
        initial_seed=11,
        exogenous_seed=22,
        max_control_cycles=1,
        max_observation_steps=2,
        wall_timeout_s=10.0,
        simulator_device="cuda:9",
        live_output_dir=layout.artifacts / "live-univtac/n0/clean",
    )
    target = layout.requests / "n0-twam/pull_out_key/clean.json"
    first = write_official_n0_clean_request(target, request)
    second = write_official_n0_clean_request(target, request)

    assert first == second
    assert load_live_univtac_request(target) == request
    assert request.execute_action_steps == 24
    assert request.checkpoint_sha256 == manifest.checkpoint_sha256
    assert request.n0_source_commit == manifest.external_commit
    assert request.simulator_device == "cuda:9"
    assert N0_LIVE_UNIVTAC_INPUT_PROFILE.profile_id in request.base_system_id
    assert "initial_state_policy" not in live_univtac_request_to_dict(request)

    repeated_request = build_official_n0_clean_request(
        manifest=manifest,
        layout=layout,
        dataset_sha256="a" * 64,
        initial_seed=11,
        exogenous_seed=22,
        max_control_cycles=1,
        max_observation_steps=2,
        wall_timeout_s=10.0,
        simulator_device="cuda:9",
        live_output_dir=layout.artifacts / "live-univtac/n0/clean",
    )
    assert canonical_json_bytes(live_univtac_request_to_dict(repeated_request)) == (
        canonical_json_bytes(live_univtac_request_to_dict(request))
    )

    robust = replace(
        request,
        initial_state_policy=InitialStatePolicy.REPLACE_INITIAL_TERMINAL_V1,
        output_dir=layout.artifacts / "live-univtac/n0/clean-robust",
    )
    robust_target = target.with_name("clean-robust.json")
    write_official_n0_clean_request(robust_target, robust)
    robust_document = json.loads(robust_target.read_text(encoding="utf-8"))
    assert robust_document["initial_state_policy"] == "replace_initial_terminal_v1"
    assert load_live_univtac_request(robust_target) == robust
    assert load_live_univtac_run(robust).content_sha256 != (
        load_live_univtac_run(request).content_sha256
    )

    diagnostic = replace(
        request,
        initial_state_policy=InitialStatePolicy.DIAGNOSTIC_ALLOW_INVALID_V1,
        output_dir=layout.artifacts / "live-univtac/n0/clean-diagnostic",
    )
    diagnostic_target = target.with_name("clean-diagnostic.json")
    write_official_n0_clean_request(diagnostic_target, diagnostic)
    diagnostic_document = json.loads(diagnostic_target.read_text(encoding="utf-8"))
    assert diagnostic_document["initial_state_policy"] == (
        "diagnostic_allow_invalid_v1"
    )
    assert load_live_univtac_request(diagnostic_target) == diagnostic
    assert load_live_univtac_run(diagnostic).content_sha256 != (
        load_live_univtac_run(request).content_sha256
    )

    watchdog = replace(
        request,
        wall_timeout_role=WallTimeoutRole.INFRASTRUCTURE_WATCHDOG_V1,
        output_dir=layout.artifacts / "live-univtac/n0/clean-watchdog",
    )
    watchdog_target = target.with_name("clean-watchdog.json")
    write_official_n0_clean_request(watchdog_target, watchdog)
    watchdog_document = json.loads(watchdog_target.read_text(encoding="utf-8"))
    assert watchdog_document["wall_timeout_role"] == ("infrastructure_watchdog_v1")
    assert load_live_univtac_request(watchdog_target) == watchdog
    assert load_live_univtac_run(watchdog).content_sha256 != (
        load_live_univtac_run(request).content_sha256
    )

    changed = build_official_n0_clean_request(
        manifest=manifest,
        layout=layout,
        dataset_sha256="b" * 64,
        initial_seed=11,
        exogenous_seed=22,
        max_control_cycles=1,
        max_observation_steps=2,
        wall_timeout_s=10.0,
        simulator_device="cuda:9",
        live_output_dir=layout.artifacts / "live-univtac/n0/clean",
    )
    with pytest.raises(FileExistsError, match="different"):
        write_official_n0_clean_request(target, changed)


def test_infrastructure_watchdog_budget_is_horizon_derived_and_non_scoring() -> None:
    assert (
        derive_n0_infrastructure_watchdog_timeout(
            requested_floor_s=1800.0,
            action_horizon=100,
        )
        == 1800.0
    )
    assert (
        derive_n0_infrastructure_watchdog_timeout(
            requested_floor_s=1800.0,
            action_horizon=500,
        )
        == 7500.0
    )

    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            derive_n0_infrastructure_watchdog_timeout(
                requested_floor_s=invalid,
                action_horizon=500,
            )


def test_trial_set_generation_is_content_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trial_set_manifest.json"
    first = write_official_n0_trial_set(
        manifest_path=target,
        task_id="pull_out_key",
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=300,
        max_observation_steps=301,
    )
    second = write_official_n0_trial_set(
        manifest_path=target,
        task_id="pull_out_key",
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=300,
        max_observation_steps=301,
    )

    assert first == second
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "action_horizon": 300,
        "exogenous_seed": 29,
        "identity_kind": "frozen_trial_set_manifest",
        "initial_seed": 17,
        "max_observation_steps": 301,
        "semantic_version": "1.0",
        "task_id": "pull_out_key",
        "task_registry_id": "robotactile_univtac_tasks_v1",
        "trial_count": 1,
        "upstream_commit": "05bcd3edb92237107efa40105292a24f1a9fd761",
    }

    with pytest.raises(FileExistsError, match="different"):
        write_official_n0_trial_set(
            manifest_path=target,
            task_id="pull_out_key",
            initial_seed=18,
            exogenous_seed=29,
            max_control_cycles=300,
            max_observation_steps=301,
        )


def test_trial_set_rejects_an_inconsistent_horizon(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must equal"):
        write_official_n0_trial_set(
            manifest_path=tmp_path / "trial_set_manifest.json",
            task_id="pull_out_key",
            initial_seed=17,
            exogenous_seed=29,
            max_control_cycles=300,
            max_observation_steps=300,
        )

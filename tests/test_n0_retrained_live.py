"""Retrained requests reuse live loading/export while retaining legacy identities."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_live_univtac_artifacts import _capture_n0, _request

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
)
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.closed_loop.runner import run_closed_loop_trial_with_evidence
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution import (
    LivePolicyKind,
    load_live_univtac_artifact,
    load_live_univtac_run,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.live_artifacts_values import live_request_identity
from robotactile_benchmark.execution.loading import load_live_univtac_request
from robotactile_benchmark.execution.request_values import live_univtac_request_to_dict
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.integrations.n0_twam.retrained import server_overrides
from robotactile_benchmark.integrations.n0_twam.retrained_live import (
    make_request,
    validate_server_receipt,
)
from robotactile_benchmark.policies.n0_input_profile import (
    N0_LIVE_UNIVTAC_INPUT_PROFILE,
    N0_RETRAINED_INPUT_PROFILE,
    prepare_n0_image,
)
from robotactile_benchmark.trials import Condition


def retrained_request(root):
    return replace(
        _request(root, Condition.CLEAN),
        task_id="pull_out_key",
        policy_kind=LivePolicyKind.N0,
        max_control_cycles=1,
        max_observation_steps=9,
        execute_action_steps=8,
        act_device_name=None,
        n0_source_commit="9" * 40,
        n0_normalizer_sha256="d" * 64,
        n0_serve_bundle_sha256="e" * 64,
        n0_prompt_manifest_sha256="f" * 64,
        n0_action_per_frame=4,
        n0_prompt_override="training prompt",
    )


def test_retrained_request_serialization_and_loading(tmp_path):
    request = retrained_request(tmp_path)
    document = live_univtac_request_to_dict(request)
    assert document["n0_action_per_frame"] == 4
    assert document["n0_prompt_override"] == "training prompt"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(document))
    loaded_request = load_live_univtac_request(path)
    assert loaded_request == request
    loaded = load_live_univtac_run(loaded_request)
    assert (
        loaded.backend_config.action_execution_contract
        == N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT
    )
    assert loaded.backend_config.physics_steps_per_action == 12
    assert loaded.run_spec.prompt == "training prompt"
    assert loaded.run_spec.execute_action_steps == 8
    assert live_request_identity(loaded)["n0_action_per_frame"] == 4


@pytest.mark.parametrize(
    "profile", [LiveCaptureProfile.PREVIEW, LiveCaptureProfile.PAPER_FULL]
)
def test_retrained_preview_full_roundtrip(tmp_path, profile):
    loaded = load_live_univtac_run(retrained_request(tmp_path))
    backend = DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )
    backend.action_spec = loaded.trial.action_spec
    evidence = run_closed_loop_trial_with_evidence(
        loaded.trial,
        loaded.run_spec,
        backend,
        DeterministicFakePolicy.for_trial(loaded.trial),
    )
    target = tmp_path / profile.value
    write_live_univtac_artifact(target, loaded, evidence, capture_profile=profile)
    reopened = load_live_univtac_artifact(target)
    assert reopened.run_content_sha256 == loaded.content_sha256
    assert reopened.run_spec == loaded.run_spec
    assert reopened.evidence.result == evidence.result
    assert reopened.run_spec.prompt == "training prompt"
    if profile is LiveCaptureProfile.PREVIEW:
        assert reopened.preview_trace is not None
    else:
        assert reopened.evidence.finalization is not None


@pytest.mark.parametrize(
    "updates",
    [
        {"n0_prompt_override": None},
        {"n0_prompt_override": ""},
        {"execute_action_steps": 24},
    ],
)
def test_incomplete_or_mismatched_retrained_requests_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        replace(retrained_request(tmp_path), **updates)


def test_legacy_default_serialization_and_identity_has_no_new_fields(tmp_path):
    loaded, _ = _capture_n0(tmp_path)
    explicit = replace(loaded.request, n0_action_per_frame=12, n0_prompt_override=None)
    old_document = live_univtac_request_to_dict(loaded.request)
    new_document = live_univtac_request_to_dict(explicit)
    assert canonical_hash(old_document) == canonical_hash(new_document)
    identity = live_request_identity(loaded)
    for key in ("n0_action_per_frame", "n0_prompt_override"):
        assert key not in old_document and key not in identity
    explicit_loaded = load_live_univtac_run(explicit)
    assert canonical_hash(identity) == canonical_hash(
        live_request_identity(explicit_loaded)
    )
    assert loaded.content_sha256 == explicit_loaded.content_sha256
    assert (
        loaded.backend_config.action_execution_contract
        == N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT
    )
    assert loaded.backend_config.physics_steps_per_action == 2


@pytest.fixture
def runner_artifact():
    artifact = {
        "schema_version": "robotactile-n0-retrained-artifact-v1",
        "artifact_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "training_step": 10000,
        "action_per_frame": 4,
        "action_hz": 10,
        "bundle_root": "/synthetic/bundle",
        "normalizer_path": "/synthetic/normalizer.json",
        "source": {"source_commit": "9" * 40},
        "tasks": {
            "pull_out_key": {
                "prompt": "training prompt",
                "config_sha256": "c" * 64,
                "normalizer_sha256": "d" * 64,
                "normalizer": {"q01": [-2.0] * 20, "q99": [3.0] * 20},
            }
        },
    }
    artifact["tasks"]["pull_out_key"]["config_sha256"] = canonical_hash(
        server_overrides(artifact, "pull_out_key")
    )
    return artifact


def test_runner_make_request_binds_actual_training_identity(tmp_path, runner_artifact):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"test_fixture":true}')
    request = make_request(
        runner_artifact,
        task="pull_out_key",
        dataset_manifest=dataset,
        upstream=tmp_path / "upstream",
        runtime=tmp_path / "runtime",
        output=tmp_path / "output",
        seed=42,
        watchdog_s=7200.0,
    )
    assert request.n0_action_per_frame == 4
    assert request.n0_prompt_override == "training prompt"
    assert request.execute_action_steps == 8
    assert request.checkpoint_sha256 == runner_artifact["checkpoint_sha256"]
    assert request.config_sha256 == canonical_hash(
        {
            "server_config_sha256": runner_artifact["tasks"]["pull_out_key"][
                "config_sha256"
            ],
            "input_profile_sha256": N0_RETRAINED_INPUT_PROFILE.sha256,
            "execution_contract": N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        }
    )
    loaded = load_live_univtac_run(request)
    assert loaded.backend_config.physics_steps_per_action == 12
    assert loaded.run_spec.prompt == "training prompt"
    with pytest.raises(ValueError, match="10 Hz"):
        make_request(
            {**runner_artifact, "action_per_frame": 12},
            task="pull_out_key",
            dataset_manifest=dataset,
            upstream=tmp_path,
            runtime=tmp_path,
            output=tmp_path / "unused",
            seed=42,
            watchdog_s=7200.0,
        )


def server_receipt(artifact):
    runtime = {
        **json.loads(json.dumps(server_overrides(artifact, "pull_out_key"))),
        "port": 29601,
        "host": "127.0.0.1",
        "action_delta_mode": "none",
        "action_per_frame": 4,
        "serve_task": "pull_out_key",
    }
    return {
        "schema_version": "robotactile-n0-retrained-server-v1",
        "artifact_sha256": artifact["artifact_sha256"],
        "task": "pull_out_key",
        "config_sha256": artifact["tasks"]["pull_out_key"]["config_sha256"],
        "runtime_config": runtime,
        "runtime_config_sha256": canonical_hash(runtime),
    }


@pytest.mark.parametrize("change", ["artifact", "task", "port", "legacy"])
def test_server_receipt_rejects_other_artifact_task_port_and_legacy(
    runner_artifact, change
):
    receipt = server_receipt(runner_artifact)
    validate_server_receipt(runner_artifact, "pull_out_key", receipt, 29601)
    if change == "artifact":
        receipt["artifact_sha256"] = "f" * 64
    elif change == "task":
        receipt["task"] = "insert_HDMI"
    elif change == "port":
        receipt["runtime_config"]["port"] = 29602
        receipt["runtime_config_sha256"] = canonical_hash(receipt["runtime_config"])
    else:
        receipt["schema_version"] = "official-release-server"
    with pytest.raises(ValueError, match="does not match"):
        validate_server_receipt(runner_artifact, "pull_out_key", receipt, 29601)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_local_tactile", False),
        ("tactile_cfg_prob", 0.5),
        ("norm_stat", {"q01": [-9.0] * 20, "q99": [9.0] * 20}),
        ("prompt", "different training prompt"),
        ("wan22_pretrained_model_name_or_path", "/different/bundle"),
        ("guidance_scale", 3),
    ],
)
def test_server_receipt_rejects_changed_contract_with_recomputed_hash(
    runner_artifact, field, value
):
    receipt = server_receipt(runner_artifact)
    receipt["runtime_config"][field] = value
    receipt["runtime_config_sha256"] = canonical_hash(receipt["runtime_config"])
    with pytest.raises(ValueError, match="does not match"):
        validate_server_receipt(runner_artifact, "pull_out_key", receipt, 29601)


def test_retrained_rgb_is_identity_and_released_rgb_still_reversed():
    image = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    new = prepare_n0_image(image, profile=N0_RETRAINED_INPUT_PROFILE, name="fixture")
    old = prepare_n0_image(image, profile=N0_LIVE_UNIVTAC_INPUT_PROFILE, name="fixture")
    np.testing.assert_array_equal(new, [[[10, 20, 30]]])
    np.testing.assert_array_equal(old, [[[30, 20, 10]]])
    np.testing.assert_array_equal(image, [[[10, 20, 30]]])


def test_schema_keeps_new_fields_optional_and_binds_four_to_eight():
    path = (
        Path(__file__).resolve().parents[1] / "schemas/live_univtac_request.schema.json"
    )
    schema = json.loads(path.read_text())
    assert schema["properties"]["n0_action_per_frame"]["enum"] == [4, 12]
    assert schema["properties"]["n0_prompt_override"]["minLength"] == 1
    assert not {"n0_action_per_frame", "n0_prompt_override"} & set(schema["required"])
    encoded = json.dumps(schema, sort_keys=True)
    assert '"n0_action_per_frame": {"const": 4}' in encoded
    assert '"execute_action_steps": {"const": 8}' in encoded

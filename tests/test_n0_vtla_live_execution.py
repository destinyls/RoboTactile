"""N0-VTLA live request, binding, and paired-dispatch contracts."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark.action_specs import QPOS8_ACTION_SPEC
from robotactile_benchmark.backends.univtac_success_profiles import (
    INSERT_HOLE_STRICT_PREDICATE_ID,
    UniVTACSuccessProfile,
)
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution import live_cli
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.loading import load_live_univtac_run
from robotactile_benchmark.execution.official_n0_vtla import (
    build_official_n0_vtla_live_binding,
)
from robotactile_benchmark.execution.request_values import (
    live_univtac_request_to_dict,
)
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    ASSET_ID,
    N0VTLAArtifactManifest,
    build_n0_vtla_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_vtla.requests import (
    build_official_n0_vtla_request,
)
from robotactile_benchmark.trials import Condition


def test_insert_hole_protocol_declares_official_as_primary_metric() -> None:
    protocol = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/protocols/univtac_insert_hole_dual_success_v1.json"
        ).read_text(encoding="utf-8")
    )
    aggregation = protocol["aggregation"]

    assert aggregation["primary_profile_id"] == "official_v1"
    assert aggregation["primary_metric_name"] == "insert_hole_official_sr"
    assert aggregation["strict_metric_role"] == "diagnostic_only"
    assert aggregation["strict_affects_official_outcome"] is False
    assert aggregation["mix_profiles_in_one_denominator"] is False


def _manifest(root: Path) -> N0VTLAArtifactManifest:
    bundle = root / "artifacts/models/n0_vtla"
    checkpoint = bundle / "checkpoint"
    normalizer = checkpoint / "assets" / ASSET_ID / "norm_stats.json"
    normalizer.parent.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"released-weights")
    (checkpoint / "config.json").write_text(
        '{"config":"sim_single_arm_tactile"}\n', encoding="utf-8"
    )
    normalizer.write_text('{"norm":true}\n', encoding="utf-8")
    return build_n0_vtla_artifact_manifest(
        bundle_root=bundle.absolute(),
        checkpoint_root=checkpoint.absolute(),
    )


def test_request_loads_qpos50_prompt_and_validates_schema(tmp_path: Path) -> None:
    deployment = (tmp_path / "deployment").absolute()
    layout = DeploymentLayout(deployment)
    manifest = _manifest(deployment)
    request = build_official_n0_vtla_request(
        manifest=manifest,
        layout=layout,
        condition=Condition.CLEAN,
        dataset_sha256="d" * 64,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=6,
        max_observation_steps=300,
        wall_timeout_s=1800.0,
        simulator_device="cuda:0",
        live_output_dir=deployment / "outputs/n0-vtla/clean",
    )

    loaded = load_live_univtac_run(request)
    document = live_univtac_request_to_dict(request)
    schema = json.loads(
        (
            Path(__file__).parents[1] / "schemas/live_univtac_request.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert document["policy_kind"] in schema["properties"]["policy_kind"]["enum"]
    assert any(
        branch.get("if", {}).get("properties", {}).get("policy_kind")
        == {"const": "n0_vtla"}
        and branch["then"]["properties"]["execute_action_steps"] == {"const": 50}
        for branch in schema["allOf"]
    )

    assert request.policy_kind is LivePolicyKind.N0_VTLA
    assert request.success_profile_id is UniVTACSuccessProfile.OFFICIAL_V1
    assert request.execute_action_steps == 50
    assert loaded.policy_identity.action_spec == QPOS8_ACTION_SPEC
    assert loaded.run_spec.prompt == "insert hole"
    assert loaded.run_spec.max_control_cycles == 6


def test_strict_request_selects_independent_insert_hole_predicate(
    tmp_path: Path,
) -> None:
    deployment = (tmp_path / "deployment").absolute()
    layout = DeploymentLayout(deployment)
    request = build_official_n0_vtla_request(
        manifest=_manifest(deployment),
        layout=layout,
        condition=Condition.CLEAN,
        dataset_sha256="f" * 64,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=6,
        max_observation_steps=300,
        wall_timeout_s=1800.0,
        simulator_device="cuda:0",
        live_output_dir=deployment / "outputs/n0-vtla/strict",
        success_profile_id=UniVTACSuccessProfile.INSERT_HOLE_STRICT_V1,
    )

    loaded = load_live_univtac_run(request)
    document = live_univtac_request_to_dict(request)

    assert document["success_profile_id"] == INSERT_HOLE_STRICT_PREDICATE_ID
    assert loaded.run_spec.success_predicate_id == INSERT_HOLE_STRICT_PREDICATE_ID
    assert (
        loaded.run_spec.sha256
        != load_live_univtac_run(
            build_official_n0_vtla_request(
                manifest=_manifest(deployment / "official"),
                layout=DeploymentLayout(deployment / "official"),
                condition=Condition.CLEAN,
                dataset_sha256="f" * 64,
                initial_seed=17,
                exogenous_seed=29,
                max_control_cycles=6,
                max_observation_steps=300,
                wall_timeout_s=1800.0,
                simulator_device="cuda:0",
                live_output_dir=deployment / "outputs/n0-vtla/official",
            )
        ).run_spec.sha256
    )


def test_source_and_artifact_binding_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = (tmp_path / "deployment").absolute()
    layout = DeploymentLayout(deployment)
    manifest = _manifest(deployment)
    request = build_official_n0_vtla_request(
        manifest=manifest,
        layout=layout,
        condition=Condition.CLEAN,
        dataset_sha256="e" * 64,
        initial_seed=3,
        exogenous_seed=5,
        max_control_cycles=6,
        max_observation_steps=300,
        wall_timeout_s=1800.0,
        simulator_device="cuda:0",
        live_output_dir=deployment / "outputs/n0-vtla/clean",
    )
    source_root = deployment / "sources/N0-VTLA"
    source_root.mkdir(parents=True)
    monkeypatch.setattr(
        "robotactile_benchmark.execution.official_n0_vtla.verify_external_checkout",
        lambda spec, root: SimpleNamespace(commit_sha=manifest.external_commit),
    )

    binding = build_official_n0_vtla_live_binding(
        request,
        manifest=manifest,
        source_root=source_root,
        endpoint="tcp://127.0.0.1:5557",
    )

    assert binding.manifest == manifest
    assert binding.source_root == source_root
    assert binding.endpoint == "tcp://127.0.0.1:5557"


def test_paired_cli_publishes_receipt_before_simulator_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    paired_result = object()
    requests = (
        SimpleNamespace(policy_kind=LivePolicyKind.N0_VTLA),
        SimpleNamespace(policy_kind=LivePolicyKind.N0_VTLA),
    )

    def execute(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(paired=paired_result, artifacts=())

    monkeypatch.setattr(
        live_cli,
        "load_live_univtac_request",
        lambda path: requests[int(Path(path).stem)],
    )
    monkeypatch.setattr(
        live_cli,
        "_n0_vtla_runtime_artifacts",
        lambda _args: (DeploymentLayout(tmp_path), SimpleNamespace(manifest=object())),
    )
    monkeypatch.setattr(live_cli, "execute_official_n0_vtla_paired_live_runs", execute)
    monkeypatch.setattr(live_cli, "_paired_payload", lambda *_args: {})

    writes: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        live_cli,
        "write_paired_execution_receipt",
        lambda path, result: writes.append((path, result)),
    )
    receipt = tmp_path / "paired-receipt.json"
    args = Namespace(
        action_execution_contract=None,
        capture_profile="preview_v1",
        command="live-univtac-paired-run",
        lifecycle_journal=None,
        n0_vtla_endpoint="tcp://127.0.0.1:5557",
        n0_vtla_source_root=tmp_path / "N0-VTLA",
        receipt=receipt,
        requests=(tmp_path / "0", tmp_path / "1"),
        root=tmp_path,
    )

    assert live_cli.handle_live_execution_command(args) == {}
    publisher = captured["pre_close_publisher"]
    assert callable(publisher)
    publisher(SimpleNamespace(paired=paired_result))
    assert writes == [(receipt, paired_result)]

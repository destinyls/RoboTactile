"""N0 live CLI action-execution contract selection tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution import live_cli
from robotactile_benchmark.execution.contracts import LivePolicyKind


def _args(
    tmp_path: Path,
    requested: str | None = None,
    capture_profile: str = "paper_full_v1",
) -> Namespace:
    return Namespace(
        action_execution_contract=requested,
        capture_profile=capture_profile,
        campaign_id=None,
        command="live-univtac-run",
        isaac_attestation_output=None,
        lifecycle_journal=None,
        n0_host="127.0.0.1",
        n0_port=29601,
        n0_server_attestation=None,
        n0_server_attestation_sha256=None,
        n0_source_root=tmp_path / "N0-TWAM",
        qualification=None,
        request=tmp_path / "request.json",
        root=tmp_path,
    )


def test_n0_live_cli_defaults_to_training_aligned_60hz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def execute(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    request = SimpleNamespace(policy_kind=LivePolicyKind.N0, task_id="insert_hole")
    runtime = SimpleNamespace(manifest=object())
    monkeypatch.setattr(live_cli, "load_live_univtac_request", lambda _path: request)
    monkeypatch.setattr(
        live_cli,
        "_n0_runtime_artifacts",
        lambda _args, _task: (DeploymentLayout(tmp_path), runtime),
    )
    monkeypatch.setattr(live_cli, "execute_official_n0_live_run", execute)
    monkeypatch.setattr(live_cli, "official_act_live_summary", lambda _item: {})

    assert live_cli.handle_live_execution_command(_args(tmp_path)) == {}
    assert captured["action_execution_contract"] == "robotactile_n0_training_60hz_ee_v1"
    assert captured["capture_profile"].value == "paper_full_v1"


def test_n0_live_cli_preserves_explicit_diagnostic_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def execute(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    request = SimpleNamespace(policy_kind=LivePolicyKind.N0, task_id="insert_hole")
    runtime = SimpleNamespace(manifest=object())
    monkeypatch.setattr(live_cli, "load_live_univtac_request", lambda _path: request)
    monkeypatch.setattr(
        live_cli,
        "_n0_runtime_artifacts",
        lambda _args, _task: (DeploymentLayout(tmp_path), runtime),
    )
    monkeypatch.setattr(live_cli, "execute_official_n0_live_run", execute)
    monkeypatch.setattr(live_cli, "official_act_live_summary", lambda _item: {})

    args = _args(tmp_path, "robotactile_fixed_endpoint_v1")
    assert live_cli.handle_live_execution_command(args) == {}
    assert captured["action_execution_contract"] == "robotactile_fixed_endpoint_v1"


def test_n0_live_cli_propagates_preview_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def execute(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    request = SimpleNamespace(policy_kind=LivePolicyKind.N0, task_id="insert_hole")
    runtime = SimpleNamespace(manifest=object())
    monkeypatch.setattr(live_cli, "load_live_univtac_request", lambda _path: request)
    monkeypatch.setattr(
        live_cli,
        "_n0_runtime_artifacts",
        lambda _args, _task: (DeploymentLayout(tmp_path), runtime),
    )
    monkeypatch.setattr(live_cli, "execute_official_n0_live_run", execute)
    monkeypatch.setattr(live_cli, "official_act_live_summary", lambda _item: {})

    assert (
        live_cli.handle_live_execution_command(
            _args(tmp_path, capture_profile="preview_v1")
        )
        == {}
    )
    assert captured["capture_profile"].value == "preview_v1"


def test_n0_paired_cli_propagates_training_contract_and_preview_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def execute(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(paired=object(), artifacts=())

    requests = (
        SimpleNamespace(policy_kind=LivePolicyKind.N0, task_id="insert_tube"),
        SimpleNamespace(policy_kind=LivePolicyKind.N0, task_id="insert_tube"),
    )
    runtime = SimpleNamespace(manifest=object())
    monkeypatch.setattr(
        live_cli,
        "load_live_univtac_request",
        lambda path: requests[int(Path(path).stem)],
    )
    monkeypatch.setattr(
        live_cli,
        "_n0_runtime_artifacts",
        lambda _args, _task: (DeploymentLayout(tmp_path), runtime),
    )
    monkeypatch.setattr(live_cli, "execute_official_n0_paired_live_runs", execute)
    monkeypatch.setattr(live_cli, "_paired_payload", lambda *_args: {})
    args = Namespace(
        action_execution_contract=None,
        capture_profile="preview_v1",
        command="live-univtac-paired-run",
        lifecycle_journal=None,
        n0_host="127.0.0.1",
        n0_port=29601,
        n0_source_root=tmp_path / "N0-TWAM",
        receipt=tmp_path / "receipt.json",
        requests=(tmp_path / "0", tmp_path / "1"),
        root=tmp_path,
    )

    assert live_cli.handle_live_execution_command(args) == {}
    assert captured["action_execution_contract"] == (
        "robotactile_n0_training_60hz_ee_v1"
    )
    assert captured["capture_profile"].value == "preview_v1"

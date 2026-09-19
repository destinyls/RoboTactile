"""Read-only official ACT runtime probe contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import robotactile_benchmark.integrations.act.runtime_probe as runtime_probe
from robotactile_benchmark.integrations.runtime_config import ACTRuntimeArtifacts
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile
from robotactile_benchmark.policies.univtac_official_act_manifest import (
    OfficialUniVTACACTArtifactManifest,
)
from scripts.act.verify_official_runtime import main


def _runtime(tmp_path: Path) -> ACTRuntimeArtifacts:
    manifest = OfficialUniVTACACTArtifactManifest.for_shared_root(
        task_id="pull_out_key",
        profile=OfficialACTProfile.UNIVTAC,
        artifact_root=(tmp_path / "artifacts").absolute(),
        upstream_root=(tmp_path / "UniVTAC").absolute(),
        checkpoint_sha256="a" * 64,
        stats_sha256="b" * 64,
        encoder_sha256="c" * 64,
    )
    return ACTRuntimeArtifacts(
        manifest=manifest,
        artifact_root=manifest.artifact_root,
        stats_sha256=manifest.stats_sha256,
        encoder_sha256=manifest.encoder_sha256,
        manifest_path=(tmp_path / "artifact_manifest.json").absolute(),
        device="cuda:2",
    )


def _all_imports(_name: str) -> None:
    return None


def test_default_probe_validates_artifacts_and_imports_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    resolver = Mock(return_value=runtime)
    loader = Mock()
    importer = Mock(side_effect=_all_imports)
    monkeypatch.setattr(runtime_probe, "resolve_act_runtime_artifacts", resolver)
    monkeypatch.setattr(runtime_probe, "_import_dependency", importer)
    monkeypatch.setattr(runtime_probe, "load_official_univtac_act_policy", loader)
    config = tmp_path / "integration_config.json"

    result = runtime_probe.probe_official_act_runtime(config)

    resolver.assert_called_once_with(config.absolute())
    assert result.artifact_valid is True
    assert result.dependency_imports_valid is True
    assert set(result.dependency_imports) == {
        "IPython",
        "numpy",
        "torch",
        "torchvision",
    }
    assert result.policy_load_requested is False
    assert result.policy_loaded is False
    assert result.policy_closed is False
    assert result.passed is True
    loader.assert_not_called()


def test_load_policy_builds_exact_identity_request_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    policy = Mock()
    loader = Mock(return_value=policy)
    monkeypatch.setattr(
        runtime_probe, "resolve_act_runtime_artifacts", Mock(return_value=runtime)
    )
    monkeypatch.setattr(
        runtime_probe, "_import_dependency", Mock(side_effect=_all_imports)
    )
    monkeypatch.setattr(runtime_probe, "load_official_univtac_act_policy", loader)

    result = runtime_probe.probe_official_act_runtime(
        tmp_path / "integration_config.json", load_policy=True
    )

    identity, request = loader.call_args.args
    assert identity.system_id == (
        "official-univtac-act.pull_out_key.univtac.policy_last.v1"
    )
    assert identity.checkpoint_sha256 == runtime.manifest.checkpoint_sha256
    assert identity.config_sha256 == runtime.manifest.config_sha256
    assert identity.consumes_tactile is True
    assert identity.supports_structural_absence is False
    assert request.manifest is runtime.manifest
    assert request.task_id == "pull_out_key"
    assert request.profile is OfficialACTProfile.UNIVTAC
    assert request.device_name == "cuda:2"
    assert request.live is True
    policy.close.assert_called_once_with()
    assert result.policy_loaded is True
    assert result.policy_closed is True
    assert result.passed is True


def test_artifact_failure_stops_before_dependency_or_policy_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importer = Mock()
    loader = Mock()
    monkeypatch.setattr(
        runtime_probe,
        "resolve_act_runtime_artifacts",
        Mock(side_effect=ValueError("checkpoint SHA256 mismatch")),
    )
    monkeypatch.setattr(runtime_probe, "_import_dependency", importer)
    monkeypatch.setattr(runtime_probe, "load_official_univtac_act_policy", loader)

    result = runtime_probe.probe_official_act_runtime(
        tmp_path / "integration_config.json", load_policy=True
    )

    assert result.artifact_valid is False
    assert result.dependency_imports_attempted is False
    assert result.dependency_imports == {}
    assert result.error_stage == "artifact_validation"
    assert result.policy_loaded is False
    assert result.passed is False
    importer.assert_not_called()
    loader.assert_not_called()


def test_dependency_failure_is_reported_without_loading_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    loader = Mock()

    def import_dependency(name: str) -> None:
        if name == "torchvision":
            raise ModuleNotFoundError("torchvision unavailable")
        return None

    monkeypatch.setattr(
        runtime_probe, "resolve_act_runtime_artifacts", Mock(return_value=runtime)
    )
    monkeypatch.setattr(runtime_probe, "_import_dependency", import_dependency)
    monkeypatch.setattr(runtime_probe, "load_official_univtac_act_policy", loader)

    result = runtime_probe.probe_official_act_runtime(
        tmp_path / "integration_config.json", load_policy=True
    )

    assert result.artifact_valid is True
    assert result.dependency_imports["torchvision"] is False
    assert result.dependency_imports_valid is False
    assert result.error_stage == "dependency_imports"
    assert result.error_type == "ModuleNotFoundError"
    assert result.policy_loaded is False
    loader.assert_not_called()


def test_script_emits_one_canonical_json_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(
        runtime_probe, "resolve_act_runtime_artifacts", Mock(return_value=runtime)
    )
    monkeypatch.setattr(
        runtime_probe, "_import_dependency", Mock(side_effect=_all_imports)
    )
    config = tmp_path / "integration_config.json"

    assert main(["--integration-config", str(config)]) == 0

    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert output.count("\n") == 1
    document = json.loads(output)
    assert output == runtime_probe.runtime_probe_json_bytes(
        runtime_probe.OfficialACTRuntimeProbeResult(
            integration_config_path=config,
            artifact_valid=True,
            dependency_imports_attempted=True,
            dependency_imports={name: True for name in document["dependency_imports"]},
            policy_load_requested=False,
            policy_loaded=False,
            policy_closed=False,
            artifact_manifest_path=runtime.manifest_path,
            task_id=runtime.manifest.task_id,
            profile=runtime.manifest.profile.value,
            device=runtime.device,
            checkpoint_sha256=runtime.manifest.checkpoint_sha256,
            config_sha256=runtime.manifest.config_sha256,
        )
    ).decode("ascii")
    assert document["closed_loop_execution_claimed"] is False
    assert document["isaac_sim_started"] is False
    assert document["policy_loaded"] is False

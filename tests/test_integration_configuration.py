"""Generated deployment configuration for both first-class models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robotactile_benchmark.cli import main
from robotactile_benchmark.integrations.act.artifacts import (
    ACTArtifactManifest,
    act_artifact_manifest_from_dict,
    act_artifact_manifest_to_dict,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    load_n0_twam_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_twam.preparation import (
    PreparedN0Artifacts,
    prepare_official_n0_artifacts,
)
from robotactile_benchmark.integrations.registry import ModelIntegrationConfig
from robotactile_benchmark.policies.n0_official import n0_training_prompt
from robotactile_benchmark.policies.univtac_official_act import OfficialACTProfile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_n0_fixture(bundle: Path) -> PreparedN0Artifacts:
    base = bundle / "base"
    checkpoint = bundle / "univtac-delta"
    for component in ("vae", "tokenizer", "text_encoder"):
        (base / component).mkdir(parents=True)
        (base / component / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "transformer").mkdir(parents=True)
    (checkpoint / "transformer/config.json").write_text(
        '{"model":"n0"}\n', encoding="utf-8"
    )
    (checkpoint / "transformer/diffusion_pytorch_model.safetensors").write_bytes(
        b"n0 checkpoint"
    )
    (checkpoint / "train_meta.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "norm").mkdir()
    task_key = "univtac_pull_out_key_rot6d_current"
    normalizer = {task_key: {"q01": [0.0] * 20, "q99": [1.0] * 20}}
    (checkpoint / "norm/pull_out_key.norm_stat_per_robot.json").write_text(
        json.dumps(normalizer), encoding="utf-8"
    )
    (checkpoint / "norm/PROMPTS.json").write_text(
        json.dumps({task_key: n0_training_prompt("pull_out_key")}), encoding="utf-8"
    )
    return prepare_official_n0_artifacts(bundle_root=bundle, task_id="pull_out_key")


def test_n0_configure_hashes_real_files_and_is_idempotent(
    tmp_path: Path, capsys: object
) -> None:
    root = tmp_path / "deployment"
    bundle = root / "artifacts/models/n0_twam"
    bundle.mkdir(parents=True)
    prepared = _write_n0_fixture(bundle)
    command = ["integrations", "configure", "--model", "n0_twam", "--root", str(root)]

    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert first == second
    config_root = bundle / "configs/pull_out_key"
    manifest = load_n0_twam_artifact_manifest(config_root / "artifact_manifest.json")
    assert manifest.checkpoint_sha256 == _sha(prepared.checkpoint_path)
    assert manifest.config_sha256 == _sha(prepared.config_path)
    assert manifest.normalizer_sha256 == _sha(prepared.normalizer_path)
    generated_config = json.loads(
        (config_root / "integration_config.json").read_text(encoding="utf-8")
    )
    assert generated_config["artifact_manifest"] == str(
        config_root / "artifact_manifest.json"
    )


def test_model_specific_configure_syntax_matches_legacy_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "deployment"
    bundle = root / "artifacts/models/n0_twam"
    bundle.mkdir(parents=True)
    _write_n0_fixture(bundle)

    assert main(["integrations", "configure", "n0-twam", "--root", str(root)]) == 0
    preferred = json.loads(capsys.readouterr().out)
    assert (
        main(["integrations", "configure", "--model", "n0_twam", "--root", str(root)])
        == 0
    )
    legacy = json.loads(capsys.readouterr().out)

    assert preferred == legacy


def test_setup_initializes_and_reports_blocked_prerequisites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "deployment"

    assert main(["setup", "--model", "n0-twam", "--root", str(root)]) == 2
    payload = json.loads(capsys.readouterr().out)

    assert payload["configuration"]["status"] == "blocked"
    assert payload["doctor"]["passed"] is False
    assert payload["evidence_level"] == "model_setup_diagnostic_only_v1"
    assert payload["live_inference_claimed"] is False
    assert payload["model"] == "n0_twam"
    assert (root / "artifacts/deployment/layout_receipt.json").is_file()


def test_n0_doctor_verifies_artifact_but_fails_closed_without_sources(
    tmp_path: Path, capsys: object
) -> None:
    root = tmp_path / "deployment"
    bundle = root / "artifacts/models/n0_twam"
    bundle.mkdir(parents=True)
    _write_n0_fixture(bundle)
    assert (
        main(["integrations", "configure", "--model", "n0_twam", "--root", str(root)])
        == 0
    )
    capsys.readouterr()  # type: ignore[attr-defined]

    assert (
        main(["integrations", "doctor", "--model", "n0_twam", "--root", str(root)]) == 2
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    checks = {item["check_id"]: item for item in payload["checks"]}

    assert checks["integration_config"]["passed"] is True
    assert checks["artifact_manifest"]["passed"] is True
    assert checks["source_n0_twam"]["passed"] is False
    assert checks["source_univtac"]["passed"] is False
    assert checks["transport"]["passed"] is True
    assert payload["live_inference_claimed"] is False


def test_act_manifest_codec_preserves_absolute_frozen_layout(tmp_path: Path) -> None:
    artifact = (tmp_path / "act").absolute()
    upstream = (tmp_path / "UniVTAC").absolute()
    manifest = ACTArtifactManifest.for_shared_root(
        task_id="pull_out_key",
        profile=OfficialACTProfile.UNIVTAC,
        artifact_root=artifact,
        upstream_root=upstream,
        checkpoint_sha256="a" * 64,
        stats_sha256="b" * 64,
        encoder_sha256="c" * 64,
    )

    restored = act_artifact_manifest_from_dict(act_artifact_manifest_to_dict(manifest))

    assert restored == manifest


def test_config_rejects_relative_manifest_escape() -> None:
    with pytest.raises(ValueError, match="escape"):
        ModelIntegrationConfig(
            schema_version="robotactile-model-integration-config-v1",
            integration_id="act",
            artifact_manifest="../outside.json",
            device="cuda:0",
            transport="in_process",
        )

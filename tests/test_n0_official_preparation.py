from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotactile_benchmark.integrations.configuration import (
    configure_n0_twam_integration,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    load_n0_twam_artifact_manifest,
    serve_task_id,
    validate_n0_twam_artifact,
)
from robotactile_benchmark.integrations.n0_twam.preparation import (
    prepare_official_n0_artifacts,
)
from robotactile_benchmark.policies.n0_official import n0_training_prompt


def _write_model_fixture(root: Path) -> None:
    base = root / "base"
    checkpoint = root / "univtac-delta"
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
    task_ids = ("pull_out_key", "lift_can", "grasp_classify")
    prompts = {}
    for task_id in task_ids:
        task = serve_task_id(task_id)
        normalizer = {task: {"q01": [0.0] * 20, "q99": [1.0] * 20}}
        (checkpoint / f"norm/{task_id}.norm_stat_per_robot.json").write_text(
            json.dumps(normalizer), encoding="utf-8"
        )
        prompts[task] = n0_training_prompt(task_id)
    (checkpoint / "norm/PROMPTS.json").write_text(json.dumps(prompts), encoding="utf-8")


def test_preparation_builds_idempotent_task_pool_and_typed_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "n0"
    root.mkdir()
    _write_model_fixture(root)
    first = prepare_official_n0_artifacts(bundle_root=root, task_id="pull_out_key")
    second = prepare_official_n0_artifacts(bundle_root=root, task_id="pull_out_key")
    lift_can = prepare_official_n0_artifacts(bundle_root=root, task_id="lift_can")
    grasp_classify = prepare_official_n0_artifacts(
        bundle_root=root, task_id="grasp_classify"
    )
    assert first == second
    assert first.serve_bundle_manifest_path != lift_can.serve_bundle_manifest_path
    assert first.serve_bundle_manifest_path.parent == first.serve_pool_root
    assert lift_can.serve_bundle_manifest_path.parent == lift_can.serve_pool_root
    assert serve_task_id("grasp_classify") == "univtac_grasp_classify_hdf5_current"
    assert grasp_classify.serve_info_path == (
        grasp_classify.serve_pool_root
        / "train/univtac_grasp_classify_hdf5_current/meta/info.json"
    )
    assert first.serve_bundle_root.joinpath("transformer").is_symlink()
    info = json.loads(first.serve_info_path.read_text(encoding="utf-8"))
    assert info["features"]["action"]["shape"] == [10]
    assert set(info["features"]) >= {
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.tactile_a",
        "observation.images.tactile_b",
    }

    generated = configure_n0_twam_integration(
        bundle_root=first.bundle_root,
        task_id="pull_out_key",
        base_root=first.base_root,
        checkpoint_root=first.checkpoint_root,
        serve_bundle_root=first.serve_bundle_root,
        serve_pool_root=first.serve_pool_root,
        checkpoint_path=first.checkpoint_path,
        model_config_path=first.config_path,
        train_meta_path=first.train_meta_path,
        normalizer_path=first.normalizer_path,
        prompt_manifest_path=first.prompt_manifest_path,
        serve_bundle_manifest_path=first.serve_bundle_manifest_path,
        serve_info_path=first.serve_info_path,
        serve_tasks_path=first.serve_tasks_path,
        manifest_path=root / "configs/pull_out_key/artifact_manifest.json",
        config_path=root / "configs/pull_out_key/integration_config.json",
        device="cuda",
    )
    loaded = load_n0_twam_artifact_manifest(generated.artifact_manifest_path)
    assert loaded.task_id == "pull_out_key"
    assert validate_n0_twam_artifact(loaded)["external_commit"] == (
        "c43a2160dd31c449d92b28eab52c0e2f09e4738a"
    )

    (first.base_root / "vae/config.json").write_text(
        '{"tampered":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="base component hash mismatch"):
        validate_n0_twam_artifact(loaded)

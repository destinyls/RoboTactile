from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HPU_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "n0_twam"
sys.path.insert(0, str(HPU_ROOT))

from hpu_training.latent.contracts import (  # noqa: E402
    TACTILE_KEYS,
    TASKS,
    VIDEO_KEYS,
    LatentSpec,
    build_work_units,
    canonical_sha256,
    load_certified_train759,
)
from hpu_training.latent.inventory import aggregate_inventory  # noqa: E402
from hpu_training.latent.vision_encoder_accel import (  # noqa: E402
    OFFICIAL_VISION_ENCODER_SHA256,
    run_official_encoder,
)
from hpu_training.latent.worker import (  # noqa: E402
    build_encoder_command,
    encoder_provenance,
)

FORMAL_TEST_NODES = ("192.0.2.107", "192.0.2.108")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _source_sha(task: str, episode_id: int) -> str:
    return hashlib.sha256(f"{task}/{episode_id}".encode()).hexdigest()


def _signed(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = canonical_sha256(result)
    return result


def _build_certified_fixture(tmp_path: Path) -> LatentSpec:
    dataset_root = tmp_path / "materialized" / "train759"
    source_episodes: list[dict[str, object]] = []
    for task in TASKS:
        for source_id in range(100):
            if source_id in {0, 1, 2, 3, 5}:
                split = "frozen"
            elif task == "grasp_classify" and source_id == 90:
                split = "quarantine"
            else:
                split = "train"
            source_episodes.append(
                {
                    "relative_path": f"{task}/clean/{source_id}.hdf5",
                    "task": task,
                    "episode_id": source_id,
                    "split": split,
                    "size_bytes": 100 + source_id,
                    "mtime_ns": source_id,
                    "sha256": _source_sha(task, source_id),
                    "usable_source_range": None,
                }
            )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "univtac_train759_absee20_v1",
        "raw_root": str(tmp_path / "raw"),
        "source_fps": 10,
        "action_per_frame": 4,
        "action_schema": "ee20_absolute_next_step",
        "quaternion_order": "wxyz",
        "gripper_source": "embodiment/joint[:,7]",
        "used_action_channel_ids": list(range(10)),
        "split_counts": {"train": 759, "frozen": 40, "quarantine": 1},
        "materialized_splits": ["train"],
        "episodes": source_episodes,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "materialized" / "source_split_manifest.json"
    _write_json(manifest_path, manifest)

    task_repos: list[dict[str, object]] = []
    total_frames = 0
    for task in TASKS:
        train_sources = [
            item
            for item in source_episodes
            if item["task"] == task and item["split"] == "train"
        ]
        repo = dataset_root / task
        _write_json(
            repo / "meta" / "info.json",
            {"fps": 10, "total_chunks": 1, "chunks_size": 1000},
        )
        episode_lines = [
            json.dumps({"episode_index": local_id, "length": 9})
            for local_id in range(len(train_sources))
        ]
        (repo / "meta" / "episodes.jsonl").write_text(
            "\n".join(episode_lines) + "\n", encoding="utf-8"
        )
        episode_map = [
            {
                "lerobot_episode_index": local_id,
                "source_episode_id": source["episode_id"],
                "source_relative_path": source["relative_path"],
                "source_sha256": source["sha256"],
                "usable_source_range": None,
                "converted_frame_count": 9,
            }
            for local_id, source in enumerate(train_sources)
        ]
        frame_count = len(train_sources) * 9
        task_receipt = _signed(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol_id": "univtac_train759_absee20_task_v1",
                "task": task,
                "repo_relative_path": f"train759/{task}",
                "physical_repo_basename": task,
                "per_repo_norm_key": task,
                "source_manifest_sha256": manifest["manifest_sha256"],
                "source_split": "train",
                "episode_count": len(train_sources),
                "frame_count": frame_count,
                "source_fps": 10,
                "action_per_frame": 4,
                "action_schema": "ee20_absolute_next_step",
                "quaternion_order": "wxyz",
                "gripper_source": "embodiment/joint[:,7]",
                "used_action_channel_ids": list(range(10)),
                "observation_keys": [*VIDEO_KEYS, *TACTILE_KEYS],
                "episode_map": episode_map,
            },
            "task_receipt_sha256",
        )
        _write_json(repo / "_robotactile_task_receipt.json", task_receipt)
        task_repos.append(
            {
                "task": task,
                "repo_relative_path": f"train759/{task}",
                "physical_repo_basename": task,
                "per_repo_norm_key": task,
                "task_receipt_relative_path": (
                    f"train759/{task}/_robotactile_task_receipt.json"
                ),
                "task_receipt_sha256": task_receipt["task_receipt_sha256"],
                "episode_count": len(train_sources),
                "frame_count": frame_count,
            }
        )
        total_frames += frame_count

    conversion_path = tmp_path / "materialized" / "train759_receipt.json"
    normalization: dict[str, object] = {}
    for name in ("norm_stat_path", "per_repo_norm_stat_path", "raw_report_path"):
        artifact = tmp_path / "materialized" / "artifacts" / f"{name}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"{name}\n", encoding="utf-8")
        normalization[name] = artifact.relative_to(tmp_path / "materialized").as_posix()
        normalization[f"{name}_sha256"] = hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
    _write_json(
        conversion_path,
        _signed(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol_id": "univtac_train759_absee20_v1",
                "dataset_path": str(dataset_root.resolve()),
                "source_manifest_relative_path": "source_split_manifest.json",
                "source_manifest_sha256": manifest["manifest_sha256"],
                "source_split_counts": {
                    "train": 759,
                    "frozen": 40,
                    "quarantine": 1,
                },
                "materialized_split": "train",
                "materialized_episode_count": 759,
                "excluded_episode_count": 41,
                "completed_task_count": 8,
                "frame_count": total_frames,
                "source_fps": 10,
                "action_per_frame": 4,
                "action_schema": "ee20_absolute_next_step",
                "quaternion_order": "wxyz",
                "gripper_source": "embodiment/joint[:,7]",
                "used_action_channel_ids": list(range(10)),
                "validation_dataset_path": None,
                "materialized_splits": ["train"],
                "task_repos": task_repos,
                "normalization": normalization,
            },
            "receipt_sha256",
        ),
    )
    official_repo = tmp_path / "official"
    model_path = tmp_path / "model"
    official_repo.mkdir()
    model_path.mkdir()
    return LatentSpec(
        dataset_root=dataset_root,
        source_manifest_path=manifest_path,
        conversion_receipt_path=conversion_path,
        official_repo=official_repo,
        model_path=model_path,
    )


def test_certified_train759_excludes_frozen_and_quarantine(tmp_path: Path) -> None:
    spec = _build_certified_fixture(tmp_path)
    records = load_certified_train759(spec)
    assert len(records) == 759
    assert all(record.source_episode_id not in {0, 1, 2, 3, 5} for record in records)
    assert not any(
        record.task == "grasp_classify" and record.source_episode_id == 90
        for record in records
    )

    receipt_path = (
        spec.dataset_root / "grasp_classify" / "_robotactile_task_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["episode_map"][0].update(
        {
            "source_episode_id": 0,
            "source_relative_path": "grasp_classify/clean/0.hdf5",
            "source_sha256": _source_sha("grasp_classify", 0),
        }
    )
    receipt.pop("task_receipt_sha256")
    receipt = _signed(receipt, "task_receipt_sha256")
    _write_json(receipt_path, receipt)
    conversion = json.loads(spec.conversion_receipt_path.read_text())
    for repo_record in conversion["task_repos"]:
        if repo_record["task"] == "grasp_classify":
            repo_record["task_receipt_sha256"] = receipt["task_receipt_sha256"]
    conversion.pop("receipt_sha256")
    _write_json(
        spec.conversion_receipt_path,
        _signed(conversion, "receipt_sha256"),
    )
    with pytest.raises(ValueError, match="refusing frozen/quarantine"):
        load_certified_train759(spec)


def test_work_plan_is_balanced_two_nodes_sixteen_explicit_units(tmp_path: Path) -> None:
    records = load_certified_train759(_build_certified_fixture(tmp_path))
    units = build_work_units(records, FORMAL_TEST_NODES)
    assert len(units) == 16
    assert {unit.worker_id for unit in units} == set(range(16))
    for node in FORMAL_TEST_NODES:
        node_units = [unit for unit in units if unit.node == node]
        assert {unit.local_device for unit in node_units} == set(range(8))
        assert sum(unit.kind == "vision" for unit in node_units) == 4
        assert sum(unit.kind == "tactile" for unit in node_units) == 4
    assert {(unit.task, unit.kind) for unit in units} == {
        (task, kind) for task in TASKS for kind in ("vision", "tactile")
    }
    assert sum(len(unit.episodes) for unit in units if unit.kind == "vision") == 759
    assert sum(len(unit.episodes) for unit in units if unit.kind == "tactile") == 759
    assert all(unit.to_json()["episode_ids"] for unit in units)


def test_official_commands_pin_true_10hz_keys_and_never_overwrite() -> None:
    common = {
        "python_path": Path("/runtime/python"),
        "official_repo": Path("/source/N0-TWAM"),
        "model_path": Path("/models/n0-twam-base"),
        "repo_path": Path("/data/train759/lift_can"),
        "episode_ids": [4, 6, 7],
    }
    vision = build_encoder_command(kind="vision", **common)
    tactile = build_encoder_command(kind="tactile", **common)
    assert vision[1].endswith("vision_encoder_accel.py")
    assert vision[2:4] == [
        "--official-script",
        "/source/N0-TWAM/script/encode_lerobot_n0_latents.py",
    ]
    assert tactile[1] == "/source/N0-TWAM/script/encode_tactile_latent.py"
    assert vision[vision.index("--target-fps") + 1] == "10"
    assert tactile[tactile.index("--target-fps") + 1] == "10"
    assert vision[vision.index("--video-keys") + 1 : -4] == list(VIDEO_KEYS)
    tactile_start = tactile.index("--tactile-keys") + 1
    assert tactile[tactile_start : tactile_start + 2] == list(TACTILE_KEYS)
    assert tactile[tactile.index("--mode") + 1] == "both"
    assert tactile[tactile.index("--local-mode") + 1] == "current"
    assert "--overwrite" not in vision and "--overwrite" not in tactile
    for episode_id in common["episode_ids"]:
        assert str(episode_id) in vision and str(episode_id) in tactile


def test_vision_wrapper_changes_only_official_umt5_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "wrapper-output.json"
    monkeypatch.setenv("ROBOTACTILE_WRAPPER_TEST_OUTPUT", str(output_path))
    source = """
import json
import os
import sys
from pathlib import Path

def load_text_encoder(path, torch_dtype, torch_device):
    return {
        "path": path,
        "torch_dtype": torch_dtype,
        "torch_device": torch_device,
    }

def main():
    loaded = load_text_encoder(
        "official-text-encoder", torch_dtype="bf16", torch_device="cpu"
    )
    Path(os.environ["ROBOTACTILE_WRAPPER_TEST_OUTPUT"]).write_text(
        json.dumps({"argv": sys.argv[1:], "loaded": loaded}),
        encoding="utf-8",
    )
"""
    official_script = tmp_path / "encode_lerobot_n0_latents.py"
    official_script.write_text(source, encoding="utf-8")
    expected_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    official_arguments = [
        "--dataset-root",
        "/data/train759/lift_can",
        "--device",
        "cuda:0",
        "--dtype",
        "bf16",
        "--prompt",
        "lift the can",
    ]

    run_official_encoder(
        official_script=official_script,
        official_arguments=official_arguments,
        expected_sha256=expected_sha256,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["argv"] == official_arguments
    assert result["loaded"] == {
        "path": "official-text-encoder",
        "torch_dtype": "bf16",
        "torch_device": "cuda:0",
    }
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_official_encoder(
            official_script=official_script,
            official_arguments=official_arguments,
            expected_sha256="0" * 64,
        )


def test_acceleration_provenance_and_persistent_cache_contract() -> None:
    official_repo = Path("/source/N0-TWAM")
    vision = encoder_provenance(kind="vision", official_repo=official_repo)
    tactile = encoder_provenance(kind="tactile", official_repo=official_repo)
    assert vision["execution"] == "robotactile_official_main_device_wrapper"
    assert vision["official_script_sha256"] == OFFICIAL_VISION_ENCODER_SHA256
    assert vision["override"] == {
        "symbol": "load_text_encoder",
        "argument": "torch_device",
        "official_value": "cpu",
        "effective_value": "cuda:0",
    }
    assert tactile["execution"] == "pinned_official_cli"
    entrypoint = (
        HPU_ROOT / "hpu_training" / "latent" / "container_worker_entrypoint.sh"
    ).read_text(encoding="utf-8")
    assert "TORCHINDUCTOR_COMPILE_THREADS=1" in entrypoint
    assert 'TORCHINDUCTOR_CACHE_DIR="$RESOLVED_CACHE_ROOT/torchinductor"' in entrypoint
    assert 'TRITON_CACHE_DIR="$RESOLVED_CACHE_ROOT/triton"' in entrypoint
    assert 'XDG_CACHE_HOME="$RESOLVED_CACHE_ROOT/xdg"' in entrypoint
    assert '[[ "$RESOLVED_CACHE_ROOT" == /mnt/data/* ]]' in entrypoint


def _fake_output(label: str, episode_id: int) -> dict[str, object]:
    return {
        "path": f"/latents/{label}/{episode_id}.pth",
        "size_bytes": 100,
        "mtime_ns": 1,
        "latent_num_frames": 3,
        "latent_height": 8,
        "latent_width": 8,
        "latent_channels": 48,
        "frame_count": 9,
        "frame_ids_sha256": "f" * 64,
        "fps": 10,
        "ori_fps": 10,
    }


def test_final_inventory_covers_all_759_and_six_outputs_each(tmp_path: Path) -> None:
    spec = _build_certified_fixture(tmp_path)
    units = build_work_units(load_certified_train759(spec), FORMAL_TEST_NODES)
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    for unit in units:
        inventory = []
        for episode in unit.episodes:
            labels = (
                VIDEO_KEYS
                if unit.kind == "vision"
                else tuple(
                    f"{mode}:{key}"
                    for mode in ("global", "local")
                    for key in TACTILE_KEYS
                )
            )
            inventory.append(
                {
                    "lerobot_episode_index": episode.lerobot_episode_index,
                    "source_episode_id": episode.source_episode_id,
                    "source_relative_path": episode.source_relative_path,
                    "source_sha256": episode.source_sha256,
                    "length": episode.length,
                    "latent_num_frames": 3,
                    "frame_ids_sha256": "f" * 64,
                    "outputs": {
                        label: _fake_output(label, episode.lerobot_episode_index)
                        for label in labels
                    },
                }
            )
        _write_json(
            receipt_dir / f"worker-{unit.worker_id:02d}.json",
            {
                "status": "complete",
                "worker_id": unit.worker_id,
                "node": unit.node,
                "local_device": unit.local_device,
                "task": unit.task,
                "kind": unit.kind,
                "target_fps": 10,
                "episode_ids": [
                    episode.lerobot_episode_index for episode in unit.episodes
                ],
                "final_missing_episode_ids": [],
                "inventory": inventory,
            },
        )
    final = aggregate_inventory(
        units=units,
        receipt_dir=receipt_dir,
        source_manifest_path=spec.source_manifest_path,
        conversion_receipt_path=spec.conversion_receipt_path,
        official_commit="cdd87b6a141667123ad2c25f452478afdb71e287",
        encoder_sha256={"vision": "a" * 64, "tactile": "b" * 64},
    )
    assert final["status"] == "complete"
    assert final["episode_count"] == 759
    assert final["vision_file_count"] == 1518
    assert final["tactile_global_file_count"] == 1518
    assert final["tactile_local_file_count"] == 1518
    assert final["source_manifest_path"] == str(spec.source_manifest_path)
    assert final["conversion_receipt_path"] == str(spec.conversion_receipt_path)
    assert len(final["episodes"]) == 759

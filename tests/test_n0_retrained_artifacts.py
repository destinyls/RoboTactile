"""Small synthetic artifacts test independent identity and namespace isolation."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotactile_benchmark.backends.univtac_contracts import load_univtac_task_registry
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.n0_twam.retrained import (
    RUNTIME_CONTENT_IDENTITY_SCHEMA,
    TRAINING_CONTRACT,
    file_sha256,
    load_retrained,
    prepare_retrained,
    read_object,
    server_overrides,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def inputs(tmp_path):
    checkpoint, base, source, metadata = [
        tmp_path / name for name in ("checkpoint", "base", "source", "metadata")
    ]
    write(checkpoint / "train_meta.json", {**TRAINING_CONTRACT, "step": 10000})
    write(checkpoint / "transformer/config.json", {"is_mot": True})
    (checkpoint / "transformer/diffusion_pytorch_model.safetensors").write_bytes(
        b"synthetic-test-not-model"
    )
    for component in ("vae", "text_encoder", "tokenizer"):
        write(base / component / "config.json", {"fixture": True})
    (base / "empty_emb.pt").write_bytes(b"synthetic-test-not-embedding")
    (source / "n0_twam").mkdir(parents=True)
    (source / "n0_twam/n0_twam_server.py").write_text("# synthetic source\n")
    norms = {}
    for task in load_univtac_task_registry().tasks:
        norms[task.task_id] = {"q01": [-2.0] * 20, "q99": [3.0] * 20}
        write(
            metadata / task.task_id / "meta/info.json",
            {
                "fps": 10,
                "features": {
                    "action": {"shape": [20]},
                    **{
                        k: {"dtype": "video"}
                        for k in TRAINING_CONTRACT["obs_cam_keys"]
                        + TRAINING_CONTRACT["tactile_keys"]
                    },
                },
            },
        )
        write(
            metadata / task.task_id / "meta/tasks.jsonl",
            {"task": f"Actual trained {task.task_id}"},
        )
    norm_path = tmp_path / "norm.json"
    write(norm_path, norms)
    return dict(
        checkpoint=checkpoint,
        base=base,
        source_root=source,
        per_repo_norm=norm_path,
        metadata_root=metadata,
        output=tmp_path / "prepared",
        source_commit="cdd87b6",
    )


def test_prepare_is_independent_no_clobber_and_task_exact(inputs):
    path = prepare_retrained(**inputs)
    artifact = load_retrained(path)
    assert artifact["source"]["verification"] == "declared_unverified"
    assert artifact["action_hz"] == 10 and artifact["action_per_frame"] == 4
    assert len(artifact["tasks"]) == 8
    assert (inputs["output"] / "serve-bundle/transformer").resolve() == inputs[
        "checkpoint"
    ] / "transformer"
    for task, entry in artifact["tasks"].items():
        assert entry["prompt"] == f"Actual trained {task}"
        override = server_overrides(artifact, task)
        assert entry["config_sha256"] == canonical_hash(override)
        assert override["action_delta_mode"] == "none"
        assert override["host"] == "127.0.0.1"
        assert override["action_per_frame"] == 4
        assert override["frame_chunk_size"] == 2
        assert override["tactile_resize"] == 128
        assert override["guidance_scale"] == override["action_guidance_scale"] == 1
    with pytest.raises(FileExistsError):
        prepare_retrained(**inputs)


@pytest.mark.parametrize("target", ["meta", "weight", "source", "base"])
def test_changed_inputs_are_rejected_without_weight_rehash(inputs, target, monkeypatch):
    path = prepare_retrained(**inputs)
    destinations = {
        "meta": inputs["checkpoint"] / "train_meta.json",
        "weight": inputs["checkpoint"]
        / "transformer/diffusion_pytorch_model.safetensors",
        "source": inputs["source_root"] / "n0_twam/n0_twam_server.py",
        "base": inputs["base"] / "vae/config.json",
    }
    changed = destinations[target]
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        load_retrained(path)


def test_preverified_content_identity_allows_metadata_only_drift(inputs):
    path = prepare_retrained(**inputs)
    artifact = read_object(path)
    artifact.pop("artifact_sha256")
    checkpoint = Path(artifact["checkpoint_path"]).resolve()
    base_paths = [Path(name).resolve() for name in artifact["base_component_stamps"]]
    artifact["runtime_content_identity"] = {
        "schema": RUNTIME_CONTENT_IDENTITY_SCHEMA,
        "verification": "sha256_verified_prelaunch",
        "checkpoint": {
            "path": str(checkpoint),
            "size": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
        },
        "base_components": {
            str(member): {
                "path": str(member),
                "size": member.stat().st_size,
                "sha256": file_sha256(member),
            }
            for member in base_paths
        },
    }
    artifact["artifact_sha256"] = canonical_hash(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")

    for member in (checkpoint, base_paths[0]):
        stat = member.stat()
        os.utime(member, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    loaded = load_retrained(path)
    assert loaded["runtime_content_identity"]["verification"] == (
        "sha256_verified_prelaunch"
    )


@pytest.mark.parametrize(
    "key,value", [("action_per_frame", 12), ("action_delta_mode", "pi05_delta")]
)
def test_training_contract_mismatch_rejected(inputs, key, value):
    path = inputs["checkpoint"] / "train_meta.json"
    meta = read_object(path)
    meta[key] = value
    write(path, meta)
    with pytest.raises(ValueError, match="training contract"):
        prepare_retrained(**inputs)
    assert not inputs["output"].exists()


def test_snapshot_receipt_and_actual_server_namespace(inputs, tmp_path):
    first = prepare_retrained(**inputs)
    artifact = load_retrained(first)
    receipt = tmp_path / "receipt.json"
    write(
        receipt,
        {
            "source_commit": "cdd87b6" + "0" * 33,
            "source_tree_sha256": artifact["source_tree_sha256"],
        },
    )
    second_inputs = {**inputs, "output": tmp_path / "second", "source_receipt": receipt}
    second = load_retrained(prepare_retrained(**second_inputs))
    assert second["source"]["verification"] == "matched_source_snapshot_receipt"
    script = Path(__file__).resolve().parents[1] / "scripts/n0_twam/serve_retrained.py"
    spec = importlib.util.spec_from_file_location("test_retrained_server", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {"nested": {"preserved": [1, 2]}}
    before = copy.deepcopy(original)
    server = SimpleNamespace(TWAM_CONFIGS={"posttrain_server": original})
    name = module.configure_server(
        server, artifact, "pull_out_key", tmp_path / "out", 29601, True
    )
    assert server.TWAM_CONFIGS["posttrain_server"] == before
    assert server.TWAM_CONFIGS[name]["prompt"] == "Actual trained pull_out_key"
    assert server.TWAM_CONFIGS[name]["enable_offload"] is True
    assert server.TWAM_CONFIGS[name]["action_per_frame"] == 4

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "scripts" / "n0_twam" / "hpu_training"
)
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hpu_training import (  # noqa: E402
    contract,
    launch_profiles,
    remote_execution,
    render_config,
)

sys.path.insert(0, str(SCRIPT_DIR / "runtime_shims"))
from flex25_compat import (  # noqa: E402
    _create_grouped_self_mask,
    _get_mask_mod,
    uses_vendor_torch25,
)


def _training_payload(root: Path) -> dict[str, object]:
    return {
        "dataset_path": str(root / "data" / "train759"),
        "base_model_path": str(root / "models" / "base"),
        "released_checkpoint_path": str(root / "models" / "base"),
        "empty_embedding_path": str(root / "models" / "base" / "empty_emb.pt"),
        "norm_stat_path": str(root / "artifacts" / "norm.json"),
        "per_repo_norm_stat_path": str(root / "artifacts" / "norm-per-repo.json"),
        "latent_inventory_path": str(root / "artifacts" / "latent_inventory.json"),
        "prompt": "use vision and tactile feedback",
        "max_latent_frames": 5,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_training_spec_rejects_validation_and_non_train759(tmp_path: Path) -> None:
    payload = _training_payload(tmp_path)
    payload["val_dataset_path"] = str(tmp_path / "frozen40")
    spec_path = tmp_path / "with-validation.json"
    _write_json(spec_path, payload)
    with pytest.raises(ValueError, match="unsupported training spec keys"):
        contract.load_training_spec(spec_path)

    payload.pop("val_dataset_path")
    payload["dataset_path"] = str(tmp_path / "data" / "frozen40")
    _write_json(spec_path, payload)
    with pytest.raises(ValueError, match="train759"):
        contract.load_training_spec(spec_path)

    payload["dataset_path"] = str(tmp_path / "data" / "train759")
    payload["released_checkpoint_path"] = str(
        tmp_path / "models" / "n0-twam-univtac-absee"
    )
    _write_json(spec_path, payload)
    with pytest.raises(ValueError, match="official pretrained base"):
        contract.load_training_spec(spec_path)


def test_formal_example_is_two_nodes_eight_processes_each() -> None:
    cluster = contract.load_cluster_spec(SCRIPT_DIR / "cluster.example.json")
    profile = launch_profiles.resolve_launch_profile(
        mode="formal",
        cluster=cluster,
        training_steps=10_000,
        smoke_steps=None,
    )
    assert cluster.nodes == ("192.0.2.107", "192.0.2.108")
    assert profile.world_size == 16
    payload = remote_execution.encode_payload(
        mode="preflight",
        rank=0,
        cluster=cluster,
        profile=profile,
        repo=Path("/mnt/data/task/source/N0-TWAM"),
        save_root=Path("/mnt/data/task/runs/example/model"),
        latent_inventory_path=Path("/mnt/data/task/latents/final_inventory.json"),
        latent_inventory_sha256="b" * 64,
        config_sha256="a" * 64,
        run_id="test-2n16",
    )
    decoded = json.loads(base64.b64decode(payload))
    assert decoded["launch_mode"] == "formal"
    assert decoded["processes_per_node"] == 8
    assert decoded["expected_num_steps"] == 10_000
    assert decoded["fsdp_topology"] == "global_shard"
    assert decoded["fsdp_shard_size"] == 16
    assert len(decoded["nodes"]) == 2
    assert decoded["container_entrypoint"].endswith("rank_entrypoint.sh")
    assert decoded["entrypoint_kind"] == "bash"
    command = remote_execution.ssh_command(cluster, cluster.nodes[0], payload)
    assert "BatchMode=yes" in command
    assert not any("password" in item.lower() for item in command)


def test_launch_profiles_cover_single_distributed_and_formal_modes() -> None:
    cluster = contract.load_cluster_spec(SCRIPT_DIR / "cluster.example.json")
    single = launch_profiles.resolve_launch_profile(
        mode="single-card-smoke",
        cluster=cluster,
        training_steps=10_000,
        smoke_steps=None,
    )
    assert single.nodes == (cluster.master_addr,)
    assert single.world_size == 1
    assert single.num_steps == 1
    assert single.visible_devices == "0"
    assert single.fsdp_topology == "global_shard"

    distributed = launch_profiles.resolve_launch_profile(
        mode="distributed-smoke",
        cluster=cluster,
        training_steps=10_000,
        smoke_steps=3,
    )
    assert distributed.nodes == cluster.nodes
    assert distributed.world_size == 16
    assert distributed.num_steps == 3
    assert distributed.fsdp_topology == "global_shard"
    assert distributed.fsdp_shard_size == 16

    formal = launch_profiles.resolve_launch_profile(
        mode="formal",
        cluster=cluster,
        training_steps=10_000,
        smoke_steps=None,
    )
    assert formal.world_size == 16
    assert formal.num_steps == 10_000

    with pytest.raises(ValueError, match="between 1 and 20"):
        launch_profiles.resolve_launch_profile(
            mode="distributed-smoke",
            cluster=cluster,
            training_steps=10_000,
            smoke_steps=21,
        )
    with pytest.raises(ValueError, match="only valid"):
        launch_profiles.resolve_launch_profile(
            mode="formal",
            cluster=cluster,
            training_steps=10_000,
            smoke_steps=1,
        )


def test_rank_entrypoint_enforces_smoke_and_temporal_contracts() -> None:
    source = (SCRIPT_DIR / "rank_entrypoint.sh").read_text(encoding="utf-8")
    assert 'N0_FSDP_TOPOLOGY="$fsdp_topology"' in source
    assert 'N0_FSDP_SHARD_SIZE="$fsdp_shard_size"' in source
    assert "single-card-smoke requires exactly one optimizer step" in source
    assert "distributed-smoke requires 2 nodes x 8 processes" in source
    assert "formal training requires 2 nodes x 8 processes" in source
    assert "right - left != 1" in source
    assert 'payload.get("fps") != 10' in source
    assert 'cfg.num_steps != int(os.environ["N0_EXPECTED_NUM_STEPS"])' in source
    assert "flex_mask_compat=" in source
    assert "-m n0_train_compat" in source
    compat_source = (SCRIPT_DIR / "runtime_shims" / "n0_train_compat.py").read_text(
        encoding="utf-8"
    )
    assert "distributed.barrier()" in compat_source
    assert "ROBOTACTILE_TRAIN_EXIT_OK" in compat_source
    assert "os._exit(0)" in compat_source
    assert "from models.model import FlexAttnFunc" in source
    assert "attention_forward_backward=grouped_flash" in source
    assert "ROBOTACTILE_GROUPED_FLASH_MAX_QUERY_TOKENS=32768" in source
    assert "N0_FLEX_ATTENTION_BACKEND=grouped_flash_attn" in source
    assert "N0_MOT_CROSS_ATTENTION_BACKEND=flash_attn" in source
    assert "TORCHINDUCTOR_COMPILE_THREADS=1" in source
    assert "PYTORCH_HIP_ALLOC_CONF" not in source
    assert "runtime/node-$node_rank" not in source
    assert 'runtime_work_dir="$runtime_root/node-$node_rank"' in source


def test_flex_mask_shim_is_restricted_to_vendor_torch25() -> None:
    assert uses_vendor_torch25("2.5.1+vendor", "6.2.41134")
    assert not uses_vendor_torch25("2.5.1+cpu", None)
    assert not uses_vendor_torch25("2.9.0+vendor", "6.2.41134")


def test_grouped_mask_shim_preserves_official_boolean_semantics() -> None:
    torch = pytest.importorskip("torch")

    seq_ids = torch.tensor([0, 0, 0, 0, 1, 1])
    frame_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    noise_ids = torch.tensor([0, 0, 1, 1, 0, 1])
    modality_ids = torch.zeros_like(seq_ids)
    grouped_mask = _create_grouped_self_mask(
        _get_mask_mod(
            seq_ids,
            frame_ids,
            noise_ids,
            modality_ids,
            window_size=2,
        )
    )
    actual_edges = {
        (int(query_index), int(key_value_index))
        for query_indices, key_value_indices in grouped_mask.groups
        for query_index in query_indices
        for key_value_index in key_value_indices
    }

    for query_index in range(seq_ids.numel()):
        for key_value_index in range(seq_ids.numel()):
            same_seq = (
                (seq_ids[query_index] == seq_ids[key_value_index])
                and seq_ids[query_index] >= 0
                and seq_ids[key_value_index] >= 0
            )
            clean_to_clean = (
                noise_ids[query_index] == 1
                and noise_ids[key_value_index] == 1
                and frame_ids[key_value_index] <= frame_ids[query_index]
            )
            noise_to_clean = (
                noise_ids[query_index] == 0
                and noise_ids[key_value_index] == 1
                and frame_ids[key_value_index] < frame_ids[query_index]
            )
            noise_to_noise = (
                noise_ids[query_index] == 0
                and noise_ids[key_value_index] == 0
                and frame_ids[key_value_index] == frame_ids[query_index]
            )
            in_window = (frame_ids[query_index] - frame_ids[key_value_index]).abs() <= 2
            expected = bool(
                same_seq
                and (clean_to_clean or noise_to_clean or noise_to_noise)
                and in_window
            )
            assert ((query_index, key_value_index) in actual_edges) is expected


def test_cluster_is_explicit_and_runtime_recipe_is_source_pinned(
    tmp_path: Path,
) -> None:
    missing_nodes = tmp_path / "cluster.json"
    _write_json(
        missing_nodes,
        {
            "ssh_user": "root",
            "ssh_port": 36000,
            "master_addr": "192.0.2.107",
            "master_port": 29620,
            "processes_per_node": 8,
        },
    )
    with pytest.raises(ValueError, match="explicitly contain exactly two"):
        contract.load_cluster_spec(missing_nodes)

    runtime = (SCRIPT_DIR / "docker_runtime.sh").read_text(encoding="utf-8")
    assert contract.PROVEN_DOCKER_IMAGE_ID in runtime
    assert contract.PROVEN_RCCL_PLUGIN_SHA256 in runtime
    assert "docker container inspect" in runtime
    assert "docker create" in runtime
    assert '--entrypoint ""' in runtime
    assert "docker rm -f" not in runtime
    committed = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SCRIPT_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    assert "10.232." not in committed


def test_renderer_hardcodes_train_only_vision_tactile_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _training_payload(tmp_path)
    spec_path = tmp_path / "training.json"
    _write_json(spec_path, payload)
    norm_path = Path(str(payload["norm_stat_path"]))
    _write_json(norm_path, {"q01": [0.0] * 20, "q99": [1.0] * 20})
    per_repo_path = Path(str(payload["per_repo_norm_stat_path"]))
    _write_json(
        per_repo_path,
        {task: {"q01": [0.0] * 20, "q99": [1.0] * 20} for task in render_config.TASKS},
    )
    spec = contract.load_training_spec(spec_path)
    repo = tmp_path / "N0-TWAM"
    target = repo / contract.ALLOWED_OFFICIAL_CHANGE
    target.parent.mkdir(parents=True)
    target.write_text("official placeholder\n", encoding="utf-8")
    monkeypatch.setattr(
        render_config,
        "verify_official_checkout",
        lambda _repo: {"commit": contract.OFFICIAL_COMMIT},
    )

    receipt = render_config.render_posttrain_config(
        repo=repo,
        spec=spec,
        save_root=tmp_path / "runs" / "model",
    )
    source = target.read_text(encoding="utf-8")
    compile(source, str(target), "exec")
    assert "cfg.val_dataset_path = None" in source
    assert "cfg.tactile_optional = False" in source
    assert "cfg.synthetic_tactile_data = False" in source
    assert "cfg.use_local_tactile = True" in source
    assert "cfg.tactile_global_zero = False" in source
    assert "cfg.tactile_cfg_prob = 0.0" in source
    assert "cfg.cfg_prob = 0.0" in source
    assert "cfg.noisy_cond_prob_tactile = 0.0" in source
    assert "cfg.tactile_diffusion_loss_weight = 1.0" in source
    assert 'cfg.max_latent_frames = _VALUES["max_latent_frames"]' in source
    assert "cfg.action_dim = 20" in source
    assert 'cfg.action_delta_mode = "none"' in source
    assert "cfg.pi05_action_horizon = 4" in source
    assert "cfg.action_per_frame = 4" in source
    assert "cfg.action_per_frame = 12" not in source
    assert receipt["vision_tactile_policy"] == "always_on"
    assert receipt["source_fps"] == 10
    assert receipt["latent_frame_stride"] == 1
    assert receipt["action_per_frame"] == 4
    assert receipt["max_latent_frames"] == 5
    assert receipt["num_steps"] == spec.num_steps


def test_official_checkout_allows_only_generated_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "N0-TWAM"
    config_path = repo / contract.ALLOWED_OFFICIAL_CHANGE
    config_path.parent.mkdir(parents=True)
    config_path.write_text("official\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            contract.OFFICIAL_REPOSITORY_URL,
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(contract, "OFFICIAL_COMMIT", head)

    config_path.write_text("generated\n", encoding="utf-8")
    contract.verify_official_checkout(repo)
    (repo / "train.py").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden changes"):
        contract.verify_official_checkout(repo)

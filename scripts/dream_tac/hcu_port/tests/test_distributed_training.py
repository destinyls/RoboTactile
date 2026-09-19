from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scripts.dream_tac.hcu_port.distributed_remote import ssh_command
from scripts.dream_tac.hcu_port.distributed_request import (
    TASKS,
    DistributedTrainingRequest,
)
from scripts.dream_tac.hcu_port.distributed_runtime import (
    build_rank_payload,
    build_resolved_experiment,
    training_overrides,
)
from scripts.dream_tac.hcu_port.optimizer_step_request import (
    HcuOptimizerStepRequest,
)

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "configs/experiments/dream_tac/hcu_distributed_2node.example.json"


def _distributed_payload() -> dict[str, object]:
    payload = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _source_request() -> HcuOptimizerStepRequest:
    return HcuOptimizerStepRequest.from_dict(
        {
            "schema_version": "robotactile-dream-tac-hcu-optimizer-step-request-v1",
            "accelerator_contract": "hygon_hcu_hip_single_device_v1",
            "dream_tac_commit": "14bab51d6862fd07124745c55cd395ea5caa9fd3",
            "donor_experiment": "cosmos_predict2_2b_480p_franka_cut_banana_20260321",
            "phase": "p2_hcu_optimizer_step",
            "run_name": "source-v11",
            "dream_tac_root": "/mnt/data/task/source/Dream-Tac-HCU",
            "python_executable": "/opt/runtime/bin/python",
            "hcu_environment_script": {
                "path": "/opt/hyhal/env.sh",
                "sha256": "1" * 64,
            },
            "runtime_pythonpath_roots": ["/mnt/data/task/runtime/site"],
            "materialization_root": "/mnt/data/task/data/dream-tac-train759",
            "output_root": "/mnt/data/task/runs/one-step",
            "source_manifest_sha256": "2" * 64,
            "materialization_receipt_sha256": "3" * 64,
            "t5_cache_sha256": "4" * 64,
            "base_checkpoint": {
                "path": "/mnt/data/task/base/model.pt",
                "sha256": "5" * 64,
            },
            "base_dcp_root": "/mnt/data/task/base-dcp/iter_000000000",
            "base_dcp_receipt": {
                "path": "/mnt/data/task/base-dcp/receipt.json",
                "sha256": "6" * 64,
            },
            "tokenizer_checkpoint": {
                "path": "/mnt/data/task/base/tokenizer.pth",
                "sha256": "7" * 64,
            },
            "overlay_receipt": {
                "path": "/mnt/data/task/receipts/overlay.json",
                "sha256": "8" * 64,
            },
            "casa_receipt": {
                "path": "/mnt/data/task/receipts/casa.json",
                "sha256": "9" * 64,
            },
            "hcu_device": 0,
            "nproc_per_node": 1,
            "master_port": 12341,
            "max_iter": 1,
            "save_iter": 1,
            "batch_size": 1,
            "num_workers": 0,
            "resume_checkpoint": None,
        }
    )


def test_formal_request_is_exact_two_node_train759_contract() -> None:
    request = DistributedTrainingRequest.from_dict(_distributed_payload())
    assert request.world_size == 16
    assert request.nproc_per_node == 8
    assert request.fsdp_shard_size == 16
    assert request.max_iter == 20_000
    assert request.effective_global_batch == 16
    assert request.dream_tac_host_root.is_absolute()
    assert request.runtime_host_root.is_absolute()
    assert request.franka_dataset_sha256 == (
        "937fc592c86dfef512cc048d622daba8edcd98b7df1b4ca853c6819a73942b99"
    )
    assert tuple(request.to_dict()["tasks"]) == TASKS


@pytest.mark.parametrize("steps", [1, 2, 20])
def test_distributed_smoke_is_bounded(steps: int) -> None:
    payload = _distributed_payload()
    payload.update(launch_mode="distributed-smoke", max_iter=steps, save_iter=1)
    assert DistributedTrainingRequest.from_dict(payload).max_iter == steps


def test_request_rejects_wrong_topology_or_unbounded_profile() -> None:
    payload = _distributed_payload()
    payload["fsdp_shard_size"] = 8
    with pytest.raises(ValueError, match="fsdp_shard_size"):
        DistributedTrainingRequest.from_dict(payload)
    payload = _distributed_payload()
    payload.update(launch_mode="distributed-smoke", max_iter=21)
    with pytest.raises(ValueError, match="distributed-smoke"):
        DistributedTrainingRequest.from_dict(payload)
    payload = _distributed_payload()
    payload["master_addr"] = payload["nodes"][1]
    with pytest.raises(ValueError, match=r"nodes\[0\]"):
        DistributedTrainingRequest.from_dict(payload)


def test_overrides_use_base_dcp_fresh_optimizer_and_hsdp() -> None:
    request = DistributedTrainingRequest.from_dict(_distributed_payload())
    overrides = training_overrides(request, _source_request())
    assert "model.config.fsdp_shard_size=16" in overrides
    assert "trainer.distributed_parallelism=fsdp" in overrides
    assert "checkpoint.dcp_async_mode_enabled=false" in overrides
    assert "checkpoint.load_training_state=false" in overrides
    assert "checkpoint.strict_resume=false" in overrides
    assert "trainer.grad_accum_iter=1" in overrides
    assert all("iter_000000001" not in value for value in overrides)


def test_rank_payload_and_resolved_receipt_cover_all_tasks() -> None:
    request_payload = _distributed_payload()
    request_payload["robotactile_root"] = str(ROOT)
    request = DistributedTrainingRequest.from_dict(request_payload)
    source = _source_request()
    encoded = build_rank_payload(request, source, node_rank=1, mode="train")
    payload = json.loads(base64.b64decode(encoded, validate=True))
    assert payload["node_rank"] == 1
    assert payload["world_size"] == 16
    assert payload["fsdp_shard_size"] == 16
    assert payload["collective_probe"]["path"].endswith(
        "distributed_collective_probe.py"
    )
    assert payload["franka_dataset"]["sha256"] == request.franka_dataset_sha256
    assert payload["franka_dataset"]["path"].endswith(
        "cosmos_policy/datasets/franka_dataset.py"
    )
    assert payload["fsdp_sources"]["mesh_mode"] == "global_1d_shard_v1"
    assert payload["fsdp_sources"]["fsdp_helper"]["sha256"] == "4" * 64
    assert payload["fsdp_sources"]["dtensor_helper"]["sha256"] == "5" * 64
    resolved = build_resolved_experiment(request, source)
    assert resolved["task_count"] == 8
    assert resolved["tasks"] == list(TASKS)
    assert resolved["distributed"]["effective_global_batch"] == 16
    assert resolved["distributed"]["fsdp_topology"] == "global_1d_16way_shard"
    assert resolved["franka_dataset"]["loading"] == "lazy_random_access_frames_v1"


def test_ssh_command_is_argv_only_and_uses_request_nodes() -> None:
    request = DistributedTrainingRequest.from_dict(_distributed_payload())
    command = ssh_command(request, request.nodes[1], "encoded-payload")
    assert command[-5:] == ("root@192.0.2.108", "bash", "-s", "--", "encoded-payload")
    assert "StrictHostKeyChecking=yes" in command

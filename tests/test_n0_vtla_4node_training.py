from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.n0_vtla.hpu_training import distributed_collective_probe as probe
from scripts.n0_vtla.hpu_training import launch_formal_mixed8_4node as launcher


def _bindings(digest: str = "a" * 64) -> dict[str, str]:
    return {name: digest for name in launcher._SOURCE_NAMES}


def _request(
    *,
    nodes: tuple[str, ...] = (
        "192.0.2.10",
        "192.0.2.11",
        "192.0.2.12",
        "192.0.2.13",
    ),
    master_addr: str = "192.0.2.10",
) -> launcher.LaunchRequest:
    return launcher.LaunchRequest.create(
        project_root=Path("/mnt/data/task/n0-vtla-test"),
        run_id="mixed8-four-node-test",
        nodes=nodes,
        ssh_user="root",
        ssh_port=36000,
        master_addr=master_addr,
        master_port=29640,
        source_bindings=_bindings(),
    )


def test_request_requires_exact_unique_4x8_topology() -> None:
    request = _request()

    assert request.nodes[0] == request.master_addr
    assert request.to_dict()["nnodes"] == 4
    assert request.to_dict()["nproc_per_node"] == 8
    assert request.to_dict()["world_size"] == 32
    assert request.to_dict()["global_batch_size"] == 64
    assert request.to_dict()["gradient_accumulation_steps"] == 2

    with pytest.raises(ValueError, match="exactly four"):
        _request(nodes=("192.0.2.10", "192.0.2.11", "192.0.2.12"))
    with pytest.raises(ValueError, match="unique"):
        _request(nodes=("192.0.2.10", "192.0.2.11", "192.0.2.12", "192.0.2.12"))
    with pytest.raises(ValueError, match=r"nodes\[0\]"):
        _request(master_addr="192.0.2.11")


def test_request_round_trip_keeps_exact_fresh_base_path() -> None:
    request = _request()
    document = {**request.to_dict(), "started_at_utc": "2026-09-02T00:00:00Z"}

    assert launcher.LaunchRequest.from_dict(document) == request
    base = dict(document["base_checkpoint"])
    base["path"] = "/mnt/data/task/old-checkpoint"
    document["base_checkpoint"] = base
    with pytest.raises(ValueError, match="base checkpoint path"):
        launcher.LaunchRequest.from_dict(document)


def test_payload_binds_exact_master_fresh_base_request_and_sources() -> None:
    request = _request()
    encoded = launcher._encode_payload(
        request,
        mode="train",
        node_rank=3,
        request_sha256="b" * 64,
    )
    payload = json.loads(base64.b64decode(encoded, validate=True))

    assert payload["node_rank"] == 3
    assert payload["node_addr"] == "192.0.2.13"
    assert payload["nnodes"] == 4
    assert payload["nproc_per_node"] == 8
    assert payload["world_size"] == 32
    assert payload["master_addr"] == "192.0.2.10"
    assert payload["master_port"] == 29640
    assert payload["fresh_base"] is True
    assert payload["launch_request"]["sha256"] == "b" * 64
    assert payload["collective_probe"]["sha256"] == "a" * 64
    assert payload["rank_entrypoint"]["sha256"] == "a" * 64
    assert payload["distributed_entrypoint"]["sha256"] == "a" * 64


def test_rank_adapter_exports_exact_parent_entrypoint_contract() -> None:
    source = Path(
        "scripts/n0_vtla/hpu_training/distributed_rank_entrypoint.sh"
    ).read_text(encoding="utf-8")

    assert "ROBOTACTILE_N0_VTLA_DISTRIBUTED_MODE=four_node" in source
    assert 'ROBOTACTILE_N0_VTLA_NNODES="$nnodes"' in source
    assert 'ROBOTACTILE_N0_VTLA_NODE_RANK="$node_rank"' in source
    assert 'ROBOTACTILE_N0_VTLA_MASTER_ADDR="$master_addr"' in source
    assert 'ROBOTACTILE_N0_VTLA_MASTER_PORT="$master_port"' in source
    assert 'preflight mixed8 "$project_root" 8 160000 "$run_id"' in source
    assert 'formal mixed8 "$project_root" 8 160000 "$run_id"' in source
    assert '--nnodes="$nnodes"' in source
    assert '--node_rank="$node_rank"' in source
    assert '--master_addr="$master_addr"' in source
    assert '--master_port="$master_port"' in source
    assert "collective probe SHA256 mismatch" in source
    assert "launch request SHA256 mismatch" in source
    for name in (
        "hcu_train_entry",
        "gradient_accumulation_patch",
        "multitask_sampler",
        "runtime_patches",
        "runtime_shims_sitecustomize",
    ):
        assert name in source
    assert "source-bound training SHA256 mismatch" in source


def test_collective_probe_covers_all_32_ranks_and_finite_sum() -> None:
    mapping = [
        probe.expected_global_rank(node_rank, local_rank)
        for node_rank in range(4)
        for local_rank in range(8)
    ]

    assert mapping == list(range(32))
    assert sum(range(1, 33)) == probe.EXPECTED_REDUCED_SUM
    source = Path(
        "scripts/n0_vtla/hpu_training/distributed_collective_probe.py"
    ).read_text(encoding="utf-8")
    assert "dist.all_reduce(reduced, op=dist.ReduceOp.SUM)" in source
    assert "dist.barrier()" in source
    assert '"world_size": world_size' in source


def test_container_runtime_reuses_pinned_isolated_contract() -> None:
    source = Path(
        "scripts/n0_vtla/hpu_training/distributed_container_runtime.sh"
    ).read_text(encoding="utf-8")

    assert (
        "sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
        in source
    )
    assert "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0" in source
    assert "--device /dev/kfd" in source
    assert "--device /dev/infiniband/rdma_cm" in source
    assert "--mount type=bind,src=/mnt/data,dst=/mnt/data" in source
    assert "--network host --ipc host" in source
    assert "docker container inspect" in source
    assert "docker create" in source
    assert 'docker rm "$container_id"' in source
    assert "docker rm -f" not in source
    assert "docker pull" not in source
    assert "pip install" not in source


def test_remote_phases_start_all_four_ssh_processes_and_keep_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tooling = tmp_path / "tooling/hpu_training"
    tooling.mkdir(parents=True)
    contents = {name: f"{name}\n".encode("utf-8") for name in launcher._SOURCE_NAMES}
    contents["distributed_container_runtime"] = b"runtime-script\n"
    paths = launcher._source_paths(tmp_path)
    for name, data in contents.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(data)
    bindings = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    request = launcher.LaunchRequest(
        project_root=tmp_path,
        run_id="concurrency-test",
        nodes=("192.0.2.10", "192.0.2.11", "192.0.2.12", "192.0.2.13"),
        ssh_user="root",
        ssh_port=36000,
        master_addr="192.0.2.10",
        master_port=29640,
        source_bindings=bindings,
    )

    def fake_ssh(
        _request_value: launcher.LaunchRequest, node: str, _payload: str
    ) -> list[str]:
        code = (
            "import sys; data=sys.stdin.buffer.read(); "
            f"print({node!r}, len(data), flush=True)"
        )
        return [sys.executable, "-c", code]

    monkeypatch.setattr(launcher, "_ssh_command", fake_ssh)
    returncodes = launcher._run_remote_phase(
        request,
        mode="preflight",
        request_sha256="b" * 64,
        log_dir=tmp_path,
        timeout_seconds=10.0,
    )

    assert returncodes == {node: 0 for node in request.nodes}
    for rank, node in enumerate(request.nodes):
        log = tmp_path / f"preflight-node-{rank:02d}-{node}.log"
        assert "runtime-script" not in log.read_text(encoding="utf-8")
        assert node in log.read_text(encoding="utf-8")


def test_state_and_request_writers_are_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "request.json"

    digest = launcher._write_new(path, {"status": "fresh"})

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        launcher._write_new(path, {"status": "replacement"})


def test_launcher_is_persistent_and_contains_no_cluster_addresses() -> None:
    source = Path(
        "scripts/n0_vtla/hpu_training/launch_formal_mixed8_4node.py"
    ).read_text(encoding="utf-8")
    committed = (
        "\n".join(
            path.read_text(encoding="utf-8")
            for path in Path("scripts/n0_vtla/hpu_training").glob("distributed_*")
        )
        + source
    )

    assert "start_new_session=True" in source
    assert 'supervisor_dir / "launcher.log"' in source
    assert 'supervisor_dir / "launch.json"' in source
    assert 'supervisor_dir / "state.json"' in source
    assert 'mode="preflight"' in source
    assert 'mode="train"' in source
    assert "10.232." not in committed

"""Immutable request for two-node Dream-Tac HCU training."""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, cast

from scripts.dream_tac.training.identity import canonical_json_sha256
from scripts.dream_tac.training.training_request import BoundFile, sha256_regular_file

from .overlay_contract import PINNED_COMMIT

REQUEST_SCHEMA: Final[str] = "robotactile-dream-tac-hcu-distributed-request-v4"
ACCELERATOR_CONTRACT: Final[str] = "hygon_hcu_hip_2node_16rank_v1"
IMAGE_REF: Final[str] = "docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
IMAGE_ID: Final[str] = (
    "sha256:09822e7616a3f9794284341aee19e1dfd531c2d2cb58168a40b14fcf818fdf9a"
)
RCCL_PLUGIN_PATH: Final[Path] = Path(
    "/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so.0.0.0"
)
RCCL_PLUGIN_SHA256: Final[str] = (
    "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
)
HCU_DEVICE_ORDER: Final[str] = "0,1,5,4,2,3,7,6"
FRANKA_DATASET_OVERLAY_SOURCE: Final[Path] = (
    Path(__file__).resolve().parent / "overlays/franka_dataset.py"
)
TASKS: Final[tuple[str, ...]] = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
LaunchMode = Literal["distributed-smoke", "formal"]

_RUN_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9_.-]{2,95}")
_SSH_USER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,31}")
_INTERFACE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.:-]{1,31}")
_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accelerator_contract",
        "batch_size",
        "container_image",
        "container_image_id",
        "dataset_split",
        "dtensor_helper_sha256",
        "dream_tac_commit",
        "franka_dataset_sha256",
        "franka_dataset_loading",
        "dream_tac_host_root",
        "fsdp_shard_size",
        "fsdp_helper_sha256",
        "fsdp_mesh_mode",
        "grad_accum_iter",
        "launch_mode",
        "master_addr",
        "master_port",
        "max_iter",
        "network_interface",
        "nodes",
        "nproc_per_node",
        "num_workers",
        "optimizer_step_request",
        "optimizer_step_result",
        "output_root",
        "rccl_plugin",
        "robotactile_root",
        "run_name",
        "runtime_host_root",
        "save_iter",
        "schema_version",
        "ssh_port",
        "ssh_user",
        "tasks",
        "visible_devices",
    }
)


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _nodes(value: object) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("nodes must contain exactly two IP addresses")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("nodes must contain IP address strings")
        parsed.append(str(ipaddress.ip_address(item)))
    if len(set(parsed)) != 2:
        raise ValueError("nodes must be unique")
    return cast(tuple[str, str], tuple(parsed))


@dataclass(frozen=True)
class DistributedTrainingRequest:
    """Source-bound two-node request with an immutable execution topology."""

    launch_mode: LaunchMode
    run_name: str
    robotactile_root: Path
    dream_tac_host_root: Path
    runtime_host_root: Path
    output_root: Path
    nodes: tuple[str, str]
    ssh_user: str
    ssh_port: int
    master_addr: str
    master_port: int
    network_interface: str
    max_iter: int
    save_iter: int
    batch_size: int
    num_workers: int
    grad_accum_iter: int
    franka_dataset_sha256: str
    fsdp_helper_sha256: str
    dtensor_helper_sha256: str
    optimizer_step_request: BoundFile
    optimizer_step_result: BoundFile

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DistributedTrainingRequest:
        if set(value) != _FIELDS:
            raise ValueError("Dream-Tac distributed request fields mismatch")
        expected: dict[str, object] = {
            "schema_version": REQUEST_SCHEMA,
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "dream_tac_commit": PINNED_COMMIT,
            "dataset_split": "train759",
            "franka_dataset_loading": "lazy_random_access_frames_v1",
            "fsdp_mesh_mode": "global_1d_shard_v1",
            "tasks": list(TASKS),
            "nproc_per_node": 8,
            "fsdp_shard_size": 16,
            "visible_devices": HCU_DEVICE_ORDER,
            "container_image": IMAGE_REF,
            "container_image_id": IMAGE_ID,
            "rccl_plugin": {
                "path": str(RCCL_PLUGIN_PATH),
                "sha256": RCCL_PLUGIN_SHA256,
            },
        }
        for name, required in expected.items():
            if value[name] != required:
                raise ValueError(f"{name} must be {required!r}")
        mode_value = value["launch_mode"]
        if mode_value not in {"distributed-smoke", "formal"}:
            raise ValueError("launch_mode must be distributed-smoke or formal")
        mode = cast(LaunchMode, mode_value)
        run_name = value["run_name"]
        if not isinstance(run_name, str) or _RUN_NAME.fullmatch(run_name) is None:
            raise ValueError("run_name must be a safe lowercase identifier")
        nodes = _nodes(value["nodes"])
        master_value = value["master_addr"]
        if not isinstance(master_value, str):
            raise ValueError("master_addr must be an IP address string")
        master_addr = str(ipaddress.ip_address(master_value))
        if master_addr != nodes[0]:
            raise ValueError("master_addr must equal nodes[0]")
        ssh_user = value["ssh_user"]
        if not isinstance(ssh_user, str) or _SSH_USER.fullmatch(ssh_user) is None:
            raise ValueError("ssh_user has an invalid format")
        interface = value["network_interface"]
        if not isinstance(interface, str) or _INTERFACE.fullmatch(interface) is None:
            raise ValueError("network_interface has an invalid format")
        max_iter = _integer(value["max_iter"], "max_iter", minimum=1, maximum=1_000_000)
        if mode == "distributed-smoke" and max_iter > 20:
            raise ValueError("distributed-smoke max_iter must be in [1, 20]")
        if mode == "formal" and max_iter <= 1_000:
            raise ValueError("formal max_iter must exceed 1000")
        save_iter = _integer(
            value["save_iter"], "save_iter", minimum=1, maximum=max_iter
        )
        franka_dataset_sha256 = _sha256(
            value["franka_dataset_sha256"], "franka_dataset_sha256"
        )
        expected_loader_sha256 = sha256_regular_file(FRANKA_DATASET_OVERLAY_SOURCE)
        if franka_dataset_sha256 != expected_loader_sha256:
            raise ValueError(
                "franka_dataset_sha256 must match the canonical lazy loader overlay"
            )
        return cls(
            launch_mode=mode,
            run_name=run_name,
            robotactile_root=_absolute_path(
                value["robotactile_root"], "robotactile_root"
            ),
            dream_tac_host_root=_absolute_path(
                value["dream_tac_host_root"], "dream_tac_host_root"
            ),
            runtime_host_root=_absolute_path(
                value["runtime_host_root"], "runtime_host_root"
            ),
            output_root=_absolute_path(value["output_root"], "output_root"),
            nodes=nodes,
            ssh_user=ssh_user,
            ssh_port=_integer(value["ssh_port"], "ssh_port", minimum=1, maximum=65535),
            master_addr=master_addr,
            master_port=_integer(
                value["master_port"], "master_port", minimum=1024, maximum=65535
            ),
            network_interface=interface,
            max_iter=max_iter,
            save_iter=save_iter,
            batch_size=_integer(
                value["batch_size"], "batch_size", minimum=1, maximum=8
            ),
            num_workers=_integer(
                value["num_workers"], "num_workers", minimum=0, maximum=16
            ),
            grad_accum_iter=_integer(
                value["grad_accum_iter"], "grad_accum_iter", minimum=1, maximum=64
            ),
            franka_dataset_sha256=franka_dataset_sha256,
            fsdp_helper_sha256=_sha256(
                value["fsdp_helper_sha256"], "fsdp_helper_sha256"
            ),
            dtensor_helper_sha256=_sha256(
                value["dtensor_helper_sha256"], "dtensor_helper_sha256"
            ),
            optimizer_step_request=BoundFile.from_value(
                value["optimizer_step_request"], "optimizer_step_request"
            ),
            optimizer_step_result=BoundFile.from_value(
                value["optimizer_step_result"], "optimizer_step_result"
            ),
        )

    @classmethod
    def load(cls, path: Path) -> DistributedTrainingRequest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dream-Tac distributed request must be an object")
        return cls.from_dict(payload)

    @property
    def nproc_per_node(self) -> int:
        return 8

    @property
    def world_size(self) -> int:
        return 16

    @property
    def fsdp_shard_size(self) -> int:
        return 16

    @property
    def effective_global_batch(self) -> int:
        return self.world_size * self.batch_size * self.grad_accum_iter

    @property
    def visible_devices(self) -> str:
        return HCU_DEVICE_ORDER

    @property
    def franka_dataset_host_path(self) -> Path:
        return self.dream_tac_host_root / "cosmos_policy/datasets/franka_dataset.py"

    @property
    def run_root(self) -> Path:
        return (
            self.output_root
            / "_robotactile_hcu_distributed"
            / self.run_name
            / self.request_sha256
        )

    @property
    def job_root(self) -> Path:
        return (
            self.output_root
            / "robotactile_dream_tac"
            / "univtac_hcu_2node"
            / self.run_name
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "batch_size": self.batch_size,
            "container_image": IMAGE_REF,
            "container_image_id": IMAGE_ID,
            "dataset_split": "train759",
            "dream_tac_commit": PINNED_COMMIT,
            "dream_tac_host_root": str(self.dream_tac_host_root),
            "franka_dataset_sha256": self.franka_dataset_sha256,
            "franka_dataset_loading": "lazy_random_access_frames_v1",
            "fsdp_helper_sha256": self.fsdp_helper_sha256,
            "fsdp_mesh_mode": "global_1d_shard_v1",
            "dtensor_helper_sha256": self.dtensor_helper_sha256,
            "fsdp_shard_size": self.fsdp_shard_size,
            "grad_accum_iter": self.grad_accum_iter,
            "launch_mode": self.launch_mode,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "max_iter": self.max_iter,
            "network_interface": self.network_interface,
            "nodes": list(self.nodes),
            "nproc_per_node": self.nproc_per_node,
            "num_workers": self.num_workers,
            "optimizer_step_request": self.optimizer_step_request.to_dict(),
            "optimizer_step_result": self.optimizer_step_result.to_dict(),
            "output_root": str(self.output_root),
            "rccl_plugin": {
                "path": str(RCCL_PLUGIN_PATH),
                "sha256": RCCL_PLUGIN_SHA256,
            },
            "robotactile_root": str(self.robotactile_root),
            "run_name": self.run_name,
            "runtime_host_root": str(self.runtime_host_root),
            "save_iter": self.save_iter,
            "schema_version": REQUEST_SCHEMA,
            "ssh_port": self.ssh_port,
            "ssh_user": self.ssh_user,
            "tasks": list(TASKS),
            "visible_devices": self.visible_devices,
        }

    @property
    def request_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


__all__ = [
    "ACCELERATOR_CONTRACT",
    "DistributedTrainingRequest",
    "FRANKA_DATASET_OVERLAY_SOURCE",
    "HCU_DEVICE_ORDER",
    "IMAGE_ID",
    "IMAGE_REF",
    "LaunchMode",
    "RCCL_PLUGIN_PATH",
    "RCCL_PLUGIN_SHA256",
    "REQUEST_SCHEMA",
    "TASKS",
]

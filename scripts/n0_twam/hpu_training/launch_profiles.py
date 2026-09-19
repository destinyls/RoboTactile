"""Bounded launch profiles for official N0-TWAM HCU training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contract import ClusterSpec

LaunchMode = Literal["single-card-smoke", "distributed-smoke", "formal"]

HCU_DEVICE_ORDER = "0,1,5,4,2,3,7,6"
DEFAULT_DISTRIBUTED_SMOKE_STEPS = 2
MAX_DISTRIBUTED_SMOKE_STEPS = 20


@dataclass(frozen=True)
class LaunchProfile:
    """Resolved topology and finite-step contract for one invocation."""

    mode: LaunchMode
    nodes: tuple[str, ...]
    processes_per_node: int
    num_steps: int
    visible_devices: str
    fsdp_topology: str
    fsdp_shard_size: int

    @property
    def world_size(self) -> int:
        return len(self.nodes) * self.processes_per_node


def resolve_launch_profile(
    *,
    mode: LaunchMode,
    cluster: ClusterSpec,
    training_steps: int,
    smoke_steps: int | None,
) -> LaunchProfile:
    """Resolve one of the only three supported execution shapes."""
    if training_steps <= 0:
        raise ValueError("training_steps must be positive")
    if mode == "single-card-smoke":
        if smoke_steps is not None:
            raise ValueError("single-card-smoke is fixed at exactly one step")
        return LaunchProfile(
            mode=mode,
            nodes=(cluster.master_addr,),
            processes_per_node=1,
            num_steps=1,
            visible_devices="0",
            fsdp_topology="global_shard",
            fsdp_shard_size=1,
        )
    if mode == "distributed-smoke":
        resolved_steps = (
            DEFAULT_DISTRIBUTED_SMOKE_STEPS if smoke_steps is None else smoke_steps
        )
        if not 1 <= resolved_steps <= MAX_DISTRIBUTED_SMOKE_STEPS:
            raise ValueError(
                "distributed smoke steps must be between 1 and "
                f"{MAX_DISTRIBUTED_SMOKE_STEPS}"
            )
        return LaunchProfile(
            mode=mode,
            nodes=cluster.nodes,
            processes_per_node=8,
            num_steps=resolved_steps,
            visible_devices=HCU_DEVICE_ORDER,
            fsdp_topology="global_shard",
            fsdp_shard_size=len(cluster.nodes) * 8,
        )
    if mode == "formal":
        if smoke_steps is not None:
            raise ValueError("--smoke-steps is only valid for distributed-smoke")
        return LaunchProfile(
            mode=mode,
            nodes=cluster.nodes,
            processes_per_node=8,
            num_steps=training_steps,
            visible_devices=HCU_DEVICE_ORDER,
            fsdp_topology="global_shard",
            fsdp_shard_size=len(cluster.nodes) * 8,
        )
    raise ValueError(f"unsupported launch mode: {mode!r}")

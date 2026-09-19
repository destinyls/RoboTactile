#!/usr/bin/env python3
"""Fail-closed 32-rank HCU collective and rank-map probe."""

from __future__ import annotations

import json
import math
import os
from typing import Final

EXPECTED_NNODES: Final[int] = 4
EXPECTED_LOCAL_WORLD_SIZE: Final[int] = 8
EXPECTED_WORLD_SIZE: Final[int] = 32
EXPECTED_REDUCED_SUM: Final[float] = 528.0


def expected_global_rank(node_rank: int, local_rank: int) -> int:
    """Return the only rank mapping accepted by the 4x8 launch contract."""

    if not 0 <= node_rank < EXPECTED_NNODES:
        raise ValueError(f"invalid node rank: {node_rank}")
    if not 0 <= local_rank < EXPECTED_LOCAL_WORLD_SIZE:
        raise ValueError(f"invalid local rank: {local_rank}")
    return node_rank * EXPECTED_LOCAL_WORLD_SIZE + local_rank


def main() -> None:
    import torch  # type: ignore[import-not-found]
    import torch.distributed as dist  # type: ignore[import-not-found]

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["GROUP_RANK"])
    if local_world_size != EXPECTED_LOCAL_WORLD_SIZE:
        raise RuntimeError(f"local world size mismatch: {local_world_size}")
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"world size mismatch: {world_size}")
    if rank != expected_global_rank(node_rank, local_rank):
        raise RuntimeError(
            f"rank map mismatch: rank={rank} node={node_rank} local={local_rank}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("each node must expose exactly eight HCU devices")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    reduced = torch.tensor(
        [float(rank + 1)], device=torch.device("cuda", local_rank), dtype=torch.float32
    )
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    observed_sum = float(reduced.item())
    if not math.isfinite(observed_sum) or observed_sum != EXPECTED_REDUCED_SUM:
        raise RuntimeError(f"32-rank all_reduce mismatch: {observed_sum}")
    document = {
        "finite_sum": observed_sum,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "protocol_id": "robotactile.n0_vtla.hcu_4node_collective_probe.v1",
        "rank": rank,
        "status": "passed",
        "world_size": world_size,
    }
    print(json.dumps(document, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

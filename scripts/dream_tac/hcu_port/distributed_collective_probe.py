"""Tiny 16-rank RCCL collective used by distributed preflight."""

from __future__ import annotations

import os
from datetime import timedelta

import torch  # type: ignore[import-not-found]
import torch.distributed as dist  # type: ignore[import-not-found]


def main() -> int:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 16:
        raise RuntimeError(f"collective probe requires world_size=16, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"RCCL all_reduce mismatch: {value.item()} != {expected}")
    dist.barrier()
    if rank == 0:
        print(f"ROBOTACTILE_DREAM_TAC_RCCL_PROBE_PASSED world_size={world_size}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

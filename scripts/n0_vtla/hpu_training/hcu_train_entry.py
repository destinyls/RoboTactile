"""Run the official N0-VTLA entry with guarded vendor-RCCL teardown."""

from __future__ import annotations

import os
import sys

import torch.distributed as dist  # type: ignore[import-not-found]
from gradient_accumulation_patch import install_gradient_accumulation_train
from multitask_sampler import install_uniform_task_sampler
from runtime_patches import install_detached_vl_ctx_policy


def _guarded_cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()


def main() -> None:
    if os.environ.get("ROBOTACTILE_N0_VTLA_TASK_BALANCED") == "1":
        source_path = install_uniform_task_sampler()
        sys.stderr.write(
            f"installed mixed8 task-balanced sampler source={source_path}\n"
        )
    if os.environ.get("ROBOTACTILE_N0_VTLA_DETACH_VL_CTX") == "1":
        source_path = install_detached_vl_ctx_policy()
        sys.stderr.write(f"installed HCU detach-vl-ctx patch source={source_path}\n")
    if os.environ.get("ROBOTACTILE_N0_VTLA_GRAD_ACCUM") == "1":
        source_path = install_gradient_accumulation_train()
        sys.stderr.write(
            f"installed HCU gradient-accumulation patch source={source_path}\n"
        )
    import scripts.train_n0vtla as upstream_entry  # type: ignore[import-not-found]
    import scripts.train_pytorch as upstream_train  # type: ignore[import-not-found]

    upstream_train.cleanup_ddp = _guarded_cleanup
    upstream_entry.main()
    if os.environ.get("ROBOTACTILE_VENDOR_TRAIN_FAST_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

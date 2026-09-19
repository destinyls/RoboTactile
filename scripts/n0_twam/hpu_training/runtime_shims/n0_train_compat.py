"""Execute the pinned official N0-TWAM trainer through runtime-only shims."""

from __future__ import annotations

import os
import runpy
import sys
from typing import NoReturn

from flex25_compat import install_vendor_flex25_compat  # type: ignore[import-not-found]


def _synchronize_and_fast_exit() -> NoReturn:
    """Synchronize successful ranks, then bypass the vendor RCCL destructor hang."""
    from torch import distributed  # type: ignore[import-not-found]

    if os.environ.get("ROBOTACTILE_VENDOR_TRAIN_FAST_EXIT") != "1":
        raise RuntimeError("vendor successful-exit compatibility is not enabled")
    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError(
            "official trainer returned without an initialized process group"
        )
    distributed.barrier()
    print(
        "ROBOTACTILE_TRAIN_EXIT_OK "
        f"rank={distributed.get_rank()} world_size={distributed.get_world_size()}",
        flush=True,
    )
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    """Install source-bound vendor compatibility, then run the official module."""
    install_vendor_flex25_compat()
    runpy.run_module("n0_twam.train", run_name="__main__", alter_sys=True)
    _synchronize_and_fast_exit()


if __name__ == "__main__":
    main()

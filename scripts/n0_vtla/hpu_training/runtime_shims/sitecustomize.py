"""Opt-in vendor PyTorch compatibility for official N0-VTLA training."""

from __future__ import annotations

import datetime
import inspect
import os


def _install_datetime_utc_compat() -> None:
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]


def _install_ddp_init_sync_compat() -> None:
    import torch  # type: ignore[import-not-found]

    ddp = torch.nn.parallel.DistributedDataParallel
    if "init_sync" in inspect.signature(ddp.__init__).parameters:
        return
    original_init = ddp.__init__

    def compatible_init(self, *args, init_sync=True, **kwargs):  # type: ignore[no-untyped-def]
        del init_sync
        return original_init(self, *args, **kwargs)

    ddp.__init__ = compatible_init


if os.environ.get("ROBOTACTILE_ENABLE_N0_VTLA_HCU_SHIMS") == "1":
    _install_datetime_utc_compat()
    _install_ddp_init_sync_compat()

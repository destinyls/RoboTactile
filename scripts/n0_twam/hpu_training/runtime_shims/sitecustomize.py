# ruff: noqa: I001
"""Vendor PyTorch namespace compatibility for the pinned official trainer.

The Lingchu PyTorch 2.5 vendor build exposes composable FSDP2 from
``torch.distributed._composable.fsdp`` while the pinned official N0-TWAM source
imports the same API from its later public namespace.  This startup shim only
aliases those two existing vendor objects; it does not replace their
implementation or modify the official checkout.
"""

from __future__ import annotations

import os


if os.environ.get("ROBOTACTILE_ENABLE_FSDP2_NAMESPACE_SHIM") == "1":
    import torch.distributed.fsdp as public_fsdp  # type: ignore[import-not-found]
    from torch.distributed._composable.fsdp import (  # type: ignore[import-not-found]
        MixedPrecisionPolicy,
        fully_shard,
    )

    if not hasattr(public_fsdp, "fully_shard"):
        public_fsdp.fully_shard = fully_shard
    if not hasattr(public_fsdp, "MixedPrecisionPolicy"):
        public_fsdp.MixedPrecisionPolicy = MixedPrecisionPolicy

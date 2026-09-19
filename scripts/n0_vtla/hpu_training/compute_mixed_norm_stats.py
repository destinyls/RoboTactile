"""Compute mixed8 train-only norm stats without dropping trailing frames."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

UPSTREAM_MODULE: Final[str] = "scripts.compute_norm_stats"
UPSTREAM_SHA256: Final[str] = (
    "ab99356fbc51d9602b55fd3a2b4efc587978bc33be09aab227bbee5ffd775efa"
)
MIXED_FRAME_COUNT: Final[int] = 144_484
NORM_BATCH_SIZE: Final[int] = 164


def _verify_upstream() -> Path:
    spec = importlib.util.find_spec(UPSTREAM_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {UPSTREAM_MODULE}")
    path = Path(spec.origin).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError(f"N0-VTLA norm source SHA256 mismatch: {digest}")
    return path


def main() -> int:
    _verify_upstream()
    normalize: Any = importlib.import_module("n0vtla.shared.normalize")
    config_module: Any = importlib.import_module("n0vtla.training.config")
    upstream: Any = importlib.import_module("scripts.compute_norm_stats")
    config = config_module.get_config("sim_single_arm_tactile")
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, num_batches = upstream.create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        NORM_BATCH_SIZE,
        config.model,
        config.num_workers,
    )
    if num_batches * NORM_BATCH_SIZE != MIXED_FRAME_COUNT:
        raise RuntimeError(
            "mixed8 normalizer must consume all frames: "
            f"{num_batches} * {NORM_BATCH_SIZE} != {MIXED_FRAME_COUNT}"
        )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    batch_count = 0
    for batch in loader:
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
        batch_count += 1
    if batch_count != num_batches:
        raise RuntimeError(
            f"mixed8 norm batch mismatch: {batch_count} != {num_batches}"
        )
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_id = data_config.asset_id
    if output_id != "univtac_mixed8_train759_qpos8":
        raise RuntimeError(f"unexpected mixed8 asset id: {output_id}")
    output_path = config.assets_dirs / output_id
    logging.info("Writing complete mixed8 normalization stats to %s", output_path)
    normalize.save(output_path, norm_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

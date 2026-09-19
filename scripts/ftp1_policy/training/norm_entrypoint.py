"""Finalize FTP-1 CLI overrides before running upstream normalization."""

from __future__ import annotations

import openpi.training.config as training_config
from scripts.zarr_compute_norm_stats import (  # type: ignore[import-not-found]
    main as upstream_main,
)


def main() -> None:
    """Run the pinned upstream normalizer with its derived data config rebuilt."""

    config = training_config.cli()
    if isinstance(config, training_config.FTP1TrainConfig):
        config.finalize_config()
    upstream_main(config)


if __name__ == "__main__":
    main()

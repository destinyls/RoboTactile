"""Dependency-free constants shared by Dream-Tac data and training stages."""

from typing import Final

DREAM_TAC_UPSTREAM_COMMIT: Final[str] = "14bab51d6862fd07124745c55cd395ea5caa9fd3"
DREAM_TAC_TRAINING_PROTOCOL: Final[str] = "dream_tac_univtac_train759_v1"
DATASET_DIR_NAME: Final[str] = "dataset"
DATASET_STATS_NAME: Final[str] = "dataset_statistics_franka.json"
DATASET_POST_NORM_STATS_NAME: Final[str] = "dataset_statistics_post_norm_franka.json"
GLOBAL_RECEIPT_NAME: Final[str] = "dream_tac_train759_receipt.json"

__all__ = [
    "DATASET_DIR_NAME",
    "DATASET_POST_NORM_STATS_NAME",
    "DATASET_STATS_NAME",
    "DREAM_TAC_TRAINING_PROTOCOL",
    "DREAM_TAC_UPSTREAM_COMMIT",
    "GLOBAL_RECEIPT_NAME",
]

"""Immutable Dream-Tac training-data contracts built on the UniVTAC split."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from scripts.n0_twam.hpu_training.data.contracts import (
    EXPECTED_SPLIT_COUNTS,
    SOURCE_FPS,
    TASK_PROMPTS,
    TASKS,
    SourceEpisode,
)
from scripts.n0_twam.hpu_training.data.manifest_io import (
    select_train_records,
    signed_payload,
)

from .training_constants import (
    DATASET_DIR_NAME,
    DATASET_STATS_NAME,
    DREAM_TAC_TRAINING_PROTOCOL,
    DREAM_TAC_UPSTREAM_COMMIT,
    GLOBAL_RECEIPT_NAME,
)

DREAM_TAC_EPISODE_PROTOCOL: Final[str] = "dream_tac_univtac_episode_v1"
DREAM_TAC_ACTION_SCHEMA: Final[str] = "xyz_rpy_gripper_absolute_next_step"
DREAM_TAC_STATE_SCHEMA: Final[str] = "xyz_rpy_absolute_current_step"
ACTION_HORIZON: Final[int] = 20
PROMPT_MANIFEST_NAME: Final[str] = "prompt_manifest.json"
T5_REQUEST_NAME: Final[str] = "t5_cache_request.json"
T5_ENCODER_RUNTIME_PROTOCOL: Final[str] = "cosmos_t5_cpu_fp32_active_tokens_v1"
EPISODE_RECEIPT_NAME: Final[str] = "_robotactile_dream_tac_episode.json"

VIDEO_SOURCES: Final[tuple[tuple[str, str], ...]] = (
    ("cam_front", "observation/head/rgb"),
    ("cam_high", "observation/wrist/rgb"),
    ("tactile_rectify_left", "tactile/left_gsmini/rgb_marker"),
    ("tactile_rectify_right", "tactile/right_gsmini/rgb_marker"),
)


def train759_records(records: Sequence[SourceEpisode]) -> tuple[SourceEpisode, ...]:
    """Select the exact 759 training records using the shared frozen split."""

    selected = tuple(
        record for task in TASKS for record in select_train_records(records, task=task)
    )
    if len(selected) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError("Dream-Tac materialization requires exact train759")
    if any(record.split != "train" for record in selected):
        raise ValueError("non-training source leaked into Dream-Tac materialization")
    return selected


def build_prompt_manifest(source_manifest_sha256: str) -> dict[str, object]:
    """Bind every HDF5 command and T5 key to the canonical task prompt."""

    prompts = [
        {"task_id": task, "prompt": TASK_PROMPTS[task], "hdf5_attr": "task_name"}
        for task in TASKS
    ]
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": DREAM_TAC_TRAINING_PROTOCOL,
        "source_manifest_sha256": source_manifest_sha256,
        "source_split": "train",
        "task_count": len(TASKS),
        "prompt_count": len(prompts),
        "prompts": prompts,
        "t5_cache_keys": [TASK_PROMPTS[task] for task in TASKS],
    }
    return signed_payload(unsigned, field="prompt_manifest_sha256")


def validate_prompt_manifest(
    payload: Mapping[str, object], *, source_manifest_sha256: str
) -> None:
    """Reject prompt drift because HDF5 attrs index the T5 cache directly."""

    expected = build_prompt_manifest(source_manifest_sha256)
    if dict(payload) != expected:
        raise ValueError("Dream-Tac prompt manifest does not match canonical prompts")


def build_t5_cache_request(
    *, prompt_manifest: Mapping[str, object], output_root: Path
) -> dict[str, object]:
    """Describe real upstream T5 generation without fabricating embeddings."""

    digest = prompt_manifest.get("prompt_manifest_sha256")
    if not isinstance(digest, str):
        raise ValueError("prompt manifest has no identity")
    dataset_root = output_root / DATASET_DIR_NAME
    unsigned: dict[str, object] = {
        "schema_version": 2,
        "protocol_id": DREAM_TAC_TRAINING_PROTOCOL,
        "status": "external_generation_required",
        "generator": (
            "cosmos_policy._src.predict2.inference.get_t5_emb."
            "CosmosT5TextEncoder.encode_prompts"
        ),
        "encoder_runtime": {
            "protocol_id": T5_ENCODER_RUNTIME_PROTOCOL,
            "device": "cpu",
            "model_compute_dtype": "torch.float32",
            "active_token_batching": True,
            "max_output_tokens": 512,
            "output_dtype": "torch.bfloat16",
            "padding_value": 0.0,
        },
        "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
        "prompt_manifest_sha256": digest,
        "expected_cache_keys": [TASK_PROMPTS[task] for task in TASKS],
        "expected_tensor_shape": [1, 512, 1024],
        "expected_tensor_dtype": "torch.bfloat16",
        "expected_output": str((dataset_root / "t5_embeddings.pkl").resolve()),
        "command": [
            "python",
            "-m",
            "scripts.dream_tac.training.t5_cache",
            "--dream-tac-root",
            "<PINNED_DREAM_TAC_ROOT>",
            "--output-root",
            str(output_root.resolve()),
        ],
    }
    return signed_payload(unsigned, field="t5_request_sha256")


__all__ = [
    "ACTION_HORIZON",
    "DATASET_DIR_NAME",
    "DATASET_STATS_NAME",
    "DREAM_TAC_ACTION_SCHEMA",
    "DREAM_TAC_EPISODE_PROTOCOL",
    "DREAM_TAC_STATE_SCHEMA",
    "DREAM_TAC_TRAINING_PROTOCOL",
    "DREAM_TAC_UPSTREAM_COMMIT",
    "EPISODE_RECEIPT_NAME",
    "GLOBAL_RECEIPT_NAME",
    "PROMPT_MANIFEST_NAME",
    "SOURCE_FPS",
    "T5_REQUEST_NAME",
    "T5_ENCODER_RUNTIME_PROTOCOL",
    "VIDEO_SOURCES",
    "build_prompt_manifest",
    "build_t5_cache_request",
    "train759_records",
    "validate_prompt_manifest",
]

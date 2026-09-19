"""No-clobber receipts for Dream-Tac train759 materialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from scripts.n0_twam.hpu_training.data.contracts import (
    TASK_PROMPTS,
    SourceEpisode,
    sha256_file,
)
from scripts.n0_twam.hpu_training.data.manifest_io import (
    load_json_object,
    receipt_sha256,
    signed_payload,
)

from .contracts import (
    ACTION_HORIZON,
    DREAM_TAC_ACTION_SCHEMA,
    DREAM_TAC_EPISODE_PROTOCOL,
    DREAM_TAC_STATE_SCHEMA,
    EPISODE_RECEIPT_NAME,
    SOURCE_FPS,
    VIDEO_SOURCES,
)
from .episode_io import verify_official_hdf5

ArtifactVerifier = Callable[..., None]


def build_episode_receipt(
    *,
    record: SourceEpisode,
    source_manifest_sha256: str,
    export: Mapping[str, object],
) -> dict[str, object]:
    """Bind one official-format episode to its immutable training source."""

    files = export.get("files")
    converted = export.get("converted_frame_count")
    raw = export.get("raw_frame_count")
    usable = export.get("usable_source_range")
    if not isinstance(files, list) or len(files) != 5:
        raise ValueError("Dream-Tac export must contain one HDF5 and four videos")
    if isinstance(converted, bool) or not isinstance(converted, int) or converted <= 0:
        raise ValueError("Dream-Tac converted frame count is invalid")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= converted:
        raise ValueError("Dream-Tac raw frame count is invalid")
    if not isinstance(usable, list) or len(usable) != 2:
        raise ValueError("Dream-Tac usable source range is invalid")
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": DREAM_TAC_EPISODE_PROTOCOL,
        "source_manifest_sha256": source_manifest_sha256,
        "source_split": "train",
        "source_relative_path": record.relative_path,
        "source_sha256": record.sha256,
        "task_id": record.task,
        "source_episode_id": record.episode_id,
        "task_name_prompt": TASK_PROMPTS[record.task],
        "source_fps": SOURCE_FPS,
        "video_fps": SOURCE_FPS,
        "raw_frame_count": raw,
        "converted_frame_count": converted,
        "temporal_contract": "observation_t_to_absolute_action_t_plus_1",
        "state_schema": DREAM_TAC_STATE_SCHEMA,
        "action_schema": DREAM_TAC_ACTION_SCHEMA,
        "action_horizon": ACTION_HORIZON,
        "quaternion_source_order": "wxyz",
        "rotation_conversion": "sign_continuous_wxyz_to_unwrapped_euler_xyz",
        "gripper_source": "embodiment/joint[t+1,7]",
        "tactile_required": True,
        "usable_source_range": usable,
        "files": files,
    }
    return signed_payload(unsigned, field="episode_receipt_sha256")


def _expected_files(receipt: Mapping[str, object]) -> dict[str, str]:
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 5:
        raise ValueError("Dream-Tac episode receipt file inventory is invalid")
    result: dict[str, str] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError("Dream-Tac episode receipt file entry is invalid")
        relative_path = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).name != relative_path
        ):
            raise ValueError("Dream-Tac artifact path must be one basename")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Dream-Tac artifact SHA256 is invalid")
        if relative_path in result:
            raise ValueError("Dream-Tac artifact inventory contains duplicates")
        result[relative_path] = digest
    return result


def validate_episode_destination(
    *,
    destination: Path,
    record: SourceEpisode,
    source_manifest_sha256: str,
    artifact_verifier: ArtifactVerifier = verify_official_hdf5,
) -> dict[str, object]:
    """Verify hashes, source identity, prompt identity, and official HDF5 shape."""

    receipt_path = destination / EPISODE_RECEIPT_NAME
    if (
        destination.is_symlink()
        or not destination.is_dir()
        or not receipt_path.is_file()
    ):
        raise FileExistsError(
            f"existing Dream-Tac episode is not receipted: {destination}"
        )
    receipt = load_json_object(receipt_path)
    receipt_sha256(receipt, field="episode_receipt_sha256")
    expected = {
        "status": "complete",
        "protocol_id": DREAM_TAC_EPISODE_PROTOCOL,
        "source_manifest_sha256": source_manifest_sha256,
        "source_split": "train",
        "source_relative_path": record.relative_path,
        "source_sha256": record.sha256,
        "task_id": record.task,
        "source_episode_id": record.episode_id,
        "task_name_prompt": TASK_PROMPTS[record.task],
        "source_fps": SOURCE_FPS,
        "video_fps": SOURCE_FPS,
        "tactile_required": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Dream-Tac episode receipt mismatch: {destination}")
    inventory = _expected_files(receipt)
    actual = {
        path.name for path in destination.iterdir() if path.name != EPISODE_RECEIPT_NAME
    }
    if actual != set(inventory):
        raise ValueError(f"Dream-Tac episode artifact set mismatch: {destination}")
    for name, digest in inventory.items():
        artifact = destination / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Dream-Tac artifact is not a regular file: {artifact}")
        if sha256_file(artifact) != digest:
            raise ValueError(f"Dream-Tac artifact SHA256 mismatch: {artifact}")
    hdf5_names = [name for name in inventory if name.endswith(".hdf5")]
    if len(hdf5_names) != 1:
        raise ValueError("Dream-Tac episode must contain exactly one HDF5")
    converted = receipt.get("converted_frame_count")
    if not isinstance(converted, int) or isinstance(converted, bool):
        raise ValueError("Dream-Tac receipt converted length is invalid")
    artifact_verifier(
        destination / hdf5_names[0],
        prompt=TASK_PROMPTS[record.task],
        task_id=record.task,
        converted_length=converted,
    )
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Dream-Tac receipt file inventory is invalid")
    stream_entries = {
        item.get("stream")
        for item in raw_files
        if isinstance(item, dict) and item.get("kind") == "video"
    }
    if stream_entries != {name for name, _ in VIDEO_SOURCES}:
        raise ValueError("Dream-Tac receipt does not prove both tactile streams")
    return receipt


def build_global_receipt(
    *,
    source_manifest_sha256: str,
    prompt_manifest_sha256: str,
    t5_request_sha256: str,
    statistics_sha256: str,
    statistics_sample_count: int,
    episode_receipts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a complete-only receipt over all 759 episode transactions."""

    entries: list[dict[str, object]] = []
    total_frames = 0
    for receipt in episode_receipts:
        digest = receipt.get("episode_receipt_sha256")
        frames = receipt.get("converted_frame_count")
        if not isinstance(digest, str) or not isinstance(frames, int):
            raise ValueError("Dream-Tac episode receipt is incomplete")
        total_frames += frames
        entries.append(
            {
                "task_id": receipt["task_id"],
                "source_episode_id": receipt["source_episode_id"],
                "source_relative_path": receipt["source_relative_path"],
                "episode_receipt_sha256": digest,
                "converted_frame_count": frames,
            }
        )
    if len(entries) != 759:
        raise ValueError("Dream-Tac global receipt requires exactly 759 episodes")
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "dream_tac_univtac_train759_v1",
        "source_manifest_sha256": source_manifest_sha256,
        "source_split_counts": {"train": 759, "frozen": 40, "quarantine": 1},
        "materialized_splits": ["train"],
        "materialized_episode_count": 759,
        "excluded_episode_count": 41,
        "source_fps": SOURCE_FPS,
        "control_hz": SOURCE_FPS,
        "action_horizon": ACTION_HORIZON,
        "frame_count": total_frames,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "t5_request_sha256": t5_request_sha256,
        "dataset_statistics_sha256": statistics_sha256,
        "statistics_sample_count": statistics_sample_count,
        "t5_cache_status": "external_generation_required",
        "training_ready": False,
        "episodes": entries,
    }
    return signed_payload(unsigned, field="receipt_sha256")


__all__ = [
    "ArtifactVerifier",
    "build_episode_receipt",
    "build_global_receipt",
    "validate_episode_destination",
]

"""Aggregate 16 worker receipts into one complete train759 inventory."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Mapping, Sequence

from ..contract import OFFICIAL_COMMIT
from .contracts import (
    EXPECTED_EPISODES,
    OFFICIAL_TACTILE_ENCODER_SHA256,
    OFFICIAL_VIDEO_ENCODER_SHA256,
    TACTILE_KEYS,
    TARGET_FPS,
    VIDEO_KEYS,
    EpisodeRecord,
    WorkUnit,
    canonical_sha256,
    sha256_file,
)


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    return payload


def _inventory_by_episode(
    receipt: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    raw_inventory = receipt.get("inventory")
    if not isinstance(raw_inventory, list):
        raise ValueError("complete worker receipt has no inventory")
    inventory: dict[int, dict[str, object]] = {}
    for raw in raw_inventory:
        if not isinstance(raw, dict):
            raise ValueError("worker inventory entry must be an object")
        episode_id = raw.get("lerobot_episode_index")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise ValueError("worker inventory episode ID is invalid")
        if episode_id in inventory:
            raise ValueError(f"duplicate worker inventory episode: {episode_id}")
        inventory[episode_id] = raw
    return inventory


def _validate_outputs(kind: str, record: Mapping[str, object]) -> dict[str, object]:
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("inventory record has no outputs")
    expected = (
        set(VIDEO_KEYS)
        if kind == "vision"
        else {f"{mode}:{key}" for mode in ("global", "local") for key in TACTILE_KEYS}
    )
    if set(outputs) != expected:
        raise ValueError(f"{kind} output key set is incomplete: {set(outputs)}")
    return outputs


def aggregate_inventory(
    *,
    units: Sequence[WorkUnit],
    receipt_dir: Path,
    source_manifest_path: Path,
    conversion_receipt_path: Path,
    official_commit: str,
    encoder_sha256: Mapping[str, str],
) -> dict[str, object]:
    if len(units) != 16:
        raise ValueError("final inventory requires exactly 16 workers")
    expected_receipts = {
        receipt_dir / f"worker-{unit.worker_id:02d}.json" for unit in units
    }
    actual_receipts = set(receipt_dir.glob("worker-*.json"))
    if actual_receipts != expected_receipts:
        raise ValueError("worker receipt set is incomplete or contains extras")

    combined: dict[tuple[str, int], dict[str, object]] = {}
    receipt_records: list[dict[str, object]] = []
    for unit in units:
        receipt_path = receipt_dir / f"worker-{unit.worker_id:02d}.json"
        receipt = _load_object(receipt_path)
        expected_ids = [episode.lerobot_episode_index for episode in unit.episodes]
        if receipt.get("status") != "complete":
            raise ValueError(f"worker did not complete: {receipt_path}")
        if (
            receipt.get("worker_id") != unit.worker_id
            or receipt.get("node") != unit.node
            or receipt.get("local_device") != unit.local_device
            or receipt.get("task") != unit.task
            or receipt.get("kind") != unit.kind
            or receipt.get("target_fps") != TARGET_FPS
            or receipt.get("episode_ids") != expected_ids
            or receipt.get("final_missing_episode_ids") != []
        ):
            raise ValueError(f"worker receipt contract mismatch: {receipt_path}")
        inventory = _inventory_by_episode(receipt)
        if set(inventory) != set(expected_ids):
            raise ValueError(f"worker inventory coverage mismatch: {receipt_path}")
        for episode in unit.episodes:
            raw = inventory[episode.lerobot_episode_index]
            if (
                raw.get("source_episode_id") != episode.source_episode_id
                or raw.get("source_relative_path") != episode.source_relative_path
                or raw.get("source_sha256") != episode.source_sha256
                or raw.get("length") != episode.length
            ):
                raise ValueError("worker inventory source identity mismatch")
            _validate_outputs(unit.kind, raw)
            key = (unit.task, episode.lerobot_episode_index)
            item = combined.setdefault(
                key,
                {
                    "task": unit.task,
                    "lerobot_episode_index": episode.lerobot_episode_index,
                    "source_episode_id": episode.source_episode_id,
                    "source_relative_path": episode.source_relative_path,
                    "source_sha256": episode.source_sha256,
                    "length": episode.length,
                },
            )
            if unit.kind in item:
                raise ValueError(f"duplicate modality inventory: {key}/{unit.kind}")
            item[unit.kind] = raw
        receipt_records.append(
            {
                "worker_id": unit.worker_id,
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "task": unit.task,
                "kind": unit.kind,
                "episode_count": len(unit.episodes),
            }
        )

    if len(combined) != EXPECTED_EPISODES:
        raise ValueError(f"final episode coverage is {len(combined)}, expected 759")
    vision_files = 0
    tactile_files = 0
    episodes_out: list[dict[str, object]] = []
    for key in sorted(combined):
        item = combined[key]
        if "vision" not in item or "tactile" not in item:
            raise ValueError(f"episode lacks a modality inventory: {key}")
        vision = item["vision"]
        tactile = item["tactile"]
        if not isinstance(vision, dict) or not isinstance(tactile, dict):
            raise ValueError(f"invalid paired inventory: {key}")
        if vision.get("latent_num_frames") != tactile.get(
            "latent_num_frames"
        ) or vision.get("frame_ids_sha256") != tactile.get("frame_ids_sha256"):
            raise ValueError(f"Vision/tactile temporal alignment mismatch: {key}")
        vision_outputs = _validate_outputs("vision", vision)
        tactile_outputs = _validate_outputs("tactile", tactile)
        vision_files += len(vision_outputs)
        tactile_files += len(tactile_outputs)
        episodes_out.append(item)
    if vision_files != EXPECTED_EPISODES * 2:
        raise ValueError("Vision latent file count is incomplete")
    if tactile_files != EXPECTED_EPISODES * 4:
        raise ValueError("tactile global/local latent file count is incomplete")

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "official_n0_twam_train759_latents_10hz_v1",
        "official_commit": official_commit,
        "official_encoder_sha256": dict(encoder_sha256),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "conversion_receipt_path": str(conversion_receipt_path),
        "conversion_receipt_sha256": sha256_file(conversion_receipt_path),
        "target_fps": TARGET_FPS,
        "episode_count": len(episodes_out),
        "worker_count": len(units),
        "vision_keys": list(VIDEO_KEYS),
        "tactile_keys": list(TACTILE_KEYS),
        "vision_file_count": vision_files,
        "tactile_global_file_count": EXPECTED_EPISODES * 2,
        "tactile_local_file_count": EXPECTED_EPISODES * 2,
        "worker_receipts": receipt_records,
        "episodes": episodes_out,
    }
    payload["inventory_sha256"] = canonical_sha256(payload)
    return payload


def validate_training_inventory(
    *,
    path: Path,
    records: Sequence[EpisodeRecord],
    source_manifest_path: Path,
    conversion_receipt_path: Path,
) -> dict[str, object]:
    """Verify a complete inventory and stat every required latent before training."""
    payload = _load_object(path)
    unsigned = dict(payload)
    claimed_sha = unsigned.pop("inventory_sha256", None)
    actual_sha = canonical_sha256(unsigned)
    if claimed_sha != actual_sha:
        raise ValueError("latent inventory SHA256 mismatch")
    expected: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "official_n0_twam_train759_latents_10hz_v1",
        "official_commit": OFFICIAL_COMMIT,
        "official_encoder_sha256": {
            "vision": OFFICIAL_VIDEO_ENCODER_SHA256,
            "tactile": OFFICIAL_TACTILE_ENCODER_SHA256,
        },
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "conversion_receipt_path": str(conversion_receipt_path),
        "conversion_receipt_sha256": sha256_file(conversion_receipt_path),
        "target_fps": TARGET_FPS,
        "episode_count": EXPECTED_EPISODES,
        "worker_count": 16,
        "vision_keys": list(VIDEO_KEYS),
        "tactile_keys": list(TACTILE_KEYS),
        "vision_file_count": EXPECTED_EPISODES * 2,
        "tactile_global_file_count": EXPECTED_EPISODES * 2,
        "tactile_local_file_count": EXPECTED_EPISODES * 2,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("latent inventory contract mismatch")
    for field, expected_path in (
        ("source_manifest_path", source_manifest_path),
        ("conversion_receipt_path", conversion_receipt_path),
    ):
        raw_path = payload.get(field)
        if not isinstance(raw_path, str) or Path(raw_path).resolve(
            strict=True
        ) != expected_path.resolve(strict=True):
            raise ValueError(f"latent inventory {field} mismatch")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != EXPECTED_EPISODES:
        raise ValueError("latent inventory episode coverage is incomplete")
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise ValueError("latent inventory episode must be an object")
        task = raw.get("task")
        episode_id = raw.get("lerobot_episode_index")
        if (
            not isinstance(task, str)
            or isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
        ):
            raise ValueError("latent inventory episode identity is invalid")
        key = (task, episode_id)
        if key in indexed:
            raise ValueError(f"duplicate latent inventory episode: {key}")
        indexed[key] = raw
    for record in records:
        key = (record.task, record.lerobot_episode_index)
        raw = indexed.get(key)
        if raw is None:
            raise ValueError(f"missing latent inventory episode: {key}")
        identity: dict[str, object] = {
            "source_episode_id": record.source_episode_id,
            "source_relative_path": record.source_relative_path,
            "source_sha256": record.source_sha256,
            "length": record.length,
        }
        if any(raw.get(field) != value for field, value in identity.items()):
            raise ValueError(f"latent inventory source identity mismatch: {key}")
        vision = raw.get("vision")
        tactile = raw.get("tactile")
        if not isinstance(vision, dict) or not isinstance(tactile, dict):
            raise ValueError(f"latent inventory modalities are missing: {key}")
        if vision.get("latent_num_frames") != tactile.get(
            "latent_num_frames"
        ) or vision.get("frame_ids_sha256") != tactile.get("frame_ids_sha256"):
            raise ValueError(f"latent inventory temporal mismatch: {key}")
        for kind, modality in (("vision", vision), ("tactile", tactile)):
            outputs = _validate_outputs(kind, modality)
            for output in outputs.values():
                if not isinstance(output, dict):
                    raise ValueError(f"invalid latent output metadata: {key}/{kind}")
                raw_path = output.get("path")
                if not isinstance(raw_path, str):
                    raise ValueError(f"latent output path is invalid: {key}/{kind}")
                output_path = Path(raw_path)
                metadata = output_path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        f"latent output is not a regular file: {output_path}"
                    )
                resolved = output_path.resolve(strict=True)
                if record.repo_path.resolve(strict=True) not in resolved.parents:
                    raise ValueError(f"latent output escapes task repo: {output_path}")
                if (
                    output.get("size_bytes") != metadata.st_size
                    or output.get("mtime_ns") != metadata.st_mtime_ns
                ):
                    raise ValueError(
                        f"latent output changed after inventory: {output_path}"
                    )
    if set(indexed) != {
        (record.task, record.lerobot_episode_index) for record in records
    }:
        raise ValueError("latent inventory contains unexpected episodes")
    return {
        "path": str(path),
        "inventory_sha256": actual_sha,
        "episode_count": EXPECTED_EPISODES,
        "latent_file_count": EXPECTED_EPISODES * 6,
    }

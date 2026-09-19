"""Expected paths and payload-level validation for official latent files."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Mapping

from .contracts import TACTILE_KEYS, TARGET_FPS, VIDEO_KEYS, Kind


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _numeric_value(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def load_repo_info(repo_path: Path) -> dict[str, object]:
    payload = json.loads((repo_path / "meta" / "info.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid LeRobot info: {repo_path}")
    if int(round(_numeric_value(payload.get("fps"), label="repo fps"))) != TARGET_FPS:
        raise ValueError(f"LeRobot repo does not preserve 10 Hz: {repo_path}")
    return payload


def episode_chunk(info: Mapping[str, object], episode_id: int) -> int:
    total_chunks = _positive_int(info.get("total_chunks", 1), label="total_chunks")
    chunks_size = _positive_int(info.get("chunks_size", 1000), label="chunks_size")
    return 0 if total_chunks <= 1 else episode_id // chunks_size


def expected_output_paths(
    *, repo_path: Path, episode_id: int, length: int, kind: Kind
) -> dict[str, Path]:
    info = load_repo_info(repo_path)
    chunk = episode_chunk(info, episode_id)
    filename = f"episode_{episode_id:06d}_0_{length}.pth"
    if kind == "vision":
        return {
            key: repo_path / "latents" / f"chunk-{chunk:03d}" / key / filename
            for key in VIDEO_KEYS
        }
    return {
        f"{mode}:{key}": (
            repo_path / "latents_tactile" / mode / f"chunk-{chunk:03d}" / key / filename
        )
        for mode in ("global", "local")
        for key in TACTILE_KEYS
    }


def _frame_ids_digest(frame_ids: list[int]) -> str:
    encoded = json.dumps(frame_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_latent_file(
    *, path: Path, root: Path, episode_id: int, length: int, label: str
) -> dict[str, object]:
    """Load one payload on CPU and reject truncated or mis-timed files."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"latent output must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if root.resolve(strict=True) not in resolved.parents:
        raise ValueError(f"latent output escapes task repo: {path}")

    import torch  # type: ignore[import-not-found]

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"latent payload is not a mapping: {path}")
    latent = payload.get("latent")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
        raise ValueError(f"latent tensor is invalid: {path}")
    frames = payload.get("latent_num_frames")
    height = payload.get("latent_height")
    width = payload.get("latent_width")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (frames, height, width)
    ):
        raise ValueError(f"latent dimensions are invalid: {path}")
    assert (
        isinstance(frames, int) and isinstance(height, int) and isinstance(width, int)
    )
    if latent.shape[0] != frames * height * width or latent.shape[1] <= 0:
        raise ValueError(f"flattened latent shape is inconsistent: {path}")
    frame_ids_raw = payload.get("frame_ids")
    if not isinstance(frame_ids_raw, list) or not frame_ids_raw:
        raise ValueError(f"frame_ids are missing: {path}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in frame_ids_raw
    ):
        raise ValueError(f"frame_ids are invalid: {path}")
    frame_ids = [int(value) for value in frame_ids_raw]
    if frame_ids != sorted(frame_ids) or frame_ids[0] != 0:
        raise ValueError(f"frame_ids are not monotonic from zero: {path}")
    if payload.get("fps") != TARGET_FPS or payload.get("ori_fps") != TARGET_FPS:
        raise ValueError(f"latent cadence is not true 10 Hz: {path}")
    if payload.get("start_frame") != 0 or payload.get("end_frame") != length:
        raise ValueError(f"latent segment boundary mismatch: {path}")
    if label.startswith("global:") and payload.get("tactile_residual_mode") != "global":
        raise ValueError(f"global tactile semantic mismatch: {path}")
    if label.startswith("local:") and payload.get("tactile_residual_mode") != "local":
        raise ValueError(f"local tactile semantic mismatch: {path}")
    if label in VIDEO_KEYS and payload.get("video_num_frames") != len(frame_ids):
        raise ValueError(f"video frame count metadata mismatch: {path}")
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "latent_num_frames": frames,
        "latent_height": height,
        "latent_width": width,
        "latent_channels": int(latent.shape[1]),
        "frame_count": len(frame_ids),
        "frame_ids_sha256": _frame_ids_digest(frame_ids),
        "fps": TARGET_FPS,
        "ori_fps": TARGET_FPS,
    }


def inventory_episode(
    *, repo_path: Path, episode_id: int, length: int, kind: Kind
) -> dict[str, object]:
    paths = expected_output_paths(
        repo_path=repo_path, episode_id=episode_id, length=length, kind=kind
    )
    outputs = {
        label: inspect_latent_file(
            path=path,
            root=repo_path,
            episode_id=episode_id,
            length=length,
            label=label,
        )
        for label, path in paths.items()
    }
    frame_counts = {
        _positive_int(output["latent_num_frames"], label="latent_num_frames")
        for output in outputs.values()
    }
    frame_digests = {str(output["frame_ids_sha256"]) for output in outputs.values()}
    if len(frame_counts) != 1 or len(frame_digests) != 1:
        raise ValueError(
            f"within-modality latent alignment mismatch: {repo_path}/{episode_id}"
        )
    return {
        "lerobot_episode_index": episode_id,
        "length": length,
        "latent_num_frames": next(iter(frame_counts)),
        "frame_ids_sha256": next(iter(frame_digests)),
        "outputs": outputs,
    }


def missing_episode_ids(
    *, repo_path: Path, episodes: list[tuple[int, int]], kind: Kind
) -> list[int]:
    """Return only wholly/partly missing IDs; existing invalid files raise."""
    missing: list[int] = []
    for episode_id, length in episodes:
        paths = expected_output_paths(
            repo_path=repo_path,
            episode_id=episode_id,
            length=length,
            kind=kind,
        )
        existing = [path.exists() for path in paths.values()]
        if not any(existing):
            missing.append(episode_id)
            continue
        for label, path in paths.items():
            if path.exists():
                inspect_latent_file(
                    path=path,
                    root=repo_path,
                    episode_id=episode_id,
                    length=length,
                    label=label,
                )
        if not all(existing):
            missing.append(episode_id)
    return missing

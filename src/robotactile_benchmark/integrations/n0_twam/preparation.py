"""Prepare the released N0-TWAM delta checkpoint for one UniVTAC task."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    BASE_REPOSITORY,
    BASE_REVISION,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    serve_task_id,
)
from robotactile_benchmark.policies.n0_official import n0_training_prompt

PREPARATION_SCHEMA_VERSION = "robotactile-n0-serve-bundle-v2"


@dataclass(frozen=True)
class PreparedN0Artifacts:
    """Paths consumed by the typed N0 artifact manifest builder."""

    bundle_root: Path
    base_root: Path
    checkpoint_root: Path
    serve_bundle_root: Path
    serve_pool_root: Path
    checkpoint_path: Path
    config_path: Path
    train_meta_path: Path
    normalizer_path: Path
    prompt_manifest_path: Path
    serve_bundle_manifest_path: Path
    serve_info_path: Path
    serve_tasks_path: Path


def _real_directory(path: Path, name: str) -> Path:
    selected = Path(path).absolute()
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError(f"{name} must be a real directory: {selected}")
    return selected


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file: {path}")
    return path


def _read_json(path: Path, name: str) -> object:
    try:
        return json.loads(_regular_file(path, name).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlink output: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing divergent existing artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary artifact already exists: {temporary}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_link(source: Path, destination: Path) -> None:
    source_real = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source_real:
            raise FileExistsError(f"serve link points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"refusing non-link serve component: {destination}")
    destination.symlink_to(source_real, target_is_directory=source_real.is_dir())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_files(name: str, source: Path) -> dict[str, dict[str, object]]:
    if source.is_symlink():
        raise ValueError(f"N0 base component cannot be a symlink: {source}")
    if source.is_file():
        return {name: {"sha256": _sha256(source), "size": source.stat().st_size}}
    if not source.is_dir():
        raise ValueError(f"N0 base component is unavailable: {source}")
    inventory: dict[str, dict[str, object]] = {}
    for current, directories, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ValueError("N0 base component cannot contain directory symlinks")
        for filename in sorted(files):
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("N0 base component must contain regular files")
            relative = path.relative_to(source).as_posix()
            inventory[f"{name}/{relative}"] = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
    if not inventory:
        raise ValueError(f"N0 base component is empty: {source}")
    return inventory


def _norm_entry(value: object, expected_task: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or expected_task not in value:
        raise ValueError(f"normalizer does not contain task {expected_task}")
    entry = value[expected_task]
    if not isinstance(entry, Mapping) or set(entry) < {"q01", "q99"}:
        raise ValueError("per-task normalizer must contain q01 and q99")
    for name in ("q01", "q99"):
        channel = entry[name]
        if (
            not isinstance(channel, Sequence)
            or isinstance(channel, (str, bytes))
            or len(channel) != 20
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in channel
            )
        ):
            raise ValueError(f"normalizer {name} must contain 20 numbers")
    return cast(Mapping[str, object], entry)


def _contains_prompt(value: object, prompt: str) -> bool:
    if value == prompt:
        return True
    if isinstance(value, Mapping):
        return any(_contains_prompt(item, prompt) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_prompt(item, prompt) for item in value)
    return False


def prepare_official_n0_artifacts(
    *, bundle_root: Path, task_id: str
) -> PreparedN0Artifacts:
    """Build an idempotent serve bundle and minimal per-task serve pool."""

    root = _real_directory(bundle_root, "N0 artifact root")
    base_root = _real_directory(root / "base", "N0 base snapshot")
    checkpoint_root = _real_directory(
        root / "univtac-delta", "N0 UniVTAC delta snapshot"
    )
    checkpoint_path = _regular_file(
        checkpoint_root / "transformer/diffusion_pytorch_model.safetensors",
        "N0 checkpoint",
    )
    config_path = _regular_file(
        checkpoint_root / "transformer/config.json", "N0 transformer config"
    )
    train_meta_path = _regular_file(
        checkpoint_root / "train_meta.json", "N0 train metadata"
    )
    prompt_manifest_path = _regular_file(
        checkpoint_root / "norm/PROMPTS.json", "N0 prompt manifest"
    )
    source_normalizer = _regular_file(
        checkpoint_root / f"norm/{task_id}.norm_stat_per_robot.json",
        "N0 task normalizer",
    )
    task_key = serve_task_id(task_id)
    entry = _norm_entry(_read_json(source_normalizer, "N0 task normalizer"), task_key)
    prompt = n0_training_prompt(task_id)
    if not _contains_prompt(
        _read_json(prompt_manifest_path, "N0 prompt manifest"), prompt
    ):
        raise ValueError("N0 prompt manifest does not contain the training prompt")

    serve_pool_root = root / "serve-pools" / task_id
    if serve_pool_root.is_symlink():
        raise ValueError("N0 serve pool cannot be a symlink")
    meta_root = serve_pool_root / "train" / task_key / "meta"
    normalizer_path = serve_pool_root / "norm_stat_per_robot.json"
    _write_exact(normalizer_path, canonical_json_bytes({task_key: dict(entry)}))
    info = {
        "features": {
            "observation.images.top": {"dtype": "video", "shape": [270, 480, 3]},
            "observation.images.wrist_l": {
                "dtype": "video",
                "shape": [270, 480, 3],
            },
            "observation.images.tactile_a": {
                "dtype": "video",
                "shape": [240, 320, 3],
            },
            "observation.images.tactile_b": {
                "dtype": "video",
                "shape": [240, 320, 3],
            },
            "action": {"dtype": "float32", "shape": [10]},
        }
    }
    serve_info_path = meta_root / "info.json"
    serve_tasks_path = meta_root / "tasks.jsonl"
    _write_exact(serve_info_path, canonical_json_bytes(info))
    _write_exact(
        serve_tasks_path,
        canonical_json_bytes({"task": prompt, "task_index": 0}),
    )

    serve_bundle_root = root / "serve-bundle"
    if serve_bundle_root.is_symlink():
        raise ValueError("N0 serve bundle root cannot be a symlink")
    serve_bundle_root.mkdir(exist_ok=True)
    components: dict[str, str] = {}
    base_files: dict[str, dict[str, object]] = {}
    for name, source in (
        ("transformer", checkpoint_root / "transformer"),
        ("vae", base_root / "vae"),
        ("tokenizer", base_root / "tokenizer"),
        ("text_encoder", base_root / "text_encoder"),
    ):
        _ensure_link(source, serve_bundle_root / name)
        components[name] = str(source.resolve(strict=True))
        if name != "transformer":
            base_files.update(_component_files(name, source))
    for name in ("assets", "empty_emb.pt"):
        source = base_root / name
        if source.exists():
            _ensure_link(source, serve_bundle_root / name)
            components[name] = str(source.resolve(strict=True))
            base_files.update(_component_files(name, source))

    manifest_path = serve_pool_root / "serve_bundle_manifest.json"
    manifest: dict[str, Any] = {
        "action_mode": "delta",
        "base_repository": BASE_REPOSITORY,
        "base_revision": BASE_REVISION,
        "base_files": base_files,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "components": components,
        "normalizer_path": str(normalizer_path),
        "prompt": prompt,
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "serve_pool_root": str(serve_pool_root),
        "serve_task_id": task_key,
        "task_id": task_id,
    }
    _write_exact(manifest_path, canonical_json_bytes(manifest))
    return PreparedN0Artifacts(
        bundle_root=root,
        base_root=base_root,
        checkpoint_root=checkpoint_root,
        serve_bundle_root=serve_bundle_root,
        serve_pool_root=serve_pool_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        train_meta_path=train_meta_path,
        normalizer_path=normalizer_path,
        prompt_manifest_path=prompt_manifest_path,
        serve_bundle_manifest_path=manifest_path,
        serve_info_path=serve_info_path,
        serve_tasks_path=serve_tasks_path,
    )


__all__ = [
    "PREPARATION_SCHEMA_VERSION",
    "PreparedN0Artifacts",
    "prepare_official_n0_artifacts",
]

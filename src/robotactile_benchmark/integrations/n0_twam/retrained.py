"""Independent artifacts for the 10 Hz absolute-EE retrained N0 checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, cast

from robotactile_benchmark.backends.univtac_contracts import (
    N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
    load_univtac_task_registry,
)
from robotactile_benchmark.contracts import canonical_hash

SCHEMA = "robotactile-n0-retrained-artifact-v1"
RUNTIME_CONTENT_IDENTITY_SCHEMA = "robotactile-n0-runtime-content-identity-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRAINING_CONTRACT: dict[str, Any] = {
    "action_dim": 20,
    "action_delta_mode": "none",
    "action_norm_method": "q01q99",
    "action_per_frame": 4,
    "pi05_action_horizon": 4,
    "used_action_channel_ids": list(range(10)),
    "use_local_tactile": True,
    "local_tactile_mode": "current",
    "tactile_global_zero": False,
    "obs_cam_keys": ["observation.images.top", "observation.images.wrist_l"],
    "tactile_keys": ["observation.images.tactile_a", "observation.images.tactile_b"],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], data)


def _stamp(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"expected file: {resolved}")
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _metadata_drift_is_preverified(
    artifact: Mapping[str, Any], path: Path, *, checkpoint: bool
) -> bool:
    """Accept metadata-only drift for content hashed immediately before launch."""

    identity = artifact.get("runtime_content_identity")
    if not isinstance(identity, Mapping) or identity.get("schema") != (
        RUNTIME_CONTENT_IDENTITY_SCHEMA
    ):
        return False
    if identity.get("verification") != "sha256_verified_prelaunch":
        return False
    if checkpoint:
        entry = identity.get("checkpoint")
        expected_sha256 = artifact.get("checkpoint_sha256")
    else:
        components = identity.get("base_components")
        if not isinstance(components, Mapping):
            return False
        entry = components.get(str(path))
        expected_sha256 = entry.get("sha256") if isinstance(entry, Mapping) else None
    if not isinstance(entry, Mapping) or not isinstance(expected_sha256, str):
        return False
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return (
        entry.get("path") == str(resolved)
        and entry.get("size") == stat.st_size
        and entry.get("sha256") == expected_sha256
        and _SHA256.fullmatch(expected_sha256) is not None
    )


def _source_identity(
    root: Path, declared: str | None, receipt: Path | None
) -> dict[str, Any]:
    files = sorted((root / "n0_twam").rglob("*.py"))
    if not files or not (root / "n0_twam/n0_twam_server.py").is_file():
        raise ValueError("N0 source root has no inference source")
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in files}
    tree_hash = canonical_hash(hashes)
    status, commit = "declared_unverified", declared
    try:
        git_root = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(git_root).resolve() == root:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if declared and not commit.startswith(declared):
                raise ValueError("declared source commit differs from actual Git HEAD")
            status = "git_head_verified_tree_separately_hashed"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if receipt is not None:
        saved = read_object(receipt)
        receipt_commit = saved.get("source_commit")
        if saved.get("source_tree_sha256") != tree_hash or not isinstance(
            receipt_commit, str
        ):
            raise ValueError("source snapshot receipt does not match source tree")
        if declared and not receipt_commit.startswith(declared):
            raise ValueError("source snapshot receipt commit mismatch")
        if status.startswith("git_") and receipt_commit != commit:
            raise ValueError("source snapshot receipt differs from Git HEAD")
        status, commit = "matched_source_snapshot_receipt", receipt_commit
    return {
        "root": str(root),
        "source_commit": commit,
        "verification": status,
        "source_tree_sha256": tree_hash,
        "python_files": hashes,
    }


def _norm(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError("per-task norm must be an object")
    result: dict[str, list[float]] = {}
    for key in ("q01", "q99"):
        vector = value.get(key)
        if not isinstance(vector, list) or len(vector) != 20:
            raise ValueError("per-task q01/q99 must each contain 20 values")
        result[key] = [float(v) for v in vector]
    if not all(math.isfinite(v) for row in result.values() for v in row):
        raise ValueError("normalizer must be finite")
    if any(hi <= lo for lo, hi in zip(result["q01"], result["q99"])):
        raise ValueError("normalizer q99 must exceed q01")
    if result == {"q01": [-1.0] * 20, "q99": [1.0] * 20}:
        raise ValueError("placeholder normalizer is not a trained normalizer")
    return result


def prepare_retrained(
    *,
    checkpoint: Path,
    base: Path,
    source_root: Path,
    per_repo_norm: Path,
    metadata_root: Path,
    output: Path,
    source_commit: str | None = None,
    source_receipt: Path | None = None,
) -> Path:
    """Freeze eight task identities and a no-clobber bundle; hash weights once."""
    checkpoint, base, source_root = (
        p.resolve(strict=True) for p in (checkpoint, base, source_root)
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite prepared artifact: {output}")
    output = output.absolute()
    meta_path = checkpoint / "train_meta.json"
    meta = read_object(meta_path)
    for key, expected in TRAINING_CONTRACT.items():
        if meta.get(key) != expected:
            raise ValueError(f"retrained training contract mismatch: {key}")
    transformer_config = checkpoint / "transformer/config.json"
    if read_object(transformer_config).get("is_mot") is not True:
        raise ValueError("retrained checkpoint must identify a MOT transformer")
    weights = checkpoint / "transformer/diffusion_pytorch_model.safetensors"
    stamp = _stamp(weights)
    weights_hash = file_sha256(weights)
    if _stamp(weights) != stamp:
        raise ValueError("checkpoint changed while hashing")
    components = {"transformer": checkpoint / "transformer"}
    for name in ("vae", "tokenizer", "text_encoder"):
        member = base / name
        if not member.is_dir():
            raise ValueError(f"missing base component: {member}")
        components[name] = member
    for name in ("assets", "empty_emb.pt"):
        if (base / name).exists():
            components[name] = base / name
    if "empty_emb.pt" not in components:
        raise ValueError("base model must provide empty_emb.pt")
    component_stamps = {
        str(member): _stamp(member)
        for name, path in components.items()
        if name != "transformer"
        for member in (sorted(path.rglob("*")) if path.is_dir() else [path])
        if member.is_file()
    }
    norms = read_object(per_repo_norm)
    tasks = tuple(t.task_id for t in load_univtac_task_registry().tasks)
    if set(norms) != set(tasks):
        raise ValueError("normalization table must cover exactly the eight tasks")
    task_data: dict[str, Any] = {}
    hashes = {
        str(meta_path): file_sha256(meta_path),
        str(transformer_config): file_sha256(transformer_config),
        str(per_repo_norm.resolve(strict=True)): file_sha256(per_repo_norm),
    }
    for task in tasks:
        folder = metadata_root / task / "meta"
        info_path, prompt_path = folder / "info.json", folder / "tasks.jsonl"
        info = read_object(info_path)
        if info.get("fps") != 10:
            raise ValueError(f"task {task} metadata must preserve true 10 Hz")
        features = info.get("features", {})
        for key in (
            TRAINING_CONTRACT["obs_cam_keys"] + TRAINING_CONTRACT["tactile_keys"]
        ):
            if features.get(key, {}).get("dtype") != "video":
                raise ValueError(f"missing trained video feature {task}: {key}")
        if features.get("action", {}).get("shape") != [20]:
            raise ValueError("retrained task action metadata must be 20D")
        rows = [
            json.loads(line)
            for line in prompt_path.read_text().splitlines()
            if line.strip()
        ]
        prompts = {row.get("task") for row in rows}
        if len(prompts) != 1 or not all(
            isinstance(p, str) and p.strip() for p in prompts
        ):
            raise ValueError(f"task {task} must have one unambiguous training prompt")
        norm, prompt = _norm(norms[task]), next(iter(prompts))
        task_data[task] = {
            "prompt": prompt,
            "normalizer": norm,
            "normalizer_sha256": canonical_hash(norm),
        }
        for path in (info_path, prompt_path):
            hashes[str(path.resolve(strict=True))] = file_sha256(path)
    source = _source_identity(source_root, source_commit, source_receipt)
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "serve-bundle"
    bundle.mkdir()
    for name, path in components.items():
        (bundle / name).symlink_to(path, target_is_directory=path.is_dir())
    (bundle / "train_meta.json").symlink_to(meta_path)
    norm_path = output / "norm_stat_absee_per_robot.json"
    _write_json(norm_path, {t: task_data[t]["normalizer"] for t in tasks})
    hashes[str(norm_path)] = file_sha256(norm_path)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA,
        "evidence_level": "retrained_diagnostic",
        "checkpoint_path": str(weights),
        "checkpoint_sha256": weights_hash,
        "checkpoint_stamp": stamp,
        "training_step": meta.get("step"),
        "train_meta_sha256": hashes[str(meta_path)],
        "metadata_hashes": hashes,
        "source": source,
        "source_tree_sha256": source["source_tree_sha256"],
        "bundle_root": str(bundle),
        "normalizer_path": str(norm_path),
        "bundle_components": {name: str(path) for name, path in components.items()},
        "base_component_stamps": component_stamps,
        "training_contract": {
            **TRAINING_CONTRACT,
            "input_color_contract": {
                "live_transform": "identity",
                "decode": "cv2.imdecode numeric RGB, no channel reversal",
                "contract_source": "scripts/n0_twam/hpu_training/data/episode.py:decode_legacy_rgb",
                "verification_scope": "local converter source; recorded training dataset not reverified",
            },
        },
        "action_execution_contract": N0_RETRAINED_10HZ_ACTION_EXECUTION_CONTRACT,
        "action_per_frame": 4,
        "action_hz": 10,
        "tasks": task_data,
    }
    for task in tasks:
        task_data[task]["config_sha256"] = canonical_hash(
            server_overrides(artifact, task)
        )
    artifact["artifact_sha256"] = canonical_hash(artifact)
    path = output / "artifact.json"
    _write_json(path, artifact)
    return path


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_retrained(path: Path) -> dict[str, Any]:
    """Verify frozen metadata/source and weight stat; never rehash the large weights."""
    artifact = read_object(path)
    digest = artifact.pop("artifact_sha256", None)
    if artifact.get("schema_version") != SCHEMA or canonical_hash(artifact) != digest:
        raise ValueError("retrained artifact identity mismatch")
    artifact["artifact_sha256"] = digest
    checkpoint_path = Path(artifact["checkpoint_path"])
    if _stamp(checkpoint_path) != artifact["checkpoint_stamp"] and not (
        _metadata_drift_is_preverified(artifact, checkpoint_path, checkpoint=True)
    ):
        raise ValueError("retrained weights changed since preparation")
    for name, target in artifact["bundle_components"].items():
        if (Path(artifact["bundle_root"]) / name).resolve(strict=True) != Path(
            target
        ).resolve(strict=True):
            raise ValueError("retrained serve bundle target changed")
    for name, stamp in artifact["base_component_stamps"].items():
        path = Path(name)
        if _stamp(path) != stamp and not _metadata_drift_is_preverified(
            artifact, path, checkpoint=False
        ):
            raise ValueError("base component changed since preparation")
    for name, expected in artifact["metadata_hashes"].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"retrained metadata changed: {name}")
    root = Path(artifact["source"]["root"])
    live = _source_identity(root, None, None)
    if live["source_tree_sha256"] != artifact["source_tree_sha256"]:
        raise ValueError("retrained inference source changed")
    for task, entry in artifact["tasks"].items():
        if canonical_hash(server_overrides(artifact, task)) != entry["config_sha256"]:
            raise ValueError("retrained server configuration identity mismatch")
    return artifact


def server_overrides(artifact: dict[str, Any], task: str) -> dict[str, Any]:
    """Exact scientific serve settings, separate from device/port/output choices."""
    entry = artifact["tasks"][task]
    return {
        **TRAINING_CONTRACT,
        "infer_mode": "server",
        "host": "127.0.0.1",
        "wan22_pretrained_model_name_or_path": artifact["bundle_root"],
        "empty_emb_path": str(Path(artifact["bundle_root"]) / "empty_emb.pt"),
        "resume_from": None,
        "init_from": None,
        "serve_task": task,
        "multitask_norm_path": artifact["normalizer_path"],
        "norm_stat": entry["normalizer"],
        "norm_stat_path": artifact["normalizer_path"],
        "prompt": entry["prompt"],
        "eval_prompt": entry["prompt"],
        "frame_chunk_size": 2,
        "height": 256,
        "width": 256,
        "tactile_resize": 128,
        "tactile_latent_height": 8,
        "tactile_latent_width": 8,
        "source_fps": 10,
        "action_hz": 10,
        "frame_stride": 1,
        "pi05_source_action_is_delta": False,
        "pi05_condition_first_frame_zero": True,
        "pi05_rot6d_relative_delta": False,
        "inverse_used_action_channel_ids": list(range(10)) + [10] * 10,
        "server_action_output_format": "absolute",
        "server_return_action_channel_ids": list(range(20)),
        "server_tactile_denoise": True,
        "max_tactile_streams": 4,
        "tactile_sensor_id_map": {
            k: i for i, k in enumerate(TRAINING_CONTRACT["tactile_keys"])
        },
        "num_inference_steps": 3,
        "action_num_inference_steps": 4,
        "guidance_scale": 1,
        "action_guidance_scale": 1,
        "video_exec_step": -1,
        "cfg_prob": 0.0,
        "tactile_cfg_prob": 0.0,
        "noisy_cond_prob_tactile": 0.0,
        "deterministic_episode_seed": True,
        "cold_seed_mode": "free",
        "delta_smooth": False,
        "enable_wandb": False,
    }

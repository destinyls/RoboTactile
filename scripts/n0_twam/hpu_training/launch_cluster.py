#!/usr/bin/env python3
"""Preflight and launch one source-pinned multi-node N0-TWAM torchrun."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import IO, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpu_training.contract import (  # noqa: E402
    ALLOWED_OFFICIAL_CHANGE,
    PROVEN_DOCKER_IMAGE,
    PROVEN_DOCKER_IMAGE_ID,
    PROVEN_RCCL_PLUGIN_FILE,
    PROVEN_RCCL_PLUGIN_SHA256,
    RUNTIME_PYTHON_OVERLAYS,
    RUNTIME_PYTHON_PATH,
    ensure_paths_exist,
    load_cluster_spec,
    load_training_spec,
    sha256_file,
    validate_run_id,
    verify_official_checkout,
)
from hpu_training.latent.contracts import (  # noqa: E402
    LatentSpec,
    load_certified_train759,
)
from hpu_training.latent.inventory import validate_training_inventory  # noqa: E402
from hpu_training.launch_profiles import resolve_launch_profile  # noqa: E402
from hpu_training.remote_execution import (  # noqa: E402
    DOCKER_RUNTIME,
    FLEX25_COMPAT,
    FSDP_NAMESPACE_SHIM,
    N0_TRAIN_COMPAT,
    RANK_ENTRYPOINT,
    launch_all,
    preflight_all,
)
from hpu_training.render_config import render_posttrain_config  # noqa: E402


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training-spec", type=Path, required=True)
    parser.add_argument("--cluster-spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--launch-mode",
        choices=("single-card-smoke", "distributed-smoke", "formal"),
        default="formal",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=None,
        help="1-20 optimizer steps; only valid for distributed-smoke",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = validate_run_id(args.run_id)
    repo = args.repo.resolve()
    run_root = args.run_root.resolve()
    training_spec_path = args.training_spec.resolve()
    cluster_spec_path = args.cluster_spec.resolve()
    training = load_training_spec(training_spec_path)
    cluster = load_cluster_spec(cluster_spec_path)
    profile = resolve_launch_profile(
        mode=args.launch_mode,
        cluster=cluster,
        training_steps=training.num_steps,
        smoke_steps=args.smoke_steps,
    )
    effective_training = replace(training, num_steps=profile.num_steps)
    verify_official_checkout(repo)
    source_manifest_path = training.dataset_path.parent / "source_split_manifest.json"
    conversion_receipt_path = training.dataset_path.parent / "train759_receipt.json"
    lock_path = repo / ".git" / "robotactile-hpu-training.lock"
    lock_handle: IO[bytes] = lock_path.open("a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"another RoboTactile training launch owns this checkout: {lock_path}"
        ) from exc
    ensure_paths_exist(
        (
            ("dataset_path", training.dataset_path),
            ("base_model_path", training.base_model_path),
            ("released_checkpoint_path", training.released_checkpoint_path),
            ("empty_embedding_path", training.empty_embedding_path),
            ("norm_stat_path", training.norm_stat_path),
            ("source_manifest_path", source_manifest_path),
            ("conversion_receipt_path", conversion_receipt_path),
            ("latent_inventory_path", training.latent_inventory_path),
            ("docker_runtime", DOCKER_RUNTIME),
            ("rank_entrypoint", RANK_ENTRYPOINT),
            ("fsdp_namespace_shim", FSDP_NAMESPACE_SHIM),
            ("flex25_compat", FLEX25_COMPAT),
            ("n0_train_compat", N0_TRAIN_COMPAT),
        )
    )
    ensure_paths_exist((("per_repo_norm_stat_path", training.per_repo_norm_stat_path),))
    records = load_certified_train759(
        LatentSpec(
            dataset_root=training.dataset_path,
            source_manifest_path=source_manifest_path,
            conversion_receipt_path=conversion_receipt_path,
            official_repo=repo,
            model_path=training.base_model_path,
        )
    )
    latent_inventory = validate_training_inventory(
        path=training.latent_inventory_path,
        records=records,
        source_manifest_path=source_manifest_path,
        conversion_receipt_path=conversion_receipt_path,
    )
    latent_inventory_file_sha256 = sha256_file(training.latent_inventory_path)
    conversion_payload = json.loads(conversion_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(conversion_payload, dict) or not isinstance(
        conversion_payload.get("normalization"), dict
    ):
        raise ValueError("conversion receipt normalization contract is missing")
    normalization = conversion_payload["normalization"]
    assert isinstance(normalization, dict)
    certified_norm = training.dataset_path.parent / str(normalization["norm_stat_path"])
    certified_per_repo_norm = training.dataset_path.parent / str(
        normalization["per_repo_norm_stat_path"]
    )
    if training.norm_stat_path.resolve(strict=True) != certified_norm.resolve(
        strict=True
    ):
        raise ValueError(
            "training norm_stat_path differs from the certified conversion"
        )
    if training.per_repo_norm_stat_path.resolve(
        strict=True
    ) != certified_per_repo_norm.resolve(strict=True):
        raise ValueError(
            "training per_repo_norm_stat_path differs from the certified conversion"
        )

    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    save_root = run_dir / "model"
    config_receipt = render_posttrain_config(
        repo=repo, spec=effective_training, save_root=save_root
    )
    config_path = repo / ALLOWED_OFFICIAL_CHANGE
    config_sha256 = sha256_file(config_path)
    manifest: dict[str, object] = {
        "run_id": run_id,
        "launch_mode": profile.mode,
        "status": "preflighting",
        "official_source": verify_official_checkout(repo),
        "training_spec_path": str(training_spec_path),
        "training_spec_sha256": sha256_file(training_spec_path),
        "cluster_spec_path": str(cluster_spec_path),
        "cluster_spec_sha256": sha256_file(cluster_spec_path),
        "config_receipt": config_receipt,
        "certified_training_data": {
            "source_episode_count": len(records),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "conversion_receipt_path": str(conversion_receipt_path),
            "conversion_receipt_sha256": sha256_file(conversion_receipt_path),
            "latent_inventory": latent_inventory,
            "latent_inventory_file_sha256": latent_inventory_file_sha256,
            "norm_stat_path": str(training.norm_stat_path),
            "per_repo_norm_stat_path": str(training.per_repo_norm_stat_path),
        },
        "nodes": list(profile.nodes),
        "processes_per_node": profile.processes_per_node,
        "world_size": profile.world_size,
        "effective_num_steps": profile.num_steps,
        "requested_num_steps": training.num_steps,
        "max_latent_frames": training.max_latent_frames,
        "fsdp_topology": profile.fsdp_topology,
        "fsdp_shard_size": profile.fsdp_shard_size,
        "save_root": str(save_root),
        "checkout_lock": str(lock_path),
        "container_runtime": {
            "image": PROVEN_DOCKER_IMAGE,
            "image_id": PROVEN_DOCKER_IMAGE_ID,
            "rccl_plugin_file": str(PROVEN_RCCL_PLUGIN_FILE),
            "rccl_plugin_sha256": PROVEN_RCCL_PLUGIN_SHA256,
            "python_path": str(RUNTIME_PYTHON_PATH),
            "python_overlays": [str(path) for path in RUNTIME_PYTHON_OVERLAYS],
            "transient_no_clobber_containers": True,
            "docker_runtime_sha256": sha256_file(DOCKER_RUNTIME),
            "rank_entrypoint_sha256": sha256_file(RANK_ENTRYPOINT),
            "fsdp_namespace_shim_sha256": sha256_file(FSDP_NAMESPACE_SHIM),
            "flex25_compat_sha256": sha256_file(FLEX25_COMPAT),
            "n0_train_compat_sha256": sha256_file(N0_TRAIN_COMPAT),
        },
    }
    status_path = run_dir / "status.json"
    _write_json(status_path, manifest)

    try:
        preflight_all(
            cluster=cluster,
            profile=profile,
            repo=repo,
            save_root=save_root,
            latent_inventory_path=training.latent_inventory_path,
            latent_inventory_sha256=latent_inventory_file_sha256,
            config_sha256=config_sha256,
            log_dir=log_dir,
            run_id=run_id,
        )
        if args.preflight_only:
            manifest["status"] = "preflight_complete"
            _write_json(status_path, manifest)
            return 0
        manifest["status"] = "running"
        _write_json(status_path, manifest)
        returncodes = launch_all(
            cluster=cluster,
            profile=profile,
            repo=repo,
            save_root=save_root,
            latent_inventory_path=training.latent_inventory_path,
            latent_inventory_sha256=latent_inventory_file_sha256,
            config_sha256=config_sha256,
            log_dir=log_dir,
            run_id=run_id,
        )
        manifest["node_returncodes"] = returncodes
        manifest["status"] = (
            "complete" if all(code == 0 for code in returncodes) else "failed"
        )
        _write_json(status_path, manifest)
        return 0 if manifest["status"] == "complete" else 1
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(status_path, manifest)
        raise


if __name__ == "__main__":
    sys.exit(main())

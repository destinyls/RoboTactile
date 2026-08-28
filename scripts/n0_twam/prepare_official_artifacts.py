#!/usr/bin/env python3
"""Download pinned N0-TWAM weights and prepare one or all frozen tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.integrations.configuration import (
    GeneratedIntegrationConfiguration,
    configure_n0_twam_integration,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    BASE_REPOSITORY,
    BASE_REVISION,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
)
from robotactile_benchmark.integrations.n0_twam.preparation import (
    prepare_official_n0_artifacts,
)
from robotactile_benchmark.integrations.registry import (
    load_model_integration_config,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)


def _download(root: Path, workers: int) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as error:
        raise SystemExit(
            "huggingface-hub is unavailable; run install_official_runtime.sh first"
        ) from error
    snapshot_download(
        repo_id=BASE_REPOSITORY,
        revision=BASE_REVISION,
        local_dir=root / "base",
        allow_patterns=(
            "assets/**",
            "empty_emb.pt",
            "text_encoder/**",
            "tokenizer/**",
            "vae/**",
        ),
        max_workers=workers,
    )
    snapshot_download(
        repo_id=CHECKPOINT_REPOSITORY,
        revision=CHECKPOINT_REVISION,
        local_dir=root / "univtac-delta",
        allow_patterns=("norm/**", "train_meta.json", "transformer/**"),
        max_workers=workers,
    )


def _prepare_task(
    *, root: Path, task_id: str, device: str, config_label: Optional[str] = None
) -> GeneratedIntegrationConfiguration:
    config_name = task_id if config_label is None else f"{task_id}-{config_label}"
    config_root = root / "configs" / config_name
    manifest_path = config_root / "artifact_manifest.json"
    config_path = config_root / "integration_config.json"
    if manifest_path.exists() != config_path.exists():
        raise ValueError("N0 task configuration pair is incomplete")
    if manifest_path.exists():
        runtime = resolve_n0_runtime_artifacts(config_path)
        if runtime.manifest.task_id != task_id:
            raise ValueError("existing N0 task configuration identity mismatch")
        integration = load_model_integration_config("n0_twam", config_path)
        if integration.device != device:
            raise ValueError("existing N0 task configuration device mismatch")
        return GeneratedIntegrationConfiguration(
            integration_id="n0_twam",
            artifact_manifest_path=manifest_path.absolute(),
            integration_config_path=config_path.absolute(),
            artifact_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )
    prepared = prepare_official_n0_artifacts(
        bundle_root=root,
        task_id=task_id,
    )
    return configure_n0_twam_integration(
        bundle_root=prepared.bundle_root,
        task_id=task_id,
        base_root=prepared.base_root,
        checkpoint_root=prepared.checkpoint_root,
        serve_bundle_root=prepared.serve_bundle_root,
        serve_pool_root=prepared.serve_pool_root,
        checkpoint_path=prepared.checkpoint_path,
        model_config_path=prepared.config_path,
        train_meta_path=prepared.train_meta_path,
        normalizer_path=prepared.normalizer_path,
        prompt_manifest_path=prepared.prompt_manifest_path,
        serve_bundle_manifest_path=prepared.serve_bundle_manifest_path,
        serve_info_path=prepared.serve_info_path,
        serve_tasks_path=prepared.serve_tasks_path,
        manifest_path=manifest_path,
        config_path=config_path,
        device=device,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-label")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    root = args.root.absolute()
    if not root.is_absolute() or root in {Path("/"), Path("/data"), Path("/data1")}:
        raise SystemExit("--root must be a narrow absolute artifact directory")
    if args.workers < 1 or args.workers > 16:
        raise SystemExit("--workers must be in [1,16]")
    if (
        args.config_label is not None
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", args.config_label) is None
    ):
        raise SystemExit("--config-label must be a safe lowercase identifier")
    registry_tasks = tuple(task.task_id for task in load_registry().tasks)
    if args.all_tasks == (args.tasks is not None):
        raise SystemExit("select exactly one of --all-tasks or one/more --task")
    tasks = registry_tasks if args.all_tasks else tuple(args.tasks)
    if len(tasks) != len(set(tasks)) or set(tasks) - set(registry_tasks):
        raise SystemExit("tasks must be unique frozen UniVTAC task IDs")
    if root.is_symlink():
        raise SystemExit("--root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        _download(root, args.workers)
    generated = tuple(
        _prepare_task(
            root=root,
            task_id=task_id,
            device=args.device,
            config_label=args.config_label,
        )
        for task_id in tasks
    )
    payload: object = (
        generated[0].to_dict()
        if len(generated) == 1
        else {
            "evidence_level": "local_artifact_configuration_only_v1",
            "live_inference_claimed": False,
            "task_count": len(generated),
            "tasks": [item.to_dict() for item in generated],
        }
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

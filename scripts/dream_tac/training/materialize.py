"""Resumable train759-only Dream-Tac dataset materialization CLI."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.n0_twam.hpu_training.data.contracts import (
    EXPECTED_SPLIT_COUNTS,
    SOURCE_MANIFEST_NAME,
    TASKS,
    SourceEpisode,
    discover_source_episodes,
    sha256_file,
)
from scripts.n0_twam.hpu_training.data.manifest_io import (
    atomic_json_no_clobber,
    load_json_object,
    load_or_create_source_manifest,
    receipt_sha256,
    write_or_verify_json,
)

from .contracts import (
    DATASET_DIR_NAME,
    DATASET_STATS_NAME,
    EPISODE_RECEIPT_NAME,
    GLOBAL_RECEIPT_NAME,
    PROMPT_MANIFEST_NAME,
    T5_REQUEST_NAME,
    build_prompt_manifest,
    build_t5_cache_request,
    train759_records,
    validate_prompt_manifest,
)
from .episode_io import (
    Hdf5Writer,
    RootOpener,
    VideoWriter,
    export_episode,
    open_hdf5_root,
    write_mp4,
    write_official_hdf5,
)
from .receipts import (
    ArtifactVerifier,
    build_episode_receipt,
    build_global_receipt,
    validate_episode_destination,
)
from .statistics import (
    derive_post_normalization_statistics,
    statistics_from_records,
    validate_dataset_statistics,
    validate_post_normalization_statistics,
)
from .training_constants import DATASET_POST_NORM_STATS_NAME


def episode_destination(output_root: Path, record: SourceEpisode) -> Path:
    """Return the recursive path discovered by upstream ``is_train=True``."""

    return (
        output_root
        / DATASET_DIR_NAME
        / "train"
        / record.task
        / f"episode_{record.episode_id:03d}"
    )


def materialize_episode(
    *,
    record: SourceEpisode,
    output_root: Path,
    source_manifest_sha256: str,
    video_writer: VideoWriter = write_mp4,
    hdf5_writer: Hdf5Writer = write_official_hdf5,
    root_opener: RootOpener = open_hdf5_root,
    artifact_verifier: ArtifactVerifier,
) -> dict[str, object]:
    """Publish one episode transaction without overwriting any prior artifact."""

    if record.split != "train":
        raise ValueError("Dream-Tac refuses to materialize non-training episodes")
    destination = episode_destination(output_root, record)
    if destination.exists():
        return validate_episode_destination(
            destination=destination,
            record=record,
            source_manifest_sha256=source_manifest_sha256,
            artifact_verifier=artifact_verifier,
        )
    attempt = (
        output_root
        / "work"
        / record.task
        / f"episode-{record.episode_id:03d}-{uuid.uuid4().hex}"
    )
    attempt.parent.mkdir(parents=True, exist_ok=True)
    try:
        export = export_episode(
            record,
            staging_root=attempt,
            prompt=_prompt(record),
            video_writer=video_writer,
            hdf5_writer=hdf5_writer,
            root_opener=root_opener,
        )
        receipt = build_episode_receipt(
            record=record,
            source_manifest_sha256=source_manifest_sha256,
            export=export,
        )
        atomic_json_no_clobber(attempt / EPISODE_RECEIPT_NAME, receipt)
        validate_episode_destination(
            destination=attempt,
            record=record,
            source_manifest_sha256=source_manifest_sha256,
            artifact_verifier=artifact_verifier,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"episode destination appeared: {destination}")
        attempt.rename(destination)
        return receipt
    except Exception as exc:
        if attempt.is_dir():
            failure = attempt / "_robotactile_failure.json"
            if not failure.exists():
                atomic_json_no_clobber(
                    failure,
                    {
                        "status": "failed_before_publish",
                        "task": record.task,
                        "episode_id": record.episode_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        raise


def _prompt(record: SourceEpisode) -> str:
    from scripts.n0_twam.hpu_training.data.contracts import TASK_PROMPTS

    return TASK_PROMPTS[record.task]


def dry_run_plan(*, raw_root: Path, output_root: Path) -> dict[str, object]:
    """Validate the source grid and report a plan without writing or hashing it."""

    records = discover_source_episodes(raw_root, hash_sources=False)
    selected = train759_records(records)
    return {
        "status": "dry_run",
        "writes_performed": False,
        "raw_root": str(raw_root.resolve(strict=True)),
        "output_root": str(output_root.resolve()),
        "source_episode_count": len(records),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "materialized_split": "train",
        "planned_episode_count": len(selected),
        "planned_tasks": list(TASKS),
        "creates_validation_tree": False,
    }


def _load_contracts(
    *, raw_root: Path, output_root: Path
) -> tuple[str, tuple[SourceEpisode, ...], dict[str, object], dict[str, object]]:
    manifest, records = load_or_create_source_manifest(
        raw_root=raw_root, output_root=output_root
    )
    manifest_sha = manifest.get("manifest_sha256")
    if not isinstance(manifest_sha, str):
        raise ValueError("source manifest has no SHA256")
    selected = train759_records(records)
    prompts = build_prompt_manifest(manifest_sha)
    request = build_t5_cache_request(prompt_manifest=prompts, output_root=output_root)
    write_or_verify_json(output_root / PROMPT_MANIFEST_NAME, prompts)
    write_or_verify_json(output_root / T5_REQUEST_NAME, request)
    return manifest_sha, selected, prompts, request


def write_statistics_artifacts(
    *, output_root: Path, statistics: Mapping[str, object]
) -> tuple[Path, Path]:
    """Publish the source and deterministic post-normalization statistics."""

    validated = validate_dataset_statistics(statistics)
    dataset_root = output_root / DATASET_DIR_NAME
    stats_path = dataset_root / DATASET_STATS_NAME
    post_stats_path = dataset_root / DATASET_POST_NORM_STATS_NAME
    write_or_verify_json(stats_path, validated)
    write_or_verify_json(
        post_stats_path, derive_post_normalization_statistics(validated)
    )
    return stats_path, post_stats_path


def materialize_train759(
    *,
    raw_root: Path,
    output_root: Path,
    tasks: Sequence[str] = TASKS,
    video_writer: VideoWriter = write_mp4,
    hdf5_writer: Hdf5Writer = write_official_hdf5,
    root_opener: RootOpener = open_hdf5_root,
    artifact_verifier: ArtifactVerifier,
) -> dict[str, object]:
    """Materialize selected task transactions and finalize only at 759/759."""

    requested = tuple(tasks)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(task not in TASKS for task in requested)
    ):
        raise ValueError("tasks must be a unique non-empty canonical subset")
    manifest_sha, records, prompts, request = _load_contracts(
        raw_root=raw_root, output_root=output_root
    )
    requested_set = set(requested)
    completed: dict[tuple[str, int], dict[str, object]] = {}
    for record in records:
        destination = episode_destination(output_root, record)
        if record.task in requested_set:
            completed[(record.task, record.episode_id)] = materialize_episode(
                record=record,
                output_root=output_root,
                source_manifest_sha256=manifest_sha,
                video_writer=video_writer,
                hdf5_writer=hdf5_writer,
                root_opener=root_opener,
                artifact_verifier=artifact_verifier,
            )
        elif destination.exists():
            completed[(record.task, record.episode_id)] = validate_episode_destination(
                destination=destination,
                record=record,
                source_manifest_sha256=manifest_sha,
                artifact_verifier=artifact_verifier,
            )
    if len(completed) != len(records):
        complete_tasks = [
            task
            for task in TASKS
            if sum(key[0] == task for key in completed)
            == sum(record.task == task for record in records)
        ]
        return {
            "status": "partial",
            "source_manifest_sha256": manifest_sha,
            "completed_episode_count": len(completed),
            "remaining_episode_count": len(records) - len(completed),
            "completed_tasks": complete_tasks,
        }
    stats, sample_count = statistics_from_records(records, root_opener=root_opener)
    stats_path, _ = write_statistics_artifacts(
        output_root=output_root, statistics=stats
    )
    episode_receipts = [
        completed[(record.task, record.episode_id)] for record in records
    ]
    receipt = build_global_receipt(
        source_manifest_sha256=manifest_sha,
        prompt_manifest_sha256=str(prompts["prompt_manifest_sha256"]),
        t5_request_sha256=str(request["t5_request_sha256"]),
        statistics_sha256=sha256_file(stats_path),
        statistics_sample_count=sample_count,
        episode_receipts=episode_receipts,
    )
    write_or_verify_json(output_root / GLOBAL_RECEIPT_NAME, receipt)
    return receipt


def verify_materialization(
    *,
    raw_root: Path,
    output_root: Path,
    artifact_verifier: ArtifactVerifier,
) -> dict[str, object]:
    """Read-only full verification; never creates a missing artifact."""

    required = (
        output_root / SOURCE_MANIFEST_NAME,
        output_root / PROMPT_MANIFEST_NAME,
        output_root / T5_REQUEST_NAME,
        output_root / GLOBAL_RECEIPT_NAME,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Dream-Tac materialization is incomplete")
    manifest, records = load_or_create_source_manifest(
        raw_root=raw_root, output_root=output_root
    )
    manifest_sha = manifest.get("manifest_sha256")
    if not isinstance(manifest_sha, str):
        raise ValueError("source manifest has no SHA256")
    selected = train759_records(records)
    prompts = load_json_object(output_root / PROMPT_MANIFEST_NAME)
    validate_prompt_manifest(prompts, source_manifest_sha256=manifest_sha)
    receipts = [
        validate_episode_destination(
            destination=episode_destination(output_root, record),
            record=record,
            source_manifest_sha256=manifest_sha,
            artifact_verifier=artifact_verifier,
        )
        for record in selected
    ]
    global_receipt = load_json_object(output_root / GLOBAL_RECEIPT_NAME)
    receipt_sha256(global_receipt, field="receipt_sha256")
    if len(receipts) != 759 or global_receipt.get("materialized_episode_count") != 759:
        raise ValueError("Dream-Tac global receipt is incomplete")
    stats_path = output_root / DATASET_DIR_NAME / DATASET_STATS_NAME
    if sha256_file(stats_path) != global_receipt.get("dataset_statistics_sha256"):
        raise ValueError("Dream-Tac dataset statistics identity mismatch")
    source_stats = validate_dataset_statistics(load_json_object(stats_path))
    post_stats_path = output_root / DATASET_DIR_NAME / DATASET_POST_NORM_STATS_NAME
    validate_post_normalization_statistics(
        source_stats, load_json_object(post_stats_path)
    )
    if (output_root / DATASET_DIR_NAME / "val").exists():
        raise ValueError("Dream-Tac output must not contain a validation split")
    return {
        "status": "verified",
        "materialized_episode_count": len(receipts),
        "source_manifest_sha256": manifest_sha,
        "dataset_statistics_post_norm_sha256": sha256_file(post_stats_path),
        "training_ready": global_receipt.get("training_ready"),
        "t5_cache_status": global_receipt.get("t5_cache_status"),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from .episode_io import verify_official_hdf5

    args = _parse_args(argv)
    if args.dry_run:
        result = dry_run_plan(raw_root=args.raw_root, output_root=args.output_root)
    elif args.verify:
        result = verify_materialization(
            raw_root=args.raw_root,
            output_root=args.output_root,
            artifact_verifier=verify_official_hdf5,
        )
    else:
        result = materialize_train759(
            raw_root=args.raw_root,
            output_root=args.output_root,
            tasks=args.tasks,
            artifact_verifier=verify_official_hdf5,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "dry_run_plan",
    "episode_destination",
    "main",
    "materialize_episode",
    "materialize_train759",
    "verify_materialization",
    "write_statistics_artifacts",
]

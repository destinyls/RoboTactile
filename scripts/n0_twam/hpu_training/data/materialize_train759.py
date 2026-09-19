"""Materialize only UniVTAC train759 for official N0-TWAM post-training."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

if __package__ in (None, ""):
    # A direct ``python path/to/materialize_train759.py`` invocation otherwise
    # leaves this ``data`` directory first on sys.path.  Its local
    # ``lerobot.py`` would then shadow the third-party ``lerobot`` package when
    # the writer imports LeRobotDataset.
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path if Path(entry or ".").resolve() != script_directory
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from hpu_training.data.contracts import SOURCE_MANIFEST_NAME, TASKS, SourceEpisode
    from hpu_training.data.lerobot import (
        verify_lerobot_payload,
        write_task_repo,
    )
    from hpu_training.data.manifest_io import (
        atomic_json_no_clobber,
        load_or_create_source_manifest,
        select_train_records,
        write_or_verify_json,
    )
    from hpu_training.data.receipts import (
        GLOBAL_RECEIPT_NAME,
        TASK_RECEIPT_NAME,
        build_global_receipt,
        build_task_receipt,
        validate_task_repo,
    )
    from hpu_training.data.stats import (
        write_normalization_artifacts,
    )
else:
    from .contracts import SOURCE_MANIFEST_NAME, TASKS, SourceEpisode
    from .lerobot import verify_lerobot_payload, write_task_repo
    from .manifest_io import (
        atomic_json_no_clobber,
        load_or_create_source_manifest,
        select_train_records,
        write_or_verify_json,
    )
    from .receipts import (
        GLOBAL_RECEIPT_NAME,
        TASK_RECEIPT_NAME,
        build_global_receipt,
        build_task_receipt,
        validate_task_repo,
    )
    from .stats import write_normalization_artifacts

TaskWriter = Callable[[Sequence[SourceEpisode], Path, int], Mapping[str, object]]


def _default_writer(
    records: Sequence[SourceEpisode], output_root: Path, image_writer_threads: int
) -> Mapping[str, object]:
    return write_task_repo(
        records,
        output_root=output_root,
        image_writer_threads=image_writer_threads,
    )


def materialize_task(
    *,
    task: str,
    records: Sequence[SourceEpisode],
    output_root: Path,
    manifest_sha256: str,
    image_writer_threads: int,
    writer: TaskWriter = _default_writer,
) -> dict[str, object]:
    """Publish one task atomically; completed task repos are validated and skipped."""

    destination = output_root / "train759" / task
    if destination.exists():
        return validate_task_repo(
            destination=destination,
            records=records,
            manifest_sha256=manifest_sha256,
        )
    attempt = output_root / "work" / task / f"attempt-{uuid.uuid4().hex}"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = writer(records, attempt, image_writer_threads)
        verify_lerobot_payload(attempt, expected_episodes=len(records))
        receipt = build_task_receipt(
            task=task,
            repo_relative_path=f"train759/{task}",
            manifest_sha256=manifest_sha256,
            records=records,
            writer_result=result,
        )
        atomic_json_no_clobber(attempt / TASK_RECEIPT_NAME, receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                f"task destination appeared concurrently: {destination}"
            )
        attempt.rename(destination)
        return validate_task_repo(
            destination=destination,
            records=records,
            manifest_sha256=manifest_sha256,
        )
    except Exception as exc:
        if attempt.is_dir():
            failure = {
                "schema_version": 1,
                "status": "failed_before_publish",
                "task": task,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failure_path = attempt / "_robotactile_failure.json"
            if not failure_path.exists():
                atomic_json_no_clobber(failure_path, failure)
        raise


def freeze_source_manifest(*, raw_root: Path, output_root: Path) -> dict[str, object]:
    """Freeze/validate all 800 sources without creating any train repository."""

    manifest, records = load_or_create_source_manifest(
        raw_root=raw_root, output_root=output_root
    )
    return {
        "status": "manifest_ready",
        "source_manifest_path": str(output_root / SOURCE_MANIFEST_NAME),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_episode_count": len(records),
        "split_counts": manifest["split_counts"],
        "materialized_task_count": 0,
    }


def materialize_train759(
    *,
    raw_root: Path,
    output_root: Path,
    tasks: Sequence[str] = TASKS,
    image_writer_threads: int = 8,
    writer: TaskWriter = _default_writer,
) -> dict[str, object]:
    """Execute resumable per-task transactions and finalize only at 759/759."""

    selected = tuple(tasks)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(task not in TASKS for task in selected)
    ):
        raise ValueError("tasks must be a unique non-empty UniVTAC task subset")
    if image_writer_threads <= 0:
        raise ValueError("image_writer_threads must be positive")
    manifest, records = load_or_create_source_manifest(
        raw_root=raw_root, output_root=output_root
    )
    manifest_sha = manifest.get("manifest_sha256")
    if not isinstance(manifest_sha, str):
        raise ValueError("source manifest has no SHA256")
    selected_receipts: dict[str, dict[str, object]] = {}
    for task in selected:
        task_records = select_train_records(records, task=task)
        selected_receipts[task] = materialize_task(
            task=task,
            records=task_records,
            output_root=output_root,
            manifest_sha256=manifest_sha,
            image_writer_threads=image_writer_threads,
            writer=writer,
        )

    all_receipts: dict[str, dict[str, object]] = {}
    for task in TASKS:
        destination = output_root / "train759" / task
        if not destination.exists():
            continue
        all_receipts[task] = validate_task_repo(
            destination=destination,
            records=select_train_records(records, task=task),
            manifest_sha256=manifest_sha,
        )
    if set(all_receipts) != set(TASKS):
        return {
            "status": "partial",
            "source_manifest_sha256": manifest_sha,
            "completed_tasks": [task for task in TASKS if task in all_receipts],
            "remaining_tasks": [task for task in TASKS if task not in all_receipts],
        }

    train_root = output_root / "train759"
    actual_dirs = {path.name for path in train_root.iterdir() if path.is_dir()}
    if actual_dirs != set(TASKS):
        raise ValueError(
            f"train759 contains unexpected repository directories: {actual_dirs}"
        )
    task_roots = {task: train_root / task for task in TASKS}
    normalization = write_normalization_artifacts(task_roots, output_root=output_root)
    receipt = build_global_receipt(
        output_root=output_root,
        manifest=manifest,
        task_receipts=all_receipts,
        normalization=normalization,
    )
    write_or_verify_json(output_root / GLOBAL_RECEIPT_NAME, receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize only UniVTAC train759 for official N0-TWAM"
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="freeze/validate the 800-source split manifest, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.manifest_only:
        result = freeze_source_manifest(
            raw_root=args.raw_root,
            output_root=args.output_root,
        )
    else:
        result = materialize_train759(
            raw_root=args.raw_root,
            output_root=args.output_root,
            tasks=args.tasks,
            image_writer_threads=args.image_writer_threads,
        )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GLOBAL_RECEIPT_NAME",
    "TASK_RECEIPT_NAME",
    "freeze_source_manifest",
    "main",
    "materialize_task",
    "materialize_train759",
    "validate_task_repo",
]

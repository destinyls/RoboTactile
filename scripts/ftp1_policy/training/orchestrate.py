"""Prepare train759 and train one joint eight-task FTP-1 model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .artifacts import completed_tasks, prepared_tasks
from .contracts import TASK_IDS, FTP1TrainingRequest
from .pipeline import (
    normalize_joint,
    parse_task,
    preflight,
    stage_sources,
    train_joint,
)
from .runtime import signed, write_or_verify, write_status


def run(
    request: FTP1TrainingRequest,
    *,
    phase: str,
    task: str | None,
    verify_source_hashes: bool,
) -> None:
    request.output_root.mkdir(parents=True, exist_ok=True)
    write_or_verify(
        request.output_root / "request_receipt.json",
        signed(
            {"request": request.to_dict(), "request_sha256": request.request_sha256},
            "request_receipt_sha256",
        ),
    )
    if task is not None and phase != "parse":
        raise ValueError("--task is only valid with --phase parse")
    if phase == "parse" and task is None:
        raise ValueError("--phase parse requires --task")
    active_phase = phase
    active_task = task
    try:
        if phase in {"preflight", "all"}:
            active_phase = "preflight"
            active_task = None
            write_status(
                request,
                phase="preflight",
                task=None,
                completed_tasks=completed_tasks(request),
                prepared_tasks=prepared_tasks(request),
            )
            preflight(request)
        if phase in {"stage", "parse", "prepare", "all"}:
            active_phase = "stage"
            active_task = None
            write_status(
                request,
                phase="stage",
                task=None,
                completed_tasks=completed_tasks(request),
                prepared_tasks=prepared_tasks(request),
            )
            stage_sources(request, verify_hashes=verify_source_hashes)
        if phase in {"prepare", "all"}:
            for target in request.tasks:
                active_phase = "parse"
                active_task = target
                write_status(
                    request,
                    phase="parse",
                    task=target,
                    completed_tasks=completed_tasks(request),
                    prepared_tasks=prepared_tasks(request),
                )
                parse_task(request, target)
            active_phase = "norm"
            active_task = None
            write_status(
                request,
                phase="norm",
                task=None,
                completed_tasks=completed_tasks(request),
                prepared_tasks=prepared_tasks(request),
            )
            config_path = normalize_joint(request)
            if phase == "all":
                active_phase = "train"
                write_status(
                    request,
                    phase="train",
                    task=None,
                    completed_tasks=completed_tasks(request),
                    prepared_tasks=prepared_tasks(request),
                )
                train_joint(request, config_path)
        elif phase == "parse":
            assert task is not None
            active_phase = "parse"
            active_task = task
            write_status(
                request,
                phase="parse",
                task=task,
                completed_tasks=completed_tasks(request),
                prepared_tasks=prepared_tasks(request),
            )
            parse_task(request, task)
        elif phase == "train":
            active_phase = "train"
            active_task = None
            config_path = request.output_root / "configs" / "joint_all8.json"
            if not config_path.is_file():
                raise ValueError("joint dataset config is absent; run prepare first")
            write_status(
                request,
                phase="train",
                task=None,
                completed_tasks=completed_tasks(request),
                prepared_tasks=prepared_tasks(request),
            )
            train_joint(request, config_path)
        completed = completed_tasks(request)
        final_phase = (
            "complete" if len(completed) == len(request.tasks) else f"{phase}_complete"
        )
        write_status(
            request,
            phase=final_phase,
            task=None,
            completed_tasks=completed,
            prepared_tasks=prepared_tasks(request),
        )
    except Exception as exc:
        write_status(
            request,
            phase=active_phase,
            task=active_task,
            completed_tasks=completed_tasks(request),
            prepared_tasks=prepared_tasks(request),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("preflight", "stage", "parse", "prepare", "train", "all"),
        required=True,
    )
    parser.add_argument("--task", choices=TASK_IDS)
    parser.add_argument("--verify-source-hashes", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parser().parse_args()
    run(
        FTP1TrainingRequest.load(args.request),
        phase=args.phase,
        task=args.task,
        verify_source_hashes=args.verify_source_hashes,
    )


if __name__ == "__main__":
    main()

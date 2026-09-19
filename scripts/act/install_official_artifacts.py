#!/usr/bin/env python3
"""Install hash-pinned official UniVTAC ACT artifacts from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.integrations.act.official_release import (
    OFFICIAL_ACT_PROFILES,
    OFFICIAL_ACT_TASKS,
    build_official_act_install_plan,
    install_official_act_release,
    load_official_act_release_lock,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("deployment/artifacts/models/act"),
        help="canonical ACT model artifact root",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=OFFICIAL_ACT_TASKS,
        help="task to install; repeat as needed (default: all eight)",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=OFFICIAL_ACT_PROFILES,
        help="profile to install; repeat as needed (default: univtac)",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        help="explicit canonical act_artifacts.lock.json",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="also install upstream logs/metadata as reference-only material",
    )
    parser.add_argument(
        "--dry-run",
        "--plan",
        action="store_true",
        dest="dry_run",
        help="print the canonical plan without network or filesystem writes",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="per-request HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="explicit no-clobber receipt path inside the artifact root",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    lock = load_official_act_release_lock(args.lock_path)
    plan = build_official_act_install_plan(
        args.artifact_root,
        lock=lock,
        tasks=args.task,
        profiles=args.profile,
        include_reference=args.include_reference,
    )
    if args.dry_run:
        sys.stdout.buffer.write(canonical_json_bytes(plan.to_dict()))
        return 0
    result = install_official_act_release(
        plan,
        timeout_s=args.timeout_s,
        receipt_path=args.receipt,
    )
    sys.stdout.buffer.write(canonical_json_bytes(result.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

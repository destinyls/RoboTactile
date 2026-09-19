#!/usr/bin/env python3
"""Prepare a separate 10 Hz N0 diagnostic bundle from the actual training artifacts.

metadata-root must contain <task>/meta/info.json and tasks.jsonl for all eight
tasks. per-repo-norm is the training JSON keyed by those task IDs. No upstream
checkout, checkpoint, base-model file, or released artifact is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robotactile_benchmark.integrations.n0_twam.retrained import prepare_retrained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "base",
        "source-root",
        "per-repo-norm",
        "metadata-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        help="declared commit; verified only against exact Git root or matching receipt",
    )
    parser.add_argument(
        "--source-receipt",
        type=Path,
        help="JSON with source_commit and source_tree_sha256",
    )
    args = parser.parse_args()
    artifact = prepare_retrained(**vars(args))
    print(artifact)


if __name__ == "__main__":
    main()

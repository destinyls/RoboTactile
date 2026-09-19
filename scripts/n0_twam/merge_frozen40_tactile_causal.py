#!/usr/bin/env python3
"""Merge eight frozen40 task shards into one offline causal result."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from robotactile_benchmark.recorded.tactile_causal_artifact import (
    write_recorded_tactile_causal_cohort_from_shards,
)

EXPECTED_TASK_IDS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(args.shard) != len(EXPECTED_TASK_IDS):
        raise ValueError("frozen40 merge requires exactly eight task shards")
    output = args.output.absolute()
    digest, summary = write_recorded_tactile_causal_cohort_from_shards(
        output,
        shard_paths=tuple(path.absolute() for path in args.shard),
        expected_task_ids=EXPECTED_TASK_IDS,
        episodes_per_task=5,
    )
    return {
        "completed_episode_count": summary["completed_episode_count"],
        "output": str(output),
        "output_sha256": digest,
        "success_rate_claimed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

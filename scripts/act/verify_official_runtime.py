#!/usr/bin/env python3
"""Verify an official ACT runtime without starting Isaac Sim."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from robotactile_benchmark.integrations.act.runtime_probe import (
    probe_official_act_runtime,
    runtime_probe_json_bytes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integration-config",
        required=True,
        type=Path,
        help="canonical ACT integration_config.json",
    )
    parser.add_argument(
        "--load-policy",
        action="store_true",
        help="strict-load the checkpoint and immediately close the policy",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = probe_official_act_runtime(
        args.integration_config,
        load_policy=args.load_policy,
    )
    sys.stdout.write(runtime_probe_json_bytes(result).decode("ascii"))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

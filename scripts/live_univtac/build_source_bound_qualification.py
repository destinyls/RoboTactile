#!/usr/bin/env python3
"""Upgrade all-task receipts to task-source-bound qualification v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotactile_benchmark.clean_baseline.qualification_builder import (
    build_source_bound_qualification,
)


def _key_value(value: str) -> tuple[str, str]:
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
        raise argparse.ArgumentTypeError("expected KEY=DEPLOYMENT_RELATIVE_PATH")
    return key, path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-qualification", required=True)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument(
        "--installation-receipt",
        action="append",
        type=_key_value,
        required=True,
    )
    parser.add_argument(
        "--observation-parity",
        action="append",
        type=_key_value,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verified = build_source_bound_qualification(
        deployment_root=args.root,
        legacy_qualification_relpath=args.legacy_qualification,
        campaign_manifest_relpath=args.campaign_manifest,
        installation_receipt_relpaths=dict(args.installation_receipt),
        observation_parity_relpaths=dict(args.observation_parity),
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "campaign_manifest_sha256": verified.campaign_manifest_sha256,
                "output": str(verified.path),
                "qualification_sha256": verified.sha256,
                "semantic_version": verified.semantic_version,
                "source_bound": verified.source_bound,
                "task_source_binding_count": len(verified.task_source_bindings),
                "tasks": list(verified.tasks),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

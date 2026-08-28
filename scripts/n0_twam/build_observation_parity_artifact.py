#!/usr/bin/env python3
"""Build source-bound N0-TWAM x UniVTAC observation parity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotactile_benchmark.integrations.n0_twam.observation_parity_builder import (
    ObservationParityInputs,
    build_and_write_observation_parity_artifact,
)
from robotactile_benchmark.integrations.n0_twam.observation_parity_io import (
    load_observation_parity_artifact,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment-lock", required=True)
    parser.add_argument("--teacher-forced-probe", required=True)
    parser.add_argument("--isaac-install-receipt", required=True)
    parser.add_argument("--n0-client-install-receipt", required=True)
    parser.add_argument("--n0-runtime-receipt", required=True)
    parser.add_argument("--tacex-install-receipt", required=True)
    parser.add_argument("--n0-artifact-manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = ObservationParityInputs(
        experiment_lock=args.experiment_lock,
        teacher_forced_probe=args.teacher_forced_probe,
        isaac_install_receipt=args.isaac_install_receipt,
        n0_client_install_receipt=args.n0_client_install_receipt,
        n0_runtime_receipt=args.n0_runtime_receipt,
        tacex_install_receipt=args.tacex_install_receipt,
        n0_artifact_manifest=args.n0_artifact_manifest,
    )
    artifact = build_and_write_observation_parity_artifact(
        deployment_root=args.root,
        task_id=args.task,
        inputs=inputs,
        output_path=args.output,
    )
    verified = load_observation_parity_artifact(args.root, args.output)
    if verified != artifact:
        raise RuntimeError("observation parity write/read mismatch")
    print(
        json.dumps(
            {
                "content_sha256": artifact.content_sha256,
                "failure_codes": list(artifact.failure_codes),
                "file_sha256": sha256_file(args.output),
                "output": str(args.output.absolute()),
                "passed": artifact.passed,
                "task_id": artifact.task_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if artifact.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Bind a joint train759 FTP-1 checkpoint to one task without release aliases."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from robotactile_benchmark.integrations.n0_twam.retrained import read_object
from scripts.ftp1_policy.serve_official import (
    FTP1PolicyEngine,
    FTP1RequestHandler,
    TaskContract,
    _load_wrapper,
    _serve,
)
from scripts.retrained_evaluation.group import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    binding = read_object(args.binding)
    task = binding["tasks"][args.task]
    logging.basicConfig(level=logging.INFO)
    wrapper = _load_wrapper(
        argparse.Namespace(
            source_root=Path(binding["source_root"]),
            expected_source_commit=binding["source_commit"],
            checkpoint_dir=Path(binding["checkpoint_root"]),
            domain_name=task["domain_name"],
            device="cuda:0",
            num_inference_steps=10,
        )
    )
    engine = FTP1PolicyEngine(
        wrapper,
        task_id=args.task,
        source_commit=binding["source_commit"],
        checkpoint_sha256=binding["checkpoint_sha256"],
        serve_bundle_sha256=binding["binding_sha256"],
        task_contract=TaskContract(
            task["prompt"], ("top", "wrist") if task["use_wrist"] else ("top",)
        ),
    )
    if engine.metadata != task["transport_metadata"]:
        raise ValueError("joint FTP-1 server metadata differs from frozen binding")
    write_json(args.receipt, engine.metadata)
    _serve(binding["endpoint"], FTP1RequestHandler(engine))


if __name__ == "__main__":
    main()

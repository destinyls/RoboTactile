"""Freeze the 10-seed, six-condition screening campaign before any new rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.integrations.runtime_config import (
    resolve_n0_runtime_artifacts,
)
from robotactile_benchmark.n0_fault_campaign.io import file_sha256
from robotactile_benchmark.n0_fault_campaign.stress_calibration import (
    calibrate_clean_artifacts,
)
from robotactile_benchmark.n0_fault_campaign.stress_group import write_once
from robotactile_benchmark.n0_fault_campaign.stress_protocol import (
    StressVariant,
    freeze_protocol,
)
from robotactile_benchmark.n0_fault_campaign.stress_provenance import code_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clean-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--development-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--excluded-seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(100000, 100010))
    )
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) != 10:
        raise ValueError("screening phase is fixed at ten seeds")
    if not set(args.development_seeds) <= set(args.excluded_seeds):
        raise ValueError("all template development seeds must be excluded")
    runtime = resolve_n0_runtime_artifacts(args.config)
    if runtime.manifest.task_id != "lift_bottle":
        raise ValueError("screening only supports lift_bottle")
    template, receipt = calibrate_clean_artifacts(
        tuple(p.resolve(strict=True) for p in args.clean_artifacts),
        allowed_seeds=tuple(args.development_seeds),
    )
    plan = freeze_protocol(
        stage="screening",
        seeds=tuple(args.seeds),
        excluded_seeds=tuple(args.excluded_seeds),
        variants=tuple(
            StressVariant(f, 5)
            for f in ("null", "fast_f1", "contact_f3", "fullframe", "skew_t3")
        ),
        binding={
            "dataset_sha256": args.dataset_sha256,
            "integration_config_sha256": file_sha256(args.config),
            "model_sha256": runtime.manifest.checkpoint_sha256,
            "code_sha256": code_sha256(args.code),
        },
        spatial_calibration_sha256=canonical_hash(template),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output / "spatial_calibration.json", template)
    write_once(args.output / "calibration_receipt.json", receipt)
    write_once(args.output / "protocol.json", plan)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_sha256": plan["protocol_sha256"],
                "planned_rollouts": plan["planned_live_rollouts"],
                "calibration": receipt,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

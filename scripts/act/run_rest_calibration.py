#!/usr/bin/env python3
"""Capture one measured ACT empty-gripper UniVTAC rest reference."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from robotactile_benchmark.backends.univtac_rest_calibration import (
    ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
)
from robotactile_benchmark.calibration import (
    build_rest_reference_from_live_artifact,
    write_rest_reference_artifact,
)
from robotactile_benchmark.closed_loop.capture import ClosedLoopExecutionEvidence
from robotactile_benchmark.deployment.layout import DeploymentLayout
from robotactile_benchmark.execution.capture_profiles import LiveCaptureProfile
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.execution.live_artifacts import (
    load_live_univtac_artifact,
    write_live_univtac_artifact,
)
from robotactile_benchmark.execution.live_univtac import (
    LiveArtifactExportReceipt,
    default_live_backend_factory,
    execute_live_univtac_run,
)
from robotactile_benchmark.execution.loading import (
    LoadedLiveUniVTACRun,
    load_live_univtac_request,
)
from robotactile_benchmark.execution.official_act import (
    build_official_act_live_binding,
    make_official_act_policy_factory,
)
from robotactile_benchmark.integrations.act.requests import (
    write_official_act_request,
)
from robotactile_benchmark.integrations.runtime_config import (
    resolve_act_runtime_artifacts,
)
from robotactile_benchmark.rest_references import ReferenceSplit
from robotactile_benchmark.trials import Condition


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-clean-request", type=Path, required=True)
    parser.add_argument("--integration-config", type=Path, required=True)
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument("--live-output", type=Path, required=True)
    parser.add_argument("--rest-output", type=Path, required=True)
    parser.add_argument("--minimum-consecutive-free-records", type=int, default=5)
    parser.add_argument("--control-cycles", type=int, default=6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.expanduser().absolute()
    layout = DeploymentLayout(root)
    base = load_live_univtac_request(args.base_clean_request)
    if (
        base.policy_kind is not LivePolicyKind.ACT
        or base.condition is not Condition.CLEAN
    ):
        raise ValueError("base request must be an ACT Clean request")
    if args.initial_seed < 0 or args.exogenous_seed < 0:
        raise ValueError("calibration seeds must be non-negative")
    if args.control_cycles < args.minimum_consecutive_free_records:
        raise ValueError("control-cycles must cover the requested free-record window")
    live_output = args.live_output.expanduser().absolute()
    request = replace(
        base,
        initial_seed=args.initial_seed,
        exogenous_seed=args.exogenous_seed,
        max_control_cycles=args.control_cycles,
        max_observation_steps=args.control_cycles + 1,
        runtime_dir=(
            layout.runtime / "live-univtac" / "act-rest-calibration" / base.task_id
        ),
        output_dir=live_output,
    )
    generated = write_official_act_request(args.request_output, request)
    runtime = resolve_act_runtime_artifacts(args.integration_config)
    if runtime.manifest.task_id != request.task_id:
        raise ValueError("integration config task differs from calibration request")
    backend_factory = partial(
        default_live_backend_factory,
        calibration_contract=ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
    )
    binding = build_official_act_live_binding(
        request,
        artifact_root=runtime.artifact_root,
        stats_sha256=runtime.stats_sha256,
        encoder_sha256=runtime.encoder_sha256,
    )
    published: list[dict[str, object]] = []

    def export_calibration(
        output: Path,
        loaded: LoadedLiveUniVTACRun,
        evidence: ClosedLoopExecutionEvidence,
    ) -> LiveArtifactExportReceipt:
        receipt = write_live_univtac_artifact(
            output,
            loaded,
            evidence,
            capture_profile=LiveCaptureProfile.PAPER_FULL,
        )
        artifact = load_live_univtac_artifact(output)
        references, validation = build_rest_reference_from_live_artifact(
            artifact,
            ReferenceSplit.CALIBRATION,
            args.minimum_consecutive_free_records,
        )
        exported = write_rest_reference_artifact(
            args.rest_output.expanduser().absolute(),
            references,
            validation,
        )
        published.append(
            {
                "artifact_root_sha256": artifact.root_receipt_sha256,
                "calibration_contract": ACT_EMPTY_GRIPPER_CALIBRATION_CONTRACT,
                "model_actions_applied": False,
                "qualified_range": [
                    validation.qualified_start_index,
                    validation.qualified_stop_index,
                ],
                "request_file_sha256": generated.request_file_sha256,
                "rest_reference_root_sha256": exported.artifact_root_sha256,
                "rest_reference_sha256": exported.rest_reference_sha256,
                "selected_step_index": validation.selected_step_index,
                "task": request.task_id,
            }
        )
        return receipt

    execute_live_univtac_run(
        request,
        backend_factory=backend_factory,
        policy_factory=make_official_act_policy_factory(binding),
        artifact_exporter=export_calibration,
    )
    if len(published) != 1:
        raise RuntimeError("rest calibration publication count mismatch")
    print(json.dumps(published[0], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry points for local registry and replay validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from robotactile_benchmark.act_fault_campaign.cli import (
    add_act_fault_campaign_subcommands,
    handle_act_fault_campaign_command,
)
from robotactile_benchmark.calibration.cli_support import (
    add_calibration_request_parser,
)
from robotactile_benchmark.clean_baseline.cli import (
    add_clean_baseline_subcommands,
    handle_clean_baseline_command,
)
from robotactile_benchmark.closed_loop.smoke import (
    smoke_summary,
    write_cpu_smoke_bundle,
)
from robotactile_benchmark.constants import REST_REFERENCE_OPERATOR_IDS
from robotactile_benchmark.deployment.cli import (
    add_deployment_subcommands,
    handle_deployment_command,
)
from robotactile_benchmark.execution.live_cli import (
    add_live_execution_subcommands,
    handle_live_execution_command,
)
from robotactile_benchmark.execution.preflight_cli import (
    add_live_preflight_parser,
    handle_live_preflight_command,
)
from robotactile_benchmark.fixtures import make_synthetic_rest_references
from robotactile_benchmark.integrations.cli import (
    add_integration_subcommands,
    handle_integration_command,
)
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix.live_cli import (
    add_live_matrix_subcommands,
    handle_live_matrix_command,
)
from robotactile_benchmark.n0_fault_campaign.cli import (
    add_n0_fault_campaign_subcommands,
    handle_n0_fault_campaign_command,
)
from robotactile_benchmark.operators import EXPECTED_OPERATOR_IDS, list_operator_ids
from robotactile_benchmark.protocol_alignment.cli import (
    add_protocol_alignment_subcommand,
    handle_protocol_alignment_command,
)
from robotactile_benchmark.recorded.alignment_cli import (
    add_expert_alignment_subcommand,
    handle_expert_alignment_command,
)
from robotactile_benchmark.recorded.cli import (
    add_recorded_n0_subcommand,
    handle_recorded_n0_command,
)
from robotactile_benchmark.replay import run_smoke_matrix, run_smoke_replay
from robotactile_benchmark.severity import severity_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotactile",
        description=(
            "Auditable optical-tactile robustness evaluation. Start with "
            "'deployment init', 'integrations list', or 'setup --model act'."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "validate-registry", help="validate and list the frozen 14-operator registry"
    )
    replay = subparsers.add_parser(
        "smoke-replay", help="run one deterministic operator replay smoke"
    )
    replay.add_argument("--output", type=Path, required=True, help="artifact output")
    replay.add_argument(
        "--operator",
        default="F6_history_residual_imprint",
        choices=sorted(EXPECTED_OPERATOR_IDS),
    )
    replay.add_argument(
        "--severity", type=int, default=3, choices=range(1, 6), help="level 1-5"
    )
    matrix = subparsers.add_parser(
        "smoke-matrix", help="run the deterministic software smoke matrix"
    )
    matrix.add_argument("--output", type=Path, required=True, help="artifact output")
    closed_loop = subparsers.add_parser(
        "closed-loop-smoke", help="exercise the causal closed-loop runner with fakes"
    )
    closed_loop.add_argument(
        "--output", type=Path, required=True, help="artifact output"
    )
    report = subparsers.add_parser(
        "report-matrix", help="render a source-bound benchmark report bundle"
    )
    report.add_argument("--root", type=Path)
    report.add_argument("--matrix-manifest", type=Path, required=True)
    report.add_argument("--matrix-output", type=Path)
    report.add_argument("--reporting-spec", type=Path, required=True)
    report.add_argument("--output", type=Path)
    visualize = subparsers.add_parser(
        "visualize-live-artifact",
        help="render a verified live trace as a paper panel and optional MP4",
    )
    visualize.add_argument("--artifact", type=Path, required=True)
    visualize.add_argument("--output", type=Path, required=True)
    visualize.add_argument("--fps", type=int, default=20)
    visualize.add_argument("--stride", type=int, default=1)
    visualize.add_argument("--max-frames", type=int)
    visualize.add_argument("--video", action="store_true")
    visualize.add_argument("--ffmpeg", default="ffmpeg")
    calibration = subparsers.add_parser(
        "build-rest-references", help="derive measured no-contact references"
    )
    calibration.add_argument("--source-live-artifact", type=Path, required=True)
    calibration.add_argument(
        "--dataset-split",
        choices=("development", "validation", "calibration"),
        required=True,
    )
    calibration.add_argument("--minimum-consecutive-free-records", type=int, default=5)
    calibration.add_argument("--output", type=Path, required=True)
    add_calibration_request_parser(subparsers)
    add_live_execution_subcommands(subparsers)
    add_live_matrix_subcommands(subparsers)
    add_integration_subcommands(subparsers)
    add_live_preflight_parser(subparsers)
    add_deployment_subcommands(subparsers)
    add_clean_baseline_subcommands(subparsers)
    add_recorded_n0_subcommand(subparsers)
    add_expert_alignment_subcommand(subparsers)
    add_protocol_alignment_subcommand(subparsers)
    add_n0_fault_campaign_subcommands(subparsers)
    add_act_fault_campaign_subcommands(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    act_fault_result = handle_act_fault_campaign_command(args)
    if act_fault_result is not None:
        print(
            json.dumps(
                act_fault_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    n0_fault_result = handle_n0_fault_campaign_command(args)
    if n0_fault_result is not None:
        print(
            json.dumps(
                n0_fault_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    protocol_alignment_result = handle_protocol_alignment_command(args)
    if protocol_alignment_result is not None:
        protocol_payload, exit_code = protocol_alignment_result
        print(
            json.dumps(
                protocol_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return exit_code
    alignment_result = handle_expert_alignment_command(args)
    if alignment_result is not None:
        print(
            json.dumps(
                alignment_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    recorded_result = handle_recorded_n0_command(args)
    if recorded_result is not None:
        print(
            json.dumps(
                recorded_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    clean_baseline_result = handle_clean_baseline_command(args)
    if clean_baseline_result is not None:
        clean_payload, exit_code = clean_baseline_result
        print(
            json.dumps(
                clean_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return exit_code
    deployment_result = handle_deployment_command(args)
    if deployment_result is not None:
        print(
            json.dumps(
                deployment_result.payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return deployment_result.exit_code
    integration_result = handle_integration_command(args)
    if integration_result is not None:
        print(
            json.dumps(
                integration_result.payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return integration_result.exit_code
    live_execution_result = handle_live_execution_command(args)
    if live_execution_result is not None:
        print(
            json.dumps(
                live_execution_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    matrix_result = handle_live_matrix_command(args)
    if matrix_result is not None:
        print(
            json.dumps(
                matrix_result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    preflight_result = handle_live_preflight_command(args)
    if preflight_result is not None:
        preflight_payload, exit_code = preflight_result
        print(
            json.dumps(
                preflight_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return exit_code
    if args.command == "validate-registry":
        payload = {
            "count": len(list_operator_ids()),
            "operator_ids": list(list_operator_ids()),
        }
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.command == "smoke-matrix":
        summary = run_smoke_matrix(args.output)
        print(
            json.dumps(
                {
                    "matrix_id": summary["matrix_id"],
                    "cell_count": summary["cell_count"],
                    "all_valid": all(
                        cell["validation_passed"] for cell in summary["cells"]
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "closed-loop-smoke":
        summary = smoke_summary(write_cpu_smoke_bundle(args.output))
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    if args.command == "report-matrix":
        from robotactile_benchmark.deployment.layout import (
            DeploymentLayout,
            initialize_deployment_layout,
            resolve_deployment_root,
        )
        from robotactile_benchmark.reporting.matrix_adapter import (
            load_matrix_manifest,
            write_matrix_report,
        )

        matrix_output = args.matrix_output
        report_output = args.output
        if matrix_output is None or report_output is None:
            layout = DeploymentLayout(resolve_deployment_root(args.root))
            initialize_deployment_layout(layout)
            matrix_manifest = load_matrix_manifest(args.matrix_manifest)
            matrix_output = (
                matrix_output or layout.outputs / "matrices" / matrix_manifest.matrix_id
            )
            report_output = (
                report_output or layout.outputs / "reports" / matrix_manifest.matrix_id
            )
        assert matrix_output is not None
        assert report_output is not None
        exported = write_matrix_report(
            args.matrix_manifest,
            matrix_output,
            args.reporting_spec,
            report_output,
        )
        print(
            json.dumps(
                exported.to_cli_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    if args.command == "visualize-live-artifact":
        from robotactile_benchmark.visualization import (
            export_live_artifact_visualization,
        )

        visualized = export_live_artifact_visualization(
            args.artifact,
            args.output,
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            video=args.video,
            ffmpeg=args.ffmpeg,
        )
        print(
            json.dumps(
                visualized.to_cli_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    if args.command == "build-rest-references":
        from robotactile_benchmark.calibration import (
            build_rest_reference_from_live_artifact,
            load_rest_reference_artifact,
            write_rest_reference_artifact,
        )
        from robotactile_benchmark.execution import load_live_univtac_artifact
        from robotactile_benchmark.rest_references import ReferenceSplit

        source = load_live_univtac_artifact(args.source_live_artifact)
        references, validation = build_rest_reference_from_live_artifact(
            source,
            ReferenceSplit(args.dataset_split),
            args.minimum_consecutive_free_records,
        )
        write_rest_reference_artifact(args.output, references, validation)
        loaded_calibration = load_rest_reference_artifact(args.output)
        print(
            json.dumps(
                {
                    "artifact_root_sha256": loaded_calibration.root_receipt_sha256,
                    "evidence_level": loaded_calibration.root_receipt.evidence_level,
                    "qualified_range": [
                        loaded_calibration.validation.qualified_start_index,
                        loaded_calibration.validation.qualified_stop_index,
                    ],
                    "rest_reference_sha256": loaded_calibration.references.sha256,
                    "selected_step_index": (
                        loaded_calibration.validation.selected_step_index
                    ),
                    "source_live_artifact_root_sha256": (
                        loaded_calibration.validation.source_live_artifact_root_sha256
                    ),
                    "validation_sha256": loaded_calibration.validation.sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    if args.command == "generate-calibration-request":
        from robotactile_benchmark.calibration.cli_support import (
            generate_calibration_request,
            resolve_calibration_command_paths,
        )

        paths = resolve_calibration_command_paths(
            root=args.root,
            task_id=args.task,
            dataset_split=args.dataset_split,
            initial_seed=args.initial_seed,
            exogenous_seed=args.exogenous_seed,
            upstream_root=args.upstream_root,
            runtime_dir=args.runtime_dir,
            live_artifact_output=args.live_artifact_output,
            output=args.output,
        )
        payload = generate_calibration_request(
            task_id=args.task,
            dataset_split=args.dataset_split,
            split_manifest_sha256=args.split_manifest_sha256,
            base_system_id=args.base_system_id,
            checkpoint_sha256=args.checkpoint_sha256,
            config_sha256=args.config_sha256,
            initial_seed=args.initial_seed,
            exogenous_seed=args.exogenous_seed,
            max_control_cycles=args.max_control_cycles,
            max_observation_steps=args.max_observation_steps,
            wall_timeout_s=args.wall_timeout_s,
            upstream_root=paths.upstream_root,
            runtime_dir=paths.runtime_dir,
            live_artifact_output=paths.live_artifact_output,
            act_device_name=args.act_device_name,
            simulator_device=args.simulator_device,
            output=paths.output,
        )
        print(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        return 0
    start_index = 3
    if args.operator == "T1_fixed_source_delay":
        start_index = int(severity_value(args.operator, args.severity))
    stop_index = start_index + 6
    sensor_slots = (
        ("left", "right")
        if args.operator in {"T3_inter_sensor_skew", "C1_sensor_identity_misrouting"}
        else ("left",)
    )
    parameters = {}
    if args.operator in REST_REFERENCE_OPERATOR_IDS:
        parameters["rest_reference_sha256"] = make_synthetic_rest_references().sha256
    if args.operator == "C2_frame_misregistration":
        parameters["realization"] = "registered_pixels"
    manifest = FaultManifest(
        operator_id=args.operator,
        severity_level=args.severity,
        operator_seed=20260814,
        start_index=start_index,
        stop_index=stop_index,
        sensor_slots=sensor_slots,
        observability=Observability.BLIND,
        parameters=parameters,
    )
    result = run_smoke_replay(args.output, manifest)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest.sha256,
                "trace_sha256": result.trace_sha256,
                "validation_passed": result.validation.passed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

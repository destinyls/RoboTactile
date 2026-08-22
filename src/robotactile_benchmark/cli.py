"""Command-line entry points for local registry and replay validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from robotactile_benchmark.calibration.cli_support import (
    add_calibration_request_parser,
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
from robotactile_benchmark.operators import EXPECTED_OPERATOR_IDS, list_operator_ids
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
    live = subparsers.add_parser(
        "live-univtac-run", help="execute one unqualified live UniVTAC request"
    )
    live.add_argument("--request", type=Path, required=True, help="request JSON")
    live.add_argument("--root", type=Path)
    live.add_argument("--config", type=Path)
    live.add_argument("--official-act-artifact-root", type=Path)
    live.add_argument("--stats-sha256")
    live.add_argument("--encoder-sha256")
    report = subparsers.add_parser(
        "report-matrix", help="render a source-bound benchmark report bundle"
    )
    report.add_argument("--root", type=Path)
    report.add_argument("--matrix-manifest", type=Path, required=True)
    report.add_argument("--matrix-output", type=Path)
    report.add_argument("--reporting-spec", type=Path, required=True)
    report.add_argument("--output", type=Path)
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
    add_live_matrix_subcommands(subparsers)
    add_integration_subcommands(subparsers)
    add_live_preflight_parser(subparsers)
    add_deployment_subcommands(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
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
        payload, exit_code = preflight_result
        print(
            json.dumps(
                payload,
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
    if args.command == "live-univtac-run":
        from robotactile_benchmark.deployment.layout import (
            DeploymentLayout,
            resolve_deployment_root,
        )
        from robotactile_benchmark.execution.loading import (
            load_live_univtac_request,
        )
        from robotactile_benchmark.execution.official_act import (
            execute_official_act_live_run,
            official_act_live_summary,
        )
        from robotactile_benchmark.integrations.runtime_config import (
            resolve_act_runtime_artifacts,
        )

        live_request = load_live_univtac_request(args.request)
        legacy = (
            args.official_act_artifact_root,
            args.stats_sha256,
            args.encoder_sha256,
        )
        if args.config is not None or any(value is None for value in legacy):
            layout = DeploymentLayout(resolve_deployment_root(args.root))
            config_path = args.config or (
                layout.model_artifacts / "act/integration_config.json"
            )
            resolved = resolve_act_runtime_artifacts(config_path)
            artifact_root = resolved.artifact_root
            stats_sha256 = resolved.stats_sha256
            encoder_sha256 = resolved.encoder_sha256
        else:
            artifact_root = args.official_act_artifact_root
            stats_sha256 = args.stats_sha256
            encoder_sha256 = args.encoder_sha256
        assert artifact_root is not None
        assert stats_sha256 is not None
        assert encoder_sha256 is not None
        live_artifact = execute_official_act_live_run(
            live_request,
            artifact_root=artifact_root,
            stats_sha256=stats_sha256,
            encoder_sha256=encoder_sha256,
        )
        print(
            json.dumps(
                official_act_live_summary(live_artifact),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
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

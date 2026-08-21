"""CLI bridge for one strict, resumable live matrix run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.matrix.io import load_matrix_manifest_file
from robotactile_benchmark.matrix.live_run import (
    live_matrix_run_summary,
    run_live_matrix,
)
from robotactile_benchmark.matrix.live_run_config import (
    load_live_matrix_run_config,
)
from robotactile_benchmark.matrix.primary_generation import (
    LIVE_MATRIX_RUN_CONFIG_PATH,
    generate_primary_matrix_bundle,
)
from robotactile_benchmark.matrix.primary_generation_contracts import (
    PrimaryMatrixGenerationSpec,
)


def add_live_matrix_subcommands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register primary generation and resumable matrix execution commands."""

    generate = subparsers.add_parser("generate-primary-matrix")
    generate.add_argument("--task", required=True)
    generate.add_argument("--dataset-sha256", required=True)
    generate.add_argument("--base-system-id")
    generate.add_argument("--tactile-checkpoint-sha256", required=True)
    generate.add_argument("--no-touch-system-id")
    generate.add_argument("--no-touch-checkpoint-sha256", required=True)
    generate.add_argument("--stats-sha256", required=True)
    generate.add_argument("--encoder-sha256", required=True)
    generate.add_argument("--initial-seed", type=int, required=True)
    generate.add_argument("--exogenous-seed", type=int, required=True)
    generate.add_argument("--operator-seed-base", type=int, default=20260821)
    generate.add_argument("--fault-start-index", type=int, default=16)
    generate.add_argument("--restoration-index", type=int, required=True)
    generate.add_argument("--fault-stop-index", type=int)
    generate.add_argument("--max-control-cycles", type=int)
    generate.add_argument("--max-observation-steps", type=int)
    generate.add_argument("--wall-timeout-s", type=float, default=1800.0)
    generate.add_argument("--upstream-root", type=Path, required=True)
    generate.add_argument("--runtime-root", type=Path, required=True)
    generate.add_argument("--official-act-artifact-root", type=Path, required=True)
    generate.add_argument("--rest-reference-artifact", type=Path, required=True)
    generate.add_argument("--act-device-name", default="cuda:0")
    generate.add_argument("--simulator-device", default="cuda:0")
    generate.add_argument("--matrix-id")
    generate.add_argument("--output", type=Path, required=True)
    live = subparsers.add_parser("run-live-matrix")
    live.add_argument("--matrix-manifest", type=Path, required=True)
    live.add_argument("--run-config", type=Path, required=True)
    live.add_argument("--matrix-output", type=Path, required=True)
    live.add_argument("--max-new-cells", type=int)


def handle_live_matrix_command(args: argparse.Namespace) -> Optional[dict[str, object]]:
    """Execute one matrix CLI command, or return ``None`` when unrelated."""

    if args.command == "run-live-matrix":
        return execute_live_matrix_cli(
            args.matrix_manifest,
            args.run_config,
            args.matrix_output,
            args.max_new_cells,
        )
    if args.command != "generate-primary-matrix":
        return None
    task = build_univtac_backend_config(args.task).task
    max_cycles = (
        task.action_horizon
        if args.max_control_cycles is None
        else args.max_control_cycles
    )
    max_observations = (
        max_cycles + 1
        if args.max_observation_steps is None
        else args.max_observation_steps
    )
    fault_stop = (
        max_observations if args.fault_stop_index is None else args.fault_stop_index
    )
    tactile_system = (
        args.base_system_id
        or f"official-univtac-act.{args.task}.univtac.policy_last.v1"
    )
    no_touch_system = (
        args.no_touch_system_id
        or f"official-univtac-act.{args.task}.vision_only.policy_last.v1"
    )
    spec = PrimaryMatrixGenerationSpec(
        task_id=args.task,
        dataset_sha256=args.dataset_sha256,
        base_system_id=tactile_system,
        tactile_checkpoint_sha256=args.tactile_checkpoint_sha256,
        no_touch_system_id=no_touch_system,
        no_touch_checkpoint_sha256=args.no_touch_checkpoint_sha256,
        stats_sha256=args.stats_sha256,
        encoder_sha256=args.encoder_sha256,
        initial_seed=args.initial_seed,
        exogenous_seed=args.exogenous_seed,
        operator_seed_base=args.operator_seed_base,
        fault_start_index=args.fault_start_index,
        restoration_index=args.restoration_index,
        fault_stop_index=fault_stop,
        max_control_cycles=max_cycles,
        max_observation_steps=max_observations,
        wall_timeout_s=args.wall_timeout_s,
        upstream_root=args.upstream_root,
        runtime_root=args.runtime_root,
        official_act_artifact_root=args.official_act_artifact_root,
        rest_reference_artifact=args.rest_reference_artifact,
        act_device_name=args.act_device_name,
        simulator_device=args.simulator_device,
        matrix_id=args.matrix_id,
    )
    status, loaded = generate_primary_matrix_bundle(args.output, spec)
    return {
        "cell_count": len(loaded.manifest.cells),
        "comparison_count": len(loaded.manifest.comparisons),
        "evidence_level": loaded.receipt.evidence_level,
        "matrix_id": loaded.manifest.matrix_id,
        "matrix_manifest": str(loaded.root / "matrix_manifest.json"),
        "matrix_manifest_sha256": loaded.manifest.sha256,
        "output": str(loaded.root),
        "receipt_file_sha256": loaded.receipt_file_sha256,
        "run_config": str(loaded.root / LIVE_MATRIX_RUN_CONFIG_PATH),
        "simulator_execution_claimed": False,
        "status": status,
    }


def execute_live_matrix_cli(
    matrix_manifest: Path,
    run_config: Path,
    matrix_output: Path,
    max_new_cells: Optional[int],
) -> dict[str, object]:
    """Load exact inputs, execute or resume, and return a canonical summary."""

    manifest = load_matrix_manifest_file(matrix_manifest)
    config = load_live_matrix_run_config(run_config, manifest)
    result = run_live_matrix(
        matrix_output,
        manifest,
        config,
        max_new_cells=max_new_cells,
    )
    return live_matrix_run_summary(manifest, result)

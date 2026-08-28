"""Small CLI bridge for clean calibration request generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration.request_artifacts import (
    write_calibration_request_bundle,
)
from robotactile_benchmark.calibration.request_contracts import CalibrationRequestSpec
from robotactile_benchmark.deployment.layout import (
    DeploymentLayout,
    initialize_deployment_layout,
    resolve_deployment_root,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.rest_references import ReferenceSplit


def add_calibration_request_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the dependency-light calibration request command."""

    parser = subparsers.add_parser(
        "generate-calibration-request",
        help="materialize a clean, no-allocation calibration request",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--dataset-split",
        choices=("development", "validation", "calibration"),
        required=True,
    )
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--base-system-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--initial-seed", type=int, required=True)
    parser.add_argument("--exogenous-seed", type=int, required=True)
    parser.add_argument("--max-control-cycles", type=int)
    parser.add_argument("--max-observation-steps", type=int)
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--live-artifact-output", type=Path)
    parser.add_argument("--act-device-name", default="cuda:0")
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument("--output", type=Path)


@dataclass(frozen=True)
class CalibrationCommandPaths:
    upstream_root: Path
    runtime_dir: Path
    live_artifact_output: Path
    output: Path


def resolve_calibration_command_paths(
    *,
    root: Optional[Path],
    task_id: str,
    dataset_split: str,
    initial_seed: int,
    exogenous_seed: int,
    upstream_root: Optional[Path],
    runtime_dir: Optional[Path],
    live_artifact_output: Optional[Path],
    output: Optional[Path],
) -> CalibrationCommandPaths:
    """Resolve omitted paths into the typed deployment layout."""

    if all(
        value is not None
        for value in (upstream_root, runtime_dir, live_artifact_output, output)
    ):
        assert upstream_root is not None
        assert runtime_dir is not None
        assert live_artifact_output is not None
        assert output is not None
        return CalibrationCommandPaths(
            upstream_root=upstream_root,
            runtime_dir=runtime_dir,
            live_artifact_output=live_artifact_output,
            output=output,
        )
    layout = DeploymentLayout(resolve_deployment_root(root))
    initialize_deployment_layout(layout)
    request_id = f"{task_id}-{dataset_split}-i{initial_seed}-e{exogenous_seed}"
    return CalibrationCommandPaths(
        upstream_root=upstream_root or layout.sources / "UniVTAC",
        runtime_dir=runtime_dir or layout.runtime / "calibration" / request_id,
        live_artifact_output=(
            live_artifact_output
            or layout.artifacts / "live-univtac" / "calibration" / request_id
        ),
        output=output or layout.requests / "calibration" / request_id,
    )


def generate_calibration_request(
    *,
    task_id: str,
    dataset_split: str,
    split_manifest_sha256: str,
    base_system_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
    initial_seed: int,
    exogenous_seed: int,
    max_control_cycles: Optional[int],
    max_observation_steps: Optional[int],
    wall_timeout_s: float,
    upstream_root: Path,
    runtime_dir: Path,
    live_artifact_output: Path,
    act_device_name: str,
    simulator_device: str,
    output: Path,
) -> dict[str, object]:
    """Generate one request bundle and return its bounded CLI summary."""

    config = build_univtac_backend_config(task_id)
    horizon = config.task.action_horizon
    spec = CalibrationRequestSpec(
        task_id=task_id,
        dataset_split=ReferenceSplit(dataset_split),
        split_manifest_sha256=split_manifest_sha256,
        base_system_id=base_system_id,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        initial_seed=initial_seed,
        exogenous_seed=exogenous_seed,
        max_control_cycles=horizon
        if max_control_cycles is None
        else max_control_cycles,
        max_observation_steps=(
            horizon + 1 if max_observation_steps is None else max_observation_steps
        ),
        wall_timeout_s=wall_timeout_s,
        upstream_root=upstream_root,
        runtime_dir=runtime_dir,
        live_artifact_output_dir=live_artifact_output,
        act_device_name=act_device_name,
        simulator_device=simulator_device,
        launcher_args=production_univtac_launcher_args(),
    )
    loaded = write_calibration_request_bundle(output, spec)
    return {
        "dataset_split": loaded.receipt.dataset_split.value,
        "evidence_level": loaded.receipt.evidence_level,
        "receipt_file_sha256": loaded.receipt_file_sha256,
        "request_file": str((Path(output).absolute() / "request.json")),
        "request_sha256": loaded.receipt.request_sha256,
        "simulator_execution_claimed": False,
        "task_id": loaded.receipt.task_id,
        "trial_manifest_sha256": loaded.receipt.trial_manifest_sha256,
    }


__all__ = [
    "CalibrationCommandPaths",
    "add_calibration_request_parser",
    "generate_calibration_request",
    "resolve_calibration_command_paths",
]

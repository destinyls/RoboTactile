"""Small CLI bridge for clean calibration request generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import (
    build_univtac_backend_config,
)
from robotactile_benchmark.calibration.request_artifacts import (
    write_calibration_request_bundle,
)
from robotactile_benchmark.calibration.request_contracts import CalibrationRequestSpec
from robotactile_benchmark.rest_references import ReferenceSplit


def add_calibration_request_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the dependency-light calibration request command."""

    parser = subparsers.add_parser("generate-calibration-request")
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
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--live-artifact-output", type=Path, required=True)
    parser.add_argument("--act-device-name", default="cuda:0")
    parser.add_argument("--simulator-device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)


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
        launcher_args={"enable_cameras": True, "headless": True},
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

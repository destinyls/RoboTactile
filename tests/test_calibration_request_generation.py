"""Clean calibration request generation and strict persistence tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robotactile_benchmark.calibration import (
    CalibrationRequestSpec,
    load_calibration_request_bundle,
    write_calibration_request_bundle,
)
from robotactile_benchmark.calibration.request_contracts import (
    CalibrationRequestError,
)
from robotactile_benchmark.rest_references import ReferenceSplit
from robotactile_benchmark.trials import Condition


def _spec(root: Path) -> CalibrationRequestSpec:
    return CalibrationRequestSpec(
        task_id="pull_out_key",
        dataset_split=ReferenceSplit.CALIBRATION,
        split_manifest_sha256="a" * 64,
        base_system_id="official-act.pull-out-key.tactile.v1",
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        initial_seed=17,
        exogenous_seed=29,
        max_control_cycles=4,
        max_observation_steps=5,
        wall_timeout_s=30.0,
        upstream_root=root / "UniVTAC",
        runtime_dir=root / "runtime",
        live_artifact_output_dir=root / "live-artifact",
        act_device_name="cuda:0",
        simulator_device="cuda:0",
        launcher_args={"enable_cameras": True, "headless": True},
    )


def _inventory(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_request_bundle_is_clean_strict_and_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    loaded = write_calibration_request_bundle(first, _spec(tmp_path))
    write_calibration_request_bundle(second, _spec(tmp_path))
    repeated = write_calibration_request_bundle(first, _spec(tmp_path))

    assert _inventory(first) == _inventory(second)
    assert loaded == repeated == load_calibration_request_bundle(first)
    assert loaded.request.condition is Condition.CLEAN
    assert loaded.request.dataset_sha256 == "a" * 64
    assert loaded.receipt.dataset_split is ReferenceSplit.CALIBRATION
    assert loaded.receipt.simulator_execution_claimed is False


def test_request_bundle_rejects_stale_cross_link_and_synthetic_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(CalibrationRequestError, match="synthetic"):
        CalibrationRequestSpec(
            **{
                **_spec(tmp_path).__dict__,
                "dataset_split": ReferenceSplit.SYNTHETIC_DEVELOPMENT,
            }
        )
    output = tmp_path / "bundle"
    write_calibration_request_bundle(output, _spec(tmp_path))
    request_path = output / "request.json"
    document = json.loads(request_path.read_text(encoding="utf-8"))
    document["initial_seed"] = 99
    request_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CalibrationRequestError, match="cross-link"):
        load_calibration_request_bundle(output)


def test_generate_calibration_request_cli_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "request-bundle"
    command = [
        sys.executable,
        "-m",
        "robotactile_benchmark.cli",
        "generate-calibration-request",
        "--task",
        "pull_out_key",
        "--dataset-split",
        "calibration",
        "--split-manifest-sha256",
        "a" * 64,
        "--base-system-id",
        "official-act.pull-out-key.tactile.v1",
        "--checkpoint-sha256",
        "b" * 64,
        "--config-sha256",
        "c" * 64,
        "--initial-seed",
        "17",
        "--exogenous-seed",
        "29",
        "--max-control-cycles",
        "4",
        "--max-observation-steps",
        "5",
        "--upstream-root",
        str(tmp_path / "UniVTAC"),
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "--live-artifact-output",
        str(tmp_path / "live-artifact"),
        "--output",
        str(output),
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == second.returncode == 0, first.stderr
    assert json.loads(first.stdout) == json.loads(second.stdout)
    assert json.loads(first.stdout)["simulator_execution_claimed"] is False
    assert load_calibration_request_bundle(output).request.task_id == "pull_out_key"

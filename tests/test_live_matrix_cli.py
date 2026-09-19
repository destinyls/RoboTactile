"""Strict live matrix run-config, resumable execution, and CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robotactile_benchmark.closed_loop.contracts import ACTION_SPEC
from robotactile_benchmark.closed_loop.fakes import (
    DeterministicFakeBackend,
    DeterministicFakePolicy,
)
from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.fixtures import make_synthetic_episode
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix import (
    LiveMatrixResourceEntry,
    LiveMatrixRunConfig,
    MatrixCellStatus,
    MatrixGridPoint,
    build_focused_phase_manifest,
    load_live_matrix_run_config,
    run_live_matrix,
)
from robotactile_benchmark.matrix.io import canonical_matrix_json_bytes
from robotactile_benchmark.matrix.live_run_config import LiveMatrixRunConfigError
from robotactile_benchmark.trials import (
    Condition,
    TrialManifest,
    system_manifest_hash,
)


def _manifest():  # type: ignore[no-untyped-def]
    clean = TrialManifest(
        task="pull_out_key",
        initial_seed=11,
        exogenous_seed=29,
        condition=Condition.CLEAN,
        base_system_id="univtac-act-tactile",
        executed_system_id="univtac-act-tactile",
        dataset_sha256="a" * 64,
        base_system_manifest_sha256=system_manifest_hash(
            "univtac-act-tactile", "b" * 64, "c" * 64, ACTION_SPEC
        ),
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        action_spec=ACTION_SPEC,
        fault_manifest_sha256=None,
        matched_no_touch_system_id=None,
    )
    fault = FaultManifest(
        operator_id="A1_stream_absence",
        severity_level=3,
        operator_seed=700,
        start_index=1,
        stop_index=4,
        sensor_slots=("left",),
        observability=Observability.BLIND,
        parameters={},
    )
    return build_focused_phase_manifest(
        matrix_id="live-cli-three-condition-v1",
        clean=clean,
        grid_points=(MatrixGridPoint("contact-window", fault),),
        no_touch_system_id="univtac-act-vision-only",
        no_touch_checkpoint_sha256="e" * 64,
        no_touch_config_sha256="f" * 64,
    )


def _write_fault(path: Path, fault: FaultManifest) -> None:
    path.write_bytes(canonical_matrix_json_bytes(fault.to_dict()))


def _config(root: Path, manifest):  # type: ignore[no-untyped-def]
    resources = {}
    for cell in manifest.cells:
        condition = cell.trial.condition
        fault_path = None
        no_touch_path = None
        if cell.fault_manifest is not None:
            fault_path = root / "resources" / f"{cell.sha256}.json"
            fault_path.parent.mkdir(parents=True, exist_ok=True)
            _write_fault(fault_path, cell.fault_manifest)
        if condition is Condition.NO_TOUCH:
            no_touch_path = root / "official-act/vision_only/policy_last.ckpt"
        resources[cell.sha256] = LiveMatrixResourceEntry(
            fault_manifest_path=fault_path,
            rest_references_path=None,
            matched_no_touch_artifact_path=no_touch_path,
        )
    return LiveMatrixRunConfig(
        matrix_manifest_sha256=manifest.sha256,
        policy_kind="act",
        max_control_cycles=4,
        max_observation_steps=5,
        execute_action_steps=1,
        wall_timeout_s=30.0,
        upstream_root=root / "UniVTAC",
        runtime_root=root / "runtime",
        act_device_name="cpu",
        simulator_device="cpu",
        launcher_args=production_univtac_launcher_args(),
        official_act_artifact_root=root / "official-act",
        stats_sha256="1" * 64,
        encoder_sha256="2" * 64,
        resources=resources,
    )


def _backend(loaded):  # type: ignore[no-untyped-def]
    return DeterministicFakeBackend(
        make_synthetic_episode(length=10),
        success_predicate_id=loaded.run_spec.success_predicate_id,
    )


def _policy(loaded):  # type: ignore[no-untyped-def]
    return DeterministicFakePolicy.for_trial(
        loaded.trial, supports_structural_absence=True
    )


def test_run_config_loads_exact_resources_and_executes_resumably(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    config = _config(tmp_path, manifest)
    config_path = tmp_path / "run-config.json"
    config_path.write_bytes(canonical_matrix_json_bytes(config.to_dict()))
    loaded = load_live_matrix_run_config(config_path, manifest)

    first = run_live_matrix(
        tmp_path / "matrix",
        manifest,
        loaded,
        backend_factory=_backend,
        policy_factory=_policy,
    )
    second = run_live_matrix(
        tmp_path / "matrix",
        manifest,
        loaded,
        backend_factory=_backend,
        policy_factory=_policy,
    )

    assert first.executed_cell_count == len(manifest.cells)
    assert first.pending_cell_count == 0
    assert all(state.status is MatrixCellStatus.COMPLETED for state in first.states)
    assert second.executed_cell_count == 0
    assert second.reused_cell_count == len(manifest.cells)


def test_run_config_rejects_missing_cell_and_cli_can_materialize_pending(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    config = _config(tmp_path, manifest)
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "run-config.json"
    manifest_path.write_bytes(canonical_matrix_json_bytes(manifest.to_dict()))
    document = config.to_dict()
    config_path.write_bytes(canonical_matrix_json_bytes(document))
    missing = dict(document)
    missing["resources"] = dict(document["resources"])
    missing["resources"].pop(next(iter(missing["resources"])))
    bad_path = tmp_path / "bad-config.json"
    bad_path.write_bytes(canonical_matrix_json_bytes(missing))
    with pytest.raises(LiveMatrixRunConfigError, match="every matrix cell"):
        load_live_matrix_run_config(bad_path, manifest)

    command = [
        sys.executable,
        "-m",
        "robotactile_benchmark.cli",
        "run-live-matrix",
        "--matrix-manifest",
        str(manifest_path),
        "--run-config",
        str(config_path),
        "--matrix-output",
        str(tmp_path / "pending-matrix"),
        "--max-new-cells",
        "0",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["executed_cell_count"] == 0
    assert payload["pending_cell_count"] == len(manifest.cells)
    assert payload["simulator_qualification_claimed"] is False

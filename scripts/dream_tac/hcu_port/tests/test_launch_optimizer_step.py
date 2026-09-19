"""Contract tests for the bounded Dream-Tac HCU optimizer-step launcher."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TextIO, cast

import pytest

from scripts.dream_tac.hcu_port import launch_optimizer_step as launcher
from scripts.dream_tac.hcu_port.optimizer_step_request import (
    ACCELERATOR_CONTRACT,
    PHASE,
    REQUEST_SCHEMA,
    HcuOptimizerStepRequest,
)
from scripts.dream_tac.hcu_port.overlay_contract import PINNED_COMMIT
from scripts.dream_tac.hcu_port.receipt import signed_receipt, verify_receipt
from scripts.dream_tac.training.training_request import DONOR_EXPERIMENT


def _request(tmp_path: Path, *, hcu_device: int = 3) -> HcuOptimizerStepRequest:
    artifacts = tmp_path / "artifacts"
    return HcuOptimizerStepRequest.from_dict(
        {
            "schema_version": REQUEST_SCHEMA,
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "dream_tac_commit": PINNED_COMMIT,
            "donor_experiment": DONOR_EXPERIMENT,
            "phase": PHASE,
            "run_name": "dream-tac-hcu-step-test",
            "dream_tac_root": str(tmp_path / "source" / "Dream-Tac-HCU"),
            "python_executable": str(tmp_path / "runtime" / "bin" / "python"),
            "hcu_environment_script": {
                "path": str(tmp_path / "runtime" / "hcu-env.sh"),
                "sha256": "9" * 64,
            },
            "runtime_pythonpath_roots": [
                str(tmp_path / "runtime" / "site"),
                str(tmp_path / "runtime" / "site-cosmos"),
            ],
            "materialization_root": str(tmp_path / "materialized"),
            "output_root": str(tmp_path / "outputs"),
            "source_manifest_sha256": "1" * 64,
            "materialization_receipt_sha256": "2" * 64,
            "t5_cache_sha256": "3" * 64,
            "base_checkpoint": {
                "path": str(artifacts / "model-480p-16fps.pt"),
                "sha256": "4" * 64,
            },
            "base_dcp_root": str(artifacts / "model-480p-16fps.dcp" / "iter_000000000"),
            "base_dcp_receipt": {
                "path": str(artifacts / "base-dcp-receipt.json"),
                "sha256": "5" * 64,
            },
            "tokenizer_checkpoint": {
                "path": str(artifacts / "tokenizer" / "tokenizer.pth"),
                "sha256": "6" * 64,
            },
            "overlay_receipt": {
                "path": str(artifacts / "overlay-receipt.json"),
                "sha256": "7" * 64,
            },
            "casa_receipt": {
                "path": str(artifacts / "casa-receipt.json"),
                "sha256": "8" * 64,
            },
            "hcu_device": hcu_device,
            "master_port": 12341,
            "nproc_per_node": 1,
            "max_iter": 1,
            "save_iter": 1,
            "batch_size": 1,
            "num_workers": 0,
            "resume_checkpoint": None,
        }
    )


def _override(command: tuple[str, ...], key: str) -> str:
    matches = tuple(item for item in command if item.startswith(f"{key}="))
    assert len(matches) == 1
    return matches[0]


def _install_gate_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = signed_receipt(
        {
            "schema_version": 1,
            "status": "passed",
            "protocol_id": "test-artifact-preflight",
        }
    )
    environment = signed_receipt(
        {
            "schema_version": 1,
            "status": "passed",
            "protocol_id": "test-environment-preflight",
        }
    )

    def write_preflight(
        request: HcuOptimizerStepRequest,
    ) -> tuple[Path, dict[str, object]]:
        return request.receipt_root / "artifact-preflight.json", preflight

    def probe_environment(
        request: HcuOptimizerStepRequest,
    ) -> dict[str, object]:
        assert request.phase == PHASE
        return environment

    monkeypatch.setattr(
        launcher,
        "write_optimizer_step_preflight_receipt",
        write_preflight,
    )
    monkeypatch.setattr(launcher, "probe_environment", probe_environment)


def _write_checkpoint_evidence(request: HcuOptimizerStepRequest) -> None:
    checkpoint = request.job_root / "checkpoints" / "iter_000000001"
    for component in ("model", "optim", "scheduler", "trainer"):
        root = checkpoint / component
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{component}.distcp").write_bytes(component.encode("ascii"))
    (checkpoint.parent / "latest_checkpoint.txt").write_text(
        f"{checkpoint.name}\n",
        encoding="utf-8",
    )


def _install_training_process(
    monkeypatch: pytest.MonkeyPatch,
    request: HcuOptimizerStepRequest,
    *,
    create_checkpoint: bool,
    log_suffix: str = "",
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def run(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(arguments))
        stream = cast(TextIO, kwargs["stdout"])
        stream.write(
            "Resuming ckpt\nLoaded checkpoint\nStarting training\nDone with training\n"
            f"{log_suffix}"
        )
        stream.flush()
        if create_checkpoint:
            _write_checkpoint_evidence(request)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def _load_result(request: HcuOptimizerStepRequest) -> dict[str, object]:
    path = request.receipt_root / launcher.LAUNCH_RESULT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    verify_receipt(payload)
    return payload


def _load_plan(request: HcuOptimizerStepRequest) -> dict[str, object]:
    path = request.receipt_root / launcher.LAUNCH_PLAN_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    verify_receipt(payload)
    return payload


def test_command_is_single_rank_and_binds_both_tactile_datasets(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    command = launcher.build_command(request)

    assert command[:4] == ("bash", "--noprofile", "--norc", "-c")
    assert command[4] == launcher._BASH_BOOTSTRAP
    assert command[5] == "robotactile-hcu-bootstrap"
    assert command[6] == str(request.hcu_environment_script.path)
    assert command[7] == str(request.python_executable)
    assert "--nnodes=1" in command
    assert "--node_rank=0" in command
    assert "--nproc_per_node=1" in command
    assert "--dryrun" not in command
    assert "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY=1" not in command
    assert "~trainer.callbacks.device_monitor" in command
    assert (
        _override(command, "checkpoint.load_path")
        == f"checkpoint.load_path={request.base_dcp_root}"
    )
    assert str(request.base_checkpoint.path) not in _override(
        command, "checkpoint.load_path"
    )
    assert (
        _override(command, "model.config.tokenizer.vae_pth")
        == f"model.config.tokenizer.vae_pth={request.tokenizer_checkpoint.path}"
    )
    for prefix in (
        "dataloader_train.dataset",
        "dataloader_train.sampler.dataset",
    ):
        assert _override(command, f"{prefix}.use_tactile") == (
            f"{prefix}.use_tactile=true"
        )
        assert _override(command, f"{prefix}.data_dir") == (
            f"{prefix}.data_dir={request.materialization_root / 'dataset'}"
        )

    resolved = launcher.build_resolved_experiment(request)
    bindings = cast(dict[str, dict[str, object]], resolved["dataset_bindings"])
    assert (
        bindings["dataloader_train.dataset"]
        == bindings["dataloader_train.sampler.dataset"]
    )
    assert bindings["dataloader_train.dataset"]["use_tactile"] is True


def test_hcu_environment_sets_all_visibility_names_and_removes_probe_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, hcu_device=5)
    monkeypatch.setenv("ROBOTACTILE_HCU_CONFIG_PROBE_ONLY", "1")

    explicit = launcher.explicit_environment(request)
    process = launcher._process_environment(request)

    assert explicit["CUDA_VISIBLE_DEVICES"] == "5"
    assert explicit["HIP_VISIBLE_DEVICES"] == "5"
    assert explicit["ROCR_VISIBLE_DEVICES"] == "5"
    assert explicit["ROBOTACTILE_HCU_TRAINING"] == "1"
    assert "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY" not in explicit
    assert "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY" not in process


def test_environment_probe_uses_exact_runtime_and_returns_signed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    observed: dict[str, object] = {}
    runtime_payload = {
        "torch_version": "2.5.1+das.opt1.dtk25042",
        "hip_version": "6.3.25405",
        "visible_device_count": 1,
        "device_name": "BW",
    }

    def run(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        observed["arguments"] = tuple(arguments)
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps(runtime_payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(launcher, "executable_path", lambda path: path)

    receipt = launcher.probe_environment(request)

    verify_receipt(receipt)
    assert observed["arguments"] == launcher._wrapped_command(
        request,
        (
            str(request.python_executable),
            "-c",
            launcher._environment_probe_script(),
        ),
    )
    assert observed["cwd"] == request.dream_tac_root
    assert observed["check"] is True
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert receipt["status"] == "passed"
    assert receipt["environment"] == runtime_payload
    assert receipt["hcu_environment_script"] == (
        request.hcu_environment_script.to_dict()
    )


def test_environment_probe_classifies_cuda_only_attention_packages_as_optional() -> (
    None
):
    script = launcher._environment_probe_script()

    required_line = next(
        line for line in script.splitlines() if line.startswith("required=")
    )
    optional_line = next(
        line for line in script.splitlines() if line.startswith("optional=")
    )
    assert "natten" not in required_line
    assert "xformers" not in required_line
    assert "natten" in optional_line
    assert "xformers" in optional_line
    assert "optional_modules" in script


def test_rc_zero_without_checkpoint_evidence_is_failed_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _install_gate_stubs(monkeypatch)
    calls = _install_training_process(
        monkeypatch,
        request,
        create_checkpoint=False,
    )

    assert launcher.execute(request) == 2

    result = _load_result(request)
    plan = _load_plan(request)
    assert len(calls) == 1
    assert calls[0] == launcher.build_command(request)
    assert plan["command"] == list(launcher.build_command(request))
    assert plan["hcu_environment_script"] == (request.hcu_environment_script.to_dict())
    assert result["status"] == "failed"
    assert result["process_returncode"] == 0
    assert result["optimizer_step_completed"] is False
    evidence = cast(dict[str, object], result["evidence"])
    assert evidence["status"] == "failed"
    components = cast(dict[str, list[object]], evidence["components"])
    assert all(not files for files in components.values())


def test_complete_checkpoint_evidence_succeeds_and_does_not_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _install_gate_stubs(monkeypatch)
    calls = _install_training_process(
        monkeypatch,
        request,
        create_checkpoint=True,
    )

    assert launcher.execute(request) == 0
    assert launcher.execute(request) == 0

    result = _load_result(request)
    assert len(calls) == 1
    assert result["status"] == "completed"
    assert result["process_returncode"] == 0
    assert result["optimizer_step_completed"] is True
    evidence = cast(dict[str, object], result["evidence"])
    assert evidence["status"] == "passed"
    assert evidence["latest_marker_value"] == "iter_000000001"
    markers = cast(dict[str, bool], evidence["required_log_markers"])
    assert all(markers.values())
    components = cast(dict[str, list[dict[str, object]]], evidence["components"])
    assert set(components) == {"model", "optim", "scheduler", "trainer"}
    assert all(len(files) == 1 for files in components.values())


def test_identifier_containing_nan_is_not_a_nonfinite_numeric_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _install_gate_stubs(monkeypatch)
    _install_training_process(
        monkeypatch,
        request,
        create_checkpoint=True,
        log_suffix="TORCH_NCCL_NAN_CHECK: 0\n",
    )

    assert launcher.execute(request) == 0
    result = _load_result(request)
    evidence = cast(dict[str, object], result["evidence"])
    assert evidence["invalid_numeric_token"] is None


@pytest.mark.parametrize("numeric_token", ["nan", "+inf", "-Infinity"])
def test_real_nonfinite_numeric_token_remains_a_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    numeric_token: str,
) -> None:
    request = _request(tmp_path)
    _install_gate_stubs(monkeypatch)
    _install_training_process(
        monkeypatch,
        request,
        create_checkpoint=True,
        log_suffix=f"applicable_loss={numeric_token}\n",
    )

    assert launcher.execute(request) == 2
    result = _load_result(request)
    evidence = cast(dict[str, object], result["evidence"])
    assert evidence["invalid_numeric_token"] == numeric_token

"""Contract tests for one Dream-Tac HCU optimizer-step request."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from scripts.dream_tac.hcu_port.optimizer_step_request import (
    HcuOptimizerStepRequest,
)


def request_payload(tmp_path: Path) -> dict[str, object]:
    def artifact(name: str, digit: str) -> dict[str, str]:
        return {
            "path": str(tmp_path / "artifacts" / name),
            "sha256": digit * 64,
        }

    return {
        "schema_version": "robotactile-dream-tac-hcu-optimizer-step-request-v1",
        "accelerator_contract": "hygon_hcu_hip_single_device_v1",
        "dream_tac_commit": "14bab51d6862fd07124745c55cd395ea5caa9fd3",
        "donor_experiment": ("cosmos_predict2_2b_480p_franka_cut_banana_20260321"),
        "phase": "p2_hcu_optimizer_step",
        "run_name": "dream-tac-hcu-one-step-v1",
        "dream_tac_root": str(tmp_path / "source" / "Dream-Tac-HCU-port"),
        "python_executable": str(tmp_path / "runtime" / "bin" / "python"),
        "hcu_environment_script": artifact("hcu-env.sh", "9"),
        "runtime_pythonpath_roots": [
            str(tmp_path / "runtime" / "site"),
            str(tmp_path / "runtime" / "site_cosmos"),
        ],
        "materialization_root": str(tmp_path / "data"),
        "output_root": str(tmp_path / "output"),
        "source_manifest_sha256": "1" * 64,
        "materialization_receipt_sha256": "2" * 64,
        "t5_cache_sha256": "3" * 64,
        "base_checkpoint": artifact("model-480p-16fps.pt", "4"),
        "base_dcp_root": str(tmp_path / "artifacts" / "base-dcp" / "iter_000000000"),
        "base_dcp_receipt": artifact("base-dcp-receipt.json", "5"),
        "tokenizer_checkpoint": artifact("tokenizer.pth", "6"),
        "overlay_receipt": artifact("overlay.json", "7"),
        "casa_receipt": artifact("casa.json", "8"),
        "hcu_device": 0,
        "nproc_per_node": 1,
        "master_port": 12341,
        "max_iter": 1,
        "save_iter": 1,
        "batch_size": 1,
        "num_workers": 0,
        "resume_checkpoint": None,
    }


def test_request_round_trip_is_single_step_and_source_bound(tmp_path: Path) -> None:
    request = HcuOptimizerStepRequest.from_dict(request_payload(tmp_path))

    assert request.to_dict() == request_payload(tmp_path)
    assert request.nproc_per_node == 1
    assert request.max_iter == 1
    assert request.batch_size == 1
    assert request.resume_checkpoint is None
    assert len(request.request_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("accelerator_contract", "nvidia_cuda_only_v1"),
        ("nproc_per_node", 2),
        ("max_iter", 2),
        ("save_iter", 2),
        ("batch_size", 2),
        ("resume_checkpoint", {"path": "/tmp/resume.pt", "sha256": "8" * 64}),
    ),
)
def test_request_rejects_non_single_step_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = request_payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        HcuOptimizerStepRequest.from_dict(payload)


def test_request_rejects_all_zero_content_identity(tmp_path: Path) -> None:
    payload = request_payload(tmp_path)
    payload["t5_cache_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="all-zero"):
        HcuOptimizerStepRequest.from_dict(payload)

    payload = request_payload(tmp_path)
    base = dict(cast(dict[str, str], payload["base_checkpoint"]))
    base["sha256"] = "0" * 64
    payload["base_checkpoint"] = base
    with pytest.raises(ValueError, match="all-zero"):
        HcuOptimizerStepRequest.from_dict(payload)


def test_request_separates_output_from_protected_inputs(tmp_path: Path) -> None:
    payload = request_payload(tmp_path)
    payload["output_root"] = str(tmp_path / "data" / "output")

    with pytest.raises(ValueError, match="protected input"):
        HcuOptimizerStepRequest.from_dict(payload)

    payload = request_payload(tmp_path)
    script = dict(cast(dict[str, str], payload["hcu_environment_script"]))
    script["path"] = str(tmp_path / "output" / "hcu-env.sh")
    payload["hcu_environment_script"] = script
    with pytest.raises(ValueError, match="environment_script.*output_root"):
        HcuOptimizerStepRequest.from_dict(payload)


def test_request_rejects_duplicate_runtime_roots(tmp_path: Path) -> None:
    payload = request_payload(tmp_path)
    root = str(tmp_path / "runtime" / "site")
    payload["runtime_pythonpath_roots"] = [root, root]

    with pytest.raises(ValueError, match="unique"):
        HcuOptimizerStepRequest.from_dict(payload)

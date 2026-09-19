"""Contract tests for the Dream-Tac P2/P3 launcher."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from scripts.dream_tac.training.launch_training import (
    build_command,
    build_overrides,
    build_resolved_experiment,
    explicit_environment,
)
from scripts.dream_tac.training.training_preflight import _resume_identity
from scripts.dream_tac.training.training_request import (
    DreamTacTrainingRequest,
    sha256_regular_file,
)


def _payload(tmp_path: Path, *, phase: str = "p2_micro") -> dict[str, object]:
    return {
        "schema_version": "robotactile-dream-tac-training-request-v1",
        "accelerator_contract": "nvidia_cuda_only_v1",
        "dream_tac_commit": "14bab51d6862fd07124745c55cd395ea5caa9fd3",
        "donor_experiment": ("cosmos_predict2_2b_480p_franka_cut_banana_20260321"),
        "phase": phase,
        "run_name": f"dream-tac-{phase}-test",
        "dream_tac_root": str(tmp_path / "source"),
        "python_executable": str(tmp_path / "runtime" / "bin" / "python"),
        "materialization_root": str(tmp_path / "data"),
        "output_root": str(tmp_path / "output"),
        "source_manifest_sha256": "1" * 64,
        "materialization_receipt_sha256": "2" * 64,
        "t5_cache_sha256": "3" * 64,
        "base_checkpoint": {
            "path": str(tmp_path / "checkpoints" / "base.pt"),
            "sha256": "4" * 64,
        },
        "resume_checkpoint": None,
        "cuda_devices": list(range(8)),
        "nproc_per_node": 8,
        "master_port": 12341,
        "max_iter": 1000 if phase == "p2_micro" else 20000,
        "save_iter": 50 if phase == "p2_micro" else 1000,
        "batch_size": 16,
        "num_workers": 0,
    }


def _dataset_override_values(overrides: set[str], *, prefix: str) -> dict[str, str]:
    marker = f"{prefix}."
    return {
        key[len(marker) :].split("=", maxsplit=1)[0]: key.split("=", maxsplit=1)[1]
        for key in overrides
        if key.startswith(marker)
    }


def test_command_and_resolved_config_force_all_tactile_contracts(
    tmp_path: Path,
) -> None:
    request = DreamTacTrainingRequest.from_dict(_payload(tmp_path))
    command = build_command(request, mode="train")
    overrides = set(build_overrides(request))
    resolved = build_resolved_experiment(request)

    assert command[0] == str(request.python_executable)
    assert "--nproc_per_node=8" in command
    assert "--config=cosmos_policy/config/config.py" in command
    assert not any(item.startswith("--config=/") for item in command)
    assert command.count("--") == 1
    direct_dataset = _dataset_override_values(
        overrides, prefix="dataloader_train.dataset"
    )
    sampler_dataset = _dataset_override_values(
        overrides, prefix="dataloader_train.sampler.dataset"
    )
    assert direct_dataset == sampler_dataset
    assert direct_dataset == {
        "data_dir": str(request.materialization_root / "dataset"),
        "chunk_size": "20",
        "t5_text_embeddings_path": str(
            request.materialization_root / "dataset" / "t5_embeddings.pkl"
        ),
        "use_tactile": "true",
        "use_proprio": "true",
        "use_wrist_images": "true",
        "use_third_person_images": "true",
        "demonstration_sampling_prob": "1.0",
        "success_rollout_sampling_prob": "0.0",
        "return_value_function_returns": "false",
    }
    assert "model.config.net.use_tactile_self_attn_bias=true" in overrides
    assert "model.config.conditioner.text.dropout_rate=0.0" in overrides
    assert "dataloader_train.batch_size=16" in overrides
    dataset = cast(dict[str, object], resolved["dataset"])
    assert dataset["split"] == "train759_only"
    assert dataset["use_tactile"] is True
    assert dataset["tactile_dropout"] is False
    dataset_bindings = cast(dict[str, dict[str, object]], resolved["dataset_bindings"])
    assert (
        dataset_bindings["dataloader_train.dataset"]
        == dataset_bindings["dataloader_train.sampler.dataset"]
    )
    assert dataset_bindings["dataloader_train.dataset"]["use_tactile"] is True
    assert explicit_environment(request)["IMAGINAIRE_OUTPUT_ROOT"] == str(
        request.output_root
    )
    assert explicit_environment(request)["DREAM_TAC_BASE_CHECKPOINT"] == str(
        request.base_checkpoint.path
    )
    assert "ROBOTACTILE_HCU_CONFIG_PROBE_ONLY" not in explicit_environment(request)
    assert "MAGINAIRE_OUTPUT_ROOT" not in explicit_environment(request)


def test_explicit_resume_loads_training_state(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["resume_checkpoint"] = {
        "path": str(tmp_path / "checkpoints" / "iter_000001000.pt"),
        "sha256": "5" * 64,
    }
    request = DreamTacTrainingRequest.from_dict(payload)
    overrides = set(build_overrides(request))

    assert "checkpoint.load_training_state=true" in overrides
    assert "checkpoint.strict_resume=true" in overrides
    assert (
        f"checkpoint.load_path={tmp_path / 'checkpoints' / 'iter_000001000.pt'}"
        in overrides
    )


def test_same_job_latest_must_match_bound_resume(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path
        / "output"
        / "robotactile_dream_tac"
        / "univtac_p2_micro"
        / "dream-tac-p2_micro-test"
        / "checkpoints"
        / "iter_000000050.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    marker = checkpoint.parent / "latest_checkpoint.txt"
    marker.write_text(f"{checkpoint.name}\n", encoding="utf-8")
    payload = _payload(tmp_path)
    payload["resume_checkpoint"] = {
        "path": str(checkpoint),
        "sha256": sha256_regular_file(checkpoint),
    }
    request = DreamTacTrainingRequest.from_dict(payload)

    identity = _resume_identity(request)

    assert identity["mode"] == "same_job_latest"
    assert identity["checkpoint_path"] == str(checkpoint)

    payload["resume_checkpoint"] = None
    without_resume = DreamTacTrainingRequest.from_dict(payload)
    with pytest.raises(ValueError, match="explicit resume_checkpoint"):
        _resume_identity(without_resume)


def test_request_phase_limits_and_cuda_topology(tmp_path: Path) -> None:
    invalid_phase = _payload(tmp_path)
    invalid_phase["max_iter"] = 1001
    with pytest.raises(ValueError, match="cannot exceed"):
        DreamTacTrainingRequest.from_dict(invalid_phase)

    invalid_devices = _payload(tmp_path)
    invalid_devices["cuda_devices"] = [0, 1]
    with pytest.raises(ValueError, match="uniquely match"):
        DreamTacTrainingRequest.from_dict(invalid_devices)


def test_training_launcher_import_does_not_load_n0() -> None:
    code = """
import sys
import scripts.dream_tac.training.launch_training
loaded = sorted(name for name in sys.modules if name.startswith('scripts.n0_twam'))
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "name",
    ("p2_micro.example.json", "p3_full.example.json"),
)
def test_checked_in_examples_match_request_contract(name: str) -> None:
    root = Path(__file__).resolve().parents[4]
    path = root / "configs" / "experiments" / "dream_tac" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    request = DreamTacTrainingRequest.from_dict(payload)

    assert request.nproc_per_node == 8
    assert request.cuda_devices == tuple(range(8))

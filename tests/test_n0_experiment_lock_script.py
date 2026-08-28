from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.n0_twam.freeze_experiment_lock import (
    _camera_receipts,
    _prompt_contract,
    _validate_args,
    _write_once,
)


def _args(tmp_path: Path, **updates: object) -> argparse.Namespace:
    values = {
        "clean_seed": 100,
        "parity_seed": 90,
        "planned_live_artifact": tmp_path / "future-artifact",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_experiment_lock_rejects_seed_reuse_and_existing_trial(tmp_path: Path) -> None:
    _validate_args(_args(tmp_path))
    with pytest.raises(ValueError, match="distinct"):
        _validate_args(_args(tmp_path, clean_seed=90))
    planned = tmp_path / "existing"
    planned.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _validate_args(_args(tmp_path, planned_live_artifact=planned))


def test_camera_inventory_and_lock_are_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "upstream"
    root.mkdir()
    (root / "camera.usd").write_text("camera")

    receipts = _camera_receipts(root, ("camera.usd",))
    assert receipts["camera.usd"]["size_bytes"] == 6
    with pytest.raises(ValueError, match="invalid"):
        _camera_receipts(root, ("../escape.usd",))

    output = tmp_path / "experiment_lock.json"
    document = {"schema_version": "test-v1", "trial": "lift_bottle"}
    first = _write_once(output, document)
    second = _write_once(output, document)
    assert first == second
    assert json.loads(output.read_text()) == document
    with pytest.raises(FileExistsError, match="differs"):
        _write_once(output, {"schema_version": "test-v2"})


def test_prompt_contract_separates_simulator_and_model_instructions() -> None:
    training = "Grasp and lift a bottle off a surface near a wall"
    contract = _prompt_contract(
        "lift_bottle",
        {"lift_bottle": training},
        "Grasp the bottle securely and lift it above the success height.",
    )

    assert contract["policy_training_prompt"] == training
    assert contract["simulator_instruction_is_model_input"] is False
    with pytest.raises(ValueError, match="policy prompts disagree"):
        _prompt_contract(
            "lift_bottle",
            {"lift_bottle": "wrong"},
            "simulator instruction",
        )

"""Focused checks for released-model queue scoring and worker routing."""

from argparse import Namespace
from pathlib import Path

from scripts.retrained_evaluation.released_clean_seed_queue import (
    valid_terminal,
    worker_command,
)


def test_terminal_requires_real_validated_observations() -> None:
    terminal = {
        "score_eligible": True,
        "validation_passed": True,
        "terminal_status": "success",
        "observation_count": 25,
    }
    assert valid_terminal(terminal)
    assert valid_terminal({**terminal, "terminal_status": "timeout"})
    assert not valid_terminal({**terminal, "terminal_status": "crash"})
    assert not valid_terminal({**terminal, "observation_count": 0})
    assert not valid_terminal({**terminal, "validation_passed": False})


def test_worker_keeps_model_native_protocol() -> None:
    args = Namespace(
        isaac_python=Path("/deployment/runtime/isaac.sh"),
        root=Path("/deployment"),
        config=Path("/config.json"),
        model="n0_twam",
        port=37200,
    )
    command = worker_command(args, Path("/request.json"))
    assert "robotactile_n0_training_60hz_ee_v1" in command
    assert "metrics_only_v1" in command
    args.model = "act"
    assert "--action-execution-contract" not in worker_command(
        args, Path("/request.json")
    )

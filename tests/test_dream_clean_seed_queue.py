from pathlib import Path

import pytest

from robotactile_benchmark.contracts import canonical_hash
from scripts.retrained_evaluation import dream_clean_seed_queue as queue
from scripts.retrained_evaluation.group import write_json
from scripts.retrained_evaluation.serve_dream import (
    EULER_CONTRACT,
    TACTILE_GATE_CONTRACT,
)


def test_freeze_binding_updates_endpoint_and_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "frozen.json"
    write_json(
        source,
        {
            "binding_sha256": "parent",
            "model": "dream_tac",
            "tasks": {"lift_can": {"prompt": "lift"}},
        },
    )

    frozen = queue._freeze_binding(source, output, port=29820)

    assert frozen["endpoint"] == "http://127.0.0.1:29820"
    assert frozen["parent_binding_sha256"] == "parent"
    assert frozen["tactile_gate_contract"] == TACTILE_GATE_CONTRACT
    assert frozen["euler_contract"] == EULER_CONTRACT
    assert frozen["binding_sha256"] == canonical_hash(
        {key: value for key, value in frozen.items() if key != "binding_sha256"}
    )


def test_freeze_binding_rejects_other_models(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_json(source, {"model": "n0_vtla"})
    with pytest.raises(ValueError, match="Dream-Tac"):
        queue._freeze_binding(source, tmp_path / "output.json", port=29820)


def test_parser_defaults_to_formal_seed_cohort() -> None:
    args = queue._parser().parse_args(
        [
            "run",
            "--binding",
            "/binding.json",
            "--task",
            "lift_can",
            "--campaign",
            "/campaign",
            "--code",
            "/code",
            "--package",
            "/package",
            "--runtime-python",
            "/runtime/python",
            "--isaac-python",
            "/isaac/python.sh",
            "--port",
            "29820",
        ]
    )
    assert args.seed_start == 1_000_000
    assert args.seed_count == 100

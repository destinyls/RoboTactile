from pathlib import Path

import pytest

from robotactile_benchmark.contracts import canonical_hash
from scripts.retrained_evaluation import ftp1_clean_seed_queue as queue
from scripts.retrained_evaluation.group import write_json


def _metadata(task: str, bundle: str) -> dict[str, object]:
    return {
        "model": "ftp1_policy",
        "serve_bundle_sha256": bundle,
        "task_id": task,
    }


def test_freeze_binding_updates_endpoint_and_transport_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "frozen.json"
    write_json(
        source,
        {
            "binding_sha256": "parent",
            "model": "ftp1_policy",
            "tasks": {
                "lift_can": {
                    "prompt": "lift",
                    "transport_metadata": _metadata("lift_can", "parent"),
                }
            },
        },
    )

    frozen = queue._freeze_binding(source, output, port=29960)

    assert frozen["endpoint"] == "tcp://127.0.0.1:29960"
    assert frozen["parent_binding_sha256"] == "parent"
    unhashed = {
        **frozen,
        "tasks": {
            "lift_can": {
                "prompt": "lift",
            }
        },
    }
    unhashed.pop("binding_sha256")
    assert frozen["binding_sha256"] == canonical_hash(unhashed)
    assert frozen["tasks"]["lift_can"]["transport_metadata"] == _metadata(
        "lift_can", frozen["binding_sha256"]
    )


def test_freeze_binding_rejects_other_models(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    write_json(source, {"model": "n0_vtla"})
    with pytest.raises(ValueError, match="FTP-1"):
        queue._freeze_binding(source, tmp_path / "output.json", port=29960)


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
            "29960",
        ]
    )
    assert args.seed_start == 1_000_000
    assert args.seed_count == 100

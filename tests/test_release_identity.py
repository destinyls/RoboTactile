"""Public repository, distribution, import, and CLI naming contract."""

from importlib import metadata
from pathlib import Path

import robotactile_benchmark

RELEASE_VERSION = "0.6.0"


def test_public_repository_and_python_identity_are_distinct() -> None:
    root = Path(__file__).resolve().parents[1]

    assert root.name == "RoboTactile"
    assert robotactile_benchmark.__name__ == "robotactile_benchmark"
    assert metadata.metadata("robotactile-benchmark")["Name"] == (
        "robotactile-benchmark"
    )
    assert robotactile_benchmark.__version__ == RELEASE_VERSION
    assert metadata.version("robotactile-benchmark") == RELEASE_VERSION

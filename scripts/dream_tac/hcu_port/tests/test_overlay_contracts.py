"""Static contract tests for the Dream-Tac HCU overlay."""

from pathlib import Path

import pytest

from scripts.dream_tac.hcu_port import overlay
from scripts.dream_tac.hcu_port.overlay import OverlayConfig


def test_pure_python_runtime_requirements_match_declared_wheel_hashes() -> None:
    requirements = (
        Path(overlay.__file__)
        .with_name("pure_python_requirements.txt")
        .read_text(encoding="utf-8")
    )
    assert len(overlay.PURE_PYTHON_RUNTIME_PINS) == 4
    for pin in overlay.PURE_PYTHON_RUNTIME_PINS:
        assert len(pin.sha256) == 64
        assert (
            f"{pin.package}=={pin.version} --hash=sha256:{pin.sha256}" in requirements
        )


def test_overlay_config_requires_explicit_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_checkout must be an absolute path"):
        OverlayConfig(
            pinned_source_checkout=tmp_path,
            target_checkout=Path("relative-target"),
            receipt=tmp_path / "receipt.json",
        )

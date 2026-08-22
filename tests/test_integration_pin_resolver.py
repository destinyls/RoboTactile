"""Tests for shell-facing canonical integration pin resolution."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_resolver() -> ModuleType:
    path = ROOT / "integrations/resolve_pin.py"
    spec = importlib.util.spec_from_file_location("resolve_pin", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolver_reads_canonical_source_fields() -> None:
    resolver = _load_resolver()

    assert resolver.resolve_field("act_runtime", "source_directory") == "WorldArena"
    assert resolver.resolve_field("isaaclab", "commit_sha") == (
        "90b79bb2d44feb8d833f260f2bf37da3487180ba"
    )
    assert resolver.resolve_field("curobo", "license_spdx") == (
        "LicenseRef-NVIDIA-NonCommercial"
    )


def test_resolver_rejects_unknown_integration() -> None:
    resolver = _load_resolver()

    with pytest.raises(SystemExit, match="unknown or duplicated"):
        resolver.resolve_field("missing", "commit_sha")

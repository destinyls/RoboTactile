"""CLI coverage for preparing all frozen N0 UniVTAC task configurations."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/n0_twam/prepare_official_artifacts.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("n0_prepare_all_tasks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Generated:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def to_dict(self) -> dict[str, object]:
        return {"integration_id": "n0_twam", "task_id": self.task_id}


def test_prepare_all_tasks_selects_complete_frozen_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    selected: list[str] = []

    def prepare(
        *, root: Path, task_id: str, device: str, config_label: str | None
    ) -> _Generated:
        assert root == tmp_path
        assert device == "cuda"
        assert config_label is None
        selected.append(task_id)
        return _Generated(task_id)

    monkeypatch.setattr(module, "_prepare_task", prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--all-tasks",
            "--skip-download",
        ],
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_count"] == 8
    assert len(selected) == 8
    assert len(set(selected)) == 8


def test_prepare_cli_requires_exactly_one_task_selection_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--root", str(tmp_path), "--skip-download"],
    )

    with pytest.raises(SystemExit, match="select exactly one"):
        module.main()


def test_existing_task_configuration_is_validated_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config_root = tmp_path / "configs/pull_out_key"
    config_root.mkdir(parents=True)
    manifest = config_root / "artifact_manifest.json"
    config = config_root / "integration_config.json"
    manifest.write_text("{}\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "resolve_n0_runtime_artifacts",
        lambda path: SimpleNamespace(manifest=SimpleNamespace(task_id="pull_out_key")),
    )
    monkeypatch.setattr(
        module,
        "load_model_integration_config",
        lambda integration_id, path: SimpleNamespace(device="cuda"),
    )
    monkeypatch.setattr(
        module,
        "prepare_official_n0_artifacts",
        lambda **kwargs: pytest.fail("valid existing config must be reused"),
    )

    generated = module._prepare_task(
        root=tmp_path, task_id="pull_out_key", device="cuda"
    )

    assert generated.artifact_manifest_path == manifest.absolute()
    assert generated.integration_config_path == config.absolute()


def test_config_label_selects_a_non_destructive_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    config_root = tmp_path / "configs/lift_bottle-source-c43a216"
    config_root.mkdir(parents=True)
    manifest = config_root / "artifact_manifest.json"
    config = config_root / "integration_config.json"
    manifest.write_text("{}\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "resolve_n0_runtime_artifacts",
        lambda path: SimpleNamespace(manifest=SimpleNamespace(task_id="lift_bottle")),
    )
    monkeypatch.setattr(
        module,
        "load_model_integration_config",
        lambda integration_id, path: SimpleNamespace(device="cuda"),
    )

    generated = module._prepare_task(
        root=tmp_path,
        task_id="lift_bottle",
        device="cuda",
        config_label="source-c43a216",
    )

    assert generated.artifact_manifest_path == manifest.absolute()
    assert generated.integration_config_path == config.absolute()

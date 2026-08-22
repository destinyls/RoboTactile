"""Repo-contained deployment layout and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotactile_benchmark.cli import _parser, main
from robotactile_benchmark.deployment import (
    DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL,
    DeploymentLayout,
    DeploymentLayoutError,
    diagnose_deployment,
    discover_repository_root,
    initialize_deployment_layout,
    load_deployment_layout_receipt,
    resolve_deployment_root,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_root_is_repo_contained_and_precedence_is_explicit(
    tmp_path: Path,
) -> None:
    environment_root = tmp_path / "from-environment"
    explicit_root = tmp_path / "from-cli"

    assert discover_repository_root(ROOT / "tests") == ROOT
    assert resolve_deployment_root(repository_start=ROOT / "docs") == (
        ROOT / "deployment"
    )
    assert (
        resolve_deployment_root(
            environ={"ROBOTACTILE_DEPLOY_ROOT": str(environment_root)},
            repository_start=ROOT,
        )
        == environment_root
    )
    assert (
        resolve_deployment_root(
            explicit_root,
            environ={"ROBOTACTILE_DEPLOY_ROOT": str(environment_root)},
            repository_start=ROOT,
        )
        == explicit_root
    )


def test_invalid_or_undiscoverable_roots_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(DeploymentLayoutError, match="must be absolute"):
        resolve_deployment_root(Path("relative"))
    with pytest.raises(DeploymentLayoutError, match="too broad"):
        resolve_deployment_root(Path("/"))
    with pytest.raises(DeploymentLayoutError, match="cannot discover"):
        resolve_deployment_root(environ={}, repository_start=tmp_path)
    with pytest.raises(DeploymentLayoutError, match="non-empty"):
        resolve_deployment_root(
            environ={"ROBOTACTILE_DEPLOY_ROOT": ""}, repository_start=ROOT
        )


def test_initialization_is_idempotent_strict_and_reloadable(tmp_path: Path) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")

    first = initialize_deployment_layout(layout)
    second = initialize_deployment_layout(layout)

    assert first == second
    assert first.evidence_level == DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL
    assert first.simulator_execution_claimed is False
    assert first.task_execution_claimed is False
    assert load_deployment_layout_receipt(layout.receipt_path) == first
    assert all(path.is_dir() for path in layout.directory_paths())
    assert layout.path("sources") == layout.root / "sources"
    assert layout.path("artifacts/models/act") == (layout.root / "artifacts/models/act")


def test_initialization_rejects_conflicts_and_symlinks(tmp_path: Path) -> None:
    conflict_root = tmp_path / "conflict"
    (conflict_root / "artifacts").mkdir(parents=True)
    (conflict_root / "sources").write_text("not a directory", encoding="utf-8")
    with pytest.raises(DeploymentLayoutError, match="must be a directory"):
        initialize_deployment_layout(DeploymentLayout(conflict_root))

    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(real, target_is_directory=True)
    with pytest.raises(DeploymentLayoutError, match="cannot be a symlink"):
        DeploymentLayout(symlink)


def test_doctor_reports_core_and_live_profiles_without_writing(tmp_path: Path) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    initialize_deployment_layout(layout)

    core = diagnose_deployment(layout, "core")
    act = diagnose_deployment(layout, "act-univtac")
    n0 = diagnose_deployment(layout, "n0-univtac")

    assert core.passed is True
    assert act.passed is False
    assert n0.passed is False
    assert {item.check_id for item in act.checks if not item.passed} == {
        "act_source",
        "isaac_python",
        "univtac_source",
    }
    assert {item.check_id for item in n0.checks if not item.passed} == {
        "isaac_python",
        "n0_source",
        "univtac_source",
    }
    assert act.to_dict()["simulator_execution_claimed"] is False


def test_public_cli_initializes_shows_and_diagnoses_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "deployment"
    help_text = _parser().format_help()
    assert "deployment" in help_text

    assert main(("deployment", "show", "--root", str(root), "--json")) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["root"] == str(root)
    assert not root.exists()

    assert main(("deployment", "init", "--root", str(root))) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["simulator_execution_claimed"] is False
    assert Path(initialized["receipt_path"]).is_file()

    assert main(("deployment", "doctor", "--root", str(root), "--profile", "core")) == 0
    diagnosed = json.loads(capsys.readouterr().out)
    assert diagnosed["passed"] is True

    assert main(("deployment", "doctor", "--root", str(root))) == 0
    default_diagnosed = json.loads(capsys.readouterr().out)
    assert default_diagnosed["profile"] == "core"


def test_root_and_model_specific_help_explain_the_golden_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _parser()

    assert "setup --model act" in parser.format_help()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(("integrations", "configure", "act", "--help"))
    assert exit_info.value.code == 0
    act_help = capsys.readouterr().out
    assert "--artifact-root" in act_help
    assert "--checkpoint" not in act_help


def test_receipt_tampering_and_unknown_layout_paths_are_rejected(
    tmp_path: Path,
) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    initialize_deployment_layout(layout)
    layout.receipt_path.write_bytes(
        layout.receipt_path.read_bytes().replace(
            b'"task_execution_claimed":false',
            b'"task_execution_claimed":true',
        )
    )

    with pytest.raises(DeploymentLayoutError):
        load_deployment_layout_receipt(layout.receipt_path)
    with pytest.raises(KeyError):
        layout.path("unknown")

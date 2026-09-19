"""Repo-contained deployment layout and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotactile_benchmark.cli import _parser, main
from robotactile_benchmark.contracts import canonical_hash
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
from robotactile_benchmark.deployment.contracts import DeploymentLayoutReceipt
from robotactile_benchmark.deployment.layout import write_deployment_layout_receipt

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
    assert layout.path("artifacts/models/dream_tac") == (
        layout.root / "artifacts/models/dream_tac"
    )
    assert layout.path("artifacts/models/n0_vtla") == (
        layout.root / "artifacts/models/n0_vtla"
    )
    assert layout.path("requests/three-condition") == (
        layout.root / "requests/three-condition"
    )


def test_initialization_accepts_frozen_four_condition_layout_additively(
    tmp_path: Path,
) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    legacy_relative = tuple(
        "requests/four-condition"
        if relative == "requests/three-condition"
        else relative
        for relative in (
            path.relative_to(layout.root).as_posix()
            for path in layout.directory_paths()
        )
    )
    layout.root.mkdir(parents=True)
    for relative in legacy_relative:
        (layout.root / relative).mkdir(parents=True, exist_ok=True)
    legacy = DeploymentLayoutReceipt(
        root_path_sha256=canonical_hash(str(layout.root)),
        directories=legacy_relative,
    )
    write_deployment_layout_receipt(layout.receipt_path, legacy)
    original = layout.receipt_path.read_bytes()

    loaded = initialize_deployment_layout(layout)

    assert loaded == legacy
    assert layout.receipt_path.read_bytes() == original
    assert (layout.root / "requests/three-condition").is_dir()
    assert (layout.root / "requests/four-condition").is_dir()


def test_initialization_accepts_pre_n0_vtla_layout_additively(
    tmp_path: Path,
) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    legacy_relative = tuple(
        "requests/four-condition"
        if relative == "requests/three-condition"
        else relative
        for relative in (
            path.relative_to(layout.root).as_posix()
            for path in layout.directory_paths()
        )
        if relative != "artifacts/models/n0_vtla"
    )
    layout.root.mkdir(parents=True)
    for relative in legacy_relative:
        (layout.root / relative).mkdir(parents=True, exist_ok=True)
    legacy = DeploymentLayoutReceipt(
        root_path_sha256=canonical_hash(str(layout.root)),
        directories=legacy_relative,
    )
    write_deployment_layout_receipt(layout.receipt_path, legacy)
    original = layout.receipt_path.read_bytes()

    loaded = initialize_deployment_layout(layout)

    assert loaded == legacy
    assert layout.receipt_path.read_bytes() == original
    assert (layout.root / "artifacts/models/n0_vtla").is_dir()
    assert (layout.root / "requests/three-condition").is_dir()
    assert (layout.root / "requests/four-condition").is_dir()


def test_initialization_accepts_pre_dream_tac_layout_additively(
    tmp_path: Path,
) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    legacy_relative = tuple(
        path.relative_to(layout.root).as_posix()
        for path in layout.directory_paths()
        if path.relative_to(layout.root).as_posix() != "artifacts/models/dream_tac"
    )
    layout.root.mkdir(parents=True)
    for relative in legacy_relative:
        (layout.root / relative).mkdir(parents=True, exist_ok=True)
    legacy = DeploymentLayoutReceipt(
        root_path_sha256=canonical_hash(str(layout.root)),
        directories=legacy_relative,
    )
    write_deployment_layout_receipt(layout.receipt_path, legacy)
    original = layout.receipt_path.read_bytes()

    loaded = initialize_deployment_layout(layout)

    assert loaded == legacy
    assert layout.receipt_path.read_bytes() == original
    assert (layout.root / "artifacts/models/dream_tac").is_dir()


def test_initialization_accepts_pre_new_model_integrations_layout(
    tmp_path: Path,
) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    omitted = {
        "artifacts/models/dream_tac",
        "artifacts/models/ftp1_policy",
        "artifacts/models/n0_vtla",
    }
    legacy_relative = tuple(
        "requests/four-condition"
        if relative == "requests/three-condition"
        else relative
        for relative in (
            path.relative_to(layout.root).as_posix()
            for path in layout.directory_paths()
        )
        if relative not in omitted
    )
    layout.root.mkdir(parents=True)
    for relative in legacy_relative:
        (layout.root / relative).mkdir(parents=True, exist_ok=True)
    legacy = DeploymentLayoutReceipt(
        root_path_sha256=canonical_hash(str(layout.root)),
        directories=legacy_relative,
    )
    write_deployment_layout_receipt(layout.receipt_path, legacy)
    original = layout.receipt_path.read_bytes()

    loaded = initialize_deployment_layout(layout)

    assert loaded == legacy
    assert layout.receipt_path.read_bytes() == original
    assert (layout.root / "artifacts/models/dream_tac").is_dir()
    assert (layout.root / "artifacts/models/ftp1_policy").is_dir()
    assert (layout.root / "artifacts/models/n0_vtla").is_dir()
    assert (layout.root / "requests/three-condition").is_dir()
    assert (layout.root / "requests/four-condition").is_dir()


def test_initialization_rejects_other_valid_legacy_inventory(tmp_path: Path) -> None:
    layout = DeploymentLayout(tmp_path / "deployment")
    layout.root.mkdir(parents=True)
    incompatible = DeploymentLayoutReceipt(
        root_path_sha256=canonical_hash(str(layout.root)),
        directories=("sources",),
    )
    write_deployment_layout_receipt(layout.receipt_path, incompatible)

    with pytest.raises(FileExistsError, match="different layout receipt"):
        initialize_deployment_layout(layout)


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

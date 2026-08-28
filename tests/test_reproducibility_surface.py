"""Executable contracts for the public reproduction entry points."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.reporting import load_reporting_spec

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PUBLIC_GUIDES = (
    ROOT / "README.md",
    ROOT / "docs/reproducibility.md",
    ROOT / "docs/quickstart.md",
    ROOT / "docs/model_integrations.md",
    ROOT / "docs/benchmark_workflow.md",
    ROOT / "docs/isaac_sim.md",
    ROOT / "scripts/README.md",
    ROOT / "scripts/live_univtac/README.md",
    ROOT / "requirements/README.md",
)


def _local_link_target(document: Path, raw_target: str) -> Optional[Path]:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith("#"):
        return None
    if target.startswith(("http://", "https://", "mailto:")):
        return None
    path_text = unquote(target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0])
    return (document.parent / path_text).resolve()


def test_public_reproduction_guides_have_resolvable_local_links() -> None:
    missing: list[str] = []
    for document in PUBLIC_GUIDES:
        assert document.is_file(), document.relative_to(ROOT)
        for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = _local_link_target(document, raw_target)
            if target is not None and not target.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not missing, "\n".join(missing)


def test_reporting_example_is_strictly_loadable_and_complete() -> None:
    spec = load_reporting_spec(ROOT / "configs/reporting_spec.example.json")

    assert set(spec.primary_operator_ids) == set(CORE_OPERATOR_IDS)
    assert spec.primary_severity_levels == (1, 2, 3, 4, 5)
    assert spec.bootstrap_resamples == 10_000
    assert not spec.matched_control_qualified


def test_bootstrap_help_is_safe_and_documents_the_default_environment() -> None:
    result = subprocess.run(
        ("bash", "scripts/bootstrap_pip.sh", "--help"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Usage: bootstrap_pip.sh" in result.stdout
    assert ".venv" in result.stdout


def test_documented_capability_boundary_is_explicit() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")

    assert "N0 UniVTAC Clean" in readme
    assert "Campaign driver pending" in readme
    assert "current complete live primary matrix is ACT-specific" in guide
    assert "Do not present those commands as an N0-TWAM fault benchmark" in guide


def test_release_install_uses_an_exact_wheel_and_locked_visualization() -> None:
    requirements = (ROOT / "requirements/README.md").read_text(encoding="utf-8")
    isaac = (ROOT / "docs/isaac_sim.md").read_text(encoding="utf-8")

    assert "robotactile_benchmark-*.whl" not in requirements
    assert "requirements/visualization.lock.txt" in requirements
    assert "requirements/visualization.lock.txt" in isaac
    assert "pip install -e" not in isaac

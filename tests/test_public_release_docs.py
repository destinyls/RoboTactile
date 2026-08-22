"""Publication-facing documentation, examples, and notice inventory."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "BENCHMARK_CARD.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/installation.md",
    "docs/deployment_layout.md",
    "docs/external_dependencies.md",
    "docs/isaac_sim.md",
    "docs/quickstart.md",
    "docs/model_integrations.md",
    "docs/evidence_levels.md",
    "requirements/README.md",
    "requirements/core.lock.txt",
    "requirements/dev.lock.txt",
    "scripts/bootstrap_pip.sh",
    "examples/act/README.md",
    "examples/act/request.json",
    "examples/n0_twam/README.md",
    "examples/n0_twam/request.json",
    "integrations/install_pinned_repo.sh",
    "integrations/install_act_runtime.sh",
    "integrations/install_n0_twam.sh",
    "integrations/install_univtac.sh",
)


def test_publication_inventory_exists_and_uses_safe_content() -> None:
    missing = tuple(path for path in REQUIRED if not (ROOT / path).is_file())
    assert not missing

    text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in REQUIRED)
    forbidden = (
        "49.235.",
        "10.232.",
        "/mnt/data/task",
        "/Users/yanglei",
        "password=",
        "simulator_qualification_claimed=true",
    )
    assert all(token not in text for token in forbidden)


def test_notice_and_examples_present_symmetric_model_boundaries() -> None:
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    act = (ROOT / "examples/act/README.md").read_text(encoding="utf-8")
    n0 = (ROOT / "examples/n0_twam/README.md").read_text(encoding="utf-8")

    assert "Apache-2.0" in notice
    assert "CC-BY-NC-SA-4.0" in notice
    assert "BSD-3-Clause" in notice
    assert "NVIDIA non-commercial" in notice
    assert "ACT" in act and "PolicyAdapter" in act
    assert "N0-TWAM" in n0 and "PolicyAdapter" in n0
    assert "unsupported_contract" in n0


def test_install_scripts_are_syntax_valid_and_contain_no_secret() -> None:
    for relative in REQUIRED:
        if not relative.endswith(".sh"):
            continue
        path = ROOT / relative
        subprocess.run(("bash", "-n", str(path)), check=True)
        content = path.read_text(encoding="utf-8")
        assert re.search(r"(?i)(password|api[_-]?key|token)=", content) is None


def test_readme_links_to_public_release_guides() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for target in (
        "BENCHMARK_CARD.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/installation.md",
        "docs/deployment_layout.md",
        "docs/external_dependencies.md",
        "docs/isaac_sim.md",
        "docs/model_integrations.md",
        "docs/quickstart.md",
        "requirements/README.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        assert f"]({target})" in readme


def test_isaac_guide_covers_install_run_and_evidence_boundaries() -> None:
    guide = (ROOT / "docs/isaac_sim.md").read_text(encoding="utf-8")

    required_tokens = (
        "Ubuntu 22.04",
        "RTX 3090",
        "Isaac Sim 4.5.0",
        "IsaacLab v2.1.1",
        "cuRobo v0.7.7",
        "nvidia-smi",
        "install_isaac_sim_4_5.sh",
        "smoke_isaac_sim_4_5.sh",
        "install_isaaclab_v2_1_1.sh",
        "install_curobo_v0_7_7.sh",
        "generate_pull_out_key_matrix.py",
        "robotactile preflight-live",
        "robotactile live-univtac-run",
        "infrastructure_launch_only",
        "unqualified_live_univtac_execution_v1",
        "simulator_qualification_claimed=false",
    )
    assert all(token in guide for token in required_tokens)


def test_isaac_guide_references_existing_repository_entry_points() -> None:
    for relative in (
        "scripts/live_univtac/install_isaac_sim_4_5.sh",
        "scripts/live_univtac/smoke_isaac_sim_4_5.sh",
        "scripts/live_univtac/install_isaaclab_v2_1_1.sh",
        "scripts/live_univtac/install_curobo_v0_7_7.sh",
        "scripts/live_univtac/generate_pull_out_key_matrix.py",
    ):
        assert (ROOT / relative).is_file()


def test_public_sources_contain_no_personal_deployment_root() -> None:
    suffixes = {".md", ".py", ".sh", ".json", ".toml", ".yml", ".yaml"}
    paths = tuple(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != Path(__file__)
        and path.suffix in suffixes
        and not (
            {
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                "deployment",
                "dist",
            }
            & set(path.parts)
        )
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "/data1/yanglei" not in text
    assert "/Users/yanglei" not in text


def test_pip_workflow_is_hash_locked_and_does_not_require_uv() -> None:
    core = (ROOT / "requirements/core.lock.txt").read_text(encoding="utf-8")
    dev = (ROOT / "requirements/dev.lock.txt").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts/bootstrap_pip.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    for payload in (core, dev):
        requirement_blocks = re.split(r"\n(?=[A-Za-z0-9])", payload)
        requirement_blocks = tuple(
            block for block in requirement_blocks if "==" in block
        )
        assert requirement_blocks
        assert all("--hash=sha256:" in block for block in requirement_blocks)
    assert "numpy==" in core and "typing-extensions==" in core
    assert all(
        token in dev for token in ("hatchling==", "mypy==", "pytest==", "ruff==")
    )
    assert "--require-hashes" in bootstrap
    assert "-m ensurepip --upgrade" in bootstrap
    assert "--no-deps" in bootstrap and "--no-build-isolation" in bootstrap
    assert " -e " not in bootstrap
    assert "uv run" not in makefile and "uv build" not in makefile
    assert "setup-uv" not in workflow and "uv run" not in workflow
    assert "python -m pip install --require-hashes" in workflow


def test_public_commands_use_python_after_virtual_environment_activation() -> None:
    public_guides = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "CONTRIBUTING.md",
            "docs/benchmark_workflow.md",
            "docs/installation.md",
            "docs/isaac_sim.md",
            "docs/paper_code_traceability.md",
            "requirements/README.md",
            "scripts/live_univtac/README.md",
        )
    }

    assert all(".venv/bin/python" not in text for text in public_guides.values())
    assert "source .venv/bin/activate" in public_guides["docs/installation.md"]
    assert "python -m pip install" in public_guides["docs/installation.md"]
    assert "python -m pytest" in public_guides["CONTRIBUTING.md"]
    assert "python scripts/live_univtac/" in public_guides["docs/benchmark_workflow.md"]

    isaac_guide = public_guides["docs/isaac_sim.md"]
    assert '"$ISAAC_SIM_PATH/python.sh"' in isaac_guide

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts/bootstrap_pip.sh").read_text(encoding="utf-8")
    assert "PYTHON ?= $(VENV)/bin/python" in makefile
    assert 'VENV_PYTHON="$VENV_PATH/bin/python"' in bootstrap


def test_public_workflow_does_not_include_cpu_only_validation_sections() -> None:
    public_guides = tuple(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "docs/quickstart.md",
            "docs/benchmark_workflow.md",
            "docs/isaac_sim.md",
        )
    )
    forbidden = (
        "CPU-only validation",
        "Five-minute CPU quickstart",
        "CPU UniVTAC contract qualification",
        "Verify the CPU contract layer",
        "Run CPU qualification before allocating the simulator",
    )
    assert all(token not in text for text in public_guides for token in forbidden)


def test_public_release_has_no_cpu_qualification_entrypoint_or_example() -> None:
    public_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "BENCHMARK_CARD.md",
            "docs/evidence_levels.md",
            "docs/installation.md",
            "docs/isaac_sim.md",
            "docs/model_integrations.md",
            "docs/paper_code_traceability.md",
            "examples/act/README.md",
            "examples/n0_twam/README.md",
        )
    )
    assert "univtac-cpu-qualify" not in public_text
    assert "policy-cpu-qualify" not in public_text
    assert "qualify_cpu.sh" not in public_text

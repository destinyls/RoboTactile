"""Publication-facing documentation, examples, and notice inventory."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from robotactile_benchmark.execution.contracts import (
    production_univtac_launcher_args,
)
from robotactile_benchmark.execution.loading import (
    load_live_univtac_request,
    load_live_univtac_run,
)
from robotactile_benchmark.integrations.act.artifacts import (
    load_act_artifact_manifest,
)
from robotactile_benchmark.integrations.dream_tac.artifacts import (
    load_dream_tac_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_twam.artifacts import (
    load_n0_twam_artifact_manifest,
)
from robotactile_benchmark.integrations.n0_vtla.artifacts import (
    load_n0_vtla_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
N0_TWAM_COMMIT = "c43a2160dd31c449d92b28eab52c0e2f09e4738a"
N0_VTLA_COMMIT = "03a0ce4d7091ca2354864796770715aa212601b7"
DREAM_TAC_COMMIT = "14bab51d6862fd07124745c55cd395ea5caa9fd3"
ACT_ARTIFACT_REPOSITORY = "byml/UniVTAC"
ACT_ARTIFACT_REVISION = "172331dbbce95bc04c3e59b22f32dc72ba5561ae"
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
    "docs/reproducibility.md",
    "docs/evidence_levels.md",
    "configs/reporting_spec.example.json",
    "requirements/README.md",
    "requirements/core.lock.txt",
    "requirements/dev.lock.txt",
    "requirements/visualization.lock.txt",
    "scripts/bootstrap_pip.sh",
    "scripts/README.md",
    "scripts/act/README.md",
    "scripts/act/install_official_artifacts.py",
    "scripts/ftp1_policy/install_isaac_client.sh",
    "scripts/ftp1_policy/README.md",
    "examples/act/README.md",
    "examples/act/artifact_manifest.example.json",
    "examples/act/request.json",
    "examples/dream_tac/README.md",
    "examples/dream_tac/artifact_manifest.example.json",
    "examples/n0_twam/README.md",
    "examples/n0_twam/artifact_manifest.example.json",
    "examples/n0_twam/request.json",
    "examples/n0_vtla/README.md",
    "examples/n0_vtla/artifact_manifest.example.json",
    "integrations/install_pinned_repo.sh",
    "integrations/act_artifacts.lock.json",
    "integrations/install_act_runtime.sh",
    "integrations/install_dream_tac.sh",
    "integrations/install_n0_twam.sh",
    "integrations/install_n0_vtla.sh",
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
    assert N0_TWAM_COMMIT in notice
    assert "N0-VTLA" in notice
    assert N0_VTLA_COMMIT in notice
    assert ACT_ARTIFACT_REPOSITORY in notice
    assert ACT_ARTIFACT_REVISION in notice
    assert "dataset card declares MIT" in notice
    assert "integrations/act_artifacts.lock.json" in notice
    assert "Upstream logs and metadata are reference material" in notice
    assert "locally executed RoboTactile results" in notice
    assert "does not currently identify an official public weight release" not in notice


def test_public_model_examples_pass_runtime_structure_gates() -> None:
    """Examples are structurally loadable before real-file/hash validation."""

    act_request = load_live_univtac_request(ROOT / "examples/act/request.json")
    n0_request = load_live_univtac_request(ROOT / "examples/n0_twam/request.json")

    for request in (act_request, n0_request):
        assert dict(request.launcher_args) == production_univtac_launcher_args()
        assert (
            request.upstream_root.resolve()
            == (ROOT / "deployment/sources/UniVTAC").resolve()
        )
        assert request.checkpoint_sha256 == "0" * 64
        assert load_live_univtac_run(request).run_spec.execute_action_steps == (
            request.execute_action_steps
        )

    assert act_request.execute_action_steps == 1
    assert n0_request.execute_action_steps == 24
    assert n0_request.n0_source_commit == N0_TWAM_COMMIT

    act_manifest = load_act_artifact_manifest(
        ROOT / "examples/act/artifact_manifest.example.json"
    )
    n0_manifest = load_n0_twam_artifact_manifest(
        ROOT / "examples/n0_twam/artifact_manifest.example.json"
    )
    dream_tac_manifest = load_dream_tac_artifact_manifest(
        ROOT / "examples/dream_tac/artifact_manifest.example.json"
    )
    vtla_manifest = load_n0_vtla_artifact_manifest(
        ROOT / "examples/n0_vtla/artifact_manifest.example.json"
    )
    assert act_manifest.checkpoint_sha256 == "0" * 64
    assert n0_manifest.checkpoint_sha256 == "0" * 64
    assert n0_manifest.external_commit == N0_TWAM_COMMIT
    assert dream_tac_manifest.external_commit == DREAM_TAC_COMMIT
    assert dream_tac_manifest.use_tactile is True
    assert dream_tac_manifest.casa_inference_contract == (
        "upstream_http_gate_missing_v1"
    )
    assert vtla_manifest.checkpoint_sha256 == "0" * 64
    assert vtla_manifest.external_commit == N0_VTLA_COMMIT


def test_install_scripts_are_syntax_valid_and_contain_no_secret() -> None:
    for relative in REQUIRED:
        if not relative.endswith(".sh"):
            continue
        path = ROOT / relative
        subprocess.run(("bash", "-n", str(path)), check=True)
        content = path.read_text(encoding="utf-8")
        assert re.search(r"(?i)(password|api[_-]?key|token)=", content) is None


def test_ftp1_public_docs_install_only_the_isolated_isaac_client() -> None:
    script = (ROOT / "scripts/ftp1_policy/install_isaac_client.sh").read_text(
        encoding="utf-8"
    )
    guide = (ROOT / "docs/model_integrations.md").read_text(encoding="utf-8")
    operational = (ROOT / "scripts/ftp1_policy/README.md").read_text(encoding="utf-8")

    for document in (guide, operational):
        assert "scripts/ftp1_policy/install_isaac_client.sh" in document
        assert "msgpack==1.1.1" in document
        assert "pyzmq==27.1.0" in document
    assert "runtime/isaac-sim-4.5.0/python.sh" in script
    assert "runtime/n0-twam" not in script
    assert "runtime/ftp1-policy/bin/python" not in script


def test_act_official_release_docs_are_pinned_and_evidence_bounded() -> None:
    paths = (
        "README.md",
        "docs/installation.md",
        "docs/external_dependencies.md",
        "docs/model_integrations.md",
        "scripts/act/README.md",
    )
    documents = {path: (ROOT / path).read_text(encoding="utf-8") for path in paths}
    combined = "\n".join(documents.values())

    for document in documents.values():
        assert ACT_ARTIFACT_REPOSITORY in document
        assert ACT_ARTIFACT_REVISION in document
        assert "scripts/act/install_official_artifacts.py" in document

    for token in (
        "encoder.pth",
        "policy_last.ckpt",
        "dataset_stats.pkl",
        "metadata.json",
        "log.log",
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    ):
        assert token in documents["scripts/act/README.md"]

    assert "upstream_reference_only_no_local_execution" in combined
    assert "integrations/act_artifacts.lock.json" in combined
    assert "ACT-official" in combined
    assert "ACT-train759" in combined
    assert "frozen40" in combined
    assert "Clean/Faulted" in combined
    assert "matched no-touch" in combined
    assert "only the `univtac`" in combined

    obsolete = (
        "No public ACT weight source is registered here",
        "does not currently claim an official public ACT weight release",
        "does not currently claim that an official public ACT checkpoint bundle",
        "does not claim a public official ACT weight bundle",
    )
    assert all(token not in combined for token in obsolete)
    assert (
        re.search(
            r"install_official_artifacts\.py\s+(?:\\\n\s*)?--root(?:\s|$)",
            combined,
        )
        is None
    )


def test_act_official_artifact_installer_help_matches_public_docs() -> None:
    installer = ROOT / "scripts/act/install_official_artifacts.py"
    completed = subprocess.run(
        (sys.executable, str(installer), "--help"),
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--artifact-root",
        "--task",
        "--profile",
        "--lock-path",
        "--include-reference",
        "--dry-run",
        "--plan",
        "--receipt",
    ):
        assert option in completed.stdout


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
        "docs/reproducibility.md",
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
        "live-univtac-paired-run",
        "infrastructure_launch_only",
        "unqualified_live_univtac_execution_v1",
        "unqualified_paired_live_univtac_execution_v1",
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
        and path.relative_to(ROOT).as_posix() != "tests/test_public_release_docs.py"
        and path.suffix in suffixes
        and not (
            {
                ".git",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                "data",
                "deployment",
                "dist",
                "outputs",
                "temp",
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

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_n0_launcher_fails_closed_on_small_fast_path_gpus() -> None:
    launcher = (ROOT / "scripts/n0_twam/serve_univtac.sh").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/n0_twam/serve_official.py").read_text(encoding="utf-8")

    assert 'GPU_MEMORY" -lt 40000' in launcher
    assert "--query-gpu=memory.total" in launcher
    assert "--debug-offload" in launcher
    assert "official fast serving requires >=40000 MiB" in launcher
    assert 'TWAM_CONFIGS["multitask_server"]' in wrapper
    assert "config.enable_offload = True" in wrapper
    assert "from n0_twam.n0_twam_server import" in wrapper


def test_n0_launcher_supports_official_single_gpu_topology() -> None:
    launcher = (ROOT / "scripts/n0_twam/serve_univtac.sh").read_text(encoding="utf-8")

    assert '[ "$NPROC" -ge 1 ]' in launcher
    assert '[ "$NPROC" -eq 1 ]' in launcher
    assert "export RANK=0" in launcher
    assert "export LOCAL_RANK=0" in launcher
    assert "export WORLD_SIZE=1" in launcher
    assert 'exec "$RUNTIME_ROOT/bin/python"' in launcher
    assert 'exec "$RUNTIME_ROOT/bin/torchrun"' in launcher
    assert "official N0 serving requires at least two GPUs" not in launcher
    assert "serve_task_id(sys.argv[1])" in launcher
    assert 'export TWAM_SERVE_TASK="$SERVE_TASK"' in launcher
    assert "univtac_${TASK}_rot6d_current" not in launcher


def test_n0_launcher_emits_rank_zero_source_bound_attestation() -> None:
    launcher = (ROOT / "scripts/n0_twam/serve_univtac.sh").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/n0_twam/serve_official.py").read_text(encoding="utf-8")

    for argument in (
        "--session-id",
        "--attestation",
        "--qualification",
        "--integration-config",
        "--n0-source-root",
    ):
        assert argument in launcher or argument in wrapper
    assert "source-bound paper serving forbids --debug-offload" in launcher
    assert "if source_bound and rank == 0:" in wrapper
    assert "build_n0_server_runtime_attestation" in wrapper
    assert "build_n0_server_runtime_attestation(" in wrapper
    assert wrapper.rindex("build_n0_server_runtime_attestation(") < wrapper.index(
        "init_logger()"
    )


def test_n0_client_installer_is_source_manifest_bound() -> None:
    installer = (ROOT / "scripts/n0_twam/install_robotactile_client.sh").read_text(
        encoding="utf-8"
    )

    assert "source_manifest_sha256=" in installer
    assert "update_source_manifest.py" in installer
    assert "--force-reinstall" in installer
    assert "--no-deps" in installer
    assert "n0_twam_runtime_install.json" in installer
    assert "system_python_modified=false" in installer
    assert "system_cuda_modified=false" in installer


def test_n0_runtime_fallback_installs_pinned_bootstrap_without_self_upgrade() -> None:
    installer = (ROOT / "scripts/n0_twam/install_official_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert 'PIP_VERSION="25.1.1"' in installer
    assert 'SETUPTOOLS_VERSION="80.9.0"' in installer
    assert 'WHEEL_VERSION="0.45.1"' in installer
    assert '"pip=$PIP_VERSION"' in installer
    assert '"setuptools=$SETUPTOOLS_VERSION"' in installer
    assert '"wheel=$WHEEL_VERSION"' in installer
    assert 'if [ "$ENVIRONMENT_BACKEND" = "venv" ]; then' in installer
    assert "SYSTEM_PYTHON_VERSION=" in installer
    assert '[ "$SYSTEM_PYTHON_VERSION" = "3.11" ]' in installer
    assert (
        'MAMBA_EXTRACT_THREADS="${ROBOTACTILE_MAMBA_EXTRACT_THREADS:-4}"' in installer
    )
    assert "--mamba-extract-threads" in installer
    assert 'MAMBA_EXTRACT_THREADS="$MAMBA_EXTRACT_THREADS"' in installer
    assert '[ "$MAMBA_EXTRACT_THREADS" -le 32 ]' in installer
    assert '"mamba_extract_threads=$MAMBA_EXTRACT_THREADS"' in installer
    assert '"bootstrap_identity=$BOOTSTRAP_IDENTITY"' in installer

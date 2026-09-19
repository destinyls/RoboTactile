"""Host runtime selection preserves deployment identity and launcher boundaries."""

import os
from pathlib import Path

import pytest

from scripts.retrained_evaluation.campaign import (
    build_process_environments,
    resolve_isaac_python,
)


def test_default_runtime(tmp_path: Path) -> None:
    root = tmp_path / "repo/deployment-sm120"
    launcher = root / "runtime/isaac-sim-4.5.0/python.sh"
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    assert resolve_isaac_python({"deployment_root": str(root)}) == launcher


def test_host_override_allows_interpreter_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repo/deployment-sm120"
    launcher = root.parent / "deployment/runtime/isaac-sim-4.5.0/python.sh"
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    interpreter = tmp_path / "external-python"
    interpreter.touch()
    (launcher.parent / "python").symlink_to(interpreter)
    assert (
        resolve_isaac_python(
            {"deployment_root": str(root), "isaac_python": str(launcher)}
        )
        == launcher
    )


@pytest.mark.parametrize("symlink", [False, True])
def test_external_launcher_rejected(tmp_path: Path, symlink: bool) -> None:
    root = tmp_path / "repo/deployment-sm120"
    root.mkdir(parents=True)
    external = tmp_path / "external-python.sh"
    external.touch()
    launcher = external
    if symlink:
        launcher = root / "python.sh"
        launcher.symlink_to(external)
    with pytest.raises(ValueError, match="within deployment repo"):
        resolve_isaac_python(
            {"deployment_root": str(root), "isaac_python": str(launcher)}
        )


def test_missing_runtime_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Isaac runtime missing"):
        resolve_isaac_python({"deployment_root": str(tmp_path / "deployment")})


def test_relative_runtime_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute paths"):
        resolve_isaac_python(
            {
                "deployment_root": str(tmp_path / "deployment"),
                "isaac_python": "python.sh",
            }
        )


def test_dream_policy_packages_are_not_exposed_to_isaac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDNN_HOME", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    root = tmp_path / "repo/deployment-sm120"
    code = tmp_path / "code"
    package = tmp_path / "package"
    shared = os.pathsep.join(("dream-python3.11", "ftp-python3.11"))
    server, isaac = build_process_environments(
        {
            "deployment_root": str(root),
            "model": "dream_tac",
            "shared_pythonpath": shared,
            "evaluation": {"reset_time_limit_s": 600},
        },
        code,
        package,
    )

    common_pythonpath = os.pathsep.join((str(package), str(code)))
    assert isaac["PYTHONPATH"] == common_pythonpath
    assert server["PYTHONPATH"] == os.pathsep.join((common_pythonpath, shared))
    assert "CUDNN_HOME" not in isaac
    assert "CUDA_HOME" not in isaac
    assert isaac["ROBOTACTILE_UNIVTAC_RESET_TIME_LIMIT_S"] == "600"
    assert "ROBOTACTILE_UNIVTAC_RESET_TIME_LIMIT_S" not in server
    assert server["CUDNN_HOME"].endswith("nvidia/cudnn")
    assert server["CUDA_HOME"].endswith("runtime/cuda-toolkit-12.8")

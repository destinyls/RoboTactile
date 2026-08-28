"""Behavior tests for the CUDA runfile's protected system-link handling."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

INSTALLER = (
    Path(__file__).parents[1]
    / "scripts"
    / "live_univtac"
    / "install_cuda_toolkit_12_4.sh"
)
TEST_LINK = "runtime/.robotactile-test-only-usr-local-cuda"


def _fake_runfile(root: Path) -> tuple[Path, str]:
    runfile = root / "cuda.run"
    runfile.write_text(
        """#!/bin/sh
set -eu
toolkit_path=""
for argument in "$@"; do
  case "$argument" in --toolkitpath=*) toolkit_path="${argument#*=}" ;; esac
done
test -n "$toolkit_path"
mkdir -p "$toolkit_path/bin"
cat > "$toolkit_path/bin/nvcc" <<'NVCC'
#!/bin/sh
printf 'Cuda compilation tools, release 12.4, V12.4.131\n'
NVCC
chmod 755 "$toolkit_path/bin/nvcc"
printf 'ran\n' > "$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-system-link-runfile"
target="${FAKE_SYSTEM_LINK_TARGET:-$toolkit_path/}"
ln -s "$target" "$ROBOTACTILE_TEST_SYSTEM_CUDA_LINK_PATH"
""",
        encoding="utf-8",
    )
    return runfile, hashlib.sha256(runfile.read_bytes()).hexdigest()


def _run(
    deploy_root: Path,
    runfile: Path,
    digest: str,
    *,
    foreign_target: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ROBOTACTILE_TEST_SYSTEM_CUDA_LINK_SANDBOX"] = "1"
    if foreign_target is not None:
        env["FAKE_SYSTEM_LINK_TARGET"] = str(foreign_target)
    return subprocess.run(
        [
            "bash",
            str(INSTALLER),
            "--root",
            str(deploy_root),
            "--runfile",
            str(runfile),
            "--sha256",
            digest,
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_exact_runfile_system_link_is_removed_and_recorded() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        runfile, digest = _fake_runfile(temporary)

        result = _run(deploy_root, runfile, digest)
        receipt = json.loads(
            (deploy_root / "artifacts/deployment/cuda_toolkit_install.json").read_text()
        )

        assert result.returncode == 0, result.stderr
        assert not (deploy_root / TEST_LINK).exists()
        assert receipt["system_cuda_link_preexisting"] == "false"
        assert receipt["system_cuda_link_observed_after_run"] == "true"
        assert receipt["system_cuda_link_restoration"] == (
            "removed_exact_install_symlink"
        )
        assert receipt["system_cuda_link_test_override"] == "true"
        assert receipt["vendor_log_path"] == "/var/log/cuda-installer.log"
        assert receipt["vendor_log_managed"] == "false"
        (deploy_root / TEST_LINK).symlink_to(temporary / "appeared-later")
        repeated = _run(deploy_root, runfile, digest)
        assert repeated.returncode != 0
        assert "protected path exists" in repeated.stderr


def test_preexisting_system_link_blocks_runfile_without_removal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        test_link = deploy_root / TEST_LINK
        test_link.parent.mkdir(parents=True)
        test_link.symlink_to(temporary / "preexisting")
        runfile, digest = _fake_runfile(temporary)

        result = _run(deploy_root, runfile, digest)

        assert result.returncode != 0
        assert "protected path exists" in result.stderr
        assert test_link.is_symlink()
        assert not (deploy_root / "artifacts/fake-system-link-runfile").exists()


def test_foreign_runfile_system_link_is_retained_and_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        foreign_target = temporary / "foreign-cuda"
        runfile, digest = _fake_runfile(temporary)

        result = _run(
            deploy_root,
            runfile,
            digest,
            foreign_target=foreign_target,
        )
        test_link = deploy_root / TEST_LINK

        assert result.returncode != 0
        assert "refusing to remove it" in result.stderr
        assert test_link.is_symlink()
        assert os.readlink(test_link) == str(foreign_target)
        assert not (
            deploy_root / "artifacts/deployment/cuda_toolkit_install.json"
        ).exists()

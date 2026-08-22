"""Behavior tests for the reproducible live UniVTAC deployment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts" / "live_univtac"
ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
CUROBO_COMMIT = "0a50de1ba72db304195d59d9d0b1ed269696047f"


def _executable_zip_member(name: str, content: str) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(name)
    member.create_system = 3
    member.external_attr = (stat.S_IFREG | 0o755) << 16
    return member


def _write_fake_isaac_archive(root: Path) -> tuple[Path, str]:
    archive = root / "isaac-sim-standalone-4.5.0-linux-x86_64.zip"
    prefix = "isaac-sim-4.5.0"
    post_install = """#!/usr/bin/env bash
set -eu
count_file="$PWD/post-install-count"
count=0
test ! -f "$count_file" || count="$(cat "$count_file")"
printf '%s\n' "$((count + 1))" > "$count_file"
printf 'isolated\n' > "$HOME/touched-by-post-install"
"""
    python_sh = """#!/usr/bin/env bash
set -eu
if [[ "$*" == *hello_world.py* ]]; then
  count_file="$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-smoke-count"
  count=0
  test ! -f "$count_file" || count="$(cat "$count_file")"
  printf '%s\n' "$((count + 1))" > "$count_file"
  {
    printf 'HOME=%s\n' "$HOME"
    printf 'XDG_CACHE_HOME=%s\n' "$XDG_CACHE_HOME"
    printf 'TMPDIR=%s\n' "$TMPDIR"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'ARGS=%s\n' "$*"
  } > "$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-smoke-env"
  test "${FAKE_SMOKE_FAIL:-0}" = 0 || exit 17
else
  printf '%s\n' "$*" >> "$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-python-calls"
fi
"""
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            _executable_zip_member(f"{prefix}/post_install.sh", post_install),
            post_install,
        )
        bundle.writestr(
            _executable_zip_member(f"{prefix}/python.sh", python_sh),
            python_sh,
        )
        bundle.writestr(
            f"{prefix}/standalone_examples/api/isaacsim.simulation_app/hello_world.py",
            "print('fake hello world')\n",
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def _run(
    script: str,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["bash", str(SCRIPTS / script), *arguments]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _isolated_env(root: Path) -> tuple[dict[str, str], Path]:
    user_home = root / "user-home"
    user_home.mkdir()
    (user_home / ".bashrc").write_text("sentinel\n", encoding="utf-8")
    env = dict(os.environ)
    env["HOME"] = str(user_home)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env, user_home


def _install_fake_isaac(
    temporary: Path,
    deploy_root: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    archive = temporary / "isaac-sim-standalone-4.5.0-linux-x86_64.zip"
    if archive.exists():
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    else:
        archive, digest = _write_fake_isaac_archive(temporary)
    return _run(
        "install_isaac_sim_4_5.sh",
        "--root",
        str(deploy_root),
        "--archive",
        str(archive),
        "--sha256",
        digest,
        env=env,
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_isaac_installer_isolated_atomic_and_idempotent() -> None:
    """Catches post-install reruns, user-HOME writes, and missing provenance."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, user_home = _isolated_env(temporary)

        first = _install_fake_isaac(temporary, deploy_root, env)
        second = _install_fake_isaac(temporary, deploy_root, env)

        install = deploy_root / "runtime" / "isaac-sim-4.5.0"
        receipt_path = (
            deploy_root / "artifacts" / "deployment" / "isaac_sim_install.json"
        )
        receipt = _read_json(receipt_path)
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert (install / "post-install-count").read_text().strip() == "1"
        assert not (user_home / "touched-by-post-install").exists()
        assert (user_home / ".bashrc").read_text() == "sentinel\n"
        assert (deploy_root / "runtime/home/touched-by-post-install").is_file()
        assert receipt["component"] == "isaac_sim"
        assert receipt["version"] == "4.5.0"
        assert receipt["status"] == "installed"
        assert (
            receipt["source_sha256"]
            == hashlib.sha256(
                (temporary / "isaac-sim-standalone-4.5.0-linux-x86_64.zip").read_bytes()
            ).hexdigest()
        )
        assert Path(str(receipt["log_path"])).name.startswith("isaac-sim-post-install-")


def test_isaac_installer_rejects_wrong_hash_and_foreign_destination() -> None:
    """Catches unchecked downloads and accidental replacement of a runtime."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        env, _ = _isolated_env(temporary)
        archive, digest = _write_fake_isaac_archive(temporary)
        bad_root = temporary / "bad-hash"
        bad = _run(
            "install_isaac_sim_4_5.sh",
            "--root",
            str(bad_root),
            "--archive",
            str(archive),
            "--sha256",
            "0" * 64,
            env=env,
        )
        occupied_root = temporary / "occupied"
        occupied = occupied_root / "runtime/isaac-sim-4.5.0"
        occupied.mkdir(parents=True)
        sentinel = occupied / "sentinel"
        sentinel.write_text("keep\n", encoding="utf-8")
        conflict = _run(
            "install_isaac_sim_4_5.sh",
            "--root",
            str(occupied_root),
            "--archive",
            str(archive),
            "--sha256",
            digest,
            env=env,
        )

        assert bad.returncode != 0
        assert "SHA-256 mismatch" in bad.stderr
        assert not (bad_root / "runtime/isaac-sim-4.5.0").exists()
        assert conflict.returncode != 0
        assert "refusing to overwrite" in conflict.stderr
        assert sentinel.read_text() == "keep\n"


def _write_fake_nvidia_smi(bin_dir: Path) -> None:
    bin_dir.mkdir()
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\nprintf 'NVIDIA GeForce RTX 3090, 550.54.14\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def test_headless_smoke_records_gpu_and_explicit_runtime_paths() -> None:
    """Catches false smoke receipts and cache writes outside the deployment root."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, user_home = _isolated_env(temporary)
        assert _install_fake_isaac(temporary, deploy_root, env).returncode == 0
        fake_bin = temporary / "bin"
        _write_fake_nvidia_smi(fake_bin)
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

        first = _run(
            "smoke_isaac_sim_4_5.sh",
            "--root",
            str(deploy_root),
            "--gpu",
            "3",
            "--run-id",
            "pytest-gpu3",
            env=env,
        )
        second = _run(
            "smoke_isaac_sim_4_5.sh",
            "--root",
            str(deploy_root),
            "--gpu",
            "3",
            "--run-id",
            "pytest-gpu3",
            env=env,
        )

        receipt = _read_json(
            deploy_root / "artifacts/deployment/isaac_sim_smoke_pytest-gpu3.json"
        )
        runtime_env = (deploy_root / "artifacts/fake-smoke-env").read_text(
            encoding="utf-8"
        )
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert (deploy_root / "artifacts/fake-smoke-count").read_text().strip() == "1"
        assert f"HOME={deploy_root}/runtime/home" in runtime_env
        assert f"XDG_CACHE_HOME={deploy_root}/runtime/cache" in runtime_env
        assert f"TMPDIR={deploy_root}/runtime/tmp" in runtime_env
        assert "CUDA_VISIBLE_DEVICES=3" in runtime_env
        assert "hello_world.py --headless" in runtime_env
        assert receipt["status"] == "passed"
        assert receipt["gpu_index"] == "3"
        assert "RTX 3090" in str(receipt["gpu_identity"])
        assert not (user_home / "touched-by-smoke").exists()


def _write_source_fixture(path: Path, installer_name: str) -> None:
    path.mkdir()
    installer = path / installer_name
    installer.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-source-installs"
""",
        encoding="utf-8",
    )
    installer.chmod(0o755)


def _write_fake_git(bin_dir: Path) -> None:
    executable = bin_dir / "git"
    executable.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
if [ "$1" = clone ]; then
  destination="${@: -1}"
  case "$*" in
    *IsaacLab*) fixture="$FAKE_ISAACLAB_SOURCE" ;;
    *curobo*) fixture="$FAKE_CUROBO_SOURCE" ;;
    *) exit 91 ;;
  esac
  mkdir -p "$destination"
  cp -R "$fixture"/. "$destination"/
  mkdir -p "$destination/.git"
elif [ "$1" = -C ] && [ "$3" = checkout ]; then
  printf '%s\n' "$5" > "$2/.fake-head"
elif [ "$1" = -C ] && [ "$3" = rev-parse ]; then
  cat "$2/.fake-head"
elif [ "$1" = -C ] && [ "$3" = status ]; then
  :
else
  exit 92
fi
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def test_pinned_isaaclab_and_curobo_installers_record_exact_commits() -> None:
    """Catches mutable source refs, wrong installers, and repeated installs."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, user_home = _isolated_env(temporary)
        assert _install_fake_isaac(temporary, deploy_root, env).returncode == 0
        fake_bin = temporary / "bin"
        fake_bin.mkdir()
        _write_fake_git(fake_bin)
        isaaclab_fixture = temporary / "IsaacLab-fixture"
        curobo_fixture = temporary / "curobo-fixture"
        _write_source_fixture(isaaclab_fixture, "isaaclab.sh")
        _write_source_fixture(curobo_fixture, "setup.py")
        env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                "FAKE_GIT_LOG": str(temporary / "git.log"),
                "FAKE_ISAACLAB_SOURCE": str(isaaclab_fixture),
                "FAKE_CUROBO_SOURCE": str(curobo_fixture),
            }
        )

        lab_first = _run(
            "install_isaaclab_v2_1_1.sh", "--root", str(deploy_root), env=env
        )
        lab_second = _run(
            "install_isaaclab_v2_1_1.sh", "--root", str(deploy_root), env=env
        )
        curobo_first = _run(
            "install_curobo_v0_7_7.sh", "--root", str(deploy_root), env=env
        )
        curobo_second = _run(
            "install_curobo_v0_7_7.sh", "--root", str(deploy_root), env=env
        )

        lab_receipt = _read_json(
            deploy_root / "artifacts/deployment/isaaclab_install.json"
        )
        curobo_receipt = _read_json(
            deploy_root / "artifacts/deployment/curobo_install.json"
        )
        installs = (
            (deploy_root / "artifacts/fake-source-installs")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        python_calls = (deploy_root / "artifacts/fake-python-calls").read_text(
            encoding="utf-8"
        )
        assert lab_first.returncode == 0, lab_first.stderr
        assert lab_second.returncode == 0, lab_second.stderr
        assert curobo_first.returncode == 0, curobo_first.stderr
        assert curobo_second.returncode == 0, curobo_second.stderr
        assert lab_receipt["source_commit"] == ISAACLAB_COMMIT
        assert lab_receipt["version"] == "v2.1.1"
        assert curobo_receipt["source_commit"] == CUROBO_COMMIT
        assert curobo_receipt["version"] == "v0.7.7"
        assert installs == ["--install"]
        assert "flatdict==4.0.1" in python_calls
        assert "warp-lang==1.0.0" in python_calls
        assert "pip install -e" in python_calls
        assert (
            deploy_root / "sources/IsaacLab/_isaac_sim"
        ).readlink() == deploy_root / "runtime/isaac-sim-4.5.0"
        assert (user_home / ".bashrc").read_text() == "sentinel\n"

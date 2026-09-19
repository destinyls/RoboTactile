"""Behavior tests for the reproducible live UniVTAC deployment scripts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import cast

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts" / "live_univtac"
ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
CUROBO_COMMIT = "0a50de1ba72db304195d59d9d0b1ed269696047f"
UNIVTAC_COMMIT = "05bcd3edb92237107efa40105292a24f1a9fd761"
VCPKG_COMMIT = "ce613c41372b23b1f51333815feb3edd87ef8a8b"
VCPKG_BASELINE = "b2cb0da531c2f1f740045bfe7c4dac59f0b2b69c"


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
elif [[ "$*" == *torch.__version__* ]]; then
  printf '2.7.0+cu128|12.8\n'
elif [[ "$*" == *isaaclab_package_identity* ]]; then
  printf '0.41.3|0.2.2|1.0.7|0.1.8|0.10.36\n'
elif [[ "$*" == *ROBOTACTILE_CUDA_ARCH* ]]; then
  printf 'ROBOTACTILE_CUDA_ARCH=86\n'
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
        bundle.writestr(
            f"{prefix}/kit/python/lib/python3.10/site-packages/torch/lib/libc10.so",
            "fake libc10\n",
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
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _write_fake_cuda_runfile(
    root: Path,
    *,
    toolkit_version: str = "12.4.1",
    nvcc_release: str = "12.4",
    nvcc_build: str = "V12.4.131",
) -> tuple[Path, str]:
    runfile = root / f"cuda_{toolkit_version}_linux.run"
    runfile.write_text(
        """#!/bin/sh
set -eu
artifacts="$ROBOTACTILE_DEPLOY_ROOT/artifacts"
count_file="$artifacts/fake-cuda-runfile-count"
count=0
test ! -f "$count_file" || count="$(cat "$count_file")"
printf '%s\n' "$((count + 1))" > "$count_file"
printf '%s\n' "$@" > "$artifacts/fake-cuda-runfile-args"
printf 'HOME=%s\nTMPDIR=%s\n' "$HOME" "$TMPDIR" > "$artifacts/fake-cuda-runfile-env"
toolkit_path=""
for argument in "$@"; do
  case "$argument" in --toolkitpath=*) toolkit_path="${argument#*=}" ;; esac
done
test -n "$toolkit_path"
mkdir -p "$toolkit_path/bin"
if [ "${FAKE_CUDA_RUNFILE_FAIL:-0}" = 1 ]; then
  printf 'partial\n' > "$toolkit_path/partial-sentinel"
  exit 17
fi
cat > "$toolkit_path/bin/nvcc" <<'NVCC'
#!/bin/sh
printf 'Cuda compilation tools, release NVCC_RELEASE, NVCC_BUILD\n'
NVCC
chmod 755 "$toolkit_path/bin/nvcc"
""".replace("NVCC_RELEASE", nvcc_release).replace("NVCC_BUILD", nvcc_build),
        encoding="utf-8",
    )
    return runfile, hashlib.sha256(runfile.read_bytes()).hexdigest()


def _install_fake_cuda(
    deploy_root: Path,
    runfile: Path,
    digest: str,
    env: dict[str, str],
    *,
    script: str = "install_cuda_toolkit_12_4.sh",
) -> subprocess.CompletedProcess[str]:
    return _run(
        script,
        "--root",
        str(deploy_root),
        "--runfile",
        str(runfile),
        "--sha256",
        digest,
        env=env,
    )


def test_cuda_runfile_installer_is_toolkit_only_isolated_and_idempotent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, user_home = _isolated_env(temporary)
        runfile, digest = _write_fake_cuda_runfile(temporary)

        first = _install_fake_cuda(deploy_root, runfile, digest, env)
        second = _install_fake_cuda(deploy_root, runfile, digest, env)
        receipt = _read_json(
            deploy_root / "artifacts/deployment/cuda_toolkit_install.json"
        )

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert (deploy_root / "artifacts/fake-cuda-runfile-count").read_text(
            encoding="utf-8"
        ).strip() == "1"
        assert (deploy_root / "artifacts/fake-cuda-runfile-args").read_text(
            encoding="utf-8"
        ).splitlines() == [
            "--silent",
            "--toolkit",
            f"--toolkitpath={deploy_root.resolve()}/runtime/cuda-toolkit-12.4",
            "--no-man-page",
            f"--tmpdir={deploy_root.resolve()}/runtime/tmp",
        ]
        runtime_env = (deploy_root / "artifacts/fake-cuda-runfile-env").read_text()
        assert f"HOME={deploy_root.resolve()}/runtime/home" in runtime_env
        assert f"TMPDIR={deploy_root.resolve()}/runtime/tmp" in runtime_env
        assert (user_home / ".bashrc").read_text() == "sentinel\n"
        assert receipt["component"] == "cuda_toolkit"
        assert receipt["version"] == "12.4.1"
        assert receipt["source_sha256"] == digest
        assert receipt["toolkit_path"] == str(
            deploy_root.resolve() / "runtime/cuda-toolkit-12.4"
        )
        assert receipt["nvcc_identity"] == "12.4|V12.4.131"
        assert receipt["driver_install_requested"] == "false"
        assert receipt["system_prefix_install_requested"] == "false"
        marker = deploy_root / (
            "runtime/.cuda-toolkit-12.4.robotactile-install-incomplete.json"
        )
        assert not marker.exists()


def test_cuda_12_8_runfile_installer_records_blackwell_capable_toolkit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, _ = _isolated_env(temporary)
        runfile, digest = _write_fake_cuda_runfile(
            temporary,
            toolkit_version="12.8.1",
            nvcc_release="12.8",
            nvcc_build="V12.8.93",
        )

        result = _install_fake_cuda(
            deploy_root,
            runfile,
            digest,
            env,
            script="install_cuda_toolkit_12_8.sh",
        )
        receipt = _read_json(
            deploy_root / "artifacts/deployment/cuda_toolkit_install.json"
        )

        assert result.returncode == 0, result.stderr
        assert receipt["version"] == "12.8.1"
        assert receipt["install_path"] == "runtime/cuda-toolkit-12.8"
        assert receipt["nvcc_identity"] == "12.8|V12.8.93"
        assert (deploy_root / "runtime/cuda-toolkit-12.8/bin/nvcc").is_file()


def test_cuda_runfile_installer_rejects_hash_mismatch_and_foreign_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        env, _ = _isolated_env(temporary)
        runfile, digest = _write_fake_cuda_runfile(temporary)
        bad_root = temporary / "bad-hash"
        bad = _install_fake_cuda(bad_root, runfile, "0" * 64, env)

        occupied_root = temporary / "occupied"
        sentinel = occupied_root / "runtime/cuda-toolkit-12.4/sentinel"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("keep\n", encoding="utf-8")
        conflict = _install_fake_cuda(occupied_root, runfile, digest, env)

        assert bad.returncode != 0
        assert "SHA-256 mismatch" in bad.stderr
        assert not (bad_root / "runtime/cuda-toolkit-12.4").exists()
        assert conflict.returncode != 0
        assert "refusing to overwrite foreign CUDA Toolkit" in conflict.stderr
        assert sentinel.read_text() == "keep\n"


def test_cuda_runfile_installer_recovers_only_its_incomplete_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, _ = _isolated_env(temporary)
        runfile, digest = _write_fake_cuda_runfile(temporary)
        env["FAKE_CUDA_RUNFILE_FAIL"] = "1"
        failed = _install_fake_cuda(deploy_root, runfile, digest, env)
        marker = (
            deploy_root
            / "runtime/.cuda-toolkit-12.4.robotactile-install-incomplete.json"
        )

        assert failed.returncode != 0
        assert marker.is_file()
        assert (deploy_root / "runtime/cuda-toolkit-12.4/partial-sentinel").is_file()

        env["FAKE_CUDA_RUNFILE_FAIL"] = "0"
        recovered = _install_fake_cuda(deploy_root, runfile, digest, env)
        assert recovered.returncode == 0, recovered.stderr
        assert not marker.exists()
        assert not (deploy_root / "runtime/cuda-toolkit-12.4/partial-sentinel").exists()
        assert (deploy_root / "artifacts/fake-cuda-runfile-count").read_text(
            encoding="utf-8"
        ).strip() == "2"


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
        f"""#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-source-installs"
{{
  printf 'CONDA_PREFIX=%s\n' "${{CONDA_PREFIX-unset}}"
  printf 'CONDA_DEFAULT_ENV=%s\n' "${{CONDA_DEFAULT_ENV-unset}}"
  printf 'PIP_CONSTRAINT=%s\n' "${{PIP_CONSTRAINT-unset}}"
  printf 'PIP_DEFAULT_TIMEOUT=%s\n' "${{PIP_DEFAULT_TIMEOUT-unset}}"
  printf 'PIP_RETRIES=%s\n' "${{PIP_RETRIES-unset}}"
  printf 'PIP_FIND_LINKS=%s\n' "${{PIP_FIND_LINKS-unset}}"
  printf 'TERM=%s\n' "${{TERM-unset}}"
}} > "$ROBOTACTILE_DEPLOY_ROOT/artifacts/fake-source-env-{installer_name}"
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


def _write_json_fixture(path: Path, document: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _prepare_cuda_installer_fixture(
    temporary: Path,
    *,
    capability: str,
    nvcc_release: str = "12.4",
) -> tuple[Path, Path, dict[str, str]]:
    deploy_root = temporary / "deploy"
    env, _ = _isolated_env(temporary)
    isaac_python = deploy_root / "runtime/isaac-sim-4.5.0/python.sh"
    isaac_python.parent.mkdir(parents=True)
    isaac_python.write_text(
        f"""#!/usr/bin/env bash
set -eu
printf '%s\n' "${{CUDA_VISIBLE_DEVICES:-unset}}" >> "$ROBOTACTILE_DEPLOY_ROOT/artifacts/cuda-probe-gpus"
case "$*" in
  *ROBOTACTILE_CUDA_ARCH*) printf 'ROBOTACTILE_CUDA_ARCH={capability}\n' ;;
  *) exit 93 ;;
esac
""",
        encoding="utf-8",
    )
    isaac_python.chmod(0o755)

    cuda_root = deploy_root / f"runtime/cuda-toolkit-{nvcc_release}"
    nvcc_build = "V12.8.93" if nvcc_release == "12.8" else f"V{nvcc_release}.131"
    nvcc = cuda_root / "bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'Cuda compilation tools, release {nvcc_release}, "
        f"{nvcc_build}\\n'\n",
        encoding="utf-8",
    )
    nvcc.chmod(0o755)

    source = deploy_root / "sources/UniVTAC"
    (source / ".git").mkdir(parents=True)
    (source / ".fake-head").write_text(f"{UNIVTAC_COMMIT}\n", encoding="utf-8")
    for setup_path in (
        "third_party/TacEx/source/tacex/setup.py",
        "third_party/TacEx/source/tacex_uipc/setup.py",
    ):
        target = source / setup_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")

    fake_bin = temporary / "bin"
    fake_bin.mkdir()
    _write_fake_git(fake_bin)
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_GIT_LOG": str(temporary / "git.log"),
        }
    )
    major = capability[:-1]
    cuda_fields = {
        "cuda_toolkit_path": str(cuda_root.resolve()),
        "cuda_toolkit_version": nvcc_release,
        "cuda_nvcc_identity": f"{nvcc_release}|{nvcc_build}",
        "cuda_architecture": f"sm_{capability}",
        "cuda_compute_capability": f"{major}.{capability[-1]}",
    }
    _write_json_fixture(
        deploy_root / "artifacts/deployment/isaaclab_install.json",
        {
            "component": "isaaclab",
            "version": "v2.1.1",
            "status": "installed",
            "torch_identity": "2.7.0+cu128|12.8",
        },
    )
    _write_json_fixture(
        deploy_root / "sources/UniVTAC.robotactile-install.json",
        {"integration_id": "univtac", "commit_sha": UNIVTAC_COMMIT},
    )
    _write_json_fixture(
        deploy_root / "artifacts/deployment/tacex_install.json",
        {
            "component": "tacex",
            "status": "installed",
            "univtac_source_commit": UNIVTAC_COMMIT,
            "torch_identity": "2.7.0+cu128|12.8",
            **cuda_fields,
        },
    )
    _write_json_fixture(
        deploy_root / "artifacts/deployment/tacex_uipc_install.json",
        {
            "component": "tacex_uipc",
            "version": "0.1.0",
            "status": "installed",
            "univtac_source_commit": UNIVTAC_COMMIT,
            "vcpkg_source_commit": VCPKG_COMMIT,
            "vcpkg_manifest_baseline": VCPKG_BASELINE,
            "cpptrace_overlay_version": "0.8.3",
            "tinygltf_overlay_version": "2.9.3",
            "tinygltf_archive_sha512": (
                "6dbcff3ea602d0aa45ddd87a87d32ab5ab5453901891dbccfbc660746fe11c5b"
                "d814d6f74707244351dd6326e17f6d9ad7c384417db126122cc4a2cba20b205c"
            ),
            "micromamba_identity": (
                "2.9.0|8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd"
            ),
            **cuda_fields,
        },
    )
    return deploy_root, cuda_root, env


def test_pinned_isaaclab_and_curobo_installers_record_exact_commits() -> None:
    """Catches mutable source refs, wrong installers, and repeated installs."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root = temporary / "deploy"
        env, user_home = _isolated_env(temporary)
        assert _install_fake_isaac(temporary, deploy_root, env).returncode == 0
        cuda_runfile, cuda_digest = _write_fake_cuda_runfile(temporary)
        assert (
            _install_fake_cuda(deploy_root, cuda_runfile, cuda_digest, env).returncode
            == 0
        )
        fake_bin = temporary / "bin"
        fake_bin.mkdir()
        _write_fake_git(fake_bin)
        isaaclab_fixture = temporary / "IsaacLab-fixture"
        curobo_fixture = temporary / "curobo-fixture"
        _write_source_fixture(isaaclab_fixture, "isaaclab.sh")
        _write_source_fixture(curobo_fixture, "setup.py")
        wheelhouse = temporary / "isaaclab-wheelhouse"
        wheelhouse.mkdir()
        env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                "FAKE_GIT_LOG": str(temporary / "git.log"),
                "FAKE_ISAACLAB_SOURCE": str(isaaclab_fixture),
                "FAKE_CUROBO_SOURCE": str(curobo_fixture),
                "CONDA_PREFIX": "/sentinel/conda-base",
                "CONDA_DEFAULT_ENV": "base",
                "CONDA_PROMPT_MODIFIER": "(base) ",
                "TERM": "dumb",
                "PIP_FIND_LINKS": str(wheelhouse),
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
        source_env = (deploy_root / "artifacts/fake-source-env-isaaclab.sh").read_text(
            encoding="utf-8"
        )
        assert lab_first.returncode == 0, lab_first.stderr
        assert lab_second.returncode == 0, lab_second.stderr
        assert curobo_first.returncode == 0, curobo_first.stderr
        assert curobo_second.returncode == 0, curobo_second.stderr
        assert lab_receipt["source_commit"] == ISAACLAB_COMMIT
        assert lab_receipt["version"] == "v2.1.1"
        assert lab_receipt["torch_identity"] == "2.7.0+cu128|12.8"
        assert lab_receipt["package_identity"] == ("0.41.3|0.2.2|1.0.7|0.1.8|0.10.36")
        assert lab_receipt["dependency_constraints_path"] == (
            "requirements/isaaclab-v2.1.1-python310.lock.txt"
        )
        constraint_path = (
            Path(__file__).parents[1]
            / "requirements/isaaclab-v2.1.1-python310.lock.txt"
        )
        constraint_text = constraint_path.read_text(encoding="utf-8")
        assert "contourpy==1.3.2" in constraint_text
        assert "fonttools==4.63.0" in constraint_text
        assert "tomli==2.4.1" in constraint_text
        assert "warp-lang==1.0.0" in constraint_text
        assert (
            lab_receipt["dependency_constraints_sha256"]
            == hashlib.sha256(constraint_path.read_bytes()).hexdigest()
        )
        assert lab_receipt["pip_default_timeout_seconds"] == "600"
        assert lab_receipt["pip_retries"] == "10"
        assert lab_receipt["pip_find_links"] == str(wheelhouse)
        assert lab_receipt["wheelhouse_preinstalled"] == "true"
        assert curobo_receipt["source_commit"] == CUROBO_COMMIT
        assert curobo_receipt["version"] == "v0.7.7"
        assert curobo_receipt["cuda_toolkit_version"] == "12.4"
        assert curobo_receipt["cuda_architecture"] == "sm_86"
        assert curobo_receipt["native_extensions_reused"] == "true"
        assert installs == ["--install none"]
        assert "flatdict==4.0.1" in python_calls
        assert "--no-index" in python_calls
        assert f"--find-links {wheelhouse}" in python_calls
        assert f"--requirement {constraint_path}" in python_calls
        assert "pip install -e" not in python_calls
        assert 'm.version("nvidia-curobo") == "0.7.7"' in python_calls
        assert "ROBOTACTILE_EXPECTED_CUROBO_SOURCE" in python_calls
        assert "CONDA_PREFIX=unset" in source_env
        assert "CONDA_DEFAULT_ENV=unset" in source_env
        assert f"PIP_CONSTRAINT={constraint_path}" in source_env
        assert "PIP_DEFAULT_TIMEOUT=600" in source_env
        assert "PIP_RETRIES=10" in source_env
        assert f"PIP_FIND_LINKS={wheelhouse}" in source_env
        assert "TERM=xterm-256color" in source_env
        assert (
            deploy_root / "sources/IsaacLab/_isaac_sim"
        ).readlink() == deploy_root / "runtime/isaac-sim-4.5.0"
        assert (user_home / ".bashrc").read_text() == "sentinel\n"


def test_tacex_installers_auto_detect_a800_and_reject_cuda_drift() -> None:
    """Catches sm_86 carryover, mismatched explicit targets, and CUDA drift."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root, cuda_root, env = _prepare_cuda_installer_fixture(
            temporary, capability="80"
        )
        arguments = (
            "--root",
            str(deploy_root),
            "--cuda-root",
            str(cuda_root),
            "--cuda-architecture",
            "auto",
            "--gpu",
            "3",
        )

        core = _run("install_tacex_univtac.sh", *arguments, env=env)
        uipc = _run("install_tacex_uipc_univtac.sh", *arguments, env=env)
        mismatch = _run(
            "install_tacex_univtac.sh",
            "--root",
            str(deploy_root),
            "--cuda-root",
            str(cuda_root),
            "--cuda-architecture",
            "86",
            "--gpu",
            "3",
            env=env,
        )
        outside_root = temporary / "outside-cuda"
        outside_root.mkdir()
        outside = _run(
            "install_tacex_univtac.sh",
            "--root",
            str(deploy_root),
            "--cuda-root",
            str(outside_root),
            "--gpu",
            "3",
            env=env,
        )

        assert core.returncode == 0, core.stderr
        assert uipc.returncode == 0, uipc.stderr
        assert (deploy_root / "artifacts/cuda-probe-gpus").read_text(
            encoding="utf-8"
        ).splitlines() == ["3", "3", "3"]
        assert mismatch.returncode != 0
        assert "does not match GPU 3 capability 80" in mismatch.stderr
        assert outside.returncode != 0
        assert "must resolve below the deployment root" in outside.stderr

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root, cuda_root, env = _prepare_cuda_installer_fixture(
            temporary, capability="80", nvcc_release="12.5"
        )
        drift = _run(
            "install_tacex_univtac.sh",
            "--root",
            str(deploy_root),
            "--cuda-root",
            str(cuda_root),
            "--gpu",
            "0",
            env=env,
        )

        assert drift.returncode != 0
        assert "CUDA toolkit must be 12.4 or 12.8; detected 12.5" in drift.stderr


def test_native_installers_accept_cuda_12_8_for_sm120_and_reject_12_4() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root, cuda_root, env = _prepare_cuda_installer_fixture(
            temporary, capability="120", nvcc_release="12.8"
        )
        arguments = (
            "--root",
            str(deploy_root),
            "--cuda-root",
            str(cuda_root),
            "--cuda-architecture",
            "120",
        )

        core = _run("install_tacex_univtac.sh", *arguments, env=env)
        uipc = _run("install_tacex_uipc_univtac.sh", *arguments, env=env)

        assert core.returncode == 0, core.stderr
        assert uipc.returncode == 0, uipc.stderr

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root, cuda_root, env = _prepare_cuda_installer_fixture(
            temporary, capability="120", nvcc_release="12.4"
        )
        rejected = _run(
            "install_tacex_univtac.sh",
            "--root",
            str(deploy_root),
            "--cuda-root",
            str(cuda_root),
            env=env,
        )

        assert rejected.returncode != 0
        assert "architecture 120 requires CUDA 12.8" in rejected.stderr


def test_tacex_installer_default_supports_3090_deployment_local_toolkit() -> None:
    """Keeps the no-flag RTX 3090 path without trusting an arbitrary toolkit."""
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        deploy_root, _, env = _prepare_cuda_installer_fixture(
            temporary, capability="86"
        )

        result = _run("install_tacex_univtac.sh", "--root", str(deploy_root), env=env)

        assert result.returncode == 0, result.stderr
        assert "TacEx is already installed" in result.stderr


def _assert_contains(text: str, *needles: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    assert not missing, missing


def test_tacex_uipc_installer_is_pinned_and_base_environment_isolated() -> None:
    uipc = (SCRIPTS / "install_tacex_uipc_univtac.sh").read_text(encoding="utf-8")
    core = (SCRIPTS / "install_tacex_univtac.sh").read_text(encoding="utf-8")
    common = (SCRIPTS / "common.sh").read_text(encoding="utf-8")
    _assert_contains(
        uipc,
        VCPKG_COMMIT,
        VCPKG_BASELINE,
        'MICROMAMBA_VERSION="2.9.0"',
        "--no-rc",
        "-u CONDA_PREFIX",
        'TMPDIR="$DEPLOY_ROOT/runtime/tmp"',
        'CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"',
        "VCPKG_FORCE_SYSTEM_BINARIES=1",
        'VCPKG_OVERLAY_PORTS="$VCPKG_OVERLAY_ROOT"',
        "cpptrace_overlay_version=0.8.3",
        "tinygltf_overlay_version=2.9.3",
        "--tinygltf-archive",
        'sha512_file "$TINYGLTF_ARCHIVE"',
        "--force-reinstall",
    )
    _assert_contains(
        core,
        'TORCH_CUDA_ARCH_LIST="$CUDA_COMPUTE_CAPABILITY"',
        "--no-cache-dir",
        "cuda_toolkit_path=$CUDA_ROOT",
    )
    _assert_contains(
        common,
        "get_device_capability(0)",
        "runtime/cuda-toolkit-12.8/bin/nvcc",
        "runtime/cuda-toolkit-12.4/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "100|101|120",
    )
    assert (
        "CMAKE_CUDA_ARCHITECTURES=86" not in uipc
        and "TORCH_CUDA_ARCH_LIST=8.6" not in core
    )


def test_univtac_task_import_qualification_has_explicit_evidence_boundary() -> None:
    qualifier = (SCRIPTS / "qualify_univtac_task_import.sh").read_text(encoding="utf-8")
    smoke = (SCRIPTS / "smoke_univtac_task_isaac.py").read_text(encoding="utf-8")
    _assert_contains(
        qualifier,
        "--task",
        "REGISTRY_RESOURCE_SHA256",
        "TASK_SOURCE_SHA256",
        "cpptrace_overlay_version=0.8.3",
        "task_instantiated=false",
        "simulator_steps_executed=0",
        "univtac_task_import_${TASK_ID}_v3.json",
    )
    _assert_contains(
        smoke,
        "_task_contract(task_id)",
        "config_class()",
        '"task_source_sha256"',
        'extension_ids = ("omni.ui",)',
        "AppLauncher(headless=True)",
    )
    assert "task_class(" not in smoke


def test_univtac_task_reset_qualification_uses_production_backend_contract() -> None:
    qualifier = (SCRIPTS / "qualify_univtac_task_reset.sh").read_text(encoding="utf-8")
    smoke = (SCRIPTS / "smoke_univtac_task_reset_isaac.py").read_text(encoding="utf-8")
    _assert_contains(
        qualifier,
        "univtac_task_import_${TASK_ID}_v3.json",
        "univtac_task_reset_${CONTRACT_RUN_ID}.json",
        "univtac_task_reset_witness_${CONTRACT_RUN_ID}.json",
        "robotactile_isaac_install-${CURRENT_SOURCE_MANIFEST_SHA256:0:16}.json",
        "runtime_source_manifest_sha256",
        "require_command timeout",
        'CUDA_VISIBLE_DEVICES="$GPU_INDEX"',
        '--action-spec "$ACTION_SPEC"',
        "task_source_sha256=$TASK_SOURCE_SHA256",
        "construction_seed=$INITIAL_SEED",
        "cublas_workspace_config=:4096:8",
        "task_success_evaluated=false",
        "isaac_headless_task_reset_observation_v2",
        "reset_viable=$RESET_VIABLE",
        "placement_failure_phase=${RESULT_FIELDS[13]}",
        "witness_sha256=$WITNESS_SHA256",
        "status=$QUALIFICATION_STATUS",
    )
    _assert_contains(
        smoke,
        "build_univtac_backend_config(task_id, action_spec=action_spec)",
        "launch_univtac_runtime",
        "UniVTACIsaacBackend",
        "PolicyEpisodeContext",
        "backend.reset(context)",
        "backend.observe()",
        '"task_instantiated": True',
        '"reset_completed": True',
        '"policy_loaded": False',
        '"action_commands_executed": 0',
    )
    assert smoke.index("_publish_result(result_path, payload)") < smoke.index(
        "backend.close()"
    )
    assert "backend.execute(" not in smoke and "check_success(" not in smoke


def test_robotactile_isaac_installer_is_manifest_bound_and_isolated() -> None:
    installer = (SCRIPTS / "install_robotactile_isaac.sh").read_text(encoding="utf-8")
    _assert_contains(
        installer,
        "update_source_manifest.py",
        "source_manifest_sha256=",
        "wheel_sha256=",
        "--force-reinstall",
        "--no-deps",
        "--wheel",
        "py3-none-any.whl",
        '"$ISAAC_SIM_PATH/python.sh"',
        "-u CONDA_PREFIX",
        "robotactile_isaac_install-${SOURCE_MANIFEST_SHA256:0:16}.json",
        'WHEEL_MANIFEST_MEMBER="robotactile_benchmark/source_manifest.sha256"',
        "wheel_source_manifest_sha256",
        "zipfile.ZipFile",
    )
    wheel_branch = installer.index('if [ -n "$WHEEL_INPUT" ]; then')
    source_check = installer.index("update_source_manifest.py")
    assert wheel_branch < source_check
    assert "\nsudo " not in installer


@pytest.mark.parametrize("runtime_matches", (True, False))
def test_robotactile_isaac_installer_uses_explicit_wheel_manifest(
    tmp_path: Path,
    runtime_matches: bool,
) -> None:
    deploy_root = tmp_path / "deploy"
    env, _ = _isolated_env(tmp_path)
    manifest_payload = b"wheel-owned source manifest\n"
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    wheel_path = tmp_path / "robotactile_benchmark-0.6.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(
            "robotactile_benchmark/source_manifest.sha256",
            manifest_payload,
        )

    isaac_python = deploy_root / "runtime/isaac-sim-4.5.0/python.sh"
    isaac_python.parent.mkdir(parents=True)
    isaac_python.write_text(
        """#!/usr/bin/env bash
set -eu
case "$*" in
  *"-m pip install"*) printf '%s\n' "$*" > "$FAKE_ISAAC_INSTALL_LOG" ;;
  *load_source_manifest*)
    printf '0.6.0|%s|1.26.0\n' "$FAKE_WHEEL_MANIFEST_SHA256"
    ;;
  *) exit 92 ;;
esac
""",
        encoding="utf-8",
    )
    isaac_python.chmod(0o755)
    _write_json_fixture(
        deploy_root / "artifacts/deployment/isaaclab_install.json",
        {
            "component": "isaaclab",
            "version": "v2.1.1",
            "status": "installed",
        },
    )

    system_python_log = tmp_path / "system-python.log"
    system_python = tmp_path / "system-python"
    system_python.write_text(
        f"""#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$FAKE_SYSTEM_PYTHON_LOG"
case "$*" in *update_source_manifest.py*) exit 73 ;; esac
exec {sys.executable} "$@"
""",
        encoding="utf-8",
    )
    system_python.chmod(0o755)
    isaac_install_log = tmp_path / "isaac-install.log"
    env.update(
        {
            "FAKE_ISAAC_INSTALL_LOG": str(isaac_install_log),
            "FAKE_SYSTEM_PYTHON_LOG": str(system_python_log),
            "FAKE_WHEEL_MANIFEST_SHA256": (
                manifest_sha256 if runtime_matches else "0" * 64
            ),
            "ROBOTACTILE_SYSTEM_PYTHON": str(system_python),
        }
    )

    result = _run(
        "install_robotactile_isaac.sh",
        "--root",
        str(deploy_root),
        "--wheel",
        str(wheel_path),
        env=env,
    )

    assert result.returncode == (0 if runtime_matches else 2), result.stderr
    assert "update_source_manifest.py" not in system_python_log.read_text(
        encoding="utf-8"
    )
    assert str(wheel_path) in isaac_install_log.read_text(encoding="utf-8")
    receipt_path = (
        deploy_root
        / "artifacts/deployment"
        / f"robotactile_isaac_install-{manifest_sha256[:16]}.json"
    )
    if not runtime_matches:
        assert "runtime identity is incompatible" in result.stderr
        assert not receipt_path.exists()
        return
    receipt = _read_json(receipt_path)
    assert receipt["source_manifest_sha256"] == manifest_sha256
    assert (
        receipt["wheel_sha256"] == hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    )
    assert receipt["runtime_identity"] == f"0.6.0|{manifest_sha256}|1.26.0"


def test_univtac_pairing_qualifier_is_live_bounded_and_fail_closed() -> None:
    qualifier = (SCRIPTS / "qualify_univtac_task_pairing.sh").read_text(
        encoding="utf-8"
    )
    smoke = (SCRIPTS / "smoke_univtac_task_pairing_isaac.py").read_text(
        encoding="utf-8"
    )
    _assert_contains(
        qualifier,
        "ensure_pinned_git_source",
        "univtac_task_pairing_${CONTRACT_RUN_ID}.json",
        '--action-spec "$ACTION_SPEC"',
        "task_source_sha256=$TASK_SOURCE_SHA256",
        "require_command timeout",
        'CUDA_VISIBLE_DEVICES="$GPU_INDEX"',
        "in_process_snapshot_replay_equivalence_v1",
        "simulator_qualification_claimed=false",
        "task_success_evaluated=false",
    )
    _assert_contains(
        smoke,
        "UniVTACPairedBackendSession",
        "_divergence_action(",
        "initial_model_visible_qpos8",
        "canonical_record.observation.proprio",
        "np.array(initial_action8",
        "session.new_backend()",
        '"canonical_reset"',
        '"snapshot_replay"',
        '"policy_loaded": False',
    )
    assert "qualification_action" not in smoke
    assert smoke.index("_publish(result_path, payload)") < smoke.index(
        "session.close()"
    )

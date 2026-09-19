"""Isolated cuRobo builds preserve shared binaries and shell argument boundaries."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.retrained_evaluation import build_curobo_overlay as overlay


def test_wrapper_quotes_paths_and_changes_only_child_environment(
    tmp_path: Path,
) -> None:
    special = tmp_path / "space ' quote $(touch BAD)"
    special.mkdir()
    isaac = special / "fake isaac.sh"
    isaac.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$PYTHONPATH" '
        '"$TORCH_EXTENSIONS_DIR" "$@"\n'
    )
    isaac.chmod(0o755)
    source, cache = special / "source", special / "cache"
    wrapper = special / "wrapper.sh"
    wrapper.write_text(overlay.wrapper_text(source, isaac, cache))
    parent_env = dict(
        os.environ, PYTHONPATH="existing path", TORCH_EXTENSIONS_DIR="old"
    )
    before = parent_env.copy()
    result = subprocess.run(
        ["bash", str(wrapper), "-m", "module name", "$(touch BAD)"],
        env=parent_env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        f"{source / 'src'}:existing path",
        str(cache),
        "-m",
        "module name",
        "$(touch BAD)",
    ]
    assert parent_env == before
    assert not (tmp_path / "BAD").exists()


def test_binary_hashes_include_only_direct_native_extensions(tmp_path: Path) -> None:
    folder = tmp_path / "src/curobo/curobolib"
    folder.mkdir(parents=True)
    (folder / "b.so").write_bytes(b"second")
    (folder / "a.so").write_bytes(b"first")
    (folder / "module.py").write_text("ignored")
    assert overlay.binary_hashes(tmp_path) == {
        "src/curobo/curobolib/a.so": hashlib.sha256(b"first").hexdigest(),
        "src/curobo/curobolib/b.so": hashlib.sha256(b"second").hexdigest(),
    }
    assert overlay.binary_hashes(tmp_path / "absent") == {}


def setup_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: Path) -> Path:
    source = tmp_path / "source"
    native = source / "src/curobo/curobolib"
    native.mkdir(parents=True)
    (native / "old.so").write_bytes(b"original sm120")
    (source / "setup.py").write_text("# source fixture\n")
    isaac = tmp_path / "python.sh"
    isaac.touch()
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "bin/nvcc").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_curobo_overlay",
            "--source",
            str(source),
            "--output",
            str(output),
            "--isaac-python",
            str(isaac),
            "--cuda-root",
            str(cuda),
            "--architecture",
            "8.0",
        ],
    )
    return source


@pytest.mark.parametrize("location", ["existing", "inside", "same"])
def test_rejects_unsafe_output_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str
) -> None:
    output = {
        "existing": tmp_path / "existing",
        "inside": tmp_path / "source/overlay",
        "same": tmp_path / "source",
    }[location]
    source = setup_inputs(tmp_path, monkeypatch, output)
    if location == "existing":
        output.mkdir()
        (output / "preserve").write_text("user data")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid output must fail before launching subprocesses")

    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    original = overlay.binary_hashes(source)
    error = FileExistsError if location == "existing" else ValueError
    with pytest.raises(error):
        overlay.main()
    assert overlay.binary_hashes(source) == original
    if location == "existing":
        assert (output / "preserve").read_text() == "user data"


def test_build_only_modifies_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "overlay"
    source = setup_inputs(tmp_path, monkeypatch, output)
    original = overlay.binary_hashes(source)
    parent_env = dict(os.environ)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "abc123\n")

    def build(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert argv[1:] == ["setup.py", "build_ext", "--inplace"]
        private = output / "curobo"
        assert kwargs["cwd"] == private
        assert overlay.binary_hashes(private) == {}
        assert kwargs["env"]["TORCH_CUDA_ARCH_LIST"] == "8.0"
        assert kwargs["env"]["PYTHONPATH"] == str(private / "src")
        (private / "src/curobo/curobolib/sm80.so").write_bytes(b"new sm80")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", build)
    overlay.main()
    receipt = json.loads((output / "build_receipt.json").read_text())
    assert receipt["returncode"] == 0
    assert receipt["original_binaries_unchanged"] is True
    assert receipt["built_binary_sha256"] == overlay.binary_hashes(output / "curobo")
    assert overlay.binary_hashes(source) == original
    assert dict(os.environ) == parent_env
    assert os.access(output / "isaac-python.sh", os.X_OK)

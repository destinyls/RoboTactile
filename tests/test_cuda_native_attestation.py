"""Behavior tests for fail-closed CUDA native-extension attestation."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1] / "scripts/live_univtac/attest_cuda_native_extensions.py"
)


def _write_binary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fixture:{path.name}\n".encode())


def _fixture(root: Path, *, architecture: str) -> tuple[list[str], Path]:
    torch_dir = root / "runtime/isaac/site-packages/torch_scatter"
    uipc_dir = root / "sources/UniVTAC/uipc/Release/bin"
    curobo_dir = root / "sources/curobo/src/curobo/curobolib"
    for name in (
        "_scatter_cuda.so",
        "_segment_coo_cuda.so",
        "_segment_csr_cuda.so",
        "_version_cuda.so",
    ):
        _write_binary(torch_dir / name)
    _write_binary(uipc_dir / "libuipc_backend_cuda.so")
    for name in (
        "geom_cu.so",
        "kinematics_fused_cu.so",
        "lbfgs_step_cu.so",
        "line_search_cu.so",
        "tensor_step_cu.so",
    ):
        _write_binary(curobo_dir / name)

    cuobjdump = root / "runtime/cuda-toolkit/bin/cuobjdump"
    cuobjdump.parent.mkdir(parents=True)
    cuobjdump.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  --list-elf) printf 'ELF file 1: sm_{architecture}.cubin\\n' ;;\n"
        f"  --list-ptx) printf 'PTX file 1: compute_{architecture}.ptx\\n' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    cuobjdump.chmod(cuobjdump.stat().st_mode | stat.S_IXUSR)
    receipt = root / "artifacts/deployment/tacex_install.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "installed"}), encoding="utf-8")
    output = root / "artifacts/deployment/cuda_native_extensions.json"
    arguments = [
        "--root",
        str(root),
        "--cuobjdump",
        str(cuobjdump),
        "--architecture",
        "120",
        "--torch-scatter-dir",
        str(torch_dir),
        "--uipc-dir",
        str(uipc_dir),
        "--curobo-dir",
        str(curobo_dir),
        "--dependency-receipt",
        str(receipt),
        "--output",
        str(output),
    ]
    return arguments, output


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_sm120_attestation_records_all_expected_native_binaries() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "deploy"
        root.mkdir()
        arguments, output = _fixture(root, architecture="120")

        first = _run(arguments)
        document = json.loads(output.read_text(encoding="utf-8"))
        repeated = _run(arguments)

        assert first.returncode == 0, first.stderr
        assert document["status"] == "passed"
        assert document["cuda_architecture"] == "sm_120"
        assert document["component_counts"] == {
            "curobo": 5,
            "tacex_uipc": 1,
            "torch_scatter": 3,
        }
        assert document["binary_count"] == 9
        assert all(
            record["cubin_architectures"] == ["120"]
            and record["target_cubin_present"] is True
            for record in document["binaries"]
        )
        assert document["non_kernel_binary_count"] == 1
        assert document["non_kernel_binaries"] == [
            {
                "component": "torch_scatter",
                "device_code_required": False,
                "path": ("runtime/isaac/site-packages/torch_scatter/_version_cuda.so"),
                "role": "cuda_version_metadata_stub",
                "sha256": document["non_kernel_binaries"][0]["sha256"],
            }
        ]
        assert repeated.returncode != 0
        assert "refusing to overwrite CUDA attestation" in repeated.stderr


def test_sm120_attestation_rejects_legacy_sm80_only_binaries() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "deploy"
        root.mkdir()
        arguments, output = _fixture(root, architecture="80")

        result = _run(arguments)

        assert result.returncode != 0
        assert "lacks required sm_120 cubin" in result.stderr
        assert not output.exists()


def test_sm120_attestation_rejects_unknown_torch_scatter_cuda_binary() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "deploy"
        root.mkdir()
        arguments, output = _fixture(root, architecture="120")
        torch_dir = root / "runtime/isaac/site-packages/torch_scatter"
        _write_binary(torch_dir / "_unexpected_cuda.so")

        result = _run(arguments)

        assert result.returncode != 0
        assert "unexpected torch-scatter CUDA extensions" in result.stderr
        assert not output.exists()

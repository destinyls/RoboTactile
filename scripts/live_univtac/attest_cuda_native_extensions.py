#!/usr/bin/env python3
"""Fail-closed CUDA code-object attestation for the UniVTAC native stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

_CUBIN_PATTERN: Final = re.compile(r"sm_(\d+)\.cubin")
_PTX_PATTERN: Final = re.compile(r"compute_(\d+)\.ptx")
_TORCH_SCATTER_KERNEL_NAMES: Final = (
    "_scatter_cuda.so",
    "_segment_coo_cuda.so",
    "_segment_csr_cuda.so",
)
_TORCH_SCATTER_METADATA_NAME: Final = "_version_cuda.so"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--torch-scatter-dir", type=Path, required=True)
    parser.add_argument("--uipc-dir", type=Path, required=True)
    parser.add_argument("--curobo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dependency-receipt",
        type=Path,
        action="append",
        default=[],
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"{label} is outside deployment root: {resolved}") from error
    return resolved


def _run_cuobjdump(cuobjdump: Path, option: str, binary: Path) -> str:
    completed = subprocess.run(
        [str(cuobjdump), option, str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"cuobjdump {option} failed for {binary}: {detail or completed.returncode}"
        )
    return completed.stdout


def _discover_binaries(
    torch_scatter_dir: Path,
    uipc_dir: Path,
    curobo_dir: Path,
) -> tuple[list[tuple[str, Path]], Path]:
    discovered_torch_scatter = {
        path.name: path for path in torch_scatter_dir.glob("*_cuda.so")
    }
    allowed_torch_scatter = set(_TORCH_SCATTER_KERNEL_NAMES) | {
        _TORCH_SCATTER_METADATA_NAME
    }
    unexpected_torch_scatter = sorted(
        set(discovered_torch_scatter) - allowed_torch_scatter
    )
    if unexpected_torch_scatter:
        raise SystemExit(
            f"unexpected torch-scatter CUDA extensions: {unexpected_torch_scatter}"
        )
    missing_torch_scatter = [
        name
        for name in (*_TORCH_SCATTER_KERNEL_NAMES, _TORCH_SCATTER_METADATA_NAME)
        if name not in discovered_torch_scatter
    ]
    if missing_torch_scatter:
        raise SystemExit(
            "expected torch-scatter CUDA extensions are absent: "
            f"{missing_torch_scatter}"
        )
    torch_scatter_kernels = [
        discovered_torch_scatter[name] for name in _TORCH_SCATTER_KERNEL_NAMES
    ]
    torch_scatter_metadata = discovered_torch_scatter[_TORCH_SCATTER_METADATA_NAME]
    curobo = sorted(curobo_dir.glob("*_cu*.so"))
    uipc = uipc_dir / "libuipc_backend_cuda.so"
    if not uipc.is_file():
        raise SystemExit(f"modified UIPC CUDA backend is absent: {uipc}")
    if len(curobo) != 5:
        raise SystemExit(f"expected five cuRobo CUDA extensions, found {len(curobo)}")
    binaries = [
        *(("torch_scatter", path) for path in torch_scatter_kernels),
        ("tacex_uipc", uipc),
        *(("curobo", path) for path in curobo),
    ]
    return binaries, torch_scatter_metadata


def _dependency_receipts(root: Path, paths: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in paths:
        resolved = _inside(root, path, "dependency receipt")
        document = json.loads(resolved.read_text(encoding="utf-8"))
        if document.get("status") != "installed":
            raise SystemExit(f"dependency receipt is not installed: {resolved}")
        records.append(
            {
                "path": str(resolved.relative_to(root)),
                "sha256": _sha256(resolved),
            }
        )
    return records


def _publish(output: Path, document: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite CUDA attestation: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cuda-attestation.", suffix=".json", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise SystemExit(
                f"refusing to overwrite CUDA attestation: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.root.resolve(strict=True)
    if not root.is_dir():
        raise SystemExit(f"deployment root is not a directory: {root}")
    architecture = str(arguments.architecture)
    if not architecture.isdigit() or len(architecture) not in (2, 3):
        raise SystemExit("architecture must be a two/three-digit CUDA target")
    cuobjdump = _inside(root, arguments.cuobjdump, "cuobjdump")
    if not os.access(cuobjdump, os.X_OK):
        raise SystemExit(f"cuobjdump is not executable: {cuobjdump}")
    torch_scatter_dir = _inside(
        root, arguments.torch_scatter_dir, "torch-scatter directory"
    )
    uipc_dir = _inside(root, arguments.uipc_dir, "modified UIPC directory")
    curobo_dir = _inside(root, arguments.curobo_dir, "cuRobo directory")

    binaries, torch_scatter_metadata = _discover_binaries(
        torch_scatter_dir, uipc_dir, curobo_dir
    )
    resolved_torch_scatter_metadata = _inside(
        root, torch_scatter_metadata, "torch-scatter metadata binary"
    )
    non_kernel_records: list[dict[str, object]] = [
        {
            "component": "torch_scatter",
            "device_code_required": False,
            "path": str(resolved_torch_scatter_metadata.relative_to(root)),
            "role": "cuda_version_metadata_stub",
            "sha256": _sha256(resolved_torch_scatter_metadata),
        }
    ]
    records: list[dict[str, object]] = []
    component_counts: dict[str, int] = {}
    for component, binary in binaries:
        resolved = _inside(root, binary, f"{component} binary")
        cubin_output = _run_cuobjdump(cuobjdump, "--list-elf", resolved)
        ptx_output = _run_cuobjdump(cuobjdump, "--list-ptx", resolved)
        cubin_architectures = sorted(set(_CUBIN_PATTERN.findall(cubin_output)))
        ptx_architectures = sorted(set(_PTX_PATTERN.findall(ptx_output)))
        if architecture not in cubin_architectures:
            raise SystemExit(
                f"{resolved} lacks required sm_{architecture} cubin; "
                f"found {cubin_architectures or 'none'}"
            )
        component_counts[component] = component_counts.get(component, 0) + 1
        records.append(
            {
                "component": component,
                "path": str(resolved.relative_to(root)),
                "sha256": _sha256(resolved),
                "cubin_architectures": cubin_architectures,
                "ptx_architectures": ptx_architectures,
                "target_cubin_present": True,
            }
        )

    output = arguments.output.resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"output is outside deployment root: {output}") from error
    document: dict[str, object] = {
        "schema_version": "robotactile.cuda_native_attestation.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "cuda_architecture": f"sm_{architecture}",
        "cuobjdump_path": str(cuobjdump.relative_to(root)),
        "cuobjdump_sha256": _sha256(cuobjdump),
        "component_counts": component_counts,
        "binary_count": len(records),
        "binaries": records,
        "non_kernel_binary_count": len(non_kernel_records),
        "non_kernel_binaries": non_kernel_records,
        "dependency_receipts": _dependency_receipts(
            root, list(arguments.dependency_receipt)
        ),
    }
    _publish(output, document)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

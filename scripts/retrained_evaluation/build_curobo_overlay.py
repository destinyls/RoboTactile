"""Build cuRobo for one architecture without modifying any installed runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path


def binary_hashes(source: Path) -> dict[str, str]:
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source / "src/curobo/curobolib").glob("*.so"))
    }


def wrapper_text(source: Path, isaac_python: Path, cache: Path) -> str:
    """Only the Isaac child receives the private import and JIT cache paths."""
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export PYTHONPATH={shlex.quote(str(source / 'src'))}"
        "${PYTHONPATH:+:$PYTHONPATH}\n"
        f"export TORCH_EXTENSIONS_DIR={shlex.quote(str(cache))}\n"
        f'exec {shlex.quote(str(isaac_python))} "$@"\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    isaac_python = args.isaac_python.resolve(strict=True)
    cuda_root = args.cuda_root.resolve(strict=True)
    if output == source or output.is_relative_to(source):
        raise ValueError("output must be separate from the original source")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if not (source / "setup.py").is_file():
        raise ValueError("source must contain the cuRobo setup.py")
    if not (cuda_root / "bin/nvcc").is_file():
        raise ValueError("CUDA toolkit must provide nvcc")
    original = binary_hashes(source)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {
        "hostname": socket.gethostname(),
        "source": str(source),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "original_binary_sha256": original,
        "isaac_python": str(isaac_python),
        "cuda_root": str(cuda_root),
        "architecture": args.architecture,
        "started_unix": started,
    }
    with (output / "input_identity.json").open("x") as stream:
        json.dump(identity, stream, indent=2)
    private = output / "curobo"
    shutil.copytree(
        source,
        private,
        ignore=shutil.ignore_patterns(
            ".git", "build", "dist", "__pycache__", "*.so", "*.egg-info"
        ),
    )
    env = dict(os.environ)
    env.update(
        CUDA_HOME=str(cuda_root),
        CUDACXX=str(cuda_root / "bin/nvcc"),
        PATH=str(cuda_root / "bin") + os.pathsep + env.get("PATH", ""),
        PYTHONPATH=str(private / "src"),
        TORCH_CUDA_ARCH_LIST=args.architecture,
        TORCH_EXTENSIONS_DIR=str(output / "torch_extensions"),
        MAX_JOBS=str(args.jobs),
        SETUPTOOLS_SCM_PRETEND_VERSION="0.7.7",
        PYTHONDONTWRITEBYTECODE="1",
    )
    with (output / "build.log").open("xb") as log:
        result = subprocess.run(
            [str(isaac_python), "setup.py", "build_ext", "--inplace"],
            cwd=private,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    unchanged = binary_hashes(source) == original
    receipt = {
        **identity,
        "finished_unix": time.time(),
        "returncode": result.returncode,
        "original_binaries_unchanged": unchanged,
        "built_binary_sha256": binary_hashes(private),
    }
    with (output / "build_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    if result.returncode or not unchanged:
        raise RuntimeError(
            "isolated build failed; see build.log and build_receipt.json"
        )
    wrapper = output / "isaac-python.sh"
    with wrapper.open("x") as stream:
        stream.write(wrapper_text(private, isaac_python, output / "torch_extensions"))
    wrapper.chmod(0o755)


if __name__ == "__main__":
    main()

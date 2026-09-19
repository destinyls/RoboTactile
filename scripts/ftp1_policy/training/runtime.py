"""Filesystem, process, and status helpers for FTP-1 training."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from .contracts import TRAINING_SCOPE, FTP1TrainingRequest, canonical_json_sha256

LOGGER = logging.getLogger(__name__)
STATUS_SCHEMA: Final[str] = "robotactile-ftp1-joint-training-status-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed(payload: Mapping[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = canonical_json_sha256(result)
    return result


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def write_or_verify(path: Path, payload: Mapping[str, object]) -> None:
    """Create an immutable JSON artifact or verify equivalent content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if load_object(path) != dict(payload):
            raise FileExistsError(f"existing artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if load_object(path) != dict(payload):
            raise
    finally:
        temporary.unlink(missing_ok=True)


def write_status(
    request: FTP1TrainingRequest,
    *,
    phase: str,
    task: str | None,
    completed_tasks: Sequence[str],
    prepared_tasks: Sequence[str],
    error: str | None = None,
) -> None:
    if error is not None:
        status = "failed"
    elif phase == "complete":
        status = "complete"
    elif phase.endswith("_complete"):
        status = "ready"
    else:
        status = "running"
    joint_complete = len(completed_tasks) == len(request.tasks)
    payload = {
        "schema_version": STATUS_SCHEMA,
        "training_scope": TRAINING_SCOPE,
        "run_id": request.run_id,
        "request_sha256": request.request_sha256,
        "status": status,
        "phase": phase,
        "task": task,
        "prepared_tasks": list(prepared_tasks),
        "prepared_task_count": len(prepared_tasks),
        "completed_tasks": list(completed_tasks),
        "completed_task_count": len(completed_tasks),
        "target_task_count": len(request.tasks),
        "joint_model_complete": joint_complete,
        "completed_checkpoint_count": 1 if joint_complete else 0,
        "target_checkpoint_count": 1,
        "error": error,
    }
    path = request.output_root / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def explicit_environment(request: FTP1TrainingRequest) -> dict[str, str]:
    temporary = request.output_root / "temp"
    compile_root = temporary / "torch_compile"
    socket_tag = hashlib.sha256(request.run_id.encode("utf-8")).hexdigest()[:8]
    socket_root = Path("/tmp") / f"rtf1-{socket_tag}"
    temporary.mkdir(parents=True, exist_ok=True)
    compile_root.mkdir(parents=True, exist_ok=True)
    socket_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    source_paths = [str(request.ftp1_root / "src"), str(request.ftp1_root)]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        source_paths.append(inherited_pythonpath)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(request.cuda_device),
            "HYDRA_FULL_ERROR": "1",
            "JAX_PLATFORM": "cpu",
            "OPENPI_DATA_HOME": str(request.openpi_data_home),
            "PYTHONHASHSEED": str(request.seed),
            "PYTHONPATH": os.pathsep.join(source_paths),
            "TMPDIR": str(socket_root),
            "TORCH_COMPILE_DIR": str(compile_root),
            "WANDB_MODE": "disabled" if not request.wandb_enabled else "online",
        }
    )
    return environment


def command_output(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_logged(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("running %s", " ".join(command))
    with log_path.open("a", encoding="utf-8") as stream:
        subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def tree_inventory(root: Path) -> list[dict[str, object]]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact tree is empty: {root}")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


__all__ = [
    "command_output",
    "explicit_environment",
    "load_object",
    "run_logged",
    "sha256_file",
    "signed",
    "tree_inventory",
    "write_or_verify",
    "write_status",
]

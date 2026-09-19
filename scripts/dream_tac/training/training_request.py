"""Immutable, source-bound requests for Dream-Tac UniVTAC training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, cast

from .identity import canonical_json_sha256
from .training_constants import DREAM_TAC_UPSTREAM_COMMIT

REQUEST_SCHEMA: Final[str] = "robotactile-dream-tac-training-request-v1"
ACCELERATOR_CONTRACT: Final[str] = "nvidia_cuda_only_v1"
DONOR_EXPERIMENT: Final[str] = "cosmos_predict2_2b_480p_franka_cut_banana_20260321"
Phase = Literal["p2_micro", "p3_full"]

_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accelerator_contract",
        "base_checkpoint",
        "batch_size",
        "cuda_devices",
        "donor_experiment",
        "dream_tac_commit",
        "dream_tac_root",
        "master_port",
        "materialization_receipt_sha256",
        "materialization_root",
        "max_iter",
        "nproc_per_node",
        "num_workers",
        "output_root",
        "phase",
        "python_executable",
        "resume_checkpoint",
        "run_name",
        "save_iter",
        "schema_version",
        "source_manifest_sha256",
        "t5_cache_sha256",
    }
)
_ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset({"path", "sha256"})
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{2,95}")


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class BoundFile:
    """One explicitly content-addressed regular file."""

    path: Path
    sha256: str

    @classmethod
    def from_value(cls, value: object, name: str) -> BoundFile:
        if not isinstance(value, dict) or set(value) != _ARTIFACT_FIELDS:
            raise ValueError(f"{name} fields must be exactly path and sha256")
        return cls(
            path=_absolute_path(value["path"], f"{name}.path"),
            sha256=_sha256(value["sha256"], f"{name}.sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class DreamTacTrainingRequest:
    """Validated P2/P3 request; paths remain external to the main wheel."""

    phase: Phase
    run_name: str
    dream_tac_root: Path
    python_executable: Path
    materialization_root: Path
    output_root: Path
    source_manifest_sha256: str
    materialization_receipt_sha256: str
    t5_cache_sha256: str
    base_checkpoint: BoundFile
    resume_checkpoint: BoundFile | None
    cuda_devices: tuple[int, ...]
    nproc_per_node: int
    master_port: int
    max_iter: int
    save_iter: int
    batch_size: int
    num_workers: int

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DreamTacTrainingRequest:
        if set(value) != _FIELDS:
            raise ValueError("Dream-Tac training request fields mismatch")
        if value["schema_version"] != REQUEST_SCHEMA:
            raise ValueError("unsupported Dream-Tac training request schema")
        if value["accelerator_contract"] != ACCELERATOR_CONTRACT:
            raise ValueError("Dream-Tac training supports NVIDIA CUDA only")
        if value["dream_tac_commit"] != DREAM_TAC_UPSTREAM_COMMIT:
            raise ValueError("Dream-Tac request commit does not match the pin")
        if value["donor_experiment"] != DONOR_EXPERIMENT:
            raise ValueError("Dream-Tac request must use the fixed tactile/CASA donor")
        phase_value = value["phase"]
        if phase_value not in {"p2_micro", "p3_full"}:
            raise ValueError("phase must be p2_micro or p3_full")
        phase = cast(Phase, phase_value)
        run_name = value["run_name"]
        if not isinstance(run_name, str) or _RUN_NAME.fullmatch(run_name) is None:
            raise ValueError("run_name must be a safe lowercase identifier")
        nproc = _integer(
            value["nproc_per_node"], "nproc_per_node", minimum=1, maximum=8
        )
        raw_devices = value["cuda_devices"]
        if not isinstance(raw_devices, list):
            raise ValueError("cuda_devices must be a list")
        devices = tuple(
            _integer(item, "cuda device", minimum=0, maximum=255)
            for item in raw_devices
        )
        if len(devices) != nproc or len(set(devices)) != len(devices):
            raise ValueError("cuda_devices must uniquely match nproc_per_node")
        max_iter = _integer(value["max_iter"], "max_iter", minimum=1, maximum=1_000_000)
        if phase == "p2_micro" and max_iter > 1_000:
            raise ValueError("P2 micro-train cannot exceed 1000 iterations")
        if phase == "p3_full" and max_iter <= 1_000:
            raise ValueError("P3 full training must exceed 1000 iterations")
        resume_value = value["resume_checkpoint"]
        resume = (
            None
            if resume_value is None
            else BoundFile.from_value(resume_value, "resume_checkpoint")
        )
        request = cls(
            phase=phase,
            run_name=run_name,
            dream_tac_root=_absolute_path(value["dream_tac_root"], "dream_tac_root"),
            python_executable=_absolute_path(
                value["python_executable"], "python_executable"
            ),
            materialization_root=_absolute_path(
                value["materialization_root"], "materialization_root"
            ),
            output_root=_absolute_path(value["output_root"], "output_root"),
            source_manifest_sha256=_sha256(
                value["source_manifest_sha256"], "source_manifest_sha256"
            ),
            materialization_receipt_sha256=_sha256(
                value["materialization_receipt_sha256"],
                "materialization_receipt_sha256",
            ),
            t5_cache_sha256=_sha256(value["t5_cache_sha256"], "t5_cache_sha256"),
            base_checkpoint=BoundFile.from_value(
                value["base_checkpoint"], "base_checkpoint"
            ),
            resume_checkpoint=resume,
            cuda_devices=devices,
            nproc_per_node=nproc,
            master_port=_integer(
                value["master_port"], "master_port", minimum=1024, maximum=65535
            ),
            max_iter=max_iter,
            save_iter=_integer(
                value["save_iter"], "save_iter", minimum=1, maximum=max_iter
            ),
            batch_size=_integer(
                value["batch_size"], "batch_size", minimum=1, maximum=64
            ),
            num_workers=_integer(
                value["num_workers"], "num_workers", minimum=0, maximum=64
            ),
        )
        request._validate_path_separation()
        return request

    @classmethod
    def load(cls, path: Path) -> DreamTacTrainingRequest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dream-Tac training request must be a JSON object")
        return cls.from_dict(payload)

    def _validate_path_separation(self) -> None:
        for protected in (self.dream_tac_root, self.materialization_root):
            if self.output_root == protected or protected in self.output_root.parents:
                raise ValueError("output_root cannot contain a protected input root")
            if protected == self.output_root or self.output_root in protected.parents:
                raise ValueError("output_root cannot be inside a protected input root")

    def to_dict(self) -> dict[str, object]:
        return {
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "base_checkpoint": self.base_checkpoint.to_dict(),
            "batch_size": self.batch_size,
            "cuda_devices": list(self.cuda_devices),
            "donor_experiment": DONOR_EXPERIMENT,
            "dream_tac_commit": DREAM_TAC_UPSTREAM_COMMIT,
            "dream_tac_root": str(self.dream_tac_root),
            "master_port": self.master_port,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "materialization_root": str(self.materialization_root),
            "max_iter": self.max_iter,
            "nproc_per_node": self.nproc_per_node,
            "num_workers": self.num_workers,
            "output_root": str(self.output_root),
            "phase": self.phase,
            "python_executable": str(self.python_executable),
            "resume_checkpoint": (
                None
                if self.resume_checkpoint is None
                else self.resume_checkpoint.to_dict()
            ),
            "run_name": self.run_name,
            "save_iter": self.save_iter,
            "schema_version": REQUEST_SCHEMA,
            "source_manifest_sha256": self.source_manifest_sha256,
            "t5_cache_sha256": self.t5_cache_sha256,
        }

    @property
    def request_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    @property
    def job_project(self) -> str:
        return "robotactile_dream_tac"

    @property
    def job_group(self) -> str:
        return f"univtac_{self.phase}"

    @property
    def job_root(self) -> Path:
        return self.output_root / self.job_project / self.job_group / self.run_name

    @property
    def latest_marker(self) -> Path:
        return self.job_root / "checkpoints" / "latest_checkpoint.txt"

    @property
    def receipt_root(self) -> Path:
        return (
            self.output_root
            / "_robotactile_receipts"
            / self.run_name
            / self.request_sha256
        )


def sha256_regular_file(path: Path) -> str:
    """Hash a stable, non-symlink regular file."""

    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"bound artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    if (metadata.st_size, metadata.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"bound artifact changed while hashing: {path}")
    return digest.hexdigest()


def verify_bound_file(artifact: BoundFile, name: str) -> Path:
    path = artifact.path.resolve(strict=True)
    if sha256_regular_file(path) != artifact.sha256:
        raise ValueError(f"{name} SHA256 mismatch")
    return path


def executable_path(path: Path) -> Path:
    """Resolve a venv symlink while requiring one executable regular target."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("python_executable is not executable")
    return resolved


__all__ = [
    "ACCELERATOR_CONTRACT",
    "DONOR_EXPERIMENT",
    "REQUEST_SCHEMA",
    "BoundFile",
    "DreamTacTrainingRequest",
    "executable_path",
    "sha256_regular_file",
    "verify_bound_file",
]

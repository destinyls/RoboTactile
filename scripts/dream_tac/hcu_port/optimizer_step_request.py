"""Immutable request for one Dream-Tac optimizer step on one Hygon HCU."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

from scripts.dream_tac.training.identity import canonical_json_sha256
from scripts.dream_tac.training.training_request import DONOR_EXPERIMENT, BoundFile

from .overlay_contract import PINNED_COMMIT

REQUEST_SCHEMA: Final[str] = "robotactile-dream-tac-hcu-optimizer-step-request-v1"
ACCELERATOR_CONTRACT: Final[str] = "hygon_hcu_hip_single_device_v1"
PHASE: Final[str] = "p2_hcu_optimizer_step"
_ZERO_SHA256: Final[str] = "0" * 64
_RUN_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9_.-]{2,95}")
_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "accelerator_contract",
        "base_checkpoint",
        "base_dcp_receipt",
        "base_dcp_root",
        "batch_size",
        "casa_receipt",
        "donor_experiment",
        "dream_tac_commit",
        "dream_tac_root",
        "hcu_device",
        "hcu_environment_script",
        "master_port",
        "materialization_receipt_sha256",
        "materialization_root",
        "max_iter",
        "nproc_per_node",
        "num_workers",
        "output_root",
        "overlay_receipt",
        "phase",
        "python_executable",
        "resume_checkpoint",
        "run_name",
        "runtime_pythonpath_roots",
        "save_iter",
        "schema_version",
        "source_manifest_sha256",
        "t5_cache_sha256",
        "tokenizer_checkpoint",
    }
)


def _absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    if value == _ZERO_SHA256:
        raise ValueError(f"{name} must not be an all-zero placeholder")
    return value


def _bound_file(value: object, name: str) -> BoundFile:
    artifact = BoundFile.from_value(value, name)
    _sha256(artifact.sha256, f"{name}.sha256")
    return artifact


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _runtime_pythonpath_roots(value: object) -> tuple[Path, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("runtime_pythonpath_roots must contain 1 to 8 paths")
    roots = tuple(
        _absolute_path(item, "runtime_pythonpath_roots item") for item in value
    )
    if len(set(roots)) != len(roots):
        raise ValueError("runtime_pythonpath_roots must be unique")
    return roots


@dataclass(frozen=True)
class HcuOptimizerStepRequest:
    """One source-bound, no-resume, single-device HCU optimizer-step request."""

    run_name: str
    dream_tac_root: Path
    python_executable: Path
    hcu_environment_script: BoundFile
    runtime_pythonpath_roots: tuple[Path, ...]
    materialization_root: Path
    output_root: Path
    source_manifest_sha256: str
    materialization_receipt_sha256: str
    t5_cache_sha256: str
    base_checkpoint: BoundFile
    base_dcp_root: Path
    base_dcp_receipt: BoundFile
    tokenizer_checkpoint: BoundFile
    overlay_receipt: BoundFile
    casa_receipt: BoundFile
    hcu_device: int
    master_port: int
    num_workers: int

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HcuOptimizerStepRequest:
        if set(value) != _FIELDS:
            raise ValueError("Dream-Tac HCU optimizer-step request fields mismatch")
        expected = {
            "schema_version": REQUEST_SCHEMA,
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "dream_tac_commit": PINNED_COMMIT,
            "donor_experiment": DONOR_EXPERIMENT,
            "phase": PHASE,
            "nproc_per_node": 1,
            "max_iter": 1,
            "save_iter": 1,
            "batch_size": 1,
            "resume_checkpoint": None,
        }
        for name, required in expected.items():
            if value[name] != required:
                raise ValueError(f"{name} must be {required!r}")
        run_name = value["run_name"]
        if not isinstance(run_name, str) or _RUN_NAME.fullmatch(run_name) is None:
            raise ValueError("run_name must be a safe lowercase identifier")
        request = cls(
            run_name=run_name,
            dream_tac_root=_absolute_path(value["dream_tac_root"], "dream_tac_root"),
            python_executable=_absolute_path(
                value["python_executable"], "python_executable"
            ),
            hcu_environment_script=_bound_file(
                value["hcu_environment_script"], "hcu_environment_script"
            ),
            runtime_pythonpath_roots=_runtime_pythonpath_roots(
                value["runtime_pythonpath_roots"]
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
            base_checkpoint=_bound_file(value["base_checkpoint"], "base_checkpoint"),
            base_dcp_root=_absolute_path(value["base_dcp_root"], "base_dcp_root"),
            base_dcp_receipt=_bound_file(value["base_dcp_receipt"], "base_dcp_receipt"),
            tokenizer_checkpoint=_bound_file(
                value["tokenizer_checkpoint"], "tokenizer_checkpoint"
            ),
            overlay_receipt=_bound_file(value["overlay_receipt"], "overlay_receipt"),
            casa_receipt=_bound_file(value["casa_receipt"], "casa_receipt"),
            hcu_device=_integer(
                value["hcu_device"], "hcu_device", minimum=0, maximum=255
            ),
            master_port=_integer(
                value["master_port"], "master_port", minimum=1024, maximum=65535
            ),
            num_workers=_integer(
                value["num_workers"], "num_workers", minimum=0, maximum=16
            ),
        )
        request._validate_path_separation()
        return request

    @classmethod
    def load(cls, path: Path) -> HcuOptimizerStepRequest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Dream-Tac HCU optimizer-step request must be an object")
        return cls.from_dict(payload)

    def _validate_path_separation(self) -> None:
        protected_roots = (
            self.dream_tac_root,
            self.materialization_root,
            self.base_dcp_root,
            *self.runtime_pythonpath_roots,
        )
        for protected in protected_roots:
            if self.output_root == protected or protected in self.output_root.parents:
                raise ValueError("output_root cannot contain a protected input root")
            if protected == self.output_root or self.output_root in protected.parents:
                raise ValueError("output_root cannot be inside a protected input root")
        environment_script = self.hcu_environment_script.path
        if (
            environment_script == self.output_root
            or self.output_root in environment_script.parents
        ):
            raise ValueError("hcu_environment_script cannot be inside output_root")

    @property
    def phase(self) -> str:
        return PHASE

    @property
    def nproc_per_node(self) -> int:
        return 1

    @property
    def max_iter(self) -> int:
        return 1

    @property
    def save_iter(self) -> int:
        return 1

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def resume_checkpoint(self) -> None:
        return None

    @property
    def job_project(self) -> str:
        return "robotactile_dream_tac"

    @property
    def job_group(self) -> str:
        return "univtac_p2_hcu_optimizer_step"

    @property
    def job_root(self) -> Path:
        return self.output_root / self.job_project / self.job_group / self.run_name

    @property
    def receipt_root(self) -> Path:
        return (
            self.output_root
            / "_robotactile_receipts"
            / self.run_name
            / self.request_sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "accelerator_contract": ACCELERATOR_CONTRACT,
            "base_checkpoint": self.base_checkpoint.to_dict(),
            "base_dcp_receipt": self.base_dcp_receipt.to_dict(),
            "base_dcp_root": str(self.base_dcp_root),
            "batch_size": self.batch_size,
            "casa_receipt": self.casa_receipt.to_dict(),
            "donor_experiment": DONOR_EXPERIMENT,
            "dream_tac_commit": PINNED_COMMIT,
            "dream_tac_root": str(self.dream_tac_root),
            "hcu_device": self.hcu_device,
            "hcu_environment_script": self.hcu_environment_script.to_dict(),
            "master_port": self.master_port,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "materialization_root": str(self.materialization_root),
            "max_iter": self.max_iter,
            "nproc_per_node": self.nproc_per_node,
            "num_workers": self.num_workers,
            "output_root": str(self.output_root),
            "overlay_receipt": self.overlay_receipt.to_dict(),
            "phase": self.phase,
            "python_executable": str(self.python_executable),
            "resume_checkpoint": None,
            "run_name": self.run_name,
            "runtime_pythonpath_roots": [
                str(path) for path in self.runtime_pythonpath_roots
            ],
            "save_iter": self.save_iter,
            "schema_version": REQUEST_SCHEMA,
            "source_manifest_sha256": self.source_manifest_sha256,
            "t5_cache_sha256": self.t5_cache_sha256,
            "tokenizer_checkpoint": self.tokenizer_checkpoint.to_dict(),
        }

    @property
    def request_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


__all__ = [
    "ACCELERATOR_CONTRACT",
    "HcuOptimizerStepRequest",
    "PHASE",
    "REQUEST_SCHEMA",
]

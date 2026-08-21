"""Typed contracts for one generated primary live-matrix request bundle."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Optional

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.matrix.live_run_config import LiveMatrixRunConfig
from robotactile_benchmark.matrix.manifest import MatrixManifest

PRIMARY_GENERATION_SEMANTIC_VERSION = "1.0"
PRIMARY_GENERATION_EVIDENCE_LEVEL = "request_generation_only_no_simulator_execution"
PRIMARY_RECEIPT_PATH = "primary_matrix_receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PrimaryMatrixGenerationError(ValueError):
    """Raised when a generated primary request bundle is invalid."""


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PrimaryMatrixGenerationError(f"{name} must be a lowercase SHA256")
    return value


def require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PrimaryMatrixGenerationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PrimaryMatrixGenerationSpec:
    """Inputs that determine one complete 14-by-5 live evaluation request."""

    task_id: str
    dataset_sha256: str
    base_system_id: str
    tactile_checkpoint_sha256: str
    no_touch_system_id: str
    no_touch_checkpoint_sha256: str
    stats_sha256: str
    encoder_sha256: str
    initial_seed: int
    exogenous_seed: int
    operator_seed_base: int
    fault_start_index: int
    restoration_index: int
    fault_stop_index: int
    max_control_cycles: int
    max_observation_steps: int
    wall_timeout_s: float
    upstream_root: Path
    runtime_root: Path
    official_act_artifact_root: Path
    rest_reference_artifact: Path
    act_device_name: str = "cuda:0"
    simulator_device: str = "cuda:0"
    matrix_id: Optional[str] = None
    semantic_version: str = PRIMARY_GENERATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in ("task_id", "base_system_id", "no_touch_system_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PrimaryMatrixGenerationError(f"{name} must be non-empty")
        for name in (
            "dataset_sha256",
            "tactile_checkpoint_sha256",
            "no_touch_checkpoint_sha256",
            "stats_sha256",
            "encoder_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        for name in (
            "initial_seed",
            "exogenous_seed",
            "operator_seed_base",
            "fault_start_index",
            "restoration_index",
            "fault_stop_index",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PrimaryMatrixGenerationError(
                    f"{name} must be a non-negative integer"
                )
        for name in ("max_control_cycles", "max_observation_steps"):
            object.__setattr__(
                self, name, require_positive_int(getattr(self, name), name)
            )
        if not (
            self.fault_start_index
            < self.restoration_index
            < self.fault_stop_index
            <= self.max_observation_steps
        ):
            raise PrimaryMatrixGenerationError(
                "fault window must satisfy start < restoration < stop <= observations"
            )
        if self.max_observation_steps != self.max_control_cycles + 1:
            raise PrimaryMatrixGenerationError(
                "max_observation_steps must equal max_control_cycles + 1"
            )
        if isinstance(self.wall_timeout_s, bool) or not isinstance(
            self.wall_timeout_s, (int, float)
        ):
            raise PrimaryMatrixGenerationError("wall_timeout_s must be real")
        timeout = float(self.wall_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise PrimaryMatrixGenerationError(
                "wall_timeout_s must be positive and finite"
            )
        object.__setattr__(self, "wall_timeout_s", timeout)
        for name in (
            "upstream_root",
            "runtime_root",
            "official_act_artifact_root",
            "rest_reference_artifact",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
            object.__setattr__(self, name, value.expanduser().absolute())
        for name in ("act_device_name", "simulator_device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PrimaryMatrixGenerationError(f"{name} must be non-empty")
        if self.matrix_id is not None and (
            not isinstance(self.matrix_id, str) or not self.matrix_id
        ):
            raise PrimaryMatrixGenerationError("matrix_id must be non-empty or null")
        if self.semantic_version != PRIMARY_GENERATION_SEMANTIC_VERSION:
            raise PrimaryMatrixGenerationError("generation spec version mismatch")

    @property
    def contract_sha256(self) -> str:
        """Hash only stable experiment identity, excluding machine paths."""

        return canonical_hash(
            {
                "task_id": self.task_id,
                "dataset_sha256": self.dataset_sha256,
                "base_system_id": self.base_system_id,
                "tactile_checkpoint_sha256": self.tactile_checkpoint_sha256,
                "no_touch_system_id": self.no_touch_system_id,
                "no_touch_checkpoint_sha256": self.no_touch_checkpoint_sha256,
                "stats_sha256": self.stats_sha256,
                "encoder_sha256": self.encoder_sha256,
                "initial_seed": self.initial_seed,
                "exogenous_seed": self.exogenous_seed,
                "operator_seed_base": self.operator_seed_base,
                "fault_window": [
                    self.fault_start_index,
                    self.restoration_index,
                    self.fault_stop_index,
                ],
                "budgets": [
                    self.max_control_cycles,
                    self.max_observation_steps,
                    self.wall_timeout_s,
                ],
                "semantic_version": self.semantic_version,
            }
        )


@dataclass(frozen=True)
class PrimaryMatrixGenerationReceipt:
    """Self-validating index over every non-receipt member in the bundle."""

    matrix_id: str
    matrix_manifest_sha256: str
    generation_contract_sha256: str
    pair_key: str
    task_id: str
    cell_count: int
    comparison_count: int
    fault_manifest_count: int
    rest_reference_artifact_root_sha256: str
    rest_reference_sha256: str
    tactile_config_sha256: str
    no_touch_config_sha256: str
    members: Mapping[str, str]
    evidence_level: str = PRIMARY_GENERATION_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = False
    task_success_claimed: bool = False
    semantic_version: str = PRIMARY_GENERATION_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.matrix_id, str) or not self.matrix_id:
            raise PrimaryMatrixGenerationError("matrix_id must be non-empty")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise PrimaryMatrixGenerationError("task_id must be non-empty")
        for name in (
            "matrix_manifest_sha256",
            "generation_contract_sha256",
            "pair_key",
            "rest_reference_artifact_root_sha256",
            "rest_reference_sha256",
            "tactile_config_sha256",
            "no_touch_config_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        expected_counts = {
            "cell_count": 142,
            "comparison_count": 70,
            "fault_manifest_count": 140,
        }
        for name, expected in expected_counts.items():
            if getattr(self, name) != expected:
                raise PrimaryMatrixGenerationError(
                    f"{name} must equal {expected} for the primary matrix"
                )
        if not isinstance(self.members, Mapping) or not self.members:
            raise PrimaryMatrixGenerationError("members must be a non-empty mapping")
        members: dict[str, str] = {}
        for path, digest in self.members.items():
            if not isinstance(path, str) or not path or path == PRIMARY_RECEIPT_PATH:
                raise PrimaryMatrixGenerationError("receipt member path is invalid")
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
                raise PrimaryMatrixGenerationError("receipt member path is unsafe")
            members[path] = require_sha256(digest, f"member {path}")
        if tuple(members) != tuple(sorted(members)):
            raise PrimaryMatrixGenerationError("receipt members must be path-sorted")
        object.__setattr__(self, "members", MappingProxyType(members))
        if self.evidence_level != PRIMARY_GENERATION_EVIDENCE_LEVEL:
            raise PrimaryMatrixGenerationError("generation evidence level mismatch")
        if self.simulator_execution_claimed is not False:
            raise PrimaryMatrixGenerationError(
                "request generation cannot claim simulator"
            )
        if self.task_success_claimed is not False:
            raise PrimaryMatrixGenerationError(
                "request generation cannot claim success"
            )
        if self.semantic_version != PRIMARY_GENERATION_SEMANTIC_VERSION:
            raise PrimaryMatrixGenerationError("generation receipt version mismatch")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix_id": self.matrix_id,
            "matrix_manifest_sha256": self.matrix_manifest_sha256,
            "generation_contract_sha256": self.generation_contract_sha256,
            "pair_key": self.pair_key,
            "task_id": self.task_id,
            "cell_count": self.cell_count,
            "comparison_count": self.comparison_count,
            "fault_manifest_count": self.fault_manifest_count,
            "rest_reference_artifact_root_sha256": (
                self.rest_reference_artifact_root_sha256
            ),
            "rest_reference_sha256": self.rest_reference_sha256,
            "tactile_config_sha256": self.tactile_config_sha256,
            "no_touch_config_sha256": self.no_touch_config_sha256,
            "members": dict(self.members),
            "evidence_level": self.evidence_level,
            "simulator_execution_claimed": self.simulator_execution_claimed,
            "task_success_claimed": self.task_success_claimed,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> PrimaryMatrixGenerationReceipt:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise PrimaryMatrixGenerationError("generation receipt fields mismatch")
        return cls(**value)


@dataclass(frozen=True)
class LoadedPrimaryMatrixGeneration:
    """Strictly reconstructed primary request bundle."""

    root: Path
    manifest: MatrixManifest
    run_config: LiveMatrixRunConfig
    receipt: PrimaryMatrixGenerationReceipt
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a pathlib.Path")
        if type(self.manifest) is not MatrixManifest:
            raise TypeError("manifest must be an exact MatrixManifest")
        if type(self.run_config) is not LiveMatrixRunConfig:
            raise TypeError("run_config must be an exact LiveMatrixRunConfig")
        if type(self.receipt) is not PrimaryMatrixGenerationReceipt:
            raise TypeError("receipt must be an exact generation receipt")
        object.__setattr__(
            self,
            "receipt_file_sha256",
            require_sha256(self.receipt_file_sha256, "receipt_file_sha256"),
        )

"""Typed contracts for generating one clean calibration execution request."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.execution.contracts import LiveUniVTACRunRequest
from robotactile_benchmark.rest_references import ReferenceSplit

CALIBRATION_REQUEST_SEMANTIC_VERSION = "1.0"
CALIBRATION_REQUEST_EVIDENCE_LEVEL = "request_generation_only_no_simulator_execution"
CALIBRATION_REQUEST_PATH = "request.json"
CALIBRATION_REQUEST_RECEIPT_PATH = "calibration_request_receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CalibrationRequestError(ValueError):
    """Raised when a calibration request bundle is not exact and reproducible."""


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CalibrationRequestError(f"{name} must be a lowercase SHA256")
    return value


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationRequestError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationRequestError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class CalibrationRequestSpec:
    """All machine-independent and runtime inputs for one clean trace request."""

    task_id: str
    dataset_split: ReferenceSplit
    split_manifest_sha256: str
    base_system_id: str
    checkpoint_sha256: str
    config_sha256: str
    initial_seed: int
    exogenous_seed: int
    max_control_cycles: int
    max_observation_steps: int
    wall_timeout_s: float
    upstream_root: Path
    runtime_dir: Path
    live_artifact_output_dir: Path
    act_device_name: str
    simulator_device: str
    launcher_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "base_system_id",
            "act_device_name",
            "simulator_device",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        split = (
            self.dataset_split
            if isinstance(self.dataset_split, ReferenceSplit)
            else ReferenceSplit(self.dataset_split)
        )
        if split is ReferenceSplit.SYNTHETIC_DEVELOPMENT:
            raise CalibrationRequestError(
                "production calibration cannot use synthetic data"
            )
        object.__setattr__(self, "dataset_split", split)
        for name in ("split_manifest_sha256", "checkpoint_sha256", "config_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("max_control_cycles", "max_observation_steps"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if isinstance(self.wall_timeout_s, bool) or not isinstance(
            self.wall_timeout_s, (int, float)
        ):
            raise CalibrationRequestError("wall_timeout_s must be a real number")
        timeout = float(self.wall_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise CalibrationRequestError("wall_timeout_s must be positive and finite")
        object.__setattr__(self, "wall_timeout_s", timeout)
        for name in ("upstream_root", "runtime_dir", "live_artifact_output_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")
            object.__setattr__(self, name, value.absolute())
        if not isinstance(self.launcher_args, Mapping):
            raise TypeError("launcher_args must be a mapping")
        object.__setattr__(self, "launcher_args", freeze_value(self.launcher_args))


@dataclass(frozen=True)
class CalibrationRequestReceipt:
    """Content binding for a generated request; it is not execution evidence."""

    request_sha256: str
    request_file_sha256: str
    trial_manifest_sha256: str
    task_id: str
    dataset_split: ReferenceSplit
    split_manifest_sha256: str
    base_system_manifest_sha256: str
    evidence_level: str = CALIBRATION_REQUEST_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = False
    semantic_version: str = CALIBRATION_REQUEST_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "request_file_sha256",
            "trial_manifest_sha256",
            "split_manifest_sha256",
            "base_system_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        if not isinstance(self.dataset_split, ReferenceSplit):
            object.__setattr__(
                self, "dataset_split", ReferenceSplit(self.dataset_split)
            )
        if self.dataset_split is ReferenceSplit.SYNTHETIC_DEVELOPMENT:
            raise CalibrationRequestError(
                "calibration receipt cannot use synthetic data"
            )
        if self.evidence_level != CALIBRATION_REQUEST_EVIDENCE_LEVEL:
            raise CalibrationRequestError("calibration request evidence level mismatch")
        if self.simulator_execution_claimed is not False:
            raise CalibrationRequestError("request generation cannot claim execution")
        if self.semantic_version != CALIBRATION_REQUEST_SEMANTIC_VERSION:
            raise CalibrationRequestError("calibration request version mismatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_sha256": self.request_sha256,
            "request_file_sha256": self.request_file_sha256,
            "trial_manifest_sha256": self.trial_manifest_sha256,
            "task_id": self.task_id,
            "dataset_split": self.dataset_split.value,
            "split_manifest_sha256": self.split_manifest_sha256,
            "base_system_manifest_sha256": self.base_system_manifest_sha256,
            "evidence_level": self.evidence_level,
            "simulator_execution_claimed": self.simulator_execution_claimed,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> CalibrationRequestReceipt:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise CalibrationRequestError("calibration request receipt fields mismatch")
        return cls(**value)


@dataclass(frozen=True)
class LoadedCalibrationRequest:
    request: LiveUniVTACRunRequest
    receipt: CalibrationRequestReceipt
    receipt_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.request) is not LiveUniVTACRunRequest:
            raise TypeError("request must be an exact LiveUniVTACRunRequest")
        if type(self.receipt) is not CalibrationRequestReceipt:
            raise TypeError("receipt must be exact CalibrationRequestReceipt")
        object.__setattr__(
            self,
            "receipt_file_sha256",
            _sha256(self.receipt_file_sha256, "receipt_file_sha256"),
        )


def frozen_receipt_mapping(receipt: CalibrationRequestReceipt) -> Mapping[str, object]:
    """Expose a read-only receipt mapping for small integrations."""

    return MappingProxyType(receipt.to_dict())

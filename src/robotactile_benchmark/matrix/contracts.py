"""Immutable contracts for benchmark matrix construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Mapping, Optional, cast

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.manifests import FaultManifest
from robotactile_benchmark.severity import severity_value
from robotactile_benchmark.trials import Condition, TrialManifest

MATRIX_SEMANTIC_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MatrixGridKind(str, Enum):
    """Registered benchmark grid shapes."""

    PRIMARY = "primary_14x5_blind"
    FOCUSED_PHASE = "focused_phase"
    FOCUSED_RESTORATION = "focused_restoration"


def require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def require_integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


@dataclass(frozen=True)
class MatrixGridPoint:
    """One requested fault instance and restoration boundary."""

    focus_id: str
    fault_manifest: FaultManifest
    restoration_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "focus_id", require_nonempty(self.focus_id, "focus_id")
        )
        if not isinstance(self.fault_manifest, FaultManifest):
            raise TypeError("fault_manifest must be a FaultManifest")
        index = require_integer(self.restoration_index, "restoration_index")
        if not self.fault_manifest.start_index < index < self.fault_manifest.stop_index:
            raise ValueError("restoration index must lie inside the fault window")
        object.__setattr__(self, "restoration_index", index)

    @property
    def sha256(self) -> str:
        return canonical_hash(
            {
                "focus_id": self.focus_id,
                "fault_manifest_sha256": self.fault_manifest.sha256,
                "restoration_index": self.restoration_index,
            }
        )


@dataclass(frozen=True)
class MatrixCellSpec:
    """One unique execution; comparison rows may share its content address."""

    trial: TrialManifest
    fault_manifest: Optional[FaultManifest]
    semantic_version: str = MATRIX_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if self.semantic_version != MATRIX_SEMANTIC_VERSION:
            raise ValueError("unsupported matrix cell semantic version")
        if not isinstance(self.trial, TrialManifest):
            raise TypeError("trial must be a TrialManifest")
        if self.fault_manifest is not None and not isinstance(
            self.fault_manifest, FaultManifest
        ):
            raise TypeError("fault_manifest must be a FaultManifest or None")
        faulted = self.trial.condition in {Condition.FAULTED, Condition.RESTORED}
        if faulted != (self.fault_manifest is not None):
            raise ValueError(
                "faulted/restored cells must carry exactly one fault manifest"
            )
        if (
            self.fault_manifest is not None
            and self.trial.fault_manifest_sha256 != self.fault_manifest.sha256
        ):
            raise ValueError("cell fault manifest does not match its trial")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def task(self) -> str:
        return self.trial.task

    @property
    def pair_key(self) -> str:
        return self.trial.pair_key

    @property
    def operator_id(self) -> Optional[str]:
        return None if self.fault_manifest is None else self.fault_manifest.operator_id

    @property
    def severity_level(self) -> Optional[int]:
        return (
            None if self.fault_manifest is None else self.fault_manifest.severity_level
        )

    @property
    def native_dose(self) -> Optional[object]:
        if self.fault_manifest is None:
            return None
        return cast(
            object,
            severity_value(
                self.fault_manifest.operator_id, self.fault_manifest.severity_level
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "trial": self.trial.to_dict(),
            "fault_manifest": (
                None if self.fault_manifest is None else self.fault_manifest.to_dict()
            ),
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> MatrixCellSpec:
        if not isinstance(value, Mapping) or set(value) != {
            "trial",
            "fault_manifest",
            "semantic_version",
        }:
            raise ValueError("matrix cell fields mismatch")
        trial = value["trial"]
        fault = value["fault_manifest"]
        if not isinstance(trial, Mapping):
            raise TypeError("matrix cell trial must be an object")
        if fault is not None and not isinstance(fault, Mapping):
            raise TypeError("matrix cell fault manifest must be an object or null")
        return cls(
            trial=TrialManifest.from_dict(trial),
            fault_manifest=None if fault is None else FaultManifest.from_dict(fault),
            semantic_version=require_nonempty(
                value["semantic_version"], "semantic_version"
            ),
        )


@dataclass(frozen=True)
class MatrixComparison:
    """Four-condition references for one scientific comparison row."""

    point_id: str
    focus_id: str
    operator_id: str
    severity_level: int
    native_dose: object
    restoration_index: int
    clean_cell_sha256: str
    faulted_cell_sha256: str
    no_touch_cell_sha256: str
    restored_cell_sha256: str

    def __post_init__(self) -> None:
        for name in ("point_id", "focus_id", "operator_id"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name))
        if self.operator_id not in CORE_OPERATOR_IDS:
            raise ValueError("comparison operator is not registered")
        severity = require_integer(self.severity_level, "severity_level", minimum=1)
        if severity > 5:
            raise ValueError("severity_level must be in [1, 5]")
        object.__setattr__(self, "severity_level", severity)
        expected_dose = severity_value(self.operator_id, severity)
        if (
            type(self.native_dose) is not type(expected_dose)
            or self.native_dose != expected_dose
        ):
            raise ValueError("comparison native dose does not match the registry")
        object.__setattr__(
            self,
            "restoration_index",
            require_integer(self.restoration_index, "restoration_index"),
        )
        for name in (
            "clean_cell_sha256",
            "faulted_cell_sha256",
            "no_touch_cell_sha256",
            "restored_cell_sha256",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))

    def to_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "focus_id": self.focus_id,
            "operator_id": self.operator_id,
            "severity_level": self.severity_level,
            "native_dose": self.native_dose,
            "restoration_index": self.restoration_index,
            "clean_cell_sha256": self.clean_cell_sha256,
            "faulted_cell_sha256": self.faulted_cell_sha256,
            "no_touch_cell_sha256": self.no_touch_cell_sha256,
            "restored_cell_sha256": self.restored_cell_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> MatrixComparison:
        fields = {
            "point_id",
            "focus_id",
            "operator_id",
            "severity_level",
            "native_dose",
            "restoration_index",
            "clean_cell_sha256",
            "faulted_cell_sha256",
            "no_touch_cell_sha256",
            "restored_cell_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("matrix comparison fields mismatch")
        return cls(**cast(Any, dict(value)))

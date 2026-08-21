"""Typed evaluator-side evidence for registered behavioral recovery signals."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_json
from robotactile_benchmark.metrics import recovery_lag
from robotactile_benchmark.reporting.contracts import (
    require_finite,
    require_integer,
    require_sha256,
)

RECOVERY_EVIDENCE_LEVEL = "unqualified_evaluator_recovery_signal_v1"
RECOVERY_EVIDENCE_SEMANTIC_VERSION = "1.0"
CLEAN_ENVELOPE_ALIGNMENT = "task_and_phase_matched"
POLICY_TASK_PROGRESS_SIGNAL_ID = "robotactile.policy.evaluator_task_progress.v1"
REGISTERED_RECOVERY_SIGNAL_IDS = frozenset({POLICY_TASK_PROGRESS_SIGNAL_ID})


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return value


def _optional_integer(value: object, name: str) -> Optional[int]:
    return None if value is None else require_integer(value, name)


@dataclass(frozen=True)
class RecoveryEvidenceBinding:
    """Immutable identity of one clean/restored recovery comparison."""

    clean_live_artifact_root_sha256: str
    restored_live_artifact_root_sha256: str
    task: str
    pair_key: str
    operator_id: str
    severity_level: int
    restoration_index: int

    def __post_init__(self) -> None:
        clean = require_sha256(
            self.clean_live_artifact_root_sha256,
            "clean_live_artifact_root_sha256",
        )
        restored = require_sha256(
            self.restored_live_artifact_root_sha256,
            "restored_live_artifact_root_sha256",
        )
        if clean == restored:
            raise ValueError("clean and restored artifact roots must differ")
        task = _require_nonempty(self.task, "task")
        pair_key = require_sha256(self.pair_key, "pair_key")
        operator = _require_nonempty(self.operator_id, "operator_id")
        if operator not in CORE_OPERATOR_IDS:
            raise ValueError("recovery operator_id must be a registered core operator")
        severity = require_integer(self.severity_level, "severity_level", minimum=1)
        if severity > 5:
            raise ValueError("severity_level must lie in [1, 5]")
        restoration = require_integer(self.restoration_index, "restoration_index")
        object.__setattr__(self, "clean_live_artifact_root_sha256", clean)
        object.__setattr__(self, "restored_live_artifact_root_sha256", restored)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "pair_key", pair_key)
        object.__setattr__(self, "operator_id", operator)
        object.__setattr__(self, "severity_level", severity)
        object.__setattr__(self, "restoration_index", restoration)

    def to_dict(self) -> dict[str, object]:
        return {
            "clean_live_artifact_root_sha256": (self.clean_live_artifact_root_sha256),
            "restored_live_artifact_root_sha256": (
                self.restored_live_artifact_root_sha256
            ),
            "task": self.task,
            "pair_key": self.pair_key,
            "operator_id": self.operator_id,
            "severity_level": self.severity_level,
            "restoration_index": self.restoration_index,
        }

    @classmethod
    def from_dict(cls, value: object) -> RecoveryEvidenceBinding:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("recovery evidence binding fields mismatch")
        return cls(
            clean_live_artifact_root_sha256=require_sha256(
                value["clean_live_artifact_root_sha256"],
                "clean_live_artifact_root_sha256",
            ),
            restored_live_artifact_root_sha256=require_sha256(
                value["restored_live_artifact_root_sha256"],
                "restored_live_artifact_root_sha256",
            ),
            task=_require_nonempty(value["task"], "task"),
            pair_key=require_sha256(value["pair_key"], "pair_key"),
            operator_id=_require_nonempty(value["operator_id"], "operator_id"),
            severity_level=require_integer(
                value["severity_level"], "severity_level", minimum=1
            ),
            restoration_index=require_integer(
                value["restoration_index"], "restoration_index"
            ),
        )


@dataclass(frozen=True)
class RecoveryEvidence:
    """Canonical quality trace and independently checkable recovery result."""

    binding: RecoveryEvidenceBinding
    signal_id: str
    clean_envelope: Tuple[float, ...]
    restored_quality: Tuple[float, ...]
    tolerance: float
    consecutive_steps: int
    recovery_lag_steps: Optional[int]
    clean_envelope_alignment: str = CLEAN_ENVELOPE_ALIGNMENT
    evidence_level: str = RECOVERY_EVIDENCE_LEVEL
    semantic_version: str = RECOVERY_EVIDENCE_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.binding, RecoveryEvidenceBinding):
            raise TypeError("binding must be a RecoveryEvidenceBinding")
        signal_id = _require_nonempty(self.signal_id, "signal_id")
        if signal_id not in REGISTERED_RECOVERY_SIGNAL_IDS:
            raise ValueError("signal_id is not a registered recovery signal")
        clean = tuple(
            require_finite(item, f"clean_envelope[{index}]")
            for index, item in enumerate(self.clean_envelope)
        )
        restored = tuple(
            require_finite(item, f"restored_quality[{index}]")
            for index, item in enumerate(self.restored_quality)
        )
        if not clean or len(clean) != len(restored):
            raise ValueError("clean envelope and restored quality must match")
        tolerance = require_finite(self.tolerance, "tolerance")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        consecutive = require_integer(
            self.consecutive_steps, "consecutive_steps", minimum=1
        )
        remaining = len(clean) - self.binding.restoration_index
        if remaining <= 0 or consecutive > remaining:
            raise ValueError("criterion requires an evaluable post-restoration window")
        cached_lag = _optional_integer(self.recovery_lag_steps, "recovery_lag_steps")
        recalculated = recovery_lag(
            restored,
            clean,
            self.binding.restoration_index,
            tolerance,
            consecutive,
        )
        if cached_lag != recalculated:
            raise ValueError("cached recovery lag is stale or inconsistent")
        if self.clean_envelope_alignment != CLEAN_ENVELOPE_ALIGNMENT:
            raise ValueError("clean envelope must be task- and phase-matched")
        if self.evidence_level != RECOVERY_EVIDENCE_LEVEL:
            raise ValueError("unsupported recovery evidence level")
        if self.semantic_version != RECOVERY_EVIDENCE_SEMANTIC_VERSION:
            raise ValueError("unsupported recovery evidence semantic version")
        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "clean_envelope", clean)
        object.__setattr__(self, "restored_quality", restored)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "consecutive_steps", consecutive)
        object.__setattr__(self, "recovery_lag_steps", recalculated)

    @classmethod
    def evaluate(
        cls,
        *,
        binding: RecoveryEvidenceBinding,
        signal_id: str,
        clean_envelope: Sequence[float],
        restored_quality: Sequence[float],
        tolerance: float,
        consecutive_steps: int,
    ) -> RecoveryEvidence:
        """Evaluate once while retaining all inputs needed for strict replay."""

        lag = recovery_lag(
            restored_quality,
            clean_envelope,
            binding.restoration_index,
            tolerance,
            consecutive_steps,
        )
        return cls(
            binding=binding,
            signal_id=signal_id,
            clean_envelope=tuple(clean_envelope),
            restored_quality=tuple(restored_quality),
            tolerance=tolerance,
            consecutive_steps=consecutive_steps,
            recovery_lag_steps=lag,
        )

    @property
    def sha256(self) -> str:
        """Hash the exact canonical file bytes consumed by report provenance."""

        raw = (canonical_json(self.to_dict()) + "\n").encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "binding": self.binding.to_dict(),
            "signal_id": self.signal_id,
            "clean_envelope": list(self.clean_envelope),
            "restored_quality": list(self.restored_quality),
            "tolerance": self.tolerance,
            "consecutive_steps": self.consecutive_steps,
            "recovery_lag_steps": self.recovery_lag_steps,
            "clean_envelope_alignment": self.clean_envelope_alignment,
            "evidence_level": self.evidence_level,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RecoveryEvidence:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("recovery evidence fields mismatch")
        clean = _require_sequence(value["clean_envelope"], "clean_envelope")
        restored = _require_sequence(value["restored_quality"], "restored_quality")
        return cls(
            binding=RecoveryEvidenceBinding.from_dict(value["binding"]),
            signal_id=_require_nonempty(value["signal_id"], "signal_id"),
            clean_envelope=tuple(
                require_finite(item, f"clean_envelope[{index}]")
                for index, item in enumerate(clean)
            ),
            restored_quality=tuple(
                require_finite(item, f"restored_quality[{index}]")
                for index, item in enumerate(restored)
            ),
            tolerance=require_finite(value["tolerance"], "tolerance"),
            consecutive_steps=require_integer(
                value["consecutive_steps"], "consecutive_steps", minimum=1
            ),
            recovery_lag_steps=_optional_integer(
                value["recovery_lag_steps"], "recovery_lag_steps"
            ),
            clean_envelope_alignment=_require_nonempty(
                value["clean_envelope_alignment"], "clean_envelope_alignment"
            ),
            evidence_level=_require_nonempty(value["evidence_level"], "evidence_level"),
            semantic_version=_require_nonempty(
                value["semantic_version"], "semantic_version"
            ),
        )

"""Immutable inputs for result aggregation and paper reporting."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Tuple

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.trials import Condition, TerminalStatus

REPORTING_SEMANTIC_VERSION = "1.0"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCORE_INELIGIBLE = frozenset(
    {TerminalStatus.UNSUPPORTED_CONTRACT, TerminalStatus.VALIDATOR_REJECTED}
)


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def require_integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def operator_axis(operator_id: str) -> str:
    """Return the paper reporting axis encoded by a canonical operator ID."""

    if operator_id not in CORE_OPERATOR_IDS:
        raise ValueError(f"unknown core operator: {operator_id}")
    return {
        "A": "availability",
        "F": "fidelity",
        "T": "temporal",
        "C": "context",
    }[operator_id[0]]


@dataclass(frozen=True)
class ReportingSpec:
    """Frozen statistical and eligibility choices for one system report."""

    system_id: str
    matched_control_qualified: bool
    minimum_clean_gain: float
    primary_operator_ids: Tuple[str, ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    primary_severity_levels: Tuple[int, ...] = (1, 2, 3, 4, 5)
    confidence_level: float = 0.95
    semantic_version: str = REPORTING_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.system_id, str) or not self.system_id:
            raise ValueError("system_id must be a non-empty string")
        if type(self.matched_control_qualified) is not bool:
            raise TypeError("matched_control_qualified must be bool")
        gain = require_finite(self.minimum_clean_gain, "minimum_clean_gain")
        if gain <= 0.0:
            raise ValueError("minimum_clean_gain must be positive")
        operators = tuple(self.primary_operator_ids)
        if not operators or len(operators) != len(set(operators)):
            raise ValueError("primary_operator_ids must be non-empty and unique")
        unknown = set(operators) - CORE_OPERATOR_IDS
        if unknown:
            raise ValueError(f"unknown primary operators: {sorted(unknown)}")
        severity_levels = tuple(self.primary_severity_levels)
        if not severity_levels or len(severity_levels) != len(set(severity_levels)):
            raise ValueError("primary_severity_levels must be non-empty and unique")
        severity_levels = tuple(
            require_integer(level, "primary severity level", minimum=1)
            for level in severity_levels
        )
        if any(level > 5 for level in severity_levels):
            raise ValueError("primary severity levels must lie in [1, 5]")
        resamples = require_integer(
            self.bootstrap_resamples, "bootstrap_resamples", minimum=1
        )
        seed = require_integer(self.bootstrap_seed, "bootstrap_seed")
        confidence = require_finite(self.confidence_level, "confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if self.semantic_version != REPORTING_SEMANTIC_VERSION:
            raise ValueError("unsupported reporting semantic version")
        object.__setattr__(self, "minimum_clean_gain", gain)
        object.__setattr__(self, "primary_operator_ids", tuple(sorted(operators)))
        object.__setattr__(
            self, "primary_severity_levels", tuple(sorted(severity_levels))
        )
        object.__setattr__(self, "bootstrap_resamples", resamples)
        object.__setattr__(self, "bootstrap_seed", seed)
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id,
            "matched_control_qualified": self.matched_control_qualified,
            "minimum_clean_gain": self.minimum_clean_gain,
            "primary_operator_ids": list(self.primary_operator_ids),
            "primary_severity_levels": list(self.primary_severity_levels),
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReportingSpec:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("reporting spec fields mismatch")
        operators = value["primary_operator_ids"]
        severity_levels = value["primary_severity_levels"]
        if not isinstance(operators, list) or not isinstance(severity_levels, list):
            raise TypeError("primary operator and severity inventories must be lists")
        return cls(
            system_id=value["system_id"],
            matched_control_qualified=value["matched_control_qualified"],
            minimum_clean_gain=value["minimum_clean_gain"],
            primary_operator_ids=tuple(operators),
            bootstrap_resamples=value["bootstrap_resamples"],
            bootstrap_seed=value["bootstrap_seed"],
            primary_severity_levels=tuple(severity_levels),
            confidence_level=value["confidence_level"],
            semantic_version=value["semantic_version"],
        )


@dataclass(frozen=True)
class OutcomeRecord:
    """One requested matrix cell reduced to its verified scoring outcome."""

    system_id: str
    task: str
    pair_key: str
    condition: Condition
    terminal_status: TerminalStatus
    score_eligible: bool
    score_success: Optional[bool]
    source_root_sha256: str
    operator_id: Optional[str] = None
    severity_level: Optional[int] = None
    recovery_eligible: bool = False
    recovery_lag_steps: Optional[int] = None
    semantic_version: str = REPORTING_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        for name in ("system_id", "task"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "pair_key", require_sha256(self.pair_key, "pair_key"))
        object.__setattr__(
            self,
            "source_root_sha256",
            require_sha256(self.source_root_sha256, "source_root_sha256"),
        )
        if type(self.condition) is not Condition:
            raise TypeError("condition must be an exact Condition")
        if type(self.terminal_status) is not TerminalStatus:
            raise TypeError("terminal_status must be an exact TerminalStatus")
        if type(self.score_eligible) is not bool:
            raise TypeError("score_eligible must be bool")
        if self.score_eligible:
            if type(self.score_success) is not bool:
                raise TypeError("eligible outcomes require an exact bool score")
            if self.terminal_status in _SCORE_INELIGIBLE:
                raise ValueError("ineligible terminal status cannot be scored")
        elif self.score_success is not None:
            raise ValueError("ineligible outcomes cannot contain a success score")
        elif self.terminal_status not in _SCORE_INELIGIBLE:
            raise ValueError("unscored outcome must preserve an ineligible status")
        fault_condition = self.condition in {Condition.FAULTED, Condition.RESTORED}
        if fault_condition:
            if self.operator_id not in CORE_OPERATOR_IDS:
                raise ValueError("faulted/restored outcome requires a core operator")
            severity = require_integer(self.severity_level, "severity_level", minimum=1)
            if severity > 5:
                raise ValueError("severity_level must be in [1, 5]")
            object.__setattr__(self, "severity_level", severity)
        elif self.operator_id is not None or self.severity_level is not None:
            raise ValueError("baseline outcomes cannot include fault metadata")
        if type(self.recovery_eligible) is not bool:
            raise TypeError("recovery_eligible must be bool")
        if self.recovery_eligible and self.condition is not Condition.RESTORED:
            raise ValueError("recovery is only defined for restored outcomes")
        if self.recovery_lag_steps is not None:
            if not self.recovery_eligible:
                raise ValueError("recovery lag requires an eligible recovery outcome")
            object.__setattr__(
                self,
                "recovery_lag_steps",
                require_integer(self.recovery_lag_steps, "recovery_lag_steps"),
            )
        if self.semantic_version != REPORTING_SEMANTIC_VERSION:
            raise ValueError("unsupported outcome semantic version")

    @property
    def axis(self) -> Optional[str]:
        return None if self.operator_id is None else operator_axis(self.operator_id)

    @property
    def cell_key(self) -> tuple[object, ...]:
        return (
            self.system_id,
            self.task,
            self.pair_key,
            self.condition.value,
            self.operator_id,
            self.severity_level,
        )

    @property
    def sha256(self) -> str:
        return canonical_hash(self)

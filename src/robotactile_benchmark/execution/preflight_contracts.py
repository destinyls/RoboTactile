"""Immutable contracts for a no-allocation live deployment preflight."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Tuple, cast

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.trials import Condition

LIVE_PREFLIGHT_SEMANTIC_VERSION = "2.0"
LIVE_PREFLIGHT_EVIDENCE_LEVEL = "live_preflight_no_simulator_execution_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LivePreflightError(ValueError):
    """A live preflight input or receipt violates its frozen contract."""


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LivePreflightError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    normalized = _nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise LivePreflightError(f"{name} must be a lowercase SHA256")
    return normalized


@dataclass(frozen=True)
class LivePreflightCheck:
    """One deterministic readiness gate and its path-free evidence."""

    check_id: str
    passed: bool
    evidence: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", _nonempty(self.check_id, "check_id"))
        if type(self.passed) is not bool:
            raise TypeError("preflight check passed must be bool")
        if not isinstance(self.evidence, Mapping) or not self.evidence:
            raise LivePreflightError("preflight check evidence must be non-empty")
        normalized: dict[str, str] = {}
        for key, value in self.evidence.items():
            normalized[_nonempty(key, "evidence key")] = _nonempty(
                value, "evidence value"
            )
        object.__setattr__(self, "evidence", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "evidence": dict(self.evidence),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: object) -> "LivePreflightCheck":
        fields = {"check_id", "evidence", "passed"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise LivePreflightError("preflight check fields mismatch")
        return cls(
            check_id=value["check_id"],
            passed=value["passed"],
            evidence=cast(Mapping[str, str], value["evidence"]),
        )


@dataclass(frozen=True)
class LivePreflightReceipt:
    """Self-validating evidence that deployment inputs were checked only."""

    request_content_sha256: str
    task_id: str
    condition: str
    policy_kind: str
    checks: Tuple[LivePreflightCheck, ...]
    passed: bool
    evidence_level: str = LIVE_PREFLIGHT_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = False
    simulator_qualification_claimed: bool = False
    semantic_version: str = LIVE_PREFLIGHT_SEMANTIC_VERSION
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_content_sha256",
            _sha256(self.request_content_sha256, "request_content_sha256"),
        )
        for name in ("task_id", "policy_kind"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        condition = _nonempty(self.condition, "condition")
        try:
            object.__setattr__(self, "condition", Condition(condition).value)
        except ValueError as error:
            raise LivePreflightError("preflight condition is invalid") from error
        checks = tuple(self.checks)
        if not checks or any(type(item) is not LivePreflightCheck for item in checks):
            raise LivePreflightError("checks must contain exact preflight checks")
        ids = tuple(item.check_id for item in checks)
        if len(ids) != len(set(ids)):
            raise LivePreflightError("preflight check ids must be unique")
        object.__setattr__(self, "checks", checks)
        derived = all(item.passed for item in checks)
        if type(self.passed) is not bool or self.passed is not derived:
            raise LivePreflightError("preflight passed flag disagrees with checks")
        if self.evidence_level != LIVE_PREFLIGHT_EVIDENCE_LEVEL:
            raise LivePreflightError("preflight evidence level mismatch")
        if (
            self.simulator_execution_claimed is not False
            or self.simulator_qualification_claimed is not False
        ):
            raise LivePreflightError("preflight cannot claim simulator execution")
        if self.semantic_version != LIVE_PREFLIGHT_SEMANTIC_VERSION:
            raise LivePreflightError("preflight semantic version mismatch")
        object.__setattr__(self, "receipt_sha256", canonical_hash(self._unsigned()))

    def _unsigned(self) -> dict[str, object]:
        return {
            "checks": [item.to_dict() for item in self.checks],
            "condition": self.condition,
            "evidence_level": self.evidence_level,
            "passed": self.passed,
            "policy_kind": self.policy_kind,
            "request_content_sha256": self.request_content_sha256,
            "semantic_version": self.semantic_version,
            "simulator_execution_claimed": self.simulator_execution_claimed,
            "simulator_qualification_claimed": self.simulator_qualification_claimed,
            "task_id": self.task_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "LivePreflightReceipt":
        if not isinstance(value, Mapping):
            raise LivePreflightError("preflight receipt must be an object")
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields:
            raise LivePreflightError("preflight receipt fields mismatch")
        checks = value["checks"]
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            raise LivePreflightError("preflight receipt checks must be a sequence")
        supplied = _sha256(value["receipt_sha256"], "receipt_sha256")
        kwargs = dict(value)
        kwargs.pop("receipt_sha256")
        kwargs["checks"] = tuple(LivePreflightCheck.from_dict(item) for item in checks)
        receipt = cls(**cast(dict[str, Any], kwargs))
        if receipt.receipt_sha256 != supplied:
            raise LivePreflightError("preflight receipt hash mismatch")
        return receipt


__all__ = [
    "LIVE_PREFLIGHT_EVIDENCE_LEVEL",
    "LIVE_PREFLIGHT_SEMANTIC_VERSION",
    "LivePreflightCheck",
    "LivePreflightError",
    "LivePreflightReceipt",
]

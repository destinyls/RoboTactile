"""Immutable receipts for the repo-contained deployment workspace."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Tuple, cast

from robotactile_benchmark.contracts import canonical_hash

DEPLOYMENT_LAYOUT_ID = "robotactile_repo_contained_v1"
DEPLOYMENT_LAYOUT_SEMANTIC_VERSION = "1.0"
DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL = "deployment_layout_initialization_only_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DeploymentLayoutError(ValueError):
    """A deployment root, directory, or receipt violates the frozen contract."""


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DeploymentLayoutError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    normalized = _nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise DeploymentLayoutError(f"{name} must be a lowercase SHA256")
    return normalized


@dataclass(frozen=True)
class DeploymentLayoutReceipt:
    """Self-validating evidence for directory creation only."""

    root_path_sha256: str
    directories: Tuple[str, ...]
    layout_id: str = DEPLOYMENT_LAYOUT_ID
    evidence_level: str = DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL
    simulator_execution_claimed: bool = False
    task_execution_claimed: bool = False
    semantic_version: str = DEPLOYMENT_LAYOUT_SEMANTIC_VERSION
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_path_sha256",
            _sha256(self.root_path_sha256, "root_path_sha256"),
        )
        directories = tuple(self.directories)
        if (
            not directories
            or len(directories) != len(set(directories))
            or any(
                not isinstance(item, str)
                or not item
                or item.startswith("/")
                or item.startswith("../")
                or "//" in item
                for item in directories
            )
        ):
            raise DeploymentLayoutError(
                "directories must be unique safe relative paths"
            )
        object.__setattr__(self, "directories", directories)
        if self.layout_id != DEPLOYMENT_LAYOUT_ID:
            raise DeploymentLayoutError("deployment layout id mismatch")
        if self.evidence_level != DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL:
            raise DeploymentLayoutError("deployment evidence level mismatch")
        if (
            self.simulator_execution_claimed is not False
            or self.task_execution_claimed is not False
        ):
            raise DeploymentLayoutError("layout receipt cannot claim execution")
        if self.semantic_version != DEPLOYMENT_LAYOUT_SEMANTIC_VERSION:
            raise DeploymentLayoutError("deployment layout version mismatch")
        object.__setattr__(self, "receipt_sha256", canonical_hash(self._unsigned()))

    def _unsigned(self) -> dict[str, object]:
        return {
            "directories": list(self.directories),
            "evidence_level": self.evidence_level,
            "layout_id": self.layout_id,
            "root_path_sha256": self.root_path_sha256,
            "semantic_version": self.semantic_version,
            "simulator_execution_claimed": self.simulator_execution_claimed,
            "task_execution_claimed": self.task_execution_claimed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: object) -> "DeploymentLayoutReceipt":
        if not isinstance(value, Mapping):
            raise DeploymentLayoutError("deployment receipt must be an object")
        if set(value) != set(cls.__dataclass_fields__):
            raise DeploymentLayoutError("deployment receipt fields mismatch")
        directories = value["directories"]
        if not isinstance(directories, Sequence) or isinstance(
            directories, (str, bytes)
        ):
            raise DeploymentLayoutError("directories must be a sequence")
        supplied = _sha256(value["receipt_sha256"], "receipt_sha256")
        kwargs = dict(value)
        kwargs.pop("receipt_sha256")
        kwargs["directories"] = tuple(directories)
        receipt = cls(**cast(dict[str, Any], kwargs))
        if receipt.receipt_sha256 != supplied:
            raise DeploymentLayoutError("deployment receipt hash mismatch")
        return receipt


__all__ = [
    "DEPLOYMENT_LAYOUT_EVIDENCE_LEVEL",
    "DEPLOYMENT_LAYOUT_ID",
    "DEPLOYMENT_LAYOUT_SEMANTIC_VERSION",
    "DeploymentLayoutError",
    "DeploymentLayoutReceipt",
]

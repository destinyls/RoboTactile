"""Immutable contracts for N0-TWAM Clean/Faulted robustness campaigns."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Tuple, cast

from robotactile_benchmark.constants import (
    CORE_OPERATOR_IDS,
    SEVERITY_REGISTRY_ID,
    SUPPORTED_SEVERITY_REGISTRY_IDS,
    operator_requires_rest_reference,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LivePolicyKind
from robotactile_benchmark.trials import Condition

from .cell_contracts import N0FaultCampaignCellSpec, N0UnsupportedContractSpec
from .contract_primitives import (
    N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION,
    N0_FAULT_CAMPAIGN_SEMANTIC_VERSION,
    N0_FAULT_GENERATION_EVIDENCE_LEVEL,
    N0_FAULT_TEMPLATE_SEED_DERIVATION,
    N0_SUPPORTED_OPERATOR_IDS,
    N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION,
    N0_UNSUPPORTED_OPERATOR_IDS,
    N0FaultCampaignError,
    N0FaultCellDisposition,
    require_identifier,
    require_integer,
    require_nonempty,
    require_relative_path,
    require_sha256,
)

__all__ = [
    "N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION",
    "N0_FAULT_CAMPAIGN_SEMANTIC_VERSION",
    "N0_FAULT_GENERATION_EVIDENCE_LEVEL",
    "N0_FAULT_TEMPLATE_SEED_DERIVATION",
    "N0_SUPPORTED_OPERATOR_IDS",
    "N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION",
    "N0_UNSUPPORTED_OPERATOR_IDS",
    "N0FaultCampaignCellSpec",
    "N0FaultCampaignError",
    "N0FaultCampaignManifest",
    "N0FaultCellDisposition",
    "N0UnsupportedContractSpec",
    "require_identifier",
    "require_integer",
    "require_nonempty",
    "require_relative_path",
    "require_sha256",
]


@dataclass(frozen=True)
class N0FaultCampaignManifest:
    """Frozen request-generation grid for one N0 fault campaign."""

    campaign_id: str
    policy_kind: LivePolicyKind
    operator_ids: Tuple[str, ...]
    severity_levels: Tuple[int, ...]
    operator_template_seed_derivation: str
    pair_count: int
    cell_count: int
    live_request_count: int
    unsupported_contract_count: int
    rest_reference_bindings: Mapping[str, Mapping[str, str]]
    cells: Tuple[N0FaultCampaignCellSpec, ...]
    severity_registry: str = SEVERITY_REGISTRY_ID
    semantic_version: str = N0_FAULT_CAMPAIGN_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", require_identifier(self.campaign_id, "campaign_id")
        )
        kind = (
            self.policy_kind
            if isinstance(self.policy_kind, LivePolicyKind)
            else LivePolicyKind(self.policy_kind)
        )
        if kind is not LivePolicyKind.N0:
            raise N0FaultCampaignError("fault campaign policy must be N0")
        object.__setattr__(self, "policy_kind", kind)
        operators = tuple(self.operator_ids)
        levels = tuple(self.severity_levels)
        self._validate_axes(operators, levels)
        cells = tuple(self.cells)
        self._validate_cells(cells, operators, levels)
        if self.severity_registry not in SUPPORTED_SEVERITY_REGISTRY_IDS:
            raise N0FaultCampaignError("unsupported campaign severity registry")
        bindings = self._freeze_bindings(self.rest_reference_bindings)
        required_rest_tasks = {
            cell.task
            for cell in cells
            if cell.operator_id is not None
            and operator_requires_rest_reference(
                cell.operator_id,
                severity_registry=self.severity_registry,
            )
        }
        if set(bindings) != required_rest_tasks:
            raise N0FaultCampaignError(
                "rest-reference bindings do not exactly cover required tasks"
            )
        object.__setattr__(self, "operator_ids", operators)
        object.__setattr__(self, "severity_levels", levels)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "rest_reference_bindings", bindings)
        if self.semantic_version != N0_FAULT_CAMPAIGN_SEMANTIC_VERSION:
            raise N0FaultCampaignError("campaign semantic version mismatch")

    def _validate_axes(
        self, operators: Tuple[str, ...], levels: Tuple[int, ...]
    ) -> None:
        if (
            operators != tuple(sorted(set(operators)))
            or not operators
            or set(operators) - CORE_OPERATOR_IDS
        ):
            raise N0FaultCampaignError(
                "operator_ids must be sorted, unique, and canonical"
            )
        if (
            levels != tuple(sorted(set(levels)))
            or not levels
            or any(level not in range(1, 6) for level in levels)
        ):
            raise N0FaultCampaignError("severity_levels must be sorted unique levels")
        if self.operator_template_seed_derivation != N0_FAULT_TEMPLATE_SEED_DERIVATION:
            raise N0FaultCampaignError("template seed derivation mismatch")

    def _validate_cells(
        self,
        cells: Tuple[N0FaultCampaignCellSpec, ...],
        operators: Tuple[str, ...],
        levels: Tuple[int, ...],
    ) -> None:
        if tuple(cell.ordinal for cell in cells) != tuple(range(len(cells))):
            raise N0FaultCampaignError("campaign cell ordinals must be dense")
        counts_are_ints = all(
            type(getattr(self, name)) is int
            for name in (
                "pair_count",
                "cell_count",
                "live_request_count",
                "unsupported_contract_count",
            )
        )
        if (
            not counts_are_ints
            or self.cell_count != len(cells)
            or self.live_request_count
            != sum(
                cell.disposition is N0FaultCellDisposition.LIVE_REQUEST
                for cell in cells
            )
            or self.unsupported_contract_count
            != sum(
                cell.disposition is N0FaultCellDisposition.UNSUPPORTED_CONTRACT
                for cell in cells
            )
        ):
            raise N0FaultCampaignError("campaign counts disagree with cells")
        pair_keys = {cell.pair_key for cell in cells}
        if self.pair_count != len(pair_keys) or self.pair_count < 1:
            raise N0FaultCampaignError("campaign pair_count mismatch")
        expected_per_pair = 1 + len(operators) * len(levels)
        if Counter(cell.pair_key for cell in cells) != Counter(
            {key: expected_per_pair for key in pair_keys}
        ):
            raise N0FaultCampaignError(
                "each pair must cover one Clean plus the requested grid"
            )
        self._validate_pair_grids(cells, operators, levels)

    @staticmethod
    def _freeze_bindings(
        value: Mapping[str, Mapping[str, str]],
    ) -> Mapping[str, Mapping[str, str]]:
        normalized: dict[str, Mapping[str, str]] = {}
        for task, binding in sorted(value.items()):
            if set(binding) != {
                "artifact_relpath",
                "artifact_root_sha256",
                "rest_reference_sha256",
            }:
                raise N0FaultCampaignError("rest-reference binding fields mismatch")
            normalized[require_nonempty(task, "rest-reference task")] = (
                MappingProxyType(
                    {
                        "artifact_relpath": require_relative_path(
                            binding["artifact_relpath"],
                            "artifact_relpath",
                            "rest_references",
                        ),
                        "artifact_root_sha256": require_sha256(
                            binding["artifact_root_sha256"], "artifact_root_sha256"
                        ),
                        "rest_reference_sha256": require_sha256(
                            binding["rest_reference_sha256"],
                            "rest_reference_sha256",
                        ),
                    }
                )
            )
        return MappingProxyType(normalized)

    @staticmethod
    def _validate_pair_grids(
        cells: Tuple[N0FaultCampaignCellSpec, ...],
        operators: Tuple[str, ...],
        levels: Tuple[int, ...],
    ) -> None:
        for pair_key in {cell.pair_key for cell in cells}:
            selected = [cell for cell in cells if cell.pair_key == pair_key]
            clean = [cell for cell in selected if cell.condition is Condition.CLEAN]
            if len(clean) != 1:
                raise N0FaultCampaignError("each pair requires exactly one Clean cell")
            expected = {(operator, level) for operator in operators for level in levels}
            actual = {
                (cast(str, cell.operator_id), cast(int, cell.severity_level))
                for cell in selected
                if cell.condition is Condition.FAULTED
            }
            if actual != expected:
                raise N0FaultCampaignError("fault grid is incomplete or duplicated")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "policy_kind": self.policy_kind.value,
            "operator_ids": list(self.operator_ids),
            "severity_levels": list(self.severity_levels),
            "operator_template_seed_derivation": self.operator_template_seed_derivation,
            "pair_count": self.pair_count,
            "cell_count": self.cell_count,
            "live_request_count": self.live_request_count,
            "unsupported_contract_count": self.unsupported_contract_count,
            "rest_reference_bindings": {
                task: dict(binding)
                for task, binding in self.rest_reference_bindings.items()
            },
            "cells": [cell.to_dict() for cell in self.cells],
            "severity_registry": self.severity_registry,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> N0FaultCampaignManifest:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise N0FaultCampaignError("campaign manifest fields mismatch")
        document = dict(value)
        cells = document["cells"]
        if not isinstance(cells, list):
            raise N0FaultCampaignError("campaign cells must be a list")
        document["cells"] = tuple(
            N0FaultCampaignCellSpec.from_dict(cell) for cell in cells
        )
        return cls(**cast(Any, document))

"""Strict materialized matrix manifest and cross-link validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.manifests import Observability
from robotactile_benchmark.matrix.contracts import (
    MATRIX_SEMANTIC_VERSION,
    MatrixCellSpec,
    MatrixComparison,
    MatrixGridKind,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.trials import Condition


@dataclass(frozen=True)
class MatrixManifest:
    """Fully materialized matrix with unique cells and shared baseline links."""

    matrix_id: str
    kind: MatrixGridKind
    pair_key: str
    cells: Tuple[MatrixCellSpec, ...]
    comparisons: Tuple[MatrixComparison, ...]
    semantic_version: str = MATRIX_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "matrix_id", require_nonempty(self.matrix_id, "matrix_id")
        )
        kind = (
            self.kind
            if isinstance(self.kind, MatrixGridKind)
            else MatrixGridKind(self.kind)
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "pair_key", require_sha256(self.pair_key, "pair_key"))
        if self.semantic_version != MATRIX_SEMANTIC_VERSION:
            raise ValueError("unsupported matrix manifest semantic version")
        cells = tuple(self.cells)
        comparisons = tuple(self.comparisons)
        self._validate_inventory(cells, comparisons)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "comparisons", comparisons)

    def _validate_inventory(
        self,
        cells: Tuple[MatrixCellSpec, ...],
        comparisons: Tuple[MatrixComparison, ...],
    ) -> None:
        if not cells or not comparisons:
            raise ValueError("matrix manifest must contain cells and comparisons")
        if any(not isinstance(item, MatrixCellSpec) for item in cells):
            raise TypeError("cells must contain MatrixCellSpec values")
        if any(not isinstance(item, MatrixComparison) for item in comparisons):
            raise TypeError("comparisons must contain MatrixComparison values")
        addresses = tuple(cell.sha256 for cell in cells)
        if len(set(addresses)) != len(addresses):
            raise ValueError("matrix cells must be content-address unique")
        if {cell.pair_key for cell in cells} != {self.pair_key}:
            raise ValueError("matrix cells must share the frozen pair key")
        if any(
            cell.fault_manifest is not None
            and cell.fault_manifest.observability is not Observability.BLIND
            for cell in cells
        ):
            raise ValueError("matrix fault manifests must all be blind")
        lookup = dict(zip(addresses, cells))
        if sum(cell.trial.condition is Condition.CLEAN for cell in cells) != 1:
            raise ValueError("matrix must contain exactly one clean baseline")
        if sum(cell.trial.condition is Condition.NO_TOUCH for cell in cells) != 1:
            raise ValueError("matrix must contain exactly one no-touch baseline")
        if len({row.point_id for row in comparisons}) != len(comparisons):
            raise ValueError("matrix comparison point IDs must be unique")
        referenced: set[str] = set()
        for row in comparisons:
            referenced.update(self._validate_row(row, lookup))
        if referenced != set(addresses):
            raise ValueError("matrix contains unreferenced cells")
        if self.kind is MatrixGridKind.PRIMARY:
            expected = {
                (operator, level)
                for operator in CORE_OPERATOR_IDS
                for level in range(1, 6)
            }
            actual = {(row.operator_id, row.severity_level) for row in comparisons}
            if len(comparisons) != 70 or actual != expected:
                raise ValueError(
                    "primary matrix must contain the complete blind 14 x 5 grid"
                )

    @staticmethod
    def _validate_row(
        row: MatrixComparison, lookup: Mapping[str, MatrixCellSpec]
    ) -> Tuple[str, ...]:
        references = {
            Condition.CLEAN: row.clean_cell_sha256,
            Condition.FAULTED: row.faulted_cell_sha256,
            Condition.NO_TOUCH: row.no_touch_cell_sha256,
        }
        for condition, address in references.items():
            cell = lookup.get(address)
            if cell is None or cell.trial.condition is not condition:
                raise ValueError(
                    "matrix comparison references the wrong cell condition"
                )
        faulted = lookup[row.faulted_cell_sha256]
        if (
            faulted.operator_id != row.operator_id
            or faulted.severity_level != row.severity_level
        ):
            raise ValueError("matrix comparison fault metadata disagrees with cell")
        MatrixManifest._validate_operator_instance(row, faulted)
        return tuple(references.values())

    @staticmethod
    def _validate_operator_instance(
        row: MatrixComparison, faulted: MatrixCellSpec
    ) -> None:
        fault = faulted.fault_manifest
        if fault is None:
            raise ValueError("matrix fault cell lost its operator instance")
        expected_point = canonical_hash(
            {
                "focus_id": row.focus_id,
                "fault_manifest_sha256": fault.sha256,
            }
        )
        if row.point_id != expected_point:
            raise ValueError("matrix point ID does not bind its operator instance")

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix_id": self.matrix_id,
            "kind": self.kind.value,
            "pair_key": self.pair_key,
            "cells": [cell.to_dict() for cell in self.cells],
            "comparisons": [row.to_dict() for row in self.comparisons],
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> MatrixManifest:
        fields = {
            "matrix_id",
            "kind",
            "pair_key",
            "cells",
            "comparisons",
            "semantic_version",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("matrix manifest fields mismatch")
        cells = value["cells"]
        comparisons = value["comparisons"]
        if not isinstance(cells, list) or not isinstance(comparisons, list):
            raise TypeError("matrix manifest cells/comparisons must be lists")
        return cls(
            matrix_id=require_nonempty(value["matrix_id"], "matrix_id"),
            kind=MatrixGridKind(require_nonempty(value["kind"], "kind")),
            pair_key=require_sha256(value["pair_key"], "pair_key"),
            cells=tuple(MatrixCellSpec.from_dict(item) for item in cells),
            comparisons=tuple(MatrixComparison.from_dict(item) for item in comparisons),
            semantic_version=require_nonempty(
                value["semantic_version"], "semantic_version"
            ),
        )

"""Final and in-progress matrix aggregate contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.matrix.contracts import (
    MATRIX_SEMANTIC_VERSION,
    MatrixGridKind,
    require_integer,
    require_nonempty,
    require_sha256,
)
from robotactile_benchmark.matrix.results import MatrixCellStatus
from robotactile_benchmark.matrix.states import MatrixCellState


@dataclass(frozen=True)
class MatrixSummary:
    """Deterministic terminal inventory; session-specific resume counts stay out."""

    matrix_id: str
    kind: MatrixGridKind
    manifest_sha256: str
    pair_key: str
    cells: Tuple[MatrixCellState, ...]
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
        object.__setattr__(
            self,
            "manifest_sha256",
            require_sha256(self.manifest_sha256, "manifest_sha256"),
        )
        object.__setattr__(self, "pair_key", require_sha256(self.pair_key, "pair_key"))
        if self.semantic_version != MATRIX_SEMANTIC_VERSION:
            raise ValueError("unsupported matrix summary semantic version")
        cells = tuple(self.cells)
        if not cells or any(not isinstance(cell, MatrixCellState) for cell in cells):
            raise ValueError("matrix summary requires typed cell states")
        if any(cell.status is MatrixCellStatus.PENDING for cell in cells):
            raise ValueError("final matrix summary cannot contain pending cells")
        addresses = tuple(cell.cell_sha256 for cell in cells)
        if len(set(addresses)) != len(addresses):
            raise ValueError("matrix summary cell addresses must be unique")
        if {cell.pair_key for cell in cells} != {self.pair_key}:
            raise ValueError("matrix summary cells must share its pair key")
        object.__setattr__(self, "cells", cells)

    @property
    def status_counts(self) -> dict[MatrixCellStatus, int]:
        return {
            status: sum(cell.status is status for cell in self.cells)
            for status in MatrixCellStatus
        }

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix_id": self.matrix_id,
            "kind": self.kind.value,
            "manifest_sha256": self.manifest_sha256,
            "pair_key": self.pair_key,
            "cells": [cell.to_dict() for cell in self.cells],
            "status_counts": {
                status.value: count for status, count in self.status_counts.items()
            },
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> MatrixSummary:
        fields = {
            "matrix_id",
            "kind",
            "manifest_sha256",
            "pair_key",
            "cells",
            "status_counts",
            "semantic_version",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("matrix summary fields mismatch")
        cells = value["cells"]
        counts = value["status_counts"]
        if not isinstance(cells, list) or not isinstance(counts, Mapping):
            raise TypeError("matrix summary cells/counts have invalid types")
        expected_keys = {status.value for status in MatrixCellStatus}
        if set(counts) != expected_keys:
            raise ValueError("matrix summary status count fields mismatch")
        summary = cls(
            matrix_id=require_nonempty(value["matrix_id"], "matrix_id"),
            kind=MatrixGridKind(require_nonempty(value["kind"], "kind")),
            manifest_sha256=require_sha256(value["manifest_sha256"], "manifest_sha256"),
            pair_key=require_sha256(value["pair_key"], "pair_key"),
            cells=tuple(MatrixCellState.from_dict(item) for item in cells),
            semantic_version=require_nonempty(
                value["semantic_version"], "semantic_version"
            ),
        )
        normalized = {
            MatrixCellStatus(key): require_integer(counts[key], f"status_counts.{key}")
            for key in expected_keys
        }
        if normalized != summary.status_counts:
            raise ValueError("matrix summary status counts disagree with cells")
        return summary


@dataclass(frozen=True)
class MatrixRunResult:
    """One orchestration invocation, including deterministic resume accounting."""

    states: Tuple[MatrixCellState, ...]
    summary: Optional[MatrixSummary]
    executed_cell_count: int
    reused_cell_count: int
    pending_cell_count: int

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if not states or any(
            not isinstance(state, MatrixCellState) for state in states
        ):
            raise ValueError("matrix run result requires typed states")
        executed = require_integer(self.executed_cell_count, "executed_cell_count")
        reused = require_integer(self.reused_cell_count, "reused_cell_count")
        pending = require_integer(self.pending_cell_count, "pending_cell_count")
        if executed + reused + pending != len(states):
            raise ValueError(
                "matrix run accounting does not cover every requested cell"
            )
        actual_pending = sum(
            state.status is MatrixCellStatus.PENDING for state in states
        )
        if actual_pending != pending:
            raise ValueError("matrix run pending count disagrees with states")
        if (self.summary is None) != (pending > 0):
            raise ValueError("matrix summary exists exactly when no cells are pending")
        if self.summary is not None and self.summary.cells != states:
            raise ValueError("matrix run summary does not match invocation states")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "executed_cell_count", executed)
        object.__setattr__(self, "reused_cell_count", reused)
        object.__setattr__(self, "pending_cell_count", pending)

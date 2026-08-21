"""Deterministic four-condition matrix builders."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.manifests import FaultManifest, Observability
from robotactile_benchmark.matrix.contracts import (
    MatrixCellSpec,
    MatrixComparison,
    MatrixGridKind,
    MatrixGridPoint,
)
from robotactile_benchmark.matrix.manifest import MatrixManifest
from robotactile_benchmark.severity import severity_value
from robotactile_benchmark.trials import (
    Condition,
    RestorationMode,
    TrialManifest,
    build_paired_trial_grid,
)


def build_primary_matrix_manifest(
    *,
    matrix_id: str,
    clean: TrialManifest,
    fault_manifests: Sequence[FaultManifest],
    restoration_index: int,
    restoration_mode: RestorationMode,
    no_touch_system_id: str,
    no_touch_checkpoint_sha256: str,
    no_touch_config_sha256: str,
) -> MatrixManifest:
    """Build the complete 14-by-5 blind grid with shared baselines."""

    faults = tuple(fault_manifests)
    expected = {
        (operator_id, severity)
        for operator_id in CORE_OPERATOR_IDS
        for severity in range(1, 6)
    }
    actual = {(item.operator_id, item.severity_level) for item in faults}
    if len(faults) != 70 or actual != expected:
        raise ValueError("primary matrix requires the complete blind 14 x 5 grid")
    if any(item.observability is not Observability.BLIND for item in faults):
        raise ValueError("primary matrix fault manifests must all be blind")
    points = tuple(
        MatrixGridPoint(
            focus_id=f"{fault.operator_id}:L{fault.severity_level}",
            fault_manifest=fault,
            restoration_index=restoration_index,
        )
        for fault in faults
    )
    return _build_matrix_manifest(
        matrix_id=matrix_id,
        kind=MatrixGridKind.PRIMARY,
        clean=clean,
        grid_points=points,
        restoration_mode=restoration_mode,
        no_touch_system_id=no_touch_system_id,
        no_touch_checkpoint_sha256=no_touch_checkpoint_sha256,
        no_touch_config_sha256=no_touch_config_sha256,
    )


def build_focused_phase_manifest(
    *,
    matrix_id: str,
    clean: TrialManifest,
    grid_points: Sequence[MatrixGridPoint],
    restoration_mode: RestorationMode,
    no_touch_system_id: str,
    no_touch_checkpoint_sha256: str,
    no_touch_config_sha256: str,
) -> MatrixManifest:
    """Build a blind sub-grid whose points target contact phases/windows."""

    return _build_matrix_manifest(
        matrix_id=matrix_id,
        kind=MatrixGridKind.FOCUSED_PHASE,
        clean=clean,
        grid_points=grid_points,
        restoration_mode=restoration_mode,
        no_touch_system_id=no_touch_system_id,
        no_touch_checkpoint_sha256=no_touch_checkpoint_sha256,
        no_touch_config_sha256=no_touch_config_sha256,
    )


def build_focused_restoration_manifest(
    *,
    matrix_id: str,
    clean: TrialManifest,
    grid_points: Sequence[MatrixGridPoint],
    restoration_mode: RestorationMode,
    no_touch_system_id: str,
    no_touch_checkpoint_sha256: str,
    no_touch_config_sha256: str,
) -> MatrixManifest:
    """Build a blind sub-grid that sweeps restoration boundaries or modes."""

    return _build_matrix_manifest(
        matrix_id=matrix_id,
        kind=MatrixGridKind.FOCUSED_RESTORATION,
        clean=clean,
        grid_points=grid_points,
        restoration_mode=restoration_mode,
        no_touch_system_id=no_touch_system_id,
        no_touch_checkpoint_sha256=no_touch_checkpoint_sha256,
        no_touch_config_sha256=no_touch_config_sha256,
    )


def _build_matrix_manifest(
    *,
    matrix_id: str,
    kind: MatrixGridKind,
    clean: TrialManifest,
    grid_points: Sequence[MatrixGridPoint],
    restoration_mode: RestorationMode,
    no_touch_system_id: str,
    no_touch_checkpoint_sha256: str,
    no_touch_config_sha256: str,
) -> MatrixManifest:
    if clean.condition is not Condition.CLEAN:
        raise ValueError("matrix construction must start from a clean trial")
    if not isinstance(restoration_mode, RestorationMode):
        restoration_mode = RestorationMode(restoration_mode)
    points = tuple(grid_points)
    if not points or any(not isinstance(point, MatrixGridPoint) for point in points):
        raise ValueError("focused matrix requires typed grid points")
    if any(
        point.fault_manifest.observability is not Observability.BLIND
        for point in points
    ):
        raise ValueError("benchmark matrix fault manifests must all be blind")
    if len({point.sha256 for point in points}) != len(points):
        raise ValueError("matrix grid points must be unique")
    ordered = tuple(
        sorted(
            points,
            key=lambda item: (
                item.fault_manifest.operator_id,
                item.fault_manifest.severity_level,
                item.focus_id,
                item.restoration_index,
                item.fault_manifest.sha256,
            ),
        )
    )
    cells_by_address: dict[str, MatrixCellSpec] = {}
    comparisons: list[MatrixComparison] = []
    clean_address = ""
    no_touch_address = ""
    for point in ordered:
        restored_fault = point.fault_manifest.reparameterized(
            stop_index=point.restoration_index
        )
        trials = build_paired_trial_grid(
            clean=clean,
            faulted_manifest=point.fault_manifest,
            restored_manifest=restored_fault,
            restoration_index=point.restoration_index,
            restoration_mode=restoration_mode,
            no_touch_system_id=no_touch_system_id,
            no_touch_checkpoint_sha256=no_touch_checkpoint_sha256,
            no_touch_config_sha256=no_touch_config_sha256,
        )
        by_condition = {trial.condition: trial for trial in trials}
        row_cells = {
            Condition.CLEAN: MatrixCellSpec(by_condition[Condition.CLEAN], None),
            Condition.NO_TOUCH: MatrixCellSpec(by_condition[Condition.NO_TOUCH], None),
            Condition.FAULTED: MatrixCellSpec(
                by_condition[Condition.FAULTED], point.fault_manifest
            ),
            Condition.RESTORED: MatrixCellSpec(
                by_condition[Condition.RESTORED], restored_fault
            ),
        }
        for cell in row_cells.values():
            cells_by_address.setdefault(cell.sha256, cell)
        clean_address = row_cells[Condition.CLEAN].sha256
        no_touch_address = row_cells[Condition.NO_TOUCH].sha256
        fault = point.fault_manifest
        comparisons.append(
            MatrixComparison(
                point_id=canonical_hash(
                    {
                        "focus_id": point.focus_id,
                        "fault_manifest_sha256": fault.sha256,
                        "restoration_index": point.restoration_index,
                        "restoration_mode": restoration_mode.value,
                    }
                ),
                focus_id=point.focus_id,
                operator_id=fault.operator_id,
                severity_level=fault.severity_level,
                native_dose=severity_value(fault.operator_id, fault.severity_level),
                restoration_index=point.restoration_index,
                clean_cell_sha256=clean_address,
                faulted_cell_sha256=row_cells[Condition.FAULTED].sha256,
                no_touch_cell_sha256=no_touch_address,
                restored_cell_sha256=row_cells[Condition.RESTORED].sha256,
            )
        )
    cells = _ordered_cells(cells_by_address.values())
    return MatrixManifest(
        matrix_id=matrix_id,
        kind=kind,
        pair_key=clean.pair_key,
        cells=cells,
        comparisons=tuple(comparisons),
    )


def _ordered_cells(cells: Iterable[MatrixCellSpec]) -> Tuple[MatrixCellSpec, ...]:
    order = {
        Condition.CLEAN: 0,
        Condition.NO_TOUCH: 1,
        Condition.FAULTED: 2,
        Condition.RESTORED: 3,
    }
    return tuple(
        sorted(
            cells,
            key=lambda cell: (
                order[cell.trial.condition],
                cell.operator_id or "",
                cell.severity_level or 0,
                cell.trial.restoration_index or 0,
                cell.sha256,
            ),
        )
    )

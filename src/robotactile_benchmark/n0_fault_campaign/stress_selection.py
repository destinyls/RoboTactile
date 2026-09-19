"""Freeze selection from the entire predeclared calibration artifact cohort."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .stress_group import read_stress_rows
from .stress_metrics import summarize_stress
from .stress_protocol import select_calibration_doses, validate_protocol


def report_frozen_groups(
    protocol: dict[str, Any],
    groups: list[Path] | None = None,
) -> dict[str, Any]:
    plan = validate_protocol(protocol)
    if "group_paths" in plan:
        roots = [Path(plan["group_paths"][str(s)]) for s in plan["seeds"]]
        if groups is not None and (
            len(groups) != len(roots) or {p.resolve() for p in groups} != set(roots)
        ):
            raise ValueError("report must include all and only the frozen group paths")
    elif groups:
        roots = [p.resolve(strict=True) for p in groups]
    else:
        raise ValueError("unbound exploratory studies require explicit group paths")
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate group paths")
    rows = []
    for root in roots:
        if "group_paths" in plan and not (root / "group.json").exists():
            # A group not yet prepared is missing, never a model failure.
            continue
        rows.extend(read_stress_rows(root))
    return summarize_stress(plan, rows)


def calibration_selection_evidence(protocol: dict[str, Any]) -> dict[str, Any]:
    plan = validate_protocol(protocol)
    if plan["stage"] != "calibration" or "group_paths" not in plan:
        raise ValueError("selection requires a fixed-path calibration protocol")
    report = report_frozen_groups(plan)
    if (
        report["missing_accepted_cells"]
        or report["invalid_attempt_rollouts"]
        or report["provisional_live_attempt_rollouts"]
        or report["unsupported_contract"]
    ):
        raise ValueError("calibration artifact cohort is incomplete or invalid")
    return {
        "schema": "n0_stress_selection_evidence_v1",
        "protocol": plan,
        "report": report,
        "selection": select_calibration_doses(plan, report),
    }

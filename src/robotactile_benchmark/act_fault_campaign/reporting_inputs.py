"""Duck-typed ACT campaign and live-artifact normalization for reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple, cast

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.reporting.contracts import require_sha256
from robotactile_benchmark.trials import Condition, TerminalStatus

_STRUCTURAL_ABSENCE_OPERATOR_IDS = frozenset({"A1_stream_absence", "A2_frame_erasure"})
_MODEL_STATUSES = frozenset(
    {
        TerminalStatus.SUCCESS,
        TerminalStatus.TASK_FAILURE,
        TerminalStatus.EARLY_STOP,
        TerminalStatus.TIMEOUT,
    }
)
_NO_DEFAULT = object()
_MISSING = object()
Category = Literal[
    "model_success",
    "model_failure",
    "unsupported_contract",
    "infrastructure_failure",
    "validator_failure",
    "missing_artifact",
]


def _field(value: object, name: str, default: object = _NO_DEFAULT) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if name in mapping:
            return mapping[name]
    else:
        result = getattr(value, name, _MISSING)
        if result is not _MISSING:
            return result
    if default is not _NO_DEFAULT:
        return default
    raise ValueError(f"required ACT reporting field is missing: {name}")


def _enum_text(value: object, name: str) -> str:
    raw = value.value if isinstance(value, Enum) else value
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{name} must be a non-empty string or string enum")
    return raw


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return cast(int, value)


@dataclass(frozen=True)
class PlannedCell:
    cell_id: str
    lookup_keys: Tuple[object, ...]
    task: str
    pair_key: str
    condition: Condition
    disposition: str
    operator_id: Optional[str]
    severity_level: Optional[int]


@dataclass(frozen=True)
class CellOutcome:
    cell: PlannedCell
    category: Category
    score: Optional[bool]


@dataclass(frozen=True)
class CampaignInputs:
    campaign_id: str
    campaign_manifest_sha256: str
    outcomes: Tuple[CellOutcome, ...]


def _lookup_keys(cell: object) -> Tuple[str, Tuple[object, ...]]:
    ordinal = _field(cell, "ordinal", None)
    relpath = _field(cell, "artifact_relpath", None)
    explicit = _field(cell, "cell_id", None)
    if ordinal is None and relpath is None and explicit is None:
        raise ValueError("ACT campaign cell lacks an artifact lookup identity")
    raw = tuple(item for item in (relpath, ordinal, explicit) if item is not None)
    candidates: list[object] = []
    for item in raw:
        for candidate in (item, str(item)):
            try:
                hash(candidate)
            except TypeError:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    identity = explicit if explicit is not None else ordinal
    if identity is None:
        identity = relpath
    return str(identity), tuple(candidates)


def _planned_cell(cell: object) -> PlannedCell:
    cell_id, lookup_keys = _lookup_keys(cell)
    condition_text = _enum_text(_field(cell, "condition"), "condition")
    if condition_text not in {Condition.CLEAN.value, Condition.FAULTED.value}:
        raise ValueError("ACT fault reporting accepts only Clean and Faulted")
    condition = Condition(condition_text)
    disposition = _enum_text(_field(cell, "disposition"), "disposition")
    if disposition not in {"live_request", "unsupported_contract"}:
        raise ValueError("unknown ACT campaign cell disposition")
    operator_value = _field(cell, "operator_id", None)
    severity_value = _field(cell, "severity_level", None)
    operator_id: Optional[str] = None
    severity_level: Optional[int] = None
    if condition is Condition.CLEAN:
        if operator_value is not None or severity_value is not None:
            raise ValueError("Clean ACT cells cannot contain fault metadata")
        if disposition != "live_request":
            raise ValueError("Clean ACT cells must be live requests")
    else:
        operator_id = _nonempty_text(operator_value, "operator_id")
        if operator_id not in CORE_OPERATOR_IDS:
            raise ValueError("Faulted ACT cell uses an unknown operator")
        severity_level = _strict_int(severity_value, "severity_level")
        if not 1 <= severity_level <= 5:
            raise ValueError("severity_level must lie in [1, 5]")
        if (
            operator_id in _STRUCTURAL_ABSENCE_OPERATOR_IDS
            and disposition != "unsupported_contract"
        ):
            raise ValueError("ACT A1/A2 cells must be unsupported contracts")
    return PlannedCell(
        cell_id=cell_id,
        lookup_keys=lookup_keys,
        task=_nonempty_text(_field(cell, "task"), "task"),
        pair_key=require_sha256(_field(cell, "pair_key"), "pair_key"),
        condition=condition,
        disposition=disposition,
        operator_id=operator_id,
        severity_level=severity_level,
    )


def _artifact_for(cell: PlannedCell, artifacts: Mapping[object, object]) -> object:
    for key in cell.lookup_keys:
        if key in artifacts:
            return artifacts[key]
    return _MISSING


def _terminal_result(artifact: object) -> object:
    if _field(artifact, "terminal_status", _MISSING) is not _MISSING:
        return artifact
    evidence = _field(artifact, "evidence", _MISSING)
    if evidence is not _MISSING:
        result = _field(evidence, "result", _MISSING)
        if result is not _MISSING:
            return result
    for name in ("result", "terminal_result"):
        result = _field(artifact, name, _MISSING)
        if result is not _MISSING:
            return result
    raise ValueError("ACT artifact does not expose a terminal result")


def _classify(cell: PlannedCell, artifact: object) -> CellOutcome:
    if cell.disposition == "unsupported_contract":
        return CellOutcome(cell, "unsupported_contract", None)
    if artifact is _MISSING:
        return CellOutcome(cell, "missing_artifact", None)
    result = _terminal_result(artifact)
    status = TerminalStatus(
        _enum_text(_field(result, "terminal_status"), "terminal_status")
    )
    eligible = _field(result, "score_eligible")
    success = _field(result, "score_success")
    if type(eligible) is not bool:
        raise TypeError("ACT artifact score_eligible must be an exact bool")
    if success is not None and type(success) is not bool:
        raise TypeError("ACT artifact score_success must be bool or None")
    if status in _MODEL_STATUSES:
        expected = status is TerminalStatus.SUCCESS
        if eligible is not True or success is not expected:
            raise ValueError("ACT model terminal result has incoherent score fields")
        return CellOutcome(
            cell, "model_success" if expected else "model_failure", expected
        )
    if status is TerminalStatus.CRASH:
        return CellOutcome(cell, "infrastructure_failure", None)
    if status is TerminalStatus.VALIDATOR_REJECTED:
        if eligible is not False or success is not None:
            raise ValueError("ACT validator result has incoherent score fields")
        return CellOutcome(cell, "validator_failure", None)
    if status is TerminalStatus.UNSUPPORTED_CONTRACT:
        if eligible is not False or success is not None:
            raise ValueError("ACT unsupported result has incoherent score fields")
        return CellOutcome(cell, "unsupported_contract", None)
    raise ValueError("unknown ACT terminal result status")


def load_campaign_inputs(
    manifest: object, artifacts_by_cell: Mapping[object, object]
) -> CampaignInputs:
    """Normalize parallel campaign contracts without importing their classes."""

    if not isinstance(artifacts_by_cell, Mapping):
        raise TypeError("artifacts_by_cell must be a mapping")
    if _enum_text(_field(manifest, "policy_kind", "act"), "policy_kind") != "act":
        raise ValueError("ACT reporting requires policy_kind=act")
    if _enum_text(_field(manifest, "profile", "univtac"), "profile") != "univtac":
        raise ValueError("ACT robustness reporting requires profile=univtac")
    raw_cells = _field(manifest, "cells")
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
        raise TypeError("ACT campaign cells must be a sequence")
    cells = tuple(_planned_cell(item) for item in raw_cells)
    if not cells:
        raise ValueError("ACT reporting requires a non-empty campaign")
    identities = {
        (
            item.task,
            item.pair_key,
            item.condition.value,
            item.operator_id,
            item.severity_level,
        )
        for item in cells
    }
    if len(identities) != len(cells):
        raise ValueError("ACT reporting campaign contains duplicate cells")
    artifacts = cast(Mapping[object, object], artifacts_by_cell)
    outcomes = tuple(_classify(cell, _artifact_for(cell, artifacts)) for cell in cells)
    return CampaignInputs(
        campaign_id=_nonempty_text(_field(manifest, "campaign_id"), "campaign_id"),
        campaign_manifest_sha256=require_sha256(
            _field(manifest, "sha256"), "campaign_manifest_sha256"
        ),
        outcomes=outcomes,
    )

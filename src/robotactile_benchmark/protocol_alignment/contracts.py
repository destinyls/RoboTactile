"""Typed contracts for bounded N0/UniVTAC public-result comparisons."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, cast

from robotactile_benchmark.backends.univtac_registry import load_registry
from robotactile_benchmark.clean_baseline.contracts import CleanCampaignProtocol
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.execution.contracts import LivePolicyKind

REFERENCE_SEMANTIC_VERSION = "1.0"
ALIGNMENT_REPORT_SEMANTIC_VERSION = "1.0"
ALIGNMENT_EVIDENCE_LEVEL = "verified_clean_artifact_protocol_alignment_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProtocolAlignmentError(ValueError):
    """A reference or generated alignment report failed strict validation."""


class GateStatus(str, Enum):
    """Tri-state gate result; missing evidence is never silently treated as pass."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class EvaluationSemantics(str, Enum):
    """Evaluator contract used to interpret protocol evidence."""

    OFFICIAL_REPRODUCTION = "official_reproduction"


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProtocolAlignmentError(f"{name} must be a non-empty string")
    return value


def _identifier(value: object, name: str) -> str:
    normalized = _nonempty(value, name)
    if _IDENTIFIER.fullmatch(normalized) is None:
        raise ProtocolAlignmentError(f"{name} must be a safe identifier")
    return normalized


def _sha256(value: object, name: str) -> str:
    normalized = _nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise ProtocolAlignmentError(f"{name} must be a lowercase SHA256")
    return normalized


def _optional_sha256(value: object, name: str) -> Optional[str]:
    return None if value is None else _sha256(value, name)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolAlignmentError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_bool(value: object, name: str) -> Optional[bool]:
    if value is None:
        return None
    if type(value) is not bool:
        raise ProtocolAlignmentError(f"{name} must be bool or null")
    return value


def _string_tuple(
    value: object, name: str, *, nonempty: bool = True
) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolAlignmentError(f"{name} must be a string sequence")
    result = tuple(_nonempty(item, name) for item in value)
    if (nonempty and not result) or len(result) != len(set(result)):
        raise ProtocolAlignmentError(f"{name} must contain unique strings")
    return result


@dataclass(frozen=True)
class N0PaperReference:
    """Public claim boundary plus RoboTactile release requirements.

    The required action execution identifies the source-bound, training-aligned
    RoboTactile evaluator.  It is not a claim about the unpublished paper-exact
    implementation.
    """

    reference_id: str
    report_url: str
    report_snapshot_sha256: Optional[str]
    metric_name: str
    reported_macro_success_rate: float
    task_ids: Tuple[str, ...]
    trials_per_task: int
    required_policy_kind: LivePolicyKind
    required_protocol_id: CleanCampaignProtocol
    required_seed_protocol: str
    required_reset_mode: str
    required_action_execution: str
    required_exception_handling: str
    claim_boundary: str
    evaluation_semantics: EvaluationSemantics = (
        EvaluationSemantics.OFFICIAL_REPRODUCTION
    )
    semantic_version: str = REFERENCE_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_id", _identifier(self.reference_id, "reference_id")
        )
        report_url = _nonempty(self.report_url, "report_url")
        if not report_url.startswith("https://"):
            raise ProtocolAlignmentError("report_url must use HTTPS")
        rate = float(self.reported_macro_success_rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ProtocolAlignmentError("reported success rate must lie in [0, 1]")
        tasks = _string_tuple(self.task_ids, "task_ids")
        registered = tuple(task.task_id for task in load_registry().tasks)
        if set(tasks) != set(registered) or len(tasks) != len(registered):
            raise ProtocolAlignmentError(
                "paper reference must cover the frozen 8 tasks"
            )
        policy = (
            self.required_policy_kind
            if isinstance(self.required_policy_kind, LivePolicyKind)
            else LivePolicyKind(self.required_policy_kind)
        )
        protocol = (
            self.required_protocol_id
            if isinstance(self.required_protocol_id, CleanCampaignProtocol)
            else CleanCampaignProtocol(self.required_protocol_id)
        )
        evaluation_semantics = (
            self.evaluation_semantics
            if isinstance(self.evaluation_semantics, EvaluationSemantics)
            else EvaluationSemantics(self.evaluation_semantics)
        )
        if (
            policy is not LivePolicyKind.N0
            or protocol is not CleanCampaignProtocol.PAPER
        ):
            raise ProtocolAlignmentError("paper reference must require N0 paper_v1")
        if evaluation_semantics is not EvaluationSemantics.OFFICIAL_REPRODUCTION:
            raise ProtocolAlignmentError(
                "N0 paper reference must use official_reproduction semantics"
            )
        for name in (
            "metric_name",
            "required_seed_protocol",
            "required_reset_mode",
            "required_action_execution",
            "required_exception_handling",
            "claim_boundary",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.semantic_version != REFERENCE_SEMANTIC_VERSION:
            raise ProtocolAlignmentError("unsupported paper reference version")
        object.__setattr__(
            self,
            "report_snapshot_sha256",
            _optional_sha256(self.report_snapshot_sha256, "report_snapshot_sha256"),
        )
        object.__setattr__(self, "reported_macro_success_rate", rate)
        object.__setattr__(self, "task_ids", tasks)
        object.__setattr__(
            self,
            "trials_per_task",
            _integer(self.trials_per_task, "trials_per_task", 1),
        )
        object.__setattr__(self, "required_policy_kind", policy)
        object.__setattr__(self, "required_protocol_id", protocol)
        object.__setattr__(self, "evaluation_semantics", evaluation_semantics)

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "report_url": self.report_url,
            "report_snapshot_sha256": self.report_snapshot_sha256,
            "metric_name": self.metric_name,
            "reported_macro_success_rate": self.reported_macro_success_rate,
            "task_ids": list(self.task_ids),
            "trials_per_task": self.trials_per_task,
            "required_policy_kind": self.required_policy_kind.value,
            "required_protocol_id": self.required_protocol_id.value,
            "required_seed_protocol": self.required_seed_protocol,
            "required_reset_mode": self.required_reset_mode,
            "required_action_execution": self.required_action_execution,
            "required_exception_handling": self.required_exception_handling,
            "claim_boundary": self.claim_boundary,
            "evaluation_semantics": self.evaluation_semantics.value,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "N0PaperReference":
        fields = set(cls.__dataclass_fields__)
        legacy_fields = fields - {"evaluation_semantics"}
        if not isinstance(value, Mapping) or frozenset(value) not in {
            frozenset(fields),
            frozenset(legacy_fields),
        }:
            raise ProtocolAlignmentError("paper reference fields mismatch")
        kwargs = dict(value)
        kwargs.setdefault(
            "evaluation_semantics", EvaluationSemantics.OFFICIAL_REPRODUCTION.value
        )
        kwargs["task_ids"] = tuple(cast(Sequence[str], kwargs["task_ids"]))
        return cls(**cast(Any, kwargs))


@dataclass(frozen=True)
class ProtocolGate:
    """One deterministic alignment decision and its affected task scope."""

    gate_id: str
    status: GateStatus
    code: str
    detail: str
    affected_tasks: Tuple[str, ...] = ()
    paper_blocking: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _identifier(self.gate_id, "gate_id"))
        status = (
            self.status
            if isinstance(self.status, GateStatus)
            else GateStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "code", _identifier(self.code, "code"))
        object.__setattr__(self, "detail", _nonempty(self.detail, "detail"))
        if type(self.paper_blocking) is not bool:
            raise ProtocolAlignmentError("paper_blocking must be bool")
        object.__setattr__(
            self,
            "affected_tasks",
            _string_tuple(self.affected_tasks, "affected_tasks", nonempty=False),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "status": self.status.value,
            "code": self.code,
            "detail": self.detail,
            "affected_tasks": list(self.affected_tasks),
            "paper_blocking": self.paper_blocking,
        }


@dataclass(frozen=True)
class TrialProtocolDiagnostic:
    """Artifact-bound terminal and execution witnesses for one planned trial."""

    ordinal: int
    task_id: str
    artifact_root_sha256: str
    terminal_status: str
    score_success: Optional[bool]
    observation_count: int
    control_cycle_count: int
    transition_count: int
    executed_action_count: int
    first_signal: Optional[str]
    last_signal: Optional[str]
    classification: str
    immediate_terminal: bool
    initial_success_check: Optional[bool]
    initial_early_stop: Optional[bool]
    detailed_predicate_witness: bool
    transition_contract_status: GateStatus
    execution_contract_status: GateStatus
    cadence_status: GateStatus
    reset_mode: Optional[str]
    action_execution_mode: Optional[str]
    failure_stage: Optional[str]
    failure_code: Optional[str]

    def __post_init__(self) -> None:
        for name in (
            "ordinal",
            "observation_count",
            "control_cycle_count",
            "transition_count",
            "executed_action_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "artifact_root_sha256",
            _sha256(self.artifact_root_sha256, "artifact_root_sha256"),
        )
        object.__setattr__(
            self, "terminal_status", _nonempty(self.terminal_status, "terminal_status")
        )
        classification = _nonempty(self.classification, "classification")
        classification = {
            "immediate_success": "first_action_success",
            "immediate_early_stop": "first_action_early_stop",
        }.get(classification, classification)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(
            self, "score_success", _optional_bool(self.score_success, "score_success")
        )
        if (
            type(self.immediate_terminal) is not bool
            or type(self.detailed_predicate_witness) is not bool
        ):
            raise ProtocolAlignmentError("trial diagnostic flags must be bool")
        for name in ("initial_success_check", "initial_early_stop"):
            object.__setattr__(self, name, _optional_bool(getattr(self, name), name))
        for name in (
            "transition_contract_status",
            "execution_contract_status",
            "cadence_status",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                value if isinstance(value, GateStatus) else GateStatus(value),
            )

    def to_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name in (
            "transition_contract_status",
            "execution_contract_status",
            "cadence_status",
        ):
            result[name] = cast(GateStatus, result[name]).value
        return result


__all__ = [
    "ALIGNMENT_EVIDENCE_LEVEL",
    "ALIGNMENT_REPORT_SEMANTIC_VERSION",
    "EvaluationSemantics",
    "GateStatus",
    "N0PaperReference",
    "ProtocolAlignmentError",
    "ProtocolGate",
    "REFERENCE_SEMANTIC_VERSION",
    "TrialProtocolDiagnostic",
]

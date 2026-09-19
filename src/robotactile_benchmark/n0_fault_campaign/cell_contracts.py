"""Cell-level contracts for N0-TWAM Clean/Faulted campaigns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, cast

from robotactile_benchmark.constants import CORE_OPERATOR_IDS
from robotactile_benchmark.trials import Condition, TerminalStatus

from .contract_primitives import (
    N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION,
    N0_UNSUPPORTED_OPERATOR_IDS,
    N0FaultCampaignError,
    N0FaultCellDisposition,
    enum_field_dict,
    require_identifier,
    require_integer,
    require_nonempty,
    require_relative_path,
    require_sha256,
)


@dataclass(frozen=True)
class N0FaultCampaignCellSpec:
    """One generated Clean, Faulted, or unsupported campaign cell."""

    ordinal: int
    task: str
    initial_seed: int
    exogenous_seed: int
    pair_key: str
    condition: Condition
    disposition: N0FaultCellDisposition
    operator_id: Optional[str]
    severity_level: Optional[int]
    operator_template_seed: Optional[int]
    request_relpath: Optional[str]
    artifact_relpath: Optional[str]
    request_file_sha256: Optional[str]
    trial_manifest_sha256: str
    fault_manifest_sha256: Optional[str]
    fault_manifest_relpath: Optional[str]
    unsupported_receipt_relpath: Optional[str]
    unsupported_receipt_sha256: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordinal", require_integer(self.ordinal, "ordinal"))
        object.__setattr__(self, "task", require_nonempty(self.task, "task"))
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(self, name, require_integer(getattr(self, name), name))
        object.__setattr__(self, "pair_key", require_sha256(self.pair_key, "pair_key"))
        condition = (
            self.condition
            if isinstance(self.condition, Condition)
            else Condition(self.condition)
        )
        disposition = (
            self.disposition
            if isinstance(self.disposition, N0FaultCellDisposition)
            else N0FaultCellDisposition(self.disposition)
        )
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "trial_manifest_sha256",
            require_sha256(self.trial_manifest_sha256, "trial_manifest_sha256"),
        )
        self._validate_condition_fields()

    def _validate_condition_fields(self) -> None:
        for name in (
            "request_file_sha256",
            "fault_manifest_sha256",
            "unsupported_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_sha256(value, name))
        if self.condition is Condition.CLEAN:
            self._validate_clean_fields()
        elif self.condition is Condition.FAULTED:
            self._validate_fault_fields()
        else:
            raise N0FaultCampaignError("N0 campaigns allow only clean/faulted")
        if self.disposition is N0FaultCellDisposition.LIVE_REQUEST:
            self._validate_live_paths()
        else:
            self._validate_unsupported_fields()

    def _validate_clean_fields(self) -> None:
        fault_only = (
            self.operator_id,
            self.severity_level,
            self.operator_template_seed,
            self.fault_manifest_sha256,
            self.fault_manifest_relpath,
            self.unsupported_receipt_relpath,
            self.unsupported_receipt_sha256,
        )
        if self.disposition is not N0FaultCellDisposition.LIVE_REQUEST or any(
            value is not None for value in fault_only
        ):
            raise N0FaultCampaignError("clean cell fields are inconsistent")

    def _validate_fault_fields(self) -> None:
        if self.operator_id not in CORE_OPERATOR_IDS:
            raise N0FaultCampaignError("fault cell has an unknown operator")
        if self.severity_level not in range(1, 6):
            raise N0FaultCampaignError("fault severity must be in [1, 5]")
        require_integer(self.operator_template_seed, "operator_template_seed")
        if self.fault_manifest_sha256 is None:
            raise N0FaultCampaignError("fault cell requires a fault manifest")
        object.__setattr__(
            self,
            "fault_manifest_relpath",
            require_relative_path(
                self.fault_manifest_relpath,
                "fault_manifest_relpath",
                "fault_manifests",
            ),
        )

    def _validate_live_paths(self) -> None:
        if self.request_file_sha256 is None:
            raise N0FaultCampaignError("live cell requires a request hash")
        object.__setattr__(
            self,
            "request_relpath",
            require_relative_path(self.request_relpath, "request_relpath", "requests"),
        )
        object.__setattr__(
            self,
            "artifact_relpath",
            require_relative_path(
                self.artifact_relpath, "artifact_relpath", "artifacts"
            ),
        )
        if self.unsupported_receipt_relpath is not None or (
            self.unsupported_receipt_sha256 is not None
        ):
            raise N0FaultCampaignError("live cell cannot carry unsupported receipt")

    def _validate_unsupported_fields(self) -> None:
        if self.operator_id not in N0_UNSUPPORTED_OPERATOR_IDS:
            raise N0FaultCampaignError("only A1/A2 may be unsupported for N0")
        if (
            any(
                value is not None
                for value in (
                    self.request_relpath,
                    self.artifact_relpath,
                    self.request_file_sha256,
                )
            )
            or self.unsupported_receipt_sha256 is None
        ):
            raise N0FaultCampaignError("unsupported cell cannot be a live request")
        object.__setattr__(
            self,
            "unsupported_receipt_relpath",
            require_relative_path(
                self.unsupported_receipt_relpath,
                "unsupported_receipt_relpath",
                "unsupported_contracts",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return enum_field_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> N0FaultCampaignCellSpec:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise N0FaultCampaignError("campaign cell fields mismatch")
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True)
class N0UnsupportedContractSpec:
    """Receipt proving that an A1/A2 cell was not represented as a live request."""

    campaign_id: str
    task: str
    initial_seed: int
    exogenous_seed: int
    pair_key: str
    operator_id: str
    severity_level: int
    operator_template_seed: int
    fault_manifest_sha256: str
    trial_manifest_sha256: str
    terminal_status: TerminalStatus = TerminalStatus.UNSUPPORTED_CONTRACT
    reason_code: str = "n0_requires_both_tactile_streams_v1"
    live_request_generated: bool = False
    simulator_execution_claimed: bool = False
    black_frame_substitution_used: bool = False
    semantic_version: str = N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "campaign_id", require_identifier(self.campaign_id, "campaign_id")
        )
        object.__setattr__(self, "task", require_nonempty(self.task, "task"))
        for name in ("initial_seed", "exogenous_seed", "operator_template_seed"):
            object.__setattr__(self, name, require_integer(getattr(self, name), name))
        object.__setattr__(self, "pair_key", require_sha256(self.pair_key, "pair_key"))
        if self.operator_id not in N0_UNSUPPORTED_OPERATOR_IDS:
            raise N0FaultCampaignError("unsupported receipt must identify A1 or A2")
        if self.severity_level not in range(1, 6):
            raise N0FaultCampaignError("unsupported severity must be in [1, 5]")
        for name in ("fault_manifest_sha256", "trial_manifest_sha256"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name))
        status = (
            self.terminal_status
            if isinstance(self.terminal_status, TerminalStatus)
            else TerminalStatus(self.terminal_status)
        )
        object.__setattr__(self, "terminal_status", status)
        if status is not TerminalStatus.UNSUPPORTED_CONTRACT:
            raise N0FaultCampaignError("unsupported receipt status mismatch")
        if self.reason_code != "n0_requires_both_tactile_streams_v1" or any(
            (
                self.live_request_generated,
                self.simulator_execution_claimed,
                self.black_frame_substitution_used,
            )
        ):
            raise N0FaultCampaignError("unsupported receipt evidence mismatch")
        if self.semantic_version != N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION:
            raise N0FaultCampaignError("unsupported receipt version mismatch")

    def to_dict(self) -> dict[str, object]:
        return enum_field_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> N0UnsupportedContractSpec:
        if not isinstance(value, Mapping) or set(value) != set(
            cls.__dataclass_fields__
        ):
            raise N0FaultCampaignError("unsupported receipt fields mismatch")
        return cls(**cast(Any, dict(value)))

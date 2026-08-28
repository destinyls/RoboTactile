"""Immutable request contracts for one bounded UniVTAC execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Optional, Union, cast

from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.trials import Condition, RestorationMode

LIVE_REQUEST_SEMANTIC_VERSION = "1.0"
UNQUALIFIED_EXECUTION_EVIDENCE = "unqualified_closed_loop_execution"
ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG: Final[str] = "--/app/hangDetector/enabled=false"
_PRODUCTION_UNIVTAC_LAUNCHER_ARG_KEYS = frozenset(
    {"enable_cameras", "headless", "kit_args"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
LauncherArgValue = Union[bool, str]


class LivePolicyKind(str, Enum):
    """Policy families supported by the live execution orchestrator."""

    ACT = "act"
    N0 = "n0"


class ArtifactExportStatus(str, Enum):
    """Whether a correctly typed live artifact was requested and available."""

    NOT_REQUESTED = "not_requested"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    EXPORTED = "exported"


class LiveExecutionUnavailableError(RuntimeError):
    """Stable fail-fast error for an unavailable live dependency."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _nonempty(value, name)


def _sha256(value: Any, name: str, *, optional: bool = False) -> Optional[str]:
    if optional and value is None:
        return None
    normalized = _nonempty(value, name)
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return cast(int, value)


def _path(value: Any, name: str, *, optional: bool = False) -> Optional[Path]:
    if optional and value is None:
        return None
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    return value.absolute()


def production_univtac_launcher_args() -> dict[str, LauncherArgValue]:
    """Return the only launcher arguments valid for a production live request."""

    return {
        "enable_cameras": True,
        "headless": True,
        "kit_args": ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG,
    }


def _validate_production_univtac_launcher_args(
    value: object,
) -> Mapping[str, LauncherArgValue]:
    if not isinstance(value, Mapping):
        raise TypeError("launcher_args must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("production launcher_args keys must be strings")
    keys = set(value)
    if keys != _PRODUCTION_UNIVTAC_LAUNCHER_ARG_KEYS:
        missing = sorted(_PRODUCTION_UNIVTAC_LAUNCHER_ARG_KEYS - keys)
        extra = sorted(keys - _PRODUCTION_UNIVTAC_LAUNCHER_ARG_KEYS)
        raise ValueError(
            f"production launcher_args fields mismatch: missing={missing}, "
            f"extra={extra}"
        )
    if value["headless"] is not True:
        raise ValueError("production launcher_args.headless must be true")
    if value["enable_cameras"] is not True:
        raise ValueError("production launcher_args.enable_cameras must be true")
    kit_args = value["kit_args"]
    if not isinstance(kit_args, str):
        raise TypeError("production launcher_args.kit_args must be a string")
    if kit_args != ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG:
        raise ValueError(
            "production launcher_args.kit_args must contain exactly "
            f"{ISAAC_DISABLE_HANG_DETECTOR_KIT_ARG!r}"
        )
    return cast(Mapping[str, LauncherArgValue], value)


@dataclass(frozen=True)
class LiveUniVTACRunRequest:
    """Explicit paths plus content identities for one execution attempt."""

    task_id: str
    condition: Condition
    policy_kind: LivePolicyKind
    base_system_id: str
    dataset_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    base_system_manifest_sha256: Optional[str]
    initial_seed: int
    exogenous_seed: int
    max_control_cycles: int
    max_observation_steps: int
    execute_action_steps: int
    wall_timeout_s: float
    upstream_root: Path
    runtime_dir: Path
    output_dir: Optional[Path]
    fault_manifest_path: Optional[Path]
    rest_references_path: Optional[Path]
    restoration_index: Optional[int]
    restoration_mode: Optional[RestorationMode]
    matched_no_touch_system_id: Optional[str]
    matched_no_touch_artifact_path: Optional[Path]
    act_device_name: Optional[str]
    simulator_device: Optional[str]
    launcher_args: Mapping[str, Any] = field(default_factory=dict)
    n0_source_commit: Optional[str] = None
    n0_normalizer_sha256: Optional[str] = None
    n0_serve_bundle_sha256: Optional[str] = None
    n0_prompt_manifest_sha256: Optional[str] = None
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION
    wall_timeout_role: WallTimeoutRole = WallTimeoutRole.SCORING_BOUNDARY_V1
    semantic_version: str = LIVE_REQUEST_SEMANTIC_VERSION

    def __post_init__(self) -> None:
        if self.semantic_version != LIVE_REQUEST_SEMANTIC_VERSION:
            raise ValueError("unsupported live request semantic version")
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(
            self, "base_system_id", _nonempty(self.base_system_id, "base_system_id")
        )
        condition = (
            self.condition
            if isinstance(self.condition, Condition)
            else Condition(self.condition)
        )
        policy_kind = (
            self.policy_kind
            if isinstance(self.policy_kind, LivePolicyKind)
            else LivePolicyKind(self.policy_kind)
        )
        restoration_mode = self.restoration_mode
        if restoration_mode is not None and not isinstance(
            restoration_mode, RestorationMode
        ):
            restoration_mode = RestorationMode(restoration_mode)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "policy_kind", policy_kind)
        object.__setattr__(self, "restoration_mode", restoration_mode)
        initial_state_policy = (
            self.initial_state_policy
            if isinstance(self.initial_state_policy, InitialStatePolicy)
            else InitialStatePolicy(self.initial_state_policy)
        )
        object.__setattr__(self, "initial_state_policy", initial_state_policy)
        wall_timeout_role = (
            self.wall_timeout_role
            if isinstance(self.wall_timeout_role, WallTimeoutRole)
            else WallTimeoutRole(self.wall_timeout_role)
        )
        object.__setattr__(self, "wall_timeout_role", wall_timeout_role)
        self._normalize_identities()
        self._normalize_budget()
        self._normalize_paths()
        self._validate_condition()
        self._validate_policy()
        launcher_args = _validate_production_univtac_launcher_args(self.launcher_args)
        object.__setattr__(self, "launcher_args", freeze_value(launcher_args))

    def _normalize_identities(self) -> None:
        for name in ("dataset_sha256", "checkpoint_sha256", "config_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self,
            "base_system_manifest_sha256",
            _sha256(
                self.base_system_manifest_sha256,
                "base_system_manifest_sha256",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "matched_no_touch_system_id",
            _optional_string(
                self.matched_no_touch_system_id, "matched_no_touch_system_id"
            ),
        )
        object.__setattr__(
            self,
            "act_device_name",
            _optional_string(self.act_device_name, "act_device_name"),
        )
        object.__setattr__(
            self,
            "simulator_device",
            _optional_string(self.simulator_device, "simulator_device"),
        )

    def _normalize_budget(self) -> None:
        for name in (
            "max_control_cycles",
            "max_observation_steps",
            "execute_action_steps",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("initial_seed", "exogenous_seed"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if isinstance(self.wall_timeout_s, bool) or not isinstance(
            self.wall_timeout_s, (int, float)
        ):
            raise TypeError("wall_timeout_s must be a real number")
        timeout = float(self.wall_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("wall_timeout_s must be positive and finite")
        object.__setattr__(self, "wall_timeout_s", timeout)

    def _normalize_paths(self) -> None:
        for name in ("upstream_root", "runtime_dir"):
            object.__setattr__(self, name, _path(getattr(self, name), name))
        for name in (
            "output_dir",
            "fault_manifest_path",
            "rest_references_path",
            "matched_no_touch_artifact_path",
        ):
            object.__setattr__(
                self, name, _path(getattr(self, name), name, optional=True)
            )

    def _validate_condition(self) -> None:
        faulted = self.condition in {Condition.FAULTED, Condition.RESTORED}
        if faulted and self.fault_manifest_path is None:
            raise ValueError("faulted/restored condition requires a fault manifest")
        if self.condition is Condition.CLEAN and self.fault_manifest_path is not None:
            raise ValueError("clean condition cannot reference a fault manifest")
        if self.condition is Condition.NO_TOUCH and (
            self.fault_manifest_path is not None
            or self.rest_references_path is not None
        ):
            raise ValueError("no-touch condition cannot reference fault artifacts")
        if self.condition is not Condition.NO_TOUCH and (
            self.matched_no_touch_system_id is not None
            or self.matched_no_touch_artifact_path is not None
        ):
            raise ValueError("matched no-touch metadata is only valid for no-touch")
        if self.condition is Condition.NO_TOUCH and (
            (self.matched_no_touch_system_id is None)
            != (self.matched_no_touch_artifact_path is None)
        ):
            raise ValueError("matched no-touch identity and artifact must be paired")
        if self.rest_references_path is not None and not faulted:
            raise ValueError("rest references require a faulted/restored condition")
        if self.condition is Condition.RESTORED:
            if self.restoration_index is None or self.restoration_mode is None:
                raise ValueError("restored condition requires restoration metadata")
            object.__setattr__(
                self,
                "restoration_index",
                _nonnegative_int(self.restoration_index, "restoration_index"),
            )
        elif self.restoration_index is not None or self.restoration_mode is not None:
            raise ValueError(
                "restoration metadata is only valid for restored condition"
            )

    def _validate_policy(self) -> None:
        if self.policy_kind is LivePolicyKind.ACT:
            if (
                self.initial_state_policy
                is not InitialStatePolicy.OFFICIAL_REPRODUCTION
            ):
                raise ValueError("initial-state replacement is only valid for N0")
            if self.wall_timeout_role is not WallTimeoutRole.SCORING_BOUNDARY_V1:
                raise ValueError(
                    "infrastructure-only wall timeout is only valid for N0"
                )
            if self.act_device_name is None:
                raise ValueError("ACT execution requires act_device_name")
            if self.execute_action_steps != 1:
                raise ValueError("ACT execute_action_steps must equal one")
            if any(
                value is not None
                for value in (
                    self.n0_source_commit,
                    self.n0_normalizer_sha256,
                    self.n0_serve_bundle_sha256,
                    self.n0_prompt_manifest_sha256,
                )
            ):
                raise ValueError("ACT request cannot carry N0 transport identities")
            return
        if self.act_device_name is not None:
            raise ValueError("N0 request cannot carry an ACT device")
        if self.execute_action_steps != 24:
            raise ValueError("official N0 execute_action_steps must equal 24")
        if (
            self.n0_source_commit is None
            or _COMMIT.fullmatch(self.n0_source_commit) is None
        ):
            raise ValueError("N0 source commit must be lowercase 40-character hex")
        for name in (
            "n0_normalizer_sha256",
            "n0_serve_bundle_sha256",
            "n0_prompt_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

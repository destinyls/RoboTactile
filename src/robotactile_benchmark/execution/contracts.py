"""Immutable request contracts for one bounded UniVTAC execution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Optional, Union, cast

from robotactile_benchmark.backends.univtac_success_profiles import (
    UniVTACSuccessProfile,
    selected_success_predicate_id,
)
from robotactile_benchmark.closed_loop.contracts import (
    InitialStatePolicy,
    WallTimeoutRole,
)
from robotactile_benchmark.contracts import freeze_value
from robotactile_benchmark.policies.n0_vtla_execution import n0_vtla_execution_steps
from robotactile_benchmark.policies.tactile_availability import (
    RGBShape,
    TactileAvailabilityMode,
    normalize_zero_shape,
    validate_availability_config,
)
from robotactile_benchmark.trials import Condition

LIVE_REQUEST_SEMANTIC_VERSION = "2.0"
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
    FTP1_POLICY = "ftp1_policy"
    N0 = "n0"
    N0_VTLA = "n0_vtla"
    DREAM_TAC = "dream_tac"


def effective_univtac_control_hz(
    policy_kind: LivePolicyKind,
    retrained_control_hz: Optional[int],
) -> Optional[int]:
    """Resolve the policy-side control cadence used by the live backend.

    Official UniVTAC ACT evaluation calls ``task.take_action`` once per model
    output, and its qpos branch advances exactly one native 120 Hz simulator
    step.  HDF5 observation spacing is a recording contract, not the ACT
    deployment cadence.  External retrained policies continue to use their
    explicitly bound cadence.
    """

    if not isinstance(policy_kind, LivePolicyKind):
        raise TypeError("policy_kind must be a LivePolicyKind")
    if policy_kind is LivePolicyKind.ACT:
        if retrained_control_hz is not None:
            raise ValueError("official ACT cannot declare a retrained control Hz")
        return None
    return retrained_control_hz


class N0ObservedTactileMode(str, Enum):
    """How the released N0 policy receives observed tactile conditioning."""

    REQUIRED = "required_v1"
    ABSENT = "observed_tactile_absent_v1"


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
    matched_no_touch_system_id: Optional[str]
    matched_no_touch_artifact_path: Optional[Path]
    act_device_name: Optional[str]
    simulator_device: Optional[str]
    launcher_args: Mapping[str, Any] = field(default_factory=dict)
    success_profile_id: UniVTACSuccessProfile = UniVTACSuccessProfile.OFFICIAL_V1
    n0_source_commit: Optional[str] = None
    n0_normalizer_sha256: Optional[str] = None
    n0_serve_bundle_sha256: Optional[str] = None
    n0_prompt_manifest_sha256: Optional[str] = None
    n0_observed_tactile_mode: N0ObservedTactileMode = N0ObservedTactileMode.REQUIRED
    initial_state_policy: InitialStatePolicy = InitialStatePolicy.OFFICIAL_REPRODUCTION
    wall_timeout_role: WallTimeoutRole = WallTimeoutRole.SCORING_BOUNDARY_V1
    semantic_version: str = LIVE_REQUEST_SEMANTIC_VERSION
    n0_action_per_frame: int = 12
    n0_prompt_override: Optional[str] = None
    retrained_prompt: Optional[str] = None
    retrained_control_hz: Optional[int] = None
    retrained_tactile_payload: Optional[str] = None
    tactile_availability_mode: TactileAvailabilityMode = (
        TactileAvailabilityMode.REQUIRED
    )
    tactile_zero_shape: Optional[RGBShape] = None
    n0_vtla_execution_profile: Optional[str] = None

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
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "policy_kind", policy_kind)
        if self.n0_vtla_execution_profile is not None:
            if policy_kind is not LivePolicyKind.N0_VTLA:
                raise ValueError(
                    "N0-VTLA execution profile cannot change another model"
                )
            n0_vtla_execution_steps(self.n0_vtla_execution_profile, self.task_id)
            if self.retrained_control_hz != 10 or self.retrained_prompt is None:
                raise ValueError("N0-VTLA 50x8 requires the retrained 10 Hz contract")
        success_profile = (
            self.success_profile_id
            if isinstance(self.success_profile_id, UniVTACSuccessProfile)
            else UniVTACSuccessProfile(self.success_profile_id)
        )
        selected_success_predicate_id(
            task_id=self.task_id,
            official_predicate_id="official_validation_only",
            profile=success_profile,
        )
        object.__setattr__(self, "success_profile_id", success_profile)
        tactile_mode = (
            self.n0_observed_tactile_mode
            if isinstance(self.n0_observed_tactile_mode, N0ObservedTactileMode)
            else N0ObservedTactileMode(self.n0_observed_tactile_mode)
        )
        object.__setattr__(self, "n0_observed_tactile_mode", tactile_mode)
        availability = TactileAvailabilityMode(self.tactile_availability_mode)
        zero_shape = normalize_zero_shape(self.tactile_zero_shape)
        validate_availability_config(availability, zero_shape, policy_kind.value)
        if availability is not TactileAvailabilityMode.REQUIRED and (
            condition is Condition.NO_TOUCH
            or tactile_mode is not N0ObservedTactileMode.REQUIRED
        ):
            raise ValueError("availability protocols cannot mix with no-touch modes")
        object.__setattr__(self, "tactile_availability_mode", availability)
        object.__setattr__(self, "tactile_zero_shape", zero_shape)
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
        faulted = self.condition is Condition.FAULTED
        if faulted and self.fault_manifest_path is None:
            raise ValueError("faulted condition requires a fault manifest")
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
            raise ValueError("rest references require a faulted condition")
        if (
            self.n0_observed_tactile_mode is N0ObservedTactileMode.ABSENT
            and not faulted
        ):
            raise ValueError("observed tactile absence requires a faulted condition")

    def _validate_policy(self) -> None:
        retrained = (
            self.retrained_prompt,
            self.retrained_control_hz,
            self.retrained_tactile_payload,
        )
        if any(value is not None for value in retrained):
            if any(value is None for value in retrained):
                raise ValueError(
                    "retrained prompt, Hz and tactile route are inseparable"
                )
            if self.policy_kind not in {
                LivePolicyKind.N0_VTLA,
                LivePolicyKind.FTP1_POLICY,
                LivePolicyKind.DREAM_TAC,
            }:
                raise ValueError(
                    "generic retrained contract requires an external policy"
                )
            _nonempty(self.retrained_prompt, "retrained_prompt")
            if type(
                self.retrained_control_hz
            ) is not int or self.retrained_control_hz not in (10, 60):
                raise ValueError("retrained control Hz must be 10 or 60")
            if self.retrained_tactile_payload not in ("rgb", "rgb_marker"):
                raise ValueError("unknown retrained tactile route")
        if self.policy_kind is LivePolicyKind.DREAM_TAC:
            if self.retrained_control_hz != 10 or self.execute_action_steps != 20:
                raise ValueError("retrained Dream-Tac requires 10 Hz and 20 actions")
            if (
                self.act_device_name is not None
                or self.n0_action_per_frame != 12
                or self.n0_prompt_override is not None
            ):
                raise ValueError("Dream-Tac cannot use ACT or N0-only options")
            if self.n0_observed_tactile_mode is not N0ObservedTactileMode.REQUIRED:
                raise ValueError("Dream-Tac requires both tactile streams")
            self._validate_external_policy_identities("Dream-Tac")
            return
        if type(
            self.n0_action_per_frame
        ) is not int or self.n0_action_per_frame not in (4, 12):
            raise ValueError("N0 action_per_frame must be 4 or 12")
        if self.n0_prompt_override is not None:
            _nonempty(self.n0_prompt_override, "n0_prompt_override")
        if self.policy_kind is not LivePolicyKind.N0 and (
            self.n0_action_per_frame != 12 or self.n0_prompt_override is not None
        ):
            raise ValueError("retrained N0 fields require policy_kind=n0")
        if self.n0_action_per_frame == 4 and self.n0_prompt_override is None:
            raise ValueError("retrained N0 requires its training task prompt")
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
            if self.n0_observed_tactile_mode is not N0ObservedTactileMode.REQUIRED:
                raise ValueError("ACT request cannot select an N0 tactile mode")
            return
        if self.act_device_name is not None:
            raise ValueError("external-policy request cannot carry an ACT device")
        if self.policy_kind is LivePolicyKind.FTP1_POLICY:
            if self.execute_action_steps != 1:
                raise ValueError(
                    "official FTP-1 temporal-ensemble execution requires "
                    "execute_action_steps=1"
                )
            if (
                self.initial_state_policy
                is not InitialStatePolicy.OFFICIAL_REPRODUCTION
            ):
                raise ValueError("FTP-1 requires official initial-state reproduction")
            if self.n0_observed_tactile_mode is not N0ObservedTactileMode.REQUIRED:
                raise ValueError("FTP-1 requires both observed tactile streams")
            self._validate_external_policy_identities("FTP-1")
            return
        family = "N0-TWAM" if self.policy_kind is LivePolicyKind.N0 else "N0-VTLA"
        required_action_steps = (
            2 * self.n0_action_per_frame
            if self.policy_kind is LivePolicyKind.N0
            else n0_vtla_execution_steps(self.n0_vtla_execution_profile, self.task_id)
        )
        if self.execute_action_steps != required_action_steps:
            raise ValueError(
                f"official {family} execute_action_steps must equal "
                f"{required_action_steps}"
            )
        self._validate_external_policy_identities(family)

    def _validate_external_policy_identities(self, family: str) -> None:
        """Validate legacy-named source identities shared by RPC integrations."""

        if (
            self.n0_source_commit is None
            or _COMMIT.fullmatch(self.n0_source_commit) is None
        ):
            raise ValueError(
                f"{family} source commit must be lowercase 40-character hex"
            )
        for name in (
            "n0_normalizer_sha256",
            "n0_serve_bundle_sha256",
            "n0_prompt_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))

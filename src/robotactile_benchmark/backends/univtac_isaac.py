"""Dependency-injected UniVTAC backend state machine for Isaac execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple

import numpy as np

from robotactile_benchmark.adapters.univtac import ContactPhaseState
from robotactile_benchmark.backends.univtac_contracts import (
    BACKEND_ID,
    EARLY_STOP_NONE_IS_FALSE_TASK_IDS,
    UniVTACBackendConfig,
    UniVTACContractError,
    UniVTACRuntimeHandshake,
    validate_packaged_univtac_config,
)
from robotactile_benchmark.backends.univtac_conversion import (
    ConvertedUniVTACObservation,
    UniVTACConversionError,
    convert_raw_observation,
    validate_action_batch,
)
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    BackendTransition,
    ExecutionBatch,
    PolicyEpisodeContext,
    ResetReceipt,
)
from robotactile_benchmark.contracts import Array, EvaluationRecord, canonical_hash


class UniVTACTask(Protocol):
    """Only the upstream task surface used by the benchmark backend."""

    plan_success: Any

    def reset(self, seed: int, instructions: list[str]) -> Any: ...

    def _get_observations(self) -> Mapping[str, Any]: ...

    def take_action(
        self, action: Any, action_type: str = "qpos", force: bool = True
    ) -> Tuple[Any, Any]: ...

    def check_success(self) -> Any: ...

    def check_early_stop(self) -> Any: ...


@dataclass(frozen=True)
class UniVTACTaskRuntime:
    """A live task plus facts collected only after AppLauncher starts."""

    task: UniVTACTask
    handshake: UniVTACRuntimeHandshake
    encode_action: Callable[[Array], Any]
    close_runtime: Callable[[], None]

    def __post_init__(self) -> None:
        if not callable(self.encode_action) or not callable(self.close_runtime):
            raise TypeError("runtime callbacks must be callable")


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise UniVTACContractError(f"{name} must be bool")


def _action_result(value: Any) -> Tuple[bool, bool]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise UniVTACContractError(
            "take_action must return (execution_success, eval_success)"
        )
    return (
        _strict_bool(value[0], "take_action execution_success"),
        _strict_bool(value[1], "take_action eval_success"),
    )


def _early_stop_result(value: Any, task_id: str) -> bool:
    if value is None and task_id in EARLY_STOP_NONE_IS_FALSE_TASK_IDS:
        return False
    return _strict_bool(value, "check_early_stop")


def resolve_backend_signal(
    *,
    success: bool,
    execution_success: bool,
    plan_success: bool,
    early_stop: bool,
    horizon_reached: bool,
) -> BackendSignal:
    if success:
        return BackendSignal.SUCCESS
    if not execution_success or not plan_success:
        return BackendSignal.TASK_FAILURE
    if early_stop:
        return BackendSignal.EARLY_STOP
    if horizon_reached:
        return BackendSignal.TIMEOUT
    return BackendSignal.RUNNING


class UniVTACIsaacBackend:
    """Single-use action-conditioned adapter around one upstream UniVTAC task."""

    backend_id = BACKEND_ID

    def __init__(
        self,
        config: UniVTACBackendConfig,
        runtime: UniVTACTaskRuntime,
    ) -> None:
        validate_packaged_univtac_config(config)
        config.validate_handshake(runtime.handshake)
        self._config = config
        self._runtime = runtime
        self.action_spec = config.action_spec
        self.success_predicate_id = config.task.success_predicate_id
        self._context: Optional[PolicyEpisodeContext] = None
        self._cached_initial: Optional[EvaluationRecord] = None
        self._initial_delivered = False
        self._reset_attempted = False
        self._closed = False
        self._terminal = False
        self._benchmark_step = 0
        self._executed_action_count = 0
        self._last_native_step: Optional[int] = None
        self._initial_native_step_id: Optional[int] = None
        self._latest_state_sha256: Optional[str] = None
        self._initial_clean_record_sha256: Optional[str] = None
        self._joint_reorder_witness_sha256: Optional[str] = None
        self._initial_canonical_joint9: Optional[Array] = None
        self._initial_model_visible_qpos8: Optional[Array] = None
        self._phase_states = {
            "left": ContactPhaseState(False),
            "right": ContactPhaseState(False),
        }

    @property
    def handshake(self) -> UniVTACRuntimeHandshake:
        return self._runtime.handshake

    @property
    def latest_state_sha256(self) -> Optional[str]:
        return self._latest_state_sha256

    @property
    def initial_clean_record_sha256(self) -> Optional[str]:
        return self._initial_clean_record_sha256

    @property
    def joint_reorder_witness_sha256(self) -> Optional[str]:
        return self._joint_reorder_witness_sha256

    @property
    def initial_canonical_joint9(self) -> Optional[Array]:
        return self._initial_canonical_joint9

    @property
    def initial_model_visible_qpos8(self) -> Optional[Array]:
        return self._initial_model_visible_qpos8

    @property
    def initial_native_step_id(self) -> Optional[int]:
        return self._initial_native_step_id

    @property
    def executed_action_count(self) -> int:
        return self._executed_action_count

    def reset(self, context: PolicyEpisodeContext) -> ResetReceipt:
        """Reset once, discard the stale return, then capture an explicit frame."""

        if self._closed:
            raise UniVTACContractError("closed backend cannot reset")
        if self._reset_attempted:
            raise UniVTACContractError("UniVTAC backend is single-use per reset")
        if (
            context.task != self._config.task.task_id
            or context.instruction != self._config.task.prompt
            or context.action_spec != self.action_spec
        ):
            raise UniVTACContractError(
                "task, prompt, or action spec does not match backend config"
            )
        self._reset_attempted = True
        self._runtime.task.reset(
            seed=context.initial_seed,
            instructions=[context.instruction],
        )
        self._context = context
        converted = self._convert(self._fresh_raw(), benchmark_step=0)
        self._accept_conversion(converted)
        self._initial_native_step_id = converted.native_step_id
        self._joint_reorder_witness_sha256 = converted.joint_reorder_witness_sha256
        self._initial_canonical_joint9 = converted.canonical_joint9
        self._initial_model_visible_qpos8 = converted.model_visible_qpos8
        self._cached_initial = converted.record
        self._initial_clean_record_sha256 = converted.record.clean_record_sha256
        return ResetReceipt(
            episode_id=context.episode_id,
            initial_seed=context.initial_seed,
            exogenous_seed=context.exogenous_seed,
            simulator_state_sha256=converted.simulator_state_sha256,
            native_reset_id=(
                "univtac-reset-"
                + canonical_hash(
                    {
                        "config_sha256": self._config.sha256,
                        "initial_seed": context.initial_seed,
                        "native_step": converted.native_step_id,
                        "state_sha256": converted.simulator_state_sha256,
                    }
                )[:20]
            ),
        )

    def observe(self) -> EvaluationRecord:
        """Return exactly the explicit post-reset observation cached as step zero."""

        if self._cached_initial is None or self._context is None:
            raise UniVTACContractError("backend must reset before observe")
        if self._initial_delivered:
            raise UniVTACContractError("initial observation was already delivered")
        self._initial_delivered = True
        return self._cached_initial

    def execute(self, actions: Array) -> ExecutionBatch:
        """Execute validated qpos8 rows until the batch or first terminal signal."""

        if self._context is None or not self._initial_delivered:
            raise UniVTACContractError("backend must reset and observe before execute")
        if self._closed:
            raise UniVTACContractError("closed backend cannot execute")
        if self._terminal:
            raise UniVTACContractError("terminal backend cannot execute again")
        batch = validate_action_batch(actions, self._config)
        transitions = []
        for row_index, row in enumerate(batch):
            encoded = self._runtime.encode_action(row)
            execution_success, returned_success = _action_result(
                self._runtime.task.take_action(
                    encoded,
                    action_type=self._config.action_mode,
                    force=self._config.force,
                )
            )
            self._executed_action_count += 1
            next_step = self._benchmark_step + 1
            converted = self._convert(self._fresh_raw(), benchmark_step=next_step)
            expected_native_step = (
                None
                if self._last_native_step is None
                else self._last_native_step + self._config.physics_steps_per_action
            )
            if converted.native_step_id != expected_native_step:
                raise UniVTACConversionError(
                    "native step continuity mismatch after take_action",
                    code="native_step_continuity",
                )
            success_check = _strict_bool(
                self._runtime.task.check_success(), "check_success"
            )
            plan_success = _strict_bool(
                self._runtime.task.plan_success, "task.plan_success"
            )
            early_stop = (
                _early_stop_result(
                    self._runtime.task.check_early_stop(),
                    self._config.task.task_id,
                )
                if self._config.task.early_stop_capable
                else False
            )
            signal = resolve_backend_signal(
                success=returned_success or success_check,
                execution_success=execution_success,
                plan_success=plan_success,
                early_stop=early_stop,
                horizon_reached=(
                    self._executed_action_count >= self._config.task.action_horizon
                ),
            )
            self._accept_conversion(converted)
            transitions.append(
                BackendTransition(
                    clean_record=converted.record,
                    signal=signal,
                    native_step_id=converted.native_step_id,
                    diagnostics={
                        "benchmark_step": next_step,
                        "batch_row_index": row_index,
                        "execution_success": execution_success,
                        "returned_success": returned_success,
                        "success_check": success_check,
                        "plan_success": plan_success,
                        "early_stop": early_stop,
                        "action_horizon": self._config.task.action_horizon,
                    },
                )
            )
            if signal is not BackendSignal.RUNNING:
                self._terminal = True
                break
        return ExecutionBatch(
            transitions=tuple(transitions),
            executed_action_count=len(transitions),
        )

    def close(self) -> None:
        """Close the injected task/AppLauncher runtime exactly once."""

        if self._closed:
            return
        self._closed = True
        self._runtime.close_runtime()

    def _fresh_raw(self) -> Mapping[str, Any]:
        raw = self._runtime.task._get_observations()
        if not isinstance(raw, Mapping):
            raise UniVTACConversionError("_get_observations must return a mapping")
        return raw

    def _convert(
        self, raw: Mapping[str, Any], benchmark_step: int
    ) -> ConvertedUniVTACObservation:
        if self._context is None:
            raise UniVTACContractError("missing reset context")
        return convert_raw_observation(
            raw,
            config=self._config,
            handshake=self._runtime.handshake,
            phase_states=self._phase_states,
            episode_id=self._context.episode_id,
            task_id=self._context.task,
            initial_seed=self._context.initial_seed,
            benchmark_step=benchmark_step,
        )

    def _accept_conversion(self, converted: ConvertedUniVTACObservation) -> None:
        self._phase_states = converted.phase_state_mapping()
        self._benchmark_step = converted.record.observation.step_index
        self._last_native_step = converted.native_step_id
        self._latest_state_sha256 = converted.simulator_state_sha256

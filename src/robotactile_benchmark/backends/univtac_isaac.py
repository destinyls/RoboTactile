"""Dependency-injected UniVTAC backend state machine for Isaac execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from robotactile_benchmark.action_specs import EE8_ACTION_SPEC
from robotactile_benchmark.adapters.univtac import ContactPhaseState
from robotactile_benchmark.backends.univtac_contracts import (
    BACKEND_ID,
    FIXED_NATIVE_STEP_CONTRACT,
    N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_ACTION_EXECUTION_CONTRACT,
    N0_STOCK_EE_NATIVE_STEP_CONTRACT,
    N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    UniVTACBackendConfig,
    UniVTACContractError,
    UniVTACRuntimeHandshake,
    validate_n0_ee_action_execution_contract,
    validate_packaged_univtac_config,
)
from robotactile_benchmark.backends.univtac_conversion import (
    ConvertedUniVTACObservation,
    UniVTACConversionError,
    convert_raw_observation,
    validate_action_batch,
)
from robotactile_benchmark.backends.univtac_signals import (
    action_result as _action_result,
)
from robotactile_benchmark.backends.univtac_signals import (
    early_stop_result as _early_stop_result,
)
from robotactile_benchmark.backends.univtac_signals import (
    resolve_backend_signal,
)
from robotactile_benchmark.backends.univtac_signals import (
    strict_bool as _strict_bool,
)
from robotactile_benchmark.backends.univtac_task_diagnostics import (
    capture_task_diagnostics,
)
from robotactile_benchmark.closed_loop.contracts import (
    BackendSignal,
    BackendTransition,
    ExecutionBatch,
    PolicyEpisodeContext,
    ResetReceipt,
)
from robotactile_benchmark.contracts import Array, EvaluationRecord, canonical_hash


def _n0_fixed_cadence_enabled(task: object) -> bool:
    value = getattr(task, "_robotactile_n0_fixed_cadence_enabled", False)
    if type(value) is not bool:
        raise UniVTACContractError("N0 fixed-cadence marker must be boolean")
    return value


def _runtime_action_execution_contract(
    task: object,
    config: UniVTACBackendConfig,
) -> str:
    """Resolve and validate the executor installed on the live task."""

    if config.action_spec != EE8_ACTION_SPEC:
        return config.action_execution_contract
    raw_contract = getattr(
        task,
        "_robotactile_n0_action_execution_contract",
        config.action_execution_contract,
    )
    selected = validate_n0_ee_action_execution_contract(raw_contract)
    fixed_cadence = _n0_fixed_cadence_enabled(task)
    selected_is_fixed = selected in {
        N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
        N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
    }
    if fixed_cadence is not selected_is_fixed:
        raise UniVTACContractError(
            "live N0 action execution marker disagrees with installed executor"
        )
    return selected


def _fixed_physics_steps_per_action(
    config: UniVTACBackendConfig,
    action_execution_contract: str,
) -> Optional[int]:
    """Return a fixed cadence only for executors that guarantee one."""

    if action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT:
        return None
    return config.physics_steps_per_action


def _native_step_contract(action_execution_contract: str) -> str:
    """Describe the live executor's native-step behavior."""

    if action_execution_contract == N0_STOCK_EE_ACTION_EXECUTION_CONTRACT:
        return N0_STOCK_EE_NATIVE_STEP_CONTRACT
    return FIXED_NATIVE_STEP_CONTRACT


def _action_execution_source(
    config: UniVTACBackendConfig,
    *,
    action_execution_contract: str,
) -> dict[str, object]:
    method = (
        "robotactile_benchmark.backends.univtac_n0_cadence.fixed_take_action"
        if action_execution_contract
        in {
            N0_FIXED_ENDPOINT_ACTION_EXECUTION_CONTRACT,
            N0_TRAINING_60HZ_ACTION_EXECUTION_CONTRACT,
        }
        else "BaseTask.take_action"
    )
    return {
        "action_type": config.action_mode,
        "method": method,
        "task_source_sha256": config.task.task_source_sha256,
        "upstream_commit": config.upstream_commit,
    }


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


@runtime_checkable
class UniVTACResetCoordinator(Protocol):
    """Coordinate one canonical reset and later in-process state restores."""

    def prepare(self, context: PolicyEpisodeContext) -> str: ...

    def initial_conversion(
        self, context: PolicyEpisodeContext
    ) -> Optional[ConvertedUniVTACObservation]: ...

    def accept(self, converted: ConvertedUniVTACObservation) -> None: ...


@dataclass(frozen=True)
class UniVTACTaskRuntime:
    """A live task plus facts collected only after AppLauncher starts."""

    task: UniVTACTask
    handshake: UniVTACRuntimeHandshake
    construction_seed: int
    encode_action: Callable[[Array], Any]
    prepare_reset: Callable[[], None]
    close_runtime: Callable[[], None]
    capture_state: Optional[Callable[[], object]] = None
    restore_state: Optional[Callable[[object], None]] = None
    snapshot_state_sha256: Optional[Callable[[], str]] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.construction_seed, bool)
            or not isinstance(self.construction_seed, int)
            or self.construction_seed < 0
        ):
            raise TypeError("construction seed must be a non-negative integer")
        if (
            not callable(self.encode_action)
            or not callable(self.prepare_reset)
            or not callable(self.close_runtime)
        ):
            raise TypeError("runtime callbacks must be callable")
        snapshot_callbacks = (
            self.capture_state,
            self.restore_state,
            self.snapshot_state_sha256,
        )
        if any(item is None for item in snapshot_callbacks) and not all(
            item is None for item in snapshot_callbacks
        ):
            raise TypeError("runtime snapshot callbacks must be a complete triple")
        if self.capture_state is not None and not all(
            callable(item) for item in snapshot_callbacks
        ):
            raise TypeError("runtime snapshot callbacks must be callable")


def _validate_task_predicate_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    success_check: bool,
    early_stop: bool,
    stage: str,
) -> None:
    for key, expected in (
        ("predicate_success", success_check),
        ("early_stop_predicate", early_stop),
    ):
        observed = diagnostics.get(key)
        if observed is None:
            continue
        if type(observed) is not bool or observed is not expected:
            raise UniVTACContractError(
                f"{stage} task diagnostic {key} disagrees with evaluator"
            )


class UniVTACIsaacBackend:
    """Single-use action-conditioned adapter around one upstream UniVTAC task."""

    backend_id = BACKEND_ID

    def __init__(
        self,
        config: UniVTACBackendConfig,
        runtime: UniVTACTaskRuntime,
        *,
        reset_coordinator: Optional[UniVTACResetCoordinator] = None,
        owns_runtime: bool = True,
    ) -> None:
        validate_packaged_univtac_config(config)
        config.validate_handshake(runtime.handshake)
        self._config = config
        self._runtime = runtime
        if reset_coordinator is not None and not isinstance(
            reset_coordinator, UniVTACResetCoordinator
        ):
            raise TypeError("reset_coordinator must satisfy its runtime protocol")
        if type(owns_runtime) is not bool:
            raise TypeError("owns_runtime must be bool")
        self._reset_coordinator = reset_coordinator
        self._owns_runtime = owns_runtime
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
        self._latest_canonical_joint9: Optional[Array] = None
        self._latest_model_visible_qpos8: Optional[Array] = None
        self._reset_mode: Optional[str] = None
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
    def latest_canonical_joint9(self) -> Optional[Array]:
        return self._latest_canonical_joint9

    @property
    def latest_model_visible_qpos8(self) -> Optional[Array]:
        return self._latest_model_visible_qpos8

    @property
    def initial_native_step_id(self) -> Optional[int]:
        return self._initial_native_step_id

    @property
    def executed_action_count(self) -> int:
        return self._executed_action_count

    @property
    def reset_mode(self) -> Optional[str]:
        return self._reset_mode

    def reset(self, context: PolicyEpisodeContext) -> ResetReceipt:
        """Reset once, discard the stale return, then capture an explicit frame."""

        if self._closed:
            raise UniVTACContractError("closed backend cannot reset")
        if self._reset_attempted:
            raise UniVTACContractError("UniVTAC backend is single-use per reset")
        if (
            context.task != self._config.task.task_id
            or context.action_spec != self.action_spec
        ):
            raise UniVTACContractError(
                "task or action spec does not match backend config"
            )
        if context.initial_seed != self._runtime.construction_seed:
            raise UniVTACContractError(
                "reset seed must match the task construction seed"
            )
        self._reset_attempted = True
        self._runtime.prepare_reset()
        if self._reset_coordinator is None:
            self._runtime.task.reset(
                seed=context.initial_seed,
                instructions=[self._config.task.prompt],
            )
            self._reset_mode = "independent_reset"
        else:
            self._reset_mode = self._reset_coordinator.prepare(context)
        self._context = context
        converted = (
            None
            if self._reset_coordinator is None
            else self._reset_coordinator.initial_conversion(context)
        )
        if converted is None:
            converted = self._convert(self._fresh_raw(), benchmark_step=0)
        if self._reset_coordinator is not None:
            self._reset_coordinator.accept(converted)
        self._accept_conversion(converted)
        self._initial_native_step_id = converted.native_step_id
        self._joint_reorder_witness_sha256 = converted.joint_reorder_witness_sha256
        self._initial_canonical_joint9 = converted.canonical_joint9
        self._initial_model_visible_qpos8 = converted.model_visible_qpos8
        self._cached_initial = converted.record
        self._initial_clean_record_sha256 = converted.record.clean_record_sha256
        success_check = _strict_bool(
            self._runtime.task.check_success(), "initial check_success"
        )
        plan_success = _strict_bool(
            self._runtime.task.plan_success, "initial task.plan_success"
        )
        early_stop = (
            _early_stop_result(
                self._runtime.task.check_early_stop(),
                self._config.task.task_id,
            )
            if self._config.task.early_stop_capable
            else False
        )
        task_diagnostics = capture_task_diagnostics(
            self._runtime.task, self._config.task.task_id
        )
        _validate_task_predicate_diagnostics(
            task_diagnostics,
            success_check=success_check,
            early_stop=early_stop,
            stage="initial",
        )
        reset_diagnostics = getattr(
            self._runtime.task, "_robotactile_n0_last_reset", None
        )
        if reset_diagnostics is not None and not isinstance(reset_diagnostics, Mapping):
            raise UniVTACContractError("N0 reset diagnostics must be a mapping")
        action_execution_contract = _runtime_action_execution_contract(
            self._runtime.task,
            self._config,
        )
        fixed_physics_steps = _fixed_physics_steps_per_action(
            self._config,
            action_execution_contract,
        )
        native_step_contract = _native_step_contract(action_execution_contract)
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
            diagnostics={
                "action_execution_contract": action_execution_contract,
                "action_execution_source": _action_execution_source(
                    self._config,
                    action_execution_contract=action_execution_contract,
                ),
                "benchmark_step": 0,
                "camera_delivery_hz": (
                    None
                    if fixed_physics_steps is None
                    else self._config.sim_hz / fixed_physics_steps
                ),
                "decimation": self._config.decimation,
                "native_step_contract": native_step_contract,
                "native_step_id": converted.native_step_id,
                "physics_steps_per_action": fixed_physics_steps,
                "sim_hz": self._config.sim_hz,
                "success_check": success_check,
                "plan_success": plan_success,
                "early_stop": early_stop,
                "n0_reset": (
                    None if reset_diagnostics is None else dict(reset_diagnostics)
                ),
                "left_contact_phase": converted.record.provenance_for(
                    "left"
                ).phase.value,
                "right_contact_phase": converted.record.provenance_for(
                    "right"
                ).phase.value,
                "task": task_diagnostics,
            },
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
        """Execute validated action rows until the batch or first terminal signal."""

        if self._context is None or not self._initial_delivered:
            raise UniVTACContractError("backend must reset and observe before execute")
        if self._closed:
            raise UniVTACContractError("closed backend cannot execute")
        if self._terminal:
            raise UniVTACContractError("terminal backend cannot execute again")
        batch = validate_action_batch(actions, self._config)
        action_execution_contract = _runtime_action_execution_contract(
            self._runtime.task,
            self._config,
        )
        fixed_physics_steps = _fixed_physics_steps_per_action(
            self._config,
            action_execution_contract,
        )
        native_step_contract = _native_step_contract(action_execution_contract)
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
            native_step_delta = (
                None
                if self._last_native_step is None
                else converted.native_step_id - self._last_native_step
            )
            if native_step_delta is None or native_step_delta < 1:
                raise UniVTACConversionError(
                    "native step continuity mismatch after take_action",
                    code="native_step_continuity",
                )
            physics_step_delta = native_step_delta * self._config.decimation
            if (
                fixed_physics_steps is not None
                and physics_step_delta != fixed_physics_steps
            ):
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
            task_diagnostics = capture_task_diagnostics(
                self._runtime.task, self._config.task.task_id
            )
            fixed_cadence_diagnostics = getattr(
                self._runtime.task, "_robotactile_n0_last_plan", None
            )
            if fixed_cadence_diagnostics is not None and not isinstance(
                fixed_cadence_diagnostics, Mapping
            ):
                raise UniVTACContractError(
                    "N0 fixed-cadence diagnostics must be a mapping"
                )
            _validate_task_predicate_diagnostics(
                task_diagnostics,
                success_check=success_check,
                early_stop=early_stop,
                stage="transition",
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
                        "action_execution_contract": action_execution_contract,
                        "action_execution_source": _action_execution_source(
                            self._config,
                            action_execution_contract=action_execution_contract,
                        ),
                        "benchmark_step": next_step,
                        "batch_row_index": row_index,
                        "native_step_delta": native_step_delta,
                        "native_step_contract": native_step_contract,
                        "physics_step_delta": physics_step_delta,
                        "physics_steps_per_action": fixed_physics_steps,
                        "observed_action_duration_s": (
                            physics_step_delta / self._config.sim_hz
                        ),
                        "execution_success": execution_success,
                        "returned_success": returned_success,
                        "success_check": success_check,
                        "plan_success": plan_success,
                        "n0_fixed_cadence": (
                            None
                            if fixed_cadence_diagnostics is None
                            else dict(fixed_cadence_diagnostics)
                        ),
                        "early_stop": early_stop,
                        "action_horizon": self._config.task.action_horizon,
                        "left_contact_phase": converted.record.provenance_for(
                            "left"
                        ).phase.value,
                        "right_contact_phase": converted.record.provenance_for(
                            "right"
                        ).phase.value,
                        "task": task_diagnostics,
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
        if self._owns_runtime:
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
        self._latest_canonical_joint9 = converted.canonical_joint9
        self._latest_model_visible_qpos8 = converted.model_visible_qpos8
